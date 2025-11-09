"""
Market Data Module - Polygon Integration
Handles real-time and historical market data ingestion using Polygon API.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import pandas as pd
from dataclasses import dataclass

from .polygon_client import PolygonClient, AggregateBar, PolygonRestClient

logger = logging.getLogger(__name__)


@dataclass
class MarketDataConfig:
    """Configuration for market data ingestion"""

    polygon_api_key: str
    default_timeframe: str = "5min"  # 1min, 5min, 15min, etc.
    cache_bars: bool = True
    max_cache_size: int = 1000  # bars per ticker


class MarketDataProvider:
    """
    Market data provider using Polygon API.
    Provides OHLCV bars, VWAP, and real-time data.
    """

    def __init__(self, config: MarketDataConfig):
        """
        Initialize market data provider.

        Args:
            config: MarketDataConfig with Polygon API settings
        """
        self.config = config
        self.client = PolygonClient(config.polygon_api_key)
        self.rest_client = self.client.rest

        # Cache for recent bars
        self.bar_cache: Dict[str, List[AggregateBar]] = {}

        logger.info("MarketDataProvider initialized with Polygon API")

    def get_bars(
        self,
        ticker: str,
        timeframe: str = "5min",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Get historical OHLCV bars for a ticker.

        Args:
            ticker: Stock ticker symbol
            timeframe: Bar timeframe (1min, 5min, 15min, 1hour, 1day)
            start_time: Start datetime (default: 24 hours ago)
            end_time: End datetime (default: now)
            limit: Maximum number of bars

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, vwap
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(days=1)

        if end_time is None:
            end_time = datetime.now()

        # Parse timeframe
        multiplier, timespan = self._parse_timeframe(timeframe)

        # Fetch from Polygon
        bars = self.rest_client.get_aggregate_bars(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_date=start_time,
            to_date=end_time,
            adjusted=True,
            sort="asc",
            limit=limit,
        )

        if not bars:
            logger.warning(f"No bars returned for {ticker}")
            return pd.DataFrame()

        # Update cache
        if self.config.cache_bars:
            self._update_cache(ticker, bars)

        # Convert to DataFrame
        df = self._bars_to_dataframe(bars)

        logger.info(f"Retrieved {len(df)} bars for {ticker} ({timeframe})")
        return df

    def get_latest_bar(
        self, ticker: str, timeframe: str = "1min"
    ) -> Optional[AggregateBar]:
        """
        Get the most recent bar for a ticker.

        Args:
            ticker: Stock ticker symbol
            timeframe: Bar timeframe

        Returns:
            Latest AggregateBar or None
        """
        # Try cache first
        if ticker in self.bar_cache and self.bar_cache[ticker]:
            return self.bar_cache[ticker][-1]

        # Fetch from API
        bars = self.get_bars(
            ticker=ticker,
            timeframe=timeframe,
            start_time=datetime.now() - timedelta(hours=1),
            limit=1,
        )

        if bars.empty:
            return None

        # Convert last row to AggregateBar
        last_row = bars.iloc[-1]
        return AggregateBar(
            ticker=ticker,
            timestamp=last_row.name,  # Use index (timestamp) instead of column
            open=last_row["open"],
            high=last_row["high"],
            low=last_row["low"],
            close=last_row["close"],
            volume=last_row["volume"],
            vwap=last_row["vwap"],
            transactions=last_row.get("transactions", 0),
        )

    def get_intraday_bars(
        self, ticker: str, date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get all intraday 1-minute bars for a specific date.

        Args:
            ticker: Stock ticker symbol
            date: Trading date (default: today)

        Returns:
            DataFrame with minute-level bars
        """
        if date is None:
            date = datetime.now()

        # Set time range for market hours (9:30 AM - 4:00 PM ET)
        start = date.replace(hour=9, minute=30, second=0, microsecond=0)
        end = date.replace(hour=16, minute=0, second=0, microsecond=0)

        return self.get_bars(
            ticker=ticker,
            timeframe="1min",
            start_time=start,
            end_time=end,
            limit=390,  # 6.5 hours * 60 minutes
        )

    def get_pre_market_data(
        self, ticker: str, date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get pre-market data (4:00 AM - 9:30 AM ET).

        Args:
            ticker: Stock ticker symbol
            date: Trading date

        Returns:
            DataFrame with pre-market bars
        """
        if date is None:
            date = datetime.now()

        start = date.replace(hour=4, minute=0, second=0, microsecond=0)
        end = date.replace(hour=9, minute=30, second=0, microsecond=0)

        return self.get_bars(
            ticker=ticker, timeframe="1min", start_time=start, end_time=end, limit=330
        )

    def calculate_vwap(self, bars_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VWAP if not already present.

        Args:
            bars_df: DataFrame with OHLCV data

        Returns:
            DataFrame with VWAP column added/recalculated
        """
        if bars_df.empty:
            return bars_df

        # If VWAP already exists from Polygon, use it
        if "vwap" in bars_df.columns and bars_df["vwap"].notna().all():
            return bars_df

        # Calculate VWAP manually
        bars_df["typical_price"] = (
            bars_df["high"] + bars_df["low"] + bars_df["close"]
        ) / 3
        bars_df["pv"] = bars_df["typical_price"] * bars_df["volume"]

        bars_df["cumulative_pv"] = bars_df["pv"].cumsum()
        bars_df["cumulative_volume"] = bars_df["volume"].cumsum()

        bars_df["vwap"] = bars_df["cumulative_pv"] / bars_df["cumulative_volume"]

        # Clean up temporary columns
        bars_df.drop(
            ["typical_price", "pv", "cumulative_pv", "cumulative_volume"],
            axis=1,
            inplace=True,
        )

        return bars_df

    def get_realtime_quote(self, ticker: str) -> Optional[Dict]:
        """
        Get the latest quote (current price) for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with current price info or None
        """
        # Use latest 1-minute bar as proxy for current price
        latest_bar = self.get_latest_bar(ticker, timeframe="1min")

        if latest_bar:
            return {
                "ticker": ticker,
                "price": latest_bar.close,
                "timestamp": latest_bar.timestamp,
                "volume": latest_bar.volume,
                "vwap": latest_bar.vwap,
            }

        return None

    def get_ticker_metadata(self, ticker: str) -> Optional[Dict]:
        """
        Get ticker fundamental data (float, market cap, etc.).

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with ticker details or None
        """
        details = self.rest_client.get_ticker_details(ticker)

        if not details:
            return None

        return {
            "ticker": ticker,
            "name": details.get("name", ""),
            "market_cap": details.get("market_cap"),
            "shares_outstanding": details.get("weighted_shares_outstanding"),
            "float": details.get("share_class_shares_outstanding"),
            "sector": details.get("sic_description"),
            "exchange": details.get("primary_exchange"),
            "currency": details.get("currency_name", "USD"),
            "last_updated": datetime.now(),
        }

    def _parse_timeframe(self, timeframe: str) -> tuple:
        """
        Parse timeframe string into Polygon API parameters.

        Args:
            timeframe: Timeframe string (e.g., "5min", "1hour", "1day")

        Returns:
            Tuple of (multiplier, timespan)
        """
        timeframe = timeframe.lower().strip()

        # Map common formats
        timeframe_map = {
            "1min": (1, "minute"),
            "5min": (5, "minute"),
            "15min": (15, "minute"),
            "30min": (30, "minute"),
            "1hour": (1, "hour"),
            "1h": (1, "hour"),
            "1day": (1, "day"),
            "1d": (1, "day"),
            "1week": (1, "week"),
            "1w": (1, "week"),
        }

        if timeframe in timeframe_map:
            return timeframe_map[timeframe]

        # Try to parse custom format (e.g., "10minute")
        import re

        match = re.match(
            r"(\d+)(minute|min|hour|h|day|d|week|w|month|quarter|year)", timeframe
        )

        if match:
            multiplier = int(match.group(1))
            unit = match.group(2)

            # Normalize unit
            unit_map = {"min": "minute", "h": "hour", "d": "day", "w": "week"}
            timespan = unit_map.get(unit, unit)

            return (multiplier, timespan)

        # Default fallback
        logger.warning(f"Unknown timeframe format: {timeframe}, using 5min")
        return (5, "minute")

    def _bars_to_dataframe(self, bars: List[AggregateBar]) -> pd.DataFrame:
        """
        Convert list of AggregateBar objects to DataFrame.

        Args:
            bars: List of AggregateBar objects

        Returns:
            DataFrame with bar data
        """
        data = []
        for bar in bars:
            data.append(
                {
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "vwap": bar.vwap,
                    "transactions": bar.transactions,
                }
            )

        df = pd.DataFrame(data)

        if not df.empty:
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)

        return df

    def _update_cache(self, ticker: str, bars: List[AggregateBar]):
        """
        Update bar cache for a ticker.

        Args:
            ticker: Stock ticker symbol
            bars: List of new bars
        """
        if ticker not in self.bar_cache:
            self.bar_cache[ticker] = []

        self.bar_cache[ticker].extend(bars)

        # Limit cache size
        if len(self.bar_cache[ticker]) > self.config.max_cache_size:
            self.bar_cache[ticker] = self.bar_cache[ticker][
                -self.config.max_cache_size :
            ]

    def clear_cache(self, ticker: Optional[str] = None):
        """
        Clear bar cache.

        Args:
            ticker: Specific ticker to clear, or None for all
        """
        if ticker:
            self.bar_cache.pop(ticker, None)
        else:
            self.bar_cache.clear()


# Factory function
def create_market_data_provider(api_key: str, **kwargs) -> MarketDataProvider:
    """
    Factory function to create MarketDataProvider.

    Args:
        api_key: Polygon API key
        **kwargs: Additional config parameters

    Returns:
        Configured MarketDataProvider instance
    """
    config = MarketDataConfig(polygon_api_key=api_key, **kwargs)
    return MarketDataProvider(config)
