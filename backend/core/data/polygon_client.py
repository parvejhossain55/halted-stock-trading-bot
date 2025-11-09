"""
Polygon.io API Client
Comprehensive integration for LULD halts, Benzinga news, and market data aggregates.

Official Documentation:
- LULD WebSocket: https://massive.com/docs/websocket/stocks/luld
- Benzinga News REST: https://massive.com/docs/rest/partners/benzinga/news
- Aggregates REST: https://massive.com/docs/rest/stocks/aggregates/custom-bars
"""

import asyncio
import json
import logging
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import requests
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================


class LULDIndicator(str, Enum):
    """LULD Indicator values"""

    NONE = ""  # No LULD in effect
    LIMIT_UP_LIMIT_DOWN = "A"  # Limit Up-Limit Down Price Band
    LIMIT_UP = "B"  # Limit Up Price Band
    LIMIT_DOWN = "C"  # Limit Down Price Band


class LULDType(str, Enum):
    """Type of LULD message"""

    LIMIT_UP_DOWN = "limit_up_down"
    LIMIT_UP = "limit_up"
    LIMIT_DOWN = "limit_down"
    CLEARED = "cleared"


@dataclass
class LULDMessage:
    """LULD (Limit Up-Limit Down) message from WebSocket"""

    event_type: str  # "LULD"
    ticker: str
    indicator: LULDIndicator
    limit_up_price: float
    limit_down_price: float
    timestamp: datetime
    exchange: str
    tape: str

    # Derived fields
    luld_type: LULDType = field(init=False)
    price_band_width: float = field(init=False)

    def __post_init__(self):
        """Calculate derived fields"""
        # Determine LULD type
        if self.indicator == LULDIndicator.NONE:
            self.luld_type = LULDType.CLEARED
        elif self.indicator == LULDIndicator.LIMIT_UP:
            self.luld_type = LULDType.LIMIT_UP
        elif self.indicator == LULDIndicator.LIMIT_DOWN:
            self.luld_type = LULDType.LIMIT_DOWN
        else:
            self.luld_type = LULDType.LIMIT_UP_DOWN

        # Calculate band width
        self.price_band_width = self.limit_up_price - self.limit_down_price

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event_type": self.event_type,
            "ticker": self.ticker,
            "indicator": self.indicator.value,
            "limit_up_price": self.limit_up_price,
            "limit_down_price": self.limit_down_price,
            "timestamp": self.timestamp.isoformat(),
            "exchange": self.exchange,
            "tape": self.tape,
            "luld_type": self.luld_type.value,
            "price_band_width": self.price_band_width,
        }


@dataclass
class BenzingaNews:
    """Benzinga news article from Polygon"""

    id: str
    publisher: str
    title: str
    author: str
    published_utc: datetime
    article_url: str
    tickers: List[str]
    amp_url: Optional[str] = None
    image_url: Optional[str] = None
    description: Optional[str] = None
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "publisher": self.publisher,
            "title": self.title,
            "author": self.author,
            "published_utc": self.published_utc.isoformat(),
            "article_url": self.article_url,
            "tickers": self.tickers,
            "amp_url": self.amp_url,
            "image_url": self.image_url,
            "description": self.description,
            "keywords": self.keywords,
        }


@dataclass
class AggregateBar:
    """Aggregate (OHLCV) bar data"""

    ticker: str
    timestamp: datetime  # Bar open timestamp
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    transactions: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap,
            "transactions": self.transactions,
        }


# ============================================================================
# POLYGON REST API CLIENT
# ============================================================================


class PolygonRestClient:
    """
    Polygon REST API client for news and aggregates.

    Implements:
    - Benzinga News (via Polygon partnership)
    - Custom Aggregate Bars (OHLCV data)
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str):
        """
        Initialize Polygon REST client.

        Args:
            api_key: Polygon API key
        """
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def get_benzinga_news(
        self,
        ticker: Optional[str] = None,
        published_utc_gte: Optional[datetime] = None,
        published_utc_lte: Optional[datetime] = None,
        order: str = "desc",
        limit: int = 100,
        sort: str = "published_utc",
    ) -> List[BenzingaNews]:
        """
        Get Benzinga news articles via Polygon partnership.

        Reference: https://massive.com/docs/rest/partners/benzinga/news

        Args:
            ticker: Filter by ticker symbol (e.g., "AAPL")
            published_utc_gte: Published on or after this datetime
            published_utc_lte: Published on or before this datetime
            order: Sort order ("asc" or "desc")
            limit: Number of results (max 1000)
            sort: Sort field (default: "published_utc")

        Returns:
            List of BenzingaNews objects
        """
        endpoint = "/benzinga/v2/news"

        params = {"order": order, "limit": min(limit, 1000), "sort": sort}

        if ticker:
            params["ticker"] = ticker.upper()

        if published_utc_gte:
            params["published_utc.gte"] = published_utc_gte.strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            )

        if published_utc_lte:
            params["published_utc.lte"] = published_utc_lte.strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            )

        try:
            url = f"{self.BASE_URL}{endpoint}"
            logger.debug(f"Fetching Benzinga news: {url}")

            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])

            news_items = []
            for item in results:
                try:
                    news = BenzingaNews(
                        id=item["id"],
                        publisher=item.get("publisher", {}).get("name", "Benzinga"),
                        title=item["title"],
                        author=item.get("author", "Unknown"),
                        published_utc=datetime.fromisoformat(
                            item["published_utc"].replace("Z", "+00:00")
                        ),
                        article_url=item.get("article_url", ""),
                        tickers=item.get("tickers", []),
                        amp_url=item.get("amp_url"),
                        image_url=item.get("image_url"),
                        description=item.get("description"),
                        keywords=item.get("keywords", []),
                    )
                    news_items.append(news)
                except Exception as e:
                    logger.warning(f"Error parsing news item: {e}")
                    continue

            logger.info(f"Retrieved {len(news_items)} Benzinga news articles")
            return news_items

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Benzinga news: {e}")
            return []

    def get_aggregate_bars(
        self,
        ticker: str,
        multiplier: int,
        timespan: str,
        from_date: datetime,
        to_date: datetime,
        adjusted: bool = True,
        sort: str = "asc",
        limit: int = 50000,
    ) -> List[AggregateBar]:
        """
        Get aggregate bars (OHLCV) for a ticker.

        Reference: https://massive.com/docs/rest/stocks/aggregates/custom-bars

        Args:
            ticker: Stock ticker symbol
            multiplier: Size of timespan multiplier (e.g., 1 for 1-minute, 5 for 5-minute)
            timespan: Size of time window ("minute", "hour", "day", "week", "month", "quarter", "year")
            from_date: Start of aggregate window
            to_date: End of aggregate window
            adjusted: Whether results are adjusted for splits
            sort: Sort order ("asc" or "desc")
            limit: Limits number of base aggregates (max 50000)

        Returns:
            List of AggregateBar objects
        """
        # Format: /v2/aggs/ticker/{stocksTicker}/range/{multiplier}/{timespan}/{from}/{to}
        from_str = from_date.strftime("%Y-%m-%d")
        to_str = to_date.strftime("%Y-%m-%d")

        endpoint = f"/v2/aggs/ticker/{ticker.upper()}/range/{multiplier}/{timespan}/{from_str}/{to_str}"

        params = {
            "adjusted": str(adjusted).lower(),
            "sort": sort,
            "limit": min(limit, 50000),
        }

        try:
            url = f"{self.BASE_URL}{endpoint}"
            logger.debug(f"Fetching aggregates: {url}")

            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if data.get("status") != "OK":
                logger.warning(f"Polygon returned non-OK status: {data.get('status')}")
                return []

            results = data.get("results", [])
            bars = []

            for bar_data in results:
                try:
                    # Polygon returns timestamps in milliseconds
                    timestamp_ms = bar_data["t"]
                    timestamp = datetime.fromtimestamp(timestamp_ms / 1000)

                    bar = AggregateBar(
                        ticker=ticker.upper(),
                        timestamp=timestamp,
                        open=bar_data["o"],
                        high=bar_data["h"],
                        low=bar_data["l"],
                        close=bar_data["c"],
                        volume=bar_data["v"],
                        vwap=bar_data.get("vw", 0.0),
                        transactions=bar_data.get("n", 0),
                    )
                    bars.append(bar)
                except Exception as e:
                    logger.warning(f"Error parsing bar data: {e}")
                    continue

            logger.info(f"Retrieved {len(bars)} aggregate bars for {ticker}")
            return bars

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching aggregates: {e}")
            return []

    def get_ticker_details(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get ticker details including float, market cap, etc.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with ticker details or None
        """
        endpoint = f"/v3/reference/tickers/{ticker.upper()}"

        try:
            url = f"{self.BASE_URL}{endpoint}"
            response = self.session.get(url)
            response.raise_for_status()

            data = response.json()

            if data.get("status") != "OK":
                return None

            return data.get("results", {})

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching ticker details: {e}")
            return None


# ============================================================================
# POLYGON WEBSOCKET CLIENT (LULD)
# ============================================================================


class PolygonWebSocketClient:
    """
    Polygon WebSocket client for real-time LULD (Limit Up-Limit Down) messages.

    Reference: https://massive.com/docs/websocket/stocks/luld
    """

    WS_URL = "wss://socket.polygon.io/stocks"

    def __init__(self, api_key: str):
        """
        Initialize Polygon WebSocket client.

        Args:
            api_key: Polygon API key
        """
        self.api_key = api_key
        self.websocket = None
        self.is_connected = False
        self.subscriptions = set()
        self.luld_callback: Optional[Callable[[LULDMessage], None]] = None
        self._running = False

    async def connect(self):
        """Establish WebSocket connection"""
        try:
            self.websocket = await websockets.connect(self.WS_URL)
            self.is_connected = True
            logger.info("Connected to Polygon WebSocket")

            # Authenticate - Polygon expects "params" field with API key
            auth_message = {"action": "auth", "params": self.api_key}
            await self.websocket.send(json.dumps(auth_message))

            # Wait for auth confirmation - may receive multiple messages
            authenticated = False
            timeout = 10  # seconds
            start_time = asyncio.get_event_loop().time()

            while asyncio.get_event_loop().time() - start_time < timeout:
                try:
                    response = await asyncio.wait_for(
                        self.websocket.recv(), timeout=5.0
                    )
                    auth_data = json.loads(response)

                    logger.debug(f"Received WebSocket message: {auth_data}")

                    # Handle list or dict response
                    if isinstance(auth_data, list):
                        for msg in auth_data:
                            status = msg.get("status")
                            if status == "auth_success":
                                authenticated = True
                                logger.info("WebSocket authentication successful")
                                break
                            elif status == "auth_failed":
                                logger.error(
                                    f"WebSocket authentication failed: {msg.get('message', 'Unknown error')}"
                                )
                                break
                            elif status == "connected":
                                logger.info(
                                    f"WebSocket connected: {msg.get('message')}"
                                )
                                # Continue waiting for auth_success
                    else:
                        status = auth_data.get("status")
                        if status == "auth_success":
                            authenticated = True
                            logger.info("WebSocket authentication successful")
                        elif status == "auth_failed":
                            logger.error(
                                f"WebSocket authentication failed: {auth_data.get('message', 'Unknown error')}"
                            )
                            break
                        elif status == "connected":
                            logger.info(
                                f"WebSocket connected: {auth_data.get('message')}"
                            )
                            # Continue waiting for auth_success

                    if authenticated:
                        break

                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for WebSocket auth response")
                    break
                except Exception as e:
                    logger.error(f"Error during WebSocket auth: {e}")
                    break

            if not authenticated:
                logger.error("WebSocket authentication failed or timed out")
                self.is_connected = False

        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            self.is_connected = False

    async def subscribe_luld(self, tickers: List[str]):
        """
        Subscribe to LULD messages for specific tickers.

        Args:
            tickers: List of ticker symbols (e.g., ["AAPL", "TSLA"])
        """
        if not self.is_connected:
            logger.error("Not connected to WebSocket")
            return

        # Format: LULD.{ticker}
        luld_channels = [f"LULD.{ticker.upper()}" for ticker in tickers]

        subscribe_message = {"action": "subscribe", "params": ",".join(luld_channels)}

        await self.websocket.send(json.dumps(subscribe_message))
        self.subscriptions.update(luld_channels)

        logger.info(f"Subscribed to LULD for: {', '.join(tickers)}")

    async def unsubscribe_luld(self, tickers: List[str]):
        """
        Unsubscribe from LULD messages.

        Args:
            tickers: List of ticker symbols
        """
        if not self.is_connected:
            return

        luld_channels = [f"LULD.{ticker.upper()}" for ticker in tickers]

        unsubscribe_message = {
            "action": "unsubscribe",
            "params": ",".join(luld_channels),
        }

        await self.websocket.send(json.dumps(unsubscribe_message))
        self.subscriptions.difference_update(luld_channels)

        logger.info(f"Unsubscribed from LULD for: {', '.join(tickers)}")

    def on_luld(self, callback: Callable[[LULDMessage], None]):
        """
        Register callback for LULD messages.

        Args:
            callback: Function to call when LULD message received
        """
        self.luld_callback = callback

    async def listen(self):
        """
        Listen for incoming WebSocket messages.
        This is a blocking call - run in asyncio task.
        """
        self._running = True

        try:
            while self._running and self.is_connected:
                try:
                    message = await asyncio.wait_for(
                        self.websocket.recv(), timeout=30.0
                    )

                    data = json.loads(message)

                    # Handle different message types
                    for item in data:
                        ev = item.get("ev")  # Event type

                        if ev == "status":
                            logger.info(f"WebSocket status: {item.get('message')}")

                        elif ev == "LULD":
                            # Parse LULD message
                            await self._handle_luld(item)

                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    if self.is_connected:
                        await self.websocket.send(json.dumps({"action": "ping"}))

                except Exception as e:
                    logger.error(f"Error receiving message: {e}")
                    break

        except Exception as e:
            logger.error(f"Error in listen loop: {e}")

        finally:
            self._running = False
            self.is_connected = False

    async def _handle_luld(self, data: Dict[str, Any]):
        """
        Handle LULD message from WebSocket.

        LULD Message Format:
        {
            "ev": "LULD",
            "sym": "AAPL",
            "i": "A",  # Indicator
            "lu": 150.50,  # Limit up price
            "ld": 145.50,  # Limit down price
            "t": 1234567890000,  # Timestamp (nanoseconds)
            "x": "Q",  # Exchange
            "z": "C"  # Tape
        }
        """
        try:
            # Parse timestamp (nanoseconds to datetime)
            timestamp_ns = data["t"]
            timestamp = datetime.fromtimestamp(timestamp_ns / 1_000_000_000)

            # Map indicator
            indicator_map = {
                "": LULDIndicator.NONE,
                "A": LULDIndicator.LIMIT_UP_LIMIT_DOWN,
                "B": LULDIndicator.LIMIT_UP,
                "C": LULDIndicator.LIMIT_DOWN,
            }

            indicator = indicator_map.get(data.get("i", ""), LULDIndicator.NONE)

            luld_msg = LULDMessage(
                event_type="LULD",
                ticker=data["sym"],
                indicator=indicator,
                limit_up_price=data["lu"],
                limit_down_price=data["ld"],
                timestamp=timestamp,
                exchange=data.get("x", ""),
                tape=data.get("z", ""),
            )

            logger.info(
                f"LULD Alert: {luld_msg.ticker} - {luld_msg.luld_type.value} | "
                f"Band: ${luld_msg.limit_down_price:.2f} - ${luld_msg.limit_up_price:.2f}"
            )

            # Call registered callback
            if self.luld_callback:
                self.luld_callback(luld_msg)

        except Exception as e:
            logger.error(f"Error handling LULD message: {e}")

    async def close(self):
        """Close WebSocket connection"""
        self._running = False
        if self.websocket:
            await self.websocket.close()
        self.is_connected = False
        logger.info("WebSocket connection closed")


# ============================================================================
# UNIFIED POLYGON CLIENT
# ============================================================================


class PolygonClient:
    """
    Unified Polygon client combining REST and WebSocket functionality.

    Provides:
    - REST: Benzinga news, aggregate bars, ticker details
    - WebSocket: Real-time LULD messages
    """

    def __init__(self, api_key: str):
        """
        Initialize unified Polygon client.

        Args:
            api_key: Polygon API key
        """
        self.api_key = api_key
        self.rest = PolygonRestClient(api_key)
        self.ws = PolygonWebSocketClient(api_key)

    # REST methods (delegate to REST client)

    def get_news(self, **kwargs) -> List[BenzingaNews]:
        """Get Benzinga news. See PolygonRestClient.get_benzinga_news for args."""
        return self.rest.get_benzinga_news(**kwargs)

    def get_bars(self, **kwargs) -> List[AggregateBar]:
        """Get aggregate bars. See PolygonRestClient.get_aggregate_bars for args."""
        return self.rest.get_aggregate_bars(**kwargs)

    def get_ticker_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get ticker details"""
        return self.rest.get_ticker_details(ticker)

    # WebSocket methods (delegate to WS client)

    async def connect_websocket(self):
        """Connect to WebSocket"""
        await self.ws.connect()

    async def subscribe_luld(self, tickers: List[str]):
        """Subscribe to LULD for tickers"""
        await self.ws.subscribe_luld(tickers)

    def on_luld(self, callback: Callable[[LULDMessage], None]):
        """Register LULD callback"""
        self.ws.on_luld(callback)

    async def start_luld_stream(self):
        """Start listening to LULD stream"""
        await self.ws.listen()

    async def close_websocket(self):
        """Close WebSocket connection"""
        await self.ws.close()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

# async def example_usage():
#     """Example demonstrating Polygon API usage"""

#     # Initialize client
#     api_key = "your_polygon_api_key"
#     client = PolygonClient(api_key)

#     # === REST API Examples ===

#     # 1. Get recent Benzinga news for a ticker
#     print("\n=== Benzinga News ===")
#     news = client.get_news(
#         ticker="AAPL",
#         published_utc_gte=datetime.now() - timedelta(days=1),
#         limit=5
#     )

#     for article in news:
#         print(f"{article.published_utc}: {article.title}")

#     # 2. Get 5-minute aggregate bars
#     print("\n=== Aggregate Bars (5-minute) ===")
#     bars = client.get_bars(
#         ticker="AAPL",
#         multiplier=5,
#         timespan="minute",
#         from_date=datetime.now() - timedelta(days=1),
#         to_date=datetime.now(),
#         limit=10
#     )

#     for bar in bars:
#         print(f"{bar.timestamp}: O={bar.open} H={bar.high} L={bar.low} C={bar.close} V={bar.volume}")

#     # === WebSocket Example ===

#     # Define LULD callback
#     def handle_luld(luld: LULDMessage):
#         print(f"\n🚨 LULD Alert: {luld.ticker}")
#         print(f"   Type: {luld.luld_type.value}")
#         print(f"   Band: ${luld.limit_down_price:.2f} - ${luld.limit_up_price:.2f}")
#         print(f"   Width: ${luld.price_band_width:.2f}")

#     # Connect and subscribe
#     print("\n=== LULD WebSocket Stream ===")
#     await client.connect_websocket()

#     # Register callback
#     client.on_luld(handle_luld)

#     # Subscribe to tickers
#     await client.subscribe_luld(["AAPL", "TSLA", "NVDA"])

#     # Start listening (this blocks)
#     print("Listening for LULD messages... (Ctrl+C to stop)")
#     try:
#         await client.start_luld_stream()
#     except KeyboardInterrupt:
#         print("\nStopping...")
#         await client.close_websocket()


# if __name__ == "__main__":
#     # Run example
#     asyncio.run(example_usage())
