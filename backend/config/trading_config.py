"""
Trading Configuration
Central configuration for all trading parameters, risk controls, and strategy settings.
"""
from enum import Enum


class TradingMode(Enum):
    """Trading mode enumeration"""

    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"


class CatalystType(Enum):
    """Catalyst types for halt classification"""

    NO_NEWS = "no_news"
    EARNINGS = "earnings"
    FDA = "fda"
    IPO_SPAC = "ipo_spac"
    MERGER = "merger"
    OFFERING = "offering"
    BUYOUT = "buyout"
    CLINICAL_TRIAL = "clinical_trial"
    PARTNERSHIP = "partnership"
    OTHER = "other"


class HaltType(Enum):
    """Types of trading halts"""

    HALT_UP = "halt_up"
    HALT_DOWN = "halt_down"
    GAP_UP = "gap_up"
    GAP_DOWN = "gap_down"


class TradingConfig:
    """Main trading configuration class"""

    # ============================================================
    # TRADING MODE
    # ============================================================
    TRADING_MODE = TradingMode.PAPER

    # ============================================================
    # MARKET DATA
    # ============================================================
    # Data providers
    MARKET_DATA_PROVIDER = "polygon"  # polygon etc.
    NEWS_PROVIDER = "polygon-benzinga"

    # Bar intervals
    BAR_INTERVAL_MINUTES = 5

    # Market hours (Eastern Time)
    MARKET_OPEN_HOUR = 9
    MARKET_OPEN_MINUTE = 30
    MARKET_CLOSE_HOUR = 16
    MARKET_CLOSE_MINUTE = 0

    # Pre/Post market
    ALLOW_PREMARKET_TRADING = False
    ALLOW_AFTERHOURS_TRADING = False

    # ============================================================
    # STOCK FILTERS
    # ============================================================
    # Price range
    MIN_STOCK_PRICE = 5.0
    MAX_STOCK_PRICE = 300.0

    # Volume and liquidity
    MIN_AVERAGE_DAILY_VOLUME = 500_000
    MIN_HALT_VOLUME = 1_000_000

    # Float
    MIN_FLOAT = 1_000_000
    MAX_FLOAT = 50_000_000

    # Market cap (optional filter)
    MIN_MARKET_CAP = None  # None = no limit
    MAX_MARKET_CAP = 5_000_000_000  # $5B

    # ============================================================
    # HALT DETECTION
    # ============================================================
    # Halt detection settings
    HALT_DETECTION_METHOD = "official"  # official, derived, or both
    MAX_HALTS_PER_TICKER_PER_DAY = 5
    MIN_HALT_DURATION_SECONDS = 300  # 5 minutes

    # Gap thresholds
    MIN_GAP_PERCENT = 5.0  # Minimum gap % to consider
    LARGE_GAP_DOWN_PERCENT = 30.0  # Avoid entry on gaps larger than this


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def get_config(key: str, default=None):
    """Get a configuration value by key"""
    return getattr(TradingConfig, key, default)


def update_config(key: str, value):
    """Update a configuration value (runtime only)"""
    if hasattr(TradingConfig, key):
        setattr(TradingConfig, key, value)
        return True
    return False
