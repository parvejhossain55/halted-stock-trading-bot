"""
Halt Detector Data Module - Polygon API Integrations

This module provides comprehensive integration with Polygon APIs for real-time
and historical financial data, including LULD halts, market data, news feeds,
and ticker metadata.

Main Components:
- PolygonClient: Unified REST and WebSocket API client
- HaltDetector: Real-time LULD halt monitoring
- MarketDataProvider: OHLCV data and technical indicators
- NewsFeedProvider: Benzinga news integration
- MetadataSync: Ticker fundamentals and market data
- UnifiedDataPipeline: Orchestrated data collection pipeline
"""

from dotenv import load_dotenv

load_dotenv()

from .polygon_client import (
    PolygonClient,
    PolygonRestClient,
    PolygonWebSocketClient,
    LULDMessage,
    BenzingaNews,
    AggregateBar,
    LULDIndicator,
    LULDType,
)

from .halt_detector import (
    HaltDetector,
    HaltEvent,
    HaltType as HaltDetectionType,  # Alias for clarity
    HaltStatus,
    WeaknessSignal,
    StrengthSignal,
    HaltMonitoringContext,
    create_halt_detector,
)

from .market_data import (
    MarketDataProvider,
    MarketDataConfig,
    create_market_data_provider,
)

from .news_feed import NewsFeedProvider, NewsContext, create_news_feed_provider

from .metadata_sync import MetadataSync, create_metadata_sync

from .data_pipeline import (
    UnifiedDataPipeline,
    PipelineConfig,
    PipelineMetrics,
    PipelineStatus,
    DataSource,
    create_data_pipeline,
)

__all__ = [
    # Main clients
    "PolygonClient",
    "PolygonRestClient",
    "PolygonWebSocketClient",
    # Data structures
    "LULDMessage",
    "BenzingaNews",
    "AggregateBar",
    "HaltEvent",
    "NewsContext",
    "LULDIndicator",
    "LULDType",
    # Halt detection
    "HaltDetector",
    "HaltEvent",
    "HaltDetectionType",
    "HaltStatus",
    "WeaknessSignal",
    "StrengthSignal",
    "HaltMonitoringContext",
    # Data providers
    "MarketDataProvider",
    "MarketDataConfig",
    "NewsFeedProvider",
    "MetadataSync",
    # Pipeline
    "UnifiedDataPipeline",
    "PipelineConfig",
    "PipelineMetrics",
    "PipelineStatus",
    "DataSource",
    # Factory functions
    "create_halt_detector",
    "create_market_data_provider",
    "create_news_feed_provider",
    "create_metadata_sync",
    "create_data_pipeline",
]

# Module version info
__version__ = "1.0.0"
__api_version__ = "polygon_v3"

# Default configurations for easy setup
DEFAULT_PIPELINE_CONFIG = PipelineConfig(
    polygon_api_key="",  # Must be set by user
    mongo_uri="mongodb://localhost:27017/",
    database_name="halt_detector",
)

DEFAULT_MARKET_CONFIG = MarketDataConfig(
    polygon_api_key="",  # Must be set by user
    default_timeframe="5min",
    cache_bars=True,
    max_cache_size=1000,
)
