"""
Unified Data Pipeline - Polygon API Integration

This module orchestrates all data collection, processing, and storage operations
using Polygon APIs. It provides a unified interface for:

- Real-time LULD halt detection and monitoring
- Historical and real-time market data (bars, quotes)
- Benzinga news feed integration
- Ticker metadata and fundamentals
- Data validation, persistence, and health monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from .polygon_client import PolygonClient, LULDMessage, BenzingaNews, AggregateBar
from .halt_detector import HaltEvent
from .news_feed import NewsContext
from .halt_detector import HaltDetector
from .market_data import MarketDataProvider, MarketDataConfig
from .news_feed import NewsFeedProvider
from .metadata_sync import MetadataSync

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Pipeline operational status"""

    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


class DataSource(Enum):
    """Available data sources"""

    LULD_HALT = "luld_halt"
    MARKET_DATA = "market_data"
    NEWS_FEED = "news_feed"
    METADATA = "metadata"


@dataclass
class PipelineConfig:
    """Configuration for the unified data pipeline"""

    polygon_api_key: str

    # Database configuration
    mongo_uri: str = "mongodb://localhost:27017/"
    database_name: str = "halt_detector"

    # Data sources to enable
    enabled_sources: List[DataSource] = field(
        default_factory=lambda: [
            DataSource.LULD_HALT,
            DataSource.MARKET_DATA,
            DataSource.NEWS_FEED,
            DataSource.METADATA,
        ]
    )

    # Halt monitoring
    tickers_to_monitor: List[str] = field(
        default_factory=lambda: [
            "SPY",
            "QQQ",
            "AAPL",
            "TSLA",
            "NVDA",
            "META",
            "AMZN",
            "MSFT",
        ]
    )
    halt_monitoring_enabled: bool = True

    # Market data
    default_timeframe: str = "5min"
    bar_cache_size: int = 1000
    real_time_data_enabled: bool = True

    # News feed
    news_cache_ttl_minutes: int = 5
    breaking_news_enabled: bool = True

    # Metadata sync
    metadata_sync_interval_hours: int = 24
    max_metadata_age_hours: int = 24

    # Health monitoring
    health_check_interval_minutes: int = 5
    error_retry_delay_seconds: int = 30
    max_retry_attempts: int = 3


@dataclass
class PipelineMetrics:
    """Real-time pipeline performance metrics"""

    uptime_seconds: float = 0.0
    messages_processed: int = 0
    errors_encountered: int = 0
    halts_detected: int = 0
    bars_fetched: int = 0
    news_items_collected: int = 0
    metadata_syncs_performed: int = 0
    last_health_check: Optional[datetime] = None
    data_sources_status: Dict[str, str] = field(default_factory=dict)


class UnifiedDataPipeline:
    """
    Unified data pipeline that orchestrates all Polygon API integrations.

    Features:
    - Real-time LULD halt monitoring and callbacks
    - Market data ingestion and caching
    - News feed collection with filtering
    - Metadata synchronization
    - Health monitoring and error handling
    - Data persistence to MongoDB
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the unified data pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.status = PipelineStatus.INITIALIZING
        self.start_time = datetime.utcnow()

        # Core clients and providers
        self.polygon_client = PolygonClient(config.polygon_api_key)
        self.mongo_client = MongoClient(config.mongo_uri)
        self.database = self.mongo_client[config.database_name]

        # Data providers (lazy initialization)
        self._halt_detector: Optional[HaltDetector] = None
        self._market_data: Optional[MarketDataProvider] = None
        self._news_feed: Optional[NewsFeedProvider] = None
        self._metadata_sync: Optional[MetadataSync] = None

        # Event callbacks
        self.halt_callbacks: List[Callable[[HaltEvent], None]] = []
        self.luld_callbacks: List[Callable[[LULDMessage], None]] = []
        self.news_callbacks: List[Callable[[BenzingaNews], None]] = []
        self.error_callbacks: List[Callable[[Exception, str], None]] = []

        # Monitoring
        self.metrics = PipelineMetrics()
        self.health_task: Optional[asyncio.Task] = None
        self.maintenance_task: Optional[asyncio.Task] = None

        # Control flags
        self._running = False
        self._shutdown_requested = False

        logger.info("UnifiedDataPipeline initialized")

    async def start(self) -> bool:
        """
        Start the data pipeline and all enabled data sources.

        Returns:
            True if started successfully, False otherwise
        """
        try:
            logger.info("Starting unified data pipeline...")
            self.status = PipelineStatus.INITIALIZING
            self._running = True

            # Initialize all enabled providers
            await self._initialize_providers()

            # Start health monitoring
            self.health_task = asyncio.create_task(self._health_monitor())

            # Start maintenance tasks
            self.maintenance_task = asyncio.create_task(self._maintenance_loop())

            # Start real-time data collection
            if DataSource.LULD_HALT in self.config.enabled_sources:
                await self._start_halt_monitoring()

            self.status = PipelineStatus.RUNNING
            logger.info("Data pipeline started successfully")

            # Keep pipeline running until shutdown
            await self._main_loop()

            return True

        except Exception as e:
            logger.error(f"Failed to start data pipeline: {e}")
            self.status = PipelineStatus.ERROR
            await self._cleanup()
            return False

    async def stop(self):
        """Stop the data pipeline gracefully"""
        logger.info("Stopping data pipeline...")
        self._shutdown_requested = True
        self.status = PipelineStatus.SHUTTING_DOWN

        await self._cleanup()

        self.status = PipelineStatus.INITIALIZING
        logger.info("Data pipeline stopped")

    async def _initialize_providers(self):
        """Initialize all configured data providers"""

        # Halt detector
        if DataSource.LULD_HALT in self.config.enabled_sources:
            self._halt_detector = HaltDetector(self.config.polygon_api_key)
            self._halt_detector.set_halt_callback(self._on_halt_detected)
            logger.info("Halt detector initialized")

        # Market data provider
        if DataSource.MARKET_DATA in self.config.enabled_sources:
            market_config = MarketDataConfig(
                polygon_api_key=self.config.polygon_api_key,
                default_timeframe=self.config.default_timeframe,
                cache_bars=True,
                max_cache_size=self.config.bar_cache_size,
            )
            self._market_data = MarketDataProvider(market_config)
            logger.info("Market data provider initialized")

        # News feed
        if DataSource.NEWS_FEED in self.config.enabled_sources:
            self._news_feed = NewsFeedProvider(self.config.polygon_api_key)
            logger.info("News feed provider initialized")

        # Metadata sync
        if DataSource.METADATA in self.config.enabled_sources:
            self._metadata_sync = MetadataSync(
                self.config.polygon_api_key, self.database
            )
            logger.info("Metadata sync initialized")

    async def _start_halt_monitoring(self):
        """Start real-time halt monitoring"""
        if self._halt_detector and self.config.tickers_to_monitor:
            logger.info(
                f"Starting halt monitoring for {len(self.config.tickers_to_monitor)} tickers"
            )
            await self._halt_detector.start_monitoring(self.config.tickers_to_monitor)

    async def _main_loop(self):
        """Main pipeline loop - keeps pipeline alive"""
        while self._running and not self._shutdown_requested:
            try:
                # Check for maintenance tasks
                await asyncio.sleep(1)

                # Update metrics
                self.metrics.uptime_seconds = (
                    datetime.utcnow() - self.start_time
                ).total_seconds()

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                self.metrics.errors_encountered += 1

                # Call error callbacks
                for callback in self.error_callbacks:
                    try:
                        callback(e, "main_loop")
                    except Exception as cb_error:
                        logger.error(f"Error in error callback: {cb_error}")

                await asyncio.sleep(self.config.error_retry_delay_seconds)

    async def _health_monitor(self):
        """Periodic health monitoring of all data sources"""
        while self._running and not self._shutdown_requested:
            try:
                await asyncio.sleep(self.config.health_check_interval_minutes * 60)

                health_status = await self._check_health()
                self.metrics.last_health_check = datetime.utcnow()
                self.metrics.data_sources_status = health_status

                # Log health status
                unhealthy = [
                    source
                    for source, status in health_status.items()
                    if status != "healthy"
                ]
                if unhealthy:
                    logger.warning(f"Unhealthy data sources: {', '.join(unhealthy)}")
                else:
                    logger.info("All data sources healthy")

            except Exception as e:
                logger.error(f"Health monitoring error: {e}")

    async def _maintenance_loop(self):
        """Periodic maintenance tasks"""
        while self._running and not self._shutdown_requested:
            try:
                # Metadata sync (daily)
                if self._metadata_sync:
                    await asyncio.sleep(self.config.metadata_sync_interval_hours * 3600)
                    await self._perform_metadata_sync()

                # Cache cleanup
                await self._cleanup_caches()

            except Exception as e:
                logger.error(f"Maintenance task error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

    async def _perform_metadata_sync(self):
        """Sync metadata for active tickers"""
        try:
            result = self._metadata_sync.sync_active_tickers()
            self.metrics.metadata_syncs_performed += 1
            logger.info(
                f"Metadata sync completed: {result.get('successful', 0)} successful"
            )
        except Exception as e:
            logger.error(f"Metadata sync failed: {e}")

    async def _cleanup_caches(self):
        """Clean up expired cache entries"""
        # This would implement cache cleanup logic for each provider
        pass

    # Event handlers

    def _on_halt_detected(self, halt_event: HaltEvent):
        """Handle halt detection event"""
        self.metrics.halts_detected += 1

        # Persist to database
        self._persist_halt_event(halt_event)

        # Call registered callbacks
        for callback in self.halt_callbacks:
            try:
                callback(halt_event)
            except Exception as e:
                logger.error(f"Error in halt callback: {e}")

    # Data access methods

    async def get_historical_bars(
        self,
        ticker: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        timeframe: Optional[str] = None,
    ) -> List[AggregateBar]:
        """Get historical bars for a ticker"""
        if not self._market_data:
            raise RuntimeError("Market data provider not initialized")

        if timeframe is None:
            timeframe = self.config.default_timeframe

        df = self._market_data.get_bars(
            ticker=ticker, timeframe=timeframe, start_time=start_time, end_time=end_time
        )

        # Convert DataFrame back to AggregateBar list (simplified)
        bars = []
        for idx, row in df.iterrows():
            bar = AggregateBar(
                ticker=ticker,
                timestamp=idx,  # Use index (timestamp) instead of column
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                vwap=row["vwap"],
                transactions=row.get("transactions", 0),
            )
            bars.append(bar)

        return bars

    async def get_news_for_ticker(
        self, ticker: str, lookback_hours: int = 24
    ) -> List[BenzingaNews]:
        """Get news for a ticker"""
        if not self._news_feed:
            raise RuntimeError("News feed provider not initialized")

        return self._news_feed.get_news_for_ticker(ticker, lookback_hours)

    async def get_halt_context_news(
        self, ticker: str, halt_time: datetime
    ) -> NewsContext:
        """Get news context around a halt event"""
        if not self._news_feed:
            raise RuntimeError("News feed provider not initialized")

        return self._news_feed.get_halt_context_news(ticker, halt_time)

    def get_ticker_metadata(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get ticker metadata"""
        if not self._metadata_sync:
            raise RuntimeError("Metadata sync not initialized")

        return self._metadata_sync.get_ticker_metadata(ticker)

    # Callback registration

    def on_halt(self, callback: Callable[[HaltEvent], None]):
        """Register halt detection callback"""
        self.halt_callbacks.append(callback)

    def on_luld(self, callback: Callable[[LULDMessage], None]):
        """Register LULD message callback"""
        self.luld_callbacks.append(callback)

    def on_news(self, callback: Callable[[BenzingaNews], None]):
        """Register news callback"""
        self.news_callbacks.append(callback)

    def on_error(self, callback: Callable[[Exception, str], None]):
        """Register error callback"""
        self.error_callbacks.append(callback)

    # Health and monitoring

    async def _check_health(self) -> Dict[str, str]:
        """Check health of all data sources"""
        health_status = {}

        try:
            # Halt detector health
            if self._halt_detector:
                is_monitoring = self._halt_detector.is_monitoring_active()
                health_status["halt_detector"] = (
                    "healthy" if is_monitoring else "unhealthy"
                )
            else:
                health_status["halt_detector"] = "disabled"

            # Market data health
            if self._market_data:
                # Test with a simple query
                test_data = self._market_data.get_latest_bar("SPY")
                health_status["market_data"] = "healthy" if test_data else "degraded"
            else:
                health_status["market_data"] = "disabled"

            # Database health
            try:
                self.database.command("ping")
                health_status["database"] = "healthy"
            except PyMongoError:
                health_status["database"] = "unhealthy"

        except Exception as e:
            logger.error(f"Health check error: {e}")
            health_status["overall"] = "error"

        return health_status

    def get_metrics(self) -> PipelineMetrics:
        """Get current pipeline metrics"""
        return self.metrics

    def get_status(self) -> Dict[str, Any]:
        """Get detailed pipeline status"""
        return {
            "status": self.status.value,
            "uptime_seconds": self.metrics.uptime_seconds,
            "enabled_sources": [source.value for source in self.config.enabled_sources],
            "tickers_monitored": len(self.config.tickers_to_monitor)
            if self._halt_detector
            else 0,
            "metrics": {
                "messages_processed": self.metrics.messages_processed,
                "halts_detected": self.metrics.halts_detected,
                "errors": self.metrics.errors_encountered,
            },
            "data_sources_status": self.metrics.data_sources_status,
            "last_health_check": self.metrics.last_health_check.isoformat()
            if self.metrics.last_health_check
            else None,
        }

    def _persist_halt_event(self, halt_event: HaltEvent):
        """Persist halt event to database"""
        try:
            halts_collection = self.database.halts
            halts_collection.insert_one(halt_event.to_dict())
        except PyMongoError as e:
            logger.error(f"Failed to persist halt event: {e}")

    async def _cleanup(self):
        """Cleanup resources on shutdown"""
        logger.info("Cleaning up pipeline resources...")

        # Cancel background tasks
        if self.health_task:
            self.health_task.cancel()
        if self.maintenance_task:
            self.maintenance_task.cancel()

        # Stop halt monitoring
        if self._halt_detector:
            await self._halt_detector.stop_monitoring()

        # Close MongoDB connection
        self.mongo_client.close()

        self._running = False
        logger.info("Pipeline cleanup completed")


# Factory function
def create_data_pipeline(config: PipelineConfig) -> UnifiedDataPipeline:
    """Factory function to create unified data pipeline"""
    return UnifiedDataPipeline(config)


# Example usage
async def example_pipeline_usage():
    """Example demonstrating pipeline usage"""

    config = PipelineConfig(
        polygon_api_key="your_polygon_api_key_here",
        tickers_to_monitor=["SPY", "AAPL", "TSLA"],
    )

    pipeline = create_data_pipeline(config)

    # Register event callbacks
    def handle_halt(halt_event: HaltEvent):
        print(f"🚨 Halt detected: {halt_event.ticker} - {halt_event.halt_type.value}")

    def handle_errors(error: Exception, source: str):
        print(f"⚠️ Error in {source}: {error}")

    pipeline.on_halt(handle_halt)
    pipeline.on_error(handle_errors)

    try:
        # Start the pipeline
        success = await pipeline.start()

        if success:
            print("Pipeline running... (Ctrl+C to stop)")
            # Pipeline runs until interrupted

    except KeyboardInterrupt:
        print("\nShutting down pipeline...")
        await pipeline.stop()
        print("Pipeline shutdown complete")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_pipeline_usage())
