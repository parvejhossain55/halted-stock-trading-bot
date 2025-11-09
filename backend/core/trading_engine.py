"""
Main Trading Engine for Halt Detector System

Orchestrates the entire trading workflow:
1. Data ingestion and halt detection
2. Signal generation and validation
3. Order execution and position management
4. Risk monitoring and kill switch
"""

from typing import Dict, Optional, List
from datetime import datetime, timedelta
import asyncio
import logging
import signal
import sys

from .data.data_pipeline import UnifiedDataPipeline, PipelineConfig
from .strategy.policy_engine import PolicyEngine, TradeDecision
from .execution.das_api import create_das_client
from .execution.order_manager import create_order_manager, ExecutionResult
from ..database.models import (
    TradingSignal,
    HaltEvent,
    SystemLog,
    DailyPerformance,
    init_db,
    create_indexes,
)
from ..config.settings import (
    MONGODB_SETTINGS,
    TRADING_MODE,
    POLYGON_API_KEY,
)


logger = logging.getLogger(__name__)


class TradingEngine:
    """Main trading engine orchestrator"""

    def __init__(self):
        self.running = False
        self.data_pipeline = None
        self.policy_engine = None
        self.das_client = None
        self.order_manager = None
        self.shutdown_event = asyncio.Event()

        # Trading state
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.current_positions = 0

    async def initialize(self) -> bool:
        """
        Initialize all trading components

        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing Halt Detector Trading Engine...")

            # Initialize database
            db_uri = f"mongodb://{MONGODB_SETTINGS['host']}:{MONGODB_SETTINGS['port']}"
            init_db(db_uri, MONGODB_SETTINGS["db"])
            create_indexes()
            logger.info("Database initialized")

            # Initialize data pipeline
            pipeline_config = PipelineConfig(
                polygon_api_key=POLYGON_API_KEY,
                tickers_to_monitor=[
                    "AAPL",
                    "TSLA",
                    "NVDA",
                    "MSFT",
                    "GOOGL",
                ],  # Initial ticker list
                halt_monitoring_enabled=True,
                real_time_data_enabled=True,
            )
            self.data_pipeline = UnifiedDataPipeline(pipeline_config)
            logger.info("Data pipeline initialized")

            # Initialize policy engine
            self.policy_engine = PolicyEngine()
            logger.info("Policy engine initialized")

            # Initialize execution layer
            self.das_client = create_das_client(paper_trading=(TRADING_MODE == "paper"))
            connected = self.das_client.connect()
            if not connected:
                logger.error("Failed to connect to DAS")
                return False

            self.order_manager = create_order_manager(self.das_client)
            logger.info("Execution layer initialized")

            # Set up data pipeline callbacks
            self.data_pipeline.on_halt(self._handle_halt_event)
            self.data_pipeline.on_error(self._handle_pipeline_error)

            logger.info("Trading Engine initialization complete")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize trading engine: {e}")
            return False

    async def start(self) -> bool:
        """
        Start the trading engine

        Returns:
            True if started successfully
        """
        try:
            if not await self.initialize():
                return False

            logger.info("Starting Halt Detector Trading Engine...")
            self.running = True

            # Start data pipeline
            pipeline_started = await self.data_pipeline.start()
            if not pipeline_started:
                logger.error("Failed to start data pipeline")
                return False

            # Start main trading loop
            asyncio.create_task(self._main_trading_loop())
            asyncio.create_task(self._risk_monitoring_loop())
            asyncio.create_task(self._performance_update_loop())

            # Set up signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            logger.info("Trading Engine started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start trading engine: {e}")
            return False

    async def stop(self):
        """Stop the trading engine"""
        logger.info("Stopping Halt Detector Trading Engine...")
        self.running = False
        self.shutdown_event.set()

        # Stop data pipeline
        if self.data_pipeline:
            await self.data_pipeline.stop()

        # Disconnect from DAS
        if self.das_client:
            self.das_client.disconnect()

        logger.info("Trading Engine stopped")

    async def _main_trading_loop(self):
        """Main trading loop - runs continuously"""
        logger.info("Main trading loop started")

        while self.running and not self.shutdown_event.is_set():
            try:
                # Check for exit opportunities on existing positions
                await self._check_exit_opportunities()

                # Small delay to prevent tight looping
                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Error in main trading loop: {e}")
                await asyncio.sleep(10)  # Longer delay on error

    async def _check_exit_opportunities(self):
        """Check for exit opportunities on existing positions"""
        try:
            positions = self.order_manager.get_positions()

            for position in positions:
                # Get current market data for this ticker
                ticker = position["ticker"]
                market_data = await self._get_market_data(ticker)

                if not market_data:
                    continue

                # Evaluate exit opportunities
                exit_decisions = self.policy_engine.evaluate_exit_opportunity(
                    position, market_data
                )

                for exit_decision in exit_decisions:
                    if exit_decision.signal.name != "NO_EXIT":
                        # Execute exit
                        trade_decision = {
                            "action": exit_decision.signal.name,
                            "ticker": ticker,
                            "quantity": exit_decision.partial_exit_percent,
                            "price": exit_decision.exit_price,
                            "confidence": exit_decision.confidence,
                            "rationale": exit_decision.exit_reason,
                        }

                        result = self.order_manager.execute_trade_decision(
                            trade_decision
                        )

                        if result.success:
                            logger.info(
                                f"Exit executed for {ticker}: {exit_decision.exit_reason}"
                            )
                        else:
                            logger.warning(
                                f"Exit failed for {ticker}: {result.message}"
                            )

        except Exception as e:
            logger.error(f"Error checking exit opportunities: {e}")

    async def _handle_halt_event(self, halt_event: HaltEvent):
        """Handle incoming halt events"""
        try:
            logger.info(
                f"Processing halt event: {halt_event.ticker} - {halt_event.halt_type}"
            )

            # Get market data for the halted ticker
            market_data = await self._get_market_data(halt_event.ticker)
            if not market_data:
                logger.warning(f"No market data available for {halt_event.ticker}")
                return

            # Get news context
            news_context = await self._get_news_context(halt_event.ticker)

            # Get account info for risk management
            account_info = self._get_account_info()

            # Evaluate trading opportunity
            trade_decision = self.policy_engine.evaluate_halt_opportunity(
                halt_event.to_dict(), market_data, news_context, account_info
            )

            if trade_decision:
                # Execute the trade
                result = self.order_manager.execute_trade_decision(
                    trade_decision.to_dict()
                )

                if result.success:
                    # Create trading signal record
                    signal = TradingSignal(
                        signal_id=f"sig_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                        ticker=halt_event.ticker,
                        action=trade_decision.action,
                        halt_type=halt_event.halt_type,
                        confidence=trade_decision.confidence,
                        signal_price=trade_decision.price,
                        catalyst_type=getattr(halt_event, "catalyst_type", None),
                        catalyst_confidence=getattr(
                            halt_event, "catalyst_confidence", 0
                        ),
                        suggested_stop_loss=trade_decision.stop_loss,
                        suggested_take_profit=trade_decision.take_profit,
                        suggested_size=trade_decision.quantity,
                        status="executed",
                        execution_notes=result.message,
                        trade_id=result.order_id,
                        halt_event_id=str(halt_event.id),
                    )
                    signal.save()

                    logger.info(
                        f"Trade executed: {trade_decision.action} {trade_decision.quantity} {halt_event.ticker}"
                    )
                else:
                    logger.warning(f"Trade execution failed: {result.message}")
            else:
                logger.info(
                    f"No trading opportunity found for {halt_event.ticker} halt"
                )

        except Exception as e:
            logger.error(f"Error handling halt event: {e}")

    async def _get_market_data(self, ticker: str) -> Optional[Dict]:
        """Get current market data for a ticker"""
        try:
            # Use data pipeline to get market data
            bars = await self.data_pipeline.get_historical_bars(ticker, limit=1)
            if bars:
                latest_bar = bars[0]
                return {
                    "price": latest_bar.close,
                    "vwap": latest_bar.vwap or latest_bar.close,
                    "volume": latest_bar.volume,
                    "high": latest_bar.high,
                    "low": latest_bar.low,
                }
            return None
        except Exception as e:
            logger.error(f"Error getting market data for {ticker}: {e}")
            return None

    async def _get_news_context(self, ticker: str) -> Optional[Dict]:
        """Get news context for a ticker"""
        try:
            news = await self.data_pipeline.get_halt_context_news(ticker)
            if news:
                return {
                    "headlines": [n.headline for n in news[:3]],  # Top 3 headlines
                    "catalyst_type": news[0].catalyst_type
                    if news[0].catalyst_type
                    else None,
                    "sentiment": news[0].sentiment if news[0].sentiment else None,
                }
            return None
        except Exception as e:
            logger.error(f"Error getting news context for {ticker}: {e}")
            return None

    def _get_account_info(self) -> Dict:
        """Get current account information for risk management"""
        return {
            "daily_pnl": self.daily_pnl,
            "current_positions": self.current_positions,
            "daily_trades": self.daily_trades,
            "account_value": 100000.0,  # Placeholder - would get from broker
        }

    async def _risk_monitoring_loop(self):
        """Monitor risk and trigger kill switch if needed"""
        while self.running and not self.shutdown_event.is_set():
            try:
                account_info = self._get_account_info()

                if self.policy_engine.should_flatten_all(account_info):
                    logger.critical("Kill switch activated - flattening all positions")
                    self.order_manager.flatten_all_positions()
                    # Could also stop trading here

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in risk monitoring: {e}")
                await asyncio.sleep(60)

    async def _performance_update_loop(self):
        """Update daily performance metrics"""
        while self.running and not self.shutdown_event.is_set():
            try:
                # Update daily performance (simplified)
                today = datetime.utcnow().date()
                performance = DailyPerformance.objects(date=today).first()

                if not performance:
                    performance = DailyPerformance(date=today)

                # Update with current metrics
                performance.total_trades = self.daily_trades
                performance.gross_pnl = self.daily_pnl
                performance.net_pnl = self.daily_pnl  # Simplified
                performance.updated_at = datetime.utcnow()
                performance.save()

                await asyncio.sleep(300)  # Update every 5 minutes

            except Exception as e:
                logger.error(f"Error updating performance: {e}")
                await asyncio.sleep(600)

    def _handle_pipeline_error(self, error: Exception, source: str):
        """Handle data pipeline errors"""
        logger.error(f"Data pipeline error from {source}: {error}")

        # Log to database
        system_log = SystemLog(
            level="error",
            category="data",
            message=f"Pipeline error from {source}",
            details={"error": str(error), "source": source},
        )
        system_log.save()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self.stop())

    async def wait_for_shutdown(self):
        """Wait for shutdown signal"""
        await self.shutdown_event.wait()


# Factory function
def create_trading_engine() -> TradingEngine:
    """Create trading engine instance"""
    return TradingEngine()


# Main entry point
async def main():
    """Main entry point for the trading engine"""
    engine = create_trading_engine()

    try:
        success = await engine.start()
        if success:
            logger.info("Trading engine started. Press Ctrl+C to stop.")
            await engine.wait_for_shutdown()
        else:
            logger.error("Failed to start trading engine")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await engine.stop()


if __name__ == "__main__":
    # Run the trading engine
    asyncio.run(main())
