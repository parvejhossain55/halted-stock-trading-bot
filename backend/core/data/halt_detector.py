# halt-detector-zed/backend/core/data/halt_detector.py
"""
Halt Detector Module - Polygon LULD Integration

Real-time LULD (Limit Up-Limit Down) halt detection using Polygon WebSocket API.
Detects price limit events, trading halts, and resumption signals for automated trading.

Official Documentation: https://massive.com/docs/websocket/stocks/luld
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

from .polygon_client import PolygonClient, LULDMessage, LULDType

logger = logging.getLogger(__name__)


# ============================================================================
# HALT DETECTION DATA STRUCTURES
# ============================================================================


class HaltType(Enum):
    """Types of trading halts that can be detected"""

    HALT_UP = "halt_up"  # Price halted due to upward move
    HALT_DOWN = "halt_down"  # Price halted due to downward move
    GAP_UP = "gap_up"  # Price gapped up on resumption
    GAP_DOWN = "gap_down"  # Price gapped down on resumption
    LIMIT_UP = "limit_up"  # Hit upper LULD limit
    LIMIT_DOWN = "limit_down"  # Hit lower LULD limit


class HaltStatus(Enum):
    """Current status of a halt event"""

    ACTIVE = "active"  # Halt is currently in effect
    RESUMED = "resumed"  # Trading has resumed
    EXPIRED = "expired"  # Halt period has ended
    CANCELLED = "cancelled"  # Halt was cancelled


class WeaknessSignal(Enum):
    """Types of weakness confirmation signals for short entries"""

    RED_CANDLE_CLOSE = "red_candle_close"
    VWAP_REJECTION = "vwap_rejection"
    LOWER_HIGH = "lower_high"
    VOLUME_DECLINE = "volume_decline"


class StrengthSignal(Enum):
    """Types of strength confirmation signals for long entries"""

    GREEN_CANDLE_CLOSE = "green_candle_close"
    VWAP_RECLAIM = "vwap_reclaim"
    HIGHER_LOW = "higher_low"
    VOLUME_SURGE = "volume_surge"


@dataclass
class HaltEvent:
    """Represents a detected halt event with full context"""

    ticker: str
    halt_type: HaltType
    halt_status: HaltStatus
    halt_detected_at: datetime

    # LULD context
    luld_message: Optional[LULDMessage] = None
    limit_up_price: Optional[float] = None
    limit_down_price: Optional[float] = None
    price_band_width: Optional[float] = None

    # Resumption data
    resumption_time: Optional[datetime] = None
    resumption_price: Optional[float] = None
    gap_percent: Optional[float] = None

    # Confirmation signals
    weakness_confirmed: bool = False
    weakness_signals: List[WeaknessSignal] = field(default_factory=list)
    strength_confirmed: bool = False
    strength_signals: List[StrengthSignal] = field(default_factory=list)

    # Context data
    pre_halt_price: Optional[float] = None
    pre_halt_volume: Optional[int] = None
    halt_count_today: int = 1
    exchange: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert halt event to dictionary"""
        return {
            "ticker": self.ticker,
            "halt_type": self.halt_type.value,
            "halt_status": self.halt_status.value,
            "halt_detected_at": self.halt_detected_at.isoformat(),
            "limit_up_price": self.limit_up_price,
            "limit_down_price": self.limit_down_price,
            "price_band_width": self.price_band_width,
            "resumption_time": self.resumption_time.isoformat()
            if self.resumption_time
            else None,
            "resumption_price": self.resumption_price,
            "gap_percent": self.gap_percent,
            "weakness_confirmed": self.weakness_confirmed,
            "strength_confirmed": self.strength_confirmed,
            "pre_halt_price": self.pre_halt_price,
            "pre_halt_volume": self.pre_halt_volume,
            "halt_count_today": self.halt_count_today,
            "exchange": self.exchange,
            "weakness_signals": [s.value for s in self.weakness_signals],
            "strength_signals": [s.value for s in self.strength_signals],
        }


# ============================================================================
# MAIN HALT DETECTOR CLASS
# ============================================================================


class HaltDetector:
    """
    Real-time halt detector using Polygon LULD WebSocket API.

    Monitors LULD messages to detect:
    - Limit up/down price bands
    - Trading halts and resumptions
    - Price action confirmation (weakness/strength signals)
    - Gap detections on resumption
    """

    def __init__(self, api_key: str):
        """
        Initialize halt detector with Polygon API key.

        Args:
            api_key: Polygon API key for WebSocket access
        """
        self.api_key = api_key
        self.client = PolygonClient(api_key)

        # Active halt tracking
        self.active_halts: Dict[str, HaltEvent] = {}

        # Historical data for analysis
        self.halt_history: List[HaltEvent] = []

        # Daily halt counts per ticker
        self.daily_halt_counts: Dict[str, int] = {}
        self.current_date = datetime.now().date()

        # Callbacks
        self.halt_callback: Optional[Callable[[HaltEvent], None]] = None
        self.resume_callback: Optional[Callable[[HaltEvent], None]] = None
        self.weakness_callback: Optional[Callable[[HaltEvent], None]] = None
        self.strength_callback: Optional[Callable[[HaltEvent], None]] = None

        # Configuration
        self.price_threshold = 0.05  # 5% move triggers halt detection
        self.volume_spike_threshold = 3.0  # 3x average volume
        self.max_halt_history = 1000  # Max historical events to keep

        logger.info("HaltDetector initialized with Polygon LULD integration")

    def set_halt_callback(self, callback: Callable[[HaltEvent], None]):
        """Set callback for new halt detections"""
        self.halt_callback = callback

    def set_resume_callback(self, callback: Callable[[HaltEvent], None]):
        """Set callback for halt resumptions"""
        self.resume_callback = callback

    def set_weakness_callback(self, callback: Callable[[HaltEvent], None]):
        """Set callback for weakness confirmation"""
        self.weakness_callback = callback

    def set_strength_callback(self, callback: Callable[[HaltEvent], None]):
        """Set callback for strength confirmation"""
        self.strength_callback = callback

    async def start_monitoring(self, tickers: List[str]):
        """
        Start real-time LULD monitoring for specified tickers.

        Args:
            tickers: List of ticker symbols to monitor
        """
        logger.info(f"Starting LULD monitoring for {len(tickers)} tickers")

        # Set up LULD message handler
        self.client.ws.on_luld(self._handle_luld_message)

        # Connect and subscribe
        await self.client.connect_websocket()
        await self.client.subscribe_luld(tickers)

        logger.info("HaltDetector monitoring started")

    async def stop_monitoring(self):
        """Stop monitoring and close connections"""
        await self.client.close_websocket()
        logger.info("HaltDetector monitoring stopped")

    def _handle_luld_message(self, luld_msg: LULDMessage):
        """
        Handle incoming LULD WebSocket messages.

        Args:
            luld_msg: Parsed LULD message from Polygon
        """
        ticker = luld_msg.ticker

        # Check if this indicates a halt or resumption
        if self._is_halt_indicated(luld_msg):
            self._process_halt_detection(luld_msg)

        elif self._is_resume_indicated(luld_msg):
            self._process_resume_detection(luld_msg)

        else:
            # Check for price action that might indicate gap or confirmation
            self._check_price_action(ticker, luld_msg)

    def _is_halt_indicated(self, luld_msg: LULDMessage) -> bool:
        """
        Determine if LULD message indicates a trading halt.

        Args:
            luld_msg: LULD message to analyze

        Returns:
            True if halt is indicated
        """
        # Check if we're already tracking this ticker
        if luld_msg.ticker in self.active_halts:
            return False

        # LULD type indicates a halt
        return luld_msg.luld_type in [
            LULDType.LIMIT_UP,
            LULDType.LIMIT_DOWN,
            LULDType.LIMIT_UP_DOWN,
        ]

    def _process_halt_detection(self, luld_msg: LULDMessage):
        """
        Process a new halt detection event.

        Args:
            luld_msg: LULD message indicating halt
        """
        ticker = luld_msg.ticker

        # Determine halt type based on LULD type
        if luld_msg.luld_type == LULDType.LIMIT_UP:
            halt_type = HaltType.LIMIT_UP
        elif luld_msg.luld_type == LULDType.LIMIT_DOWN:
            halt_type = HaltType.LIMIT_DOWN
        else:  # LIMIT_UP_DOWN
            # Need additional context - default to unknown for now
            halt_type = HaltType.HALT_UP  # Will be refined with price action

        # Create halt event
        halt_event = HaltEvent(
            ticker=ticker,
            halt_type=halt_type,
            halt_status=HaltStatus.ACTIVE,
            halt_detected_at=luld_msg.timestamp,
            luld_message=luld_msg,
            limit_up_price=luld_msg.limit_up_price,
            limit_down_price=luld_msg.limit_down_price,
            price_band_width=luld_msg.price_band_width,
            exchange=luld_msg.exchange,
        )

        # Update daily count
        self._update_daily_halt_count(ticker)

        # Store as active halt
        self.active_halts[ticker] = halt_event

        logger.warning(
            f"HALT DETECTED: {ticker} - {halt_type.value} | "
            f"Band: ${halt_event.limit_down_price:.2f} - ${halt_event.limit_up_price:.2f} | "
            f"Width: ${halt_event.price_band_width:.2f}"
        )

        # Trigger callback
        if self.halt_callback:
            self.halt_callback(halt_event)

    def _is_resume_indicated(self, luld_msg: LULDMessage) -> bool:
        """
        Check if LULD message indicates trading resumption.

        Args:
            luld_msg: LULD message to analyze

        Returns:
            True if resume is indicated
        """
        ticker = luld_msg.ticker

        # Must have an active halt for this ticker
        if ticker not in self.active_halts:
            return False

        # LULD type indicates cleared limit bands (resumption)
        return luld_msg.luld_type == LULDType.CLEARED

    def _process_resume_detection(self, luld_msg: LULDMessage):
        """
        Process a halt resumption event.

        Args:
            luld_msg: LULD message indicating resumption
        """
        ticker = luld_msg.ticker
        halt_event = self.active_halts.get(ticker)

        if not halt_event:
            logger.warning(f"Resume detected for unknown halt: {ticker}")
            return

        # Update halt event with resumption data
        halt_event.halt_status = HaltStatus.RESUMED
        halt_event.resumption_time = luld_msg.timestamp

        # Note: Actual resumption price would need to be obtained from
        # first trade after resumption via a different mechanism

        logger.info(
            f"HALT RESUMED: {ticker} - Duration: "
            f"{(luld_msg.timestamp - halt_event.halt_detected_at).total_seconds():.0f}s"
        )

        # Move to history
        self._archive_halt(halt_event)

        # Trigger callback
        if self.resume_callback:
            self.resume_callback(halt_event)

    def _check_price_action(self, ticker: str, luld_msg: LULDMessage):
        """
        Check for price action that might indicate confirmation or gap.

        This method would typically receive price data from another source
        (e.g., aggregate bars) to check for weakness/strength signals.

        Args:
            ticker: Stock ticker
            luld_msg: Latest LULD message
        """
        # This is where we'd integrate with the market data provider
        # to check for confirmation signals after resumption
        pass

    def check_weakness_confirmation(
        self,
        ticker: str,
        current_candle_data: Dict[str, Any],
        vwap: Optional[float] = None,
    ) -> bool:
        """
        Check if weakness is confirmed after halt-up resumption.

        Args:
            ticker: Stock ticker
            current_candle_data: Current price bar data
            vwap: Current VWAP value

        Returns:
            True if weakness confirmed
        """
        halt_event = self.active_halts.get(ticker)
        if not halt_event or halt_event.halt_type not in [
            HaltType.HALT_UP,
            HaltType.GAP_UP,
        ]:
            return False

        if halt_event.weakness_confirmed:
            return True  # Already confirmed

        # Check for weakness signals
        weakness_signals = self._analyze_weakness_signals(current_candle_data, vwap)

        if weakness_signals:
            halt_event.weakness_confirmed = True
            halt_event.weakness_signals = weakness_signals

            logger.info(
                f"Weakness confirmed for {ticker}: {', '.join(s.value for s in weakness_signals)}"
            )

            if self.weakness_callback:
                self.weakness_callback(halt_event)

            return True

        return False

    def check_strength_confirmation(
        self,
        ticker: str,
        current_candle_data: Dict[str, Any],
        vwap: Optional[float] = None,
    ) -> bool:
        """
        Check if strength is confirmed after halt-down resumption.

        Args:
            ticker: Stock ticker
            current_candle_data: Current price bar data
            vwap: Current VWAP value

        Returns:
            True if strength confirmed
        """
        halt_event = self.active_halts.get(ticker)
        if not halt_event or halt_event.halt_type not in [
            HaltType.HALT_DOWN,
            HaltType.GAP_DOWN,
        ]:
            return False

        if halt_event.strength_confirmed:
            return True  # Already confirmed

        # Check for strength signals
        strength_signals = self._analyze_strength_signals(current_candle_data, vwap)

        if strength_signals:
            halt_event.strength_confirmed = True
            halt_event.strength_signals = strength_signals

            logger.info(
                f"Strength confirmed for {ticker}: {', '.join(s.value for s in strength_signals)}"
            )

            if self.strength_callback:
                self.strength_callback(halt_event)

            return True

        return False

    def _analyze_weakness_signals(
        self, candle_data: Dict[str, Any], vwap: Optional[float]
    ) -> List[WeaknessSignal]:
        """
        Analyze candle data for weakness confirmation signals.

        Args:
            candle_data: OHLCV candle data
            vwap: Current VWAP

        Returns:
            List of weakness signals detected
        """
        signals = []
        close = candle_data.get("close", 0)
        open = candle_data.get("open", 0)

        # Red candle (close < open)
        if close < open:
            signals.append(WeaknessSignal.RED_CANDLE_CLOSE)

        # VWAP rejection (close below VWAP)
        if vwap and close < vwap:
            signals.append(WeaknessSignal.VWAP_REJECTION)

        # Lower high pattern
        if candle_data.get("higher_high", False) == False:
            signals.append(WeaknessSignal.LOWER_HIGH)

        # Volume decline (if available)
        if candle_data.get("volume_decline", False):
            signals.append(WeaknessSignal.VOLUME_DECLINE)

        return signals

    def _analyze_strength_signals(
        self, candle_data: Dict[str, Any], vwap: Optional[float]
    ) -> List[StrengthSignal]:
        """
        Analyze candle data for strength confirmation signals.

        Args:
            candle_data: OHLCV candle data
            vwap: Current VWAP

        Returns:
            List of strength signals detected
        """
        signals = []
        close = candle_data.get("close", 0)
        open = candle_data.get("open", 0)
        high = candle_data.get("high", 0)
        low = candle_data.get("low", 0)

        # Green candle (close > open)
        if close > open:
            signals.append(StrengthSignal.GREEN_CANDLE_CLOSE)

        # VWAP reclaim (close above VWAP)
        if vwap and close > vwap:
            signals.append(StrengthSignal.VWAP_RECLAIM)

        # Higher low pattern
        if candle_data.get("lower_low", False) == False:
            signals.append(StrengthSignal.HIGHER_LOW)

        # Volume surge (if available)
        if candle_data.get("volume_surge", False):
            signals.append(StrengthSignal.VOLUME_SURGE)

        return signals

    def _update_daily_halt_count(self, ticker: str):
        """Update daily halt count for ticker"""
        today = datetime.now().date()

        # Reset counts if new day
        if today != self.current_date:
            self.daily_halt_counts.clear()
            self.current_date = today

        self.daily_halt_counts[ticker] = self.daily_halt_counts.get(ticker, 0) + 1

        # Update active halt count
        if ticker in self.active_halts:
            self.active_halts[ticker].halt_count_today = self.daily_halt_counts[ticker]

    def _archive_halt(self, halt_event: HaltEvent):
        """Move completed halt to history"""
        self.halt_history.append(halt_event)

        # Limit history size
        if len(self.halt_history) > self.max_halt_history:
            self.halt_history.pop(0)

    def get_active_halt(self, ticker: str) -> Optional[HaltEvent]:
        """Get active halt for ticker"""
        return self.active_halts.get(ticker)

    def get_all_active_halts(self) -> List[HaltEvent]:
        """Get all active halts"""
        return list(self.active_halts.values())

    def get_halt_history(
        self, ticker: Optional[str] = None, limit: int = 100
    ) -> List[HaltEvent]:
        """Get halt history with optional filtering"""
        history = self.halt_history

        if ticker:
            history = [h for h in history if h.ticker == ticker]

        return history[-limit:]  # Most recent

    def get_halt_statistics(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for halts"""
        halts = self.get_halt_history(ticker) if ticker else self.halt_history

        if not halts:
            return {"total_halts": 0}

        stats = {
            "total_halts": len(halts),
            "active_halts": len(
                [h for h in halts if h.halt_status == HaltStatus.ACTIVE]
            ),
            "resumed_halts": len(
                [h for h in halts if h.halt_status == HaltStatus.RESUMED]
            ),
            "avg_duration_seconds": 0.0,
            "by_type": {},
        }

        # Calculate average duration for resumed halts
        durations = []
        for halt in halts:
            if halt.resumption_time and halt.halt_detected_at:
                duration = (
                    halt.resumption_time - halt.halt_detected_at
                ).total_seconds()
                durations.append(duration)

        if durations:
            stats["avg_duration_seconds"] = sum(durations) / len(durations)

        # Group by halt type
        from collections import defaultdict

        by_type = defaultdict(int)
        for halt in halts:
            by_type[halt.halt_type.value] += 1

        stats["by_type"] = dict(by_type)

        return stats

    def clear_halt(self, ticker: str):
        """Manually clear halt for ticker"""
        if ticker in self.active_halts:
            halt_event = self.active_halts.pop(ticker)
            self._archive_halt(halt_event)
            logger.info(f"Manually cleared halt for {ticker}")

    def is_monitoring_active(self) -> bool:
        """Check if monitoring is active"""
        return self.client.ws.is_connected


# ============================================================================
# ASYNC MONITORING CONTEXT MANAGER
# ============================================================================


class HaltMonitoringContext:
    """
    Context manager for async halt monitoring.

    Usage:
        async with HaltMonitoringContext(api_key, tickers) as detector:
            # Monitor halts
            pass
    """

    def __init__(self, api_key: str, tickers: List[str]):
        self.api_key = api_key
        self.tickers = tickers
        self.detector = HaltDetector(api_key)

    async def __aenter__(self):
        await self.detector.start_monitoring(self.tickers)
        return self.detector

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.detector.stop_monitoring()
        if exc_type:
            logger.error(f"Halt monitoring context error: {exc_val}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def create_halt_detector(api_key: str) -> HaltDetector:
    """Factory function to create HaltDetector instance"""
    return HaltDetector(api_key)


# Example usage
async def example_halt_monitoring():
    """Example demonstrating halt monitoring"""

    def on_halt(halt_event: HaltEvent):
        print(f"\n🚨 Halt Detected: {halt_event.ticker}")
        print(f"   Type: {halt_event.halt_type.value}")
        print(
            f"   Price Band: ${halt_event.limit_down_price:.2f} - ${halt_event.limit_up_price:.2f}"
        )

    def on_weakness(halt_event: HaltEvent):
        print(f"\n⚠️ Weakness Confirmed: {halt_event.ticker}")
        print(f"   Signals: {', '.join(s.value for s in halt_event.weakness_signals)}")

    api_key = "your_polygon_api_key"
    tickers_to_monitor = ["AAPL", "TSLA", "NVDA", "SPY"]

    async with HaltMonitoringContext(api_key, tickers_to_monitor) as detector:
        # Set up callbacks
        detector.set_halt_callback(on_halt)
        detector.set_weakness_callback(on_weakness)

        print("Monitoring LULD halts... (Ctrl+C to stop)")

        try:
            # Keep monitoring active
            while detector.is_monitoring_active():
                await asyncio.sleep(1)

                # You could add price data checks here for confirmation signals
                # detector.check_weakness_confirmation("AAPL", candle_data, vwap)

        except KeyboardInterrupt:
            print("\nStopping halt monitoring...")

    print("Halt monitoring completed.")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_halt_monitoring())
