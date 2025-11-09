"""
Entry Rules for Halt Detector Trading Strategy

This module implements the entry logic for different halt types:
- Halt Up: Short after first red candle confirmation
- Gap Halt Up: Short after gap fill + weakness
- Halt Down: Long at resumption (if gap not too large)
- Gap Down Halt: Long after first green candle confirmation
"""

from typing import Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from ...database.models import HaltType, PositionSide


logger = logging.getLogger(__name__)


class EntrySignal(Enum):
    """Entry signal types"""

    HALT_UP_SHORT = "halt_up_short"
    GAP_HALT_UP_SHORT = "gap_halt_up_short"
    HALT_DOWN_LONG = "halt_down_long"
    GAP_DOWN_HALT_LONG = "gap_down_halt_long"
    PASS = "pass"


@dataclass
class EntryDecision:
    """Entry decision result"""

    signal: EntrySignal
    confidence: float  # 0-1
    stop_loss: float
    take_profit: float
    rationale: str
    risk_amount: float


class EntryRules:
    """Entry rules engine for halt-based trading"""

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()

    def _default_config(self) -> Dict:
        """Default configuration for entry rules"""
        return {
            "min_gap_percent": 5.0,
            "max_gap_down_percent": 30.0,  # Avoid entries on gaps larger than this
            "min_volume_threshold": 1_000_000,
            "max_float_rotation": 50.0,  # % of float traded pre-halt
            "vwap_distance_threshold": 2.0,  # % distance from VWAP
            "confirmation_candles": 1,  # Candles to wait for confirmation
            "default_stop_loss_percent": 2.0,
            "default_take_profit_percent": 5.0,
            "risk_per_trade_percent": 1.0,  # % of account to risk per trade
        }

    def evaluate_entry(
        self, halt_event: Dict, market_data: Dict, news_context: Dict = None
    ) -> EntryDecision:
        """
        Evaluate entry opportunity for a halt event

        Args:
            halt_event: Halt event data
            market_data: Current market data (price, volume, VWAP, etc.)
            news_context: News/catalyst context (optional)

        Returns:
            EntryDecision with signal and parameters
        """
        try:
            halt_type = halt_event.get("halt_type")
            ticker = halt_event.get("ticker")
            halt_price = halt_event.get("halt_price", 0)
            resume_price = halt_event.get("resume_price", halt_price)

            # Basic validation
            if not self._validate_basic_conditions(halt_event, market_data):
                return self._create_pass_decision("Basic validation failed")

            # Route to specific entry logic
            if halt_type == HaltType.HALT_UP.value:
                return self._evaluate_halt_up_entry(
                    halt_event, market_data, news_context
                )
            elif halt_type == HaltType.HALT_DOWN.value:
                return self._evaluate_halt_down_entry(
                    halt_event, market_data, news_context
                )
            elif halt_type == HaltType.GAP_UP.value:
                return self._evaluate_gap_halt_up_entry(
                    halt_event, market_data, news_context
                )
            elif halt_type == HaltType.GAP_DOWN.value:
                return self._evaluate_gap_down_halt_entry(
                    halt_event, market_data, news_context
                )
            else:
                return self._create_pass_decision(f"Unknown halt type: {halt_type}")

        except Exception as e:
            logger.error(f"Error evaluating entry for {halt_event.get('ticker')}: {e}")
            return self._create_pass_decision(f"Error: {str(e)}")

    def _validate_basic_conditions(self, halt_event: Dict, market_data: Dict) -> bool:
        """Validate basic entry conditions"""
        try:
            # Check volume threshold
            pre_halt_volume = halt_event.get("pre_halt_volume", 0)
            if pre_halt_volume < self.config["min_volume_threshold"]:
                return False

            # Check float rotation
            float_rotation = halt_event.get("float_rotation", 0)
            if float_rotation > self.config["max_float_rotation"]:
                return False

            # Check VWAP distance
            distance_from_vwap = halt_event.get("distance_from_vwap", 0)
            if abs(distance_from_vwap) > self.config["vwap_distance_threshold"]:
                return False

            return True

        except Exception as e:
            logger.error(f"Error in basic validation: {e}")
            return False

    def _evaluate_halt_up_entry(
        self, halt_event: Dict, market_data: Dict, news_context: Dict = None
    ) -> EntryDecision:
        """Evaluate entry for halt up events (short opportunity)"""
        resume_price = halt_event.get("resume_price", 0)
        halt_price = halt_event.get("halt_price", 0)

        # Check if price opened below VWAP (weakness confirmation)
        vwap = market_data.get("vwap", 0)
        current_price = market_data.get("price", resume_price)

        if current_price < vwap:
            # Calculate risk parameters
            stop_loss = current_price * (
                1 + self.config["default_stop_loss_percent"] / 100
            )
            take_profit = current_price * (
                1 - self.config["default_take_profit_percent"] / 100
            )
            risk_amount = self._calculate_risk_amount(current_price, stop_loss)

            return EntryDecision(
                signal=EntrySignal.HALT_UP_SHORT,
                confidence=0.7,  # Basic confidence for MVP
                stop_loss=stop_loss,
                take_profit=take_profit,
                rationale="Halt up with weakness confirmation below VWAP",
                risk_amount=risk_amount,
            )

        return self._create_pass_decision("No weakness confirmation for halt up")

    def _evaluate_halt_down_entry(
        self, halt_event: Dict, market_data: Dict, news_context: Dict = None
    ) -> EntryDecision:
        """Evaluate entry for halt down events (long opportunity)"""
        resume_price = halt_event.get("resume_price", 0)
        halt_price = halt_event.get("halt_price", 0)
        gap_percent = halt_event.get("gap_percent", 0)

        # Avoid entries on large gap downs
        if gap_percent > self.config["max_gap_down_percent"]:
            return self._create_pass_decision(f"Gap too large: {gap_percent}%")

        # Check if price shows strength
        vwap = market_data.get("vwap", 0)
        current_price = market_data.get("price", resume_price)

        if current_price > vwap:
            # Calculate risk parameters
            stop_loss = current_price * (
                1 - self.config["default_stop_loss_percent"] / 100
            )
            take_profit = current_price * (
                1 + self.config["default_take_profit_percent"] / 100
            )
            risk_amount = self._calculate_risk_amount(current_price, stop_loss)

            return EntryDecision(
                signal=EntrySignal.HALT_DOWN_LONG,
                confidence=0.6,  # Slightly lower confidence for long trades
                stop_loss=stop_loss,
                take_profit=take_profit,
                rationale="Halt down with strength confirmation above VWAP",
                risk_amount=risk_amount,
            )

        return self._create_pass_decision("No strength confirmation for halt down")

    def _evaluate_gap_halt_up_entry(
        self, halt_event: Dict, market_data: Dict, news_context: Dict = None
    ) -> EntryDecision:
        """Evaluate entry for gap halt up events (short opportunity)"""
        gap_percent = halt_event.get("gap_percent", 0)

        # Must have minimum gap
        if gap_percent < self.config["min_gap_percent"]:
            return self._create_pass_decision(f"Gap too small: {gap_percent}%")

        # Look for gap fill + weakness
        resume_price = halt_event.get("resume_price", 0)
        current_price = market_data.get("price", resume_price)
        vwap = market_data.get("vwap", 0)

        # Gap fill would be trading back towards pre-halt levels
        # For MVP, simplified logic: weakness below VWAP
        if current_price < vwap:
            stop_loss = current_price * (
                1 + self.config["default_stop_loss_percent"] / 100
            )
            take_profit = current_price * (
                1 - self.config["default_take_profit_percent"] / 100
            )
            risk_amount = self._calculate_risk_amount(current_price, stop_loss)

            return EntryDecision(
                signal=EntrySignal.GAP_HALT_UP_SHORT,
                confidence=0.75,  # Higher confidence for gap fills
                stop_loss=stop_loss,
                take_profit=take_profit,
                rationale=f"Gap halt up {gap_percent}% with weakness confirmation",
                risk_amount=risk_amount,
            )

        return self._create_pass_decision("No gap fill weakness for gap halt up")

    def _evaluate_gap_down_halt_entry(
        self, halt_event: Dict, market_data: Dict, news_context: Dict = None
    ) -> EntryDecision:
        """Evaluate entry for gap down halt events (long opportunity)"""
        gap_percent = halt_event.get("gap_percent", 0)

        # Must have minimum gap
        if gap_percent < self.config["min_gap_percent"]:
            return self._create_pass_decision(f"Gap too small: {gap_percent}%")

        # Avoid large gaps
        if gap_percent > self.config["max_gap_down_percent"]:
            return self._create_pass_decision(f"Gap too large: {gap_percent}%")

        # Look for strength confirmation
        resume_price = halt_event.get("resume_price", 0)
        current_price = market_data.get("price", resume_price)
        vwap = market_data.get("vwap", 0)

        if current_price > vwap:
            stop_loss = current_price * (
                1 - self.config["default_stop_loss_percent"] / 100
            )
            take_profit = current_price * (
                1 + self.config["default_take_profit_percent"] / 100
            )
            risk_amount = self._calculate_risk_amount(current_price, stop_loss)

            return EntryDecision(
                signal=EntrySignal.GAP_DOWN_HALT_LONG,
                confidence=0.65,
                stop_loss=stop_loss,
                take_profit=take_profit,
                rationale=f"Gap down halt {gap_percent}% with strength confirmation",
                risk_amount=risk_amount,
            )

        return self._create_pass_decision("No strength confirmation for gap down halt")

    def _calculate_risk_amount(self, entry_price: float, stop_loss: float) -> float:
        """Calculate risk amount per trade"""
        # For MVP, use fixed percentage. In production, this would be dynamic
        # based on account size, position sizing algorithms, etc.
        risk_percent = self.config["risk_per_trade_percent"] / 100
        risk_per_share = abs(entry_price - stop_loss)
        return risk_per_share  # This would be multiplied by position size later

    def _create_pass_decision(self, reason: str) -> EntryDecision:
        """Create a pass decision"""
        return EntryDecision(
            signal=EntrySignal.PASS,
            confidence=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            rationale=reason,
            risk_amount=0.0,
        )


# Factory function
def create_entry_rules(config: Dict = None) -> EntryRules:
    """Create entry rules instance"""
    return EntryRules(config)
