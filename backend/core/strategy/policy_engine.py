"""
Policy Engine for Halt Detector Trading System

Combines entry/exit rules with risk management to make final trading decisions.
"""

from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime
import logging

from .entry_rules import EntryRules, EntryDecision, EntrySignal
from .exit_rules import ExitRules, ExitDecision, ExitSignal
from ...database.models import PositionSide


logger = logging.getLogger(__name__)


@dataclass
class TradeDecision:
    """Final trade decision"""

    action: str  # "BUY", "SELL", "SHORT", "COVER", "HOLD", "CLOSE"
    quantity: int
    price: float
    confidence: float
    rationale: str
    risk_amount: float
    stop_loss: float
    take_profit: float


class PolicyEngine:
    """Policy engine that combines strategy rules with risk management"""

    def __init__(
        self,
        entry_rules: EntryRules = None,
        exit_rules: ExitRules = None,
        risk_config: Dict = None,
    ):
        self.entry_rules = entry_rules or EntryRules()
        self.exit_rules = exit_rules or ExitRules()
        self.risk_config = risk_config or self._default_risk_config()

    def _default_risk_config(self) -> Dict:
        """Default risk management configuration"""
        return {
            "max_daily_loss": 1000.00,
            "max_position_loss": 200.00,
            "max_exposure_percent": 25.0,  # % of account
            "max_concurrent_trades": 5,
            "daily_trade_limit": 20,
            "min_confidence_threshold": 0.6,
            "max_risk_per_trade_percent": 1.0,
        }

    def evaluate_halt_opportunity(
        self,
        halt_event: Dict,
        market_data: Dict,
        news_context: Dict = None,
        account_info: Dict = None,
    ) -> Optional[TradeDecision]:
        """
        Evaluate a halt event for potential trade opportunity

        Args:
            halt_event: Halt event data
            market_data: Current market data
            news_context: News/catalyst context
            account_info: Current account status

        Returns:
            TradeDecision or None if no trade
        """
        try:
            # Get entry decision
            entry_decision = self.entry_rules.evaluate_entry(
                halt_event, market_data, news_context
            )

            # If no entry signal, return None
            if entry_decision.signal == EntrySignal.PASS:
                return None

            # Check risk management constraints
            if not self._validate_risk_constraints(entry_decision, account_info):
                logger.info(f"Risk constraints not met for {halt_event.get('ticker')}")
                return None

            # Calculate position size
            quantity = self._calculate_position_size(
                entry_decision, market_data, account_info
            )

            if quantity <= 0:
                logger.info(
                    f"Invalid position size calculated for {halt_event.get('ticker')}"
                )
                return None

            # Convert entry signal to trade action
            action = self._entry_signal_to_action(entry_decision.signal)

            return TradeDecision(
                action=action,
                quantity=quantity,
                price=market_data.get("price", 0),
                confidence=entry_decision.confidence,
                rationale=entry_decision.rationale,
                risk_amount=entry_decision.risk_amount * quantity,
                stop_loss=entry_decision.stop_loss,
                take_profit=entry_decision.take_profit,
            )

        except Exception as e:
            logger.error(f"Error evaluating halt opportunity: {e}")
            return None

    def evaluate_exit_opportunity(
        self, position: Dict, current_market_data: Dict, account_info: Dict = None
    ) -> List[TradeDecision]:
        """
        Evaluate exit opportunities for existing positions

        Args:
            position: Current position data
            current_market_data: Current market data
            account_info: Current account status

        Returns:
            List of TradeDecision objects for exits
        """
        try:
            decisions = []

            # Get exit decisions
            entry_price = position.get("entry_price", 0)
            entry_time = position.get("opened_at", datetime.utcnow())

            exit_decisions = self.exit_rules.evaluate_exit(
                position, current_market_data, entry_price, entry_time
            )

            for exit_decision in exit_decisions:
                if exit_decision.signal == ExitSignal.NO_EXIT:
                    continue

                # Convert exit signal to trade action
                action = self._exit_signal_to_action(exit_decision.signal, position)

                # Calculate exit quantity
                position_quantity = position.get("quantity", 0)
                exit_quantity = int(
                    position_quantity * exit_decision.partial_exit_percent / 100
                )

                if exit_quantity <= 0:
                    continue

                decisions.append(
                    TradeDecision(
                        action=action,
                        quantity=exit_quantity,
                        price=exit_decision.exit_price
                        or current_market_data.get("price", 0),
                        confidence=exit_decision.confidence,
                        rationale=exit_decision.exit_reason,
                        risk_amount=0.0,  # Exits don't add risk
                        stop_loss=0.0,
                        take_profit=0.0,
                    )
                )

            return decisions

        except Exception as e:
            logger.error(f"Error evaluating exit opportunity: {e}")
            return []

    def _validate_risk_constraints(
        self, entry_decision: EntryDecision, account_info: Dict = None
    ) -> bool:
        """Validate risk management constraints"""
        if not account_info:
            return True  # Allow trades if no account info provided

        try:
            # Check confidence threshold
            if entry_decision.confidence < self.risk_config["min_confidence_threshold"]:
                return False

            # Check daily loss limit
            daily_pnl = account_info.get("daily_pnl", 0)
            max_daily_loss = self.risk_config["max_daily_loss"]
            if daily_pnl <= -max_daily_loss:
                logger.warning("Daily loss limit reached")
                return False

            # Check concurrent trades limit
            current_positions = account_info.get("current_positions", 0)
            max_concurrent = self.risk_config["max_concurrent_trades"]
            if current_positions >= max_concurrent:
                logger.info("Maximum concurrent trades reached")
                return False

            # Check daily trade limit
            daily_trades = account_info.get("daily_trades", 0)
            max_daily_trades = self.risk_config["daily_trade_limit"]
            if daily_trades >= max_daily_trades:
                logger.info("Daily trade limit reached")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating risk constraints: {e}")
            return False

    def _calculate_position_size(
        self,
        entry_decision: EntryDecision,
        market_data: Dict,
        account_info: Dict = None,
    ) -> int:
        """Calculate position size based on risk parameters"""
        try:
            entry_price = market_data.get("price", 0)
            if entry_price <= 0:
                return 0

            # Default position size calculation
            # In production, this would be more sophisticated
            risk_per_share = abs(entry_price - entry_decision.stop_loss)
            max_risk_per_trade = self.risk_config["max_risk_per_trade_percent"] / 100

            # Assume account size if not provided
            account_size = (
                account_info.get("account_value", 100000) if account_info else 100000
            )

            max_risk_amount = account_size * max_risk_per_trade
            position_size_risk = (
                max_risk_amount / risk_per_share if risk_per_share > 0 else 0
            )

            # Limit by position loss constraint
            max_position_loss = self.risk_config["max_position_loss"]
            position_size_loss = (
                max_position_loss / risk_per_share if risk_per_share > 0 else 0
            )

            # Take the minimum of the constraints
            position_size = min(position_size_risk, position_size_loss)

            # Ensure minimum position size (at least 100 shares for round lots)
            position_size = max(position_size, 100)

            # Round to nearest 100 shares
            return int(position_size // 100 * 100)

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0

    def _entry_signal_to_action(self, signal: EntrySignal) -> str:
        """Convert entry signal to trade action"""
        signal_to_action = {
            EntrySignal.HALT_UP_SHORT: "SHORT",
            EntrySignal.GAP_HALT_UP_SHORT: "SHORT",
            EntrySignal.HALT_DOWN_LONG: "BUY",
            EntrySignal.GAP_DOWN_HALT_LONG: "BUY",
        }
        return signal_to_action.get(signal, "HOLD")

    def _exit_signal_to_action(self, signal: ExitSignal, position: Dict) -> str:
        """Convert exit signal to trade action"""
        side = position.get("side")

        # For long positions, exit with SELL
        # For short positions, exit with COVER
        if side == PositionSide.LONG.value:
            return "SELL"
        elif side == PositionSide.SHORT.value:
            return "COVER"
        else:
            return "CLOSE"

    def should_flatten_all(self, account_info: Dict) -> bool:
        """Determine if all positions should be flattened (kill switch)"""
        if not account_info:
            return False

        # Check daily loss limit
        daily_pnl = account_info.get("daily_pnl", 0)
        max_daily_loss = self.risk_config["max_daily_loss"]

        if daily_pnl <= -max_daily_loss:
            logger.critical("Kill switch activated: Daily loss limit exceeded")
            return True

        # Check exposure limit
        current_exposure = account_info.get("current_exposure_percent", 0)
        max_exposure = self.risk_config["max_exposure_percent"]

        if current_exposure >= max_exposure:
            logger.warning("High exposure detected, consider reducing positions")
            # For MVP, don't auto-flatten on exposure, just warn
            return False

        return False


# Factory function
def create_policy_engine(
    entry_config: Dict = None, exit_config: Dict = None, risk_config: Dict = None
) -> PolicyEngine:
    """Create policy engine instance"""
    entry_rules = EntryRules(entry_config)
    exit_rules = ExitRules(exit_config)
    return PolicyEngine(entry_rules, exit_rules, risk_config)
