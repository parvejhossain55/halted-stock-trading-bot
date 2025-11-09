"""
Exit Rules for Halt Detector Trading Strategy

This module implements exit logic:
- VWAP reversion exits
- Gap-fill completion
- Trend change detection
- Trailing stops
- Partial exits (25%, 50%, 75%)
"""

from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from ...database.models import PositionSide


logger = logging.getLogger(__name__)


class ExitSignal(Enum):
    """Exit signal types"""

    VWAP_REVERSION = "vwap_reversion"
    GAP_FILL = "gap_fill"
    TREND_CHANGE = "trend_change"
    TRAILING_STOP = "trailing_stop"
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TIME_EXIT = "time_exit"
    NO_EXIT = "no_exit"


@dataclass
class ExitDecision:
    """Exit decision result"""

    signal: ExitSignal
    confidence: float  # 0-1
    exit_price: float
    exit_reason: str
    partial_exit_percent: float = 100.0  # 100% = full exit


class ExitRules:
    """Exit rules engine for halt-based trading"""

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()

    def _default_config(self) -> Dict:
        """Default configuration for exit rules"""
        return {
            "vwap_reversion_threshold": 0.1,  # % from VWAP for reversion exit
            "gap_fill_threshold": 0.5,  # % gap fill for exit
            "trend_change_period": 3,  # candles to detect trend change
            "trailing_stop_activation": 1.0,  # % profit before trailing stop activates
            "trailing_stop_distance": 0.5,  # % distance for trailing stop
            "max_hold_time_minutes": 60,  # Maximum time in position
            "partial_exit_levels": [25, 50, 75],  # % levels for partial exits
            "take_profit_scales": [1.0, 2.0, 3.0],  # Take profit multipliers
        }

    def evaluate_exit(
        self,
        position: Dict,
        current_market_data: Dict,
        entry_price: float,
        entry_time: datetime,
    ) -> List[ExitDecision]:
        """
        Evaluate exit opportunities for a position

        Args:
            position: Current position data
            current_market_data: Current market data
            entry_price: Original entry price
            entry_time: When position was entered

        Returns:
            List of ExitDecision objects (can have multiple partial exits)
        """
        try:
            ticker = position.get("ticker")
            side = position.get("side")
            current_price = current_market_data.get("price", 0)
            vwap = current_market_data.get("vwap", 0)

            exits = []

            # Check for stop loss first (highest priority)
            stop_loss_exit = self._check_stop_loss(position, current_price)
            if stop_loss_exit:
                exits.append(stop_loss_exit)

            # Check for take profit targets
            take_profit_exits = self._check_take_profit_targets(
                position, current_price, entry_price
            )
            exits.extend(take_profit_exits)

            # Check for VWAP reversion
            vwap_exit = self._check_vwap_reversion(position, current_price, vwap)
            if vwap_exit:
                exits.append(vwap_exit)

            # Check for gap fill (if applicable)
            gap_fill_exit = self._check_gap_fill(position, current_market_data)
            if gap_fill_exit:
                exits.append(gap_fill_exit)

            # Check for trend change
            trend_exit = self._check_trend_change(position, current_market_data)
            if trend_exit:
                exits.append(trend_exit)

            # Check for time-based exit
            time_exit = self._check_time_exit(entry_time)
            if time_exit:
                exits.append(time_exit)

            # Check for trailing stop
            trailing_exit = self._check_trailing_stop(
                position, current_price, entry_price
            )
            if trailing_exit:
                exits.append(trailing_exit)

            # If no exits triggered, return no exit signal
            if not exits:
                exits.append(
                    ExitDecision(
                        signal=ExitSignal.NO_EXIT,
                        confidence=0.0,
                        exit_price=0.0,
                        exit_reason="No exit conditions met",
                    )
                )

            return exits

        except Exception as e:
            logger.error(f"Error evaluating exit for {position.get('ticker')}: {e}")
            return [
                ExitDecision(
                    signal=ExitSignal.NO_EXIT,
                    confidence=0.0,
                    exit_price=0.0,
                    exit_reason=f"Error: {str(e)}",
                )
            ]

    def _check_stop_loss(
        self, position: Dict, current_price: float
    ) -> Optional[ExitDecision]:
        """Check if stop loss has been hit"""
        stop_loss = position.get("stop_loss", 0)
        side = position.get("side")

        if stop_loss == 0:
            return None

        # For long positions, exit if price <= stop_loss
        # For short positions, exit if price >= stop_loss
        if (side == PositionSide.LONG.value and current_price <= stop_loss) or (
            side == PositionSide.SHORT.value and current_price >= stop_loss
        ):
            return ExitDecision(
                signal=ExitSignal.STOP_LOSS,
                confidence=1.0,  # Stop loss is mandatory
                exit_price=stop_loss,
                exit_reason=f"Stop loss hit at {stop_loss}",
            )

        return None

    def _check_take_profit_targets(
        self, position: Dict, current_price: float, entry_price: float
    ) -> List[ExitDecision]:
        """Check take profit targets with partial exits"""
        exits = []
        side = position.get("side")

        # Calculate profit percentage
        if side == PositionSide.LONG.value:
            profit_percent = ((current_price - entry_price) / entry_price) * 100
        else:  # SHORT
            profit_percent = ((entry_price - current_price) / entry_price) * 100

        # Check each take profit level
        for i, target_percent in enumerate(self.config["take_profit_scales"]):
            if profit_percent >= target_percent:
                # Determine partial exit percentage
                partial_percent = (
                    self.config["partial_exit_levels"][i]
                    if i < len(self.config["partial_exit_levels"])
                    else 100.0
                )

                # Calculate target price
                if side == PositionSide.LONG.value:
                    target_price = entry_price * (1 + target_percent / 100)
                else:
                    target_price = entry_price * (1 - target_percent / 100)

                exits.append(
                    ExitDecision(
                        signal=ExitSignal.TAKE_PROFIT,
                        confidence=0.9,
                        exit_price=target_price,
                        exit_reason=f"Take profit target {target_percent}% hit",
                        partial_exit_percent=partial_percent,
                    )
                )

        return exits

    def _check_vwap_reversion(
        self, position: Dict, current_price: float, vwap: float
    ) -> Optional[ExitDecision]:
        """Check for VWAP reversion exit"""
        side = position.get("side")
        threshold = self.config["vwap_reversion_threshold"]

        # For long positions, exit if price falls below VWAP
        # For short positions, exit if price rises above VWAP
        if (
            side == PositionSide.LONG.value
            and current_price <= vwap * (1 - threshold / 100)
        ) or (
            side == PositionSide.SHORT.value
            and current_price >= vwap * (1 + threshold / 100)
        ):
            signal_type = (
                "below VWAP" if side == PositionSide.LONG.value else "above VWAP"
            )
            return ExitDecision(
                signal=ExitSignal.VWAP_REVERSION,
                confidence=0.8,
                exit_price=vwap,
                exit_reason=f"VWAP reversion: price moved {signal_type}",
            )

        return None

    def _check_gap_fill(
        self, position: Dict, market_data: Dict
    ) -> Optional[ExitDecision]:
        """Check for gap fill completion"""
        # This would require tracking the original gap
        # For MVP, simplified logic based on position context
        gap_info = position.get("gap_info", {})
        if not gap_info:
            return None

        gap_percent = gap_info.get("gap_percent", 0)
        current_price = market_data.get("price", 0)
        entry_price = position.get("entry_price", 0)

        # Simplified gap fill detection
        fill_threshold = self.config["gap_fill_threshold"] / 100
        if abs(current_price - entry_price) / entry_price >= fill_threshold:
            return ExitDecision(
                signal=ExitSignal.GAP_FILL,
                confidence=0.85,
                exit_price=current_price,
                exit_reason=f"Gap fill {gap_percent}% completed",
            )

        return None

    def _check_trend_change(
        self, position: Dict, market_data: Dict
    ) -> Optional[ExitDecision]:
        """Check for trend change signals"""
        # This would require price action analysis over multiple candles
        # For MVP, simplified logic
        side = position.get("side")
        current_price = market_data.get("price", 0)
        entry_price = position.get("entry_price", 0)

        # Simple trend change: if position has moved against us significantly
        adverse_move_percent = 2.0  # 2% adverse move
        if (
            side == PositionSide.LONG.value
            and current_price <= entry_price * (1 - adverse_move_percent / 100)
        ) or (
            side == PositionSide.SHORT.value
            and current_price >= entry_price * (1 + adverse_move_percent / 100)
        ):
            return ExitDecision(
                signal=ExitSignal.TREND_CHANGE,
                confidence=0.7,
                exit_price=current_price,
                exit_reason="Trend change detected",
            )

        return None

    def _check_time_exit(self, entry_time: datetime) -> Optional[ExitDecision]:
        """Check for time-based exit"""
        elapsed_minutes = (datetime.utcnow() - entry_time).total_seconds() / 60

        if elapsed_minutes >= self.config["max_hold_time_minutes"]:
            return ExitDecision(
                signal=ExitSignal.TIME_EXIT,
                confidence=0.6,
                exit_price=0.0,  # Market order
                exit_reason=f"Maximum hold time {self.config['max_hold_time_minutes']} minutes exceeded",
            )

        return None

    def _check_trailing_stop(
        self, position: Dict, current_price: float, entry_price: float
    ) -> Optional[ExitDecision]:
        """Check trailing stop conditions"""
        side = position.get("side")
        trailing_stop = position.get("trailing_stop", 0)

        if trailing_stop == 0:
            return None

        # Check if trailing stop has been hit
        if (side == PositionSide.LONG.value and current_price <= trailing_stop) or (
            side == PositionSide.SHORT.value and current_price >= trailing_stop
        ):
            return ExitDecision(
                signal=ExitSignal.TRAILING_STOP,
                confidence=0.95,
                exit_price=trailing_stop,
                exit_reason="Trailing stop hit",
            )

        return None

    def update_trailing_stop(
        self, position: Dict, current_price: float, entry_price: float
    ) -> float:
        """Update trailing stop level based on current profit"""
        side = position.get("side")

        # Calculate current profit percentage
        if side == PositionSide.LONG.value:
            profit_percent = ((current_price - entry_price) / entry_price) * 100
        else:
            profit_percent = ((entry_price - current_price) / entry_price) * 100

        # Only activate trailing stop after minimum profit
        if profit_percent < self.config["trailing_stop_activation"]:
            return position.get("trailing_stop", 0)

        # Update trailing stop
        trailing_distance = self.config["trailing_stop_distance"]
        if side == PositionSide.LONG.value:
            new_trailing_stop = current_price * (1 - trailing_distance / 100)
            current_trailing = position.get("trailing_stop", 0)
            return (
                max(new_trailing_stop, current_trailing)
                if current_trailing > 0
                else new_trailing_stop
            )
        else:  # SHORT
            new_trailing_stop = current_price * (1 + trailing_distance / 100)
            current_trailing = position.get("trailing_stop", float("inf"))
            return (
                min(new_trailing_stop, current_trailing)
                if current_trailing < float("inf")
                else new_trailing_stop
            )


# Factory function
def create_exit_rules(config: Dict = None) -> ExitRules:
    """Create exit rules instance"""
    return ExitRules(config)
