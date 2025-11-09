"""
Order Manager for Halt Detector Trading System

Manages order lifecycle, execution, and position tracking.
"""

from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime
import logging
import uuid

from .das_api import DASApiClient, OrderRequest, OrderResponse, OrderType, TimeInForce
from ...database.models import Order, Position, Trade, OrderStatus, PositionSide


logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of order execution"""

    success: bool
    order_id: str
    message: str
    filled_quantity: int = 0
    avg_fill_price: Optional[float] = None


class OrderManager:
    """Manages order execution and position tracking"""

    def __init__(self, das_client: DASApiClient):
        self.das_client = das_client
        self.active_orders = {}  # order_id -> Order model instance

    def execute_trade_decision(
        self, trade_decision: Dict, signal_id: str = None
    ) -> ExecutionResult:
        """
        Execute a trade decision from the policy engine

        Args:
            trade_decision: Decision from policy engine
            signal_id: Associated trading signal ID

        Returns:
            ExecutionResult with execution details
        """
        try:
            action = trade_decision.get("action")
            ticker = trade_decision.get("ticker", "")
            quantity = trade_decision.get("quantity", 0)
            price = trade_decision.get("price", 0)

            if not action or not ticker or quantity <= 0:
                return ExecutionResult(
                    success=False,
                    order_id="",
                    message="Invalid trade decision parameters",
                )

            # Create order request
            order_request = self._create_order_request(trade_decision)

            # Submit order
            response = self.das_client.submit_order(order_request)

            if response.status == OrderStatus.FILLED:
                # Create trade record for filled orders
                self._create_trade_record(trade_decision, response, signal_id)

                # Update positions
                self._update_positions(trade_decision, response)

                return ExecutionResult(
                    success=True,
                    order_id=response.order_id,
                    message="Order executed successfully",
                    filled_quantity=response.filled_quantity,
                    avg_fill_price=response.avg_fill_price,
                )
            elif response.status == OrderStatus.SUBMITTED:
                # Create order record for pending orders
                self._create_order_record(trade_decision, response, signal_id)

                return ExecutionResult(
                    success=True,
                    order_id=response.order_id,
                    message="Order submitted successfully",
                )
            else:
                return ExecutionResult(
                    success=False,
                    order_id=response.order_id,
                    message=f"Order rejected: {response.message}",
                )

        except Exception as e:
            logger.error(f"Error executing trade decision: {e}")
            return ExecutionResult(
                success=False, order_id="", message=f"Execution error: {str(e)}"
            )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order"""
        try:
            success = self.das_client.cancel_order(order_id)
            if success:
                # Update order status in database
                order = Order.objects(order_id=order_id).first()
                if order:
                    order.status = OrderStatus.CANCELLED
                    order.cancelled_at = datetime.utcnow()
                    order.save()
                    logger.info(f"Order {order_id} cancelled successfully")
                return True
            else:
                logger.warning(f"Failed to cancel order {order_id}")
                return False
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get current status of an order"""
        try:
            response = self.das_client.get_order_status(order_id)
            if response:
                return {
                    "order_id": response.order_id,
                    "status": response.status.value,
                    "filled_quantity": response.filled_quantity,
                    "avg_fill_price": response.avg_fill_price,
                    "timestamp": response.timestamp,
                    "message": response.message,
                }
            return None
        except Exception as e:
            logger.error(f"Error getting order status for {order_id}: {e}")
            return None

    def get_open_orders(self) -> List[Dict]:
        """Get all open orders"""
        try:
            orders = Order.objects(
                status__in=[OrderStatus.PENDING.value, OrderStatus.SUBMITTED.value]
            )
            return [
                {
                    "order_id": order.order_id,
                    "ticker": order.ticker,
                    "side": order.side,
                    "quantity": order.quantity,
                    "status": order.status,
                    "created_at": order.created_at,
                }
                for order in orders
            ]
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []

    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        try:
            positions = Position.objects()
            return [
                {
                    "position_id": pos.position_id,
                    "ticker": pos.ticker,
                    "side": pos.side,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "stop_loss": pos.stop_loss,
                    "take_profit": pos.take_profit,
                    "opened_at": pos.opened_at,
                }
                for pos in positions
            ]
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def flatten_all_positions(self) -> List[ExecutionResult]:
        """Emergency flatten all positions"""
        try:
            logger.warning("Emergency position flattening initiated")

            results = []
            positions = self.get_positions()

            for position in positions:
                ticker = position["ticker"]
                quantity = position["quantity"]
                side = position["side"]

                if quantity > 0:
                    # Create market order to close position
                    trade_decision = {
                        "action": "SELL"
                        if side == PositionSide.LONG.value
                        else "COVER",
                        "ticker": ticker,
                        "quantity": quantity,
                        "price": 0,  # Market order
                        "confidence": 1.0,
                        "rationale": "Emergency position flattening",
                    }

                    result = self.execute_trade_decision(trade_decision)
                    results.append(result)

            logger.warning(
                f"Emergency flattening completed: {len(results)} orders submitted"
            )
            return results

        except Exception as e:
            logger.error(f"Error during emergency flattening: {e}")
            return []

    def _create_order_request(self, trade_decision: Dict) -> OrderRequest:
        """Create DAS order request from trade decision"""
        action = trade_decision["action"]
        ticker = trade_decision["ticker"]
        quantity = trade_decision["quantity"]
        price = trade_decision.get("price", 0)
        stop_loss = trade_decision.get("stop_loss")
        take_profit = trade_decision.get("take_profit")

        # Determine order type
        if action in ["BUY", "SELL", "SHORT", "COVER"]:
            order_type = OrderType.MARKET  # Use market orders for MVP
            limit_price = None
            stop_price = None
        else:
            # For more complex orders, would implement limit/stop logic
            order_type = OrderType.MARKET
            limit_price = None
            stop_price = None

        return OrderRequest(
            ticker=ticker,
            side=action,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=TimeInForce.DAY,
        )

    def _create_order_record(
        self, trade_decision: Dict, response: OrderResponse, signal_id: str = None
    ):
        """Create order record in database"""
        try:
            order = Order(
                order_id=response.order_id,
                ticker=trade_decision["ticker"],
                side=trade_decision["action"],
                quantity=trade_decision["quantity"],
                order_type=response.das_order_id or "market",
                status=response.status,
                filled_quantity=response.filled_quantity,
                avg_fill_price=response.avg_fill_price,
                das_order_id=response.das_order_id,
                submitted_at=response.timestamp,
            )
            order.save()
            logger.info(f"Order record created: {response.order_id}")

        except Exception as e:
            logger.error(f"Error creating order record: {e}")

    def _create_trade_record(
        self, trade_decision: Dict, response: OrderResponse, signal_id: str = None
    ):
        """Create trade record for filled orders"""
        try:
            trade_id = str(uuid.uuid4())
            action = trade_decision["action"]
            ticker = trade_decision["ticker"]
            quantity = response.filled_quantity
            fill_price = response.avg_fill_price or 0

            # Determine side
            if action in ["BUY", "SHORT"]:
                side = PositionSide.SHORT if action == "SHORT" else PositionSide.LONG
            else:  # SELL, COVER
                side = PositionSide.LONG if action == "COVER" else PositionSide.SHORT

            trade = Trade(
                trade_id=trade_id,
                ticker=ticker,
                side=side.value,
                entry_price=fill_price,
                entry_time=datetime.utcnow(),
                quantity=quantity,
                halt_type=trade_decision.get("halt_type"),
                catalyst_type=trade_decision.get("catalyst_type"),
                entry_signal=trade_decision.get("rationale"),
                stop_loss=trade_decision.get("stop_loss"),
                take_profit=trade_decision.get("take_profit"),
                risk_amount=trade_decision.get("risk_amount", 0),
                status="open",
            )
            trade.save()
            logger.info(f"Trade record created: {trade_id}")

        except Exception as e:
            logger.error(f"Error creating trade record: {e}")

    def _update_positions(self, trade_decision: Dict, response: OrderResponse):
        """Update position records"""
        try:
            ticker = trade_decision["ticker"]
            action = trade_decision["action"]
            quantity = response.filled_quantity
            fill_price = response.avg_fill_price or 0

            # Find existing position
            position = Position.objects(ticker=ticker).first()

            if action in ["BUY", "SHORT"]:
                # Opening or increasing position
                if position:
                    # Update existing position
                    old_quantity = position.quantity
                    old_avg_price = position.entry_price

                    if action == "BUY":
                        # Long position
                        total_value = (
                            old_quantity * old_avg_price + quantity * fill_price
                        )
                        position.quantity += quantity
                        position.entry_price = total_value / position.quantity
                    else:  # SHORT
                        # Short position - for simplicity, treat as negative quantity
                        position.quantity -= quantity
                        position.entry_price = fill_price

                    position.last_updated = datetime.utcnow()
                    position.save()
                else:
                    # Create new position
                    position_id = str(uuid.uuid4())
                    side = (
                        PositionSide.SHORT if action == "SHORT" else PositionSide.LONG
                    )

                    position = Position(
                        position_id=position_id,
                        ticker=ticker,
                        side=side.value,
                        quantity=quantity if action == "BUY" else -quantity,
                        entry_price=fill_price,
                        current_price=fill_price,
                        stop_loss=trade_decision.get("stop_loss"),
                        take_profit=trade_decision.get("take_profit"),
                    )
                    position.save()

            elif action in ["SELL", "COVER"]:
                # Closing or reducing position
                if position:
                    if action == "SELL":
                        position.quantity -= quantity
                    else:  # COVER
                        position.quantity += quantity

                    # Close position if quantity reaches zero
                    if position.quantity == 0:
                        position.delete()
                    else:
                        position.last_updated = datetime.utcnow()
                        position.save()

            logger.info(f"Position updated for {ticker}: {action} {quantity} shares")

        except Exception as e:
            logger.error(f"Error updating positions: {e}")


# Factory function
def create_order_manager(das_client: DASApiClient) -> OrderManager:
    """Create order manager instance"""
    return OrderManager(das_client)
