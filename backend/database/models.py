"""
MongoDB Models for Halt Detector Trading System

This module defines the data structures for all collections in MongoDB.
Uses mongoengine for schema validation and document management.
"""

from mongoengine import (
    Document,
    StringField,
    FloatField,
    IntField,
    BooleanField,
    DateTimeField,
    DictField,
)
from datetime import datetime
from enum import Enum


class HaltType(str, Enum):
    """Types of trading halts"""

    HALT_UP = "halt_up"
    HALT_DOWN = "halt_down"
    GAP_UP = "gap_up"
    GAP_DOWN = "gap_down"


class TradeAction(str, Enum):
    """Trade actions"""

    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"
    PASS = "pass"


class OrderStatus(str, Enum):
    """Order lifecycle status"""

    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class CatalystType(str, Enum):
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


class PositionSide(str, Enum):
    """Position side"""

    LONG = "long"
    SHORT = "short"


class HaltEvent(Document):
    """Detected halt events"""

    meta = {"collection": "halt_events"}

    ticker = StringField(required=True, index=True)
    halt_type = StringField(required=True, choices=[h.value for h in HaltType])
    halt_time = DateTimeField(required=True, index=True)
    resume_time = DateTimeField()

    # Price context
    halt_price = FloatField()
    resume_price = FloatField()
    gap_percent = FloatField()  # Gap up/down percentage

    # Market context
    pre_halt_volume = IntField()
    float_rotation = FloatField()  # % of float traded pre-halt
    distance_from_vwap = FloatField()  # At halt time

    # Halt metadata
    halt_count_today = IntField(default=1)  # Number of halts for this ticker today
    halt_duration_seconds = IntField()

    # Weakness/strength indicators
    weakness_confirmed = BooleanField(default=False)
    strength_confirmed = BooleanField(default=False)
    confirmation_time = DateTimeField()

    created_at = DateTimeField(default=datetime.utcnow)


class Trade(Document):
    """Executed trades"""

    meta = {"collection": "trades"}

    # Trade identification
    trade_id = StringField(required=True, unique=True, primary_key=True)
    ticker = StringField(required=True, index=True)
    side = StringField(required=True, choices=[s.value for s in PositionSide])

    # Entry
    entry_price = FloatField(required=True)
    entry_time = DateTimeField(required=True, index=True)
    quantity = IntField(required=True)

    # Exit
    exit_price = FloatField()
    exit_time = DateTimeField()
    exit_reason = StringField()  # "stop_loss", "take_profit", "vwap_reversion", etc.

    # PnL
    realized_pnl = FloatField()
    realized_pnl_percent = FloatField()
    commission = FloatField(default=0.0)
    net_pnl = FloatField()

    # Trade context
    halt_type = StringField()
    catalyst_type = StringField()
    entry_signal = StringField()  # What triggered entry

    # Risk parameters
    stop_loss = FloatField()
    take_profit = FloatField()
    risk_amount = FloatField()

    # Status
    status = StringField(default="open", choices=["open", "closed", "cancelled"])

    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)


class Position(Document):
    """Current open positions"""

    meta = {"collection": "positions"}

    # Position identification
    position_id = StringField(required=True, unique=True, primary_key=True)
    ticker = StringField(required=True, index=True)
    side = StringField(required=True, choices=[s.value for s in PositionSide])

    # Position details
    quantity = IntField(required=True)
    entry_price = FloatField(required=True)
    current_price = FloatField()

    # PnL tracking
    unrealized_pnl = FloatField(default=0.0)
    unrealized_pnl_percent = FloatField(default=0.0)

    # Risk parameters
    stop_loss = FloatField()
    take_profit = FloatField()

    # Timestamps
    opened_at = DateTimeField(required=True, default=datetime.utcnow)
    last_updated = DateTimeField(default=datetime.utcnow)

    # Related trade
    trade_id = StringField()


class Order(Document):
    """Order management"""

    meta = {"collection": "orders"}

    # Order identification
    order_id = StringField(required=True, unique=True, primary_key=True)
    ticker = StringField(required=True, index=True)

    # Order details
    side = StringField(required=True)  # "buy", "sell", "short", "cover"
    quantity = IntField(required=True)
    order_type = StringField(required=True)  # "market", "limit", "stop"
    limit_price = FloatField()
    stop_price = FloatField()

    # Status
    status = StringField(required=True, choices=[s.value for s in OrderStatus])
    filled_quantity = IntField(default=0)
    avg_fill_price = FloatField()

    # Execution details
    das_order_id = StringField()  # DAS broker order ID
    submitted_at = DateTimeField()
    filled_at = DateTimeField()
    cancelled_at = DateTimeField()

    # Error tracking
    reject_reason = StringField()

    created_at = DateTimeField(default=datetime.utcnow)
    updated_at = DateTimeField(default=datetime.utcnow)


class TradingSignal(Document):
    """Trading signals generated by strategy engine"""

    meta = {"collection": "trading_signals"}

    signal_id = StringField(required=True, unique=True, primary_key=True)
    ticker = StringField(required=True, index=True)

    # Signal details
    action = StringField(required=True)  # "buy", "short", "pass"
    halt_type = StringField(required=True)
    confidence = FloatField()  # 0-1

    # Price context
    signal_price = FloatField()
    vwap = FloatField()

    # Catalyst
    catalyst_type = StringField()
    catalyst_confidence = FloatField()

    # Risk parameters
    suggested_stop_loss = FloatField()
    suggested_take_profit = FloatField()
    suggested_size = IntField()

    # Status
    status = StringField(
        default="pending"
    )  # "pending", "executed", "skipped", "expired"
    execution_notes = StringField()

    # Related objects
    trade_id = StringField()
    halt_event_id = StringField()

    created_at = DateTimeField(default=datetime.utcnow, index=True)
    expires_at = DateTimeField()


class DailyPerformance(Document):
    """Daily performance tracking"""

    meta = {"collection": "daily_performance"}

    date = DateTimeField(required=True, unique=True, primary_key=True)

    # Trade counts
    total_trades = IntField(default=0)
    winning_trades = IntField(default=0)
    losing_trades = IntField(default=0)

    # PnL
    gross_pnl = FloatField(default=0.0)
    commission = FloatField(default=0.0)
    net_pnl = FloatField(default=0.0)

    # Win rate
    win_rate = FloatField()

    # Average trade
    avg_win = FloatField()
    avg_loss = FloatField()
    avg_trade = FloatField()

    # Best/Worst
    best_trade_pnl = FloatField()
    worst_trade_pnl = FloatField()

    # By halt type
    halt_up_trades = IntField(default=0)
    halt_down_trades = IntField(default=0)
    gap_up_trades = IntField(default=0)
    gap_down_trades = IntField(default=0)

    updated_at = DateTimeField(default=datetime.utcnow)


class SystemLog(Document):
    """System event logs"""

    meta = {"collection": "system_logs"}

    timestamp = DateTimeField(required=True, default=datetime.utcnow, index=True)
    level = StringField(
        required=True, index=True
    )  # "info", "warning", "error", "critical"
    category = StringField(
        required=True, index=True
    )  # "trading", "data", "execution", etc.
    message = StringField(required=True)
    details = DictField()
    ticker = StringField(index=True)


def init_db(db_uri: str, db_name: str):
    """Initialize database connection"""
    from mongoengine import connect

    connect(db=db_name, host=db_uri, alias="default")


def create_indexes():
    """Create database indexes for optimal performance"""
    try:
        HaltEvent.ensure_indexes()
        Trade.ensure_indexes()
        Position.ensure_indexes()
        Order.ensure_indexes()
        TradingSignal.ensure_indexes()
        SystemLog.ensure_indexes()
    except Exception as e:
        print(f"Warning: Failed to create indexes: {e}")
        # Continue without indexes for now
