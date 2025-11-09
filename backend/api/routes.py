"""
FastAPI Routes for Halt Detector Trading System

REST API endpoints using FastAPI and Pydantic models.
"""

from typing import List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# Database models
from ..database.models import (
    Position,
    Trade,
    TradingSignal,
    HaltEvent,
    DailyPerformance,
    SystemLog,
)

# Core components
from ..core.trading_components import get_trading_components

# Create router
router = APIRouter()


class PositionResponse(BaseModel):
    """Position response model"""

    position_id: str
    ticker: str
    side: str
    quantity: int
    entry_price: float
    current_price: Optional[float]
    unrealized_pnl: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    opened_at: datetime


class TradeResponse(BaseModel):
    """Trade response model"""

    trade_id: str
    ticker: str
    side: str
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    pnl: Optional[float]
    status: str
    entry_time: datetime
    exit_time: Optional[datetime]


class SignalResponse(BaseModel):
    """Trading signal response model"""

    signal_id: str
    ticker: str
    action: str
    halt_type: str
    confidence: float
    status: str
    created_at: datetime


class HaltEventResponse(BaseModel):
    """Halt event response model"""

    ticker: str
    halt_type: str
    halt_time: datetime
    resume_time: Optional[datetime]
    halt_price: Optional[float]
    resume_price: Optional[float]
    gap_percent: Optional[float]


class OrderRequest(BaseModel):
    """Order request model"""

    ticker: str = Field(..., description="Stock ticker symbol")
    side: str = Field(..., description="Order side: BUY, SELL, SHORT, COVER")
    quantity: int = Field(..., description="Number of shares")
    price: Optional[float] = Field(None, description="Limit price (optional)")


class OrderResponse(BaseModel):
    """Order response model"""

    order_id: str
    status: str
    message: str
    filled_quantity: Optional[int] = None
    avg_fill_price: Optional[float] = None


class MarketDataResponse(BaseModel):
    """Market data response model"""

    ticker: str
    price: float
    vwap: float
    volume: int
    high: float
    low: float
    last_updated: datetime


class NewsItem(BaseModel):
    """News item model"""

    headline: str
    source: str
    published_at: datetime
    url: str


class PerformanceResponse(BaseModel):
    """Performance response model"""

    date: str
    total_trades: int
    gross_pnl: float
    net_pnl: float
    win_rate: float


@router.get("/positions", response_model=List[PositionResponse])
async def get_positions(
    ticker: Optional[str] = Query(None, description="Filter by ticker")
):
    """Get all positions"""
    try:
        query = {}
        if ticker:
            query["ticker"] = ticker

        positions = Position.objects(**query)
        return [
            PositionResponse(
                position_id=pos.position_id,
                ticker=pos.ticker,
                side=pos.side,
                quantity=pos.quantity,
                entry_price=pos.entry_price,
                current_price=pos.current_price,
                unrealized_pnl=pos.unrealized_pnl,
                stop_loss=pos.stop_loss,
                take_profit=pos.take_profit,
                opened_at=pos.opened_at,
            )
            for pos in positions
        ]
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving positions: {str(e)}"
        )


@router.get("/trades", response_model=List[TradeResponse])
async def get_trades(
    ticker: Optional[str] = Query(None, description="Filter by ticker"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, description="Maximum number of results", ge=1, le=1000),
):
    """Get trades with optional filtering"""
    try:
        query = {}
        if ticker:
            query["ticker"] = ticker
        if status:
            query["status"] = status

        trades = Trade.objects(**query).order_by("-entry_time")[:limit]
        return [
            TradeResponse(
                trade_id=trade.trade_id,
                ticker=trade.ticker,
                side=trade.side,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                quantity=trade.quantity,
                pnl=trade.realized_pnl,
                status=trade.status,
                entry_time=trade.entry_time,
                exit_time=trade.exit_time,
            )
            for trade in trades
        ]
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving trades: {str(e)}"
        )


@router.get("/signals", response_model=List[SignalResponse])
async def get_signals(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, description="Maximum number of results", ge=1, le=500),
):
    """Get trading signals"""
    try:
        query = {}
        if status:
            query["status"] = status

        signals = TradingSignal.objects(**query).order_by("-created_at")[:limit]
        return [
            SignalResponse(
                signal_id=sig.signal_id,
                ticker=sig.ticker,
                action=sig.action,
                halt_type=sig.halt_type,
                confidence=sig.confidence,
                status=sig.status,
                created_at=sig.created_at,
            )
            for sig in signals
        ]
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving signals: {str(e)}"
        )


@router.get("/halts", response_model=List[HaltEventResponse])
async def get_halt_events(
    ticker: Optional[str] = Query(None, description="Filter by ticker"),
    halt_type: Optional[str] = Query(None, description="Filter by halt type"),
    date: Optional[str] = Query(None, description="Filter by date (YYYY-MM-DD)"),
    limit: int = Query(100, description="Maximum number of results", ge=1, le=1000),
):
    """Get halt events"""
    try:
        query = {}
        if ticker:
            query["ticker"] = ticker
        if halt_type:
            query["halt_type"] = halt_type
        if date:
            try:
                query["halt_time__date"] = datetime.strptime(date, "%Y-%m-%d").date()
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="Invalid date format. Use YYYY-MM-DD"
                )

        halts = HaltEvent.objects(**query).order_by("-halt_time")[:limit]
        return [
            HaltEventResponse(
                ticker=halt.ticker,
                halt_type=halt.halt_type,
                halt_time=halt.halt_time,
                resume_time=halt.resume_time,
                halt_price=halt.halt_price,
                resume_price=halt.resume_price,
                gap_percent=halt.gap_percent,
            )
            for halt in halts
        ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving halt events: {str(e)}"
        )


@router.post("/orders/submit", response_model=OrderResponse)
async def submit_order(order: OrderRequest):
    """Submit a new order"""
    try:
        # Get trading components
        _, order_manager = get_trading_components()

        # Prepare trade decision
        trade_decision = {
            "action": order.side,
            "ticker": order.ticker,
            "quantity": order.quantity,
            "price": order.price or 0,
            "confidence": 1.0,  # Manual orders have full confidence
            "rationale": f"Manual {order.side} order",
        }

        # Execute order
        result = order_manager.execute_trade_decision(trade_decision)

        return OrderResponse(
            order_id=result.order_id,
            status="submitted" if result.success else "rejected",
            message=result.message,
            filled_quantity=result.filled_quantity,
            avg_fill_price=result.avg_fill_price,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting order: {str(e)}")


@router.post("/orders/{order_id}/cancel")
async def cancel_order(order_id: str):
    """Cancel an order"""
    try:
        _, order_manager = get_trading_components()
        success = order_manager.cancel_order(order_id)

        if success:
            return {"message": f"Order {order_id} cancelled successfully"}
        else:
            raise HTTPException(
                status_code=400, detail=f"Failed to cancel order {order_id}"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cancelling order: {str(e)}")


@router.post("/positions/flatten")
async def flatten_positions():
    """Emergency flatten all positions"""
    try:
        _, order_manager = get_trading_components()
        results = order_manager.flatten_all_positions()

        return {
            "message": "Emergency position flattening initiated",
            "orders_submitted": len(results),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error flattening positions: {str(e)}"
        )


@router.get("/market-data/{ticker}", response_model=MarketDataResponse)
async def get_market_data(ticker: str):
    """Get market data for a ticker"""
    try:
        # For MVP, return mock data
        # In production, this would query real market data
        mock_data = MarketDataResponse(
            ticker=ticker,
            price=150.0,
            vwap=149.50,
            volume=1000000,
            high=152.0,
            low=148.0,
            last_updated=datetime.utcnow(),
        )
        return mock_data

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting market data: {str(e)}"
        )


@router.get("/news/{ticker}")
async def get_news(ticker: str):
    """Get news for a ticker"""
    try:
        # For MVP, return mock news
        mock_news = [
            NewsItem(
                headline=f"{ticker} announces quarterly results",
                source="Mock News",
                published_at=datetime.utcnow() - timedelta(hours=2),
                url=f"https://news.example.com/{ticker}",
            )
        ]
        return {"news": [news.dict() for news in mock_news]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting news: {str(e)}")


@router.get("/settings")
async def get_settings():
    """Get current trading settings"""
    from ..config.settings import TRADING_CONFIG, TRADING_MODE

    return {
        "trading_mode": TRADING_MODE,
        "risk_settings": {
            "max_daily_loss": TRADING_CONFIG["max_daily_loss"],
            "max_position_loss": TRADING_CONFIG["max_position_loss"],
            "max_exposure_percent": TRADING_CONFIG["max_exposure_percent"],
            "max_concurrent_trades": TRADING_CONFIG["max_concurrent_trades"],
        },
        "filters": {
            "min_price": TRADING_CONFIG["min_price"],
            "max_price": TRADING_CONFIG["max_price"],
            "min_volume": TRADING_CONFIG["min_volume"],
        },
    }


@router.get("/performance")
async def get_performance():
    """Get today's performance metrics"""
    try:
        today = datetime.utcnow().date()
        performance = DailyPerformance.objects(date=today).first()

        if performance:
            return PerformanceResponse(
                date=performance.date.isoformat(),
                total_trades=performance.total_trades,
                gross_pnl=performance.gross_pnl,
                net_pnl=performance.net_pnl,
                win_rate=performance.win_rate or 0.0,
            )
        else:
            return PerformanceResponse(
                date=today.isoformat(),
                total_trades=0,
                gross_pnl=0.0,
                net_pnl=0.0,
                win_rate=0.0,
            )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting performance: {str(e)}"
        )


@router.get("/performance/daily")
async def get_daily_performance():
    """Get daily performance history"""
    try:
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=30)

        performances = DailyPerformance.objects(
            date__gte=start_date, date__lte=end_date
        ).order_by("date")

        return {
            "performance": [
                PerformanceResponse(
                    date=perf.date.isoformat(),
                    total_trades=perf.total_trades,
                    gross_pnl=perf.gross_pnl,
                    net_pnl=perf.net_pnl,
                    win_rate=perf.win_rate or 0.0,
                ).dict()
                for perf in performances
            ]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting daily performance: {str(e)}"
        )


@router.post("/kill-switch")
async def activate_kill_switch():
    """Activate emergency kill switch"""
    try:
        # For MVP, just log the activation
        # In production, this would trigger actual kill switch
        import logging

        logger = logging.getLogger(__name__)
        logger.critical("Kill switch activated via API")

        return {"message": "Kill switch activated"}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error activating kill switch: {str(e)}"
        )


@router.get("/logs/trades")
async def get_trade_logs(
    limit: int = Query(100, description="Maximum number of results", ge=1, le=1000)
):
    """Get trade logs"""
    try:
        trades = Trade.objects.all().order_by("-entry_time")[:limit]

        logs = [
            {
                "timestamp": trade.entry_time.isoformat(),
                "ticker": trade.ticker,
                "action": trade.side,
                "quantity": trade.quantity,
                "price": trade.entry_price,
                "pnl": trade.realized_pnl,
            }
            for trade in trades
        ]

        return {"logs": logs}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting trade logs: {str(e)}"
        )


@router.get("/logs/system")
async def get_system_logs(
    level: Optional[str] = Query(None, description="Filter by log level"),
    limit: int = Query(100, description="Maximum number of results", ge=1, le=1000),
):
    """Get system logs"""
    try:
        query = {}
        if level:
            query["level"] = level

        logs = SystemLog.objects(**query).order_by("-timestamp")[:limit]

        data = [
            {
                "timestamp": log.timestamp.isoformat(),
                "level": log.level,
                "category": log.category,
                "message": log.message,
            }
            for log in logs
        ]

        return {"logs": data}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting system logs: {str(e)}"
        )
