"""
Structured logging system for the trading bot.
Provides comprehensive logging for trades, signals, executions, and system events.
"""

import logging
import json
from datetime import datetime
from typing import Any, Dict, Optional
from enum import Enum


class LogLevel(Enum):
    """Log severity levels"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(Enum):
    """Log categories for filtering and analysis"""

    SYSTEM = "SYSTEM"
    DATA = "DATA"
    HALT = "HALT"
    SIGNAL = "SIGNAL"
    AI = "AI"
    EXECUTION = "EXECUTION"
    RISK = "RISK"
    PNL = "PNL"
    BACKTEST = "BACKTEST"


class StructuredLogger:
    """
    Structured logger that outputs JSON-formatted logs for easy parsing and analysis.
    """

    def __init__(self, name: str = "trading_bot"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Console handler with JSON formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self._get_json_formatter())

        # File handler for all logs
        file_handler = logging.FileHandler("logs/trading_bot.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(self._get_json_formatter())

        # Error file handler
        error_handler = logging.FileHandler("logs/errors.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(self._get_json_formatter())

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)

    def _get_json_formatter(self):
        """Returns a JSON formatter for structured logging"""
        return logging.Formatter("%(message)s")

    def _format_log(
        self,
        level: LogLevel,
        category: LogCategory,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        ticker: Optional[str] = None,
    ) -> str:
        """Format log entry as JSON"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level.value,
            "category": category.value,
            "message": message,
        }

        if ticker:
            log_entry["ticker"] = ticker

        if data:
            log_entry["data"] = data

        return json.dumps(log_entry)

    def debug(
        self,
        category: LogCategory,
        message: str,
        data: Optional[Dict] = None,
        ticker: Optional[str] = None,
    ):
        """Log debug message"""
        log_msg = self._format_log(LogLevel.DEBUG, category, message, data, ticker)
        self.logger.debug(log_msg)

    def info(
        self,
        category: LogCategory,
        message: str,
        data: Optional[Dict] = None,
        ticker: Optional[str] = None,
    ):
        """Log info message"""
        log_msg = self._format_log(LogLevel.INFO, category, message, data, ticker)
        self.logger.info(log_msg)

    def warning(
        self,
        category: LogCategory,
        message: str,
        data: Optional[Dict] = None,
        ticker: Optional[str] = None,
    ):
        """Log warning message"""
        log_msg = self._format_log(LogLevel.WARNING, category, message, data, ticker)
        self.logger.warning(log_msg)

    def error(
        self,
        category: LogCategory,
        message: str,
        data: Optional[Dict] = None,
        ticker: Optional[str] = None,
    ):
        """Log error message"""
        log_msg = self._format_log(LogLevel.ERROR, category, message, data, ticker)
        self.logger.error(log_msg)

    def critical(
        self,
        category: LogCategory,
        message: str,
        data: Optional[Dict] = None,
        ticker: Optional[str] = None,
    ):
        """Log critical message"""
        log_msg = self._format_log(LogLevel.CRITICAL, category, message, data, ticker)
        self.logger.critical(log_msg)

    # Convenience methods for common log types

    def log_halt_detected(self, ticker: str, halt_type: str, price: float, data: Dict):
        """Log halt detection"""
        self.info(
            LogCategory.HALT,
            f"Halt detected: {halt_type}",
            data={"halt_type": halt_type, "price": price, **data},
            ticker=ticker,
        )

    def log_signal_generated(
        self, ticker: str, signal_type: str, confidence: float, data: Dict
    ):
        """Log trading signal generation"""
        self.info(
            LogCategory.SIGNAL,
            f"Signal generated: {signal_type}",
            data={"signal_type": signal_type, "confidence": confidence, **data},
            ticker=ticker,
        )

    def log_ai_decision(
        self, ticker: str, decision: str, probabilities: Dict, rationale: str
    ):
        """Log AI decision"""
        self.info(
            LogCategory.AI,
            f"AI decision: {decision}",
            data={
                "decision": decision,
                "probabilities": probabilities,
                "rationale": rationale,
            },
            ticker=ticker,
        )

    def log_order_sent(
        self,
        ticker: str,
        side: str,
        quantity: int,
        order_type: str,
        price: Optional[float] = None,
    ):
        """Log order submission"""
        self.info(
            LogCategory.EXECUTION,
            f"Order sent: {side} {quantity} shares",
            data={
                "side": side,
                "quantity": quantity,
                "order_type": order_type,
                "price": price,
            },
            ticker=ticker,
        )

    def log_order_filled(
        self, ticker: str, side: str, quantity: int, fill_price: float, order_id: str
    ):
        """Log order fill"""
        self.info(
            LogCategory.EXECUTION,
            f"Order filled: {side} {quantity} @ ${fill_price}",
            data={
                "side": side,
                "quantity": quantity,
                "fill_price": fill_price,
                "order_id": order_id,
            },
            ticker=ticker,
        )

    def log_risk_violation(
        self, risk_type: str, current_value: float, limit: float, action: str
    ):
        """Log risk limit violation"""
        self.warning(
            LogCategory.RISK,
            f"Risk violation: {risk_type}",
            data={
                "risk_type": risk_type,
                "current_value": current_value,
                "limit": limit,
                "action": action,
            },
        )

    def log_pnl_update(
        self, ticker: str, realized_pnl: float, unrealized_pnl: float, total_pnl: float
    ):
        """Log PnL update"""
        self.info(
            LogCategory.PNL,
            f"PnL Update",
            data={
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "total_pnl": total_pnl,
            },
            ticker=ticker,
        )

    def log_trade_closed(
        self,
        ticker: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        hold_time: str,
    ):
        """Log trade closure"""
        self.info(
            LogCategory.PNL,
            f"Trade closed with {'profit' if pnl > 0 else 'loss'}: ${pnl:.2f}",
            data={
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "hold_time": hold_time,
            },
            ticker=ticker,
        )

    def log_system_event(self, event: str, data: Optional[Dict] = None):
        """Log system event"""
        self.info(LogCategory.SYSTEM, event, data=data)


# Global logger instance
logger = StructuredLogger()


# Export convenience functions
def log_halt_detected(ticker: str, halt_type: str, price: float, data: Dict):
    logger.log_halt_detected(ticker, halt_type, price, data)


def log_signal_generated(ticker: str, signal_type: str, confidence: float, data: Dict):
    logger.log_signal_generated(ticker, signal_type, confidence, data)


def log_ai_decision(ticker: str, decision: str, probabilities: Dict, rationale: str):
    logger.log_ai_decision(ticker, decision, probabilities, rationale)


def log_order_sent(
    ticker: str,
    side: str,
    quantity: int,
    order_type: str,
    price: Optional[float] = None,
):
    logger.log_order_sent(ticker, side, quantity, order_type, price)


def log_order_filled(
    ticker: str, side: str, quantity: int, fill_price: float, order_id: str
):
    logger.log_order_filled(ticker, side, quantity, fill_price, order_id)


def log_risk_violation(risk_type: str, current_value: float, limit: float, action: str):
    logger.log_risk_violation(risk_type, current_value, limit, action)


def log_pnl_update(
    ticker: str, realized_pnl: float, unrealized_pnl: float, total_pnl: float
):
    logger.log_pnl_update(ticker, realized_pnl, unrealized_pnl, total_pnl)


def log_trade_closed(
    ticker: str, entry_price: float, exit_price: float, pnl: float, hold_time: str
):
    logger.log_trade_closed(ticker, entry_price, exit_price, pnl, hold_time)


def log_system_event(event: str, data: Optional[Dict] = None):
    logger.log_system_event(event, data)
