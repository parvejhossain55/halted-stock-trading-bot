"""
Trading Components Manager

Handles lazy initialization and management of trading components
to avoid circular imports in FastAPI application.
"""

from typing import Optional, Tuple
from .execution.das_api import create_das_client
from .execution.order_manager import create_order_manager
from ..config.settings import TRADING_MODE


# Global variables for lazy initialization
_das_client = None
_order_manager = None


def get_trading_components() -> Tuple[Optional[object], Optional[object]]:
    """
    Lazy initialization of trading components

    Returns:
        Tuple of (das_client, order_manager)
    """
    global _das_client, _order_manager

    if _das_client is None:
        _das_client = create_das_client(paper_trading=(TRADING_MODE == "paper"))
        _das_client.connect()

    if _order_manager is None:
        _order_manager = create_order_manager(_das_client)

    return _das_client, _order_manager


def reset_trading_components():
    """Reset trading components (useful for testing)"""
    global _das_client, _order_manager

    if _das_client:
        _das_client.disconnect()

    _das_client = None
    _order_manager = None
