#!/usr/bin/env python3
"""
Script to run the Halt Detector Trading Engine

Usage:
    python scripts/run_trading_engine.py
"""

import sys
import os
import asyncio

# Set up proper Python path for backend package
backend_dir = os.path.join(os.path.dirname(__file__), '..', 'backend')
sys.path.insert(0, os.path.dirname(backend_dir))  # Add parent directory of backend

# Import and initialize logging (this sets up file logging)
from backend.config.logging import get_logger

from backend.core.trading_engine import main

logger = get_logger(__name__)


if __name__ == "__main__":
    logger.info("Starting Halt Detector Trading Engine via script")

    # Run the trading engine
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Trading engine stopped by user")
        print("\nTrading engine stopped by user")
    except Exception as e:
        logger.error(f"Error running trading engine: {e}")
        print(f"Error running trading engine: {e}")
        sys.exit(1)
