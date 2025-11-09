#!/usr/bin/env python3
"""
Test script for paper trading functionality

Usage:
    python scripts/test_paper_trading.py
"""

import sys
import os
import asyncio
import logging

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from core.execution.das_api import create_das_client
from core.execution.order_manager import create_order_manager
from database.models import init_db, create_indexes


async def test_paper_trading():
    """Test paper trading functionality"""
    print("Testing Halt Detector Paper Trading...")

    try:
        # Initialize database
        print("Initializing database...")
        from config.settings import MONGODB_SETTINGS
        db_uri = f"mongodb://{MONGODB_SETTINGS['host']}:{MONGODB_SETTINGS['port']}"
        init_db(db_uri, MONGODB_SETTINGS['db'])
        create_indexes()
        print("Database initialized")

        # Create DAS client (paper trading mode)
        print("Creating DAS client...")
        das_client = create_das_client(paper_trading=True)
        connected = das_client.connect()
        print(f"DAS client connected: {connected}")

        # Create order manager
        print("Creating order manager...")
        order_manager = create_order_manager(das_client)

        # Test order submission
        print("Testing order submission...")
        trade_decision = {
            'action': 'BUY',
            'ticker': 'AAPL',
            'quantity': 100,
            'price': 150.0,
            'confidence': 0.8,
            'rationale': 'Test paper trade'
        }

        result = order_manager.execute_trade_decision(trade_decision)
        print(f"Order result: {result.success} - {result.message}")

        # Check positions
        print("Checking positions...")
        positions = order_manager.get_positions()
        print(f"Current positions: {len(positions)}")

        for pos in positions:
            print(f"  {pos['ticker']}: {pos['quantity']} shares at ${pos['entry_price']}")

        # Test position flattening
        print("Testing position flattening...")
        flatten_results = order_manager.flatten_all_positions()
        print(f"Flatten results: {len(flatten_results)} orders submitted")

        print("Paper trading test completed successfully!")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Run test
    success = asyncio.run(test_paper_trading())
    sys.exit(0 if success else 1)
