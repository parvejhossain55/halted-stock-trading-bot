#!/usr/bin/env python3
"""
Daily Metadata Sync Script

This script runs daily to refresh ticker metadata including:
- Float (shares outstanding)
- Market capitalization
- Average daily volume (ADV)
- Liquidity metrics

Data sources: Polygon.io API or Yahoo Finance as fallback
"""

import sys
import os
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../backend'))

from database.models import TickerMetadata
from config.trading_config import DATA_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/metadata_sync.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MetadataSync:
    """Handles daily synchronization of ticker metadata"""

    def __init__(self):
        self.polygon_api_key = os.getenv('POLYGON_API_KEY')
        self.yahoo_fallback = True
        self.tickers_updated = 0
        self.tickers_failed = 0

    def get_active_tickers(self) -> List[str]:
        """
        Get list of tickers to update.
        Includes: current watchlist + recently halted stocks + high-volume movers
        """
        try:
            # TODO: Query MongoDB for active tickers
            # For now, return example list
            tickers = ['AAPL', 'TSLA', 'SPY', 'QQQ']
            logger.info(f"Retrieved {len(tickers)} tickers for metadata update")
            return tickers
        except Exception as e:
            logger.error(f"Error fetching active tickers: {e}")
            return []

    def fetch_polygon_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch metadata from Polygon.io API"""
        try:
            # TODO: Implement Polygon API call
            # Example structure:
            # response = requests.get(
            #     f"https://api.polygon.io/v3/reference/tickers/{ticker}",
            #     params={'apiKey': self.polygon_api_key}
            # )

            data = {
                'ticker': ticker,
                'float': 0,  # shares_outstanding
                'market_cap': 0,
                'average_volume': 0,
                'last_updated': datetime.utcnow()
            }
            return data
        except Exception as e:
            logger.warning(f"Polygon fetch failed for {ticker}: {e}")
            return None

    def fetch_yahoo_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch metadata from Yahoo Finance as fallback"""
        try:
            # TODO: Implement Yahoo Finance API/scraping
            data = {
                'ticker': ticker,
                'float': 0,
                'market_cap': 0,
                'average_volume': 0,
                'last_updated': datetime.utcnow()
            }
            return data
        except Exception as e:
            logger.warning(f"Yahoo fetch failed for {ticker}: {e}")
            return None

    def update_ticker_metadata(self, ticker: str) -> bool:
        """Update metadata for a single ticker"""
        try:
            # Try Polygon first
            data = self.fetch_polygon_data(ticker)

            # Fallback to Yahoo if Polygon fails
            if not data and self.yahoo_fallback:
                data = self.fetch_yahoo_data(ticker)

            if not data:
                logger.error(f"All data sources failed for {ticker}")
                self.tickers_failed += 1
                return False

            # TODO: Save to MongoDB
            # TickerMetadata.update_or_create(ticker, data)

            self.tickers_updated += 1
            logger.info(f"Updated metadata for {ticker}")
            return True

        except Exception as e:
            logger.error(f"Error updating {ticker}: {e}")
            self.tickers_failed += 1
            return False

    def run(self):
        """Main execution function"""
        start_time = datetime.utcnow()
        logger.info("=" * 60)
        logger.info("Starting daily metadata sync")
        logger.info(f"Start time: {start_time}")

        # Get tickers to update
        tickers = self.get_active_tickers()

        if not tickers:
            logger.warning("No tickers to update")
            return

        # Update each ticker
        for ticker in tickers:
            self.update_ticker_metadata(ticker)

        # Log summary
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()

        logger.info("=" * 60)
        logger.info("Metadata sync completed")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Tickers updated: {self.tickers_updated}")
        logger.info(f"Tickers failed: {self.tickers_failed}")
        logger.info(f"Success rate: {(self.tickers_updated / len(tickers) * 100):.1f}%")
        logger.info("=" * 60)


def main():
    """Entry point for cron job"""
    try:
        sync = MetadataSync()
        sync.run()
    except Exception as e:
        logger.critical(f"Critical error in metadata sync: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
