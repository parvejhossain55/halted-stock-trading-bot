# halt-detector-zed/backend/core/data/metadata_sync.py
"""
Metadata Sync Module - Polygon Integration

This module handles the synchronization of ticker metadata using Polygon API.
Includes float size, market cap, average daily volume (ADV), and other fundamentals.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from .polygon_client import PolygonClient

logger = logging.getLogger(__name__)


class MetadataSync:
    """
    Handles daily synchronization of ticker metadata using Polygon API.
    Stores data in MongoDB for fast retrieval by trading systems.
    """

    def __init__(self, api_key: str, db_client):
        """
        Initialize metadata sync with Polygon API and MongoDB.

        Args:
            api_key: Polygon API key
            db_client: MongoDB client instance
        """
        self.api_key = api_key
        self.client = PolygonClient(api_key)
        self.rest_client = self.client.rest
        self.db = db_client
        self.collection: Collection = self.db.metadata.ticker_details

        # Create indexes for faster queries
        try:
            self.collection.create_index([("ticker", 1)], unique=True)
            self.collection.create_index([("last_updated", -1)])
            self.collection.create_index([("market_cap", -1)])
            self.collection.create_index([("float", -1)])
        except PyMongoError as e:
            logger.warning(f"Failed to create indexes: {e}")

        logger.info("MetadataSync initialized with Polygon API integration")

    def sync_ticker(self, ticker: str) -> Dict[str, Any]:
        """
        Sync metadata for a single ticker using Polygon API.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary containing ticker metadata and sync status
        """
        try:
            logger.info(f"Syncing metadata for {ticker}")

            # Fetch from Polygon API
            ticker_data = self.rest_client.get_ticker_details(ticker.upper())

            if not ticker_data:
                logger.warning(f"No data available for {ticker}")
                return self._create_error_result(ticker, "No data available")

            # Process and store the data
            metadata = self._process_ticker_data(ticker_data)

            # Upsert to MongoDB
            result = self.collection.update_one(
                {"ticker": ticker.upper()}, {"$set": metadata}, upsert=True
            )

            if result.upserted_id or result.modified_count > 0:
                logger.info(f"Successfully synced metadata for {ticker}")
                return {
                    "ticker": ticker.upper(),
                    "status": "success",
                    "operation": "upserted" if result.upserted_id else "updated",
                    "data": metadata,
                }
            else:
                return self._create_error_result(ticker, "No changes made")

        except Exception as e:
            logger.error(f"Error syncing {ticker}: {str(e)}")
            return self._create_error_result(ticker, str(e))

    def sync_multiple_tickers(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Sync metadata for multiple tickers.

        Args:
            tickers: List of ticker symbols

        Returns:
            Summary of sync operation
        """
        results = []
        successful = 0
        failed = 0

        for ticker in tickers:
            result = self.sync_ticker(ticker)

            if result.get("status") == "success":
                successful += 1
            else:
                failed += 1

            results.append(result)

        summary = {
            "total_tickers": len(tickers),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(tickers) if tickers else 0,
            "results": results,
            "timestamp": datetime.utcnow(),
        }

        logger.info(f"Batch sync completed: {successful}/{len(tickers)} successful")
        return summary

    def sync_active_tickers(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Sync all tickers that have been actively traded in recent history.

        Args:
            days_back: How many days back to look for active tickers

        Returns:
            Sync summary
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)

            # Find active tickers from recent trades and halts
            pipeline = [
                {
                    "$match": {
                        "$or": [
                            {"entry_time": {"$gte": cutoff_date}},
                            {"halt_detected_at": {"$gte": cutoff_date}},
                        ]
                    }
                },
                {"$group": {"_id": None, "tickers": {"$addToSet": "$ticker"}}},
            ]

            # Get unique tickers from trades collection
            trades_result = self.db.trades.aggregate(pipeline)
            active_tickers = set()

            for result in trades_result:
                active_tickers.update(result.get("tickers", []))

            # Also check halts collection
            halts_result = self.db.halts.aggregate(pipeline)
            for result in halts_result:
                active_tickers.update(result.get("tickers", []))

            active_tickers = list(active_tickers)

            logger.info(f"Found {len(active_tickers)} active tickers to sync")

            return self.sync_multiple_tickers(active_tickers)

        except Exception as e:
            logger.error(f"Error syncing active tickers: {e}")
            return {"error": str(e)}

    def _process_ticker_data(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw Polygon API data into standardized format.

        Args:
            api_data: Raw data from Polygon ticker details API

        Returns:
            Processed metadata dictionary
        """
        # Extract core information
        processed = {
            "ticker": api_data.get("ticker", "").upper(),
            "name": api_data.get("name", ""),
            "description": api_data.get("description", ""),
            "sic_code": api_data.get("sic_code"),
            "sic_description": api_data.get("sic_description"),
            "cik": api_data.get("cik"),
            "composite_figi": api_data.get("composite_figi"),
            "share_class_figi": api_data.get("share_class_figi"),
            "currency_name": api_data.get("currency_name", "USD"),
            "primary_exchange": api_data.get("primary_exchange"),
            "type": api_data.get("type"),
            "market": api_data.get("market"),
            # Financial metrics
            "market_cap": api_data.get("market_cap"),
            "shares_outstanding": api_data.get("weighted_shares_outstanding"),
            "float": api_data.get("share_class_shares_outstanding"),
            # Trading attributes
            "delisted_utc": api_data.get("delisted_utc"),
            "list_date": api_data.get("list_date"),
            "is_active": api_data.get("active", True),
            # Additional metadata
            "source": "polygon",
            "last_updated": datetime.utcnow(),
            "api_version": api_data.get("api_version"),
            "update_count": 1,
        }

        # Handle dates
        for date_field in ["delisted_utc", "list_date"]:
            if processed[date_field]:
                try:
                    # Convert ISO string to datetime if needed
                    if isinstance(processed[date_field], str):
                        processed[date_field] = datetime.fromisoformat(
                            processed[date_field].replace("Z", "+00:00")
                        )
                except ValueError:
                    processed[date_field] = None

        # Calculate additional derived fields
        processed.update(self._calculate_derived_fields(processed))

        return processed

    def _calculate_derived_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate additional derived fields from the raw data.

        Args:
            data: Processed ticker data

        Returns:
            Dictionary of derived fields
        """
        derived = {}

        # Float rotation calculation would require volume data
        # This is a placeholder for now
        derived["float_percentage"] = None

        # Market cap category
        market_cap = data.get("market_cap")
        if market_cap:
            if market_cap >= 10_000_000_000:  # $10B+
                derived["market_cap_category"] = "large_cap"
            elif market_cap >= 2_000_000_000:  # $2B-$10B
                derived["market_cap_category"] = "mid_cap"
            elif market_cap >= 300_000_000:  # $300M-$2B
                derived["market_cap_category"] = "small_cap"
            else:  # <$300M
                derived["market_cap_category"] = "micro_cap"
        else:
            derived["market_cap_category"] = None

        # Float category
        float_shares = data.get("float")
        if float_shares:
            if float_shares <= 10_000_000:  # <=10M
                derived["float_category"] = "nano_float"
            elif float_shares <= 50_000_000:  # 10M-50M
                derived["float_category"] = "low_float"
            elif float_shares <= 200_000_000:  # 50M-200M
                derived["float_category"] = "medium_float"
            else:  # >200M
                derived["float_category"] = "high_float"
        else:
            derived["float_category"] = None

        return derived

    def get_ticker_metadata(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached metadata for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Metadata dictionary or None if not found
        """
        try:
            result = self.collection.find_one({"ticker": ticker.upper()})
            return result
        except PyMongoError as e:
            logger.error(f"Error retrieving metadata for {ticker}: {e}")
            return None

    def get_tickers_by_criteria(
        self,
        min_market_cap: Optional[float] = None,
        max_market_cap: Optional[float] = None,
        min_float: Optional[float] = None,
        max_float: Optional[float] = None,
        market_cap_categories: Optional[List[str]] = None,
        float_categories: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Query tickers that match specific criteria.

        Args:
            min_market_cap: Minimum market cap
            max_market_cap: Maximum market cap
            min_float: Minimum float
            max_float: Maximum float
            market_cap_categories: List of market cap categories
            float_categories: List of float categories
            limit: Maximum number of results

        Returns:
            List of matching ticker metadata
        """
        query = {"is_active": True}

        if min_market_cap is not None:
            query["market_cap"] = {"$gte": min_market_cap}
        if max_market_cap is not None:
            query["market_cap"] = query.get("market_cap", {})
            query["market_cap"]["$lte"] = max_market_cap

        if min_float is not None:
            query["float"] = {"$gte": min_float}
        if max_float is not None:
            query["float"] = query.get("float", {})
            query["float"]["$lte"] = max_float

        if market_cap_categories:
            query["market_cap_category"] = {"$in": market_cap_categories}

        if float_categories:
            query["float_category"] = {"$in": float_categories}

        try:
            cursor = self.collection.find(query).limit(limit)
            return list(cursor)
        except PyMongoError as e:
            logger.error(f"Error querying tickers: {e}")
            return []

    def is_metadata_stale(self, ticker: str, max_age_hours: int = 24) -> bool:
        """
        Check if cached metadata is stale and needs refresh.

        Args:
            ticker: Stock ticker symbol
            max_age_hours: Maximum age before considered stale

        Returns:
            True if stale or missing, False otherwise
        """
        metadata = self.get_ticker_metadata(ticker)

        if not metadata:
            return True

        last_updated = metadata.get("last_updated")
        if not last_updated:
            return True

        age = datetime.utcnow() - last_updated
        return age.total_seconds() > (max_age_hours * 3600)

    def _create_error_result(self, ticker: str, error: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            "ticker": ticker.upper(),
            "status": "error",
            "error": error,
            "timestamp": datetime.utcnow(),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get metadata sync statistics.

        Returns:
            Statistics about the metadata collection
        """
        try:
            total_tickers = self.collection.count_documents({})
            active_tickers = self.collection.count_documents({"is_active": True})

            # Market cap distribution
            mc_dist = {}
            mc_dist_cursor = self.collection.aggregate(
                [
                    {"$match": {"is_active": True, "market_cap": {"$exists": True}}},
                    {"$group": {"_id": "$market_cap_category", "count": {"$sum": 1}}},
                ]
            )
            for doc in mc_dist_cursor:
                mc_dist[doc["_id"]] = doc["count"]

            # Float distribution
            float_dist = {}
            float_dist_cursor = self.collection.aggregate(
                [
                    {"$match": {"is_active": True, "float": {"$exists": True}}},
                    {"$group": {"_id": "$float_category", "count": {"$sum": 1}}},
                ]
            )
            for doc in float_dist_cursor:
                float_dist[doc["_id"]] = doc["count"]

            return {
                "total_tickers": total_tickers,
                "active_tickers": active_tickers,
                "market_cap_distribution": mc_dist,
                "float_distribution": float_dist,
                "generated_at": datetime.utcnow(),
            }

        except PyMongoError as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}


# Factory function
def create_metadata_sync(api_key: str, db_client) -> MetadataSync:
    """
    Factory function to create MetadataSync instance.

    Args:
        api_key: Polygon API key
        db_client: MongoDB client

    Returns:
        Configured MetadataSync instance
    """
    return MetadataSync(api_key, db_client)


# Example usage and testing
if __name__ == "__main__":
    # Example usage (requires actual API key and MongoDB)
    import os
    from pymongo import MongoClient

    # This would be called from the main application
    api_key = os.getenv("POLYGON_API_KEY")
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

    if api_key:
        mongo_client = MongoClient(mongo_uri)
        sync = create_metadata_sync(api_key, mongo_client)

        # Test sync single ticker
        result = sync.sync_ticker("AAPL")
        print(f"Sync result: {result}")

        # Test sync multiple
        summary = sync.sync_multiple_tickers(["TSLA", "NVDA", "MSFT"])
        print(
            f"Batch sync: {summary['successful']}/{summary['total_tickers']} successful"
        )

    else:
        print("POLYGON_API_KEY environment variable not set")
