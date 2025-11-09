"""
Configuration Settings for Halt Detector Trading System

Environment-based configuration for FastAPI application.
"""

import os
from pathlib import Path
from typing import Optional

# Build paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Environment variables with defaults
DEBUG = os.environ.get("DEBUG", "True").lower() == "true"
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")

# MongoDB Configuration
MONGODB_SETTINGS = {
    "host": os.environ.get("MONGO_HOST", "localhost"),
    "port": int(os.environ.get("MONGO_PORT", 27017)),
    "db": os.environ.get("MONGO_DB", "halt_detector"),
    "username": os.environ.get("MONGO_USER", ""),
    "password": os.environ.get("MONGO_PASSWORD", ""),
}

# Redis Configuration
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB = int(os.environ.get("REDIS_DB", 0))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD")

# Celery Configuration
CELERY_BROKER_URL = (
    f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    if REDIS_PASSWORD
    else f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
)
CELERY_RESULT_BACKEND = CELERY_BROKER_URL

# Trading System Configuration
TRADING_MODE = os.environ.get("PAPER_TRADING_MODE", "True").lower() == "true"

# API Keys
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")

# DAS Trading Configuration
DAS_API_ENDPOINT = os.environ.get("DAS_API_ENDPOINT", "")
DAS_API_KEY = os.environ.get("DAS_API_KEY", "")

# Server Configuration
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))

# CORS Settings
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

# Logging Configuration
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Create logs directory if it doesn't exist
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Trading Configuration
TRADING_CONFIG = {
    # Risk settings
    "max_daily_loss": float(os.environ.get("MAX_DAILY_LOSS", 1000.00)),
    "max_position_loss": float(os.environ.get("MAX_POSITION_LOSS", 200.00)),
    "max_exposure_percent": float(os.environ.get("MAX_EXPOSURE_PERCENT", 25.0)),
    "max_concurrent_trades": int(os.environ.get("MAX_CONCURRENT_TRADES", 5)),
    "daily_trade_limit": int(os.environ.get("DAILY_TRADE_LIMIT", 20)),
    "min_confidence_threshold": float(os.environ.get("MIN_CONFIDENCE_THRESHOLD", 0.6)),
    "max_risk_per_trade_percent": float(
        os.environ.get("MAX_RISK_PER_TRADE_PERCENT", 1.0)
    ),
    # Stock filters
    "min_price": float(os.environ.get("MIN_STOCK_PRICE", 5.0)),
    "max_price": float(os.environ.get("MAX_STOCK_PRICE", 300.0)),
    "min_volume": int(os.environ.get("MIN_VOLUME", 500000)),
    # Strategy settings
    "vwap_reversion_threshold": float(os.environ.get("VWAP_REVERSION_THRESHOLD", 0.1)),
    "gap_fill_threshold": float(os.environ.get("GAP_FILL_THRESHOLD", 0.5)),
    "trailing_stop_distance": float(os.environ.get("TRAILING_STOP_DISTANCE", 0.5)),
    "max_hold_time_minutes": int(os.environ.get("MAX_HOLD_TIME_MINUTES", 60)),
}

# Application metadata
APP_NAME = "Halt Detector Trading System"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Event-driven intraday equity trading system"
