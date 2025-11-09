"""
Logging Configuration for Halt Detector Trading System

Configures structured logging to console and files with proper formatting.
"""

import os
import logging
import logging.handlers
from pathlib import Path
from .settings import LOG_LEVEL, LOG_FORMAT, LOGS_DIR

# Ensure logs directory exists
LOGS_DIR.mkdir(exist_ok=True)

# Log file paths
TRADING_LOG = LOGS_DIR / "trading.log"
ERROR_LOG = LOGS_DIR / "error.log"
DEBUG_LOG = LOGS_DIR / "debug.log"

# Convert string log level to logging constant
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

def get_log_level(level_str: str) -> int:
    """Convert string log level to logging constant"""
    return LOG_LEVEL_MAP.get(level_str.upper(), logging.INFO)

def setup_logging():
    """Set up comprehensive logging configuration"""

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set root logger level
    root_logger.setLevel(logging.DEBUG)

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )

    standard_formatter = logging.Formatter(LOG_FORMAT)

    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(get_log_level(LOG_LEVEL))
    console_handler.setFormatter(standard_formatter)
    root_logger.addHandler(console_handler)

    # Trading log handler (INFO and above)
    trading_handler = logging.handlers.RotatingFileHandler(
        TRADING_LOG,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    trading_handler.setLevel(logging.INFO)
    trading_handler.setFormatter(standard_formatter)
    root_logger.addHandler(trading_handler)

    # Error log handler (ERROR and above)
    error_handler = logging.handlers.RotatingFileHandler(
        ERROR_LOG,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)

    # Debug log handler (DEBUG and above) - only in development
    if os.environ.get("DEBUG", "False").lower() == "true":
        debug_handler = logging.handlers.RotatingFileHandler(
            DEBUG_LOG,
            maxBytes=50*1024*1024,  # 50MB
            backupCount=3
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(debug_handler)

    # Set specific log levels for noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("polygon").setLevel(logging.WARNING)
    logging.getLogger("motor").setLevel(logging.WARNING)
    logging.getLogger("pymongo").setLevel(logging.WARNING)

    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")
    logger.info(f"Trading logs: {TRADING_LOG}")
    logger.info(f"Error logs: {ERROR_LOG}")
    if os.environ.get("DEBUG", "False").lower() == "true":
        logger.info(f"Debug logs: {DEBUG_LOG}")

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)

# Initialize logging when module is imported
setup_logging()
