"""
FastAPI Application for Halt Detector Trading System

Main entry point for the REST API server.
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Database and core imports
from .database.models import init_db, create_indexes
from .config.settings import (
    MONGODB_SETTINGS,
    TRADING_MODE,
    POLYGON_API_KEY,
)

# API routes
from .api.routes import router as api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Halt Detector API Server...")

    # Initialize database
    try:
        db_uri = f"mongodb://{MONGODB_SETTINGS['host']}:{MONGODB_SETTINGS['port']}"
        init_db(db_uri, MONGODB_SETTINGS["db"])
        create_indexes()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        # Continue without database for API-only mode

    # Initialize trading components (lazy loading)
    app.state.das_client = None
    app.state.order_manager = None

    yield

    # Shutdown
    logger.info("Shutting down Halt Detector API Server...")
    if app.state.das_client:
        app.state.das_client.disconnect()


# Create FastAPI app
app = FastAPI(
    title="Halt Detector Trading System",
    description="Event-driven intraday equity trading system",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "halt-detector-api",
        "trading_mode": TRADING_MODE,
    }


# System status endpoint
@app.get("/api/status")
async def get_system_status():
    """Get system status"""
    return {
        "status": "operational",
        "trading_mode": TRADING_MODE,
        "api_keys_configured": {
            "polygon": bool(POLYGON_API_KEY),
        },
    }


# Include API routes
app.include_router(api_router, prefix="/api")


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    # Don't expose internal details in production
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True,
        log_level="info",
    )
