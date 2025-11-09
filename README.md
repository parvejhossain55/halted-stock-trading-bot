# Halt Detector Trading System

A real-time trading system that detects and responds to LULD (Limit Up-Limit Down) halt events in U.S. equities using Polygon WebSocket API and automated execution via DAS CMD API.

## Overview

This system monitors stock price halts in real-time and can execute automated trading strategies based on halt events. It provides both programmatic API access and direct trading engine control.

## Features

- **Real-time Halt Detection**: Monitors LULD events via Polygon WebSocket
- **Automated Trading**: Executes trades based on configurable strategies
- **Risk Management**: Built-in position limits and emergency controls
- **REST API**: FastAPI-based endpoints for monitoring and control
- **Comprehensive Logging**: Structured logging to files and console
- **Docker Support**: Containerized deployment with Docker Compose

## Project Structure

```
halt-detector-zed/
├── backend/              # Python FastAPI backend
│   ├── api/             # REST API endpoints
│   ├── core/            # Core trading logic
│   │   ├── data/        # Data ingestion (Polygon, news feeds)
│   │   ├── strategy/    # Trading strategies and rules
│   │   └── execution/   # Order execution (DAS API)
│   ├── database/        # MongoDB models
│   ├── config/          # Configuration and logging
│   └── logs/            # Application logs
├── scripts/            # Utility scripts
├── tests/              # Test suites
├── docs/               # Documentation
├── docker-compose.yml  # Docker services
├── Makefile           # Development commands
└── requirements.txt   # Python dependencies
```

## Technology Stack

- **Backend**: Python 3.10+, FastAPI
- **Database**: MongoDB
- **Real-time Data**: Polygon.io WebSocket API
- **Execution**: Cobra Trading DAS CMD API
- **Deployment**: Docker & Docker Compose
- **Logging**: Structured logging with file rotation

## Quick Start

### 1. Environment Setup
```bash
# Clone and setup
git clone <repository-url>
cd halt-detector-zed

# Use Makefile for easy setup
make dev-setup
```

### 2. Configure API Keys
Edit `.env` file with your API keys:
```bash
POLYGON_API_KEY=your_polygon_key_here
# Add other required API keys
```

### 3. Run the System
```bash
# Start trading engine
make run

# Or start API server
make run-api

# Check system health
make health
```

## Usage

### Trading Engine
The core trading engine monitors halt events and executes trades:
```bash
make run  # Runs continuously until Ctrl+C
```

### REST API
Access trading data and controls via REST API:
```bash
make run-api  # Starts FastAPI server on port 8000
```

### Available Endpoints
- `GET /health` - System health check
- `GET /api/status` - Trading system status
- `GET /api/positions` - Current positions
- `GET /api/trades` - Trade history
- `GET /api/halts` - Halt events
- `POST /api/orders` - Submit orders
- `POST /api/kill-switch` - Emergency stop

## Configuration

### Environment Variables
Key settings in `.env`:
- `POLYGON_API_KEY` - Real-time market data
- `MONGO_HOST` - Database connection
- `PAPER_TRADING_MODE` - Enable paper trading
- `LOG_LEVEL` - Logging verbosity

### Trading Parameters
Configure risk and trading settings in `backend/config/settings.py`:
- Daily loss limits
- Position sizing
- Entry/exit rules
- Risk management

## Development

### Makefile Commands
```bash
make help          # Show all commands
make test          # Run tests
make lint          # Code quality checks
make format        # Format code
make clean         # Clean temporary files
make health        # System health check
```

### Project Structure Details
- `backend/core/` - Trading logic and strategies
- `backend/api/` - REST API endpoints
- `backend/database/` - MongoDB models
- `backend/config/` - Configuration and logging
- `scripts/` - Utility scripts
- `docs/` - Documentation

## Logging

Logs are automatically saved to:
- `backend/logs/trading.log` - Main trading activity
- `backend/logs/error.log` - Error messages
- `backend/logs/debug.log` - Debug information (when DEBUG=true)

View logs:
```bash
tail -f backend/logs/trading.log
```

## Security

- **API Keys**: Store securely in `.env` file (never commit)
- **Environment Variables**: Use for production deployments
- **Database**: Enable authentication in production
- **Logs**: Contain sensitive trading data - monitor access

## License

Proprietary - All rights reserved

---

**⚠️ Disclaimer**: This is an automated trading system. Trading involves substantial risk. Past performance does not guarantee future results. Use at your own risk.
