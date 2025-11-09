# Halt Detector Architecture

## Overview

The Halt Detector is a real-time trading system that monitors LULD (Limit Up-Limit Down) halt events in U.S. equities and executes automated trading strategies. The system connects to Polygon WebSocket API for real-time market data and uses DAS CMD API for order execution.

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         REST API Clients                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────────┐ │
│  │ Positions│ │  Trades  │ │  Halts   │ │ Manual Controls    │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────────┘ │
└─────────────────────┬───────────────────────────────────────────┘
                      │ REST API (FastAPI)
┌─────────────────────┴───────────────────────────────────────────┐
│                    Backend (Python)                             │
│                                                                   │
│  ┌───────────────────┐  ┌────────────────┐  ┌──────────────┐  │
│  │   Data Layer      │  │   Strategy     │  │ Execution    │  │
│  │                   │  │   Engine       │  │ Layer        │  │
│  │ • Halt Detection  │  │ • Entry Rules  │  │ • DAS API     │  │
│  │ • Market Data     │  │ • Exit Rules   │  │ • Order Mgmt  │  │
│  │ • News Feed       │  │ • Risk Mgmt    │  │ • Position    │  │
│  │                   │  │                │  │   Tracking    │  │
│  └───────────────────┘  └────────────────┘  └──────────────┘  │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Database Layer (MongoDB)                    │   │
│  │  • Trades  • Positions  • Halts  • System Logs           │   │
│  └─────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼────┐      ┌───────▼──────┐    ┌──────▼──────┐
    │ Polygon │      │   Benzinga   │    │    DAS      │
    │ WebSocket│      │   News API   │    │  CMD API   │
    └─────────┘      └──────────────┘    └─────────────┘
```

## Core Components

### 1. Data Layer (`backend/core/data/`)

**Purpose:** Ingest and process real-time market data and halt events.

**Components:**
- `halt_detector.py` - Monitors Polygon WebSocket for LULD events
- `market_data.py` - Fetches market data and quotes
- `news_feed.py` - Integrates news data (planned)
- `polygon_client.py` - WebSocket client for Polygon API

**Data Flow:**
1. Connect to Polygon WebSocket
2. Monitor real-time price/quote updates
3. Detect LULD halt events
4. Store halt data in MongoDB

### 2. Strategy Engine (`backend/core/strategy/`)

**Purpose:** Implement trading rules and risk management.

**Components:**
- `entry_rules.py` - Define entry conditions for halt events
- `exit_rules.py` - Define exit conditions and stop losses
- `policy_engine.py` - Combine rules into trading decisions

**Trading Logic:**
- **Halt Up:** Short after price confirmation
- **Halt Down:** Long after price confirmation
- **Risk Controls:** Position limits, stop losses, daily loss limits

### 3. Execution Layer (`backend/core/execution/`)

**Purpose:** Interface with DAS trading API for order execution.

**Components:**
- `das_api.py` - DAS CMD API client
- `order_manager.py` - Order lifecycle management

**Execution Flow:**
1. Receive trading signal from strategy engine
2. Validate risk parameters
3. Submit order to DAS API
4. Monitor order status and fills

### 4. API Layer (`backend/api/`)

**Purpose:** Provide REST API for monitoring and control.

**Components:**
- `routes.py` - FastAPI route definitions

**Endpoints:**
- System health and status
- Trading data (positions, trades, halts)
- Manual order submission
- Configuration access

### 5. Database Layer (`backend/database/`)

**Purpose:** Persistent storage for all trading data.

**Collections:**
- `trades` - Executed trades with P&L
- `positions` - Current open positions
- `halts` - Historical halt events
- `system_logs` - Application logs

## Data Flow

### Halt Detection Flow
```
1. Polygon WebSocket → Real-time price updates
2. Halt Detector → Identify LULD events
3. Strategy Engine → Evaluate trading opportunity
4. Risk Manager → Validate position limits
5. Order Manager → Submit to DAS API
6. Database → Store trade results
```

### API Request Flow
```
Client → FastAPI → Database → Response
```

## Technology Stack

**Backend:**
- Python 3.10+
- FastAPI (web framework)
- MongoDB (database)
- WebSockets (real-time data)

**External APIs:**
- Polygon.io (market data & halts)
- DAS CMD API (order execution)
- Benzinga (news feed - planned)

**Infrastructure:**
- Docker & Docker Compose
- Makefile (development automation)

## Development Workflow

**Current Implementation:**
1. ✅ Real-time halt detection via Polygon
2. ✅ Basic trading strategy rules
3. ✅ DAS API integration for execution
4. ✅ REST API for monitoring
5. ✅ MongoDB for data persistence
6. ✅ Docker containerization

**Future Enhancements:**
- Advanced strategy rules
- News sentiment analysis
- Performance analytics
- Web dashboard
- Backtesting framework

## Deployment

**Development:**
```bash
make dev-setup  # Setup environment
make run        # Start trading engine
make run-api    # Start API server (separate terminal)
```

**Production:**
```bash
docker-compose up -d
```

## Monitoring

**Logs:**
- `backend/logs/trading.log` - Trading activity
- `backend/logs/error.log` - Error messages
- `backend/logs/debug.log` - Debug information

**Health Checks:**
```bash
make health     # System status
curl localhost:8000/health  # API health
```

---

**Version:** 1.0
**Status:** Active Development
