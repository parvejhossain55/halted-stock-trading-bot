# API Reference

## Overview

This document describes the REST API endpoints for the Halt Detector Trading System. The API provides access to trading data, system monitoring, and basic trading controls.

## Base URL
```
http://localhost:8000/api
```

## Authentication
Currently no authentication required (for development). In production, JWT tokens will be implemented.

---

## Endpoints

### System Health

#### GET /health
Check system health status.

**Response:**
```json
{
  "status": "healthy",
  "service": "halt-detector-api",
  "trading_mode": "paper"
}
```

#### GET /api/status
Get detailed system status.

**Response:**
```json
{
  "status": "operational",
  "trading_mode": "paper",
  "api_keys_configured": {
    "polygon": true
  }
}
```

---

### Trading Data

#### GET /api/positions
Get all current positions.

**Response:**
```json
{
  "positions": [
    {
      "position_id": "pos_123",
      "ticker": "AAPL",
      "side": "long",
      "quantity": 100,
      "entry_price": 150.25,
      "current_price": 151.00,
      "unrealized_pnl": 75.00,
      "opened_at": "2024-01-15T09:35:00Z"
    }
  ]
}
```

#### GET /api/trades
Get trade history.

**Query Parameters:**
- `ticker`: Filter by ticker symbol
- `status`: Filter by status (open, closed, cancelled)
- `limit`: Maximum results (default: 100)

**Response:**
```json
{
  "trades": [
    {
      "trade_id": "trade_456",
      "ticker": "TSLA",
      "side": "short",
      "entry_price": 250.00,
      "exit_price": 245.00,
      "quantity": 50,
      "pnl": 250.00,
      "status": "closed",
      "entry_time": "2024-01-15T10:00:00Z",
      "exit_time": "2024-01-15T10:30:00Z"
    }
  ]
}
```

#### GET /api/halts
Get detected halt events.

**Query Parameters:**
- `ticker`: Filter by ticker
- `limit`: Maximum results (default: 100)

**Response:**
```json
{
  "halts": [
    {
      "ticker": "NVDA",
      "halt_type": "halt_up",
      "halt_time": "2024-01-15T09:31:25Z",
      "resume_time": "2024-01-15T09:36:30Z",
      "halt_price": 450.20,
      "resume_price": 455.80,
      "gap_percent": 1.24
    }
  ]
}
```

---

### Trading Operations

#### POST /api/orders/submit
Submit a new order.

**Request:**
```json
{
  "ticker": "AAPL",
  "side": "buy",
  "quantity": 100,
  "price": 150.50
}
```

**Response:**
```json
{
  "order_id": "ord_789",
  "status": "submitted",
  "message": "Order submitted successfully"
}
```

#### POST /api/orders/{order_id}/cancel
Cancel an existing order.

**Response:**
```json
{
  "message": "Order cancelled successfully"
}
```

#### POST /api/positions/flatten
Emergency flatten all positions.

**Response:**
```json
{
  "message": "Emergency position flattening initiated",
  "orders_submitted": 3
}
```

---

### Configuration

#### GET /api/settings
Get current trading settings.

**Response:**
```json
{
  "trading_mode": "paper",
  "risk_settings": {
    "max_daily_loss": 1000.00,
    "max_position_loss": 200.00,
    "max_exposure_percent": 25.0,
    "max_concurrent_trades": 5
  },
  "filters": {
    "min_price": 5.0,
    "max_price": 300.0,
    "min_volume": 500000
  }
}
```

---

### Logs

#### GET /api/logs/trades
Get trade execution logs.

**Query Parameters:**
- `limit`: Maximum results (default: 100)

**Response:**
```json
{
  "logs": [
    {
      "timestamp": "2024-01-15T10:00:00Z",
      "ticker": "TSLA",
      "action": "SHORT",
      "quantity": 50,
      "price": 250.00,
      "pnl": 250.00
    }
  ]
}
```

#### GET /api/logs/system
Get system event logs.

**Query Parameters:**
- `level`: Filter by level (info, warning, error)
- `limit`: Maximum results (default: 100)

**Response:**
```json
{
  "logs": [
    {
      "timestamp": "2024-01-15T09:30:00Z",
      "level": "info",
      "category": "trading",
      "message": "Halt detected for AAPL"
    }
  ]
}
```

---

## Error Handling

All errors return standard HTTP status codes with JSON error details:

```json
{
  "detail": "Error message description"
}
```

Common status codes:
- `200` - Success
- `400` - Bad Request
- `404` - Not Found
- `500` - Internal Server Error

---

## Rate Limits

- Basic endpoints: 100 requests/minute
- Trading operations: 50 requests/minute

---

## Development Notes

- API is under active development
- Endpoints may change without notice
- Authentication will be added in future versions
- WebSocket support planned for real-time updates
