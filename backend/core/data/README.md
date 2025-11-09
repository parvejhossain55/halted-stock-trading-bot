# Polygon API Integrations - Comprehensive Guide

## Overview

This module provides complete integration with Polygon APIs for real-time and historical financial data, specifically designed for halt-based trading strategies. The implementation follows Polygon's official documentation and includes LULD halts, market data, news feeds, and metadata synchronization.

## Key Features

- **Real-time LULD Halt Detection**: WebSocket-based monitoring of Limit Up-Limit Down price bands
- **OHLCV Market Data**: Historical and real-time bars with VWAP calculations
- **Benzinga News Integration**: Premium news feed for catalyst identification
- **Ticker Fundamentals**: Float, market cap, and other metadata
- **Unified Data Pipeline**: Orchestrated data collection with health monitoring

## Architecture

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Polygon       │────│   Unified Data   │────│   MongoDB       │
│   WebSocket     │    │   Pipeline       │    │   Storage       │
│   (LULD)        │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Polygon       │────│   HaltDetector   │────│   Halt Events   │
│   REST API      │    │   Provider       │    │   Collection    │
│   (News/Data)   │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Installation & Setup

### Prerequisites

- Python 3.10+
- Polygon.io API account with premium data access
- MongoDB 6.0+
- Valid API credentials

### Installation

The data module is included in the main project dependencies:

```bash
pip install -r requirements.txt
```

### Configuration

Set your Polygon API key in environment variables:

```bash
export POLYGON_API_KEY="your_polygon_api_key_here"
export MONGO_URI="mongodb://localhost:27017/"
export MONGO_DATABASE="halt_detector"
```

## Usage Examples

### Basic Polygon Client

```python
from backend.core.data import PolygonClient

# Initialize client
client = PolygonClient(api_key="your_api_key")

# Get news for a ticker
news = client.get_news(ticker="AAPL", limit=10)
for article in news:
    print(f"{article.published_utc}: {article.title}")

# Get aggregate bars
bars = client.get_bars(
    ticker="TSLA",
    multiplier=5,
    timespan="minute",
    from_date=datetime.now() - timedelta(days=1),
    to_date=datetime.now()
)
print(f"Retrieved {len(bars)} 5-minute bars")
```

### Real-time Halt Detection

```python
from backend.core.data import HaltDetector, HaltMonitoringContext

async def monitor_halts():
    detector = HaltDetector(api_key="your_api_key")

    # Set up callbacks
    def on_halt(halt_event):
        print(f"Halt detected: {halt_event.ticker} - {halt_event.halt_type}")

    detector.set_halt_callback(on_halt)

    # Monitor specific tickers
    tickers = ["AAPL", "TSLA", "NVDA", "SPY"]

    async with HaltMonitoringContext(api_key, tickers) as detector:
        await detector.start_monitoring(tickers)
        # Monitoring runs indefinitely until interrupted

# Run the monitor
asyncio.run(monitor_halts())
```

### Unified Data Pipeline

```python
from backend.core.data import (
    UnifiedDataPipeline,
    PipelineConfig,
    DataSource
)

async def run_pipeline():
    config = PipelineConfig(
        polygon_api_key="your_api_key",
        tickers_to_monitor=["SPY", "QQQ", "AAPL"],
        enabled_sources=[
            DataSource.LULD_HALT,
            DataSource.MARKET_DATA,
            DataSource.NEWS_FEED,
            DataSource.METADATA
        ]
    )

    pipeline = UnifiedDataPipeline(config)

    # Register callbacks
    def handle_halt(halt_event):
        print(f"🚨 Halt: {halt_event.ticker}")

    pipeline.on_halt(handle_halt)

    # Start pipeline
    success = await pipeline.start()
    if success:
        print("Pipeline running...")
        # Pipeline runs until stopped

asyncio.run(run_pipeline())
```

## API Reference

### PolygonClient

Main client combining REST and WebSocket functionality.

**REST Methods:**
- `get_news(**kwargs)`: Get Benzinga news articles
- `get_bars(**kwargs)`: Get OHLCV aggregate bars
- `get_ticker_info(ticker)`: Get ticker fundamentals

**WebSocket Methods:**
- `connect_websocket()`: Establish WebSocket connection
- `subscribe_luld(tickers)`: Subscribe to LULD messages
- `on_luld(callback)`: Register LULD message handler

### HaltDetector

Real-time halt detection using LULD WebSocket.

**Key Methods:**
- `set_halt_callback(callback)`: Register halt detection handler
- `start_monitoring(tickers)`: Begin monitoring tickers
- `check_weakness_confirmation()`: Analyze weakness signals
- `check_strength_confirmation()`: Analyze strength signals

### MarketDataProvider

OHLCV data provider with caching.

**Key Methods:**
- `get_bars(ticker, timeframe, start, end)`: Get historical bars
- `get_latest_bar(ticker)`: Get most recent bar
- `calculate_vwap(bars_df)`: Calculate VWAP

### NewsFeedProvider

Benzinga news integration.

**Key Methods:**
- `get_news_for_ticker(ticker, hours)`: Get news for ticker
- `get_halt_context_news(ticker, halt_time)`: Get news around halt

### UnifiedDataPipeline

Orchestrated data collection system.

**Key Methods:**
- `start()`: Start all enabled data sources
- `stop()`: Stop all data sources
- `get_historical_bars()`: Get historical market data
- `get_news_for_ticker()`: Get news data
- `get_metrics()`: Get pipeline performance metrics

## Data Structures

### LULDMessage

WebSocket message format for LULD events:

```python
@dataclass
class LULDMessage:
    event_type: str = "LULD"
    ticker: str = ""
    indicator: LULDIndicator = LULDIndicator.NONE
    limit_up_price: float = 0.0
    limit_down_price: float = 0.0
    timestamp: datetime
    exchange: str = ""
    tape: str = ""
```

### AggregateBar

OHLCV bar structure:

```python
@dataclass
class AggregateBar:
    ticker: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    transactions: int
```

### HaltEvent

Comprehensive halt event data:

```python
@dataclass
class HaltEvent:
    ticker: str
    halt_type: HaltType
    halt_status: HaltStatus
    halt_detected_at: datetime
    luld_message: Optional[LULDMessage] = None
    limit_up_price: Optional[float] = None
    limit_down_price: Optional[float] = None
    weakness_confirmed: bool = False
    strength_confirmed: bool = False
    halt_count_today: int = 1
    exchange: Optional[str] = None
```

## Error Handling

All modules implement comprehensive error handling:

- **API Rate Limits**: Automatic retry with exponential backoff
- **Network Issues**: Connection recovery and reconnection logic
- **Data Validation**: Input validation and error reporting
- **Database Errors**: Graceful degradation and logging

## Performance Considerations

### Optimization Features

- **Caching**: Recent data cached in memory to reduce API calls
- **Async I/O**: Non-blocking WebSocket and HTTP operations
- **Connection Pooling**: Reused HTTP connections
- **Batch Operations**: Bulk data operations where possible

### Rate Limits

- **WebSocket**: Unlimited LULD messages (real-time feed)
- **REST API**: 5 requests/second (Benzinga/News), 2 requests/minute (Aggregates - unlimited for premium)

### Memory Management

- **Cache Limits**: Configurable cache sizes per provider
- **Data Retention**: Automatic cleanup of old cache entries
- **Connection Cleanup**: Proper resource cleanup on shutdown

## Monitoring & Health Checks

### Built-in Monitoring

- **Pipeline Metrics**: Messages processed, errors, uptime
- **Data Source Health**: Individual provider status checks
- **Performance Tracking**: Latency and throughput monitoring

### Health Check Example

```python
pipeline = UnifiedDataPipeline(config)
status = pipeline.get_status()

print(f"Pipeline status: {status['status']}")
print(f"Uptime: {status['uptime_seconds']:.1f} seconds")
print(f"Halts detected: {status['metrics']['halts_detected']}")

# Data source health
for source, health in status['data_sources_status'].items():
    print(f"{source}: {health}")
```

## Best Practices

### Production Deployment

1. **Environment Variables**: Never hardcode API keys
2. **Connection Pooling**: Configure MongoDB connection pooling
3. **Monitoring**: Set up alerts for pipeline failures
4. **Backup**: Regular database backups and model snapshots

### Performance Tuning

1. **Cache Sizing**: Adjust based on ticker count and timeframes
2. **WebSocket Subscriptions**: Subscribe only to needed tickers
3. **Batch Operations**: Use bulk inserts for database operations
4. **Async Patterns**: Leverage asyncio for concurrent operations

### Error Recovery

1. **Graceful Degradation**: Continue operation if non-critical sources fail
2. **Auto-Reconnection**: Automatic WebSocket reconnection
3. **Data Consistency**: Database transactions for critical operations
4. **Audit Logging**: Complete audit trail of all operations

## Troubleshooting

### Common Issues

**WebSocket Connection Failed:**
- Verify API key permissions
- Check firewall settings
- Confirm Polygon account status

**No Halt Data:**
- Verify ticker symbols are active
- Check subscription tier includes LULD data
- Confirm market hours

**Slow API Responses:**
- Check rate limits
- Verify network connectivity
- Consider adding retry logic

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Set Polygon client debug logging
import http.client
http.client.HTTPConnection.debuglevel = 1
```

## API Documentation Links

- **LULD WebSocket**: https://massive.com/docs/websocket/stocks/luld
- **Benzinga News REST**: https://massive.com/docs/rest/partners/benzinga/news
- **Aggregates REST**: https://massive.com/docs/rest/stocks/aggregates/custom-bars
- **Polygon API Reference**: https://polygon.io/docs

## Support

For issues with Polygon API integration:
1. Check official Polygon documentation
2. Verify API key and account permissions
3. Review error logs and status codes
4. Contact Polygon support if needed

---

**Last Updated**: January 2024
**Version**: 1.0.0
**Compatibility**: Polygon API v3+