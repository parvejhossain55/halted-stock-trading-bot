# Deployment Guide

## Overview

This guide covers deploying the Halt Detector Trading System for both development and production environments. The system consists of a Python backend with FastAPI, MongoDB database, and Docker containerization.

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows
- **Python**: 3.10+
- **MongoDB**: 6.0+ (local or cloud)
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: 10GB+ for logs and data

### Required Accounts & API Keys
1. **Polygon.io Account** - Real-time market data and halt events
2. **Cobra Trading Account** - DAS CMD API for order execution (optional for testing)

## Quick Start (Development)

### 1. Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd halt-detector-zed

# Use Makefile for easy setup
make dev-setup
```

### 2. Configure API Keys
Edit `.env` file:
```bash
POLYGON_API_KEY=your_polygon_api_key_here
# Optional: DAS credentials for live trading
```

### 3. Start the System
```bash
# Start trading engine
make run

# In another terminal, start API server
make run-api
```

## Manual Setup (Alternative)

### 1. Python Environment
```bash
# Create virtual environment
python3 -m venv myenv
source myenv/bin/activate  # Linux/Mac
# myenv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. MongoDB Setup
```bash
# Using Docker (recommended)
docker run -d --name mongodb -p 27017:27017 mongo:6.0

# Or install MongoDB locally and start service
sudo systemctl start mongod
```

### 3. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys
nano .env
```

## Running the System

### Development Mode
```bash
# Terminal 1: Start trading engine
source myenv/bin/activate
python3 scripts/run_trading_engine.py

# Terminal 2: Start API server
source myenv/bin/activate
cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode (Docker)
```bash
# Build and start all services
docker-compose up -d

# Check logs
docker-compose logs -f
```

### Production Mode (Manual)
```bash
# Install as system service
sudo cp scripts/halt-detector.service /etc/systemd/system/
sudo systemctl enable halt-detector
sudo systemctl start halt-detector

# Start API server with Gunicorn
gunicorn backend.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Configuration

### Environment Variables
```bash
# Database
MONGO_HOST=localhost
MONGO_PORT=27017
MONGO_DB=halt_detector

# APIs
POLYGON_API_KEY=your_key_here

# Trading
PAPER_TRADING_MODE=True  # Set to False for live trading

# Logging
LOG_LEVEL=INFO
```

### Trading Parameters
Configure in `backend/config/settings.py`:
- Risk limits and position sizing
- Entry/exit rules
- API rate limits

## Monitoring & Health Checks

### System Health
```bash
# Makefile health check
make health

# API health endpoint
curl http://localhost:8000/health

# System status
curl http://localhost:8000/api/status
```

### Logs
```bash
# View trading logs
tail -f backend/logs/trading.log

# View error logs
tail -f backend/logs/error.log

# View debug logs (if DEBUG=true)
tail -f backend/logs/debug.log
```

### Database Status
```bash
# Check MongoDB connection
mongosh --eval "db.stats()" halt_detector

# List collections
mongosh --eval "db.listCollections()" halt_detector
```

## Troubleshooting

### Common Issues

**1. MongoDB Connection Failed**
```bash
# Check if MongoDB is running
sudo systemctl status mongod

# Or for Docker
docker ps | grep mongo

# Reset database
make db-reset
```

**2. Polygon API Authentication Failed**
```bash
# Check API key in .env
cat .env | grep POLYGON

# Verify key format (should start with polygon_ or similar)
```

**3. Trading Engine Won't Start**
```bash
# Check Python environment
source myenv/bin/activate && python3 --version

# Check dependencies
pip list | grep -E "(fastapi|mongo|polygon)"

# Check logs for errors
tail -20 backend/logs/error.log
```

**4. API Server Not Responding**
```bash
# Check if port 8000 is in use
netstat -tlnp | grep 8000

# Kill conflicting process or change port
```

### Debug Mode
```bash
# Enable debug logging
echo "DEBUG=True" >> .env

# Restart services
make restart
```

## Backup & Recovery

### Database Backup
```bash
# Using Makefile
make backup-db

# Manual backup
mongodump --uri="mongodb://localhost:27017/halt_detector" --out=backups/$(date +%Y%m%d_%H%M%S)
```

### Database Restore
```bash
# Using Makefile
make restore-db BACKUP_DIR=backups/20241110_030000

# Manual restore
mongorestore --uri="mongodb://localhost:27017/halt_detector" backups/20241110_030000
```

## Security Considerations

### API Keys
- Store securely in `.env` file
- Never commit to version control
- Rotate keys regularly
- Use separate keys for dev/prod

### Network Security
- Run behind reverse proxy in production
- Enable SSL/TLS
- Implement rate limiting
- Use firewall rules

### Database Security
- Enable MongoDB authentication
- Use strong passwords
- Regular security updates
- Backup encryption

## Scaling Considerations

### Performance Optimization
- Use MongoDB indexes for query performance
- Implement connection pooling
- Cache frequently accessed data
- Monitor memory usage

### High Availability
- MongoDB replica sets for production
- Load balancing for API servers
- Redundant trading engines
- Automated failover procedures

## Support & Maintenance

### Daily Tasks
- Monitor system health
- Review trading logs
- Check API rate limits
- Verify database backups

### Weekly Tasks
- Update dependencies
- Review error logs
- Performance optimization
- Security updates

### Emergency Procedures
1. **Stop Trading**: `make stop` or API kill switch
2. **Check Logs**: Review error logs for issues
3. **Database Backup**: Create emergency backup
4. **Restart Services**: `make restart`
5. **Contact Support**: If issues persist

---

**Version:** 1.0
**Status:** Active Development
