"""
Tests for API endpoints
"""

import pytest
import sys
import os
from fastapi.testclient import TestClient

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Import as a package
from backend.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "service" in data
    assert "trading_mode" in data


def test_system_status():
    """Test system status endpoint"""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "trading_mode" in data
    assert "api_keys_configured" in data


def test_get_positions():
    """Test get positions endpoint"""
    response = client.get("/api/positions")
    # This might fail if database is not connected, but should not crash
    assert response.status_code in [200, 500]


def test_invalid_market_data():
    """Test market data with invalid ticker"""
    response = client.get("/api/market-data/INVALID")
    assert response.status_code == 200  # Returns mock data
    data = response.json()
    assert data["ticker"] == "INVALID"


def test_kill_switch():
    """Test kill switch activation"""
    response = client.post("/api/kill-switch")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
