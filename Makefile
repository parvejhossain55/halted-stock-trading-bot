# Halt Detector Trading System - Makefile
# ======================================
# Streamlined, robust version with variables, proper virtualenv usage, and fixed backups.

SHELL := /bin/bash

# =========================
# Variables
# =========================
VENV := myenv
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip
DB_URI := mongodb://localhost:27017
DB_NAME := halt_detector
BACKUP_DIR := backups

# =========================
# Phony targets
# =========================
.PHONY: help setup install run run-api test test-unit test-integration \
        db-init db-reset docker-build docker-up docker-down docker-logs \
        lint format clean clean-all env-activate env-deactivate \
        health config-show config-template docs-build \
        backup-db restore-db quick-start dev-setup ci-test \
        start stop restart logs

# =========================
# Help
# =========================
help: ## Show this help message
	@echo "Halt Detector Trading System - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort \
	| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =========================
# Environment Setup
# =========================
setup: ## Set up development environment
	@echo "Setting up virtual environment..."
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	@echo "Setup complete. Activate with 'make env-activate'"

install: ## Install dependencies
	@echo "Installing Python dependencies..."
	$(PIP) install -r requirements.txt

env-activate: ## Show command to activate virtual environment
	@echo "Run: source $(VENV)/bin/activate"

env-deactivate: ## Show command to deactivate virtual environment
	@echo "Run: deactivate"

# =========================
# Running Application
# =========================
run: ## Run trading engine
	@echo "Starting Halt Detector Trading Engine..."
	$(PYTHON) scripts/run_trading_engine.py

run-api: ## Run FastAPI server
	@echo "Starting FastAPI server..."
	cd backend && $(PYTHON) -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# =========================
# Testing
# =========================
test: ## Run all tests
	@echo "Running all tests..."
	$(PYTHON) -m pytest tests/ -v

test-unit: ## Run unit tests only
	@echo "Running unit tests..."
	$(PYTHON) -m pytest tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "Running integration tests..."
	$(PYTHON) -m pytest tests/integration/ -v

# =========================
# Database Operations
# =========================
db-init: ## Initialize DB and create indexes
	@echo "Initializing database..."
	$(PYTHON) -c "from backend.database.models import init_db, create_indexes; \
init_db('$(DB_URI)', '$(DB_NAME)'); create_indexes(); print('DB initialized')"

db-reset: ## Reset database (drop all collections)
	@echo "Resetting database..."
	$(PYTHON) -c "from mongoengine import connect; \
db = connect(db='$(DB_NAME)', host='$(DB_URI)'); \
[db.drop_collection(c) for c in db.list_collection_names()]; \
print('Database reset complete')"

# =========================
# Docker Operations
# =========================
docker-build: ## Build Docker images
	docker-compose build

docker-up: ## Start Docker services
	docker-compose up -d

docker-down: ## Stop Docker services
	docker-compose down

docker-logs: ## Show Docker logs
	docker-compose logs -f

# =========================
# Code Quality
# =========================
lint: ## Run flake8
	$(PYTHON) -m flake8 backend/ scripts/ tests/ --max-line-length=100

format: ## Format code with black
	$(PYTHON) -m black backend/ scripts/ tests/ --line-length=100

# =========================
# Cleanup
# =========================
clean: ## Clean temporary files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

clean-all: clean ## Clean everything including venv and docker
	rm -rf $(VENV)
	docker-compose down -v

# =========================
# Health Check
# =========================
health: ## Check system health
	@echo "Checking API server..."
	@curl -s http://localhost:8000/health || echo "API server not running"
	@pgrep -f "python3 scripts/run_trading_engine.py" > /dev/null && \
		echo "Trading engine running" || echo "Trading engine not running"

# =========================
# Configuration
# =========================
config-show: ## Show environment config
	@echo "Current environment configuration:"
	@cat .env 2>/dev/null || echo "No .env file found"

config-template: ## Copy .env.example to .env
	cp .env.example .env
	@echo "Please edit .env with actual API keys and configuration."

# =========================
# Documentation
# =========================
docs-build: ## Build documentation
	@echo "Documentation is in docs/"

# =========================
# Backup & Restore
# =========================
backup-db: ## Backup MongoDB
	@TIMESTAMP=$$(date +%Y%m%d_%H%M%S); \
	BACKUP_PATH=$(BACKUP_DIR)/$$TIMESTAMP; \
	mkdir -p $$BACKUP_PATH; \
	mongodump --uri="$(DB_URI)/$(DB_NAME)" --out=$$BACKUP_PATH; \
	echo "Backup saved to $$BACKUP_PATH"

restore-db: ## Restore MongoDB (BACKUP_DIR required)
ifndef BACKUP_DIR
	$(error Please specify BACKUP_DIR, e.g., make restore-db BACKUP_DIR=backups/20251110_030000)
endif
	mongorestore --uri="$(DB_URI)/$(DB_NAME)" $(BACKUP_DIR)

# =========================
# Quick Start / Dev Setup
# =========================
quick-start: setup docker-up run ## Setup, start docker, run app
	@echo "Quick start complete!"

dev-setup: setup config-template db-init ## Dev environment setup
	@echo "Development environment ready!"

# =========================
# CI/CD
# =========================
ci-test: install test lint ## Run CI pipeline
	@echo "CI tests passed!"

# =========================
# Aliases
# =========================
start: run
stop: docker-down
restart: docker-down docker-up
logs: docker-logs
