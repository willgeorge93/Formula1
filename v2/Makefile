.PHONY: install install-dev test lint typecheck format clean collect train serve help

# Default target
help:
	@echo "F1 Predictor - Available commands:"
	@echo ""
	@echo "  make install      Install production dependencies"
	@echo "  make install-dev  Install development dependencies"
	@echo "  make test         Run tests with coverage"
	@echo "  make lint         Run linting checks"
	@echo "  make typecheck    Run type checking"
	@echo "  make format       Format code with ruff"
	@echo "  make clean        Remove build artifacts"
	@echo "  make collect      Collect F1 data"
	@echo "  make train        Train the model"
	@echo "  make serve        Start the API server"
	@echo ""

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Testing
test:
	pytest tests/ -v --cov=f1_predictor --cov-report=term-missing

test-fast:
	pytest tests/ -v -x --tb=short

# Code quality
lint:
	ruff check src/ tests/

typecheck:
	mypy src/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Data and training
collect:
	f1-predictor collect --start-year 2014

train:
	f1-predictor train

# Server
serve:
	f1-predictor serve --reload

serve-prod:
	f1-predictor serve --host 0.0.0.0 --port 8000

# Full pipeline
pipeline: collect train
	@echo "Pipeline complete!"

# Development setup
dev-setup: install-dev
	f1-predictor init
	@echo "Development environment ready!"
