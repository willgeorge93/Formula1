# F1 Predictor v2.0.0 Release Notes

## Overview

Complete rewrite of the Formula 1 race prediction system, converting the original Jupyter notebook-based approach into a production-ready Python package with CLI, REST API, and comprehensive testing.

---

## New Features

### Modular Package Architecture
- Clean separation of concerns across 8 submodules
- Type-safe configuration with Pydantic Settings
- Custom exception hierarchy for better error handling
- Structured logging with configurable levels

### Command-Line Interface
- `f1-predictor collect` - Fetch F1 data from Ergast API and web sources
- `f1-predictor train` - Train the XGBoost prediction model
- `f1-predictor predict <season>` - Generate season predictions
- `f1-predictor evaluate` - Evaluate model performance
- `f1-predictor models` - List registered model versions
- `f1-predictor serve` - Start the REST API server
- `f1-predictor init` - Initialize project data directories
- `f1-predictor version` - Display version information

### REST API
- FastAPI-based with automatic OpenAPI documentation
- `/health` - Health check endpoint
- `/api/v1/predict/race` - Single race predictions
- `/api/v1/predict/season/{season}` - Full season predictions
- `/api/v1/standings/{season}` - Championship standings
- `/api/v1/models` - Model registry information

### Data Collection
- Ergast API client with rate limiting and retry logic
- Wikipedia scraper for weather and circuit data
- F1-Fansite scraper for additional weather information
- Pydantic schemas for data validation
- Multiple storage backends (SQLite, CSV, Parquet)

### Feature Engineering Pipeline
- Qualifying time parsing (M:SS.mmm format)
- Weather text normalization and cleaning
- Split time calculations with gap handling
- Driver age computation at race time
- Configurable feature selection

### Machine Learning
- XGBoost regressor with tuned hyperparameters:
  - gamma: 0.1
  - learning_rate: 0.2
  - max_depth: 6
  - n_estimators: 150
  - reg_lambda: 0.2
  - subsample: 1.0
- sklearn ColumnTransformer with 15 transformations
- Model versioning and registry system
- Reproducible training with configurable random state

### Post-processing
- Prediction to position ranking within races
- F1 points system application (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)
- Driver and constructor championship standings calculation

### Evaluation Metrics
- Spearman rank correlation
- Pearson correlation coefficient
- R² score
- RMSE (Root Mean Square Error)
- Position tolerance analysis (±1, ±2, ±3)

---

## Technical Improvements

### Code Quality
- Type hints throughout the codebase
- Pydantic models for runtime validation
- Ruff for linting and formatting
- Mypy for static type checking
- pytest with coverage reporting

### Testing
- 35 unit tests covering core functionality
- pytest fixtures for common test data
- Mocked external API calls
- Coverage reporting with term-missing output

### Configuration
- YAML-based default configuration
- Environment variable overrides
- Pydantic Settings for type-safe config access
- Separate dev/prod configurations

### Documentation
- Comprehensive README with usage examples
- Makefile for common development tasks
- Example environment file template
- Inline docstrings for all public functions

---

## Dependencies

### Production
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- httpx >= 0.25.0
- beautifulsoup4 >= 4.12.0
- pydantic >= 2.0.0
- pydantic-settings >= 2.0.0
- typer >= 0.12.0
- rich >= 13.0.0
- fastapi >= 0.100.0
- uvicorn >= 0.23.0
- joblib >= 1.3.0
- tenacity >= 8.2.0
- PyYAML >= 6.0
- scipy >= 1.10.0

### Development
- pytest >= 7.4.0
- pytest-cov >= 4.1.0
- pytest-asyncio >= 0.21.0
- mypy >= 1.5.0
- ruff >= 0.1.0

---

## Migration from v1

The v2 package is a complete rewrite and does not share code with the original notebooks. To migrate:

1. Install the package: `pip install -e .` (from the v2 directory)
2. Initialize data directories: `f1-predictor init`
3. Collect data: `f1-predictor collect --start-year 2014`
4. Train model: `f1-predictor train`
5. Generate predictions: `f1-predictor predict 2024`

Existing CSV data from the notebooks can be imported via the storage module.

---

## Known Limitations

- Weather data scraping depends on external website availability
- Ergast API rate limiting may slow bulk data collection
- Model requires historical data from 2014+ for optimal performance

---

## Requirements

- Python >= 3.10
- pip >= 21.0

---

## Installation

```bash
cd v2
pip install -e .          # Production install
pip install -e ".[dev]"   # Development install with test dependencies
```

---

## Quick Start

```bash
# Initialize
f1-predictor init

# Collect data (2014-present)
f1-predictor collect

# Train model
f1-predictor train

# Predict 2024 season
f1-predictor predict 2024

# Start API server
f1-predictor serve
```
