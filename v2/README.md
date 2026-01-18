# F1 Predictor v2

A production-ready Python package for predicting Formula 1 race results, driver standings, and constructor standings using machine learning.

## Overview

This package uses an XGBoost regression model trained on historical F1 data (2014-2019) to predict race outcomes. The model achieves:

- **Driver Standings**: 0.98 Pearson correlation, 0.96 R² score
- **Constructor Standings**: 0.99 Pearson correlation, 0.98 R² score
- **Position Accuracy**: 52% within ±2 positions

## Installation

```bash
# Clone the repository
git clone https://github.com/willgeorge93/Formula1.git
cd Formula1/v2

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

### 1. Initialize the project

```bash
f1-predictor init
```

### 2. Collect data

```bash
f1-predictor collect --start-year 2014 --end-year 2023
```

### 3. Train the model

```bash
f1-predictor train --test-season 2020
```

### 4. Generate predictions

```bash
f1-predictor predict 2020
```

### 5. Start the API server

```bash
f1-predictor serve --port 8000
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `f1-predictor init` | Initialize project directories |
| `f1-predictor collect` | Collect F1 data from APIs |
| `f1-predictor train` | Train the prediction model |
| `f1-predictor predict <season>` | Generate predictions |
| `f1-predictor evaluate` | Evaluate model performance |
| `f1-predictor models` | List registered models |
| `f1-predictor serve` | Start the API server |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/standings/{season}` | GET | Get predicted standings |
| `/api/v1/models` | GET | List available models |
| `/api/v1/model` | GET | Get active model info |

## Project Structure

```
v2/
├── src/f1_predictor/
│   ├── core/           # Configuration, constants, exceptions
│   ├── data/           # Data collection and storage
│   ├── features/       # Feature engineering
│   ├── models/         # ML models and training
│   ├── postprocessing/ # Prediction conversion
│   ├── evaluation/     # Metrics and reports
│   ├── cli/            # Command-line interface
│   └── api/            # FastAPI application
├── tests/              # Unit tests
├── config/             # Configuration files
└── data/               # Local data storage
```

## Configuration

Configuration is managed via environment variables or `.env` file:

```bash
# Copy example config
cp .env.example .env

# Edit as needed
F1_DEBUG=false
F1_LOG_LEVEL=INFO
F1_MODEL__TEST_SEASON=2020
F1_STORAGE__BACKEND=sqlite
```

## Data Sources

- **Ergast API**: Race schedules, results, qualifying, drivers, circuits
- **Wikipedia**: Weather conditions, race distance
- **F1-Fansite**: Detailed weather data

## Model Architecture

The prediction pipeline uses:

1. **Feature Engineering**:
   - CountVectorizer for weather text
   - OneHotEncoder for 9 categorical features
   - StandardScaler for 5 numerical features

2. **XGBoost Regressor** with tuned hyperparameters:
   - gamma: 0.1
   - learning_rate: 0.2
   - max_depth: 6
   - n_estimators: 150
   - reg_lambda: 0.2

3. **Post-processing**:
   - Convert time gaps to race positions
   - Apply F1 points system
   - Aggregate to championship standings

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
ruff check src/

# Run type checking
mypy src/
```

## License

MIT License

## Author

William George
