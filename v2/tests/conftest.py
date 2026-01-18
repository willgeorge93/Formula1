"""Pytest fixtures for F1 Predictor tests."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_race_data() -> pd.DataFrame:
    """Sample race data for testing."""
    return pd.DataFrame({
        "season": [2020, 2020, 2020, 2020, 2020, 2020],
        "round": [1, 1, 1, 2, 2, 2],
        "race_name": ["australian", "australian", "australian", "bahrain", "bahrain", "bahrain"],
        "name": ["Lewis Hamilton", "Max Verstappen", "Valtteri Bottas"] * 2,
        "constructor": ["mercedes", "red_bull", "mercedes"] * 2,
        "grid": [1, 2, 3, 1, 2, 3],
        "qual_position": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "q_best": [80.5, 80.8, 81.0, 79.5, 79.8, 80.0],
        "q_worst": [81.5, 81.8, 82.0, 80.5, 80.8, 81.0],
        "q_mean": [81.0, 81.3, 81.5, 80.0, 80.3, 80.5],
        "ageDuringRace": [12775, 8200, 11000, 12780, 8205, 11005],
        "locality": ["Melbourne", "Melbourne", "Melbourne", "Sakhir", "Sakhir", "Sakhir"],
        "country": ["Australia", "Australia", "Australia", "Bahrain", "Bahrain", "Bahrain"],
        "type": ["Road", "Road", "Road", "Road", "Road", "Road"],
        "direction": ["Clockwise", "Clockwise", "Clockwise", "Clockwise", "Clockwise", "Clockwise"],
        "length": [5.303, 5.303, 5.303, 5.412, 5.412, 5.412],
        "weather": ["sunny warm", "sunny warm", "sunny warm", "clear hot", "clear hot", "clear hot"],
        "finish_position": [1, 2, 3, 1, 2, 3],
        "points": [25.0, 18.0, 15.0, 25.0, 18.0, 15.0],
        "filled_splits": [0.0, 5.5, 12.3, 0.0, 4.2, 9.8],
        "status": ["Finished", "Finished", "Finished", "Finished", "Finished", "Finished"],
    })


@pytest.fixture
def sample_qualifying_times() -> dict:
    """Sample qualifying time data."""
    return {
        "valid_time": "1:23.456",
        "valid_time_2": "1:05.123",
        "invalid_time": "invalid",
        "none_time": None,
    }


@pytest.fixture
def sample_weather_text() -> dict:
    """Sample weather text data."""
    return {
        "clean": "sunny warm 25°c",
        "dirty": "Weather: Sunny, warm 25-30°c<br/>with some clouds",
        "empty": "",
        "none": None,
    }


@pytest.fixture
def sample_predictions() -> pd.DataFrame:
    """Sample predictions for testing postprocessing."""
    return pd.DataFrame({
        "season": [2020] * 6,
        "round": [1, 1, 1, 2, 2, 2],
        "race_name": ["australian"] * 3 + ["bahrain"] * 3,
        "name": ["Lewis Hamilton", "Max Verstappen", "Valtteri Bottas"] * 2,
        "constructor": ["mercedes", "red_bull", "mercedes"] * 2,
        "pred": [0.0, 5.5, 12.3, 0.0, 4.2, 9.8],
        "finish_position": [1, 2, 3, 1, 2, 3],
    })


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Temporary data directory for tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "raw").mkdir()
    (data_dir / "processed").mkdir()
    (data_dir / "models").mkdir()
    return data_dir
