"""Tests for ML model."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from f1_predictor.models.xgboost_model import F1XGBoostModel
from f1_predictor.models.transformers import create_column_transformer
from f1_predictor.core.exceptions import ModelNotFittedError


class TestColumnTransformer:
    """Tests for column transformer creation."""

    def test_transformer_creation(self):
        """Test that transformer is created successfully."""
        transformer = create_column_transformer()
        assert transformer is not None

    def test_transformer_has_all_features(self):
        """Test that transformer includes all expected transformers."""
        transformer = create_column_transformer()
        transformer_names = [name for name, _, _ in transformer.transformers]

        expected = [
            "weather", "direction", "country", "locality", "type",
            "season", "round", "qual_position", "grid", "race_name",
            "q_mean", "q_best", "q_worst", "length", "ageDuringRace"
        ]

        for name in expected:
            assert name in transformer_names


class TestF1XGBoostModel:
    """Tests for F1XGBoostModel."""

    def test_model_initialization(self):
        """Test model initialization with default parameters."""
        model = F1XGBoostModel()

        assert model.gamma == 0.1
        assert model.learning_rate == 0.2
        assert model.max_depth == 6
        assert model.n_estimators == 150
        assert model.reg_lambda == 0.2
        assert model.subsample == 1.0
        assert not model.is_fitted

    def test_model_custom_params(self):
        """Test model initialization with custom parameters."""
        model = F1XGBoostModel(
            gamma=0.5,
            learning_rate=0.1,
            max_depth=8,
        )

        assert model.gamma == 0.5
        assert model.learning_rate == 0.1
        assert model.max_depth == 8

    def test_predict_before_fit_raises(self):
        """Test that predict raises error before fitting."""
        model = F1XGBoostModel()

        with pytest.raises(ModelNotFittedError):
            model.predict(pd.DataFrame())

    def test_score_before_fit_raises(self):
        """Test that score raises error before fitting."""
        model = F1XGBoostModel()

        with pytest.raises(ModelNotFittedError):
            model.score(pd.DataFrame(), pd.Series())

    def test_save_before_fit_raises(self, tmp_path):
        """Test that save raises error before fitting."""
        model = F1XGBoostModel()

        with pytest.raises(ModelNotFittedError):
            model.save(tmp_path / "model.pkl")

    def test_get_params(self):
        """Test getting model parameters."""
        model = F1XGBoostModel()
        params = model.get_params()

        assert "gamma" in params
        assert "learning_rate" in params
        assert "max_depth" in params
        assert params["gamma"] == 0.1

    @pytest.mark.skipif(
        True, reason="Requires training data"
    )
    def test_fit_and_predict(self, sample_race_data):
        """Test model fitting and prediction."""
        model = F1XGBoostModel(n_estimators=10)

        # Prepare data
        X = sample_race_data.drop(
            columns=["filled_splits", "finish_position", "points", "status", "name", "constructor"]
        )
        y = sample_race_data["filled_splits"]

        # Fit
        model.fit(X, y)
        assert model.is_fitted

        # Predict
        predictions = model.predict(X)
        assert len(predictions) == len(y)
