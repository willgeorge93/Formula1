"""XGBoost model implementation with tuned hyperparameters."""

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from f1_predictor.core.config import get_settings
from f1_predictor.core.exceptions import ModelNotFittedError, ModelNotFoundError
from f1_predictor.core.logging import get_logger
from f1_predictor.models.transformers import create_column_transformer, get_feature_names

logger = get_logger(__name__)


class F1XGBoostModel:
    """
    XGBoost model for F1 race prediction.

    Hyperparameters from notebook tuning (96,040 combinations tested):
    - gamma: 0.1
    - learning_rate: 0.2
    - max_depth: 6
    - n_estimators: 150
    - reg_lambda: 0.2
    - subsample: 1

    Target: filled_splits (time gap to winner in seconds)

    Performance metrics (from notebook):
    - Driver Standings: Pearson 0.9807, R² 0.9601, RMSE 17.342
    - Constructor Standings: Pearson 0.9941, R² 0.9823, RMSE 22.036
    """

    def __init__(
        self,
        gamma: Optional[float] = None,
        learning_rate: Optional[float] = None,
        max_depth: Optional[int] = None,
        n_estimators: Optional[int] = None,
        reg_lambda: Optional[float] = None,
        subsample: Optional[float] = None,
    ) -> None:
        """
        Initialize the XGBoost model.

        Args:
            gamma: Minimum loss reduction for split (default from config)
            learning_rate: Boosting learning rate (default from config)
            max_depth: Maximum tree depth (default from config)
            n_estimators: Number of boosting rounds (default from config)
            reg_lambda: L2 regularization (default from config)
            subsample: Subsample ratio (default from config)
        """
        self.settings = get_settings().model

        # Use provided params or defaults from config (tuned values)
        self.gamma = gamma if gamma is not None else self.settings.xgb_gamma
        self.learning_rate = (
            learning_rate if learning_rate is not None else self.settings.xgb_learning_rate
        )
        self.max_depth = max_depth if max_depth is not None else self.settings.xgb_max_depth
        self.n_estimators = (
            n_estimators if n_estimators is not None else self.settings.xgb_n_estimators
        )
        self.reg_lambda = (
            reg_lambda if reg_lambda is not None else self.settings.xgb_reg_lambda
        )
        self.subsample = subsample if subsample is not None else self.settings.xgb_subsample

        self._pipeline: Optional[Pipeline] = None
        self._is_fitted: bool = False

    def _create_pipeline(self) -> Pipeline:
        """Create sklearn pipeline with column transformer and XGBoost."""
        xgb = XGBRegressor(
            gamma=self.gamma,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            reg_lambda=self.reg_lambda,
            subsample=self.subsample,
            n_jobs=-1,
            random_state=self.settings.random_state,
            verbosity=0,
        )

        return Pipeline(
            [
                ("preprocessor", create_column_transformer()),
                ("regressor", xgb),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "F1XGBoostModel":
        """
        Train the model on provided data.

        Args:
            X: Feature DataFrame
            y: Target Series (filled_splits)

        Returns:
            Self for method chaining
        """
        logger.info(f"Training XGBoost model on {len(X)} samples")

        self._pipeline = self._create_pipeline()
        self._pipeline.fit(X, y)
        self._is_fitted = True

        train_score = self._pipeline.score(X, y)
        logger.info(f"Training R² score: {train_score:.4f}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions (time gaps).

        Args:
            X: Feature DataFrame

        Returns:
            Array of predicted time gaps

        Raises:
            ModelNotFittedError: If model hasn't been fitted
        """
        if not self._is_fitted:
            raise ModelNotFittedError("Model must be fitted before prediction")
        return self._pipeline.predict(X)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Return R² score on provided data.

        Args:
            X: Feature DataFrame
            y: Target Series

        Returns:
            R² score

        Raises:
            ModelNotFittedError: If model hasn't been fitted
        """
        if not self._is_fitted:
            raise ModelNotFittedError("Model must be fitted before scoring")
        return self._pipeline.score(X, y)

    def save(self, path: Path) -> None:
        """
        Save model to disk.

        Args:
            path: File path for saving

        Raises:
            ModelNotFittedError: If model hasn't been fitted
        """
        if not self._is_fitted:
            raise ModelNotFittedError("Model must be fitted before saving")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "F1XGBoostModel":
        """
        Load model from disk.

        Args:
            path: File path to load from

        Returns:
            Loaded model instance

        Raises:
            ModelNotFoundError: If model file doesn't exist
        """
        path = Path(path)
        if not path.exists():
            raise ModelNotFoundError(f"Model file not found: {path}")

        instance = cls()
        instance._pipeline = joblib.load(path)
        instance._is_fitted = True
        logger.info(f"Model loaded from {path}")

        return instance

    def get_feature_importances(self) -> pd.DataFrame:
        """
        Get feature importances from the XGBoost model.

        Returns:
            DataFrame with feature names and importance scores

        Raises:
            ModelNotFittedError: If model hasn't been fitted
        """
        if not self._is_fitted:
            raise ModelNotFittedError("Model must be fitted first")

        xgb_model = self._pipeline.named_steps["regressor"]
        preprocessor = self._pipeline.named_steps["preprocessor"]

        # Get feature names from transformers
        feature_names = get_feature_names(preprocessor)
        importances = xgb_model.feature_importances_

        return pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False)

    def get_params(self) -> dict[str, Any]:
        """Get model hyperparameters."""
        return {
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators,
            "reg_lambda": self.reg_lambda,
            "subsample": self.subsample,
        }

    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._is_fitted
