"""Model training pipeline."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from f1_predictor.core.config import get_settings
from f1_predictor.core.logging import get_logger
from f1_predictor.data.storage import get_storage
from f1_predictor.evaluation.metrics import EvaluationMetrics, calculate_metrics
from f1_predictor.features.pipeline import FeaturePipeline
from f1_predictor.models.registry import ModelRegistry
from f1_predictor.models.xgboost_model import F1XGBoostModel
from f1_predictor.postprocessing.standings import calculate_standings_from_predictions

logger = get_logger(__name__)


class ModelTrainer:
    """
    Orchestrates model training, evaluation, and saving.

    Handles the full training pipeline from data loading through
    model persistence.
    """

    def __init__(
        self,
        model: Optional[F1XGBoostModel] = None,
        registry: Optional[ModelRegistry] = None,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            model: Model instance to train (default: new F1XGBoostModel)
            registry: Model registry for versioning (default: new registry)
        """
        self.model = model or F1XGBoostModel()
        self.registry = registry or ModelRegistry()
        self.settings = get_settings()

    def train(
        self,
        df: Optional[pd.DataFrame] = None,
        test_season: Optional[int] = None,
        save_model: bool = True,
    ) -> dict:
        """
        Train the model and evaluate performance.

        Args:
            df: Training data (default: load from storage)
            test_season: Season for testing (default: from config)
            save_model: Whether to save the trained model

        Returns:
            Dictionary with training results and metrics
        """
        test_season = test_season or self.settings.model.test_season

        # Load data if not provided
        if df is None:
            storage = get_storage()
            df = storage.load_dataframe("main_df")

        logger.info(f"Training model with test season {test_season}")

        # Prepare data for training
        pipeline = FeaturePipeline()
        X_train, X_test, y_train, y_test = pipeline.prepare_for_training(
            df, test_season=test_season
        )

        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")

        # Train the model
        self.model.fit(X_train, y_train)

        # Calculate metrics
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)

        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        train_metrics = calculate_metrics(y_train, y_pred_train)
        test_metrics = calculate_metrics(y_test, y_pred_test)

        # Evaluate standings predictions
        test_df = df[df["season"] == test_season].copy()
        standings_metrics = self._evaluate_standings(test_df, X_test, y_pred_test)

        results = {
            "train_r2": train_score,
            "test_r2": test_score,
            "train_metrics": train_metrics.to_dict(),
            "test_metrics": test_metrics.to_dict(),
            "standings_metrics": standings_metrics,
            "test_season": test_season,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

        logger.info(f"Training R²: {train_score:.4f}")
        logger.info(f"Test R²: {test_score:.4f}")

        # Save model
        if save_model:
            version = self.registry.save_model(
                self.model,
                metrics=results,
                set_active=True,
            )
            results["model_version"] = version

        return results

    def _evaluate_standings(
        self,
        test_df: pd.DataFrame,
        X_test: pd.DataFrame,
        y_pred: pd.Series,
    ) -> dict:
        """Evaluate standings predictions."""
        from f1_predictor.postprocessing.positions import predictions_to_positions
        from f1_predictor.postprocessing.points import position_to_points

        # Add predictions to test data
        eval_df = test_df.copy()
        eval_df = eval_df.reset_index(drop=True)

        # Match predictions to eval_df
        if len(y_pred) == len(eval_df):
            eval_df["pred"] = y_pred
        else:
            # Handle index mismatch
            eval_df["pred"] = 0.0

        # Convert predictions to positions
        eval_df["pred_position"] = predictions_to_positions(
            eval_df, pred_column="pred"
        )

        # Calculate predicted points
        eval_df["pred_points"] = eval_df["pred_position"].apply(position_to_points)

        # Calculate standings
        driver_standings, constructor_standings = calculate_standings_from_predictions(
            eval_df
        )

        # Compare with actual standings
        # (In a full implementation, we'd load actual standings and compare)
        metrics = {
            "driver_standings": driver_standings.to_dict("records")[:10],
            "constructor_standings": constructor_standings.to_dict("records")[:5],
        }

        return metrics

    def cross_validate(
        self,
        df: pd.DataFrame,
        seasons: Optional[list[int]] = None,
    ) -> dict:
        """
        Perform walk-forward cross-validation across seasons.

        Args:
            df: Full dataset
            seasons: Seasons to use as test sets

        Returns:
            Cross-validation results
        """
        if seasons is None:
            seasons = [2018, 2019, 2020]

        results = []

        for test_season in seasons:
            logger.info(f"Cross-validating with test season {test_season}")

            # Create fresh model for each fold
            model = F1XGBoostModel()

            # Prepare data
            pipeline = FeaturePipeline()
            X_train, X_test, y_train, y_test = pipeline.prepare_for_training(
                df, test_season=test_season
            )

            # Train and evaluate
            model.fit(X_train, y_train)
            test_score = model.score(X_test, y_test)

            y_pred = model.predict(X_test)
            metrics = calculate_metrics(y_test, y_pred)

            results.append(
                {
                    "test_season": test_season,
                    "test_r2": test_score,
                    **metrics.to_dict(),
                }
            )

        # Calculate aggregate metrics
        mean_r2 = sum(r["test_r2"] for r in results) / len(results)
        mean_rmse = sum(r["rmse"] for r in results) / len(results)

        return {
            "folds": results,
            "mean_r2": mean_r2,
            "mean_rmse": mean_rmse,
        }
