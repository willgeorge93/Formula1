"""Evaluation metrics for F1 predictions."""

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    spearman: float
    pearson: float
    r2: float
    mse: float
    rmse: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "spearman": self.spearman,
            "pearson": self.pearson,
            "r2": self.r2,
            "mse": self.mse,
            "rmse": self.rmse,
        }

    def __str__(self) -> str:
        """Format as string."""
        return (
            f"Spearman: {self.spearman:.4f}, "
            f"Pearson: {self.pearson:.4f}, "
            f"R²: {self.r2:.4f}, "
            f"RMSE: {self.rmse:.3f}"
        )


def calculate_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> EvaluationMetrics:
    """
    Calculate all evaluation metrics.

    From notebook evaluation cells.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        EvaluationMetrics object with all metrics
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return EvaluationMetrics(
            spearman=0.0, pearson=0.0, r2=0.0, mse=0.0, rmse=0.0
        )

    # Calculate correlations
    spearman_corr, _ = stats.spearmanr(y_pred, y_true)
    pearson_corr, _ = stats.pearsonr(y_pred, y_true)

    # Calculate regression metrics
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return EvaluationMetrics(
        spearman=round(float(spearman_corr), 4),
        pearson=round(float(pearson_corr), 4),
        r2=round(float(r2), 4),
        mse=round(float(mse), 2),
        rmse=round(float(rmse), 3),
    )


def calculate_position_tolerance(
    true_positions: Union[pd.Series, np.ndarray],
    pred_positions: Union[pd.Series, np.ndarray],
    tolerance: int = 1,
) -> float:
    """
    Calculate percentage of predictions within position tolerance.

    From notebook: match_+/-1, match_+/-2, match_+/-3 calculations

    Args:
        true_positions: Actual race positions
        pred_positions: Predicted race positions
        tolerance: Maximum position difference allowed

    Returns:
        Percentage of predictions within tolerance (0-1)
    """
    true_positions = np.array(true_positions)
    pred_positions = np.array(pred_positions)

    diff = np.abs(true_positions - pred_positions)
    within_tolerance = np.sum(diff <= tolerance)

    return round(within_tolerance / len(true_positions), 4)


def calculate_exact_match_rate(
    true_positions: Union[pd.Series, np.ndarray],
    pred_positions: Union[pd.Series, np.ndarray],
) -> float:
    """
    Calculate percentage of exact position matches.

    Args:
        true_positions: Actual race positions
        pred_positions: Predicted race positions

    Returns:
        Percentage of exact matches (0-1)
    """
    true_positions = np.array(true_positions)
    pred_positions = np.array(pred_positions)

    matches = np.sum(true_positions == pred_positions)
    return round(matches / len(true_positions), 4)


def calculate_all_tolerances(
    true_positions: Union[pd.Series, np.ndarray],
    pred_positions: Union[pd.Series, np.ndarray],
) -> dict[str, float]:
    """
    Calculate position accuracy at multiple tolerance levels.

    Args:
        true_positions: Actual race positions
        pred_positions: Predicted race positions

    Returns:
        Dictionary with tolerance levels and accuracies
    """
    return {
        "exact": calculate_exact_match_rate(true_positions, pred_positions),
        "within_1": calculate_position_tolerance(true_positions, pred_positions, 1),
        "within_2": calculate_position_tolerance(true_positions, pred_positions, 2),
        "within_3": calculate_position_tolerance(true_positions, pred_positions, 3),
    }


def calculate_standings_correlation(
    pred_standings: pd.DataFrame,
    true_standings: pd.DataFrame,
    points_column: str = "total_points",
) -> EvaluationMetrics:
    """
    Calculate correlation metrics for championship standings.

    Args:
        pred_standings: Predicted standings DataFrame
        true_standings: Actual standings DataFrame
        points_column: Column name for points

    Returns:
        EvaluationMetrics for standings comparison
    """
    # Ensure same order
    pred_points = pred_standings[points_column].values
    true_points = true_standings[points_column].values

    return calculate_metrics(true_points, pred_points)
