"""Convert positions to championship points."""

import pandas as pd

from f1_predictor.core.constants import POINTS_SYSTEM


def position_to_points(position: int) -> int:
    """
    Convert finishing position to championship points.

    Uses the current F1 points system:
    1st: 25, 2nd: 18, 3rd: 15, 4th: 12, 5th: 10,
    6th: 8, 7th: 6, 8th: 4, 9th: 2, 10th: 1

    From notebook: points() function

    Args:
        position: Race finishing position

    Returns:
        Championship points awarded
    """
    return POINTS_SYSTEM.get(int(position), 0)


def calculate_race_points(
    df: pd.DataFrame,
    position_column: str = "pred_position",
) -> pd.Series:
    """
    Calculate championship points for each race result.

    Args:
        df: DataFrame with position column
        position_column: Name of the position column

    Returns:
        Series with championship points
    """
    return df[position_column].apply(position_to_points)


def apply_points_system(
    df: pd.DataFrame,
    position_column: str = "position",
    points_system: dict[int, int] = None,
) -> pd.DataFrame:
    """
    Apply a points system to race results.

    Args:
        df: DataFrame with race results
        position_column: Column containing positions
        points_system: Custom points mapping (default: F1 system)

    Returns:
        DataFrame with points column added
    """
    if points_system is None:
        points_system = POINTS_SYSTEM

    df = df.copy()
    df["points"] = df[position_column].apply(lambda p: points_system.get(int(p), 0))

    return df


def get_points_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics for points distribution.

    Args:
        df: DataFrame with points column

    Returns:
        Dictionary with summary statistics
    """
    if "points" not in df.columns:
        return {}

    return {
        "total_points": df["points"].sum(),
        "mean_points": df["points"].mean(),
        "median_points": df["points"].median(),
        "max_points": df["points"].max(),
        "points_scoring_results": (df["points"] > 0).sum(),
        "podiums": (df["points"] >= 15).sum(),  # 1st, 2nd, 3rd
        "wins": (df["points"] == 25).sum(),
    }
