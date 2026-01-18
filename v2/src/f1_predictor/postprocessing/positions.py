"""Convert predictions to race positions."""

from typing import Optional

import pandas as pd


def predictions_to_positions(
    df: pd.DataFrame,
    pred_column: str = "pred",
    groupby_columns: Optional[list[str]] = None,
) -> pd.Series:
    """
    Convert time gap predictions to race positions.

    Lower predicted gap = better position (winner has smallest gap).
    Predictions are ranked within each race to determine positions.

    From notebook: The indexer function that sorts predictions within each race.

    Args:
        df: DataFrame with predictions
        pred_column: Name of the prediction column
        groupby_columns: Columns to group by for per-race ranking

    Returns:
        Series with predicted positions
    """
    if groupby_columns is None:
        groupby_columns = ["season", "round"]

    # Check for required columns
    for col in groupby_columns:
        if col not in df.columns:
            if col == "round" and "race_name" in df.columns:
                groupby_columns = ["season", "race_name"]
                break

    def rank_within_group(group: pd.DataFrame) -> pd.Series:
        """Rank predictions within a race (1 = best/smallest gap)."""
        return group[pred_column].rank(method="min").astype(int)

    return df.groupby(groupby_columns, group_keys=False).apply(rank_within_group)


def predictions_to_positions_detailed(
    df: pd.DataFrame,
    pred_column: str = "pred",
) -> pd.DataFrame:
    """
    Convert predictions to positions with detailed race information.

    Args:
        df: DataFrame with predictions and race metadata
        pred_column: Name of the prediction column

    Returns:
        DataFrame with predicted positions and race info
    """
    df = df.copy()

    # Get unique races
    race_cols = ["season", "round"] if "round" in df.columns else ["season", "race_name"]

    results = []

    for _, race_group in df.groupby(race_cols):
        race_df = race_group.copy()

        # Rank predictions (lower is better)
        race_df["pred_position"] = race_df[pred_column].rank(method="min").astype(int)

        # Sort by predicted position
        race_df = race_df.sort_values("pred_position")

        results.append(race_df)

    return pd.concat(results, ignore_index=True)


def compare_positions(
    df: pd.DataFrame,
    true_col: str = "finish_position",
    pred_col: str = "pred_position",
) -> pd.DataFrame:
    """
    Compare true vs predicted positions.

    Args:
        df: DataFrame with true and predicted positions
        true_col: Column name for true positions
        pred_col: Column name for predicted positions

    Returns:
        DataFrame with comparison metrics
    """
    df = df.copy()

    # Calculate position difference
    df["position_diff"] = df[pred_col] - df[true_col]
    df["position_diff_abs"] = df["position_diff"].abs()

    # Calculate match indicators
    df["exact_match"] = df[true_col] == df[pred_col]
    df["within_1"] = df["position_diff_abs"] <= 1
    df["within_2"] = df["position_diff_abs"] <= 2
    df["within_3"] = df["position_diff_abs"] <= 3

    return df
