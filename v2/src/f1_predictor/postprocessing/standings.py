"""Calculate championship standings from race results."""

from typing import Optional, Tuple

import pandas as pd

from f1_predictor.postprocessing.points import position_to_points


def calculate_driver_standings(
    df: pd.DataFrame,
    position_column: str = "pred_position",
    name_column: str = "name",
) -> pd.DataFrame:
    """
    Calculate driver championship standings from race results.

    From notebook: driver_points aggregation logic

    Args:
        df: DataFrame with race results
        position_column: Column with finishing positions
        name_column: Column with driver names

    Returns:
        DataFrame with driver standings
    """
    df = df.copy()

    # Calculate points for each result
    df["points"] = df[position_column].apply(position_to_points)

    # Aggregate by driver
    standings = (
        df.groupby(name_column)
        .agg(
            total_points=("points", "sum"),
            races=("points", "count"),
            wins=("points", lambda x: (x == 25).sum()),
            podiums=("points", lambda x: (x >= 15).sum()),
            points_finishes=("points", lambda x: (x > 0).sum()),
        )
        .reset_index()
    )

    # Sort by points (descending)
    standings = standings.sort_values("total_points", ascending=False)

    # Add position
    standings["position"] = range(1, len(standings) + 1)

    # Reorder columns
    columns = [
        "position",
        name_column,
        "total_points",
        "wins",
        "podiums",
        "points_finishes",
        "races",
    ]
    standings = standings[columns]

    return standings.reset_index(drop=True)


def calculate_constructor_standings(
    df: pd.DataFrame,
    position_column: str = "pred_position",
    constructor_column: str = "constructor",
) -> pd.DataFrame:
    """
    Calculate constructor championship standings from race results.

    From notebook: constructor_points aggregation logic

    Args:
        df: DataFrame with race results
        position_column: Column with finishing positions
        constructor_column: Column with constructor names

    Returns:
        DataFrame with constructor standings
    """
    df = df.copy()

    # Calculate points for each result
    df["points"] = df[position_column].apply(position_to_points)

    # Aggregate by constructor
    standings = (
        df.groupby(constructor_column)
        .agg(
            total_points=("points", "sum"),
            races=("points", "count"),
            wins=("points", lambda x: (x == 25).sum()),
            podiums=("points", lambda x: (x >= 15).sum()),
            points_finishes=("points", lambda x: (x > 0).sum()),
        )
        .reset_index()
    )

    # Sort by points (descending)
    standings = standings.sort_values("total_points", ascending=False)

    # Add position
    standings["position"] = range(1, len(standings) + 1)

    # Reorder columns
    columns = [
        "position",
        constructor_column,
        "total_points",
        "wins",
        "podiums",
        "points_finishes",
        "races",
    ]
    standings = standings[columns]

    return standings.reset_index(drop=True)


def calculate_standings_from_predictions(
    df: pd.DataFrame,
    position_column: str = "pred_position",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate both driver and constructor standings from predictions.

    Args:
        df: DataFrame with predictions and metadata
        position_column: Column with predicted positions

    Returns:
        Tuple of (driver_standings, constructor_standings)
    """
    driver_standings = calculate_driver_standings(
        df, position_column=position_column
    )
    constructor_standings = calculate_constructor_standings(
        df, position_column=position_column
    )

    return driver_standings, constructor_standings


def compare_standings(
    predicted: pd.DataFrame,
    actual: pd.DataFrame,
    name_column: str = "name",
) -> pd.DataFrame:
    """
    Compare predicted standings with actual standings.

    Args:
        predicted: DataFrame with predicted standings
        actual: DataFrame with actual standings
        name_column: Column used for joining

    Returns:
        DataFrame with comparison metrics
    """
    # Merge on name/constructor
    comparison = pd.merge(
        predicted,
        actual,
        on=name_column,
        suffixes=("_pred", "_actual"),
    )

    # Calculate differences
    comparison["position_diff"] = (
        comparison["position_pred"] - comparison["position_actual"]
    )
    comparison["points_diff"] = (
        comparison["total_points_pred"] - comparison["total_points_actual"]
    )
    comparison["exact_position"] = (
        comparison["position_pred"] == comparison["position_actual"]
    )

    return comparison


def format_standings_table(
    standings: pd.DataFrame,
    top_n: Optional[int] = None,
) -> str:
    """
    Format standings as a text table.

    Args:
        standings: DataFrame with standings
        top_n: Number of rows to show (default: all)

    Returns:
        Formatted table string
    """
    if top_n:
        standings = standings.head(top_n)

    # Determine columns to show
    if "name" in standings.columns:
        display_cols = ["position", "name", "total_points", "wins"]
    else:
        display_cols = ["position", "constructor", "total_points", "wins"]

    display_cols = [c for c in display_cols if c in standings.columns]

    return standings[display_cols].to_string(index=False)
