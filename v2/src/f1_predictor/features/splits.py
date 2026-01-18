"""Split time calculations for race performance metrics."""

from typing import Optional

import numpy as np
import pandas as pd

from f1_predictor.core.constants import NO_FAULT_STATUSES


def calculate_split_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate split times (gap to winner) for each race result.

    The split time represents the time difference between each driver
    and the race winner. This is used as the primary target variable
    for the prediction model.

    Args:
        df: DataFrame with columns: season, round, finish_position,
            time_millis, status

    Returns:
        DataFrame with additional columns: split_times, filled_splits
    """
    df = df.copy()

    # Convert time to seconds if in milliseconds
    if "time_millis" in df.columns:
        df["time_seconds"] = df["time_millis"] / 1000
    elif "time" in df.columns:
        df["time_seconds"] = df["time"]
    else:
        df["time_seconds"] = np.nan

    # Find winner's time for each race
    df["winner_time"] = df.groupby(["season", "round"])["time_seconds"].transform(
        lambda x: x[df.loc[x.index, "finish_position"] == 1].iloc[0]
        if len(x[df.loc[x.index, "finish_position"] == 1]) > 0
        else np.nan
    )

    # Calculate raw split times (gap to winner)
    df["split_times"] = df["time_seconds"] - df["winner_time"]

    # Fill splits for lapped cars and DNFs
    df["filled_splits"] = df.apply(
        lambda row: _compute_filled_split(
            row["split_times"],
            row.get("status", ""),
            row.get("finish_position", None),
        ),
        axis=1,
    )

    # Forward fill within races for remaining NaN values
    df["filled_splits"] = df.groupby(["season", "round"])["filled_splits"].transform(
        lambda x: x.ffill().bfill()
    )

    return df


def _compute_filled_split(
    split: Optional[float],
    status: str,
    position: Optional[int],
) -> Optional[float]:
    """
    Compute filled split time accounting for lapped cars.

    From notebook: split_compute function

    Args:
        split: Raw split time in seconds
        status: Race finish status (e.g., "Finished", "+1 Lap", "Retired")
        position: Finish position

    Returns:
        Adjusted split time
    """
    if pd.isna(split) or split is None:
        # For DNFs and lapped cars, use a penalty based on position
        if position is not None and not pd.isna(position):
            # Rough estimate: 2 seconds per position behind
            return position * 2.0
        return None

    # Check if car was lapped
    if status and "Lap" in str(status):
        try:
            # Extract number of laps down, e.g., "+2 Laps" -> 2
            parts = str(status).replace("+", "").split()
            if parts and parts[0].isdigit():
                laps_down = int(parts[0])
                # Multiply split by laps to account for being lapped
                return split * max(laps_down, 1)
        except (ValueError, IndexError):
            pass

    return split


def calculate_gap_to_leader(
    df: pd.DataFrame, time_col: str = "time_seconds"
) -> pd.Series:
    """
    Calculate gap to race leader for each row.

    Args:
        df: DataFrame with race data
        time_col: Column name containing lap/finish times

    Returns:
        Series with gap to leader
    """

    def get_leader_time(group: pd.DataFrame) -> pd.Series:
        leader_time = group[group["finish_position"] == 1][time_col]
        if len(leader_time) > 0:
            return group[time_col] - leader_time.iloc[0]
        return pd.Series(np.nan, index=group.index)

    return df.groupby(["season", "round"], group_keys=False).apply(get_leader_time)


def is_finished(status: str) -> bool:
    """
    Check if a driver finished the race (not DNF).

    Args:
        status: Race finish status

    Returns:
        True if driver finished, False if DNF
    """
    if not status:
        return False
    return status in NO_FAULT_STATUSES
