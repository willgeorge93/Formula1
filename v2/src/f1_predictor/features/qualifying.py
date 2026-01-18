"""Qualifying time parsing and feature engineering."""

from typing import Optional

import numpy as np


def parse_qualifying_time(time_str: Optional[str]) -> Optional[float]:
    """
    Parse qualifying time from M:SS.mmm format to seconds.

    Extracted from notebook: qual_time_formatter function

    Args:
        time_str: Time string in M:SS.mmm format (e.g., "1:23.456")

    Returns:
        Time in seconds as float, or None if parsing fails

    Examples:
        >>> parse_qualifying_time("1:23.456")
        83.456
        >>> parse_qualifying_time("1:05.123")
        65.123
        >>> parse_qualifying_time(None)
        None
    """
    if time_str is None or time_str == "" or str(time_str) == "nan":
        return None

    try:
        time_str = str(time_str).strip()

        # Handle format: M:SS.mmm
        if ":" not in time_str:
            return None

        parts = time_str.split(":")
        if len(parts) != 2:
            return None

        minutes = int(parts[0])

        # Handle seconds and milliseconds
        if "." in parts[1]:
            seconds_parts = parts[1].split(".")
            seconds = int(seconds_parts[0])
            milliseconds = float(f"0.{seconds_parts[1]}")
        else:
            seconds = int(parts[1])
            milliseconds = 0.0

        return (minutes * 60) + seconds + milliseconds

    except (ValueError, IndexError, TypeError):
        return None


def calculate_quali_best(
    q1: Optional[float], q2: Optional[float], q3: Optional[float]
) -> Optional[float]:
    """
    Calculate best (fastest) qualifying time across all sessions.

    Args:
        q1: Q1 time in seconds
        q2: Q2 time in seconds
        q3: Q3 time in seconds

    Returns:
        Best time in seconds, or None if no valid times
    """
    times = [t for t in [q1, q2, q3] if t is not None and t > 0 and not np.isnan(t)]
    return min(times) if times else None


def calculate_quali_worst(
    q1: Optional[float], q2: Optional[float], q3: Optional[float]
) -> Optional[float]:
    """
    Calculate worst (slowest) qualifying time across all sessions.

    Args:
        q1: Q1 time in seconds
        q2: Q2 time in seconds
        q3: Q3 time in seconds

    Returns:
        Worst time in seconds, or None if no valid times
    """
    times = [t for t in [q1, q2, q3] if t is not None and t > 0 and not np.isnan(t)]
    return max(times) if times else None


def calculate_quali_mean(
    q1: Optional[float], q2: Optional[float], q3: Optional[float]
) -> Optional[float]:
    """
    Calculate mean qualifying time across all valid sessions.

    Args:
        q1: Q1 time in seconds
        q2: Q2 time in seconds
        q3: Q3 time in seconds

    Returns:
        Mean time in seconds, or None if no valid times
    """
    times = [t for t in [q1, q2, q3] if t is not None and t > 0 and not np.isnan(t)]
    return sum(times) / len(times) if times else None


def process_qualifying_times(
    q1_str: Optional[str], q2_str: Optional[str], q3_str: Optional[str]
) -> dict[str, Optional[float]]:
    """
    Process all qualifying times and calculate derived features.

    Args:
        q1_str: Q1 time string
        q2_str: Q2 time string
        q3_str: Q3 time string

    Returns:
        Dictionary with q1, q2, q3, q_best, q_worst, q_mean
    """
    q1 = parse_qualifying_time(q1_str)
    q2 = parse_qualifying_time(q2_str)
    q3 = parse_qualifying_time(q3_str)

    return {
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "q_best": calculate_quali_best(q1, q2, q3),
        "q_worst": calculate_quali_worst(q1, q2, q3),
        "q_mean": calculate_quali_mean(q1, q2, q3),
    }
