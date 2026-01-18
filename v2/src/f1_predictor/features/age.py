"""Age calculation features for drivers."""

from datetime import date, datetime
from typing import Optional, Union

import numpy as np
import pandas as pd


def calculate_age_at_race(
    date_of_birth: Union[str, date, datetime],
    race_date: Union[str, date, datetime],
) -> Optional[int]:
    """
    Calculate driver's age in days at the time of a race.

    From notebook: ageDuringRace calculation

    Args:
        date_of_birth: Driver's date of birth
        race_date: Date of the race

    Returns:
        Age in days, or None if dates are invalid
    """
    try:
        # Parse date of birth
        if isinstance(date_of_birth, str):
            dob = datetime.strptime(date_of_birth, "%Y-%m-%d").date()
        elif isinstance(date_of_birth, datetime):
            dob = date_of_birth.date()
        elif isinstance(date_of_birth, date):
            dob = date_of_birth
        else:
            return None

        # Parse race date
        if isinstance(race_date, str):
            race = datetime.strptime(race_date, "%Y-%m-%d").date()
        elif isinstance(race_date, datetime):
            race = race_date.date()
        elif isinstance(race_date, date):
            race = race_date
        else:
            return None

        # Calculate age in days
        age_delta = race - dob
        return age_delta.days

    except (ValueError, TypeError, AttributeError):
        return None


def calculate_age_years(age_days: Optional[int]) -> Optional[float]:
    """
    Convert age in days to age in years.

    Args:
        age_days: Age in days

    Returns:
        Age in years (float), or None if invalid
    """
    if age_days is None or pd.isna(age_days):
        return None
    return age_days / 365.25


def get_age_bracket(age_days: Optional[int]) -> Optional[int]:
    """
    Convert age in days to age bracket (years).

    From notebook: age_bracket function

    Args:
        age_days: Age in days

    Returns:
        Age in whole years, or None if invalid
    """
    if age_days is None or pd.isna(age_days):
        return None
    return int(age_days // 365)


def process_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process age-related features for a DataFrame.

    Args:
        df: DataFrame with 'date_of_birth' and 'date' columns

    Returns:
        DataFrame with 'ageDuringRace' column added
    """
    df = df.copy()

    # Ensure date columns are datetime
    if "date_of_birth" in df.columns:
        df["date_of_birth"] = pd.to_datetime(df["date_of_birth"])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Calculate age in days
    if "date_of_birth" in df.columns and "date" in df.columns:
        df["ageDuringRace"] = df.apply(
            lambda row: calculate_age_at_race(
                row["date_of_birth"], row["date"]
            ),
            axis=1,
        )
    else:
        df["ageDuringRace"] = np.nan

    return df


def is_peak_age(age_days: Optional[int]) -> bool:
    """
    Check if driver is in peak performance age range (23-27 years).

    Based on notebook analysis showing peak at age 25.

    Args:
        age_days: Age in days

    Returns:
        True if in peak age range
    """
    if age_days is None or pd.isna(age_days):
        return False

    age_years = age_days / 365.25
    return 23 <= age_years <= 27


def is_veteran(age_days: Optional[int], threshold_years: int = 33) -> bool:
    """
    Check if driver is a veteran (above threshold age).

    Based on notebook analysis showing decline after 33.

    Args:
        age_days: Age in days
        threshold_years: Age threshold for veteran status

    Returns:
        True if driver is above threshold age
    """
    if age_days is None or pd.isna(age_days):
        return False

    age_years = age_days / 365.25
    return age_years > threshold_years
