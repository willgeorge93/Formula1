"""Feature engineering pipeline orchestration."""

from typing import Optional

import numpy as np
import pandas as pd

from f1_predictor.core.constants import ALL_FEATURES
from f1_predictor.core.logging import get_logger
from f1_predictor.features.age import process_age_features
from f1_predictor.features.qualifying import process_qualifying_times
from f1_predictor.features.splits import calculate_split_times
from f1_predictor.features.weather import clean_weather_text

logger = get_logger(__name__)


class FeaturePipeline:
    """
    Feature engineering pipeline for F1 race data.

    Transforms raw race data into the feature matrix used for
    model training and prediction.
    """

    def __init__(self, include_target: bool = True) -> None:
        """
        Initialize the feature pipeline.

        Args:
            include_target: Whether to include target variable (filled_splits)
        """
        self.include_target = include_target

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature transformations to the input DataFrame.

        Args:
            df: Raw merged DataFrame with race data

        Returns:
            Transformed DataFrame with engineered features
        """
        logger.info(f"Processing {len(df)} rows through feature pipeline")

        df = df.copy()

        # Step 1: Process qualifying times
        df = self._process_qualifying(df)

        # Step 2: Process age features
        df = process_age_features(df)

        # Step 3: Clean weather text
        df = self._process_weather(df)

        # Step 4: Calculate split times (target variable)
        if self.include_target and "time_millis" in df.columns:
            df = calculate_split_times(df)

        # Step 5: Fill missing values
        df = self._fill_missing_values(df)

        # Step 6: Select and order final columns
        df = self._select_columns(df)

        logger.info(f"Feature pipeline complete. Output shape: {df.shape}")

        return df

    def _process_qualifying(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process qualifying time columns."""
        # Check if we have raw qualifying time strings
        has_q_strings = all(col in df.columns for col in ["q1", "q2", "q3"])

        if has_q_strings:
            # Check if already processed (numeric) or need parsing (strings)
            sample_q1 = df["q1"].dropna().iloc[0] if len(df["q1"].dropna()) > 0 else None

            if sample_q1 is not None and isinstance(sample_q1, str) and ":" in str(sample_q1):
                # Parse string format qualifying times
                quali_features = df.apply(
                    lambda row: pd.Series(
                        process_qualifying_times(row["q1"], row["q2"], row["q3"])
                    ),
                    axis=1,
                )
                df = df.drop(columns=["q1", "q2", "q3"])
                df = pd.concat([df, quali_features], axis=1)

        # Calculate derived features if not present
        if "q_best" not in df.columns and "q1" in df.columns:
            df["q_best"] = df[["q1", "q2", "q3"]].min(axis=1)
            df["q_worst"] = df[["q1", "q2", "q3"]].max(axis=1)
            df["q_mean"] = df[["q1", "q2", "q3"]].mean(axis=1)

        return df

    def _process_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize weather text."""
        if "weather" in df.columns:
            df["weather"] = df["weather"].apply(clean_weather_text)
        else:
            df["weather"] = ""

        return df

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with appropriate defaults."""
        # Fill missing categorical values
        categorical_cols = ["direction", "type", "locality", "country", "race_name"]
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")

        # Fill missing numerical values with column median (within season)
        numerical_cols = ["q_best", "q_worst", "q_mean", "length", "ageDuringRace", "grid"]
        for col in numerical_cols:
            if col in df.columns:
                df[col] = df.groupby("season")[col].transform(
                    lambda x: x.fillna(x.median())
                )
                # Fill any remaining NaN with overall median
                df[col] = df[col].fillna(df[col].median())

        # Fill weather with empty string
        if "weather" in df.columns:
            df["weather"] = df["weather"].fillna("")

        return df

    def _select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and order final columns for the model."""
        # Core feature columns
        feature_cols = [
            "season",
            "round",
            "race_name",
            "name",
            "constructor",
            "grid",
            "qual_position",
            "q_best",
            "q_worst",
            "q_mean",
            "ageDuringRace",
            "circuit_id",
            "locality",
            "country",
            "type",
            "direction",
            "length",
            "weather",
        ]

        # Target columns
        target_cols = ["finish_position", "points", "status"]
        if self.include_target:
            target_cols.append("filled_splits")

        # Select available columns
        available_cols = [col for col in feature_cols + target_cols if col in df.columns]

        return df[available_cols]

    def get_feature_columns(self) -> list[str]:
        """Get list of feature column names used by the model."""
        return ALL_FEATURES

    def prepare_for_training(
        self, df: pd.DataFrame, test_season: Optional[int] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for model training with train/test split.

        Args:
            df: Transformed DataFrame
            test_season: Season to use for testing (default: 2020)

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        from f1_predictor.core.config import get_settings

        settings = get_settings()
        test_season = test_season or settings.model.test_season

        # Split by season
        train_mask = df["season"] < test_season
        test_mask = df["season"] == test_season

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        # Separate target
        y_train = train_df.pop("filled_splits")
        y_test = test_df.pop("filled_splits")

        # Remove other target columns
        for col in ["finish_position", "points", "status"]:
            if col in train_df.columns:
                train_df.pop(col)
            if col in test_df.columns:
                test_df.pop(col)

        # Remove identifier columns not used for prediction
        drop_cols = ["name", "constructor", "circuit_id"]
        for col in drop_cols:
            if col in train_df.columns:
                train_df = train_df.drop(columns=[col])
            if col in test_df.columns:
                test_df = test_df.drop(columns=[col])

        return train_df, test_df, y_train, y_test
