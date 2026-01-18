"""Tests for position conversion."""

import pytest
import pandas as pd

from f1_predictor.postprocessing.positions import (
    predictions_to_positions,
    compare_positions,
)


class TestPredictionsToPositions:
    """Tests for predictions_to_positions function."""

    def test_basic_ranking(self, sample_predictions):
        """Test basic prediction ranking."""
        positions = predictions_to_positions(sample_predictions)

        # Lower prediction = better position (1st)
        # Race 1: Hamilton (0.0) should be 1st, Verstappen (5.5) should be 2nd
        race1 = sample_predictions[sample_predictions["round"] == 1]
        pos1 = positions[race1.index]

        # Hamilton should be position 1 (lowest pred)
        hamilton_idx = race1[race1["name"] == "Lewis Hamilton"].index[0]
        assert pos1[hamilton_idx] == 1

    def test_per_race_ranking(self, sample_predictions):
        """Test that ranking is done per race."""
        positions = predictions_to_positions(sample_predictions)

        # Each race should have positions 1, 2, 3
        for round_num in [1, 2]:
            race_mask = sample_predictions["round"] == round_num
            race_positions = positions[race_mask]
            assert sorted(race_positions.tolist()) == [1, 2, 3]


class TestComparePositions:
    """Tests for compare_positions function."""

    def test_position_comparison(self, sample_predictions):
        """Test position comparison calculations."""
        sample_predictions["pred_position"] = [1, 2, 3, 1, 2, 3]
        sample_predictions["true_position"] = [1, 2, 3, 2, 1, 3]

        result = compare_positions(
            sample_predictions,
            true_col="true_position",
            pred_col="pred_position",
        )

        assert "position_diff" in result.columns
        assert "exact_match" in result.columns
        assert "within_1" in result.columns

    def test_exact_match_calculation(self, sample_predictions):
        """Test exact match calculation."""
        sample_predictions["pred_position"] = [1, 2, 3, 1, 2, 3]
        sample_predictions["true_position"] = [1, 2, 3, 1, 2, 3]

        result = compare_positions(
            sample_predictions,
            true_col="true_position",
            pred_col="pred_position",
        )

        # All should be exact matches
        assert result["exact_match"].all()
