"""Tests for points calculation."""

import pytest

from f1_predictor.postprocessing.points import (
    position_to_points,
    calculate_race_points,
)
from f1_predictor.core.constants import POINTS_SYSTEM


class TestPositionToPoints:
    """Tests for position_to_points function."""

    def test_top_ten_positions(self):
        """Test points for top 10 positions."""
        assert position_to_points(1) == 25
        assert position_to_points(2) == 18
        assert position_to_points(3) == 15
        assert position_to_points(4) == 12
        assert position_to_points(5) == 10
        assert position_to_points(6) == 8
        assert position_to_points(7) == 6
        assert position_to_points(8) == 4
        assert position_to_points(9) == 2
        assert position_to_points(10) == 1

    def test_outside_top_ten(self):
        """Test positions outside top 10."""
        assert position_to_points(11) == 0
        assert position_to_points(20) == 0

    def test_matches_constant(self):
        """Test that function matches POINTS_SYSTEM constant."""
        for pos, points in POINTS_SYSTEM.items():
            assert position_to_points(pos) == points


class TestCalculateRacePoints:
    """Tests for calculate_race_points function."""

    def test_race_points_calculation(self, sample_predictions):
        """Test race points calculation."""
        sample_predictions["position"] = [1, 2, 3, 1, 2, 3]
        points = calculate_race_points(sample_predictions, position_column="position")

        expected = [25, 18, 15, 25, 18, 15]
        assert points.tolist() == expected
