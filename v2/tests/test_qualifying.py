"""Tests for qualifying time parsing."""

import pytest

from f1_predictor.features.qualifying import (
    parse_qualifying_time,
    calculate_quali_best,
    calculate_quali_worst,
    calculate_quali_mean,
    process_qualifying_times,
)


class TestParseQualifyingTime:
    """Tests for parse_qualifying_time function."""

    def test_valid_time_parsing(self):
        """Test parsing valid qualifying times."""
        assert parse_qualifying_time("1:23.456") == pytest.approx(83.456, rel=1e-3)
        assert parse_qualifying_time("1:05.123") == pytest.approx(65.123, rel=1e-3)
        assert parse_qualifying_time("1:30.000") == pytest.approx(90.0, rel=1e-3)

    def test_edge_cases(self):
        """Test edge cases."""
        assert parse_qualifying_time("0:59.999") == pytest.approx(59.999, rel=1e-3)
        assert parse_qualifying_time("2:00.000") == pytest.approx(120.0, rel=1e-3)

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        assert parse_qualifying_time(None) is None
        assert parse_qualifying_time("") is None
        assert parse_qualifying_time("invalid") is None
        assert parse_qualifying_time("nan") is None

    def test_whitespace_handling(self):
        """Test handling of whitespace."""
        assert parse_qualifying_time(" 1:23.456 ") == pytest.approx(83.456, rel=1e-3)


class TestQualifyingAggregations:
    """Tests for qualifying time aggregation functions."""

    def test_quali_best(self):
        """Test finding best qualifying time."""
        assert calculate_quali_best(83.0, 82.0, 81.5) == pytest.approx(81.5)
        assert calculate_quali_best(83.0, None, 81.5) == pytest.approx(81.5)
        assert calculate_quali_best(None, None, None) is None

    def test_quali_worst(self):
        """Test finding worst qualifying time."""
        assert calculate_quali_worst(83.0, 82.0, 81.5) == pytest.approx(83.0)
        assert calculate_quali_worst(83.0, None, 81.5) == pytest.approx(83.0)
        assert calculate_quali_worst(None, None, None) is None

    def test_quali_mean(self):
        """Test calculating mean qualifying time."""
        assert calculate_quali_mean(83.0, 82.0, 81.0) == pytest.approx(82.0)
        assert calculate_quali_mean(83.0, None, 81.0) == pytest.approx(82.0)
        assert calculate_quali_mean(None, None, None) is None


class TestProcessQualifyingTimes:
    """Tests for full qualifying time processing."""

    def test_full_processing(self):
        """Test complete qualifying time processing."""
        result = process_qualifying_times("1:23.000", "1:22.500", "1:22.000")

        assert result["q1"] == pytest.approx(83.0)
        assert result["q2"] == pytest.approx(82.5)
        assert result["q3"] == pytest.approx(82.0)
        assert result["q_best"] == pytest.approx(82.0)
        assert result["q_worst"] == pytest.approx(83.0)
        assert result["q_mean"] == pytest.approx(82.5)

    def test_partial_times(self):
        """Test processing with missing sessions."""
        result = process_qualifying_times("1:23.000", None, None)

        assert result["q1"] == pytest.approx(83.0)
        assert result["q2"] is None
        assert result["q3"] is None
        assert result["q_best"] == pytest.approx(83.0)
        assert result["q_worst"] == pytest.approx(83.0)
        assert result["q_mean"] == pytest.approx(83.0)
