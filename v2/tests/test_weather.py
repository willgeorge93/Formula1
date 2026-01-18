"""Tests for weather text cleaning."""

import pytest

from f1_predictor.features.weather import (
    clean_weather_text,
    extract_temperature,
    extract_weather_conditions,
)


class TestCleanWeatherText:
    """Tests for clean_weather_text function."""

    def test_basic_cleaning(self):
        """Test basic text cleaning."""
        result = clean_weather_text("Sunny, warm")
        assert "sunny" in result.lower()
        assert "warm" in result.lower()

    def test_html_removal(self):
        """Test HTML tag removal."""
        result = clean_weather_text("<p>Sunny weather</p>")
        assert "<" not in result
        assert ">" not in result

    def test_temperature_range_averaging(self):
        """Test temperature range conversion to average."""
        result = clean_weather_text("20-30°c")
        assert "25" in result or "25.0" in result

    def test_special_characters(self):
        """Test special character handling."""
        result = clean_weather_text("Weather: Sunny\xa0warm")
        assert "\xa0" not in result

    def test_empty_inputs(self):
        """Test handling of empty inputs."""
        assert clean_weather_text(None) == ""
        assert clean_weather_text("") == ""
        assert clean_weather_text("nan") == ""
        assert clean_weather_text("None") == ""


class TestExtractTemperature:
    """Tests for extract_temperature function."""

    def test_celsius_extraction(self):
        """Test extraction of Celsius temperatures."""
        assert extract_temperature("sunny 25°c") == pytest.approx(25.0)
        assert extract_temperature("warm 30 °c") == pytest.approx(30.0)

    def test_no_temperature(self):
        """Test when no temperature is found."""
        assert extract_temperature("sunny warm") is None
        assert extract_temperature("") is None
        assert extract_temperature(None) is None


class TestExtractWeatherConditions:
    """Tests for extract_weather_conditions function."""

    def test_condition_extraction(self):
        """Test extraction of weather conditions."""
        conditions = extract_weather_conditions("sunny warm dry")
        assert "sunny" in conditions
        assert "warm" in conditions
        assert "dry" in conditions

    def test_empty_input(self):
        """Test handling of empty input."""
        assert extract_weather_conditions("") == []
        assert extract_weather_conditions(None) == []

    def test_mixed_conditions(self):
        """Test extraction of mixed conditions."""
        conditions = extract_weather_conditions("partly cloudy with rain")
        assert "cloudy" in conditions
        assert "rain" in conditions
        assert "partly" in conditions
