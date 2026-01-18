"""Weather text cleaning and normalization."""

import re
from typing import Optional

import numpy as np


def clean_weather_text(weather: Optional[str]) -> str:
    """
    Clean and normalize weather text from various sources.

    Extracted from notebook functions: text_filter, range_filter,
    race_weather_extract

    Args:
        weather: Raw weather text from Wikipedia or F1-Fansite

    Returns:
        Cleaned and normalized weather text
    """
    if weather is None or weather in ("None", "nan", "") or (isinstance(weather, float) and np.isnan(weather)):
        return ""

    text = str(weather).lower()

    # Character replacements from notebook
    replacements = [
        ("\xa0", " "),
        ("º", "°"),
        ("&amp;", ""),
        (",", " "),
        ("/>", ""),
        ("/p>", ""),
        ("p>", ""),
        ("/a>", ""),
        ("<p>", ""),
        ("</p>", ""),
        ("<br>", " "),
        ("<br/>", " "),
        ("☁", "clouds"),
        ("☂", "rain"),
        ("<", ""),
        (">", ""),
        ("weather:", ""),
        ("/", " "),
        (";", " "),
        (":", " "),
        ("(", " "),
        (")", " "),
    ]

    for old, new in replacements:
        text = text.replace(old, new)

    # Word spacing fixes from notebook
    spacing_fixes = [
        ("dryovercast", "dry overcast"),
        ("dryclouded", "dry clouded"),
        ("drysunny", "dry sunny"),
        ("dryclear", "dry clear"),
        ("clear26°c", "clear 26°c"),
        ("overcast22°c", "overcast 22°c"),
        ("sunny", "sunny "),
        ("temperature", "temperature "),
        ("cloudy", "cloudy "),
        ("clear", "clear "),
        ("later", "later "),
        ("dry", "dry "),
        ("times", "times "),
    ]

    for old, new in spacing_fixes:
        text = text.replace(old, new)

    # Remove citations like [1], [2], etc.
    text = _remove_citations(text)

    # Handle temperature ranges (convert to average)
    text = _normalize_temperature_ranges(text)

    # Remove rogue word endings
    text = _remove_rogue_endings(text)

    # Clean up multiple spaces
    text = " ".join(text.split())

    return text.strip()


def _remove_citations(text: str) -> str:
    """Remove Wikipedia citation brackets like [1], [2], etc."""
    return re.sub(r"\[\d+\]", "", text)


def _normalize_temperature_ranges(text: str) -> str:
    """
    Convert temperature ranges to averages.

    Examples:
        "20-25°c" -> "22.5°c"
        "20 to 25" -> "22.5"
    """
    # Handle hyphen ranges like "20-25"
    hyphen_pattern = r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)"
    matches = re.findall(hyphen_pattern, text)
    for match in matches:
        try:
            val1, val2 = float(match[0]), float(match[1])
            avg = (val1 + val2) / 2
            text = text.replace(f"{match[0]}-{match[1]}", str(avg))
        except (ValueError, IndexError):
            continue

    # Handle "X to Y" ranges
    to_pattern = r"(\d+\.?\d*)\s+to\s+(\d+\.?\d*)"
    matches = re.findall(to_pattern, text)
    for match in matches:
        try:
            val1, val2 = float(match[0]), float(match[1])
            avg = (val1 + val2) / 2
            text = text.replace(f"{match[0]} to {match[1]}", str(avg))
        except (ValueError, IndexError):
            continue

    # Handle en-dash ranges like "20–25"
    endash_pattern = r"(\d+\.?\d*)–(\d+\.?\d*)"
    matches = re.findall(endash_pattern, text)
    for match in matches:
        try:
            val1, val2 = float(match[0]), float(match[1])
            avg = (val1 + val2) / 2
            text = text.replace(f"{match[0]}–{match[1]}", str(avg))
        except (ValueError, IndexError):
            continue

    return text


def _remove_rogue_endings(text: str) -> str:
    """Clean up malformed word endings."""
    # Fix "temperature s" -> "temperatures"
    text = re.sub(r"temperature\s+s\s", "temperatures ", text)

    # Fix standalone "ing"
    text = re.sub(r"\s+ing[\s\.\,\:\;]+", "ing ", text)

    return text


def extract_temperature(weather: str) -> Optional[float]:
    """
    Extract temperature value from weather text.

    Args:
        weather: Cleaned weather text

    Returns:
        Temperature in Celsius, or None if not found
    """
    if not weather:
        return None

    # Look for patterns like "25°c", "25 °c", "25c"
    patterns = [
        r"(\d+\.?\d*)\s*°\s*c",
        r"(\d+\.?\d*)\s*degrees?\s*c",
        r"(\d+\.?\d*)c\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, weather.lower())
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    return None


def extract_weather_conditions(weather: str) -> list[str]:
    """
    Extract weather condition keywords from text.

    Args:
        weather: Cleaned weather text

    Returns:
        List of weather condition keywords
    """
    if not weather:
        return []

    conditions = []
    weather_lower = weather.lower()

    # Check for common weather keywords
    keywords = [
        "sunny",
        "cloudy",
        "overcast",
        "rain",
        "wet",
        "dry",
        "clear",
        "warm",
        "hot",
        "cold",
        "cool",
        "humid",
        "windy",
        "partly",
        "scattered",
        "showers",
    ]

    for keyword in keywords:
        if keyword in weather_lower:
            conditions.append(keyword)

    return conditions
