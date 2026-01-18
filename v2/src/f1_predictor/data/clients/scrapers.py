"""Web scrapers for supplementary F1 data."""

from typing import Optional

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from f1_predictor.core.config import get_settings
from f1_predictor.core.constants import RACE_NAME_MAPPINGS
from f1_predictor.core.exceptions import ScrapingError
from f1_predictor.core.logging import get_logger
from f1_predictor.data.schemas.models import Weather

logger = get_logger(__name__)


class WikipediaScraper:
    """Scraper for Wikipedia F1 race pages."""

    def __init__(self) -> None:
        self.settings = get_settings().scraping
        self._client = httpx.Client(
            timeout=30,
            headers={"User-Agent": self.settings.user_agent},
            follow_redirects=True,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        reraise=True,
    )
    def get_weather(self, url: str) -> Optional[str]:
        """
        Extract weather from Wikipedia race infobox.

        Args:
            url: Wikipedia race page URL

        Returns:
            Weather text or None if not found
        """
        if not url:
            return None

        try:
            response = self._client.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Find infobox table
            table = soup.find("table", attrs={"class": "infobox"})
            if not table:
                return None

            # Search for weather row
            rows = table.find_all("tr")
            for row in rows:
                th = row.find("th", attrs={"scope": "row"})
                if th and "Weather" in str(th):
                    td = row.find("td")
                    if td:
                        return td.text.strip()

            return None
        except httpx.HTTPError as e:
            logger.warning(f"Failed to fetch Wikipedia page: {e}")
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        reraise=True,
    )
    def get_distance(self, url: str) -> Optional[str]:
        """
        Extract race distance from Wikipedia race infobox.

        Args:
            url: Wikipedia race page URL

        Returns:
            Distance string or None if not found
        """
        if not url:
            return None

        try:
            response = self._client.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            table = soup.find("table", attrs={"class": "infobox"})
            if not table:
                return None

            rows = table.find_all("tr")
            for row in rows:
                th = row.find("th", attrs={"scope": "row"})
                if th and "Distance" in str(th):
                    td = row.find("td")
                    if td:
                        text = td.text
                        # Extract km value
                        if "km" in text:
                            parts = text.split(",")
                            for part in parts:
                                if "km" in part:
                                    return part.split("km")[0].strip()
            return None
        except httpx.HTTPError as e:
            logger.warning(f"Failed to fetch Wikipedia page: {e}")
            return None

    def get_circuits_info(self) -> list[dict]:
        """
        Get circuit information from Wikipedia circuits list.

        Returns:
            List of circuit info dictionaries
        """
        url = "https://en.wikipedia.org/wiki/List_of_Formula_One_circuits"
        try:
            response = self._client.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            tables = soup.find_all("table", attrs={"class": "wikitable"})
            if len(tables) < 2:
                return []

            table = tables[1]  # Second wikitable contains circuit details
            body = table.find("tbody")
            if not body:
                return []

            circuits = []
            for row in body.find_all("tr"):
                cells = [
                    td.text.strip().strip("✔")
                    for td in row.find_all("td")
                    if td.text.strip()
                ]
                if len(cells) >= 4:
                    circuits.append(
                        {
                            "name": cells[0] if len(cells) > 0 else None,
                            "type": cells[1] if len(cells) > 1 else None,
                            "direction": cells[2] if len(cells) > 2 else None,
                            "location": cells[3] if len(cells) > 3 else None,
                            "length": cells[4] if len(cells) > 4 else None,
                        }
                    )

            return circuits
        except httpx.HTTPError as e:
            logger.warning(f"Failed to fetch circuits list: {e}")
            return []

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()


class F1FansiteScraper:
    """Scraper for F1-Fansite.com weather data."""

    def __init__(self) -> None:
        self.settings = get_settings().scraping
        self._client = httpx.Client(
            timeout=30,
            headers={"User-Agent": self.settings.user_agent},
            follow_redirects=True,
        )

    def _normalize_race_name(self, name: str) -> str:
        """
        Convert race name to URL-friendly format.

        Args:
            name: Original race name (e.g., "Australian Grand Prix")

        Returns:
            Normalized name for URL (e.g., "australian")
        """
        # Remove "Grand Prix" and convert to lowercase
        parts = name.lower().replace(" grand prix", "").split()
        normalized = "-".join(parts)

        # Apply mappings for special cases
        return RACE_NAME_MAPPINGS.get(normalized, normalized)

    def get_weather(self, year: int, race_name: str) -> Optional[str]:
        """
        Get weather data from F1-Fansite for a specific race.

        Args:
            year: Season year
            race_name: Race name

        Returns:
            Weather text or None if not found
        """
        normalized_name = self._normalize_race_name(race_name)

        # Try multiple URL patterns
        for url_template in self.settings.f1fansite_urls:
            url = url_template.format(year=year, race=normalized_name)
            weather = self._try_fetch_weather(url)
            if weather:
                return weather

        return None

    def _try_fetch_weather(self, url: str) -> Optional[str]:
        """
        Try to fetch weather from a specific URL.

        Args:
            url: Full URL to try

        Returns:
            Weather text or None if not found
        """
        try:
            response = self._client.get(url)

            # Check for 404
            if response.status_code == 404:
                return None

            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Check for 404 page class
            body = soup.find("body")
            if body and body.get("class"):
                if "error404" in body.get("class", []):
                    return None

            # Search for weather in <p> tags
            sections = soup.find_all("p")
            for section in sections:
                if "Weather" in str(section):
                    text = str(section)
                    cats = text.split("br")
                    for cat in cats:
                        if "Weather" in cat:
                            # Clean up the weather text
                            weather = cat.replace("<p>", "").replace("</p>", "")
                            weather = weather.replace("<br/>", "").replace("<br>", "")
                            return weather.strip()

            return None
        except httpx.HTTPError:
            return None

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()


def collect_weather(
    season: int,
    round_num: int,
    race_name: str,
    wikipedia_url: Optional[str] = None,
) -> Weather:
    """
    Collect weather data from all available sources.

    Args:
        season: Season year
        round_num: Race round number
        race_name: Race name
        wikipedia_url: Optional Wikipedia URL for the race

    Returns:
        Weather object with collected data
    """
    settings = get_settings().scraping
    weather_text = None
    source = "none"

    # Try Wikipedia first
    if settings.wikipedia_enabled and wikipedia_url:
        scraper = WikipediaScraper()
        try:
            weather_text = scraper.get_weather(wikipedia_url)
            if weather_text:
                source = "wikipedia"
        finally:
            scraper.close()

    # Fall back to F1-Fansite
    if not weather_text and settings.f1fansite_enabled:
        scraper = F1FansiteScraper()
        try:
            weather_text = scraper.get_weather(season, race_name)
            if weather_text:
                source = "f1fansite"
        finally:
            scraper.close()

    return Weather(
        season=season,
        round=round_num,
        race_name=race_name,
        weather_text=weather_text,
        source=source,
    )
