"""Ergast API client for F1 data retrieval."""

import time
from typing import Any, Iterator

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from f1_predictor.core.config import get_settings
from f1_predictor.core.exceptions import ErgastAPIError
from f1_predictor.core.logging import get_logger
from f1_predictor.data.schemas.models import (
    Circuit,
    Driver,
    QualifyingResult,
    Race,
    RaceResult,
    Standing,
)

logger = get_logger(__name__)


class ErgastClient:
    """
    Client for Ergast F1 API.

    Provides methods to fetch F1 data including races, results,
    qualifying, drivers, circuits, and standings.
    """

    def __init__(self) -> None:
        self.settings = get_settings().ergast
        self._client = httpx.Client(
            base_url=self.settings.base_url,
            timeout=self.settings.timeout,
            headers={"User-Agent": "F1Predictor/2.0"},
        )
        self._last_request_time: float = 0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.settings.rate_limit:
            time.sleep(self.settings.rate_limit - elapsed)
        self._last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _get(self, endpoint: str) -> dict[str, Any]:
        """
        Make GET request with rate limiting and retries.

        Args:
            endpoint: API endpoint (without base URL)

        Returns:
            Parsed JSON response MRData object

        Raises:
            ErgastAPIError: If the API request fails
        """
        self._rate_limit()
        url = f"{endpoint}.json"
        logger.debug(f"Fetching: {url}")

        try:
            response = self._client.get(url)
            response.raise_for_status()
            return response.json()["MRData"]
        except httpx.HTTPStatusError as e:
            raise ErgastAPIError(
                f"API request failed: {e.response.status_code}",
                status_code=e.response.status_code,
            ) from e
        except httpx.RequestError as e:
            raise ErgastAPIError(f"Request error: {e}") from e

    def get_races(self, year: int) -> list[Race]:
        """
        Get all races for a season.

        Maps to: /f1/{year}.json

        Args:
            year: Season year

        Returns:
            List of Race objects
        """
        data = self._get(f"/{year}")
        races = data["RaceTable"]["Races"]
        return [Race.model_validate(r) for r in races]

    def get_results(self, season: int, round_num: int) -> list[RaceResult]:
        """
        Get race results for a specific race.

        Maps to: /f1/{season}/{race}/results.json

        Args:
            season: Season year
            round_num: Race round number

        Returns:
            List of RaceResult objects
        """
        data = self._get(f"/{season}/{round_num}/results")
        if not data["RaceTable"]["Races"]:
            return []
        results = data["RaceTable"]["Races"][0]["Results"]
        return [RaceResult.model_validate(r) for r in results]

    def get_qualifying(self, season: int, round_num: int) -> list[QualifyingResult]:
        """
        Get qualifying results for a specific race.

        Maps to: /f1/{season}/{race}/qualifying.json

        Args:
            season: Season year
            round_num: Race round number

        Returns:
            List of QualifyingResult objects
        """
        data = self._get(f"/{season}/{round_num}/qualifying")
        if not data["RaceTable"]["Races"]:
            return []
        qualifying = data["RaceTable"]["Races"][0]["QualifyingResults"]
        return [QualifyingResult.model_validate(q) for q in qualifying]

    def get_drivers(self, year: int) -> list[Driver]:
        """
        Get all drivers for a season.

        Maps to: /f1/{year}/drivers.json

        Args:
            year: Season year

        Returns:
            List of Driver objects
        """
        data = self._get(f"/{year}/drivers")
        drivers = data["DriverTable"]["Drivers"]
        return [Driver.model_validate(d) for d in drivers]

    def get_circuits(self, year: int) -> list[Circuit]:
        """
        Get all circuits for a season.

        Maps to: /f1/{year}/circuits.json

        Args:
            year: Season year

        Returns:
            List of Circuit objects
        """
        data = self._get(f"/{year}/circuits")
        circuits = data["CircuitTable"]["Circuits"]
        return [Circuit.model_validate(c) for c in circuits]

    def get_driver_standings(self, year: int) -> list[Standing]:
        """
        Get driver championship standings for a season.

        Maps to: /f1/{year}/driverStandings.json

        Args:
            year: Season year

        Returns:
            List of Standing objects
        """
        data = self._get(f"/{year}/driverStandings")
        standings_list = data["StandingsTable"]["StandingsLists"]
        if not standings_list:
            return []
        standings = standings_list[0]["DriverStandings"]
        return [Standing.model_validate(s) for s in standings]

    def get_constructor_standings(self, year: int) -> list[Standing]:
        """
        Get constructor championship standings for a season.

        Maps to: /f1/{year}/constructorStandings.json

        Args:
            year: Season year

        Returns:
            List of Standing objects
        """
        data = self._get(f"/{year}/constructorStandings")
        standings_list = data["StandingsTable"]["StandingsLists"]
        if not standings_list:
            return []
        standings = standings_list[0]["ConstructorStandings"]
        return [Standing.model_validate(s) for s in standings]

    def get_seasons(self, start: int, end: int) -> Iterator[int]:
        """
        Generate season years for iteration.

        Args:
            start: Start year (inclusive)
            end: End year (inclusive)

        Yields:
            Season year
        """
        yield from range(start, end + 1)

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "ErgastClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
