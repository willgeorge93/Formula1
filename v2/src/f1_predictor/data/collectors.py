"""Data collection orchestration."""

from datetime import datetime
from typing import Optional

import pandas as pd
from rich.progress import Progress, SpinnerColumn, TextColumn

from f1_predictor.core.config import get_settings
from f1_predictor.core.constants import CONSTRUCTOR_MAPPINGS
from f1_predictor.core.logging import get_logger
from f1_predictor.data.clients.ergast import ErgastClient
from f1_predictor.data.clients.scrapers import WikipediaScraper, collect_weather
from f1_predictor.data.storage import Storage, get_storage

logger = get_logger(__name__)


class DataCollector:
    """
    Orchestrates data collection from all sources.

    Collects F1 data from Ergast API and web scrapers,
    then stores it in the configured storage backend.
    """

    def __init__(self, storage: Optional[Storage] = None) -> None:
        self.storage = storage or get_storage()
        self.settings = get_settings()

    def collect_all(
        self,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        include_weather: bool = True,
    ) -> None:
        """
        Collect all F1 data for the specified year range.

        Args:
            start_year: Start year (default: from config)
            end_year: End year (default: current year)
            include_weather: Whether to collect weather data
        """
        start = start_year or self.settings.model.train_seasons_start
        end = end_year or datetime.now().year

        logger.info(f"Collecting F1 data for seasons {start}-{end}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            task = progress.add_task("Collecting races...", total=None)
            races_df = self.collect_races(start, end)

            progress.update(task, description="Collecting results...")
            results_df = self.collect_results(start, end)

            progress.update(task, description="Collecting qualifying...")
            qualifying_df = self.collect_qualifying(start, end)

            progress.update(task, description="Collecting drivers...")
            drivers_df = self.collect_drivers(start, end)

            progress.update(task, description="Collecting circuits...")
            circuits_df = self.collect_circuits(start, end)

            if include_weather:
                progress.update(task, description="Collecting weather...")
                weather_df = self.collect_weather_data(races_df)
            else:
                weather_df = pd.DataFrame()

            progress.update(task, description="Building main dataset...")
            main_df = self.build_main_dataset(
                races_df, results_df, qualifying_df, drivers_df, circuits_df, weather_df
            )

            progress.update(task, description="Saving data...")
            self._save_all(
                races_df,
                results_df,
                qualifying_df,
                drivers_df,
                circuits_df,
                weather_df,
                main_df,
            )

        logger.info("Data collection complete!")

    def collect_races(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Collect race schedule data."""
        all_races = []

        with ErgastClient() as client:
            for year in client.get_seasons(start_year, end_year):
                races = client.get_races(year)
                for race in races:
                    all_races.append(
                        {
                            "season": race.season,
                            "round": race.round,
                            "race_name": race.race_name,
                            "circuit_id": race.circuit.circuit_id,
                            "date": race.date,
                            "url": race.url,
                            "locality": race.circuit.location.locality
                            if race.circuit.location
                            else None,
                            "country": race.circuit.location.country
                            if race.circuit.location
                            else None,
                            "lat": race.circuit.location.lat
                            if race.circuit.location
                            else None,
                            "long": race.circuit.location.long
                            if race.circuit.location
                            else None,
                        }
                    )

        return pd.DataFrame(all_races)

    def collect_results(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Collect race results data."""
        all_results = []

        with ErgastClient() as client:
            for year in client.get_seasons(start_year, end_year):
                races = client.get_races(year)
                for race in races:
                    results = client.get_results(year, race.round)
                    for result in results:
                        constructor_id = result.constructor.constructor_id
                        # Apply constructor mappings for consistency
                        constructor_id = CONSTRUCTOR_MAPPINGS.get(
                            constructor_id, constructor_id
                        )

                        all_results.append(
                            {
                                "season": year,
                                "round": race.round,
                                "circuit_id": race.circuit.circuit_id,
                                "driver_id": result.driver.driver_id,
                                "constructor": constructor_id,
                                "grid": result.grid,
                                "finish_position": result.position,
                                "points": result.points,
                                "status": result.status,
                                "time_millis": result.time_millis,
                            }
                        )

        return pd.DataFrame(all_results)

    def collect_qualifying(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Collect qualifying results data."""
        all_qualifying = []

        with ErgastClient() as client:
            for year in client.get_seasons(start_year, end_year):
                races = client.get_races(year)
                for race in races:
                    qualifying = client.get_qualifying(year, race.round)
                    for qual in qualifying:
                        all_qualifying.append(
                            {
                                "season": year,
                                "round": race.round,
                                "circuit_id": race.circuit.circuit_id,
                                "driver_id": qual.driver.driver_id,
                                "constructor": qual.constructor.constructor_id,
                                "qual_position": qual.position,
                                "q1": qual.q1,
                                "q2": qual.q2,
                                "q3": qual.q3,
                            }
                        )

        return pd.DataFrame(all_qualifying)

    def collect_drivers(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Collect driver information."""
        all_drivers = {}

        with ErgastClient() as client:
            for year in client.get_seasons(start_year, end_year):
                drivers = client.get_drivers(year)
                for driver in drivers:
                    # Use latest info for each driver
                    all_drivers[driver.driver_id] = {
                        "driver_id": driver.driver_id,
                        "name": driver.full_name,
                        "given_name": driver.given_name,
                        "family_name": driver.family_name,
                        "nationality": driver.nationality,
                        "date_of_birth": driver.date_of_birth,
                        "code": driver.code,
                    }

        return pd.DataFrame(list(all_drivers.values()))

    def collect_circuits(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Collect circuit information."""
        all_circuits = {}

        # Get basic circuit info from Ergast
        with ErgastClient() as client:
            for year in client.get_seasons(start_year, end_year):
                circuits = client.get_circuits(year)
                for circuit in circuits:
                    all_circuits[circuit.circuit_id] = {
                        "circuit_id": circuit.circuit_id,
                        "circuit_name": circuit.circuit_name,
                        "locality": circuit.location.locality
                        if circuit.location
                        else None,
                        "country": circuit.location.country
                        if circuit.location
                        else None,
                        "lat": circuit.location.lat if circuit.location else None,
                        "long": circuit.location.long if circuit.location else None,
                        "url": circuit.url,
                    }

        circuits_df = pd.DataFrame(list(all_circuits.values()))

        # Enhance with Wikipedia scraped data
        scraper = WikipediaScraper()
        try:
            wiki_circuits = scraper.get_circuits_info()
            if wiki_circuits:
                wiki_df = pd.DataFrame(wiki_circuits)
                # Map and merge circuit attributes (type, direction, length)
                # This is a simplified version - full mapping would need circuit name matching
                circuits_df["type"] = "Road"  # Default
                circuits_df["direction"] = "Clockwise"  # Default
                circuits_df["length"] = None
        finally:
            scraper.close()

        return circuits_df

    def collect_weather_data(self, races_df: pd.DataFrame) -> pd.DataFrame:
        """Collect weather data for all races."""
        weather_data = []

        for _, race in races_df.iterrows():
            weather = collect_weather(
                season=race["season"],
                round_num=race["round"],
                race_name=race["race_name"],
                wikipedia_url=race.get("url"),
            )
            weather_data.append(
                {
                    "season": weather.season,
                    "round": weather.round,
                    "race_name": weather.race_name,
                    "weather": weather.weather_text,
                    "weather_source": weather.source,
                }
            )

        return pd.DataFrame(weather_data)

    def build_main_dataset(
        self,
        races_df: pd.DataFrame,
        results_df: pd.DataFrame,
        qualifying_df: pd.DataFrame,
        drivers_df: pd.DataFrame,
        circuits_df: pd.DataFrame,
        weather_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build the main dataset by merging all data sources.

        This creates the feature matrix used for model training.
        """
        from f1_predictor.features.pipeline import FeaturePipeline

        # Merge results and qualifying
        merged = pd.merge(
            results_df,
            qualifying_df,
            on=["season", "round", "circuit_id", "driver_id", "constructor"],
            how="outer",
        )

        # Merge with races
        merged = pd.merge(
            merged,
            races_df[["season", "round", "circuit_id", "race_name", "date", "locality", "country"]],
            on=["season", "round", "circuit_id"],
            how="left",
        )

        # Merge with drivers
        merged = pd.merge(
            merged, drivers_df, on="driver_id", how="left"
        )

        # Merge with circuits
        merged = pd.merge(
            merged,
            circuits_df[["circuit_id", "type", "direction", "length"]],
            on="circuit_id",
            how="left",
        )

        # Merge with weather
        if not weather_df.empty:
            merged = pd.merge(
                merged,
                weather_df[["season", "round", "weather"]],
                on=["season", "round"],
                how="left",
            )
        else:
            merged["weather"] = ""

        # Apply feature engineering
        pipeline = FeaturePipeline()
        main_df = pipeline.transform(merged)

        # Sort by season, round, finish position
        main_df = main_df.sort_values(
            by=["season", "round", "finish_position"]
        ).reset_index(drop=True)

        return main_df

    def _save_all(
        self,
        races_df: pd.DataFrame,
        results_df: pd.DataFrame,
        qualifying_df: pd.DataFrame,
        drivers_df: pd.DataFrame,
        circuits_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        main_df: pd.DataFrame,
    ) -> None:
        """Save all datasets to storage."""
        self.storage.save_dataframe(races_df, "races")
        self.storage.save_dataframe(results_df, "results")
        self.storage.save_dataframe(qualifying_df, "qualifying")
        self.storage.save_dataframe(drivers_df, "drivers")
        self.storage.save_dataframe(circuits_df, "circuits")
        if not weather_df.empty:
            self.storage.save_dataframe(weather_df, "weather")
        self.storage.save_dataframe(main_df, "main_df")

    def load_main_dataset(self) -> pd.DataFrame:
        """Load the main dataset from storage."""
        return self.storage.load_dataframe("main_df")
