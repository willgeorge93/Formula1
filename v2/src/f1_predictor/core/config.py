"""Configuration management using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ErgastSettings(BaseSettings):
    """Ergast API configuration."""

    model_config = SettingsConfigDict(env_prefix="F1_ERGAST__")

    base_url: str = "https://ergast.com/api/f1"
    rate_limit: float = 0.25  # seconds between requests
    timeout: int = 30
    retries: int = 3


class ScrapingSettings(BaseSettings):
    """Web scraping configuration."""

    model_config = SettingsConfigDict(env_prefix="F1_SCRAPING__")

    wikipedia_enabled: bool = True
    f1fansite_enabled: bool = True
    f1fansite_urls: list[str] = [
        "https://www.f1-fansite.com/f1-result/race-results-{year}-{race}-f1-grand-prix/",
        "https://www.f1-fansite.com/f1-result/race-results-{year}-{race}-f1-gp/",
        "https://www.f1-fansite.com/f1-result/race-result-{year}-{race}-f1-gp/",
        "https://www.f1-fansite.com/f1-result/{year}-{race}-grand-prix-race-results/",
        "https://www.f1-fansite.com/f1-result/{year}-{race}-grand-prix-results/",
    ]
    user_agent: str = "F1Predictor/2.0"


class ModelSettings(BaseSettings):
    """ML model configuration with tuned hyperparameters."""

    model_config = SettingsConfigDict(env_prefix="F1_MODEL__")

    # XGBoost tuned parameters (from 96,040 combinations tested)
    xgb_gamma: float = 0.1
    xgb_learning_rate: float = 0.2
    xgb_max_depth: int = 6
    xgb_n_estimators: int = 150
    xgb_reg_lambda: float = 0.2
    xgb_subsample: float = 1.0

    # Training settings
    test_season: int = 2020  # Season to use for testing
    train_seasons_start: int = 2014
    random_state: int = 42


class StorageSettings(BaseSettings):
    """Data storage configuration."""

    model_config = SettingsConfigDict(env_prefix="F1_STORAGE__")

    backend: Literal["sqlite", "csv", "parquet"] = "sqlite"
    data_dir: Path = Path("data")
    models_dir: Path = Path("data/models")
    cache_enabled: bool = True
    cache_ttl: int = 86400  # 24 hours


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_prefix="F1_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = "F1 Predictor"
    debug: bool = False
    log_level: str = "INFO"

    ergast: ErgastSettings = Field(default_factory=ErgastSettings)
    scraping: ScrapingSettings = Field(default_factory=ScrapingSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)


@lru_cache
def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()
