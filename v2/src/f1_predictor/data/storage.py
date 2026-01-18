"""Storage backends for F1 data."""

import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd

from f1_predictor.core.config import get_settings
from f1_predictor.core.exceptions import StorageError
from f1_predictor.core.logging import get_logger

logger = get_logger(__name__)


class Storage(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def save_dataframe(self, df: pd.DataFrame, name: str) -> None:
        """Save a DataFrame to storage."""
        pass

    @abstractmethod
    def load_dataframe(self, name: str) -> pd.DataFrame:
        """Load a DataFrame from storage."""
        pass

    @abstractmethod
    def exists(self, name: str) -> bool:
        """Check if a dataset exists in storage."""
        pass

    @abstractmethod
    def list_datasets(self) -> list[str]:
        """List all available datasets."""
        pass


class SQLiteStorage(Storage):
    """SQLite storage backend."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        settings = get_settings().storage
        self.db_path = db_path or settings.data_dir / "f1_predictor.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            # Create metadata table for tracking datasets
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _datasets (
                    name TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(self.db_path)

    def save_dataframe(self, df: pd.DataFrame, name: str) -> None:
        """
        Save a DataFrame to SQLite.

        Args:
            df: DataFrame to save
            name: Table name
        """
        try:
            with self._get_connection() as conn:
                df.to_sql(name, conn, if_exists="replace", index=False)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO _datasets (name, updated_at)
                    VALUES (?, CURRENT_TIMESTAMP)
                """,
                    (name,),
                )
                conn.commit()
            logger.info(f"Saved {len(df)} rows to table '{name}'")
        except Exception as e:
            raise StorageError(f"Failed to save DataFrame: {e}") from e

    def load_dataframe(self, name: str) -> pd.DataFrame:
        """
        Load a DataFrame from SQLite.

        Args:
            name: Table name

        Returns:
            Loaded DataFrame
        """
        try:
            with self._get_connection() as conn:
                return pd.read_sql(f"SELECT * FROM {name}", conn)
        except Exception as e:
            raise StorageError(f"Failed to load DataFrame: {e}") from e

    def exists(self, name: str) -> bool:
        """Check if a table exists."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name=?
            """,
                (name,),
            )
            return cursor.fetchone() is not None

    def list_datasets(self) -> list[str]:
        """List all available tables."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT name FROM _datasets ORDER BY name
            """
            )
            return [row[0] for row in cursor.fetchall()]

    def execute_query(self, query: str, params: tuple = ()) -> pd.DataFrame:
        """Execute a custom SQL query."""
        with self._get_connection() as conn:
            return pd.read_sql(query, conn, params=params)


class CSVStorage(Storage):
    """CSV file storage backend."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        settings = get_settings().storage
        self.data_dir = data_dir or settings.data_dir / "csv"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, name: str) -> Path:
        """Get file path for a dataset."""
        return self.data_dir / f"{name}.csv"

    def save_dataframe(self, df: pd.DataFrame, name: str) -> None:
        """
        Save a DataFrame to CSV.

        Args:
            df: DataFrame to save
            name: Dataset name (without extension)
        """
        try:
            path = self._get_path(name)
            df.to_csv(path, index=False)
            logger.info(f"Saved {len(df)} rows to '{path}'")
        except Exception as e:
            raise StorageError(f"Failed to save DataFrame: {e}") from e

    def load_dataframe(self, name: str) -> pd.DataFrame:
        """
        Load a DataFrame from CSV.

        Args:
            name: Dataset name (without extension)

        Returns:
            Loaded DataFrame
        """
        try:
            path = self._get_path(name)
            return pd.read_csv(path)
        except FileNotFoundError as e:
            raise StorageError(f"Dataset '{name}' not found") from e
        except Exception as e:
            raise StorageError(f"Failed to load DataFrame: {e}") from e

    def exists(self, name: str) -> bool:
        """Check if a dataset file exists."""
        return self._get_path(name).exists()

    def list_datasets(self) -> list[str]:
        """List all available CSV files."""
        return [p.stem for p in self.data_dir.glob("*.csv")]


class ParquetStorage(Storage):
    """Parquet file storage backend for large datasets."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        settings = get_settings().storage
        self.data_dir = data_dir or settings.data_dir / "parquet"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, name: str) -> Path:
        """Get file path for a dataset."""
        return self.data_dir / f"{name}.parquet"

    def save_dataframe(self, df: pd.DataFrame, name: str) -> None:
        """
        Save a DataFrame to Parquet.

        Args:
            df: DataFrame to save
            name: Dataset name (without extension)
        """
        try:
            path = self._get_path(name)
            df.to_parquet(path, index=False)
            logger.info(f"Saved {len(df)} rows to '{path}'")
        except Exception as e:
            raise StorageError(f"Failed to save DataFrame: {e}") from e

    def load_dataframe(self, name: str) -> pd.DataFrame:
        """
        Load a DataFrame from Parquet.

        Args:
            name: Dataset name (without extension)

        Returns:
            Loaded DataFrame
        """
        try:
            path = self._get_path(name)
            return pd.read_parquet(path)
        except FileNotFoundError as e:
            raise StorageError(f"Dataset '{name}' not found") from e
        except Exception as e:
            raise StorageError(f"Failed to load DataFrame: {e}") from e

    def exists(self, name: str) -> bool:
        """Check if a dataset file exists."""
        return self._get_path(name).exists()

    def list_datasets(self) -> list[str]:
        """List all available Parquet files."""
        return [p.stem for p in self.data_dir.glob("*.parquet")]


def get_storage(backend: Optional[str] = None) -> Storage:
    """
    Get storage backend instance.

    Args:
        backend: Storage type ("sqlite", "csv", "parquet") or None for default

    Returns:
        Storage instance
    """
    settings = get_settings().storage
    backend = backend or settings.backend

    if backend == "sqlite":
        return SQLiteStorage()
    elif backend == "csv":
        return CSVStorage()
    elif backend == "parquet":
        return ParquetStorage()
    else:
        raise StorageError(f"Unknown storage backend: {backend}")
