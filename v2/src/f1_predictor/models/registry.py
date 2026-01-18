"""Model versioning and registry."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from f1_predictor.core.config import get_settings
from f1_predictor.core.exceptions import ModelNotFoundError
from f1_predictor.core.logging import get_logger
from f1_predictor.models.xgboost_model import F1XGBoostModel

logger = get_logger(__name__)


class ModelRegistry:
    """
    Registry for managing model versions.

    Provides versioning, loading, and metadata management for
    trained models.
    """

    def __init__(self, models_dir: Optional[Path] = None) -> None:
        """
        Initialize the registry.

        Args:
            models_dir: Directory for storing models (default from config)
        """
        settings = get_settings().storage
        self.models_dir = models_dir or settings.models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_file = self.models_dir / "registry.json"
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """Load registry metadata from file."""
        if self._metadata_file.exists():
            with open(self._metadata_file) as f:
                return json.load(f)
        return {"models": {}, "active": None}

    def _save_metadata(self) -> None:
        """Save registry metadata to file."""
        with open(self._metadata_file, "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)

    def save_model(
        self,
        model: F1XGBoostModel,
        version: Optional[str] = None,
        metrics: Optional[dict[str, Any]] = None,
        set_active: bool = False,
    ) -> str:
        """
        Save a model with versioning.

        Args:
            model: Model to save
            version: Version string (default: timestamp-based)
            metrics: Training metrics to store
            set_active: Whether to set as active model

        Returns:
            Version string
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_path = self.models_dir / f"model_{version}.pkl"
        model.save(model_path)

        # Store metadata
        self._metadata["models"][version] = {
            "path": str(model_path),
            "created_at": datetime.now().isoformat(),
            "params": model.get_params(),
            "metrics": metrics or {},
        }

        if set_active:
            self._metadata["active"] = version

        self._save_metadata()
        logger.info(f"Registered model version: {version}")

        return version

    def load_model(self, version: Optional[str] = None) -> F1XGBoostModel:
        """
        Load a model by version.

        Args:
            version: Version to load (default: active model)

        Returns:
            Loaded model

        Raises:
            ModelNotFoundError: If version not found
        """
        if version is None:
            version = self._metadata.get("active")

        if version is None:
            raise ModelNotFoundError("No active model and no version specified")

        if version not in self._metadata["models"]:
            raise ModelNotFoundError(f"Model version not found: {version}")

        model_info = self._metadata["models"][version]
        model_path = Path(model_info["path"])

        return F1XGBoostModel.load(model_path)

    def get_active_version(self) -> Optional[str]:
        """Get the active model version."""
        return self._metadata.get("active")

    def set_active(self, version: str) -> None:
        """
        Set the active model version.

        Args:
            version: Version to set as active

        Raises:
            ModelNotFoundError: If version not found
        """
        if version not in self._metadata["models"]:
            raise ModelNotFoundError(f"Model version not found: {version}")

        self._metadata["active"] = version
        self._save_metadata()
        logger.info(f"Set active model: {version}")

    def list_versions(self) -> list[dict]:
        """
        List all registered model versions.

        Returns:
            List of model info dictionaries
        """
        versions = []
        for version, info in self._metadata["models"].items():
            versions.append(
                {
                    "version": version,
                    "created_at": info["created_at"],
                    "is_active": version == self._metadata.get("active"),
                    "metrics": info.get("metrics", {}),
                }
            )

        # Sort by creation date (newest first)
        versions.sort(key=lambda x: x["created_at"], reverse=True)
        return versions

    def get_model_info(self, version: str) -> dict:
        """
        Get information about a specific model version.

        Args:
            version: Version to query

        Returns:
            Model info dictionary

        Raises:
            ModelNotFoundError: If version not found
        """
        if version not in self._metadata["models"]:
            raise ModelNotFoundError(f"Model version not found: {version}")

        info = self._metadata["models"][version]
        return {
            "version": version,
            "is_active": version == self._metadata.get("active"),
            **info,
        }

    def delete_version(self, version: str) -> None:
        """
        Delete a model version.

        Args:
            version: Version to delete

        Raises:
            ModelNotFoundError: If version not found
        """
        if version not in self._metadata["models"]:
            raise ModelNotFoundError(f"Model version not found: {version}")

        model_info = self._metadata["models"][version]
        model_path = Path(model_info["path"])

        # Delete file
        if model_path.exists():
            model_path.unlink()

        # Remove from registry
        del self._metadata["models"][version]

        # Clear active if it was this version
        if self._metadata.get("active") == version:
            self._metadata["active"] = None

        self._save_metadata()
        logger.info(f"Deleted model version: {version}")

    def cleanup_old_versions(self, keep: int = 5) -> list[str]:
        """
        Remove old model versions, keeping the most recent.

        Args:
            keep: Number of versions to keep

        Returns:
            List of deleted version strings
        """
        versions = self.list_versions()

        # Keep the most recent N versions and the active version
        active = self._metadata.get("active")
        to_delete = []

        for i, v in enumerate(versions):
            if i >= keep and v["version"] != active:
                to_delete.append(v["version"])

        for version in to_delete:
            self.delete_version(version)

        return to_delete
