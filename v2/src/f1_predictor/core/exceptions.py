"""Custom exceptions for F1 Predictor."""


class F1PredictorError(Exception):
    """Base exception for F1 Predictor."""

    pass


class DataCollectionError(F1PredictorError):
    """Error during data collection."""

    pass


class ErgastAPIError(DataCollectionError):
    """Error from Ergast API."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ScrapingError(DataCollectionError):
    """Error during web scraping."""

    pass


class ModelError(F1PredictorError):
    """Error related to ML model operations."""

    pass


class ModelNotFittedError(ModelError):
    """Model has not been fitted yet."""

    pass


class ModelNotFoundError(ModelError):
    """Model file not found."""

    pass


class ValidationError(F1PredictorError):
    """Data validation error."""

    pass


class MissingDataError(ValidationError):
    """Required data is missing."""

    pass


class InvalidDataError(ValidationError):
    """Data format is invalid."""

    pass


class StorageError(F1PredictorError):
    """Error during data storage operations."""

    pass


class ConfigurationError(F1PredictorError):
    """Configuration error."""

    pass
