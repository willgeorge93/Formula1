"""Logging configuration for F1 Predictor."""

import logging
import sys
from typing import Optional

from f1_predictor.core.config import get_settings


def setup_logging(
    level: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages

    Returns:
        Configured logger instance
    """
    settings = get_settings()
    log_level = level or settings.log_level

    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Create package logger
    logger = logging.getLogger("f1_predictor")
    logger.setLevel(getattr(logging, log_level.upper()))

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Name of the module (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"f1_predictor.{name}")
