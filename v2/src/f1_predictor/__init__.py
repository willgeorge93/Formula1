"""
F1 Predictor - Formula 1 Race Prediction using Machine Learning.

A production-ready package for predicting F1 race results, driver standings,
and constructor standings using XGBoost regression trained on historical data.
"""

__version__ = "2.0.0"
__author__ = "William George"

from f1_predictor.core.config import Settings, get_settings

__all__ = ["Settings", "get_settings", "__version__"]
