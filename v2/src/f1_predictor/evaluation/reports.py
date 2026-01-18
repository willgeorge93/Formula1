"""Evaluation report generation."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from f1_predictor.evaluation.metrics import (
    EvaluationMetrics,
    calculate_all_tolerances,
    calculate_metrics,
)


@dataclass
class EvaluationReport:
    """Container for evaluation report data."""

    model_version: str
    test_season: int
    created_at: datetime
    raw_metrics: EvaluationMetrics
    position_metrics: dict[str, float]
    driver_standings_metrics: Optional[EvaluationMetrics] = None
    constructor_standings_metrics: Optional[EvaluationMetrics] = None
    driver_standings: Optional[pd.DataFrame] = None
    constructor_standings: Optional[pd.DataFrame] = None
    additional_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "model_version": self.model_version,
            "test_season": self.test_season,
            "created_at": self.created_at.isoformat(),
            "raw_metrics": self.raw_metrics.to_dict(),
            "position_metrics": self.position_metrics,
            "driver_standings_metrics": (
                self.driver_standings_metrics.to_dict()
                if self.driver_standings_metrics
                else None
            ),
            "constructor_standings_metrics": (
                self.constructor_standings_metrics.to_dict()
                if self.constructor_standings_metrics
                else None
            ),
            "additional_info": self.additional_info,
        }

    def summary(self) -> str:
        """Generate a text summary of the report."""
        lines = [
            f"Evaluation Report - Model {self.model_version}",
            f"Test Season: {self.test_season}",
            f"Generated: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Raw Prediction Metrics:",
            f"  {self.raw_metrics}",
            "",
            "Position Prediction Accuracy:",
            f"  Exact match: {self.position_metrics.get('exact', 0):.1%}",
            f"  Within ±1: {self.position_metrics.get('within_1', 0):.1%}",
            f"  Within ±2: {self.position_metrics.get('within_2', 0):.1%}",
            f"  Within ±3: {self.position_metrics.get('within_3', 0):.1%}",
        ]

        if self.driver_standings_metrics:
            lines.extend(
                [
                    "",
                    "Driver Standings Metrics:",
                    f"  {self.driver_standings_metrics}",
                ]
            )

        if self.constructor_standings_metrics:
            lines.extend(
                [
                    "",
                    "Constructor Standings Metrics:",
                    f"  {self.constructor_standings_metrics}",
                ]
            )

        if self.driver_standings is not None:
            lines.extend(
                [
                    "",
                    "Predicted Driver Standings (Top 5):",
                ]
            )
            for _, row in self.driver_standings.head(5).iterrows():
                lines.append(
                    f"  {int(row['position'])}. {row['name']} - "
                    f"{int(row['total_points'])} pts"
                )

        if self.constructor_standings is not None:
            lines.extend(
                [
                    "",
                    "Predicted Constructor Standings:",
                ]
            )
            for _, row in self.constructor_standings.iterrows():
                lines.append(
                    f"  {int(row['position'])}. {row['constructor']} - "
                    f"{int(row['total_points'])} pts"
                )

        return "\n".join(lines)


def generate_report(
    y_true: pd.Series,
    y_pred: pd.Series,
    true_positions: pd.Series,
    pred_positions: pd.Series,
    model_version: str,
    test_season: int,
    pred_driver_standings: Optional[pd.DataFrame] = None,
    true_driver_standings: Optional[pd.DataFrame] = None,
    pred_constructor_standings: Optional[pd.DataFrame] = None,
    true_constructor_standings: Optional[pd.DataFrame] = None,
) -> EvaluationReport:
    """
    Generate a comprehensive evaluation report.

    Args:
        y_true: True target values (filled_splits)
        y_pred: Predicted target values
        true_positions: Actual race positions
        pred_positions: Predicted race positions
        model_version: Model version string
        test_season: Season used for testing
        pred_driver_standings: Predicted driver standings
        true_driver_standings: Actual driver standings
        pred_constructor_standings: Predicted constructor standings
        true_constructor_standings: Actual constructor standings

    Returns:
        EvaluationReport object
    """
    # Calculate raw prediction metrics
    raw_metrics = calculate_metrics(y_true, y_pred)

    # Calculate position accuracy metrics
    position_metrics = calculate_all_tolerances(true_positions, pred_positions)

    # Calculate standings metrics if available
    driver_standings_metrics = None
    constructor_standings_metrics = None

    if (
        pred_driver_standings is not None
        and true_driver_standings is not None
    ):
        driver_standings_metrics = calculate_metrics(
            true_driver_standings["total_points"],
            pred_driver_standings["total_points"],
        )

    if (
        pred_constructor_standings is not None
        and true_constructor_standings is not None
    ):
        constructor_standings_metrics = calculate_metrics(
            true_constructor_standings["total_points"],
            pred_constructor_standings["total_points"],
        )

    return EvaluationReport(
        model_version=model_version,
        test_season=test_season,
        created_at=datetime.now(),
        raw_metrics=raw_metrics,
        position_metrics=position_metrics,
        driver_standings_metrics=driver_standings_metrics,
        constructor_standings_metrics=constructor_standings_metrics,
        driver_standings=pred_driver_standings,
        constructor_standings=pred_constructor_standings,
    )


def print_report(report: EvaluationReport) -> None:
    """Print a formatted evaluation report."""
    print(report.summary())
