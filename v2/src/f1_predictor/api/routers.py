"""API routes for F1 Predictor."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from f1_predictor import __version__
from f1_predictor.models.xgboost_model import F1XGBoostModel

router = APIRouter()


# Request/Response models
class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    model_loaded: bool
    model_version: Optional[str]


class PredictionRequest(BaseModel):
    """Request for race prediction."""

    season: int
    round: int


class StandingsRequest(BaseModel):
    """Request for season standings prediction."""

    season: int


class DriverStanding(BaseModel):
    """Driver standing entry."""

    position: int
    name: str
    total_points: int
    wins: int


class ConstructorStanding(BaseModel):
    """Constructor standing entry."""

    position: int
    constructor: str
    total_points: int
    wins: int


class StandingsResponse(BaseModel):
    """Response with championship standings."""

    season: int
    driver_standings: list[DriverStanding]
    constructor_standings: list[ConstructorStanding]


class ModelInfoResponse(BaseModel):
    """Model information response."""

    version: Optional[str]
    is_loaded: bool
    params: Optional[dict]


def get_model(request: Request) -> Optional[F1XGBoostModel]:
    """Dependency to get loaded model."""
    return getattr(request.app.state, "model", None)


# Routes
@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check(request: Request) -> HealthResponse:
    """Check API health status."""
    model = get_model(request)
    return HealthResponse(
        status="healthy",
        version=__version__,
        model_loaded=model is not None,
        model_version=getattr(request.app.state, "model_version", None),
    )


@router.get("/api/v1/model", response_model=ModelInfoResponse, tags=["models"])
async def get_model_info(request: Request) -> ModelInfoResponse:
    """Get information about the loaded model."""
    model = get_model(request)
    return ModelInfoResponse(
        version=getattr(request.app.state, "model_version", None),
        is_loaded=model is not None,
        params=model.get_params() if model else None,
    )


@router.get(
    "/api/v1/standings/{season}",
    response_model=StandingsResponse,
    tags=["predictions"],
)
async def get_standings(
    season: int,
    request: Request,
) -> StandingsResponse:
    """Get predicted championship standings for a season."""
    model = get_model(request)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        from f1_predictor.data.storage import get_storage
        from f1_predictor.postprocessing.positions import predictions_to_positions
        from f1_predictor.postprocessing.standings import (
            calculate_standings_from_predictions,
        )

        # Load data
        storage = get_storage()
        df = storage.load_dataframe("main_df")

        # Filter to season
        season_df = df[df["season"] == season].copy()

        if len(season_df) == 0:
            raise HTTPException(
                status_code=404, detail=f"No data found for season {season}"
            )

        # Prepare features
        X = season_df.drop(
            columns=[
                "filled_splits",
                "finish_position",
                "points",
                "status",
                "name",
                "constructor",
                "circuit_id",
            ],
            errors="ignore",
        )

        # Generate predictions
        predictions = model.predict(X)
        season_df["pred"] = predictions
        season_df["pred_position"] = predictions_to_positions(season_df)

        # Calculate standings
        driver_standings, constructor_standings = calculate_standings_from_predictions(
            season_df
        )

        return StandingsResponse(
            season=season,
            driver_standings=[
                DriverStanding(
                    position=int(row["position"]),
                    name=row["name"],
                    total_points=int(row["total_points"]),
                    wins=int(row["wins"]),
                )
                for _, row in driver_standings.head(20).iterrows()
            ],
            constructor_standings=[
                ConstructorStanding(
                    position=int(row["position"]),
                    constructor=row["constructor"],
                    total_points=int(row["total_points"]),
                    wins=int(row["wins"]),
                )
                for _, row in constructor_standings.iterrows()
            ],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/models", tags=["models"])
async def list_models():
    """List all available model versions."""
    from f1_predictor.models.registry import ModelRegistry

    registry = ModelRegistry()
    versions = registry.list_versions()

    return {
        "models": [
            {
                "version": v["version"],
                "created_at": v["created_at"],
                "is_active": v["is_active"],
                "metrics": v.get("metrics", {}),
            }
            for v in versions
        ]
    }


@router.post("/api/v1/models/{version}/activate", tags=["models"])
async def activate_model(version: str, request: Request):
    """Set a model version as active."""
    from f1_predictor.models.registry import ModelRegistry

    try:
        registry = ModelRegistry()
        registry.set_active(version)

        # Reload the model
        request.app.state.model = registry.load_model(version)
        request.app.state.model_version = version

        return {"status": "success", "active_version": version}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
