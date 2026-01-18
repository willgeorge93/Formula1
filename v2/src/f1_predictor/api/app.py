"""FastAPI application for F1 predictions."""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from f1_predictor import __version__
from f1_predictor.api.routers import router
from f1_predictor.core.config import get_settings
from f1_predictor.core.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting F1 Predictor API")
    try:
        from f1_predictor.models.registry import ModelRegistry

        registry = ModelRegistry()
        app.state.model = registry.load_model()
        app.state.model_version = registry.get_active_version()
        logger.info(f"Loaded model version: {app.state.model_version}")
    except Exception as e:
        logger.warning(f"Could not load model: {e}")
        app.state.model = None
        app.state.model_version = None

    yield

    # Shutdown
    logger.info("Shutting down F1 Predictor API")
    app.state.model = None


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="F1 Predictor API",
        description="Formula 1 Race Prediction Service",
        version=__version__,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(router)

    return app


app = create_app()
