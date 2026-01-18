"""CLI application using Typer."""

from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from f1_predictor import __version__
from f1_predictor.core.config import get_settings
from f1_predictor.core.logging import setup_logging

app = typer.Typer(
    name="f1-predictor",
    help="Formula 1 Race Prediction CLI",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()


@app.callback(invoke_without_command=True)
def callback(
    ctx: typer.Context,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    """F1 Predictor - Formula 1 Race Prediction using Machine Learning."""
    if verbose:
        setup_logging(level="DEBUG")
    else:
        setup_logging()

    if ctx.invoked_subcommand is None:
        console.print(f"[bold blue]F1 Predictor[/bold blue] version {__version__}")
        console.print("\nUse [bold]f1-predictor --help[/bold] to see available commands.")


@app.command()
def version() -> None:
    """Show version information."""
    console.print(f"[bold blue]F1 Predictor[/bold blue] version {__version__}")


@app.command()
def init(
    data_dir: Annotated[str, typer.Option(help="Data directory path")] = "data",
) -> None:
    """Initialize the project structure."""
    base_path = Path(data_dir)
    directories = [
        base_path / "raw",
        base_path / "processed",
        base_path / "models",
        base_path / "cache",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]Created[/green] {directory}")

    console.print("\n[bold green]Initialization complete![/bold green]")


@app.command()
def collect(
    start_year: Annotated[int, typer.Option(help="Start year for data collection")] = 2014,
    end_year: Annotated[Optional[int], typer.Option(help="End year (default: current year)")] = None,
    no_weather: Annotated[bool, typer.Option(help="Skip weather data collection")] = False,
) -> None:
    """Collect F1 data from Ergast API and web sources."""
    from f1_predictor.data.collectors import DataCollector

    if end_year is None:
        end_year = datetime.now().year

    console.print(f"[bold]Collecting F1 data for seasons {start_year}-{end_year}[/bold]")

    collector = DataCollector()
    collector.collect_all(
        start_year=start_year,
        end_year=end_year,
        include_weather=not no_weather,
    )

    console.print("[bold green]Data collection complete![/bold green]")


@app.command()
def train(
    test_season: Annotated[Optional[int], typer.Option(help="Season to use for testing")] = None,
    output_dir: Annotated[Optional[str], typer.Option(help="Directory to save model")] = None,
) -> None:
    """Train the prediction model."""
    from f1_predictor.models.training import ModelTrainer

    settings = get_settings()
    test_season = test_season or settings.model.test_season

    console.print(f"[bold]Training model with test season {test_season}[/bold]")

    trainer = ModelTrainer()
    results = trainer.train(test_season=test_season)

    # Display results
    table = Table(title="Training Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Training R²", f"{results['train_r2']:.4f}")
    table.add_row("Test R²", f"{results['test_r2']:.4f}")
    table.add_row("Test RMSE", f"{results['test_metrics']['rmse']:.3f}")
    table.add_row("Test Pearson", f"{results['test_metrics']['pearson']:.4f}")
    table.add_row("Model Version", results.get("model_version", "N/A"))

    console.print(table)
    console.print("[bold green]Training complete![/bold green]")


@app.command()
def predict(
    season: Annotated[int, typer.Argument(help="Season to predict")],
    output: Annotated[Optional[str], typer.Option(help="Output file for predictions")] = None,
) -> None:
    """Generate predictions for a season."""
    from f1_predictor.data.storage import get_storage
    from f1_predictor.models.registry import ModelRegistry
    from f1_predictor.postprocessing.positions import predictions_to_positions
    from f1_predictor.postprocessing.standings import calculate_standings_from_predictions

    console.print(f"[bold]Generating predictions for {season} season[/bold]")

    # Load model
    registry = ModelRegistry()
    model = registry.load_model()

    # Load data
    storage = get_storage()
    df = storage.load_dataframe("main_df")

    # Filter to season
    season_df = df[df["season"] == season].copy()

    if len(season_df) == 0:
        console.print(f"[red]No data found for season {season}[/red]")
        raise typer.Exit(1)

    # Prepare features (remove target columns)
    X = season_df.drop(
        columns=["filled_splits", "finish_position", "points", "status", "name", "constructor", "circuit_id"],
        errors="ignore",
    )

    # Generate predictions
    predictions = model.predict(X)
    season_df["pred"] = predictions
    season_df["pred_position"] = predictions_to_positions(season_df)

    # Calculate standings
    driver_standings, constructor_standings = calculate_standings_from_predictions(season_df)

    # Display results
    console.print("\n[bold cyan]Predicted Driver Standings[/bold cyan]")
    driver_table = Table()
    driver_table.add_column("Pos", style="cyan")
    driver_table.add_column("Driver", style="white")
    driver_table.add_column("Points", style="green")

    for _, row in driver_standings.head(10).iterrows():
        driver_table.add_row(
            str(int(row["position"])),
            row["name"],
            str(int(row["total_points"])),
        )

    console.print(driver_table)

    console.print("\n[bold cyan]Predicted Constructor Standings[/bold cyan]")
    constructor_table = Table()
    constructor_table.add_column("Pos", style="cyan")
    constructor_table.add_column("Constructor", style="white")
    constructor_table.add_column("Points", style="green")

    for _, row in constructor_standings.iterrows():
        constructor_table.add_row(
            str(int(row["position"])),
            row["constructor"],
            str(int(row["total_points"])),
        )

    console.print(constructor_table)

    # Save to file if requested
    if output:
        driver_standings.to_csv(output, index=False)
        console.print(f"\n[green]Saved predictions to {output}[/green]")


@app.command()
def evaluate(
    model_version: Annotated[Optional[str], typer.Option(help="Model version to evaluate")] = None,
    test_season: Annotated[Optional[int], typer.Option(help="Season to evaluate on")] = None,
) -> None:
    """Evaluate model performance."""
    from f1_predictor.data.storage import get_storage
    from f1_predictor.evaluation.metrics import calculate_metrics
    from f1_predictor.features.pipeline import FeaturePipeline
    from f1_predictor.models.registry import ModelRegistry

    settings = get_settings()
    test_season = test_season or settings.model.test_season

    console.print(f"[bold]Evaluating model on {test_season} season[/bold]")

    # Load model
    registry = ModelRegistry()
    model = registry.load_model(version=model_version)

    # Load and prepare data
    storage = get_storage()
    df = storage.load_dataframe("main_df")

    pipeline = FeaturePipeline()
    _, X_test, _, y_test = pipeline.prepare_for_training(df, test_season=test_season)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)

    # Display results
    table = Table(title=f"Evaluation Results (Season {test_season})")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Spearman Correlation", f"{metrics.spearman:.4f}")
    table.add_row("Pearson Correlation", f"{metrics.pearson:.4f}")
    table.add_row("R² Score", f"{metrics.r2:.4f}")
    table.add_row("RMSE", f"{metrics.rmse:.3f}")

    console.print(table)


@app.command()
def models() -> None:
    """List available model versions."""
    from f1_predictor.models.registry import ModelRegistry

    registry = ModelRegistry()
    versions = registry.list_versions()

    if not versions:
        console.print("[yellow]No models registered[/yellow]")
        return

    table = Table(title="Registered Models")
    table.add_column("Version", style="cyan")
    table.add_column("Created", style="white")
    table.add_column("Active", style="green")
    table.add_column("Test R²", style="yellow")

    for v in versions:
        active = "Y" if v["is_active"] else ""
        test_r2 = v.get("metrics", {}).get("test_r2", "N/A")
        if isinstance(test_r2, float):
            test_r2 = f"{test_r2:.4f}"

        table.add_row(
            v["version"],
            v["created_at"][:19],
            active,
            str(test_r2),
        )

    console.print(table)


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Host to bind to")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="Port to bind to")] = 8000,
    reload: Annotated[bool, typer.Option(help="Enable auto-reload for development")] = False,
) -> None:
    """Start the API server."""
    import uvicorn

    console.print(f"[bold]Starting API server on {host}:{port}[/bold]")
    uvicorn.run(
        "f1_predictor.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
