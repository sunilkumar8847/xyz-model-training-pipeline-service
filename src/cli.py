"""
model-training-pipeline/src/cli.py
CLI commands for training, evaluation, and model promotion.
Usage:
  python -m src.cli train --trigger MANUAL
  python -m src.cli evaluate --run-id <uuid>
  python -m src.cli promote --version 5 --stage production
"""
from __future__ import annotations

import asyncio
import logging
import sys
from typing import Optional
from uuid import UUID

import click
from rich.console import Console
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """XYZ MDM Model Training Pipeline CLI"""
    pass


@cli.command()
@click.option("--trigger", default="MANUAL", help="MANUAL|SCHEDULED|DRIFT_DETECTED|EMERGENCY")
@click.option("--reason", default="CLI trigger", help="Reason for training run")
@click.option("--tune-weights", is_flag=True, help="Use Optuna to tune ensemble weights")
def train(trigger: str, reason: str, tune_weights: bool):
    """Launch a new training pipeline run."""
    asyncio.run(_train_async(trigger, reason, tune_weights))


async def _train_async(trigger: str, reason: str, tune_weights: bool):
    from src.core.config import settings
    from src.domain.models import RetrainingTrigger, TrainingRun
    from src.pipeline.training_pipeline import TrainingPipeline
    from src.repositories.training_run_repository import get_session_factory

    console.print(f"[bold green]Starting training run...[/bold green]")
    console.print(f"  Trigger: {trigger}")
    console.print(f"  Reason:  {reason}")
    console.print(f"  MLflow:  {settings.MLFLOW_TRACKING_URI}")

    async with get_session_factory()() as session:
        run = TrainingRun(
            trigger=RetrainingTrigger(trigger),
            triggered_by=f"cli:{reason}",
        )
        pipeline = TrainingPipeline(db_session=session)
        run = await pipeline.run(run, tune_weights=tune_weights)

    _print_run_summary(run)


@cli.command()
@click.option("--version", required=True, help="MLflow model version")
@click.option("--stage", required=True, type=click.Choice(["staging", "canary", "production"]))
def promote(version: str, stage: str):
    """Promote a model version to a deployment stage."""
    from src.registry.mlflow_registry import MLflowModelRegistry
    registry = MLflowModelRegistry()

    if stage == "staging":
        success = registry.promote_to_staging(version)
    else:
        success = registry.promote_to_production(version)

    if success:
        console.print(f"[bold green]✓[/bold green] Model v{version} promoted to {stage}")
    else:
        console.print(f"[bold red]✗[/bold red] Promotion failed")
        sys.exit(1)


@cli.command()
@click.option("--reason", required=True, help="Reason for rollback")
def rollback(reason: str):
    """Emergency rollback to previous champion."""
    from src.registry.mlflow_registry import MLflowModelRegistry
    registry = MLflowModelRegistry()
    version = registry.rollback(reason)

    if version:
        console.print(f"[bold green]✓[/bold green] Rolled back to version {version}")
    else:
        console.print(f"[bold red]✗[/bold red] Rollback failed — no previous version available")
        sys.exit(1)


@cli.command()
def list_models():
    """List all registered model versions."""
    from src.registry.mlflow_registry import MLflowModelRegistry
    registry = MLflowModelRegistry()
    versions = registry.list_versions()

    table = Table(title="Registered Models")
    table.add_column("Version", style="cyan")
    table.add_column("Stage", style="magenta")
    table.add_column("F1 Score", style="green")
    table.add_column("Run ID")
    table.add_column("Created")

    for v in versions:
        table.add_row(
            v["version"],
            v["stage"],
            v.get("f1_score") or "N/A",
            (v.get("run_id") or "")[:8],
            str(v.get("created_at") or ""),
        )

    console.print(table)


def _print_run_summary(run):
    from rich.panel import Panel
    status_color = "green" if run.status.value == "COMPLETED" else "red"
    console.print(Panel(
        f"[bold]Run ID:[/bold] {run.run_id}\n"
        f"[bold]Status:[/bold] [{status_color}]{run.status.value}[/{status_color}]\n"
        f"[bold]Training Pairs:[/bold] {run.n_training_pairs}\n"
        f"[bold]Ensemble F1:[/bold] {run.ensemble_f1 or 'N/A'}\n"
        f"[bold]Model Version:[/bold] {run.model_version or 'Not registered'}\n"
        f"[bold]Error:[/bold] {run.error_message or 'None'}",
        title="Training Run Summary",
    ))


if __name__ == "__main__":
    cli()
