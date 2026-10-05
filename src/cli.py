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
    console.print(f"  MLflow:  {settings.MLFLOW_TRACKING_URI}  [{settings.mlflow_registry_scope}]")
    console.print(f"  Data:    {settings.data_mode}")

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


@cli.command("publish-triton")
@click.option("--version", default=None, help="Registry version to publish (e.g. 5)")
@click.option("--stage", default=None, help="Publish the highest version in this stage instead")
@click.option("--repo", default=None, help="Triton model repository (default TRITON_MODEL_REPOSITORY)")
@click.option("--allow-scratch-registry", is_flag=True,
              help="Publish from a local file/sqlite registry (development experiments only)")
def publish_triton(version: Optional[str], stage: Optional[str], repo: Optional[str],
                   allow_scratch_registry: bool):
    """Publish a registered model version into the Triton model repository."""
    from src.core.config import settings
    from src.registry.mlflow_registry import MODEL_NAME, configure_mlflow_environment
    from src.registry.triton_publisher import TritonPublishError, publish_registered_model

    configure_mlflow_environment()
    try:
        result = publish_registered_model(
            repository=repo or settings.TRITON_MODEL_REPOSITORY,
            model_name=MODEL_NAME, version=version, stage=stage,
            allow_scratch_registry=allow_scratch_registry,
        )
    except TritonPublishError as exc:
        console.print(f"[bold red]Publish refused:[/bold red] {exc}")
        sys.exit(2)
    _print_publish(result)


def _print_publish(result):
    state = "already published (identical artifact)" if result.already_published else "published"
    console.print(f"[bold green]{result.registry_model_name} v{result.registry_version} {state}[/bold green]")
    for key in ("registry_uri", "registry_stage", "triton_model_name", "triton_version",
                "serving_model_version", "mlflow_run_id", "model_dir", "onnx_sha256",
                "feature_catalog_version", "dataset_version", "dataset_id"):
        console.print(f"  {key:24s} {getattr(result, key)}")


@cli.command()
@click.option("--version", required=True, help="Registry version to release (e.g. 6)")
@click.option("--repo", default=None, help="Triton model repository (default TRITON_MODEL_REPOSITORY)")
def release(version: str, repo: Optional[str]):
    """
    The ONE promotion path: promote a version to Production in the MLflow server
    registry, then publish exactly that version to Triton.

    If publishing is refused the version is returned to its previous stage, so the
    registry never says "Production" for a model Triton does not have. The serving
    instance then selects it with CHAMPION_MODEL_VERSION=v<version>.
    """
    from mlflow import MlflowClient
    from src.core.config import settings
    from src.registry.mlflow_registry import MODEL_NAME, configure_mlflow_environment
    from src.registry.triton_publisher import TritonPublishError, publish_registered_model

    if configure_mlflow_environment() != "server":
        console.print("[bold red]Release refused:[/bold red] the registry is a local scratch "
                      "store; releases are made from the MLflow server registry only.")
        sys.exit(2)
    client = MlflowClient()
    try:
        mv = client.get_model_version(MODEL_NAME, str(version))
    except Exception as exc:
        console.print(f"[bold red]Release refused:[/bold red] {MODEL_NAME} v{version} not found: {exc}")
        sys.exit(2)
    previous_stage = mv.current_stage or "None"
    previous_production = [m.version for m in client.search_model_versions(f"name='{MODEL_NAME}'")
                           if m.current_stage == "Production" and m.version != mv.version]

    client.transition_model_version_stage(MODEL_NAME, mv.version, "Production",
                                          archive_existing_versions=True)
    try:
        result = publish_registered_model(
            repository=repo or settings.TRITON_MODEL_REPOSITORY,
            model_name=MODEL_NAME, version=str(version), require_stage="Production",
        )
    except TritonPublishError as exc:
        client.transition_model_version_stage(MODEL_NAME, mv.version, previous_stage)
        for v in previous_production:
            client.transition_model_version_stage(MODEL_NAME, v, "Production")
        console.print(f"[bold red]Release refused:[/bold red] {exc}")
        console.print(f"  v{version} returned to stage {previous_stage}")
        sys.exit(2)
    console.print(f"[bold green]Released {MODEL_NAME} v{version}: Production in the registry and "
                  f"published to Triton[/bold green]")
    _print_publish(result)
    console.print(f"  serve it with            CHAMPION_MODEL_VERSION=v{version}")


@cli.command()
@click.option("--version", required=True, help="Registry version to evaluate")
@click.option("--data-dir", default=None, help="Seed corpus directory (default SYNTHETIC_DATA_DIR)")
@click.option("--out", default=None, help="Also write the report to this JSON file")
def benchmark(version: str, data_dir: Optional[str], out: Optional[str]):
    """Score a registry version on the frozen benchmark and report decision-tier metrics."""
    import json
    from src.core.config import settings
    from src.evaluators.benchmark import DECISION_TIERS, BenchmarkError, run_benchmark
    from src.registry.mlflow_registry import MODEL_NAME, configure_mlflow_environment

    configure_mlflow_environment()
    data_dir = data_dir or settings.SYNTHETIC_DATA_DIR
    if not data_dir:
        console.print("[bold red]No dataset directory (SYNTHETIC_DATA_DIR / --data-dir)[/bold red]")
        sys.exit(2)
    try:
        report = asyncio.run(run_benchmark(str(version), data_dir, MODEL_NAME))
    except BenchmarkError as exc:
        console.print(f"[bold red]Benchmark failed:[/bold red] {exc}")
        sys.exit(2)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    console.print(f"[bold]{MODEL_NAME} v{version}[/bold]  benchmark {report['benchmark_id']}  "
                  f"dataset {report['dataset_id']}")
    console.print(f"  pairs {report['pairs']}  true matches {report['true_matches']}  "
                  f"true non-matches {report['true_non_matches']}   [SYNTHETIC bootstrap data]")
    table = Table(title="Decision tiers")
    for col in ("Tier", "Score >=", "Pairs", "True matches", "True non-matches", "Match rate"):
        table.add_column(col)
    for name, lower in DECISION_TIERS:
        t = report["tiers"][name]
        table.add_row(name, f"{lower:.2f}", str(t["pairs"]), str(t["true_matches"]),
                      str(t["true_non_matches"]), "-" if t["match_rate"] is None else f"{t['match_rate']:.4f}")
    console.print(table)
    am, anm, rv = report["auto_match"], report["auto_non_match"], report["review"]
    console.print(f"  AUTO_MATCH      precision {am['precision']}  recall {am['recall']}  "
                  f"wrong merges {am['wrong_merges']} of {am['pairs']}")
    console.print(f"  AUTO_NON_MATCH  precision {anm['precision']}  recall {anm['recall']}  "
                  f"missed matches {anm['missed_matches']} of {anm['pairs']}")
    console.print(f"  human review    {rv['pairs']} pairs ({rv['share_of_pairs']})")
    console.print(f"  at 0.5          {report['at_threshold_0_5']}")


@cli.command()
@click.option("--version", required=True, help="Registry version to trace")
@click.option("--data-dir", default=None, help="Dataset directory to verify (default SYNTHETIC_DATA_DIR)")
@click.option("--repo", default=None, help="Triton model repository (default TRITON_MODEL_REPOSITORY)")
@click.option("--triton-url", default=None, help="Running Triton HTTP URL, e.g. http://127.0.0.1:8000")
@click.option("--inference-url", default=None, help="Running inference service, e.g. http://127.0.0.1:8090")
@click.option("--json-out", default=None, help="Write the trace to this JSON file")
def lineage(version: str, data_dir: Optional[str], repo: Optional[str], triton_url: Optional[str],
            inference_url: Optional[str], json_out: Optional[str]):
    """Trace and verify: dataset -> training run -> model version -> ONNX -> Triton -> inference."""
    import json
    from src.core.config import settings
    from src.registry.lineage import OK, trace_lineage
    from src.registry.mlflow_registry import MODEL_NAME, configure_mlflow_environment

    configure_mlflow_environment()
    trace = trace_lineage(
        str(version), MODEL_NAME,
        data_dir=data_dir or settings.SYNTHETIC_DATA_DIR,
        triton_repository=repo or settings.TRITON_MODEL_REPOSITORY,
        triton_url=triton_url, inference_url=inference_url,
    )
    for hop in trace.hops:
        colour = "green" if hop.status == OK else ("red" if hop.status == "BROKEN" else "yellow")
        console.print(f"[bold {colour}]{hop.name:18s} {hop.status}[/bold {colour}]")
        for k, v in hop.facts.items():
            console.print(f"    {k:34s} {v}")
        for problem in hop.problems:
            console.print(f"    [{colour}]! {problem}[/{colour}]")
    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(trace.as_dict(), f, indent=2)
    if trace.broken:
        console.print("[bold red]LINEAGE BROKEN[/bold red]")
        sys.exit(1)
    console.print("[bold green]LINEAGE COMPLETE[/bold green]" if trace.complete
                  else "[bold yellow]LINEAGE INCOMPLETE (hops skipped or not recorded)[/bold yellow]")
    sys.exit(0 if trace.complete else 3)


@cli.command("migrate-registry")
@click.option("--from", "source", required=True, help="Source scratch store, e.g. sqlite:///D:/.../mlflow.db")
def migrate_registry_cmd(source: str):
    """Copy registered versions from a local scratch store into the MLflow server registry."""
    from src.core.config import settings
    from src.registry.migrate import RegistryMigrationError, migrate_registry
    from src.registry.mlflow_registry import MODEL_NAME, configure_mlflow_environment, registry_uri

    if configure_mlflow_environment() != "server":
        console.print("[bold red]Refused:[/bold red] MLFLOW_TRACKING_URI must be the MLflow server.")
        sys.exit(2)
    location = settings.MLFLOW_ARTIFACT_LOCATION
    if location.lower().startswith("file:"):
        location = None       # a client-local path means nothing to the server
    try:
        done = migrate_registry(source, registry_uri(), MODEL_NAME,
                                settings.MLFLOW_EXPERIMENT_NAME, artifact_location=location)
    except RegistryMigrationError as exc:
        console.print(f"[bold red]Migration refused:[/bold red] {exc}")
        sys.exit(2)
    for m in done:
        state = "already migrated" if m.skipped else "migrated"
        console.print(f"  v{m.version}  {state}  {m.source_run_id} -> {m.target_run_id}  [{m.stage}]")


if __name__ == "__main__":
    cli()
