"""
model-training-pipeline/src/registry/migrate.py

One-off migration of registered model versions from a LOCAL SCRATCH MLflow store
(file/sqlite) into the authoritative MLflow SERVER registry.

Why: models were being registered in a sqlite file on a developer machine while the
MLflow server stayed empty, so "version 5" existed in one place only and the next
server registration would have been "version 1" — older than the model Triton serves.

What it does, per source version (ascending):
  * copies the run — parameters, metrics, tags — into a NEW run on the server, tagged
    migrated_from_registry / migrated_from_run_id (run ids cannot be preserved);
  * copies every artifact byte-for-byte (the ONNX file keeps its sha256);
  * registers it, and REQUIRES the server to assign the same version number;
  * restores the stage.

It is idempotent (a version already migrated from the same run is skipped) and refuses
to continue if the numbering would diverge. It never alters the source store.
"""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

MIGRATED_FROM_RUN = "migrated_from_run_id"
MIGRATED_FROM_REGISTRY = "migrated_from_registry"


class RegistryMigrationError(RuntimeError):
    pass


@dataclass
class MigratedVersion:
    version: str
    source_run_id: str
    target_run_id: str
    stage: str
    skipped: bool


def migrate_registry(
    source_uri: str,
    target_uri: str,
    model_name: str,
    experiment_name: str,
    artifact_location: Optional[str] = None,
) -> List[MigratedVersion]:
    from mlflow import MlflowClient
    from mlflow.entities import Metric, Param, RunTag

    if not target_uri.lower().startswith(("http://", "https://")):
        raise RegistryMigrationError("the migration target must be the MLflow server (http://...)")
    src = MlflowClient(tracking_uri=source_uri, registry_uri=source_uri)
    dst = MlflowClient(tracking_uri=target_uri, registry_uri=target_uri)

    versions = sorted(src.search_model_versions(f"name='{model_name}'"), key=lambda m: int(m.version))
    if not versions:
        raise RegistryMigrationError(f"{model_name} has no versions in {source_uri}")

    experiment = dst.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else dst.create_experiment(
        experiment_name, artifact_location=artifact_location)
    try:
        dst.get_registered_model(model_name)
    except Exception:
        dst.create_registered_model(model_name)
    existing = {int(m.version): m for m in dst.search_model_versions(f"name='{model_name}'")}

    out: List[MigratedVersion] = []
    for mv in versions:
        n = int(mv.version)
        if n in existing:
            already = existing[n]
            if already.tags.get(MIGRATED_FROM_RUN) != mv.run_id:
                raise RegistryMigrationError(
                    f"server already has {model_name} v{n} from a different run "
                    f"({already.run_id}); refusing to mix two histories")
            out.append(MigratedVersion(str(n), mv.run_id, already.run_id, already.current_stage, True))
            continue
        expected = max(existing) + 1 if existing else 1
        if n != expected:
            raise RegistryMigrationError(
                f"cannot keep version numbers: the server would assign v{expected} to the "
                f"model that is v{n} in the source")

        run = src.get_run(mv.run_id)
        new_run = dst.create_run(
            experiment_id, start_time=run.info.start_time,
            tags={**{k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")},
                  "mlflow.runName": run.data.tags.get("mlflow.runName", f"migrated-{mv.run_id[:8]}"),
                  MIGRATED_FROM_RUN: mv.run_id, MIGRATED_FROM_REGISTRY: source_uri})
        rid = new_run.info.run_id
        metrics = [Metric(k, v, run.info.end_time or run.info.start_time, 0)
                   for k, v in run.data.metrics.items()]
        params = [Param(k, v) for k, v in run.data.params.items()]
        for i in range(0, max(len(metrics), 1), 500):
            dst.log_batch(rid, metrics=metrics[i:i + 500], params=params if i == 0 else [])
        with tempfile.TemporaryDirectory() as tmp:
            local = src.download_artifacts(mv.run_id, "", tmp)
            dst.log_artifacts(rid, local)
        dst.set_terminated(rid, status=run.info.status, end_time=run.info.end_time)

        source_path = mv.source.split(f"{mv.run_id}/artifacts/")[-1] if "/artifacts/" in mv.source else "xgboost_model"
        created = dst.create_model_version(
            model_name, source=f"{dst.get_run(rid).info.artifact_uri}/{source_path}", run_id=rid,
            tags={**dict(mv.tags), MIGRATED_FROM_RUN: mv.run_id, MIGRATED_FROM_REGISTRY: source_uri})
        if int(created.version) != n:
            raise RegistryMigrationError(
                f"server assigned v{created.version} to the model that is v{n} in the source")
        if mv.current_stage and mv.current_stage != "None":
            dst.transition_model_version_stage(model_name, created.version, mv.current_stage)
        existing[n] = created
        logger.info(f"Migrated {model_name} v{n}: run {mv.run_id} -> {rid}")
        out.append(MigratedVersion(str(n), mv.run_id, rid, mv.current_stage or "None", False))
    return out
