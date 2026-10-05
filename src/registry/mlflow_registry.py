"""
model-training-pipeline/src/registry/mlflow_registry.py

MLflow-backed model registry.
Handles: experiment tracking, model versioning, artifact storage,
champion/challenger promotion, and rollback.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import mlflow.pytorch
import mlflow.xgboost
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

from src.core.config import settings
from src.core.exceptions import InfrastructureUnavailableError
from src.domain.models import (
    TrainingDataset,
    ModelEvaluation, ModelStatus, TrainedModel, TrainingRun
)

logger = logging.getLogger(__name__)


def get_git_sha() -> str:
    """
    Git SHA of the training code, for reproducibility (LLD PART III §3.2).
    Prefers GIT_SHA (set by CI/container builds, where .git is usually absent),
    then falls back to the working tree. Returns "unknown" rather than raising —
    a missing SHA must not fail a training run, but it is recorded as missing.
    """
    env_sha = os.getenv("GIT_SHA")
    if env_sha:
        return env_sha
    try:
        import subprocess
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return sha.decode().strip()
    except Exception:
        logger.warning("Could not determine git SHA — run provenance will be incomplete")
        return "unknown"

def get_git_dirty() -> Optional[bool]:
    """
    True if the training code has uncommitted changes, False if the tree is clean,
    None if that cannot be determined. A git_sha alone is misleading when the tree is
    dirty: the commit does not contain the code that actually produced the model.
    """
    env = os.getenv("GIT_DIRTY")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes")
    try:
        import subprocess
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return bool(out.strip())
    except Exception:
        return None


def configure_mlflow_environment() -> str:
    """
    Point MLflow at the configured registry and return its scope ("server" or
    "local-scratch"). With the server registry, artifacts are stored in S3 and written
    by the client, so the S3 endpoint and credentials from this service's settings are
    exported for MLflow's S3 client (only where the process has not set them already).
    """
    tracking_uri = settings.MLFLOW_TRACKING_URI.replace("localhost", "127.0.0.1")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(
        (settings.MLFLOW_REGISTRY_URI or settings.MLFLOW_TRACKING_URI).replace("localhost", "127.0.0.1")
    )
    if settings.mlflow_registry_scope == "server":
        if settings.S3_ENDPOINT_URL:
            os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", settings.S3_ENDPOINT_URL)
        if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
            os.environ.setdefault("AWS_ACCESS_KEY_ID", settings.AWS_ACCESS_KEY_ID)
            os.environ.setdefault("AWS_SECRET_ACCESS_KEY", settings.AWS_SECRET_ACCESS_KEY)
        os.environ.setdefault("AWS_DEFAULT_REGION", settings.S3_REGION)
    return settings.mlflow_registry_scope


def registry_uri() -> str:
    """The registry this process talks to, as recorded in lineage."""
    return (settings.MLFLOW_REGISTRY_URI or settings.MLFLOW_TRACKING_URI).replace("localhost", "127.0.0.1")


def label_sources_text(label_sources: Dict) -> str:
    """{"synthetic": 6606} -> "synthetic=6606" (safe as an MLflow tag / Triton parameter)."""
    return ",".join(f"{k}={v}" for k, v in sorted((label_sources or {}).items())) or "none"


# The registered model name in MLflow
MODEL_NAME = "xyz-mdm-matcher"


class MLflowModelRegistry:
    """
    Wraps MLflow for model lifecycle management.
    Responsibilities:
      - Start/end experiment runs
      - Log parameters, metrics, artifacts
      - Register model versions
      - Promote champion, archive old versions
      - Rollback on degradation
    """

    def __init__(self):
        # Force 127.0.0.1 to avoid localhost→::1 IPv6 double-timeout on Windows
        self.scope = configure_mlflow_environment()
        if self.scope != "server":
            logger.warning(
                "MLflow registry is a LOCAL SCRATCH store (%s). Models registered here "
                "cannot be published to Triton; the authoritative registry is the MLflow "
                "server (MLFLOW_TRACKING_URI=http://...).", settings.MLFLOW_TRACKING_URI,
            )
        self._client = MlflowClient()
        self._ensure_experiment()

    @staticmethod
    def _mlflow_timeout_ctx():
        """Context manager that limits MLflow calls to 3s with no retries."""
        import socket as _s, os as _os
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            old_timeout = _s.getdefaulttimeout()
            old_retries = _os.environ.get("MLFLOW_HTTP_REQUEST_MAX_RETRIES")
            _s.setdefaulttimeout(3)
            _os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "0"
            try:
                yield
            finally:
                _s.setdefaulttimeout(old_timeout)
                if old_retries is None:
                    _os.environ.pop("MLFLOW_HTTP_REQUEST_MAX_RETRIES", None)
                else:
                    _os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = old_retries

        return _ctx()

    def _ensure_experiment(self):
        """Create MLflow experiment if it doesn't exist."""
        try:
            with self._mlflow_timeout_ctx():
                experiment = mlflow.get_experiment_by_name(settings.MLFLOW_EXPERIMENT_NAME)
                if experiment is None:
                    mlflow.create_experiment(
                        settings.MLFLOW_EXPERIMENT_NAME,
                        artifact_location=settings.MLFLOW_ARTIFACT_LOCATION,
                    )
                    logger.info(f"Created MLflow experiment: {settings.MLFLOW_EXPERIMENT_NAME}")
        except Exception as e:
            logger.warning(f"MLflow experiment init failed: {e}")

    def start_run(self, run: TrainingRun) -> str:
        """Start an MLflow run. Returns mlflow_run_id."""
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

        tags = {
            "trigger": run.trigger.value,
            "triggered_by": run.triggered_by,
            "feature_store_version": run.feature_store_version,
            "service_version": settings.SERVICE_VERSION,
            # Lineage: which labels, which feature definition, which code, which models.
            "feature_catalog_version": run.feature_store_version,
            "dataset_version": run.dataset_version or "unknown",
            "git_sha": get_git_sha(),
            "git_dirty": str(get_git_dirty()).lower(),
            "enabled_trainers": settings.ENABLED_TRAINERS,
            "environment": settings.ENVIRONMENT.value,
            "registry_scope": settings.mlflow_registry_scope,
            **self._dataset_tags(run),
        }

        mlflow_run = mlflow.start_run(
            run_name=f"training-{run.run_id.hex[:8]}-{run.trigger.value.lower()}",
            tags=tags,
        )

        # Log hyperparameters
        mlflow.log_params({
            "transformer_model": settings.TRANSFORMER_MODEL_NAME,
            "transformer_lr": settings.TRANSFORMER_LEARNING_RATE,
            "transformer_epochs": settings.TRANSFORMER_EPOCHS,
            "transformer_batch_size": settings.TRANSFORMER_BATCH_SIZE,
            "gnn_hidden_dim": settings.GNN_HIDDEN_DIM,
            "gnn_layers": settings.GNN_NUM_LAYERS,
            "gnn_epochs": settings.GNN_EPOCHS,
            "xgb_n_estimators": settings.XGB_N_ESTIMATORS,
            "xgb_max_depth": settings.XGB_MAX_DEPTH,
            "xgb_lr": settings.XGB_LEARNING_RATE,
            "ensemble_transformer_w": settings.ENSEMBLE_TRANSFORMER_WEIGHT,
            "ensemble_gnn_w": settings.ENSEMBLE_GNN_WEIGHT,
            "ensemble_xgb_w": settings.ENSEMBLE_XGB_WEIGHT,
            "min_f1_threshold": settings.MIN_F1_THRESHOLD,
            "training_lookback_days": settings.TRAINING_LOOKBACK_DAYS,
        })

        logger.info(f"MLflow run started: {mlflow_run.info.run_id}")
        return mlflow_run.info.run_id

    @staticmethod
    def _dataset_tags(run: TrainingRun) -> Dict[str, str]:
        """Dataset lineage as flat tags. Only what the run actually recorded."""
        prov = run.dataset_provenance or []
        tags = {
            "data_mode": settings.data_mode,
            "label_sources": label_sources_text(run.label_sources),
        }
        if prov:
            tags["dataset_id"] = ",".join(p["dataset_id"] for p in prov)
            tags["dataset_manifest_sha256"] = ",".join(p["manifest_sha256"] for p in prov)
            tags["dataset_generator_version"] = ",".join(str(p.get("generator_version")) for p in prov)
            tags["dataset_seed"] = ",".join(str(p.get("seed")) for p in prov)
        return tags

    def log_lineage(self, run: TrainingRun) -> None:
        """Lineage that is only known after feature extraction."""
        if run.feature_as_of is not None:
            mlflow.set_tag("feature_as_of", run.feature_as_of.isoformat())
        if run.dataset_provenance:
            mlflow.log_dict({"datasets": run.dataset_provenance,
                             "label_sources": run.label_sources,
                             "data_mode": settings.data_mode,
                             "dataset_version": run.dataset_version},
                            "dataset_provenance.json")

    def log_data_stats(self, run: TrainingRun):
        """Log training data statistics."""
        mlflow.log_metrics({
            "n_training_pairs": run.n_training_pairs,
            "n_positive": run.n_positive,
            "n_negative": run.n_negative,
            "positive_rate": run.n_positive / max(run.n_training_pairs, 1),
        })

    def log_split_stats(self, stats: Dict[str, int]) -> None:
        """Per-split sizes and label counts, plus residual entity-record overlap, so
        every evaluation metric can be tied to the exact split it came from."""
        if stats:
            mlflow.log_metrics({f"split_{k}": float(v) for k, v in stats.items()})

    def _export_serving_artifact(self, models: Dict, dataset: Optional[TrainingDataset]):
        """Verified ONNX serving artifact for the XGBoost component (see onnx_export)."""
        from src.registry.onnx_export import OnnxExportError, export_xgboost_to_onnx

        if dataset is None or dataset.feature_matrix is None or not dataset.feature_names:
            raise OnnxExportError(
                "Cannot build the ONNX serving artifact: the training dataset with its "
                "Feature Store feature names is required for the parity check."
            )
        rows_idx = dataset.test_indices or list(range(len(dataset.feature_matrix)))
        component = "xgboost" if models.get("is_partial_ensemble") else "xgboost_component_of_ensemble"
        return export_xgboost_to_onnx(
            models["xgb"]["model"],
            dataset.feature_names,
            dataset.feature_matrix[rows_idx],
            component=component,
        )

    def register_model(
        self,
        models: Dict,
        evaluation: ModelEvaluation,
        run: TrainingRun,
        dataset: Optional[TrainingDataset] = None,
    ) -> Optional[str]:
        """
        Register the trained ensemble model in MLflow Model Registry.
        Only registers if F1 exceeds minimum threshold.
        Returns model version string or None if rejected.

        When XGBoost was trained, its verified ONNX serving artifact is logged to the
        same run BEFORE registration, so a registered version always carries the exact
        artifact Triton will serve. `dataset` supplies the real rows (and the Feature
        Store's ordered feature names) used for the ONNX parity check.
        """
        if evaluation.f1_score < settings.MIN_F1_THRESHOLD:
            logger.warning(
                f"Model rejected: F1={evaluation.f1_score:.4f} < "
                f"minimum {settings.MIN_F1_THRESHOLD}"
            )
            mlflow.log_metric("registration_rejected", 1)
            mlflow.end_run(status="FAILED")
            return None

        logger.info(
            f"Registering model: F1={evaluation.f1_score:.4f} "
            f"(threshold: {settings.MIN_F1_THRESHOLD})"
        )

        # Log evaluation metrics
        mlflow.log_metrics({
            "final_f1": evaluation.f1_score,
            "final_precision": evaluation.precision,
            "final_recall": evaluation.recall,
            "final_auc_roc": evaluation.auc_roc,
            "final_avg_precision": evaluation.average_precision,
            "n_test_samples": evaluation.n_test_samples,
        })

        # Log confusion matrix as artifact
        with tempfile.TemporaryDirectory() as tmpdir:
            cm_path = os.path.join(tmpdir, "confusion_matrix.json")
            with open(cm_path, "w") as f:
                json.dump(evaluation.confusion_matrix, f)
            mlflow.log_artifact(cm_path, "evaluation")

            # Log feature importance
            if evaluation.feature_importance:
                fi_path = os.path.join(tmpdir, "feature_importance.json")
                with open(fi_path, "w") as f:
                    json.dump(evaluation.feature_importance, f, indent=2)
                mlflow.log_artifact(fi_path, "evaluation")

        # Log each model that was ACTUALLY trained. A model that was skipped
        # (ENABLED_TRAINERS) is recorded as not-trained rather than logged as an
        # artifact, so the registry never implies an ensemble member exists when
        # it does not.
        component_uris = {}
        serving: Dict = {}

        if models["xgb"].get("model") is not None:
            # Logged as a run artifact only. It used to be registered a second time
            # under "<MODEL_NAME>-xgb", giving every run two registry entries whose
            # version numbers could drift apart.
            mlflow.xgboost.log_model(
                models["xgb"]["model"],
                "xgboost_model",
            )
            component_uris["xgboost_model_uri"] = "runs:/{run_id}/xgboost_model"

            onnx_export = self._export_serving_artifact(models, dataset)
            with tempfile.TemporaryDirectory() as onnx_dir:
                onnx_path = os.path.join(onnx_dir, "model.onnx")
                with open(onnx_path, "wb") as f:
                    f.write(onnx_export.onnx_bytes)
                sig_path = os.path.join(onnx_dir, "serving_signature.json")
                with open(sig_path, "w") as f:
                    json.dump(onnx_export.signature, f, indent=2)
                mlflow.log_artifacts(onnx_dir, "onnx_model")
            component_uris["onnx_model_uri"] = "runs:/{run_id}/onnx_model/model.onnx"
            serving = {
                "onnx_sha256": onnx_export.sha256,
                "feature_names_sha256": onnx_export.signature["feature_names_sha256"],
                "serving_component": onnx_export.signature["component"],
            }

        if models["gnn"].get("model") is not None:
            mlflow.pytorch.log_model(models["gnn"]["model"], "gnn_model")
            component_uris["gnn_model_uri"] = "runs:/{run_id}/gnn_model"

        if models["transformer"].get("model") is not None:
            mlflow.pytorch.log_model(models["transformer"]["model"], "transformer_model")
            component_uris["transformer_model_uri"] = "runs:/{run_id}/transformer_model"

            if models["transformer"].get("tokenizer"):
                with tempfile.TemporaryDirectory() as tok_dir:
                    models["transformer"]["tokenizer"].save_pretrained(tok_dir)
                    mlflow.log_artifacts(tok_dir, "transformer_tokenizer")
                component_uris["transformer_tokenizer_uri"] = "runs:/{run_id}/transformer_tokenizer"

        active_run_id = mlflow.active_run().info.run_id
        component_uris = {k: v.format(run_id=active_run_id) for k, v in component_uris.items()}

        skipped = {
            name: models[name].get("skipped_reason")
            for name in ("transformer", "gnn", "xgb")
            if not models[name].get("trained", models[name].get("model") is not None)
        }

        # Log ensemble config
        ensemble_config = {
            "weights": models["weights"],
            "transformer_f1": models["transformer"]["f1"],
            "gnn_f1": models["gnn"]["f1"],
            "xgb_f1": models["xgb"]["f1"],
            "ensemble_f1": evaluation.f1_score,
            "feature_store_version": run.feature_store_version,
            "is_partial_ensemble": models.get("is_partial_ensemble", False),
            "enabled_trainers": settings.ENABLED_TRAINERS,
            "skipped_models": skipped,
        }
        mlflow.log_dict(ensemble_config, "ensemble_config.json")

        # Ensemble manifest — binds every component artifact, the weights and the
        # full provenance under one ensemble_version, so a served model can be
        # traced back to exactly how it was produced (SERVICE_CONTRACTS.md §4.2).
        manifest = {
            "ensemble_version": run.model_version or f"run-{run.run_id.hex[:8]}",
            "is_partial_ensemble": models.get("is_partial_ensemble", False),
            "enabled_trainers": sorted(settings.enabled_trainers),
            "skipped_models": skipped,
            "feature_catalog_version": run.feature_store_version,
            "dataset_version": run.dataset_version,
            "dataset": {
                "dataset_version": run.dataset_version,
                "data_mode": settings.data_mode,
                "label_sources": dict(run.label_sources or {}),
                "datasets": list(run.dataset_provenance or []),
            },
            "feature_as_of": run.feature_as_of.isoformat() if run.feature_as_of else None,
            "git_sha": get_git_sha(),
            "git_dirty": get_git_dirty(),
            "environment": settings.ENVIRONMENT.value,
            "registry_scope": settings.mlflow_registry_scope,
            "registry_uri": registry_uri(),
            "training_run_id": str(run.run_id),
            "mlflow_run_id": active_run_id,
            "mlflow_model_name": MODEL_NAME,
            **component_uris,
            "weights": {
                "transformer": models["weights"][0],
                "gnn": models["weights"][1],
                "xgboost": models["weights"][2],
            },
            "metrics": {
                "precision": evaluation.precision,
                "recall": evaluation.recall,
                "f1": evaluation.f1_score,
                "auc": evaluation.auc_roc,
            },
            "created_at": datetime.utcnow().isoformat() + "Z",
            **serving,
        }
        mlflow.log_dict(manifest, "ensemble_manifest.json")

        # A model must be reproducible from a commit. Outside development a dirty
        # working tree is refused; in development it is recorded (git_dirty) so the
        # lineage never claims a commit that does not contain the code that ran.
        if get_git_dirty() and settings.ENVIRONMENT.value in ("staging", "production"):
            mlflow.end_run(status="FAILED")
            raise InfrastructureUnavailableError(
                "MLflow Model Registry",
                "refusing to register a model built from a working tree with uncommitted "
                "changes; commit the code first",
            )

        # Register in Model Registry
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/xgboost_model"

        try:
            mv = mlflow.register_model(
                model_uri=model_uri,
                name=MODEL_NAME,
                tags={
                    "f1_score": str(evaluation.f1_score),
                    "precision": str(evaluation.precision),
                    "recall": str(evaluation.recall),
                    "training_run_id": str(run.run_id),
                    "trigger": run.trigger.value,
                    "dataset_version": str(run.dataset_version),
                    "feature_catalog_version": str(run.feature_store_version),
                    "git_sha": get_git_sha(),
                    "git_dirty": str(get_git_dirty()).lower(),
                    "registry_scope": settings.mlflow_registry_scope,
                    **({"onnx_sha256": serving["onnx_sha256"]} if serving else {}),
                    **self._dataset_tags(run),
                },
            )
            # MLflow returns the version as an int on some backends (SQLite) and a
            # str on others. TrainingRun.model_version is a VARCHAR column, so a raw
            # int fails the UPDATE with an asyncpg DataError AFTER the model was
            # already registered — losing the run record for a successful training.
            version = str(mv.version)
            # The manifest was written before the registry assigned a version, so it
            # carried the run id as ensemble_version. Re-log it with the real identity.
            manifest["ensemble_version"] = f"v{version}"
            manifest["registered_model_version"] = version
            mlflow.log_dict(manifest, "ensemble_manifest.json")
            logger.info(f"Model registered as {MODEL_NAME} version {version}")
        except Exception as e:
            # Previously this set version = "1" and carried on, reporting a model
            # version that was never registered. A registration failure is a failure.
            logger.error(f"MLflow model registration failed: {e}")
            mlflow.end_run(status="FAILED")
            raise InfrastructureUnavailableError(
                "MLflow Model Registry", f"{type(e).__name__}: {e}"
            ) from e

        mlflow.end_run(status="FINISHED")
        return version

    def get_current_champion(self) -> Optional[Dict]:
        """
        Get the currently deployed production model version and its metrics.
        """
        try:
            with self._mlflow_timeout_ctx():
                versions = self._client.get_latest_versions(MODEL_NAME, stages=["Production"])
            if not versions:
                return None

            v = versions[0]
            return {
                "version": v.version,
                "run_id": v.run_id,
                "f1_score": float(v.tags.get("f1_score", 0)),
                "created_at": v.creation_timestamp,
            }
        except Exception as e:
            logger.warning(f"Could not fetch champion model: {e}")
            return None

    def promote_to_staging(self, version: str) -> bool:
        """Move model version to Staging."""
        try:
            self._client.transition_model_version_stage(
                name=MODEL_NAME,
                version=version,
                stage="Staging",
                archive_existing_versions=False,
            )
            logger.info(f"Model {MODEL_NAME} v{version} promoted to Staging")
            return True
        except Exception as e:
            logger.error(f"Staging promotion failed: {e}")
            return False

    def promote_to_production(self, version: str) -> bool:
        """
        Promote a model version to Production.
        Archives the previous production version.
        """
        try:
            self._client.transition_model_version_stage(
                name=MODEL_NAME,
                version=version,
                stage="Production",
                archive_existing_versions=True,  # Auto-archives previous champion
            )
            logger.info(f"Model {MODEL_NAME} v{version} promoted to PRODUCTION")
            return True
        except Exception as e:
            logger.error(f"Production promotion failed: {e}")
            return False

    def rollback(self, reason: str) -> Optional[str]:
        """
        Rollback to the previous production model.
        Archives the current version, promotes the previous archived version.
        Returns the version rolled back to, or None on failure.
        """
        try:
            # Get all versions
            all_versions = self._client.search_model_versions(f"name='{MODEL_NAME}'")
            archived = [v for v in all_versions if v.current_stage == "Archived"]

            if not archived:
                logger.error("No archived version to rollback to")
                return None

            # Most recently archived
            archived.sort(key=lambda v: v.last_updated_timestamp, reverse=True)
            rollback_version = archived[0].version

            # Archive current production
            prod_versions = self._client.get_latest_versions(MODEL_NAME, stages=["Production"])
            for v in prod_versions:
                self._client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=v.version,
                    stage="Archived",
                )

            # Promote previous to production
            self._client.transition_model_version_stage(
                name=MODEL_NAME,
                version=rollback_version,
                stage="Production",
            )

            logger.info(
                f"ROLLBACK: Model rolled back to {MODEL_NAME} v{rollback_version}. "
                f"Reason: {reason}"
            )
            return rollback_version

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return None

    def list_versions(self) -> List[Dict]:
        """List all registered model versions with their metrics."""
        try:
            with self._mlflow_timeout_ctx():
                versions = self._client.search_model_versions(f"name='{MODEL_NAME}'")
            return [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "f1_score": v.tags.get("f1_score"),
                    "run_id": v.run_id,
                    "created_at": v.creation_timestamp,
                }
                for v in sorted(versions, key=lambda v: int(v.version), reverse=True)
            ]
        except Exception as e:
            logger.error(f"Failed to list versions: {e}")
            return []
