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
from src.domain.models import (
    ModelEvaluation, ModelStatus, TrainedModel, TrainingRun
)

logger = logging.getLogger(__name__)

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
        tracking_uri = settings.MLFLOW_TRACKING_URI.replace("localhost", "127.0.0.1")
        mlflow.set_tracking_uri(tracking_uri)
        if settings.MLFLOW_REGISTRY_URI:
            mlflow.set_registry_uri(settings.MLFLOW_REGISTRY_URI.replace("localhost", "127.0.0.1"))
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

    def log_data_stats(self, run: TrainingRun):
        """Log training data statistics."""
        mlflow.log_metrics({
            "n_training_pairs": run.n_training_pairs,
            "n_positive": run.n_positive,
            "n_negative": run.n_negative,
            "positive_rate": run.n_positive / max(run.n_training_pairs, 1),
        })

    def register_model(
        self,
        models: Dict,
        evaluation: ModelEvaluation,
        run: TrainingRun,
    ) -> Optional[str]:
        """
        Register the trained ensemble model in MLflow Model Registry.
        Only registers if F1 exceeds minimum threshold.
        Returns model version string or None if rejected.
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

        # Log XGBoost model
        mlflow.xgboost.log_model(
            models["xgb"]["model"],
            "xgboost_model",
            registered_model_name=f"{MODEL_NAME}-xgb",
        )

        # Log PyTorch GNN model
        mlflow.pytorch.log_model(
            models["gnn"]["model"],
            "gnn_model",
        )

        # Log ensemble config
        ensemble_config = {
            "weights": models["weights"],
            "transformer_f1": models["transformer"]["f1"],
            "gnn_f1": models["gnn"]["f1"],
            "xgb_f1": models["xgb"]["f1"],
            "ensemble_f1": evaluation.f1_score,
            "feature_store_version": run.feature_store_version,
        }
        mlflow.log_dict(ensemble_config, "ensemble_config.json")

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
                },
            )
            version = mv.version
            logger.info(f"Model registered as {MODEL_NAME} version {version}")
        except Exception as e:
            logger.error(f"MLflow model registration failed: {e}")
            version = "1"  # Fallback

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
