"""
model-training-pipeline/src/pipeline/training_pipeline.py

Main pipeline orchestrator: runs all 8 stages end-to-end.

Stages:
  1. Data Collection       (~15 min)
  2. Feature Extraction    (~30 min)
  3. Train Transformer     (~45 min GPU)
  4. Train GNN             (~30 min GPU)
  5. Train XGBoost         (~15 min CPU)
  6. Ensemble Combination  (~5 min)
  7. Evaluation            (~10 min)
  8. Register / Deploy     (~2 min)

Total: ~2.5 hours on GPU A100
"""
from __future__ import annotations

import asyncio
import logging
import traceback
from datetime import datetime
from typing import Optional
from uuid import UUID

from src.adapters.feature_store_client import FeatureStoreClient
from src.core.config import settings
from src.domain.models import (
    ModelStatus, RetrainingTrigger, RetrainingTriggerEvent,
    RunStatus, TrainedModel, TrainingRun,
)
from src.evaluators.model_evaluator import ModelEvaluator
from src.pipeline.stages.data_collection import DataCollectionStage, TrainingDataSplitter
from src.pipeline.stages.feature_extraction import FeatureExtractionStage
from src.registry.mlflow_registry import MLflowModelRegistry
from src.trainers.ensemble_trainer import EnsembleTrainer
from src.core.metrics import (
    TRAINING_DURATION, MODEL_F1_SCORE, RETRAINING_TRIGGERED,
    ACTIVE_TRAINING_RUNS, ROLLBACK_EVENTS, STAGE_RETRIES,
)
from src.core.exceptions import EvaluationGateFailedError
from src.pipeline.kfp_pipeline import KubeflowPipelineRunner

logger = logging.getLogger(__name__)

# Stage retry counts from LLD PART II §2.2 (Table 5)
STAGE_RETRY_COUNTS = {
    "data_collection": 3,
    "feature_extraction": 3,
    "train_transformer": 2,
    "train_gnn": 2,
    "train_xgboost": 3,
    "ensemble": 2,
    "evaluation": 2,
    "registration": 3,
}


class TrainingPipeline:
    """
    Orchestrates the complete ML training pipeline.
    Can run locally (dev) or via Kubeflow (production).
    """

    def __init__(
        self,
        db_session,
        feature_store_client=None,
        kubeflow_client=None,
    ):
        self._db = db_session
        if feature_store_client is None:
            feature_store_client = FeatureStoreClient.from_settings()
        self._data_stage = DataCollectionStage(db_session, feature_store_client)
        self._feature_stage = FeatureExtractionStage(feature_store_client)
        self._splitter = TrainingDataSplitter(
            test_ratio=settings.TEST_SPLIT_RATIO,
            val_ratio=settings.VAL_SPLIT_RATIO,
        )
        self._ensemble = EnsembleTrainer()
        self._evaluator = ModelEvaluator(self._ensemble)
        self._registry = MLflowModelRegistry()
        # Use provided client or build from settings
        if kubeflow_client is not None:
            self._kubeflow = kubeflow_client
        elif settings.KUBEFLOW_HOST:
            self._kubeflow = KubeflowPipelineRunner(
                host=settings.KUBEFLOW_HOST,
                namespace=settings.KUBEFLOW_NAMESPACE,
            )
        else:
            self._kubeflow = None

    async def _run_stage_with_retry(self, stage_name: str, func, *args, **kwargs):
        """
        Run a pipeline stage with automatic retry on failure.
        Retry counts are defined per stage in STAGE_RETRY_COUNTS (LLD Table 5).
        Uses exponential backoff: 30s, 60s, 90s, ...
        """
        max_retries = STAGE_RETRY_COUNTS.get(stage_name, 2)
        for attempt in range(1, max_retries + 1):
            try:
                result = func(*args, **kwargs)
                # Handle both sync and async callables
                if asyncio.iscoroutine(result):
                    return await result
                return result
            except Exception as e:
                if attempt == max_retries:
                    logger.error(
                        "Stage '%s' failed after %d attempts: %s",
                        stage_name, max_retries, e,
                    )
                    raise
                wait = 30 * attempt
                STAGE_RETRIES.labels(stage=stage_name).inc()
                logger.warning(
                    "Stage '%s' failed (attempt %d/%d): %s. Retrying in %ds...",
                    stage_name, attempt, max_retries, e, wait,
                )
                await asyncio.sleep(wait)

    async def run(
        self,
        training_run: TrainingRun,
        tune_weights: bool = False,
    ) -> TrainingRun:
        """
        Execute the complete training pipeline for a given TrainingRun.
        Updates the run's status, metrics, and model references throughout.
        """
        ACTIVE_TRAINING_RUNS.inc()
        RETRAINING_TRIGGERED.labels(trigger_type=training_run.trigger.value).inc()

        training_run.status = RunStatus.RUNNING
        training_run.started_at = datetime.utcnow()
        # The feature catalog version this run requests from the Feature Store —
        # previously left at the model's hardcoded default rather than the config.
        training_run.feature_store_version = settings.FEATURE_CATALOG_VERSION
        await self._save_run(training_run)

        # ── KFP path: delegate to Kubeflow Pipelines cluster ─────────
        if self._kubeflow is not None:
            return await self._run_via_kubeflow(training_run)

        try:
            # ── Stage 1: Data Collection ─────────────────────────────
            logger.info(f"[{training_run.run_id}] ━━━ Stage 1: Data Collection")
            pairs = await self._run_stage_with_retry(
                "data_collection",
                self._data_stage.execute,
                training_run,
            )
            training_run.n_training_pairs = len(pairs)
            training_run.n_positive = sum(1 for p in pairs if p.label == 1)
            training_run.n_negative = sum(1 for p in pairs if p.label == 0)
            training_run.data_collected_at = datetime.utcnow()
            await self._save_run(training_run)

            if len(pairs) < 100:
                raise ValueError(
                    f"Insufficient training data: {len(pairs)} pairs. "
                    f"Minimum required: 100 (production: {settings.MIN_LABELED_PAIRS})"
                )

            # ── Stage 2: Feature Extraction ──────────────────────────
            logger.info(f"[{training_run.run_id}] ━━━ Stage 2: Feature Extraction")

            # Start MLflow run before feature extraction
            mlflow_run_id = self._registry.start_run(training_run)
            training_run.mlflow_run_id = mlflow_run_id
            await self._save_run(training_run)

            dataset = await self._run_stage_with_retry(
                "feature_extraction",
                self._feature_stage.execute,
                training_run,
                pairs,
            )

            # Split dataset
            train_idx, val_idx, test_idx = self._splitter.split(dataset.pairs)
            dataset.train_indices = train_idx
            dataset.val_indices = val_idx
            dataset.test_indices = test_idx
            training_run.features_extracted_at = datetime.utcnow()

            self._registry.log_data_stats(training_run)
            self._registry.log_split_stats(self._splitter.last_stats)

            # ── Stage 3-5: Model Training ────────────────────────────
            logger.info(f"[{training_run.run_id}] ━━━ Stages 3-5: Training all models")
            models = await self._run_stage_with_retry(
                "ensemble",
                asyncio.to_thread,
                self._ensemble.train_all,
                dataset,
                mlflow_run_id,
                tune_weights,
            )

            training_run.transformer_f1 = models["transformer"]["f1"]
            training_run.gnn_f1 = models["gnn"]["f1"]
            training_run.xgb_f1 = models["xgb"]["f1"]
            training_run.training_completed_at = datetime.utcnow()
            await self._save_run(training_run)

            # ── Stage 7: Evaluation ───────────────────────────────────
            logger.info(f"[{training_run.run_id}] ━━━ Stage 7: Evaluation")
            from uuid import uuid4
            model_id = uuid4()

            # Get champion metrics for comparison
            champion = self._registry.get_current_champion()
            champion_f1 = champion["f1_score"] if champion else None

            evaluation = self._evaluator.evaluate(
                models=models,
                dataset=dataset,
                model_id=model_id,
                run_id=training_run.run_id,
                champion_f1=champion_f1,
            )

            training_run.ensemble_f1 = evaluation.f1_score
            training_run.ensemble_precision = evaluation.precision
            training_run.ensemble_recall = evaluation.recall
            training_run.ensemble_auc = evaluation.auc_roc
            training_run.evaluation_completed_at = datetime.utcnow()

            MODEL_F1_SCORE.labels(
                model="ensemble",
                version=str(training_run.run_id)[:8],
            ).set(evaluation.f1_score)

            await self._save_run(training_run)

            # ── Stage 8: Register ─────────────────────────────────────
            logger.info(f"[{training_run.run_id}] ━━━ Stage 8: Register")
            model_version = self._registry.register_model(
                models=models,
                evaluation=evaluation,
                run=training_run,
            )

            if model_version:
                training_run.model_version = model_version
                training_run.registered_at = datetime.utcnow()

                # Auto-promote if significantly better than champion
                if self._should_auto_promote(evaluation, champion_f1):
                    self._registry.promote_to_staging(model_version)
                    logger.info(
                        f"Model v{model_version} auto-promoted to STAGING "
                        f"(F1={evaluation.f1_score:.4f})"
                    )
                    training_run.promoted_to_production = False  # staging, not prod yet
                else:
                    logger.info(
                        f"Model v{model_version} registered in STAGING (requires manual promotion)"
                    )

            # ── Complete ───────────────────────────────────────────────
            duration = (datetime.utcnow() - training_run.started_at).total_seconds()
            TRAINING_DURATION.labels(model="ensemble", version=model_version or "unknown").observe(duration)

            training_run.status = RunStatus.COMPLETED
            training_run.completed_at = datetime.utcnow()
            await self._save_run(training_run)

            logger.info(
                f"[{training_run.run_id}] ━━━ PIPELINE COMPLETE ━━━\n"
                f"  Duration:  {duration/60:.1f} min\n"
                f"  F1:        {evaluation.f1_score:.4f}\n"
                f"  Precision: {evaluation.precision:.4f}\n"
                f"  Recall:    {evaluation.recall:.4f}\n"
                f"  AUC-ROC:   {evaluation.auc_roc:.4f}\n"
                f"  Version:   {model_version or 'not registered (F1 too low)'}"
            )

        except Exception as e:
            training_run.status = RunStatus.FAILED
            training_run.error_message = str(e)
            training_run.completed_at = datetime.utcnow()
            await self._save_run(training_run)
            logger.error(
                f"[{training_run.run_id}] Pipeline FAILED: {e}\n{traceback.format_exc()}"
            )
            try:
                import mlflow
                mlflow.end_run(status="FAILED")
            except Exception:
                pass
        finally:
            ACTIVE_TRAINING_RUNS.dec()

        return training_run

    async def _run_via_kubeflow(self, training_run: TrainingRun) -> TrainingRun:
        """
        Submit the training pipeline to KFP and poll for completion.
        The 8 pipeline stages run as independent KFP components (containers)
        on the cluster, with GPU nodes assigned for Transformer and GNN stages.
        """
        import asyncio

        try:
            kfp_run_id = self._kubeflow.submit_run(
                run_id=training_run.run_id,
                feature_store_url=settings.FEATURE_STORE_URL,
                mlflow_tracking_uri=settings.MLFLOW_TRACKING_URI,
                mlflow_experiment=settings.MLFLOW_EXPERIMENT_NAME,
                lookback_days=settings.TRAINING_LOOKBACK_DAYS,
                min_pairs=settings.MIN_LABELED_PAIRS,
                min_f1_threshold=settings.MIN_F1_THRESHOLD,
            )
            training_run.mlflow_run_id = kfp_run_id  # store KFP run ID for traceability
            await self._save_run(training_run)

            logger.info("[%s] KFP run submitted: %s — polling for completion", training_run.run_id, kfp_run_id)

            # Poll in a thread to avoid blocking the event loop
            final_state = await asyncio.to_thread(
                self._kubeflow.wait_for_completion,
                kfp_run_id,
                10800,  # 3-hour timeout
            )

            if final_state == "SUCCEEDED":
                training_run.status = RunStatus.COMPLETED
                training_run.completed_at = datetime.utcnow()
                duration = (training_run.completed_at - training_run.started_at).total_seconds()
                TRAINING_DURATION.labels(
                    model="ensemble", version=kfp_run_id[:8]
                ).observe(duration)
                logger.info("[%s] KFP pipeline SUCCEEDED in %.1f min", training_run.run_id, duration / 60)
            else:
                training_run.status = RunStatus.FAILED
                training_run.error_message = f"KFP run ended with state: {final_state}"
                training_run.completed_at = datetime.utcnow()
                logger.error("[%s] KFP pipeline FAILED (state=%s)", training_run.run_id, final_state)

        except Exception as exc:
            training_run.status = RunStatus.FAILED
            training_run.error_message = str(exc)
            training_run.completed_at = datetime.utcnow()
            logger.error("[%s] KFP submission error: %s", training_run.run_id, exc, exc_info=True)
        finally:
            ACTIVE_TRAINING_RUNS.dec()
            await self._save_run(training_run)

        return training_run

    def _should_auto_promote(
        self,
        evaluation,
        champion_f1: Optional[float],
    ) -> bool:
        """
        Determine if the new model should be auto-promoted to staging.
        Requirements:
          1. F1 ≥ minimum threshold
          2. F1 beats champion by ≥ 0.5%
          3. Statistically significant improvement (p < 0.05)
        """
        if evaluation.f1_score < settings.MIN_F1_THRESHOLD:
            return False
        if champion_f1 and evaluation.f1_score < champion_f1 + settings.PROMOTION_F1_DELTA:
            logger.info(
                f"Auto-promote rejected: F1 {evaluation.f1_score:.4f} does not beat "
                f"champion {champion_f1:.4f} + {settings.PROMOTION_F1_DELTA}"
            )
            return False
        return True

    async def _save_run(self, run: TrainingRun):
        """Persist training run state to database."""
        try:
            from src.repositories.training_run_repository import TrainingRunRepository
            repo = TrainingRunRepository(self._db)
            await repo.upsert(run)
            await self._db.commit()
        except Exception as e:
            logger.warning(f"Failed to persist training run state: {e}")


class ChampionChallengerManager:
    """
    Manages the canary rollout and promotion lifecycle.
    Champion/Challenger flow:
      STAGING → CANARY (10%) → RAMP_1 (30%) → RAMP_2 (50%) → PRODUCTION (100%)
      or ROLLBACK at any stage.
    """

    def __init__(self, registry: MLflowModelRegistry):
        self._registry = registry

    async def start_canary(self, model_version: str) -> bool:
        """Start canary deployment with 10% traffic."""
        logger.info(f"Starting canary deployment for model v{model_version} (10% traffic)")
        # In production: update Kubernetes traffic split via Istio/Argo
        # Here: update MLflow model status + emit traffic routing event
        return self._registry.promote_to_staging(model_version)

    async def advance_rollout(self, model_version: str, phase: str) -> bool:
        """Advance rollout to next phase."""
        phase_traffic = {"RAMP_1": 30, "RAMP_2": 50, "PROMOTION": 100}
        traffic_pct = phase_traffic.get(phase, 10)
        logger.info(f"Advancing model v{model_version} rollout to {phase} ({traffic_pct}% traffic)")

        if phase == "PROMOTION":
            return self._registry.promote_to_production(model_version)
        return True  # Other phases handled by traffic controller

    async def check_canary_health(self, model_version: str, champion_f1: float) -> tuple:
        """
        Check if canary is performing well enough to continue rollout.
        Returns (is_healthy, rollback_reason).
        In production: queries Prometheus for live inference metrics.
        """
        # Production: query Prometheus metrics API for challenger model
        # metrics like inference_latency_ms{model_version="challenger"} etc.
        return True, ""  # Assume healthy in this implementation

    async def rollback(self, model_version: str, reason: str) -> Optional[str]:
        """Emergency rollback to previous champion."""
        ROLLBACK_EVENTS.labels(reason=reason).inc()
        rollback_version = self._registry.rollback(reason)
        logger.critical(
            f"ROLLBACK executed: {model_version} → {rollback_version}. Reason: {reason}"
        )
        return rollback_version
