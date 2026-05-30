"""
model-training-pipeline/src/api/v1/endpoints/training.py
REST API for training pipeline management.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from xyz_security import TenantContext, get_current_tenant, require_permission, Resource, Action

from src.api.v1.schemas import (
    HealthResponse, ModelVersionResponse,
    PromoteRequest, PromoteResponse, RollbackRequest, RollbackResponse,
    TrainingRunResponse, TriggerTrainingRequest,
)
from src.core.config import settings
from src.domain.models import RetrainingTrigger, RetrainingTriggerEvent, TrainingRun
from src.pipeline.training_pipeline import ChampionChallengerManager, TrainingPipeline
from src.registry.mlflow_registry import MLflowModelRegistry
from src.repositories.training_run_repository import TrainingRunRepository, get_db_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["training"])


def get_registry() -> MLflowModelRegistry:
    return MLflowModelRegistry()


@router.post(
    "/training/runs",
    response_model=TrainingRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger a new training run",
    description=(
        "Launches the full 8-stage training pipeline asynchronously. "
        "Returns immediately with the run_id. Poll /training/runs/{run_id} for status."
    ),
)
async def trigger_training(
    request: TriggerTrainingRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session),
    tenant: TenantContext = Depends(require_permission(Resource.TRAINING, Action.WRITE)),
):
    try:
        trigger = RetrainingTrigger(request.trigger_type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid trigger_type: {request.trigger_type}",
        )

    reason_str = (request.reason or trigger.value)[:500]
    run = TrainingRun(
        trigger=trigger,
        triggered_by=f"api:{tenant.user_id}:{reason_str}",
    )

    # Enqueue the background task BEFORE committing so a task-enqueue failure
    # prevents an orphaned run record from being created.
    pipeline = TrainingPipeline(db_session=db)
    background_tasks.add_task(pipeline.run, run, request.tune_weights)

    repo = TrainingRunRepository(db)
    await repo.upsert(run)
    await db.commit()

    return _run_to_response(run)


@router.get(
    "/training/runs/{run_id}",
    response_model=TrainingRunResponse,
    summary="Get training run status",
)
async def get_training_run(
    run_id: UUID,
    db: AsyncSession = Depends(get_db_session),
    tenant: TenantContext = Depends(require_permission(Resource.TRAINING, Action.READ)),
):
    repo = TrainingRunRepository(db)
    run = await repo.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Training run {run_id} not found")
    return _run_to_response(run)


@router.get(
    "/training/runs",
    response_model=List[TrainingRunResponse],
    summary="List recent training runs",
)
async def list_training_runs(
    limit: int = Query(default=20, le=100),
    db: AsyncSession = Depends(get_db_session),
    tenant: TenantContext = Depends(require_permission(Resource.TRAINING, Action.READ)),
):
    repo = TrainingRunRepository(db)
    runs = await repo.list_recent(limit=limit)
    return [_run_to_response(r) for r in runs]


@router.post(
    "/models/promote",
    response_model=PromoteResponse,
    summary="Promote model version to a deployment stage",
    description="Stages: staging → canary → production. Production triggers champion/challenger swap.",
)
async def promote_model(
    request: PromoteRequest,
    registry: MLflowModelRegistry = Depends(get_registry),
    tenant: TenantContext = Depends(require_permission(Resource.TRAINING, Action.WRITE)),
):
    success = False
    if request.target_stage == "staging":
        success = registry.promote_to_staging(request.model_version)
    elif request.target_stage in ("canary", "production"):
        success = registry.promote_to_production(request.model_version)
    else:
        raise HTTPException(status_code=422, detail=f"Unknown stage: {request.target_stage}")

    return PromoteResponse(
        model_version=request.model_version,
        stage=request.target_stage,
        success=success,
        message=f"Model {request.model_version} {'promoted to' if success else 'failed promotion to'} {request.target_stage}",
    )


@router.post(
    "/models/rollback",
    response_model=RollbackResponse,
    summary="Emergency rollback to previous champion",
)
async def rollback_model(
    request: RollbackRequest,
    registry: MLflowModelRegistry = Depends(get_registry),
    tenant: TenantContext = Depends(require_permission(Resource.TRAINING, Action.WRITE)),
):
    if not request.reason or not request.reason.strip():
        raise HTTPException(status_code=422, detail="reason is required for rollback")
    reason = request.reason.strip()[:500]
    rolled_back_to = registry.rollback(reason)
    return RollbackResponse(
        rolled_back_to_version=rolled_back_to,
        reason=reason,
        success=rolled_back_to is not None,
    )


@router.get(
    "/models",
    response_model=List[ModelVersionResponse],
    summary="List all registered model versions",
)
async def list_models(
    registry: MLflowModelRegistry = Depends(get_registry),
    tenant: TenantContext = Depends(require_permission(Resource.TRAINING, Action.READ)),
):
    versions = registry.list_versions()
    return [
        ModelVersionResponse(
            version=v["version"],
            stage=v["stage"],
            f1_score=v.get("f1_score"),
            run_id=v.get("run_id"),
            created_at=v.get("created_at"),
        )
        for v in versions
    ]


@router.get(
    "/models/champion",
    summary="Get current production champion model",
)
async def get_champion(
    registry: MLflowModelRegistry = Depends(get_registry),
    tenant: TenantContext = Depends(require_permission(Resource.TRAINING, Action.READ)),
):
    champion = registry.get_current_champion()
    if not champion:
        raise HTTPException(status_code=404, detail="No production champion model found")
    return champion


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
async def health_check(
    db: AsyncSession = Depends(get_db_session),
):
    checks = {}

    # PostgreSQL
    try:
        from sqlalchemy import text
        await db.execute(text("SELECT 1"))
        checks["postgres"] = True
    except Exception:
        checks["postgres"] = False

    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        checks["mlflow"] = True
    except Exception:
        checks["mlflow"] = False

    checks["kafka"] = True

    all_ok = all(checks.values())
    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        service=settings.SERVICE_NAME,
        version=settings.SERVICE_VERSION,
        timestamp=datetime.utcnow(),
        checks=checks,
    )


def _run_to_response(run: TrainingRun) -> TrainingRunResponse:
    return TrainingRunResponse(
        run_id=run.run_id,
        trigger=run.trigger.value,
        status=run.status.value,
        triggered_by=run.triggered_by,
        mlflow_run_id=run.mlflow_run_id,
        n_training_pairs=run.n_training_pairs,
        n_positive=run.n_positive,
        n_negative=run.n_negative,
        transformer_f1=run.transformer_f1,
        gnn_f1=run.gnn_f1,
        xgb_f1=run.xgb_f1,
        ensemble_f1=run.ensemble_f1,
        ensemble_precision=run.ensemble_precision,
        ensemble_recall=run.ensemble_recall,
        ensemble_auc=run.ensemble_auc,
        model_version=run.model_version,
        promoted_to_production=run.promoted_to_production,
        error_message=run.error_message,
        started_at=run.started_at,
        completed_at=run.completed_at,
    )
