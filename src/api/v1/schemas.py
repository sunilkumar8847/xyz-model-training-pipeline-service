"""
model-training-pipeline/src/api/v1/schemas.py
Pydantic DTOs for all API contracts.
"""
from __future__ import annotations
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field


class TriggerTrainingRequest(BaseModel):
    trigger_type: str = Field(default="MANUAL", description="MANUAL | SCHEDULED | DRIFT_DETECTED | EMERGENCY")
    reason: str = Field(default="", description="Human-readable reason for triggering")
    tune_weights: bool = Field(default=False, description="If true, use Optuna to tune ensemble weights")


class TrainingRunResponse(BaseModel):
    run_id: UUID
    trigger: str
    status: str
    triggered_by: str
    mlflow_run_id: Optional[str]
    n_training_pairs: int
    n_positive: int
    n_negative: int
    transformer_f1: Optional[float]
    gnn_f1: Optional[float]
    xgb_f1: Optional[float]
    ensemble_f1: Optional[float]
    ensemble_precision: Optional[float]
    ensemble_recall: Optional[float]
    ensemble_auc: Optional[float]
    model_version: Optional[str]
    promoted_to_production: bool
    error_message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]


class PromoteRequest(BaseModel):
    model_version: str = Field(description="MLflow model version to promote")
    target_stage: str = Field(description="staging | canary | production")


class PromoteResponse(BaseModel):
    model_version: str
    stage: str
    success: bool
    message: str


class RollbackRequest(BaseModel):
    reason: str = Field(description="Reason for rollback")


class RollbackResponse(BaseModel):
    rolled_back_to_version: Optional[str]
    reason: str
    success: bool


class ModelVersionResponse(BaseModel):
    version: str
    stage: str
    f1_score: Optional[str]
    run_id: Optional[str]
    created_at: Optional[int]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: datetime
    checks: Dict[str, bool]
