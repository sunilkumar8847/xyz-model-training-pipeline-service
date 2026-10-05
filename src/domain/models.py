"""
model-training-pipeline/src/domain/models.py

Core domain models for the Model Training Pipeline.
Pure Python dataclasses — no ORM or framework concerns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4


# ─── Enums ──────────────────────────────────────────────────────────────────

class RunStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ModelStatus(str, Enum):
    TRAINING = "TRAINING"
    STAGING = "STAGING"
    CANARY = "CANARY"
    PRODUCTION = "PRODUCTION"
    ARCHIVED = "ARCHIVED"
    ROLLED_BACK = "ROLLED_BACK"


class RetrainingTrigger(str, Enum):
    SCHEDULED = "SCHEDULED"
    DRIFT_DETECTED = "DRIFT_DETECTED"
    NEW_LABELS = "NEW_LABELS"
    MANUAL = "MANUAL"
    EMERGENCY = "EMERGENCY"


class ModelType(str, Enum):
    TRANSFORMER = "TRANSFORMER"
    GNN = "GNN"
    XGBOOST = "XGBOOST"
    ENSEMBLE = "ENSEMBLE"


class DeploymentPhase(str, Enum):
    CANARY = "CANARY"        # 10% traffic, 24h
    RAMP_1 = "RAMP_1"        # 30% traffic, 24h
    RAMP_2 = "RAMP_2"        # 50% traffic, 48h
    PROMOTION = "PROMOTION"  # 100% traffic, permanent
    ROLLED_BACK = "ROLLED_BACK"


# ─── Training Data ───────────────────────────────────────────────────────────

@dataclass
class LabeledPair:
    """A labeled entity pair for training (positive = match, negative = no match)."""
    entity_id_1: str
    entity_id_2: str
    tenant_id: str
    label: int          # 1 = match, 0 = no match
    confidence: float   # 0.0-1.0 confidence in label
    source: str         # hitl_merge | hitl_reject | auto_merge | synthetic | active_learning
    labeled_at: datetime = field(default_factory=datetime.utcnow)
    id: UUID = field(default_factory=uuid4)


@dataclass
class TrainingDataset:
    """A fully prepared training dataset for one pipeline run."""
    run_id: UUID
    pairs: List[LabeledPair]
    feature_matrix: Optional["np.ndarray"] = None  # (N, 50) float32
    labels: Optional["np.ndarray"] = None          # (N,) int32
    entity_texts_1: Optional[List[str]] = None     # For transformer
    entity_texts_2: Optional[List[str]] = None     # For transformer
    train_indices: Optional[List[int]] = None
    val_indices: Optional[List[int]] = None
    test_indices: Optional[List[int]] = None
    # Column names of feature_matrix, in order, as declared by the Feature Store's
    # offline response (None when the source did not declare them).
    feature_names: Optional[List[str]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def n_samples(self) -> int:
        return len(self.pairs)

    @property
    def n_positive(self) -> int:
        return sum(1 for p in self.pairs if p.label == 1)

    @property
    def n_negative(self) -> int:
        return sum(1 for p in self.pairs if p.label == 0)

    @property
    def positive_rate(self) -> float:
        return self.n_positive / max(self.n_samples, 1)


# ─── Training Run ────────────────────────────────────────────────────────────

@dataclass
class TrainingRun:
    """
    A complete end-to-end training pipeline execution.
    Tracks all stages from data collection to model registration.
    """
    run_id: UUID = field(default_factory=uuid4)
    trigger: RetrainingTrigger = RetrainingTrigger.MANUAL
    status: RunStatus = RunStatus.PENDING
    mlflow_run_id: Optional[str] = None
    mlflow_experiment_id: Optional[str] = None

    # Stage timestamps
    started_at: Optional[datetime] = None
    data_collected_at: Optional[datetime] = None
    features_extracted_at: Optional[datetime] = None
    training_completed_at: Optional[datetime] = None
    evaluation_completed_at: Optional[datetime] = None
    registered_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Stage metrics
    n_training_pairs: int = 0
    n_positive: int = 0
    n_negative: int = 0

    # Dataset identity (LLD PART III §3.2) — a deterministic fingerprint of the
    # exact labeled pairs used, so a run can be reproduced from its dataset_version.
    dataset_version: Optional[str] = None
    # Lineage of the labels behind dataset_version (not persisted to the run table;
    # written to MLflow tags and the ensemble manifest):
    #   label_sources        {source: n_pairs}, e.g. {"synthetic": 6606}
    #   dataset_provenance   identity of each file-based dataset that was read
    #   feature_as_of        the point-in-time bound the features were read at
    label_sources: Dict = field(default_factory=dict)
    dataset_provenance: List[Dict] = field(default_factory=list)
    feature_as_of: Optional[datetime] = None

    # Model metrics (from evaluation)
    transformer_f1: Optional[float] = None
    gnn_f1: Optional[float] = None
    xgb_f1: Optional[float] = None
    ensemble_f1: Optional[float] = None
    ensemble_precision: Optional[float] = None
    ensemble_recall: Optional[float] = None
    ensemble_auc: Optional[float] = None

    # Registered model info
    model_version: Optional[str] = None
    model_artifact_uri: Optional[str] = None
    promoted_to_production: bool = False
    error_message: Optional[str] = None

    # Configuration snapshot
    hyperparameters: Dict = field(default_factory=dict)
    feature_store_version: str = "v2.0.0"
    triggered_by: str = "system"

    def duration_seconds(self, stage_end: Optional[datetime] = None) -> Optional[float]:
        if self.started_at and stage_end:
            return (stage_end - self.started_at).total_seconds()
        return None


# ─── Trained Model ───────────────────────────────────────────────────────────

@dataclass
class TrainedModel:
    """
    A model artifact registered in MLflow, ready for deployment.
    Tracks champion/challenger status and canary rollout state.
    """
    model_id: UUID = field(default_factory=uuid4)
    run_id: UUID = field(default_factory=uuid4)
    mlflow_model_name: str = "xyz-mdm-matcher"
    mlflow_version: Optional[str] = None
    model_type: ModelType = ModelType.ENSEMBLE
    status: ModelStatus = ModelStatus.STAGING
    artifact_uri: Optional[str] = None
    onnx_artifact_uri: Optional[str] = None

    # Performance
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    auc_roc: float = 0.0
    inference_p95_ms: float = 0.0

    # Deployment
    deployment_phase: Optional[DeploymentPhase] = None
    traffic_pct: int = 0
    canary_started_at: Optional[datetime] = None
    promoted_at: Optional[datetime] = None
    rolled_back_at: Optional[datetime] = None
    rollback_reason: Optional[str] = None

    # Lineage
    training_data_hash: Optional[str] = None
    feature_store_version: str = "v2.0.0"
    framework_version: str = "pytorch-2.1"
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_production(self) -> bool:
        return self.status == ModelStatus.PRODUCTION

    @property
    def is_canary(self) -> bool:
        return self.status == ModelStatus.CANARY


# ─── Evaluation Results ──────────────────────────────────────────────────────

@dataclass
class ModelEvaluation:
    """Comprehensive evaluation results for a trained model."""
    model_id: UUID
    run_id: UUID
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    average_precision: float
    confusion_matrix: List[List[int]]  # [[TN, FP], [FN, TP]]
    threshold: float = 0.5
    n_test_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0
    # Per-threshold analysis
    thresholds: List[float] = field(default_factory=list)
    precision_at_threshold: List[float] = field(default_factory=list)
    recall_at_threshold: List[float] = field(default_factory=list)
    # Feature importance (from XGBoost/SHAP)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    # Statistical significance vs champion
    champion_f1: Optional[float] = None
    p_value: Optional[float] = None
    is_significantly_better: bool = False
    # Bias check (LLD PART VI §6.2)
    bias_passes: bool = True
    bias_f1_per_tenant: Dict[str, float] = field(default_factory=dict)
    evaluated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def passes_promotion_criteria(self) -> bool:
        """True if this model is eligible for production promotion."""
        if self.f1_score < 0.91:
            return False
        if self.champion_f1 and self.f1_score < self.champion_f1 + 0.005:
            return False
        if not self.bias_passes:
            return False
        return True


# ─── Retraining Trigger Event ────────────────────────────────────────────────

@dataclass
class RetrainingTriggerEvent:
    """Event that triggers a new training run (from Kafka or scheduler)."""
    trigger_type: RetrainingTrigger
    priority: str = "NORMAL"  # CRITICAL | HIGH | NORMAL
    reason: str = ""
    drift_score: Optional[float] = None
    new_label_count: Optional[int] = None
    triggered_at: datetime = field(default_factory=datetime.utcnow)
    event_id: UUID = field(default_factory=uuid4)


# ─── Canary Metrics ──────────────────────────────────────────────────────────

@dataclass
class CanaryMetrics:
    """Live metrics collected during canary deployment."""
    model_id: UUID
    inference_count: int = 0
    error_count: int = 0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    rolling_f1: Optional[float] = None
    override_rate: float = 0.0  # HITL override rate
    collected_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def error_rate(self) -> float:
        return self.error_count / max(self.inference_count, 1)

    def should_rollback(
        self,
        champion_f1: float,
        champion_p95_ms: float,
    ) -> Tuple[bool, str]:
        """Returns (should_rollback, reason)."""
        if self.rolling_f1 and self.rolling_f1 < champion_f1 - 0.01:
            return True, f"F1 {self.rolling_f1:.3f} < champion {champion_f1:.3f} - 0.01"
        if champion_p95_ms > 0 and self.p95_latency_ms > champion_p95_ms * 1.5:
            return True, f"P95 latency {self.p95_latency_ms:.0f}ms exceeds 150% of champion"
        if self.error_rate > 0.01:
            return True, f"Error rate {self.error_rate:.3%} > 1%"
        return False, ""
