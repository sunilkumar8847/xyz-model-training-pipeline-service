"""
model-training-pipeline/src/core/config.py
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import Field, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import secrets


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─── Service Identity ────────────────────────────────────────────
    SERVICE_NAME: str = "model-training-pipeline"
    SERVICE_VERSION: str = "3.0.0"
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    LOG_LEVEL: str = "INFO"

    # ─── API ─────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8110
    API_WORKERS: int = 2

    # ─── PostgreSQL (Training run registry) ──────────────────────────
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "xyzmdm"
    POSTGRES_USER: str = "xyzmdm"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_POOL_SIZE: int = 10

    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @computed_field
    @property
    def SYNC_DATABASE_URL(self) -> str:
        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # ─── MLflow ──────────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "xyz-mdm-matching"
    MLFLOW_ARTIFACT_LOCATION: str = "s3://xyz-mdm-artifacts/mlflow"
    MLFLOW_REGISTRY_URI: Optional[str] = None  # Falls back to tracking URI

    # ─── S3 (Model Artifacts) ────────────────────────────────────────
    S3_ARTIFACT_BUCKET: str = "xyz-mdm-artifacts"
    S3_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    S3_ENDPOINT_URL: Optional[str] = None  # LocalStack

    # ─── Feature Store ───────────────────────────────────────────────
    FEATURE_STORE_URL: str = "http://localhost:8115"
    # Training is a batch job: on-the-fly computation of 100 uncached pairs can take tens of
    # seconds on CPU (longer while the Feature Store loads its embedding model).
    FEATURE_STORE_TIMEOUT_SECONDS: int = 120
    FEATURE_STORE_BATCH_SIZE: int = 100          # POST /api/v1/features/batch limit
    FEATURE_STORE_SERVICE_USER: str = "svc-model-training"
    FEATURE_STORE_SERVICE_ROLES: str = "PLATFORM_ADMIN"
    # "offline" = POST /features/offline, point-in-time correct (LLD requirement).
    # "online"  = POST /features/batch, current values, computes on-the-fly for cache
    #             misses. Explicit opt-in only — it cannot prevent label leakage.
    FEATURE_STORE_RETRIEVAL_MODE: str = "offline"
    # Feature catalog version; MUST match the Feature Store's FEATURE_VERSION, because
    # training and serving are only comparable when built from the same catalog.
    FEATURE_CATALOG_VERSION: str = "v2.0.0"
    # Fail the run if fewer than this fraction of requested pairs come back with
    # features. Guards against "trained on almost nothing" passing as success.
    FEATURE_COVERAGE_MIN_RATIO: float = 0.95

    # ─── Trainer selection (execution profile) ────────────────────────
    # Which models actually get trained. Default = the full production ensemble.
    # Set to "xgboost" for the local CPU/16GB profile, where BERT fine-tuning and a
    # real GNN are not feasible. Disabled models are NOT faked: they are reported as
    # not-trained, excluded from the ensemble, and the remaining weights renormalized.
    ENABLED_TRAINERS: str = "transformer,gnn,xgboost"

    @property
    def enabled_trainers(self) -> set[str]:
        return {t.strip().lower() for t in self.ENABLED_TRAINERS.split(",") if t.strip()}

    @property
    def is_partial_ensemble(self) -> bool:
        return self.enabled_trainers != {"transformer", "gnn", "xgboost"}

    # ─── Synthetic development data ──────────────────────────────────
    # A directory produced by the synthetic data generator at the repository root
    # (`python -m synthetic_data --profile dev`). When set, Stage 1 adds that dataset's
    # TRAIN-split labeled pairs, and the online Feature Store client can send the
    # synthetic records' fields for on-the-fly feature computation.
    #
    # DEVELOPMENT / TEST ONLY. Rejected when ENVIRONMENT is staging or production:
    # production must never fall back to synthetic labels. Unset (the default) means
    # no synthetic data is read at all.
    SYNTHETIC_DATA_DIR: Optional[str] = None

    # LEGACY. The old demo dataset (demo/data/training_pairs.json) is no longer a
    # training source. This field exists only so a stale value is REJECTED rather than
    # silently ignored (model_config uses extra="ignore", so removing the field would
    # make an old .env.local quietly change behaviour).
    DEMO_DATA_DIR: Optional[str] = None

    @model_validator(mode="after")
    def _validate_training_data_sources(self) -> "Settings":
        check_training_data_sources(self)
        return self

    # ─── Kafka (Retraining triggers) ─────────────────────────────────
    KAFKA_BROKERS: str = "localhost:9092"
    KAFKA_RETRAINING_TOPIC: str = "mdm.retraining.triggers"
    KAFKA_CONSUMER_GROUP: str = "model-training-consumer"

    @computed_field
    @property
    def KAFKA_BROKERS_LIST(self) -> List[str]:
        return self.KAFKA_BROKERS.split(",")

    # ─── Training Hyperparameters ────────────────────────────────────
    TRANSFORMER_MODEL_NAME: str = "bert-base-uncased"
    TRANSFORMER_MAX_LENGTH: int = 128
    TRANSFORMER_LEARNING_RATE: float = 2e-5
    TRANSFORMER_BATCH_SIZE: int = 64
    TRANSFORMER_EPOCHS: int = 3
    TRANSFORMER_WARMUP_RATIO: float = 0.1
    TRANSFORMER_DROPOUT: float = 0.2
    TRANSFORMER_WEIGHT_DECAY: float = 0.01

    GNN_HIDDEN_DIM: int = 256
    GNN_NUM_LAYERS: int = 3
    GNN_DROPOUT: float = 0.2
    GNN_LEARNING_RATE: float = 1e-3
    GNN_EPOCHS: int = 100

    XGB_N_ESTIMATORS: int = 1000
    XGB_MAX_DEPTH: int = 8
    XGB_LEARNING_RATE: float = 0.05
    XGB_SUBSAMPLE: float = 0.8
    XGB_COLSAMPLE_BYTREE: float = 0.8
    XGB_EARLY_STOPPING_ROUNDS: int = 50

    ENSEMBLE_TRANSFORMER_WEIGHT: float = 0.50
    ENSEMBLE_GNN_WEIGHT: float = 0.30
    ENSEMBLE_XGB_WEIGHT: float = 0.20

    # ─── Evaluation Thresholds ────────────────────────────────────────
    MIN_F1_THRESHOLD: float = 0.91         # Minimum to register in MLflow
    TARGET_F1_THRESHOLD: float = 0.94     # Target production quality
    PROMOTION_F1_DELTA: float = 0.005     # Must beat champion by 0.5%
    PROMOTION_PVALUE: float = 0.05        # Statistical significance
    MAX_LATENCY_P95_MS: float = 100.0     # Latency regression guard

    # ─── Data Collection ─────────────────────────────────────────────
    TRAINING_LOOKBACK_DAYS: int = 30
    MIN_LABELED_PAIRS: int = 50000
    NEGATIVE_SAMPLING_RATIO: int = 5       # 1 positive : 5 negatives
    ACTIVE_LEARNING_BATCH_SIZE: int = 2000
    TEST_SPLIT_RATIO: float = 0.15
    VAL_SPLIT_RATIO: float = 0.10

    # ─── Scheduling ──────────────────────────────────────────────────
    RETRAINING_SCHEDULE: str = "0 2 * * 0"  # Sunday 2 AM UTC
    DRIFT_CHECK_INTERVAL_SECONDS: int = 3600

    # ─── Canary Deployment ───────────────────────────────────────────
    CANARY_INITIAL_TRAFFIC_PCT: int = 10
    CANARY_PHASE_DURATION_HOURS: int = 24
    CANARY_MIN_INFERENCES: int = 1000
    ROLLBACK_F1_DROP_THRESHOLD: float = 0.01
    ROLLBACK_ERROR_RATE_THRESHOLD: float = 0.01
    ROLLBACK_LATENCY_INCREASE_PCT: float = 0.50

    # ─── Kubeflow (Production orchestration) ─────────────────────────
    KUBEFLOW_HOST: Optional[str] = None  # None = run locally
    KUBEFLOW_NAMESPACE: str = "kubeflow"

    # ─── Model Inference Service ─────────────────────────────────────
    MODEL_INFERENCE_SERVICE_URL: str = "http://localhost:8090"


SYNTHETIC_DATA_ENVIRONMENTS = frozenset({Environment.DEVELOPMENT, Environment.TEST})


def check_training_data_sources(s: "Settings") -> None:
    """
    Guard the training-data configuration. Called when settings are loaded AND again
    where synthetic data is actually read, because settings can be mutated at runtime
    (validators do not re-run on attribute assignment).
    """
    if s.DEMO_DATA_DIR:
        raise ValueError(
            "DEMO_DATA_DIR is no longer supported: the old demo dataset is not a "
            "training source. For local development generate a synthetic dataset "
            "(`python -m synthetic_data --profile dev`) and set SYNTHETIC_DATA_DIR "
            "to its output directory. Remove DEMO_DATA_DIR from your environment."
        )
    if s.SYNTHETIC_DATA_DIR and s.ENVIRONMENT not in SYNTHETIC_DATA_ENVIRONMENTS:
        raise ValueError(
            f"SYNTHETIC_DATA_DIR is set but ENVIRONMENT={s.ENVIRONMENT.value}. "
            f"Synthetic training data is only permitted in "
            f"{sorted(e.value for e in SYNTHETIC_DATA_ENVIRONMENTS)}; staging and "
            f"production must use real label sources only."
        )


settings = Settings()
