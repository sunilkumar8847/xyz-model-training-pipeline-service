"""
model-training-pipeline/src/core/config.py
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import Field, computed_field, field_validator
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
    FEATURE_STORE_TIMEOUT_SECONDS: int = 30

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


settings = Settings()
