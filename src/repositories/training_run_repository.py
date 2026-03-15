"""
model-training-pipeline/src/repositories/training_run_repository.py
PostgreSQL persistence for training runs and trained models.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

from sqlalchemy import Column, DateTime, Float, Boolean, Integer, String, Text, Index, select, update
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from src.core.config import settings
from src.domain.models import ModelStatus, RunStatus, RetrainingTrigger, TrainedModel, TrainingRun

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class TrainingRunORM(Base):
    __tablename__ = "training_runs"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    trigger: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="PENDING")
    mlflow_run_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    mlflow_experiment_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    triggered_by: Mapped[str] = mapped_column(String(100), default="system")
    feature_store_version: Mapped[str] = mapped_column(String(50), default="v2.0.0")

    # Data stats
    n_training_pairs: Mapped[int] = mapped_column(Integer, default=0)
    n_positive: Mapped[int] = mapped_column(Integer, default=0)
    n_negative: Mapped[int] = mapped_column(Integer, default=0)

    # Metrics
    transformer_f1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gnn_f1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    xgb_f1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ensemble_f1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ensemble_precision: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ensemble_recall: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ensemble_auc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Model info
    model_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    model_artifact_uri: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    promoted_to_production: Mapped[bool] = mapped_column(Boolean, default=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    data_collected_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    features_extracted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    training_completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    evaluation_completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    registered_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_tr_status", "status"),
        Index("ix_tr_trigger", "trigger"),
        Index("ix_tr_created", "created_at"),
    )


class TrainedModelORM(Base):
    __tablename__ = "trained_models"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    mlflow_model_name: Mapped[str] = mapped_column(String(255), default="xyz-mdm-matcher")
    mlflow_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    model_type: Mapped[str] = mapped_column(String(50), default="ENSEMBLE")
    status: Mapped[str] = mapped_column(String(50), default="STAGING")
    artifact_uri: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    f1_score: Mapped[float] = mapped_column(Float, default=0.0)
    precision: Mapped[float] = mapped_column(Float, default=0.0)
    recall: Mapped[float] = mapped_column(Float, default=0.0)
    auc_roc: Mapped[float] = mapped_column(Float, default=0.0)
    inference_p95_ms: Mapped[float] = mapped_column(Float, default=0.0)

    traffic_pct: Mapped[int] = mapped_column(Integer, default=0)
    feature_store_version: Mapped[str] = mapped_column(String(50), default="v2.0.0")
    rollback_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    promoted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    rolled_back_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_tm_status", "status"),
        Index("ix_tm_run_id", "run_id"),
    )


# ─── Engine ───────────────────────────────────────────────────────────────────

_engine = None
_session_factory = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.DATABASE_URL,
            pool_size=settings.POSTGRES_POOL_SIZE,
            future=True,
        )
    return _engine


def get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), class_=AsyncSession, expire_on_commit=False)
    return _session_factory


async def get_db_session():
    async with get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ─── Repository ──────────────────────────────────────────────────────────────

class TrainingRunRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def upsert(self, run: TrainingRun) -> TrainingRun:
        existing = await self._session.get(TrainingRunORM, run.run_id)
        if existing:
            existing.status = run.status.value
            existing.mlflow_run_id = run.mlflow_run_id
            existing.n_training_pairs = run.n_training_pairs
            existing.n_positive = run.n_positive
            existing.n_negative = run.n_negative
            existing.transformer_f1 = run.transformer_f1
            existing.gnn_f1 = run.gnn_f1
            existing.xgb_f1 = run.xgb_f1
            existing.ensemble_f1 = run.ensemble_f1
            existing.ensemble_precision = run.ensemble_precision
            existing.ensemble_recall = run.ensemble_recall
            existing.ensemble_auc = run.ensemble_auc
            existing.model_version = run.model_version
            existing.promoted_to_production = run.promoted_to_production
            existing.error_message = run.error_message
            existing.started_at = run.started_at
            existing.data_collected_at = run.data_collected_at
            existing.features_extracted_at = run.features_extracted_at
            existing.training_completed_at = run.training_completed_at
            existing.evaluation_completed_at = run.evaluation_completed_at
            existing.registered_at = run.registered_at
            existing.completed_at = run.completed_at
        else:
            orm = TrainingRunORM(
                id=run.run_id,
                trigger=run.trigger.value,
                status=run.status.value,
                triggered_by=run.triggered_by,
                feature_store_version=run.feature_store_version,
            )
            self._session.add(orm)
        return run

    async def get(self, run_id: UUID) -> Optional[TrainingRun]:
        orm = await self._session.get(TrainingRunORM, run_id)
        return self._to_domain(orm) if orm else None

    async def list_recent(self, limit: int = 20) -> List[TrainingRun]:
        result = await self._session.execute(
            select(TrainingRunORM).order_by(TrainingRunORM.created_at.desc()).limit(limit)
        )
        return [self._to_domain(r) for r in result.scalars().all()]

    async def get_latest_successful(self) -> Optional[TrainingRun]:
        result = await self._session.execute(
            select(TrainingRunORM)
            .where(TrainingRunORM.status == "COMPLETED")
            .order_by(TrainingRunORM.completed_at.desc())
            .limit(1)
        )
        orm = result.scalar_one_or_none()
        return self._to_domain(orm) if orm else None

    @staticmethod
    def _to_domain(orm: TrainingRunORM) -> TrainingRun:
        return TrainingRun(
            run_id=orm.id,
            trigger=RetrainingTrigger(orm.trigger),
            status=RunStatus(orm.status),
            mlflow_run_id=orm.mlflow_run_id,
            n_training_pairs=orm.n_training_pairs,
            n_positive=orm.n_positive,
            n_negative=orm.n_negative,
            transformer_f1=orm.transformer_f1,
            gnn_f1=orm.gnn_f1,
            xgb_f1=orm.xgb_f1,
            ensemble_f1=orm.ensemble_f1,
            ensemble_precision=orm.ensemble_precision,
            ensemble_recall=orm.ensemble_recall,
            ensemble_auc=orm.ensemble_auc,
            model_version=orm.model_version,
            promoted_to_production=orm.promoted_to_production,
            error_message=orm.error_message,
            started_at=orm.started_at,
            data_collected_at=orm.data_collected_at,
            features_extracted_at=orm.features_extracted_at,
            training_completed_at=orm.training_completed_at,
            evaluation_completed_at=orm.evaluation_completed_at,
            registered_at=orm.registered_at,
            completed_at=orm.completed_at,
            triggered_by=orm.triggered_by,
            feature_store_version=orm.feature_store_version,
        )
