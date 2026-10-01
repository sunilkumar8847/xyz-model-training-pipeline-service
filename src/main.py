"""
model-training-pipeline/src/main.py
FastAPI application + lifespan for the model training pipeline service.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
import uvicorn
from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from uuid import uuid4

import os

from src.api.v1.endpoints.training import router as training_router
from xyz_security import get_current_tenant
from src.core.config import settings
from src.core.exceptions import TrainingPipelineError
from src.repositories.training_run_repository import Base, get_engine, get_session_factory

# ─── Logging ──────────────────────────────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.dev.ConsoleRenderer()
        if settings.ENVIRONMENT.value == "development"
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(),
)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    logger.info(
        f"Starting {settings.SERVICE_NAME} v{settings.SERVICE_VERSION} "
        f"[{settings.ENVIRONMENT.value}]"
    )

    # 1. Create database tables
    try:
        async with get_engine().begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database schema: READY")
    except Exception as e:
        logger.error(f"Database init failed: {e}")

    # 2. Configure MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        logger.info(f"MLflow tracking URI: {settings.MLFLOW_TRACKING_URI}")
    except Exception as e:
        logger.warning(f"MLflow init failed: {e}")

    # 3. Start Kafka consumer for retraining triggers
    kafka_task = None
    try:
        from src.workers.retraining_trigger_consumer import (
            PipelineLauncher, RetrainingTriggerConsumer
        )
        launcher = PipelineLauncher(session_factory=get_session_factory())
        consumer = RetrainingTriggerConsumer(pipeline_launcher=launcher)
        kafka_task = asyncio.create_task(_run_with_restart(consumer.run))
        logger.info("Kafka retraining trigger consumer: STARTED")
    except Exception as e:
        logger.warning(f"Kafka consumer init failed (non-critical): {e}")

    # 4. Start scheduled retraining worker
    scheduler_task = None
    try:
        from src.workers.retraining_trigger_consumer import (
            PipelineLauncher, ScheduledRetrainingWorker
        )
        launcher = PipelineLauncher(session_factory=get_session_factory())
        sched_worker = ScheduledRetrainingWorker(launcher)
        scheduler_task = asyncio.create_task(_run_with_restart(sched_worker.start))
        logger.info(f"Scheduled retraining worker: STARTED (cron: {settings.RETRAINING_SCHEDULE})")
    except Exception as e:
        logger.warning(f"Scheduler init failed (non-critical): {e}")

    # 5. Start drift detection worker
    drift_task = None
    drift_detector = None
    try:
        from src.workers.drift_detector import DriftDetector
        drift_detector = DriftDetector()
        await asyncio.wait_for(drift_detector.start(), timeout=5.0)
        drift_task = asyncio.create_task(_run_with_restart(drift_detector.run))
        logger.info(
            f"Drift detection worker: STARTED "
            f"(interval={settings.DRIFT_CHECK_INTERVAL_SECONDS}s, "
            f"KL threshold={0.1}, F1 drop threshold={0.03})"
        )
    except BaseException as e:
        logger.warning(f"Drift detector init failed (non-critical): {e}")

    logger.info(f"{settings.SERVICE_NAME} startup complete. Docs: /docs")
    yield

    # Shutdown
    logger.info("Shutting down model-training-pipeline...")
    for task in [kafka_task, scheduler_task, drift_task]:
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    if drift_detector:
        try:
            await drift_detector.stop()
        except Exception:
            pass
    await get_engine().dispose()
    logger.info("Shutdown complete")


async def _run_with_restart(coro_fn):
    """Run a coroutine with automatic restart on failure."""
    while True:
        try:
            await coro_fn()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Worker crashed: {e}. Restarting in 30s...")
            await asyncio.sleep(30)


# ─── App Factory ──────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="XYZ MDM — Model Training Pipeline",
        description=(
            "Automated ML lifecycle management: "
            "BERT + GNN + XGBoost ensemble training, MLflow registry, "
            "champion/challenger deployment, drift-triggered retraining."
        ),
        version="1.0.0",
        contact={"name": "XYZ MDM Platform", "email": "engineering@xyzmdm.com"},
        servers=[{"url": "http://localhost:8036", "description": "Integration"}],
        docs_url="/swagger-ui.html",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    _cors_origins = [o.strip() for o in os.environ.get("CORS_ALLOWED_ORIGINS", "").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Verified-Tenant-ID",
                       "X-Verified-User-ID", "X-Verified-Roles", "X-Verified-Platform",
                       "X-Request-Timestamp", "X-Gateway-Signature"],
    )

    app.include_router(training_router, prefix="/api", dependencies=[Depends(get_current_tenant)])

    # ── Structured error envelope (LLD PART IX) ──────────────────────
    @app.exception_handler(TrainingPipelineError)
    async def training_pipeline_error_handler(request: Request, exc: TrainingPipelineError):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "correlation_id": str(uuid4()),
                }
            },
        )

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": settings.SERVICE_NAME,
            "version": settings.SERVICE_VERSION,
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=1,  # Training pipeline: 1 worker (GPU memory)
        reload=False,
        log_level="info",
    )
