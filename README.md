# XYZ MDM 3.0 — Model Training Pipeline Service

> **"The model that ships today is always worse than the model shipping next week. Continuous Learning. Automatic Improvement. Production Excellence."**

Automated ML model lifecycle management for the XYZ MDM 3.0 AI plane. Orchestrates end-to-end model training from HITL data collection through champion/challenger production deployment, with continuous drift detection and automatic retraining.

---

## Overview

| Attribute | Value |
|---|---|
| **Service Name** | model-training-pipeline |
| **Bounded Context** | AI Intelligence Context (AI Plane) |
| **Language** | Python 3.11 + PyTorch 2.0 |
| **Criticality** | P0 |
| **Port** | 8007 |
| **Orchestration** | Kubeflow Pipelines 2.0 |
| **Experiment Tracking** | MLflow 2.x |
| **Compute** | AWS SageMaker / Kubernetes GPU Nodes |
| **Artifact Storage** | S3 |
| **Training Data** | Feature Store (point-in-time) |

---

## Service SLOs

| Metric | Target | Minimum | Alert |
|---|---|---|---|
| Training Time (1M samples) | < 2 hours | < 4 hours | > 3 hours |
| F1 Score (Production) | > 94% | > 91% | < 92% |
| Drift Detection Time | < 1 hour | < 2 hours | > 2 hours |
| Retraining Frequency | Weekly | Bi-weekly | > 14 days |
| Rollback Time | < 60 seconds | < 5 minutes | > 2 minutes |

---

## Architecture

The service uses **Domain-Driven Design (DDD) with a Pipeline Pattern**. Eight discrete stages with rich business rules — promotion criteria, rollback thresholds, and champion/challenger management — all converging on a single execution path regardless of trigger source.

```
  ┌─────────────────────────────────────────────────────────────┐
  │           MODEL TRAINING PIPELINE (Kubeflow)                │
  ├─────────────────────────────────────────────────────────────┤
  │                                                             │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
  │  │   Data   │─▶│ Feature  │─▶│  Model   │─▶│  Model   │   │
  │  │Collection│  │ Extract  │  │ Training │  │ Evaluate │   │
  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
  │       ▲                                          │          │
  │       │                                          ▼          │
  │  ┌──────────┐                           ┌──────────────┐   │
  │  │  Kafka   │                           │    MLflow    │   │
  │  │ Consumer │                           │   Registry   │   │
  │  └──────────┘                           └──────┬───────┘   │
  │                                                │           │
  │  ┌──────────┐                        ┌─────────▼────────┐  │
  │  │  Drift   │───────────────────────▶│    Champion /    │  │
  │  │ Detector │                        │    Challenger    │  │
  │  └──────────┘                        └──────────────────┘  │
  └─────────────────────────────────────────────────────────────┘

  Trigger Sources: Scheduler | Kafka | REST API | Emergency
```

---

## Pipeline Stages

| Stage | Duration | Compute | Retry | Output |
|---|---|---|---|---|
| Data Collection | 15 min | CPU | 3x | Labeled pairs dataset |
| Feature Extraction | 30 min | CPU | 3x | Feature matrix (50 dims) |
| Train Transformer | 45 min | GPU A100 | 2x | BERT-base weights |
| Train GNN | 30 min | GPU A100 | 2x | GraphSAGE weights |
| Train XGBoost | 15 min | CPU | 3x | XGBoost model |
| Ensemble Combination | 5 min | CPU | 2x | Combined model |
| Evaluate | 10 min | GPU | 2x | Metrics + SHAP |
| Register | 2 min | CPU | 3x | MLflow entry |

**Total pipeline duration: ~2.5 hours**

---

## Ensemble Model

| Model | Weight | Architecture | Parameters | Strength |
|---|---|---|---|---|
| **Transformer** | 50% | BERT-base fine-tuned | 110M | Semantic similarity |
| **GNN** | 30% | GraphSAGE 3-layer | 5M | Relational patterns |
| **XGBoost** | 20% | 1000 trees, depth 8 | 2M | Tabular features |

Ensemble weights are tuned per training run using **Optuna** (50 trials, 60s timeout) on the validation set.

---

## Training Data

### Label Sources

| Source | Label | Volume / Week | Quality |
|---|---|---|---|
| HITL Approved Merges | Positive | ~5,000 pairs | High (human) |
| HITL Rejected | Negative | ~3,000 pairs | High (human) |
| Auto-Merge (score > 0.99) | Positive | ~10,000 pairs | Medium |
| Synthetic Negatives | Negative | ~15,000 pairs | Medium |
| Active Learning | Uncertain | ~2,000 pairs | High (targeted) |

### Data Requirements

| Requirement | Minimum | Target |
|---|---|---|
| Labeled pairs per training | 50,000 | 250,000+ |
| Positive:Negative ratio | 1:3 | 1:5 |
| Temporal spread | 30 days | 90 days |
| Tenant diversity | 10 tenants | All active tenants |

---

## Evaluation & Promotion

### Accuracy Thresholds

| Metric | Minimum | Target | Stretch |
|---|---|---|---|
| Precision | ≥ 92% | ≥ 95% | ≥ 98% |
| Recall | ≥ 90% | ≥ 93% | ≥ 96% |
| F1 Score | ≥ 91% | ≥ 94% | ≥ 97% |
| AUC-ROC | ≥ 0.94 | ≥ 0.96 | ≥ 0.98 |

### Promotion Criteria (all must pass)

- F1 score must beat current champion by ≥ 0.5%
- Statistical significance: p-value < 0.05 on held-out test set
- Latency: P95 inference < 100ms (no regression)
- Canary: 10% traffic for 24h with no anomalies

---

## Champion / Challenger Rollout

| Phase | Champion | Challenger | Duration | Trigger |
|---|---|---|---|---|
| Canary | 90% | 10% | 24 hours | Auto |
| Ramp 1 | 70% | 30% | 24 hours | Auto |
| Ramp 2 | 50% | 50% | 48 hours | Auto |
| Promotion | 0% | 100% | Permanent | Auto |

### Rollback Triggers (automatic)

| Trigger | Threshold | Action |
|---|---|---|
| F1 Drop | Challenger < Champion - 1% | Auto-rollback |
| Latency Spike | P95 > 150ms (50% increase) | Auto-rollback |
| Error Rate | > 1% inference errors | Auto-rollback |
| Manual Override | On-call decision | Instant rollback |

---

## Drift Detection & Retraining

### Drift Types

| Type | Detection Method | Threshold |
|---|---|---|
| Data Drift | KL divergence on feature distributions | KL > 0.1 sustained 7 days |
| Concept Drift | F1 score degradation on live traffic | F1 drops > 2% over 7 days |
| Label Drift | HITL override rate change | Override rate > 15% |
| Prediction Drift | Score distribution shift | Jensen-Shannon > 0.05 |

### Retraining Triggers

| Trigger | Condition | Priority |
|---|---|---|
| Scheduled | Every Sunday 2:00 AM UTC | Normal |
| Drift Detected | Any drift threshold exceeded | High |
| New Labels | > 10,000 new labeled pairs | Normal |
| Manual | Triggered by ML team | Varies |
| Emergency | F1 < 90% (critical degradation) | Critical |

---

## Project Structure

```
xyz-model-training-pipeline/
├── pyproject.toml
├── alembic.ini
├── pytest.ini
├── Dockerfile
├── .env.example
├── migrations/
│   ├── env.py
│   └── versions/
│       └── 001_initial_schema.py
└── src/
    ├── main.py                              # FastAPI app + lifespan
    ├── cli.py                               # CLI: train/promote/rollback/list-models
    ├── core/
    │   ├── config.py                        # Pydantic Settings
    │   └── metrics.py                       # Prometheus metrics
    ├── domain/
    │   └── models.py                        # LabeledPair, TrainingDataset,
    │                                        # TrainingRun, TrainedModel,
    │                                        # ModelEvaluation, CanaryMetrics,
    │                                        # RetrainingTriggerEvent
    ├── pipeline/
    │   ├── training_pipeline.py             # 8-stage orchestrator +
    │   │                                    # ChampionChallengerManager
    │   └── stages/
    │       ├── data_collection.py           # HITL, auto-merge, synthetic neg,
    │       │                                # active learning
    │       └── feature_extraction.py        # Point-in-time Feature Store retrieval
    ├── trainers/
    │   └── ensemble_trainer.py              # TransformerTrainer + GNNTrainer +
    │                                        # XGBoostTrainer + EnsembleTrainer
    │                                        # (Optuna weight tuning)
    ├── evaluators/
    │   └── model_evaluator.py               # SHAP, Mann-Whitney U, ONNX export
    ├── registry/
    │   └── mlflow_registry.py               # Full MLflow lifecycle + rollback
    ├── repositories/
    │   └── training_run_repository.py       # SQLAlchemy ORM + async repository
    ├── workers/
    │   └── retraining_trigger_consumer.py   # Kafka consumer + APScheduler cron
    └── api/v1/
        ├── schemas.py                       # Pydantic v2 DTOs
        └── endpoints/
            └── training.py                  # All REST endpoints
```

---

## Prerequisites

- Python 3.11+
- Docker + Docker Compose (for shared infrastructure)
- Shared infra running: PostgreSQL, Kafka, MLflow, LocalStack (S3)
- Feature Store service running on port 8006

---

## Local Setup

### 1. Start shared infrastructure

```bash
cd infra
docker compose up -d
```

Verify all required services are healthy:

```bash
docker compose ps
# mlflow        → running (port 5000)
# kafka         → healthy
# postgres      → healthy
# localstack    → healthy
```

### 2. Install dependencies

```bash
cd xyz-model-training-pipeline
pip install -e ".[dev]"
# or if using Poetry:
poetry install --no-root
```

### 3. Configure environment

```bash
cp .env.example .env
```

Key variables to verify:

```env
# Service
SERVICE_PORT=8007
ENVIRONMENT=development

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Feature Store
FEATURE_STORE_URL=http://localhost:8006

# S3 Artifacts
S3_ARTIFACT_BUCKET=xyz-mdm-artifacts
S3_ENDPOINT_URL=https://localhost:4566
AWS_ACCESS_KEY_ID=test
AWS_SECRET_ACCESS_KEY=test

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=model_training
POSTGRES_USER=training_user
POSTGRES_PASSWORD=training_pass

# Kafka
KAFKA_BROKERS=localhost:9092

# Model thresholds
MIN_F1_THRESHOLD=0.91
ENSEMBLE_TRANSFORMER_WEIGHT=0.50
ENSEMBLE_GNN_WEIGHT=0.30
ENSEMBLE_XGB_WEIGHT=0.20

# Scheduling
RETRAINING_SCHEDULE=0 2 * * 0
DRIFT_CHECK_INTERVAL_SECONDS=3600
```

### 4. Run database migrations

```bash
alembic upgrade head
```

### 5. Start the service

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8007 --reload
```

Expected startup output:
```
Connected to PostgreSQL
MLflow tracking: CONNECTED (http://localhost:5000)
Feature Store: CONNECTED (http://localhost:8006)
Kafka consumer: STARTED (topic: mdm.model.retraining)
Retraining scheduler: STARTED (cron: 0 2 * * 0)
Drift check scheduler: STARTED (interval: 3600s)
model-training-pipeline startup complete. Docs at /docs
```

---

## API Reference

**Swagger UI:** http://localhost:8007/docs  
**MLflow UI:** http://localhost:5000  
**Metrics:** http://localhost:8007/metrics

### Trigger a training run

```
POST /api/v1/training/runs
```

```json
{
  "trigger": "MANUAL",
  "reason": "Initial training run",
  "tenant_id": "tenant-123"
}
```

Returns `202 Accepted` with `run_id`.

### Get training run status

```
GET /api/v1/training/runs/{run_id}
```

### List recent training runs

```
GET /api/v1/training/runs
```

### Promote a model version

```
POST /api/v1/models/promote
```

```json
{
  "version": 5,
  "stage": "production"
}
```

### Emergency rollback

```
POST /api/v1/models/rollback
```

```json
{
  "reason": "F1 degradation observed in production"
}
```

### List all model versions

```
GET /api/v1/models
```

### Get current champion model

```
GET /api/v1/models/champion
```

### Health check

```
GET /api/v1/health
```

---

## CLI

The service ships with a CLI for operations team use:

```bash
# Trigger a training run manually
python -m src.cli train --trigger MANUAL --reason "quarterly refresh"

# Promote a specific model version to production
python -m src.cli promote --version 5 --stage production

# Emergency rollback
python -m src.cli rollback --reason "F1 degradation"

# List all registered model versions
python -m src.cli list-models
```

---

## Observability

### Prometheus Metrics

| Metric | Type | Labels | Alert |
|---|---|---|---|
| `training_duration_seconds` | Histogram | `model`, `version` | > 4 hours |
| `model_f1_score` | Gauge | `model`, `version` | < 0.91 |
| `drift_score` | Gauge | `drift_type` | > 0.1 |
| `retraining_triggered_total` | Counter | `trigger_type` | — |
| `rollback_events_total` | Counter | `reason` | > 0 |
| `labeled_pairs_total` | Counter | `source` | < 1,000/week |

All metrics available at: `GET /metrics`

---

## Infrastructure Dependencies

| Service | Purpose | Port | Container |
|---|---|---|---|
| `mlflow` | Experiment tracking + model registry | 5000 | `xyz-mlflow` |
| `postgres` | Training run metadata | 5432 | `xyz-postgres` |
| `kafka` | Retraining trigger events | 9092 | `xyz-kafka` |
| `localstack` | S3 model artifact storage | 4566 | `xyz-localstack` |
| `feature-store` | Point-in-time training data | 8006 | *(external service)* |

**S3 bucket:** `xyz-mdm-artifacts`  
**S3 artifact path:** `models/{model_name}/{version}/`  
**Kafka topic:** `mdm.model.retraining`

---

## Key Design Decisions

**DDD with Pipeline Pattern** — 8 discrete pipeline stages with explicit retry policies per stage (GPU stages retry 2x, CPU stages 3x). Rich domain models capture business rules like promotion criteria and rollback thresholds explicitly, not scattered through procedural code.

**Single trigger path** — Whether triggered by Kafka event, REST API, APScheduler cron, or emergency F1 threshold breach, all execution flows through the same `TrainingPipeline.run()` method. No special cases.

**Optuna ensemble tuning** — Rather than hardcoding ensemble weights, each training run uses Optuna to find optimal Transformer/GNN/XGBoost weights on the validation set (50 trials, 60s budget). This means weights adapt as the data distribution evolves.

**Statistical promotion gate** — Mann-Whitney U test (p < 0.05) is required before any model promotion. A challenger that looks better on aggregate metrics but lacks statistical significance is held back.

**ONNX export on registration** — All models are exported to ONNX format at registration time, ensuring the inference service can load any registered model version without framework dependencies.

**Automatic rollback** — The `ChampionChallengerManager` monitors live canary metrics every 60 seconds. Rollback requires no human intervention — it fires automatically when any threshold is breached and completes in under 60 seconds.

---

## Running Tests

```bash
pytest tests/unit/ -v
pytest tests/integration/ -v --require-infra
```

---

## Upstream / Downstream

| Service | Relationship |
|---|---|
| **feature-store** (port 8006) | Upstream — provides point-in-time training features via `POST /api/v1/features/offline` |
| **model-inference-service** (port 8000) | Downstream — loads trained models from MLflow registry for real-time serving |
| **matching-service** (port 8001) | Downstream — uses promoted champion model for entity pair scoring |
| **HITL workflows** | Upstream — labeled pairs from human reviewers feed the training data pipeline |