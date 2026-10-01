"""
LIVE smoke test of the local XGBoost path, on the real code path end to end.

Dependency type:
  - REAL Feature Store service (offline retrieval, point-in-time) — FEATURE_STORE_URL
  - REAL synthetic development labels — SYNTHETIC_DATA_DIR (train split only)
  - REAL splitter, EnsembleTrainer (XGBoost only), ModelEvaluator, xgboost, MLflow
  - MLflow isolated to a temporary SQLite store (does not touch the dev registry)

Skipped when the Feature Store is unreachable or SYNTHETIC_DATA_DIR is unset.

    ./.venv/Scripts/python.exe -m pytest tests/integration/test_xgboost_smoke_live.py -v -s

Metrics printed here are from a small SUBSET of SYNTHETIC data. They verify that the
code path works; they say nothing about production accuracy.
"""
from __future__ import annotations

import random

import httpx
import numpy as np
import pytest

from src.core.config import settings

SUBSET_POSITIVES = 120  # plus ~5x negatives


def _feature_store_up() -> bool:
    try:
        return httpx.get(f"{settings.FEATURE_STORE_URL}/api/v1/health", timeout=5).status_code == 200
    except Exception:
        return False


@pytest.fixture
def live_prerequisites(tmp_path, monkeypatch):
    if not settings.SYNTHETIC_DATA_DIR:
        pytest.skip("SYNTHETIC_DATA_DIR is not configured")
    if not _feature_store_up():
        pytest.skip(f"Feature Store not reachable at {settings.FEATURE_STORE_URL}")
    if settings.FEATURE_STORE_RETRIEVAL_MODE != "offline":
        pytest.fail("smoke test requires FEATURE_STORE_RETRIEVAL_MODE=offline")

    import mlflow
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    monkeypatch.setattr(settings, "ENABLED_TRAINERS", "xgboost")
    yield mlflow
    while mlflow.active_run():
        mlflow.end_run()


async def test_xgboost_smoke_on_real_feature_store(live_prerequisites, tmp_path):
    import mlflow.xgboost
    import xgboost as xgb
    from src.adapters.feature_store_client import FeatureStoreClient
    from src.adapters.synthetic_data import load_synthetic_labeled_pairs
    from src.domain.models import RetrainingTrigger, TrainingRun
    from src.evaluators.model_evaluator import ModelEvaluator
    from src.pipeline.stages.data_collection import TrainingDataSplitter
    from src.pipeline.stages.feature_extraction import FeatureExtractionStage
    from src.trainers.ensemble_trainer import EnsembleTrainer

    mlflow = live_prerequisites

    # 1. Controlled subset of the REAL synthetic training split
    all_pairs = load_synthetic_labeled_pairs(settings.SYNTHETIC_DATA_DIR)
    rng = random.Random(0)
    pos = [p for p in all_pairs if p.label == 1]
    neg = [p for p in all_pairs if p.label == 0]
    subset = rng.sample(pos, SUBSET_POSITIVES) + rng.sample(neg, SUBSET_POSITIVES * 5)

    # 2. REAL offline features from the running Feature Store
    run = TrainingRun(trigger=RetrainingTrigger.MANUAL, triggered_by="phase3-smoke")
    dataset = await FeatureExtractionStage(FeatureStoreClient.from_settings()).execute(run, subset)
    assert dataset.feature_matrix.shape == (len(subset), 50), dataset.feature_matrix.shape
    assert dataset.labels.shape == (len(subset),)
    assert dataset.feature_names is not None and len(dataset.feature_names) == 50
    assert int(dataset.labels.sum()) == SUBSET_POSITIVES

    # 3. REAL split, train, evaluate
    splitter = TrainingDataSplitter(settings.TEST_SPLIT_RATIO, settings.VAL_SPLIT_RATIO)
    dataset.train_indices, dataset.val_indices, dataset.test_indices = splitter.split(dataset.pairs)
    trainer = EnsembleTrainer()
    with mlflow.start_run():
        models = trainer.train_all(dataset)
        evaluation = ModelEvaluator(trainer).evaluate(models, dataset, model_id="smoke", run_id=run.run_id)

    model = models["xgb"]["model"]
    assert isinstance(model, xgb.XGBClassifier) and model.n_features_in_ == 50
    assert models["transformer"]["trained"] is False and models["gnn"]["trained"] is False
    assert evaluation.n_test_samples == len(dataset.test_indices)

    # 4. Real predictions, serialization and reload
    X_test = dataset.feature_matrix[dataset.test_indices]
    proba = model.predict_proba(X_test)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    path = tmp_path / "xgboost_model"
    mlflow.xgboost.save_model(model, str(path))
    reloaded = mlflow.xgboost.load_model(str(path))
    assert np.allclose(reloaded.predict_proba(X_test)[:, 1], proba)
    single = reloaded.predict_proba(X_test[:1])
    assert single.shape == (1, 2)

    print(
        f"\n[SMOKE — synthetic subset, NOT production] pairs={len(subset)} "
        f"split={len(dataset.train_indices)}/{len(dataset.val_indices)}/{len(dataset.test_indices)} "
        f"test pos/neg={evaluation.n_positive}/{evaluation.n_negative} "
        f"P={evaluation.precision:.4f} R={evaluation.recall:.4f} F1={evaluation.f1_score:.4f} "
        f"AUC={evaluation.auc_roc:.4f} AP={evaluation.average_precision:.4f} "
        f"CM={evaluation.confusion_matrix} trees={model.get_booster().num_boosted_rounds()}"
    )
