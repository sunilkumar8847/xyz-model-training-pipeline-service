"""
Phase 3 contract tests for the local XGBoost training path.

Dependency type:
  - REAL xgboost, scikit-learn, shap and MLflow (MLflow pointed at a temporary SQLite
    store and artifact directory, never the developer's store).
  - Feature data: a small in-memory 50-column matrix with a known signal. These tests
    exercise the training/evaluation CODE; they make no claim about model quality.
  - Feature Store: test double (httpx.MockTransport) where Stage 2 is involved.
"""
from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from src.core.config import Environment, settings
from src.core.exceptions import InfrastructureUnavailableError, InsufficientTrainingDataError
from src.domain.models import LabeledPair, RetrainingTrigger, TrainingDataset, TrainingRun
from src.pipeline.stages.data_collection import TrainingDataSplitter, enforce_minimum_training_pairs

N_FEATURES = 50
FEATURE_NAMES = sorted(f"f{i:02d}_signal" if i < 5 else f"f{i:02d}_noise" for i in range(N_FEATURES))
T1 = "00000000-0000-0000-0000-000000000001"


def _run() -> TrainingRun:
    return TrainingRun(trigger=RetrainingTrigger.MANUAL, triggered_by="unit-test")


def _dataset(n_clusters=160, seed=0) -> TrainingDataset:
    """Clusters of 3 records (3 positives each) + 5x random negatives. Positives get
    higher values on the first five columns, so a real model can learn something."""
    rng = np.random.default_rng(seed)
    pairs, rows, labels = [], [], []
    for c in range(n_clusters):
        ents = [f"c{c}r{r}" for r in range(3)]
        for i in range(3):
            for j in range(i + 1, 3):
                pairs.append(LabeledPair(ents[i], ents[j], T1, 1, 1.0, "synthetic"))
    n_pos = len(pairs)
    seen = set()
    while len(pairs) < n_pos * 6:
        a, b = rng.choice(n_clusters, 2, replace=False)
        e1, e2 = f"c{a}r{rng.integers(3)}", f"c{b}r{rng.integers(3)}"
        if (e1, e2) not in seen:
            seen.add((e1, e2))
            pairs.append(LabeledPair(e1, e2, T1, 0, 1.0, "synthetic"))
    for p in pairs:
        x = rng.uniform(0, 1, N_FEATURES)
        x[:5] = np.clip(rng.normal(0.8 if p.label else 0.35, 0.18, 5), 0, 1)
        rows.append(x)
        labels.append(p.label)
    ds = TrainingDataset(
        run_id=_run().run_id, pairs=pairs,
        feature_matrix=np.array(rows, dtype=np.float32),
        labels=np.array(labels, dtype=np.int32),
        feature_names=FEATURE_NAMES,
    )
    ds.train_indices, ds.val_indices, ds.test_indices = TrainingDataSplitter(0.15, 0.10).split(pairs)
    return ds


@pytest.fixture
def isolated_mlflow(tmp_path, monkeypatch):
    import mlflow
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setattr(settings, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(settings, "MLFLOW_REGISTRY_URI", uri)
    monkeypatch.setattr(settings, "MLFLOW_ARTIFACT_LOCATION", (tmp_path / "artifacts").as_uri())
    monkeypatch.setattr(settings, "MLFLOW_EXPERIMENT_NAME", "phase3-unit")
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    # mlflow caches the active experiment id process-wide; select one that exists in
    # THIS fresh store so ids from an earlier test's store never leak in.
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
    yield mlflow
    while mlflow.active_run():
        mlflow.end_run()


@pytest.fixture
def xgb_only(monkeypatch):
    monkeypatch.setattr(settings, "ENABLED_TRAINERS", "xgboost")
    monkeypatch.setattr(settings, "XGB_N_ESTIMATORS", 200)  # keep unit tests fast


# ─── 1. Environment-aware minimum ─────────────────────────────────────────────

class TestMinimumTrainingPairs:
    @pytest.mark.parametrize("env", [Environment.DEVELOPMENT, Environment.TEST])
    def test_development_dataset_size_allowed(self, monkeypatch, env):
        monkeypatch.setattr(settings, "ENVIRONMENT", env)
        monkeypatch.setattr(settings, "MIN_LABELED_PAIRS", 50000)
        enforce_minimum_training_pairs(6606)  # the synthetic dev training split

    @pytest.mark.parametrize("env", [Environment.STAGING, Environment.PRODUCTION])
    def test_staging_and_production_enforce_the_minimum(self, monkeypatch, env):
        monkeypatch.setattr(settings, "ENVIRONMENT", env)
        monkeypatch.setattr(settings, "MIN_LABELED_PAIRS", 50000)
        with pytest.raises(InsufficientTrainingDataError) as exc:
            enforce_minimum_training_pairs(6606)
        assert exc.value.code == "MT_1001" and exc.value.minimum == 50000

    def test_production_passes_at_the_minimum(self, monkeypatch):
        monkeypatch.setattr(settings, "ENVIRONMENT", Environment.PRODUCTION)
        monkeypatch.setattr(settings, "MIN_LABELED_PAIRS", 50000)
        enforce_minimum_training_pairs(50000)

    def test_shipped_minimum_unchanged(self):
        assert type(settings).model_fields["MIN_LABELED_PAIRS"].default == 50000


# ─── 2-4. Stage 2 data contract ───────────────────────────────────────────────

def _offline_transport(vectors: dict, names):
    def handler(request):
        body = json.loads(request.content)
        results = [
            {"entity_id_1": a, "entity_id_2": b, "feature_vector": vectors[(a, b)],
             "computed_at": "2026-09-30T00:00:00"}
            for a, b in body["entity_pairs"] if (a, b) in vectors
        ]
        return httpx.Response(200, json={
            "request_id": "00000000-0000-0000-0000-0000000000aa",
            "pairs_requested": len(body["entity_pairs"]), "pairs_found": len(results),
            "as_of_timestamp": body["as_of_timestamp"], "feature_version": body["feature_version"],
            "feature_names": names, "results": results, "download_url": None, "message": "ok",
        })
    return httpx.MockTransport(handler)


class TestStage2Contract:
    async def test_matrix_is_n_by_50_labels_preserved_names_carried(self):
        from src.adapters.feature_store_client import FeatureStoreClient
        from src.pipeline.stages.feature_extraction import FeatureExtractionStage

        pairs = [LabeledPair("A", "B", T1, 1, 1.0, "synthetic"),
                 LabeledPair("C", "D", T1, 0, 1.0, "synthetic")]
        vectors = {("A", "B"): [0.9] * 50, ("C", "D"): [0.1] * 50}
        client = FeatureStoreClient("http://fs", transport=_offline_transport(vectors, FEATURE_NAMES))
        ds = await FeatureExtractionStage(client).execute(_run(), pairs)

        assert ds.feature_matrix.shape == (2, 50)
        assert ds.labels.shape == (2,)
        assert ds.labels.tolist() == [1, 0]
        assert ds.feature_names == FEATURE_NAMES
        assert np.allclose(ds.feature_matrix[0], 0.9) and np.allclose(ds.feature_matrix[1], 0.1)

    async def test_missing_vector_is_not_fabricated(self, monkeypatch):
        """A pair the store cannot serve is absent from the matrix — never zero-filled."""
        from src.adapters.feature_store_client import FeatureStoreClient
        from src.pipeline.stages.feature_extraction import FeatureExtractionStage

        monkeypatch.setattr(settings, "FEATURE_COVERAGE_MIN_RATIO", 0.5)
        pairs = [LabeledPair("A", "B", T1, 1, 1.0, "synthetic"),
                 LabeledPair("C", "D", T1, 0, 1.0, "synthetic")]
        client = FeatureStoreClient("http://fs", transport=_offline_transport({("A", "B"): [0.9] * 50}, FEATURE_NAMES))
        ds = await FeatureExtractionStage(client).execute(_run(), pairs)
        assert ds.feature_matrix.shape == (1, 50)
        assert [(p.entity_id_1, p.entity_id_2) for p in ds.pairs] == [("A", "B")]

    async def test_declared_names_must_match_matrix_width(self):
        from src.adapters.feature_store_client import FeatureStoreClient
        from src.pipeline.stages.feature_extraction import FeatureExtractionStage

        pairs = [LabeledPair("A", "B", T1, 1, 1.0, "synthetic")]
        client = FeatureStoreClient("http://fs", transport=_offline_transport({("A", "B"): [0.9] * 50}, FEATURE_NAMES[:49]))
        with pytest.raises(ValueError, match="49 feature names"):
            await FeatureExtractionStage(client).execute(_run(), pairs)


# ─── 7-10. Real XGBoost: train, predict, serialize, evaluate ─────────────────

class TestXGBoostTrainer:
    def test_returns_a_real_fitted_model(self, xgb_only, isolated_mlflow):
        import xgboost as xgb
        from src.trainers.ensemble_trainer import XGBoostTrainer

        ds = _dataset()
        model, val_f1 = XGBoostTrainer().train(ds, mlflow_run_id=None)
        assert isinstance(model, xgb.XGBClassifier)
        assert model.n_features_in_ == 50
        assert model.get_booster().num_boosted_rounds() > 0
        assert 0.0 <= val_f1 <= 1.0

    def test_class_imbalance_handled_with_scale_pos_weight_not_resampling(self, xgb_only):
        from src.trainers.ensemble_trainer import XGBoostTrainer

        ds = _dataset()
        model, _ = XGBoostTrainer().train(ds)
        y_tr = ds.labels[ds.train_indices]
        expected = (y_tr == 0).sum() / (y_tr == 1).sum()
        assert model.get_params()["scale_pos_weight"] == pytest.approx(expected)
        assert len(ds.pairs) == ds.feature_matrix.shape[0]  # nothing resampled

    def test_predicts_a_probability_for_one_50_dim_vector(self, xgb_only):
        from src.trainers.ensemble_trainer import XGBoostTrainer

        model, _ = XGBoostTrainer().train(_dataset())
        proba = model.predict_proba(np.full((1, 50), 0.5, dtype=np.float32))
        assert proba.shape == (1, 2)
        assert 0.0 <= proba[0, 1] <= 1.0

    def test_runs_on_cpu_here(self, xgb_only):
        """This environment has no CUDA; training must not need it."""
        from src.trainers.ensemble_trainer import XGBoostTrainer

        model, _ = XGBoostTrainer().train(_dataset())
        assert model.get_params()["device"] == "cpu"
        assert model.get_params()["tree_method"] == "hist"

    def test_deterministic(self, xgb_only):
        from src.trainers.ensemble_trainer import XGBoostTrainer

        ds = _dataset()
        a, _ = XGBoostTrainer().train(ds)
        b, _ = XGBoostTrainer().train(ds)
        X = ds.feature_matrix[ds.test_indices]
        assert np.array_equal(a.predict_proba(X), b.predict_proba(X))

    def test_mlflow_flavor_round_trip_preserves_predictions(self, xgb_only, tmp_path):
        """Same flavor the registry uses (mlflow.xgboost)."""
        import mlflow.xgboost
        from src.trainers.ensemble_trainer import XGBoostTrainer

        ds = _dataset()
        model, _ = XGBoostTrainer().train(ds)
        path = tmp_path / "xgboost_model"
        mlflow.xgboost.save_model(model, str(path))
        reloaded = mlflow.xgboost.load_model(str(path))
        X = ds.feature_matrix[ds.test_indices]
        assert reloaded.n_features_in_ == 50
        assert np.allclose(model.predict_proba(X), reloaded.predict_proba(X))


class TestEvaluator:
    def test_evaluates_a_real_model_on_the_test_split(self, xgb_only, isolated_mlflow):
        from sklearn.metrics import f1_score
        from src.evaluators.model_evaluator import ModelEvaluator
        from src.trainers.ensemble_trainer import EnsembleTrainer

        ds = _dataset()
        trainer = EnsembleTrainer()
        with isolated_mlflow.start_run():
            models = trainer.train_all(ds)
            ev = ModelEvaluator(trainer).evaluate(models, ds, model_id="m", run_id=_run().run_id)

        y_true = ds.labels[ds.test_indices]
        y_pred = (models["xgb"]["model"].predict_proba(ds.feature_matrix[ds.test_indices])[:, 1] >= 0.5).astype(int)
        assert ev.n_test_samples == len(ds.test_indices)
        assert ev.n_positive == int(y_true.sum())
        assert ev.f1_score == pytest.approx(f1_score(y_true, y_pred))
        assert sum(map(sum, ev.confusion_matrix)) == len(ds.test_indices)
        for m in (ev.precision, ev.recall, ev.auc_roc, ev.average_precision):
            assert 0.0 <= m <= 1.0

    def test_shap_importance_uses_real_feature_names(self, xgb_only, isolated_mlflow):
        """Previously keyed feat_0..feat_49, discarding the catalog names."""
        from src.evaluators.model_evaluator import ModelEvaluator
        from src.trainers.ensemble_trainer import EnsembleTrainer

        ds = _dataset()
        trainer = EnsembleTrainer()
        with isolated_mlflow.start_run():
            models = trainer.train_all(ds)
            ev = ModelEvaluator(trainer).evaluate(models, ds, model_id="m", run_id=_run().run_id)
        assert set(ev.feature_importance) == set(FEATURE_NAMES)
        top5 = sorted(ev.feature_importance, key=ev.feature_importance.get, reverse=True)[:5]
        assert all(name.endswith("_signal") for name in top5), top5


# ─── 11. Registration never fabricates a version ─────────────────────────────

class TestRegistration:
    def test_registry_failure_raises_instead_of_inventing_version_1(
        self, xgb_only, isolated_mlflow, monkeypatch,
    ):
        import mlflow
        from src.evaluators.model_evaluator import ModelEvaluator
        from src.registry.mlflow_registry import MLflowModelRegistry
        from src.trainers.ensemble_trainer import EnsembleTrainer

        ds = _dataset()
        trainer = EnsembleTrainer()
        registry = MLflowModelRegistry()
        run = _run()
        monkeypatch.setattr(settings, "MIN_F1_THRESHOLD", 0.0)  # isolate the registry path
        registry.start_run(run)
        models = trainer.train_all(ds)
        ev = ModelEvaluator(trainer).evaluate(models, ds, model_id="m", run_id=run.run_id)

        def boom(*a, **k):
            raise ConnectionError("registry unreachable")
        monkeypatch.setattr(mlflow, "register_model", boom)

        with pytest.raises(InfrastructureUnavailableError, match="registry unreachable"):
            registry.register_model(models=models, evaluation=ev, run=run, dataset=ds)

    def test_successful_registration_logs_model_and_lineage(self, xgb_only, isolated_mlflow, monkeypatch):
        from mlflow import MlflowClient
        from src.evaluators.model_evaluator import ModelEvaluator
        from src.registry.mlflow_registry import MLflowModelRegistry
        from src.trainers.ensemble_trainer import EnsembleTrainer

        monkeypatch.setattr(settings, "MIN_F1_THRESHOLD", 0.0)
        ds = _dataset()
        trainer = EnsembleTrainer()
        registry = MLflowModelRegistry()
        run = _run()
        run.dataset_version = "ds-test-123"
        run_id = registry.start_run(run)
        registry.log_split_stats({"test_pairs": 10})
        models = trainer.train_all(ds, mlflow_run_id=run_id)
        ev = ModelEvaluator(trainer).evaluate(models, ds, model_id="m", run_id=run.run_id)
        version = registry.register_model(models=models, evaluation=ev, run=run, dataset=ds)

        client = MlflowClient()
        r = client.get_run(run_id)
        assert version is not None
        assert r.data.tags["dataset_version"] == "ds-test-123"
        assert r.data.tags["feature_catalog_version"] == run.feature_store_version
        assert r.data.tags["enabled_trainers"] == "xgboost"
        assert r.data.metrics["split_test_pairs"] == 10
        assert r.data.params["is_partial_ensemble"] == "True"
        artifacts = {a.path for a in client.list_artifacts(run_id)}
        assert "xgboost_model" in artifacts
        assert "transformer_model" not in artifacts and "gnn_model" not in artifacts
        # The serving artifact is logged with the version it belongs to.
        assert {a.path for a in client.list_artifacts(run_id, "onnx_model")} == {
            "onnx_model/model.onnx", "onnx_model/serving_signature.json"}
        import mlflow
        manifest = json.loads(mlflow.artifacts.load_text(f"runs:/{run_id}/ensemble_manifest.json"))
        assert manifest["registered_model_version"] == version
        assert manifest["ensemble_version"] == f"v{version}"   # was the run id before
        assert manifest["onnx_sha256"] and manifest["feature_names_sha256"]


# ─── Stage 2: tenant-correct batching (regression) ───────────────────────────

T2 = "00000000-0000-0000-0000-000000000002"


def _tenant_scoped_transport(store: dict, seen: list):
    """Like the real Feature Store: answers ONLY for the tenant in the request header.
    store: {(tenant, e1, e2): vector}."""
    def handler(request):
        tenant = request.headers["x-verified-tenant-id"]
        body = json.loads(request.content)
        seen.append((tenant, [tuple(p) for p in body["entity_pairs"]]))
        results = [
            {"entity_id_1": a, "entity_id_2": b, "feature_vector": store[(tenant, a, b)],
             "computed_at": "2026-09-30T00:00:00"}
            for a, b in body["entity_pairs"] if (tenant, a, b) in store
        ]
        return httpx.Response(200, json={
            "request_id": "00000000-0000-0000-0000-0000000000bb",
            "pairs_requested": len(body["entity_pairs"]), "pairs_found": len(results),
            "as_of_timestamp": body["as_of_timestamp"], "feature_version": body["feature_version"],
            "feature_names": FEATURE_NAMES, "results": results, "download_url": None, "message": "ok",
        })
    return httpx.MockTransport(handler)


class TestTenantBatching:
    async def test_interleaved_tenants_get_full_coverage(self, monkeypatch):
        """Regression: batches were requested under batch[0]'s tenant, so interleaved
        tenants lost ~half their pairs (48.8% on the dev set)."""
        from src.adapters.feature_store_client import FeatureStoreClient
        from src.pipeline.stages.feature_extraction import FeatureExtractionStage

        monkeypatch.setattr(settings, "FEATURE_COVERAGE_MIN_RATIO", 1.0)
        pairs, store = [], {}
        for i in range(250):  # strictly alternating tenants
            t = T1 if i % 2 == 0 else T2
            pairs.append(LabeledPair(f"E{i}a", f"E{i}b", t, i % 2, 1.0, "synthetic"))
            store[(t, f"E{i}a", f"E{i}b")] = [i / 1000] * 50
        seen = []
        client = FeatureStoreClient("http://fs", transport=_tenant_scoped_transport(store, seen))
        ds = await FeatureExtractionStage(client).execute(_run(), pairs)

        assert ds.feature_matrix.shape == (250, 50)
        # every request carried pairs of exactly the tenant in its header
        for tenant, requested in seen:
            expected = {(p.entity_id_1, p.entity_id_2) for p in pairs if p.tenant_id == tenant}
            assert set(requested) <= expected

    async def test_same_entity_ids_in_two_tenants_do_not_collide(self):
        """Entity ids are tenant-scoped; results must be keyed by tenant too."""
        from src.adapters.feature_store_client import FeatureStoreClient
        from src.pipeline.stages.feature_extraction import FeatureExtractionStage

        pairs = [LabeledPair("A", "B", T1, 1, 1.0, "synthetic"),
                 LabeledPair("A", "B", T2, 0, 1.0, "synthetic")]
        store = {(T1, "A", "B"): [0.9] * 50, (T2, "A", "B"): [0.1] * 50}
        client = FeatureStoreClient("http://fs", transport=_tenant_scoped_transport(store, []))
        ds = await FeatureExtractionStage(client).execute(_run(), pairs)

        by_tenant = {p.tenant_id: ds.feature_matrix[i] for i, p in enumerate(ds.pairs)}
        assert np.allclose(by_tenant[T1], 0.9)
        assert np.allclose(by_tenant[T2], 0.1)
