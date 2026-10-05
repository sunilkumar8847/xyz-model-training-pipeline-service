"""
Tests for the ONNX serving artifact and the registry -> Triton publisher.

Dependency type: REAL xgboost, onnxmltools, onnxruntime and MLflow (MLflow isolated
to a temporary SQLite registry + artifact dir). Triton itself is not needed: these
tests verify the repository the publisher writes, not a running server.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.core.config import settings
from src.registry.onnx_export import (
    N_FEATURES, OnnxExportError, export_xgboost_to_onnx, feature_names_sha256,
)
from src.registry.triton_publisher import (
    TritonPublishError, publish_registered_model, triton_model_name,
)
from tests.unit.test_xgboost_training_path import FEATURE_NAMES, _dataset, _run


# These tests exercise publishing mechanics against a throw-away sqlite registry, which
# the publisher refuses by default (only the MLflow server registry is authoritative).
SCRATCH = {"allow_scratch_registry": True}

@pytest.fixture
def isolated_mlflow(tmp_path, monkeypatch):
    import mlflow
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    for key, value in (("MLFLOW_TRACKING_URI", uri), ("MLFLOW_REGISTRY_URI", uri),
                       ("MLFLOW_ARTIFACT_LOCATION", (tmp_path / "artifacts").as_uri()),
                       ("MLFLOW_EXPERIMENT_NAME", "triton-publish-unit"),
                       ("ENABLED_TRAINERS", "xgboost"), ("XGB_N_ESTIMATORS", 150),
                       ("MIN_F1_THRESHOLD", 0.0)):
        monkeypatch.setattr(settings, key, value)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    # mlflow caches the active experiment id process-wide; select one that exists in
    # THIS fresh store so ids from an earlier test's store never leak in.
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
    yield mlflow
    while mlflow.active_run():
        mlflow.end_run()


@pytest.fixture
def trained():
    from src.trainers.ensemble_trainer import EnsembleTrainer

    ds = _dataset()
    import mlflow
    with mlflow.start_run(nested=True) if mlflow.active_run() else _null():
        models = EnsembleTrainer().train_all(ds)
    return ds, models


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _register(ds, models):
    """Run the REAL registration path; returns (registry_version, mlflow_run_id)."""
    from src.evaluators.model_evaluator import ModelEvaluator
    from src.registry.mlflow_registry import MLflowModelRegistry
    from src.trainers.ensemble_trainer import EnsembleTrainer

    registry = MLflowModelRegistry()
    run = _run()
    run.dataset_version = "ds-unit-1"
    run_id = registry.start_run(run)
    ev = ModelEvaluator(EnsembleTrainer()).evaluate(models, ds, model_id="m", run_id=run.run_id)
    version = registry.register_model(models=models, evaluation=ev, run=run, dataset=ds)
    return version, run_id


# ─── ONNX export ──────────────────────────────────────────────────────────────

class TestOnnxExport:
    def test_export_matches_xgboost_and_honours_contract(self, isolated_mlflow, trained):
        import onnx
        import onnxruntime as ort

        ds, models = trained
        X = ds.feature_matrix[ds.test_indices]
        exp = export_xgboost_to_onnx(models["xgb"]["model"], FEATURE_NAMES, X)

        m = onnx.load_from_string(exp.onnx_bytes)
        assert m.ir_version <= 9
        sess = ort.InferenceSession(exp.onnx_bytes, providers=["CPUExecutionProvider"])
        assert [(i.name, i.shape[1]) for i in sess.get_inputs()] == [("input", 50)]
        got = sess.run(["probabilities"], {"input": X})[0][:, 1]
        want = models["xgb"]["model"].predict_proba(X)[:, 1]
        assert np.abs(got - want).max() <= 1e-5
        assert exp.signature["parity"]["max_abs_diff"] <= 1e-5
        assert exp.signature["feature_names"] == FEATURE_NAMES
        assert exp.signature["feature_names_sha256"] == feature_names_sha256(FEATURE_NAMES)
        assert len(exp.signature["probes"]) > 0

    def test_deterministic_bytes_and_outputs(self, isolated_mlflow, trained):
        ds, models = trained
        X = ds.feature_matrix[ds.test_indices]
        a = export_xgboost_to_onnx(models["xgb"]["model"], FEATURE_NAMES, X)
        b = export_xgboost_to_onnx(models["xgb"]["model"], FEATURE_NAMES, X)
        assert a.sha256 == b.sha256

    def test_wrong_feature_name_count_rejected(self, isolated_mlflow, trained):
        ds, models = trained
        with pytest.raises(OnnxExportError, match="exactly 50"):
            export_xgboost_to_onnx(models["xgb"]["model"], FEATURE_NAMES[:49], ds.feature_matrix)

    def test_duplicate_feature_names_rejected(self, isolated_mlflow, trained):
        ds, models = trained
        with pytest.raises(OnnxExportError, match="duplicates"):
            export_xgboost_to_onnx(models["xgb"]["model"], [FEATURE_NAMES[0]] * 50, ds.feature_matrix)

    def test_wrong_width_verification_rows_rejected(self, isolated_mlflow, trained):
        ds, models = trained
        with pytest.raises(OnnxExportError, match="verification_rows"):
            export_xgboost_to_onnx(models["xgb"]["model"], FEATURE_NAMES, ds.feature_matrix[:, :49])

    def test_registration_without_feature_names_fails(self, isolated_mlflow, trained):
        """A version must never be registered without its serving artifact."""
        ds, models = trained
        ds.feature_names = None
        with pytest.raises(OnnxExportError, match="feature names"):
            _register(ds, models)


# ─── Publisher ────────────────────────────────────────────────────────────────

class TestPublisher:
    def test_publishes_the_registered_artifact(self, isolated_mlflow, trained, tmp_path):
        ds, models = trained
        version, run_id = _register(ds, models)
        repo = tmp_path / "repo"
        repo.mkdir()

        r = publish_registered_model(str(repo), "xyz-mdm-matcher", version=version, **SCRATCH)

        name = triton_model_name(version)
        assert r.triton_model_name == name == f"customer_matcher_v{version}"
        assert r.triton_version == "1" and r.serving_model_version == f"v{version}"
        assert r.registry_version == version and r.mlflow_run_id == run_id
        onnx_file = repo / name / "1" / "model.onnx"
        assert onnx_file.is_file()

        # The published bytes are exactly the registered artifact.
        import mlflow
        logged = Path(mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="onnx_model/model.onnx", dst_path=str(tmp_path / "dl")))
        assert onnx_file.read_bytes() == logged.read_bytes()

        config = (repo / name / "config.pbtxt").read_text()
        for expected in (f'name: "{name}"', 'platform: "onnxruntime_onnx"', 'name: "input"',
                         "dims: [ 50 ]", 'name: "probabilities"', "dims: [ 2 ]",
                         f'key: "registry_version" value: {{ string_value: "{version}" }}',
                         f'key: "mlflow_run_id" value: {{ string_value: "{run_id}" }}',
                         f'key: "feature_names_sha256" value: {{ string_value: "{feature_names_sha256(FEATURE_NAMES)}" }}',
                         "versions: [ 1 ]"):
            assert expected in config, expected

        manifest = json.loads((repo / name / "serving_manifest.json").read_text())
        assert manifest["feature_names"] == FEATURE_NAMES
        assert manifest["dataset_version"] == "ds-unit-1"
        # nothing left behind in the repository except the published model
        assert sorted(p.name for p in repo.iterdir()) == [name]

    def test_republishing_identical_artifact_is_a_noop(self, isolated_mlflow, trained, tmp_path):
        ds, models = trained
        version, _ = _register(ds, models)
        repo = tmp_path / "repo"
        repo.mkdir()
        publish_registered_model(str(repo), "xyz-mdm-matcher", version=version, **SCRATCH)
        again = publish_registered_model(str(repo), "xyz-mdm-matcher", version=version, **SCRATCH)
        assert again.already_published is True

    def test_refuses_to_overwrite_a_different_artifact(self, isolated_mlflow, trained, tmp_path):
        ds, models = trained
        version, _ = _register(ds, models)
        repo = tmp_path / "repo"
        (repo / triton_model_name(version) / "1").mkdir(parents=True)
        (repo / triton_model_name(version) / "1" / "model.onnx").write_bytes(b"something else")
        with pytest.raises(TritonPublishError, match="DIFFERENT artifact"):
            publish_registered_model(str(repo), "xyz-mdm-matcher", version=version, **SCRATCH)

    def test_corrupt_artifact_refused(self, isolated_mlflow, trained, tmp_path, monkeypatch):
        """Bytes that differ from what training verified must never be served."""
        import mlflow

        ds, models = trained
        version, run_id = _register(ds, models)
        real_download = mlflow.artifacts.download_artifacts

        def corrupting(*a, **k):
            path = Path(real_download(*a, **k))
            if path.name == "model.onnx":
                path.write_bytes(path.read_bytes()[:-16] + b"\x00" * 16)
            return str(path)

        monkeypatch.setattr(mlflow.artifacts, "download_artifacts", corrupting)
        repo = tmp_path / "repo"
        repo.mkdir()
        with pytest.raises(TritonPublishError, match="sha256"):
            publish_registered_model(str(repo), "xyz-mdm-matcher", version=version, **SCRATCH)
        assert list(repo.iterdir()) == []

    def test_unknown_version_refused(self, isolated_mlflow, trained, tmp_path):
        ds, models = trained
        _register(ds, models)
        with pytest.raises(TritonPublishError, match="not found"):
            publish_registered_model(str(tmp_path), "xyz-mdm-matcher", version="999", **SCRATCH)

    def test_invalid_version_string_refused(self, isolated_mlflow, tmp_path):
        with pytest.raises(TritonPublishError, match="invalid registry version"):
            publish_registered_model(str(tmp_path), "xyz-mdm-matcher", version="latest", **SCRATCH)

    def test_stage_resolves_to_highest_version_in_stage(self, isolated_mlflow, trained, tmp_path):
        from mlflow import MlflowClient

        ds, models = trained
        v1, _ = _register(ds, models)
        v2, _ = _register(ds, models)
        client = MlflowClient()
        client.transition_model_version_stage("xyz-mdm-matcher", v1, "Production")
        repo = tmp_path / "repo"
        repo.mkdir()
        r = publish_registered_model(str(repo), "xyz-mdm-matcher", stage="Production", **SCRATCH)
        assert r.registry_version == v1 != v2

    def test_full_ensemble_weights_refused(self, isolated_mlflow, trained, tmp_path, monkeypatch):
        """The ONNX file is the XGBoost component only; never serve it as an ensemble."""
        import mlflow

        ds, models = trained
        version, _ = _register(ds, models)
        real_load = mlflow.artifacts.load_text

        def ensemble_manifest(uri):
            text = real_load(uri)
            if uri.endswith("ensemble_manifest.json"):
                m = json.loads(text)
                m["weights"] = {"transformer": 0.5, "gnn": 0.3, "xgboost": 0.2}
                return json.dumps(m)
            return text

        monkeypatch.setattr(mlflow.artifacts, "load_text", ensemble_manifest)
        repo = tmp_path / "repo"
        repo.mkdir()
        with pytest.raises(TritonPublishError, match="XGBoost component"):
            publish_registered_model(str(repo), "xyz-mdm-matcher", version=version, **SCRATCH)

    def test_missing_repository_refused(self, isolated_mlflow, tmp_path):
        with pytest.raises(TritonPublishError, match="does not exist"):
            publish_registered_model(str(tmp_path / "nope"), "xyz-mdm-matcher", version="1", **SCRATCH)
