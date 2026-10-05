"""
Tests for the "durable, traceable bootstrap lifecycle" phase (training side):

  dataset identity + bootstrap mode, tenant-safe deduplication, dataset/label lineage
  in the registry, the single authoritative registry, registry -> Triton traceability,
  the frozen benchmark and decision-tier evaluation, and the lineage trace.

Dependency type: REAL synthetic generator output (written to tmp_path), REAL xgboost /
onnx / MLflow (isolated sqlite registry). No network services.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from src.core.config import Environment, Settings, check_training_data_sources, settings
from src.domain.models import LabeledPair
from src.evaluators.benchmark import DECISION_TIERS, benchmark_id, tier_of, tier_report
from src.pipeline.stages.data_collection import DataCollectionStage
from src.registry.triton_publisher import TritonPublishError, publish_registered_model
from tests.unit.test_triton_publishing import SCRATCH, _register, isolated_mlflow, trained  # noqa: F401
from tests.unit.test_xgboost_training_path import _dataset, _run

REPO_ROOT = Path(__file__).resolve().parents[3]
T1 = "00000000-0000-0000-0000-000000000001"
T2 = "00000000-0000-0000-0000-000000000002"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """A real seed corpus written by the repository-root generator."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    sd = pytest.importorskip("synthetic_data")
    out = tmp_path_factory.mktemp("seed_corpus")
    config = dataclasses.replace(sd.get_profile("smoke"), n_identities=20)
    identities, records, pairs = sd.SyntheticDatasetGenerator(config).generate()
    manifest = sd.write_dataset(out, config, identities, records, pairs)
    return out, manifest, hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest()


def pair(e1, e2, tenant=T1, label=1, source="synthetic"):
    return LabeledPair(e1, e2, tenant, label, 1.0, source)


# ─── 1 / 2. Dataset identity ──────────────────────────────────────────────────

class TestDatasetIdentity:
    def test_provenance_is_read_from_the_verified_manifest(self, corpus):
        from src.adapters.synthetic_data import dataset_provenance
        out, manifest, digest = corpus
        prov = dataset_provenance(str(out))
        assert prov["dataset_id"] == manifest["dataset_id"]
        assert prov["manifest_sha256"] == digest
        assert prov["content_sha256"] == manifest["identity"]["content_sha256"]
        assert prov["generator_version"] == manifest["generator_version"]
        assert prov["seed"] == manifest["seed"] and prov["label_source"] == "synthetic"
        assert prov["tenants"] == manifest["identity"]["tenants"]

    def test_modified_dataset_is_refused(self, corpus, tmp_path):
        from src.adapters.synthetic_data import SyntheticDataError, load_synthetic_labeled_pairs
        d = tmp_path / "copy"
        shutil.copytree(corpus[0], d)
        with open(d / "pairs.parquet", "ab") as f:
            f.write(b"tampered")
        with pytest.raises(SyntheticDataError, match="does not match the manifest"):
            load_synthetic_labeled_pairs(str(d))

    def test_dataset_without_identity_is_refused(self, corpus, tmp_path):
        from src.adapters.synthetic_data import SyntheticDataError, load_synthetic_labeled_pairs
        d = tmp_path / "legacy"
        shutil.copytree(corpus[0], d)
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        for k in ("identity", "dataset_id", "file_sha256"):
            m.pop(k)
        (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        with pytest.raises(SyntheticDataError, match="no dataset identity"):
            load_synthetic_labeled_pairs(str(d))

    def test_train_and_benchmark_splits_do_not_overlap(self, corpus):
        from src.adapters.synthetic_data import load_benchmark_pairs, load_synthetic_labeled_pairs
        train = load_synthetic_labeled_pairs(str(corpus[0]))
        bench = [p for p, _ in load_benchmark_pairs(str(corpus[0]))]
        assert train and bench
        key = lambda p: (p.tenant_id, *sorted((p.entity_id_1, p.entity_id_2)))  # noqa: E731
        assert not {key(p) for p in train} & {key(p) for p in bench}
        ents = lambda ps: {(p.tenant_id, e) for p in ps for e in (p.entity_id_1, p.entity_id_2)}  # noqa: E731
        assert not ents(train) & ents(bench)          # no shared entity record at all


class TestBootstrapMode:
    def _cfg(self, env, corpus, **kw):
        return Settings(ENVIRONMENT=env, SYNTHETIC_DATA_DIR=str(corpus[0]),
                        MLFLOW_TRACKING_URI="http://mlflow:5000", **kw)

    @pytest.mark.parametrize("env", [Environment.STAGING, Environment.PRODUCTION])
    def test_synthetic_refused_without_bootstrap_mode(self, corpus, env):
        with pytest.raises(ValueError, match="BOOTSTRAP_MODE"):
            self._cfg(env, corpus)

    def test_bootstrap_mode_requires_a_pin(self, corpus):
        with pytest.raises(ValueError, match="BOOTSTRAP_DATASET_MANIFEST_SHA256"):
            self._cfg(Environment.PRODUCTION, corpus, BOOTSTRAP_MODE=True)

    def test_pinned_corpus_is_accepted_and_reported_as_bootstrap(self, corpus, monkeypatch):
        from src.adapters import synthetic_data as sdmod
        s = self._cfg(Environment.PRODUCTION, corpus, BOOTSTRAP_MODE=True,
                      BOOTSTRAP_DATASET_MANIFEST_SHA256=corpus[2])
        assert s.data_mode == "bootstrap"
        monkeypatch.setattr(sdmod, "settings", s)
        assert sdmod.dataset_provenance(str(corpus[0]))["dataset_id"] == corpus[1]["dataset_id"]

    def test_unpinned_directory_is_refused_in_bootstrap_mode(self, corpus, monkeypatch):
        from src.adapters import synthetic_data as sdmod
        s = self._cfg(Environment.PRODUCTION, corpus, BOOTSTRAP_MODE=True,
                      BOOTSTRAP_DATASET_MANIFEST_SHA256="0" * 64)
        monkeypatch.setattr(sdmod, "settings", s)
        with pytest.raises(sdmod.SyntheticDataError, match="not an approved bootstrap corpus"):
            sdmod.load_synthetic_labeled_pairs(str(corpus[0]))

    def test_minimum_pairs_is_still_enforced_in_bootstrap_mode(self, corpus, monkeypatch):
        """Bootstrap mode opens ONE gate; the 50,000-pair minimum stays."""
        from src.core.exceptions import InsufficientTrainingDataError
        from src.pipeline.stages import data_collection as dc
        s = self._cfg(Environment.PRODUCTION, corpus, BOOTSTRAP_MODE=True,
                      BOOTSTRAP_DATASET_MANIFEST_SHA256=corpus[2])
        monkeypatch.setattr(dc, "settings", s)
        with pytest.raises(InsufficientTrainingDataError):
            dc.enforce_minimum_training_pairs(6606)

    def test_data_mode_labels(self, corpus):
        assert Settings(ENVIRONMENT=Environment.DEVELOPMENT, SYNTHETIC_DATA_DIR=None).data_mode == "customer"
        assert self._cfg(Environment.DEVELOPMENT, corpus).data_mode == "development"

    def test_staging_requires_the_server_registry(self):
        with pytest.raises(ValueError, match="MLflow server registry"):
            Settings(ENVIRONMENT=Environment.STAGING, SYNTHETIC_DATA_DIR=None,
                     MLFLOW_TRACKING_URI="sqlite:///local.db")

    def test_registry_scope(self):
        dev = Environment.DEVELOPMENT
        assert Settings(ENVIRONMENT=dev, MLFLOW_TRACKING_URI="http://localhost:5000").mlflow_registry_scope == "server"
        assert Settings(ENVIRONMENT=dev, MLFLOW_TRACKING_URI="sqlite:///x.db").mlflow_registry_scope == "local-scratch"
        assert Settings(ENVIRONMENT=dev, MLFLOW_TRACKING_URI="file:///tmp/m").mlflow_registry_scope == "local-scratch"


# ─── 3. Tenant-safe deduplication / dataset version ───────────────────────────

class TestTenantSafeDedup:
    def test_same_ids_in_two_tenants_are_both_kept(self):
        """The key used to be the two entity ids only: the second tenant's pair vanished."""
        stage = DataCollectionStage(db_session=None)
        kept = stage._deduplicate([pair("E1", "E2", T1), pair("E1", "E2", T2), pair("E2", "E1", T2)])
        assert [(p.tenant_id, p.entity_id_1, p.entity_id_2) for p in kept] == [(T1, "E1", "E2"), (T2, "E1", "E2")]

    def test_duplicates_within_a_tenant_are_still_removed(self):
        stage = DataCollectionStage(db_session=None)
        kept = stage._deduplicate([pair("A", "B"), pair("B", "A"), pair("A", "B"), pair("A", "C")])
        assert len(kept) == 2

    def test_dataset_version_distinguishes_tenants(self):
        v = DataCollectionStage._compute_dataset_version
        assert v([pair("E1", "E2", T1)]) != v([pair("E1", "E2", T2)])
        assert v([pair("E1", "E2", T1), pair("E1", "E2", T2)]).startswith("ds-2-")

    def test_dataset_version_changes_with_label_source_and_dataset_content(self):
        v = DataCollectionStage._compute_dataset_version
        pairs = [pair("E1", "E2"), pair("E3", "E4", label=0)]
        assert v(pairs, ["synds-aaa"]) != v(pairs, ["synds-bbb"])      # same ids, other content
        assert v(pairs) != v([pair("E1", "E2", source="hitl_merge"), pair("E3", "E4", label=0)])
        assert v(pairs) != v([pair("E1", "E2", label=0), pair("E3", "E4", label=0)])

    def test_dataset_version_is_order_independent(self):
        v = DataCollectionStage._compute_dataset_version
        a, b = pair("E1", "E2"), pair("E4", "E3", T2, 0)
        assert v([a, b], ["x", "y"]) == v([b, a], ["y", "x"])
        assert v([pair("E2", "E1")]) == v([pair("E1", "E2")])


# ─── Lineage recorded at registration ─────────────────────────────────────────

def _register_with_dataset(ds, models, provenance):
    from src.evaluators.model_evaluator import ModelEvaluator
    from src.registry.mlflow_registry import MLflowModelRegistry
    from src.trainers.ensemble_trainer import EnsembleTrainer

    registry = MLflowModelRegistry()
    run = _run()
    run.label_sources = {"synthetic": len(ds.pairs)}
    run.dataset_provenance = [provenance] if provenance else []
    run.dataset_version = DataCollectionStage._compute_dataset_version(
        ds.pairs, [provenance["dataset_id"]] if provenance else [])
    run.feature_as_of = datetime(2026, 10, 5, 12, 0, 0)
    run_id = registry.start_run(run)
    registry.log_lineage(run)
    ev = ModelEvaluator(EnsembleTrainer()).evaluate(models, ds, model_id="m", run_id=run.run_id)
    version = registry.register_model(models=models, evaluation=ev, run=run, dataset=ds)
    return version, run_id, run


class TestLineageAtRegistration:
    def test_dataset_identity_is_in_run_tags_manifest_and_version_tags(self, isolated_mlflow, trained, corpus):
        from mlflow import MlflowClient
        from src.adapters.synthetic_data import dataset_provenance
        ds, models = trained
        prov = dataset_provenance(str(corpus[0]))
        version, run_id, run = _register_with_dataset(ds, models, prov)

        client = MlflowClient()
        tags = client.get_run(run_id).data.tags
        assert tags["dataset_id"] == prov["dataset_id"]
        assert tags["dataset_manifest_sha256"] == prov["manifest_sha256"]
        assert tags["dataset_version"] == run.dataset_version
        assert tags["label_sources"] == f"synthetic={len(ds.pairs)}"
        assert tags["data_mode"] in ("development", "bootstrap", "customer")
        assert tags["feature_as_of"] == "2026-10-05T12:00:00"
        assert tags["git_dirty"] in ("true", "false", "none")
        assert tags["registry_scope"] == "local-scratch"

        manifest = json.loads(isolated_mlflow.artifacts.load_text(f"runs:/{run_id}/ensemble_manifest.json"))
        assert manifest["dataset"]["datasets"][0]["dataset_id"] == prov["dataset_id"]
        assert manifest["dataset"]["label_sources"] == {"synthetic": len(ds.pairs)}
        assert manifest["feature_as_of"] == "2026-10-05T12:00:00"
        assert manifest["registered_model_version"] == version

        mv = client.get_model_version("xyz-mdm-matcher", version)
        assert mv.tags["dataset_id"] == prov["dataset_id"]
        assert mv.tags["onnx_sha256"] == manifest["onnx_sha256"]
        prov_file = json.loads(isolated_mlflow.artifacts.load_text(f"runs:/{run_id}/dataset_provenance.json"))
        assert prov_file["datasets"][0]["content_sha256"] == prov["content_sha256"]

    def test_only_one_registered_model_name_is_created(self, isolated_mlflow, trained):
        """Every run used to be registered twice (…-xgb), with independent numbering."""
        from mlflow import MlflowClient
        ds, models = trained
        _register(ds, models)
        assert [m.name for m in MlflowClient().search_registered_models()] == ["xyz-mdm-matcher"]

    def test_dirty_tree_is_refused_outside_development(self, isolated_mlflow, trained, monkeypatch):
        from src.core.exceptions import InfrastructureUnavailableError
        from src.registry import mlflow_registry as reg
        ds, models = trained
        monkeypatch.setenv("GIT_DIRTY", "true")
        monkeypatch.setattr(reg.settings, "ENVIRONMENT", Environment.PRODUCTION)
        with pytest.raises(InfrastructureUnavailableError, match="uncommitted"):
            _register(ds, models)

    def test_git_dirty_is_reported_from_the_environment(self, monkeypatch):
        from src.registry.mlflow_registry import get_git_dirty
        monkeypatch.setenv("GIT_DIRTY", "false")
        assert get_git_dirty() is False
        monkeypatch.setenv("GIT_DIRTY", "true")
        assert get_git_dirty() is True


# ─── 13 / 14. One registry, traceable publish ─────────────────────────────────

class TestRegistryToTriton:
    def test_scratch_registry_cannot_be_published(self, isolated_mlflow, trained, tmp_path):
        ds, models = trained
        version, _ = _register(ds, models)
        repo = tmp_path / "repo"
        repo.mkdir()
        with pytest.raises(TritonPublishError, match="local scratch store"):
            publish_registered_model(str(repo), "xyz-mdm-matcher", version=version)
        assert list(repo.iterdir()) == []

    def test_publish_writes_dataset_lineage_and_tags_the_registry(self, isolated_mlflow, trained, corpus, tmp_path):
        from mlflow import MlflowClient
        from src.adapters.synthetic_data import dataset_provenance
        from src.registry.lineage import parse_config_parameters
        ds, models = trained
        prov = dataset_provenance(str(corpus[0]))
        version, run_id, run = _register_with_dataset(ds, models, prov)
        repo = tmp_path / "repo"
        repo.mkdir()
        r = publish_registered_model(str(repo), "xyz-mdm-matcher", version=version, **SCRATCH)

        params = parse_config_parameters((Path(r.model_dir) / "config.pbtxt").read_text(encoding="utf-8"))
        assert params["dataset_id"] == prov["dataset_id"] == r.dataset_id
        assert params["dataset_manifest_sha256"] == prov["manifest_sha256"]
        assert params["dataset_version"] == run.dataset_version
        assert params["label_sources"] == f"synthetic={len(ds.pairs)}"
        assert params["mlflow_run_id"] == run_id and params["registry_version"] == version
        assert params["registry_scope"] == "local-scratch"

        mv = MlflowClient().get_model_version("xyz-mdm-matcher", version)   # registry -> Triton
        assert mv.tags["triton_model"] == f"customer_matcher_v{version}"
        assert mv.tags["triton_onnx_sha256"] == r.onnx_sha256 == params["onnx_sha256"]

    def test_synthetic_labels_without_a_dataset_id_cannot_be_published(self, isolated_mlflow, trained, tmp_path):
        ds, models = trained
        version, _, _ = _register_with_dataset(ds, models, provenance=None)
        with pytest.raises(TritonPublishError, match="records no dataset_id"):
            publish_registered_model(str(tmp_path), "xyz-mdm-matcher", version=version, **SCRATCH)

    def test_require_stage_refuses_a_version_that_is_not_in_production(self, isolated_mlflow, trained, tmp_path):
        from mlflow import MlflowClient
        ds, models = trained
        version, _ = _register(ds, models)
        with pytest.raises(TritonPublishError, match="not 'Production'"):
            publish_registered_model(str(tmp_path), "xyz-mdm-matcher", version=version,
                                     require_stage="Production", **SCRATCH)
        MlflowClient().transition_model_version_stage("xyz-mdm-matcher", version, "Production")
        r = publish_registered_model(str(tmp_path), "xyz-mdm-matcher", version=version,
                                     require_stage="Production", **SCRATCH)
        assert r.registry_stage == "Production"


class TestLineageTrace:
    def _published(self, isolated_mlflow, trained, corpus, tmp_path):
        from src.adapters.synthetic_data import dataset_provenance
        ds, models = trained
        prov = dataset_provenance(str(corpus[0]))
        version, run_id, _ = _register_with_dataset(ds, models, prov)
        repo = tmp_path / "repo"
        repo.mkdir()
        publish_registered_model(str(repo), "xyz-mdm-matcher", version=version, **SCRATCH)
        return version, run_id, repo, prov

    def test_intact_chain_verifies_hop_by_hop(self, isolated_mlflow, trained, corpus, tmp_path):
        from src.registry.lineage import OK, SKIPPED, trace_lineage
        version, run_id, repo, prov = self._published(isolated_mlflow, trained, corpus, tmp_path)
        trace = trace_lineage(version, "xyz-mdm-matcher", data_dir=str(corpus[0]),
                              triton_repository=str(repo))
        by = {h.name: h for h in trace.hops}
        assert [h.name for h in trace.hops] == [
            "dataset", "training_run", "model_version", "onnx_artifact", "triton_model", "inference_service"]
        for name in ("dataset", "training_run", "model_version", "onnx_artifact", "triton_model"):
            assert by[name].status == OK, (name, by[name].problems)
        assert by["inference_service"].status == SKIPPED       # not asked for here
        assert not trace.broken
        assert by["dataset"].facts["dataset_id"] == prov["dataset_id"] == by["training_run"].facts["dataset_id"]
        assert by["triton_model"].facts["config.mlflow_run_id"] == run_id
        assert (by["onnx_artifact"].facts["onnx_sha256 (artifact re-hashed)"]
                == by["triton_model"].facts["onnx_sha256 (file re-hashed)"])

    def test_swapped_triton_file_breaks_the_chain(self, isolated_mlflow, trained, corpus, tmp_path):
        from src.registry.lineage import BROKEN, trace_lineage
        version, _, repo, _ = self._published(isolated_mlflow, trained, corpus, tmp_path)
        model = repo / f"customer_matcher_v{version}" / "1" / "model.onnx"
        model.write_bytes(model.read_bytes() + b"\x00")
        trace = trace_lineage(version, "xyz-mdm-matcher", data_dir=str(corpus[0]), triton_repository=str(repo))
        hop = {h.name: h for h in trace.hops}["triton_model"]
        assert hop.status == BROKEN and trace.broken
        assert any("NOT the registry's artifact" in p for p in hop.problems)

    def test_different_dataset_on_disk_breaks_the_chain(self, isolated_mlflow, trained, corpus, tmp_path):
        from src.registry.lineage import BROKEN, trace_lineage
        sd = sys.modules["synthetic_data"]
        other = tmp_path / "other_corpus"
        cfg = dataclasses.replace(sd.get_profile("smoke"), n_identities=20, seed=7)
        sd.write_dataset(other, cfg, *sd.SyntheticDatasetGenerator(cfg).generate())
        version, _, repo, _ = self._published(isolated_mlflow, trained, corpus, tmp_path)
        trace = trace_lineage(version, "xyz-mdm-matcher", data_dir=str(other), triton_repository=str(repo))
        assert {h.name: h for h in trace.hops}["dataset"].status == BROKEN

    def test_missing_dataset_identity_is_reported_not_invented(self, isolated_mlflow, trained, tmp_path):
        from src.registry.lineage import NOT_RECORDED, trace_lineage
        ds, models = trained
        version, _ = _register(ds, models)        # a run with no dataset identity
        trace = trace_lineage(version, "xyz-mdm-matcher", data_dir=None, triton_repository=None)
        assert {h.name: h for h in trace.hops}["dataset"].status == NOT_RECORDED
        assert not trace.complete


# ─── Registry migration ───────────────────────────────────────────────────────

class TestRegistryMigration:
    def test_target_must_be_the_server(self):
        from src.registry.migrate import RegistryMigrationError, migrate_registry
        with pytest.raises(RegistryMigrationError, match="MLflow server"):
            migrate_registry("sqlite:///a.db", "sqlite:///b.db", "m", "e")


# ─── 15 / 16. Benchmark + decision tiers ──────────────────────────────────────

class TestTierReport:
    def test_tier_boundaries_match_the_inference_service(self):
        assert [t for t, _ in DECISION_TIERS] == [
            "AUTO_MATCH", "LIKELY_MATCH", "UNCERTAIN", "LIKELY_NON", "AUTO_NON_MATCH"]
        cases = [(1.0, "AUTO_MATCH"), (0.95, "AUTO_MATCH"), (0.9499, "LIKELY_MATCH"),
                 (0.85, "LIKELY_MATCH"), (0.8499, "UNCERTAIN"), (0.50, "UNCERTAIN"),
                 (0.4999, "LIKELY_NON"), (0.15, "LIKELY_NON"), (0.1499, "AUTO_NON_MATCH"),
                 (0.0, "AUTO_NON_MATCH")]
        for score, tier in cases:
            assert tier_of(score) == tier

    def test_tiers_agree_with_the_inference_service_source(self):
        """Cross-check against the real serving code when it is checked out alongside."""
        src = REPO_ROOT / "xyz-model-inference-service-main" / "xyz-model-inference-service-main" \
            / "src" / "services" / "inference_service.py"
        if not src.is_file():
            pytest.skip("inference service source not available")
        import re
        text = src.read_text(encoding="utf-8")
        body = text[text.index("def _classify_decision"):]
        thresholds = [float(x) for x in re.findall(r"score >= ([0-9.]+)", body)[:4]]
        assert thresholds == [lower for _, lower in DECISION_TIERS[:4]]

    def test_auto_match_precision_and_recall(self):
        #        4 matches auto-merged, 1 wrong merge, 1 match in review, 1 match rejected
        labels = [1, 1, 1, 1, 0, 1, 1, 0, 0, 0]
        scores = [.99, .98, .97, .96, .96, .70, .05, .40, .10, .01]
        r = tier_report(labels, scores)
        assert r["auto_match"] == {"pairs": 5, "correct_merges": 4, "wrong_merges": 1,
                                   "precision": 0.8, "recall": round(4 / 6, 6)}
        assert r["auto_non_match"]["pairs"] == 3
        assert r["auto_non_match"]["missed_matches"] == 1
        assert r["auto_non_match"]["precision"] == round(2 / 3, 6)
        assert r["review"]["pairs"] == 2 and r["review"]["true_matches"] == 1
        assert sum(t["pairs"] for t in r["tiers"].values()) == 10
        assert r["tiers"]["UNCERTAIN"]["pairs"] == 1 and r["tiers"]["LIKELY_NON"]["pairs"] == 1

    def test_empty_tier_reports_none_not_zero_or_one(self):
        r = tier_report([1, 0], [0.6, 0.4])
        assert r["auto_match"]["pairs"] == 0 and r["auto_match"]["precision"] is None
        assert r["auto_match"]["recall"] == 0.0            # one true match, none auto-merged

    def test_breakdowns_by_tenant_and_negative_kind(self):
        labels = [1, 0, 1, 0]
        scores = [.99, .99, .98, .01]
        r = tier_report(labels, scores, tenants=[T1, T1, T2, T2],
                        negative_kinds=[None, "household", None, "random_pair"])
        assert r["by_tenant"][T1]["auto_match_precision"] == 0.5
        assert r["by_tenant"][T1]["auto_match_wrong_merges"] == 1
        assert r["by_tenant"][T2]["auto_match_precision"] == 1.0
        assert r["non_matches_by_kind"]["household"]["AUTO_MATCH"] == 1
        assert r["non_matches_by_kind"]["random_pair"]["AUTO_NON_MATCH"] == 1

    def test_at_0_5_matches_sklearn(self):
        from sklearn.metrics import f1_score, precision_score, recall_score
        rng = np.random.default_rng(3)
        labels = rng.integers(0, 2, 200)
        scores = np.clip(labels * 0.6 + rng.normal(0.2, 0.25, 200), 0, 1)
        r = tier_report(labels, scores)["at_threshold_0_5"]
        pred = (scores >= 0.5).astype(int)
        assert r["precision"] == pytest.approx(precision_score(labels, pred), abs=1e-6)
        assert r["recall"] == pytest.approx(recall_score(labels, pred), abs=1e-6)
        assert r["f1"] == pytest.approx(f1_score(labels, pred), abs=1e-6)


class TestBenchmark:
    def test_benchmark_id_is_frozen_by_content(self, corpus):
        from src.adapters.synthetic_data import load_benchmark_pairs
        pairs = [p for p, _ in load_benchmark_pairs(str(corpus[0]))]
        a = benchmark_id(pairs, "synds-x")
        assert a == benchmark_id(list(reversed(pairs)), "synds-x")
        assert a != benchmark_id(pairs, "synds-y")
        flipped = dataclasses.replace(pairs[0], label=1 - pairs[0].label)
        assert a != benchmark_id([flipped] + pairs[1:], "synds-x")
        assert a.startswith(f"bench-{len(pairs)}-")

    async def test_benchmark_scores_the_serving_artifact_and_records_it(self, isolated_mlflow, trained, corpus):
        """The model is scored through its ONNX serving artifact, on every benchmark pair."""
        from mlflow import MlflowClient
        from src.adapters.synthetic_data import dataset_provenance, load_benchmark_pairs
        from src.evaluators.benchmark import run_benchmark
        from tests.unit.test_xgboost_training_path import FEATURE_NAMES

        ds, models = trained
        version, run_id, _ = _register_with_dataset(ds, models, dataset_provenance(str(corpus[0])))
        bench = [p for p, _ in load_benchmark_pairs(str(corpus[0]))]
        rng = np.random.default_rng(11)

        class Features:
            _feature_version = "v2.0.0"
            last_feature_names = FEATURE_NAMES
            calls = []

            async def get_offline_features(self, entity_pairs, tenant_id, as_of_timestamp):
                self.calls.append(tenant_id)
                return {f"{a}:{b}": rng.uniform(0, 1, 50).tolist() for a, b in entity_pairs}

        fc = Features()
        report = await run_benchmark(version, str(corpus[0]), "xyz-mdm-matcher", feature_client=fc)

        assert report["pairs"] == len(bench)
        assert report["benchmark_id"] == benchmark_id(bench, corpus[1]["dataset_id"])
        assert report["dataset_id"] == corpus[1]["dataset_id"] and report["split"] == "holdout"
        assert sorted(set(fc.calls)) == sorted({p.tenant_id for p in bench})     # per tenant
        assert sum(t["pairs"] for t in report["tiers"].values()) == len(bench)
        client = MlflowClient()
        mv = client.get_model_version("xyz-mdm-matcher", version)
        assert mv.tags["benchmark_id"] == report["benchmark_id"]
        stored = json.loads(isolated_mlflow.artifacts.load_text(
            f"runs:/{run_id}/benchmark/{report['benchmark_id']}.json"))
        assert stored["auto_match"] == report["auto_match"]
        assert stored["onnx_sha256"] == mv.tags["onnx_sha256"]

    async def test_incomplete_feature_coverage_fails_the_benchmark(self, isolated_mlflow, trained, corpus):
        from src.adapters.synthetic_data import dataset_provenance
        from src.evaluators.benchmark import BenchmarkError, run_benchmark
        from tests.unit.test_xgboost_training_path import FEATURE_NAMES
        ds, models = trained
        version, _, _ = _register_with_dataset(ds, models, dataset_provenance(str(corpus[0])))

        class Partial:
            _feature_version = "v2.0.0"
            last_feature_names = FEATURE_NAMES

            async def get_offline_features(self, entity_pairs, tenant_id, as_of_timestamp):
                return {f"{a}:{b}": [0.5] * 50 for a, b in list(entity_pairs)[1:]}   # one missing

        with pytest.raises(BenchmarkError, match="have no features"):
            await run_benchmark(version, str(corpus[0]), "xyz-mdm-matcher", feature_client=Partial())


# ─── Feature version consistency (client side) ────────────────────────────────

class TestFeatureVersionCheck:
    async def test_other_feature_version_in_the_answer_is_rejected(self):
        import httpx
        from src.adapters.feature_store_client import FeatureStoreClient

        def handler(request):
            return httpx.Response(200, json={
                "feature_version": "v1.0.0", "feature_names": [f"f{i}" for i in range(50)],
                "results": [{"entity_id_1": "A", "entity_id_2": "B", "feature_vector": [0.5] * 50}]})

        client = FeatureStoreClient(base_url="http://fs", retrieval_mode="offline",
                                    feature_version="v2.0.0", transport=httpx.MockTransport(handler))
        with pytest.raises(ValueError, match="feature_version"):
            await client._fetch_offline([("A", "B")], T1, datetime(2026, 10, 5))
