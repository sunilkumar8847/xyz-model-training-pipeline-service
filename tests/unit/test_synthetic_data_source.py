"""
Unit tests for the SYNTHETIC development training-data source.

Replaces the Stage-1 half of test_demo_data_source.py. The old tests proved that
Stage 1 read demo/data/training_pairs.json when DEMO_DATA_DIR was set; that path no
longer exists. These tests prove the replacement is explicit, gated and correctly
labelled:

  - synthetic data is only read when SYNTHETIC_DATA_DIR is explicitly set
  - with nothing configured, no synthetic or demo data is consumed
  - provenance is "synthetic", never "demo_synthetic"
  - the old demo dataset is refused, not silently loaded
  - staging/production refuse synthetic data
  - only the TRAIN split is returned; holdout pairs are never trained on

Dependency type: NONE for most tests — a minimal synthetic dataset is written to
tmp_path with pyarrow in the generator's schema. One compatibility test uses the real
repository-root generator (synthetic_data) and is skipped if it is not importable.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.adapters.feature_store_client import FeatureStoreClient
from src.adapters.synthetic_data import (
    SyntheticDataError,
    load_synthetic_entity_fields,
    load_synthetic_labeled_pairs,
    load_synthetic_manifest,
)
from src.core.config import Environment, Settings, check_training_data_sources, settings
from src.domain.models import RetrainingTrigger, TrainingRun
from src.pipeline.stages.data_collection import DataCollectionStage

TENANT = "00000000-0000-0000-0000-000000000001"
EVENT_TS = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)


def _run() -> TrainingRun:
    return TrainingRun(trigger=RetrainingTrigger.MANUAL, triggered_by="unit-test")


def _pair(e1, e2, label, split="train", label_source="synthetic"):
    return {
        "entity_id_1": e1, "entity_id_2": e2, "tenant_id": TENANT, "label": label,
        "confidence": 1.0, "label_source": label_source, "event_ts": EVENT_TS,
        "canonical_id_1": f"CANON-{e1}", "canonical_id_2": f"CANON-{e2 if label == 0 else e1}",
        "negative_kind": None if label else "random_pair",
        "difficulty": "clear", "split": split, "n_variations": 0,
    }


def _write_synthetic_dir(path: Path, pairs, entities=None, manifest=None) -> Path:
    """A minimal dataset in the generator's on-disk schema."""
    path.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(pairs), path / "pairs.parquet")
    entities = entities if entities is not None else [
        {"entity_id": "C1", "fields_json": json.dumps({"name": "Ada Lovelace"})},
    ]
    pq.write_table(pa.Table.from_pylist(entities), path / "entities.parquet")
    # A caller-supplied manifest is written verbatim (those tests are about bad
    # provenance). The default one is sealed with a valid identity, as the generator
    # does: datasets without an identity are refused.
    if manifest is None:
        manifest = _seal(path, {
            "generator": "synthetic_data", "source": "synthetic", "is_production_data": False,
        })
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _seal(path: Path, manifest: dict) -> dict:
    import hashlib

    files = {n: hashlib.sha256((path / n).read_bytes()).hexdigest()
             for n in ("entities.parquet", "pairs.parquet")}
    identity = {"source": "synthetic", "label_source": "synthetic",
                "note": "hand-built test dataset", "files": files}
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {**manifest, "identity": identity, "file_sha256": files,
            "dataset_id": "synds-" + hashlib.sha256(canonical.encode("ascii")).hexdigest()[:16]}


@pytest.fixture
def clean_sources(monkeypatch):
    """Isolate every test from the developer's .env.local."""
    monkeypatch.setattr(settings, "SYNTHETIC_DATA_DIR", None)
    monkeypatch.setattr(settings, "DEMO_DATA_DIR", None)
    monkeypatch.setattr(settings, "ENVIRONMENT", Environment.DEVELOPMENT)


# ─── 1. Explicit selection ────────────────────────────────────────────────────

async def test_synthetic_pairs_loaded_when_explicitly_configured(tmp_path, monkeypatch, clean_sources):
    d = _write_synthetic_dir(tmp_path, [_pair("C1", "C2", 1), _pair("C3", "C4", 0)])
    monkeypatch.setattr(settings, "SYNTHETIC_DATA_DIR", str(d))

    pairs = await DataCollectionStage(db_session=None).execute(_run())

    assert sorted((p.entity_id_1, p.entity_id_2, p.label) for p in pairs) == [
        ("C1", "C2", 1), ("C3", "C4", 0),
    ]
    assert all(p.tenant_id == TENANT for p in pairs)


# ─── 2. Nothing consumed by default ───────────────────────────────────────────

async def test_no_training_data_consumed_by_default(clean_sources):
    """With no source configured, Stage 1 must not pick up synthetic OR demo data."""
    pairs = await DataCollectionStage(db_session=None).execute(_run())
    assert pairs == []


def test_synthetic_data_dir_default_is_unset():
    """The shipped default is no synthetic data — production behaviour."""
    assert Settings.model_fields["SYNTHETIC_DATA_DIR"].default is None


def test_old_demo_dataset_not_read_even_if_present_on_disk(clean_sources):
    """demo/data may still exist for showcase scripts; nothing in training reads it."""
    import src.pipeline.stages.data_collection as dc
    source = Path(dc.__file__).read_text(encoding="utf-8")
    assert "training_pairs.json" not in source
    assert "_collect_demo_pairs" not in source
    assert "DEMO_DATA_DIR" not in source


# ─── 3. Provenance ────────────────────────────────────────────────────────────

async def test_provenance_is_synthetic(tmp_path, monkeypatch, clean_sources):
    monkeypatch.setattr(settings, "SYNTHETIC_DATA_DIR", str(
        _write_synthetic_dir(tmp_path, [_pair("C1", "C2", 1)])))

    pairs = await DataCollectionStage(db_session=None).execute(_run())

    assert {p.source for p in pairs} == {"synthetic"}
    assert "demo_synthetic" not in {p.source for p in pairs}


def test_demo_synthetic_label_source_is_rejected(tmp_path, clean_sources):
    d = _write_synthetic_dir(tmp_path, [_pair("C1", "C2", 1, label_source="demo_synthetic")])
    with pytest.raises(SyntheticDataError, match="label_source"):
        load_synthetic_labeled_pairs(str(d))


def test_manifest_must_declare_synthetic_source(tmp_path, clean_sources):
    d = _write_synthetic_dir(tmp_path, [_pair("C1", "C2", 1)],
                             manifest={"source": "hitl", "is_production_data": False})
    with pytest.raises(SyntheticDataError, match="not a synthetic dataset"):
        load_synthetic_manifest(str(d))


def test_manifest_must_declare_not_production(tmp_path, clean_sources):
    d = _write_synthetic_dir(tmp_path, [_pair("C1", "C2", 1)],
                             manifest={"source": "synthetic"})
    with pytest.raises(SyntheticDataError, match="is_production_data"):
        load_synthetic_manifest(str(d))


def test_old_demo_directory_layout_is_refused(tmp_path, clean_sources):
    """
    Pointing SYNTHETIC_DATA_DIR at the old demo/data layout (JSON files, a manifest
    without the provenance fields) must fail loudly, not load demo pairs.
    """
    (tmp_path / "training_pairs.json").write_text("[]", encoding="utf-8")
    (tmp_path / "customers.json").write_text("[]", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"generator_version": "1.0.0", "seed": 42}), encoding="utf-8")
    with pytest.raises(SyntheticDataError):
        load_synthetic_labeled_pairs(str(tmp_path))


def test_missing_manifest_is_refused(tmp_path, clean_sources):
    with pytest.raises(SyntheticDataError, match="not found"):
        load_synthetic_manifest(str(tmp_path))


# ─── Split handling ───────────────────────────────────────────────────────────

def test_only_train_split_is_returned(tmp_path, clean_sources):
    d = _write_synthetic_dir(tmp_path, [
        _pair("C1", "C2", 1, split="train"),
        _pair("C5", "C6", 1, split="holdout"),
    ])
    pairs = load_synthetic_labeled_pairs(str(d))
    assert [(p.entity_id_1, p.entity_id_2) for p in pairs] == [("C1", "C2")]


def test_labeled_at_is_naive_utc(tmp_path, clean_sources):
    d = _write_synthetic_dir(tmp_path, [_pair("C1", "C2", 1)])
    (pair,) = load_synthetic_labeled_pairs(str(d))
    assert pair.labeled_at.tzinfo is None
    assert pair.labeled_at == EVENT_TS.replace(tzinfo=None)


# ─── Environment gate: production never falls back to synthetic ─────────────

@pytest.mark.parametrize("env", [Environment.STAGING, Environment.PRODUCTION])
def test_synthetic_data_refused_outside_development(tmp_path, monkeypatch, clean_sources, env):
    d = _write_synthetic_dir(tmp_path, [_pair("C1", "C2", 1)])
    monkeypatch.setattr(settings, "SYNTHETIC_DATA_DIR", str(d))
    monkeypatch.setattr(settings, "ENVIRONMENT", env)
    with pytest.raises(ValueError, match="only permitted in"):
        load_synthetic_labeled_pairs(str(d))


@pytest.mark.parametrize("env", [Environment.STAGING, Environment.PRODUCTION])
def test_settings_refuse_synthetic_data_at_startup_outside_development(tmp_path, env):
    """The startup validator catches a misconfigured deploy before any run starts."""
    with pytest.raises(ValueError, match="only permitted in"):
        Settings(ENVIRONMENT=env, SYNTHETIC_DATA_DIR=str(tmp_path), DEMO_DATA_DIR=None)


@pytest.mark.parametrize("env", [Environment.DEVELOPMENT, Environment.TEST])
def test_settings_allow_synthetic_data_in_development_and_test(tmp_path, env):
    s = Settings(ENVIRONMENT=env, SYNTHETIC_DATA_DIR=str(tmp_path), DEMO_DATA_DIR=None)
    assert s.SYNTHETIC_DATA_DIR == str(tmp_path)


# ─── Legacy DEMO_DATA_DIR fails loudly ────────────────────────────────────────

def test_legacy_demo_data_dir_rejected_at_startup(tmp_path):
    """
    model_config uses extra="ignore"; if the field were simply deleted, an old
    .env.local would be silently ignored. It must be rejected with guidance instead.
    """
    with pytest.raises(ValueError, match="DEMO_DATA_DIR is no longer supported"):
        Settings(DEMO_DATA_DIR=str(tmp_path), SYNTHETIC_DATA_DIR=None)


def test_legacy_demo_data_dir_rejected_at_runtime(tmp_path, monkeypatch, clean_sources):
    """Setting it on the live settings object (bypassing the validator) is also caught."""
    monkeypatch.setattr(settings, "DEMO_DATA_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="DEMO_DATA_DIR is no longer supported"):
        check_training_data_sources(settings)


# ─── Feature Store client entity catalog ─────────────────────────────────────

def test_from_settings_uses_synthetic_catalog(tmp_path, monkeypatch, clean_sources):
    entities = [
        {"entity_id": "C1", "fields_json": json.dumps({"name": "Ada Lovelace"})},
        # An omitted key must stay omitted (it drives str_schema_similarity).
        {"entity_id": "C2", "fields_json": json.dumps({"name": "Alan Turing", "email": None})},
    ]
    monkeypatch.setattr(settings, "SYNTHETIC_DATA_DIR", str(
        _write_synthetic_dir(tmp_path, [_pair("C1", "C2", 0)], entities=entities)))

    client = FeatureStoreClient.from_settings()

    assert client._entity_fields("C1") == {"name": "Ada Lovelace"}
    assert client._entity_fields("C2") == {"name": "Alan Turing", "email": None}
    assert "phone" not in client._entity_fields("C2")
    assert client._entity_fields("unknown") is None


def test_from_settings_without_synthetic_data_has_no_catalog(clean_sources):
    assert FeatureStoreClient.from_settings()._entity_fields is None


def test_entity_catalog_also_enforces_provenance(tmp_path, clean_sources):
    d = _write_synthetic_dir(tmp_path, [_pair("C1", "C2", 1)],
                             manifest={"source": "demo", "is_production_data": False})
    with pytest.raises(SyntheticDataError):
        load_synthetic_entity_fields(str(d))


# ─── Compatibility with the real generator ───────────────────────────────────

def test_reads_a_dataset_written_by_the_real_generator(tmp_path, clean_sources):
    """
    Guards against schema drift between the repository-root generator and this
    reader. Skipped when the generator package is not importable.
    """
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    synthetic_data = pytest.importorskip("synthetic_data")

    config = synthetic_data.get_profile("smoke")
    identities, records, pairs = synthetic_data.SyntheticDatasetGenerator(config).generate()
    synthetic_data.write_dataset(tmp_path, config, identities, records, pairs)

    loaded = load_synthetic_labeled_pairs(str(tmp_path))
    expected_train = sum(1 for p in pairs if p.split == "train")
    assert len(loaded) == expected_train > 0
    assert {p.source for p in loaded} == {"synthetic"}
    assert {p.label for p in loaded} == {0, 1}

    catalog = load_synthetic_entity_fields(str(tmp_path))
    assert len(catalog) == len(records)
