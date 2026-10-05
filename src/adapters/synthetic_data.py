"""
model-training-pipeline/src/adapters/synthetic_data.py

Reads a SYNTHETIC development dataset produced by the repository-root generator
(`python -m synthetic_data --profile dev`). Development/test only.

This is deliberately a set of plain functions, not a training-data "provider"
abstraction — that is planned for a later phase, once the real production label
sources (MDM / HITL / active learning) have concrete contracts to implement against.

Every read passes four gates:

  1. ENVIRONMENT gate — refused outside development/test unless bootstrap mode is
     enabled (src.core.config.check_training_data_sources).
  2. PROVENANCE gate  — the directory's manifest.json must declare
     source == "synthetic" and is_production_data == false, and every pair row must
     carry label_source == "synthetic". This is what stops the old demo dataset
     (demo/data/, label source "demo_synthetic") from being consumed by mistake.
  3. INTEGRITY gate   — the manifest must carry a dataset identity; dataset_id must be
     the hash of that identity, and both Parquet files must have the recorded sha256.
     A dataset that was edited after generation is refused.
  4. BOOTSTRAP gate   — in staging/production the manifest's own sha256 must be listed
     in BOOTSTRAP_DATASET_MANIFEST_SHA256 (it pins the file hashes, hence the data).

The synthetic dataset is the company-owned bootstrap/seed corpus. Its identity
(dataset_id, hashes, generator version, seed, tenants) is returned by
dataset_provenance() and recorded in the lineage of every model trained on it.
"""
from __future__ import annotations

import hashlib
import json
from datetime import timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.core.config import SYNTHETIC_DATA_ENVIRONMENTS, check_training_data_sources, settings
from src.domain.models import LabeledPair

SYNTHETIC_SOURCE = "synthetic"
MANIFEST_FILE = "manifest.json"
PAIRS_FILE = "pairs.parquet"
ENTITIES_FILE = "entities.parquet"
TRAIN_SPLIT = "train"
HOLDOUT_SPLIT = "holdout"

EntityFields = Dict[str, Optional[str]]


class SyntheticDataError(ValueError):
    """The configured directory is not a valid synthetic dataset."""


def load_synthetic_manifest(data_dir: str) -> dict:
    """Read and verify manifest.json. Raises SyntheticDataError if provenance is wrong."""
    check_training_data_sources(settings)

    path = Path(data_dir) / MANIFEST_FILE
    if not path.is_file():
        raise SyntheticDataError(
            f"{path} not found. SYNTHETIC_DATA_DIR must point at a directory written by "
            f"`python -m synthetic_data` (it contains {MANIFEST_FILE}, {PAIRS_FILE} and "
            f"{ENTITIES_FILE})."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))

    if manifest.get("source") != SYNTHETIC_SOURCE:
        raise SyntheticDataError(
            f"{path}: source={manifest.get('source')!r}, expected {SYNTHETIC_SOURCE!r}. "
            f"This directory is not a synthetic dataset (the old demo dataset in demo/data "
            f"is not accepted as a training source)."
        )
    if manifest.get("is_production_data") is not False:
        raise SyntheticDataError(
            f"{path}: is_production_data must be explicitly false for synthetic data, "
            f"got {manifest.get('is_production_data')!r}."
        )
    _verify_identity(Path(data_dir), manifest)
    return manifest


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_identity(data_dir: Path, manifest: dict) -> str:
    """Integrity + bootstrap gates. Returns sha256(manifest.json)."""
    path = data_dir / MANIFEST_FILE
    identity = manifest.get("identity")
    if not identity or not manifest.get("dataset_id") or not manifest.get("file_sha256"):
        raise SyntheticDataError(
            f"{path} has no dataset identity (generator {manifest.get('generator_version')!r}). "
            f"Regenerate it with `python -m synthetic_data` (generator >= 2.1.0): a model "
            f"must be traceable to the exact data it was trained on."
        )
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    expected_id = "synds-" + hashlib.sha256(canonical.encode("ascii")).hexdigest()[:16]
    if manifest["dataset_id"] != expected_id:
        raise SyntheticDataError(
            f"{path}: dataset_id {manifest['dataset_id']!r} does not match its identity block."
        )
    for name, expected in manifest["file_sha256"].items():
        actual = _sha256_file(data_dir / name)
        if actual != expected:
            raise SyntheticDataError(
                f"{data_dir / name}: content does not match the manifest (sha256 "
                f"{actual[:12]} != {expected[:12]}). The dataset was modified after generation."
            )
    manifest_hash = _sha256_file(path)
    if settings.ENVIRONMENT not in SYNTHETIC_DATA_ENVIRONMENTS:
        if not settings.BOOTSTRAP_MODE or manifest_hash not in settings.bootstrap_manifest_allowlist:
            raise SyntheticDataError(
                f"{path} (sha256 {manifest_hash}) is not an approved bootstrap corpus for "
                f"ENVIRONMENT={settings.ENVIRONMENT.value}. Only a dataset whose manifest hash is "
                f"listed in BOOTSTRAP_DATASET_MANIFEST_SHA256 may be used, with BOOTSTRAP_MODE=true."
            )
    return manifest_hash


def dataset_provenance(data_dir: str) -> dict:
    """
    The verified identity of the dataset in `data_dir`, for model lineage. Every value
    is read from the manifest AFTER it has been checked against the files.
    """
    manifest = load_synthetic_manifest(data_dir)
    identity = manifest["identity"]
    return {
        "dataset_id": manifest["dataset_id"],
        "manifest_sha256": _sha256_file(Path(data_dir) / MANIFEST_FILE),
        "content_sha256": dict(identity.get("content_sha256", {})),
        "file_sha256": dict(manifest["file_sha256"]),
        "generator": identity.get("generator"),
        "generator_version": identity.get("generator_version"),
        "seed": identity.get("seed"),
        "reference_time": identity.get("reference_time"),
        "tenants": list(identity.get("tenants", [])),
        "label_source": identity.get("label_source", SYNTHETIC_SOURCE),
        "is_production_data": False,
    }


def load_synthetic_labeled_pairs(data_dir: str) -> List[LabeledPair]:
    """
    Labeled pairs from the dataset's TRAIN split only. The generator's holdout split
    is reserved for the frozen benchmark (load_benchmark_pairs) and is never returned
    here.
    """
    return [pair for pair, _ in _load_split(data_dir, TRAIN_SPLIT)]


def load_benchmark_pairs(data_dir: str) -> List[tuple]:
    """
    The FROZEN BENCHMARK: the dataset's holdout split, as (LabeledPair, info) tuples
    where info carries negative_kind and difficulty. These identities are never in the
    train split (the generator splits by identity), and training never reads them, so
    every model version can be scored on the same pairs.
    """
    return _load_split(data_dir, HOLDOUT_SPLIT)


def _load_split(data_dir: str, split: str) -> List[tuple]:
    import pyarrow.parquet as pq

    load_synthetic_manifest(data_dir)
    rows = pq.read_table(Path(data_dir) / PAIRS_FILE).to_pylist()

    bad_sources = sorted({r["label_source"] for r in rows} - {SYNTHETIC_SOURCE})
    if bad_sources:
        raise SyntheticDataError(
            f"{Path(data_dir) / PAIRS_FILE}: unexpected label_source value(s) {bad_sources}; "
            f"every synthetic pair must have label_source={SYNTHETIC_SOURCE!r}."
        )

    pairs = []
    for r in rows:
        if r["split"] != split:
            continue
        labeled_at = r["event_ts"]
        # Domain timestamps are naive UTC throughout this service.
        if labeled_at.tzinfo is not None:
            labeled_at = labeled_at.astimezone(timezone.utc).replace(tzinfo=None)
        pairs.append((LabeledPair(
            entity_id_1=r["entity_id_1"],
            entity_id_2=r["entity_id_2"],
            tenant_id=r["tenant_id"],
            label=int(r["label"]),
            confidence=float(r["confidence"]),
            source=SYNTHETIC_SOURCE,
            labeled_at=labeled_at,
        ), {"negative_kind": r.get("negative_kind"), "difficulty": r.get("difficulty")}))
    return pairs


def load_synthetic_entity_fields(data_dir: str) -> Dict[str, EntityFields]:
    """
    entity_id -> fields, used by the ONLINE Feature Store client to send record fields
    for on-the-fly computation. A record that omits a field key keeps it omitted —
    that distinction is meaningful to the str_schema_similarity feature.
    """
    import pyarrow.parquet as pq

    load_synthetic_manifest(data_dir)
    rows = pq.read_table(
        Path(data_dir) / ENTITIES_FILE, columns=["entity_id", "fields_json"]
    ).to_pylist()
    return {r["entity_id"]: json.loads(r["fields_json"]) for r in rows}
