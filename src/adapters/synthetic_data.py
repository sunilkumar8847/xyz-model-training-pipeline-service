"""
model-training-pipeline/src/adapters/synthetic_data.py

Reads a SYNTHETIC development dataset produced by the repository-root generator
(`python -m synthetic_data --profile dev`). Development/test only.

This is deliberately a set of plain functions, not a training-data "provider"
abstraction — that is planned for a later phase, once the real production label
sources (MDM / HITL / active learning) have concrete contracts to implement against.

Every read passes two gates:

  1. ENVIRONMENT gate — refused outside development/test
     (src.core.config.check_training_data_sources).
  2. PROVENANCE gate  — the directory's manifest.json must declare
     source == "synthetic" and is_production_data == false, and every pair row must
     carry label_source == "synthetic". This is what stops the old demo dataset
     (demo/data/, label source "demo_synthetic") from being consumed by mistake.
"""
from __future__ import annotations

import json
from datetime import timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.core.config import check_training_data_sources, settings
from src.domain.models import LabeledPair

SYNTHETIC_SOURCE = "synthetic"
MANIFEST_FILE = "manifest.json"
PAIRS_FILE = "pairs.parquet"
ENTITIES_FILE = "entities.parquet"
TRAIN_SPLIT = "train"

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
    return manifest


def load_synthetic_labeled_pairs(data_dir: str) -> List[LabeledPair]:
    """
    Labeled pairs from the dataset's TRAIN split only. The generator's holdout split
    is reserved for evaluation/scoring and is never returned here.
    """
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
        if r["split"] != TRAIN_SPLIT:
            continue
        labeled_at = r["event_ts"]
        # Domain timestamps are naive UTC throughout this service.
        if labeled_at.tzinfo is not None:
            labeled_at = labeled_at.astimezone(timezone.utc).replace(tzinfo=None)
        pairs.append(LabeledPair(
            entity_id_1=r["entity_id_1"],
            entity_id_2=r["entity_id_2"],
            tenant_id=r["tenant_id"],
            label=int(r["label"]),
            confidence=float(r["confidence"]),
            source=SYNTHETIC_SOURCE,
            labeled_at=labeled_at,
        ))
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
