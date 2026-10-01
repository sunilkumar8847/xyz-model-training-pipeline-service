"""
model-training-pipeline/src/pipeline/stages/data_collection.py

Stage 1: Collect labeled entity pairs from all label sources.

Label Sources:
  1. HITL Approved Merges  → Positive (~5,000/week, high quality)
  2. HITL Rejected Merges  → Negative (~3,000/week, high quality)
  3. Auto-Merge (>0.99)    → Positive (~10,000/week, medium quality)
  4. Synthetic Negatives   → Negative (~15,000/week, generated)
  5. Active Learning       → Uncertain (~2,000/week, high-value)

Target: ≥50,000 labeled pairs per training run.
Positive:Negative ratio target: 1:5
"""
from __future__ import annotations

import hashlib
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID

import numpy as np

from src.core.config import Environment, settings
from src.core.exceptions import InsufficientTrainingDataError
from src.domain.models import LabeledPair, RetrainingTriggerEvent, TrainingDataset, TrainingRun

logger = logging.getLogger(__name__)


# Environments where MIN_LABELED_PAIRS is a hard requirement (LLD PART IV §4.2:
# minimum 50,000 labeled pairs per training run). Development and test run on small
# synthetic datasets and only warn.
MIN_PAIRS_ENFORCED_ENVIRONMENTS = frozenset({Environment.STAGING, Environment.PRODUCTION})


def enforce_minimum_training_pairs(n_pairs: int) -> None:
    """
    Staging/production: fewer than MIN_LABELED_PAIRS raises. Before this, the minimum
    was only ever logged as a warning, in every environment — production included.
    Development/test: the same shortfall is logged as a warning and training proceeds,
    because the synthetic development dataset is far smaller than production minimums.
    """
    if n_pairs >= settings.MIN_LABELED_PAIRS:
        return
    message = f"Insufficient training data: {n_pairs} < minimum {settings.MIN_LABELED_PAIRS}"
    if settings.ENVIRONMENT in MIN_PAIRS_ENFORCED_ENVIRONMENTS:
        raise InsufficientTrainingDataError(n_pairs, settings.MIN_LABELED_PAIRS)
    logger.warning(
        f"{message} — allowed because ENVIRONMENT={settings.ENVIRONMENT.value}; "
        f"enforced in {sorted(e.value for e in MIN_PAIRS_ENFORCED_ENVIRONMENTS)}"
    )


class DataCollectionStage:
    """
    Collects training data from all label sources.
    Applies quality filtering, deduplication, and class balancing.
    """

    def __init__(self, db_session, feature_store_client=None):
        self._db = db_session
        self._feature_store = feature_store_client

    async def execute(self, run: TrainingRun) -> List[LabeledPair]:
        """
        Collect all labeled pairs for this training run.
        Returns a balanced, deduplicated dataset.
        """
        logger.info(f"[Run {run.run_id}] Stage 1: Data Collection started")
        start = datetime.utcnow()

        all_pairs: List[LabeledPair] = []

        # 1. HITL Approved Merges (Positive)
        hitl_positive = await self._collect_hitl_approved(
            lookback_days=settings.TRAINING_LOOKBACK_DAYS
        )
        all_pairs.extend(hitl_positive)
        logger.info(f"  HITL positive: {len(hitl_positive)} pairs")

        # 2. HITL Rejected Merges (Negative)
        hitl_negative = await self._collect_hitl_rejected(
            lookback_days=settings.TRAINING_LOOKBACK_DAYS
        )
        all_pairs.extend(hitl_negative)
        logger.info(f"  HITL negative: {len(hitl_negative)} pairs")

        # 3. Auto-Merge Inferred Positives
        auto_positive = await self._collect_auto_merges(
            lookback_days=settings.TRAINING_LOOKBACK_DAYS
        )
        all_pairs.extend(auto_positive)
        logger.info(f"  Auto-merge positive: {len(auto_positive)} pairs")

        # 4. Synthetic Negatives (random far-apart pairs)
        n_synthetic = max(
            0,
            len([p for p in all_pairs if p.label == 1]) * settings.NEGATIVE_SAMPLING_RATIO
            - len([p for p in all_pairs if p.label == 0])
        )
        synthetic_neg = await self._generate_synthetic_negatives(n_synthetic)
        all_pairs.extend(synthetic_neg)
        logger.info(f"  Synthetic negative: {len(synthetic_neg)} pairs")

        # 5. Active Learning Samples (uncertain pairs from inference)
        active = await self._collect_active_learning_samples()
        all_pairs.extend(active)
        logger.info(f"  Active learning: {len(active)} pairs")

        # 6. Synthetic development data — explicit opt-in via SYNTHETIC_DATA_DIR,
        #    development/test only (enforced in src.adapters.synthetic_data).
        if settings.SYNTHETIC_DATA_DIR:
            synthetic = self._collect_synthetic_pairs(settings.SYNTHETIC_DATA_DIR)
            all_pairs.extend(synthetic)
            logger.warning(
                f"  SYNTHETIC development data: {len(synthetic)} pairs from "
                f"{settings.SYNTHETIC_DATA_DIR} — not production labels"
            )

        # Deduplicate by pair hash
        all_pairs = self._deduplicate(all_pairs)

        # Quality filter: drop low-confidence labels
        all_pairs = [p for p in all_pairs if p.confidence >= 0.7]

        # Dataset fingerprint: deterministic over the exact (pair, label) set, so two
        # runs over the same labels get the same dataset_version and a run can be
        # reproduced from it (LLD PART III §3.2).
        run.dataset_version = self._compute_dataset_version(all_pairs)
        logger.info(f"[Run {run.run_id}] dataset_version={run.dataset_version}")

        # Log summary
        n_pos = sum(1 for p in all_pairs if p.label == 1)
        n_neg = sum(1 for p in all_pairs if p.label == 0)
        duration = (datetime.utcnow() - start).total_seconds()

        logger.info(
            f"[Run {run.run_id}] Data collection complete: "
            f"{len(all_pairs)} total ({n_pos} pos, {n_neg} neg) in {duration:.1f}s"
        )

        enforce_minimum_training_pairs(len(all_pairs))

        return all_pairs

    async def _collect_hitl_approved(self, lookback_days: int) -> List[LabeledPair]:
        """
        Query HITL service for human-approved merges.
        In production: calls the HITL workflow service API.
        """
        # Production: query HITL service database for approved merges
        # SELECT entity_id_1, entity_id_2, tenant_id, approved_at
        # FROM hitl_decisions
        # WHERE decision = 'APPROVE' AND approved_at > NOW() - INTERVAL '{lookback_days} days'
        # For now: return empty (real data from HITL service in production)
        return []

    async def _collect_hitl_rejected(self, lookback_days: int) -> List[LabeledPair]:
        """Query HITL service for human-rejected matches."""
        return []

    async def _collect_auto_merges(self, lookback_days: int) -> List[LabeledPair]:
        """
        Collect high-confidence auto-merges (score > 0.99) as positive labels.
        These are inferred positives from production inference.
        """
        return []

    async def _generate_synthetic_negatives(self, n: int) -> List[LabeledPair]:
        """
        Generate synthetic negative pairs by randomly sampling
        entities from different tenants or very different clusters.
        These are guaranteed non-matches (cross-tenant pairs are never duplicates).
        """
        if n <= 0:
            return []

        # Production: query entity table for random pairs from different tenants
        # or entities with very different blocking keys
        logger.info(f"  Generating {n} synthetic negatives")
        return []

    async def _collect_active_learning_samples(self) -> List[LabeledPair]:
        """
        Collect uncertain pairs (score ~0.5) that have been labeled
        by the active learning workflow.
        """
        return []

    def _collect_synthetic_pairs(self, data_dir: str) -> List[LabeledPair]:
        """
        TRAIN-split labeled pairs from a synthetic development dataset. The reader
        enforces the environment gate and verifies provenance (source == "synthetic"),
        so the old demo dataset cannot be consumed through this path.
        """
        from src.adapters.synthetic_data import load_synthetic_labeled_pairs

        return load_synthetic_labeled_pairs(data_dir)

    @staticmethod
    def _compute_dataset_version(pairs: List[LabeledPair]) -> str:
        """
        Content hash of the labeled dataset. Order-independent (pairs are sorted
        first) so the same labels always yield the same version, regardless of the
        order the sources returned them in.
        Format: ds-<n_pairs>-<sha256[:12]>
        """
        import hashlib

        fingerprint = sorted(
            f"{min(p.entity_id_1, p.entity_id_2)}:{max(p.entity_id_1, p.entity_id_2)}:{p.label}"
            for p in pairs
        )
        digest = hashlib.sha256("\n".join(fingerprint).encode()).hexdigest()[:12]
        return f"ds-{len(pairs)}-{digest}"

    def _deduplicate(self, pairs: List[LabeledPair]) -> List[LabeledPair]:
        """Remove duplicate pair (same entity_id_1 + entity_id_2, regardless of order)."""
        seen = set()
        unique = []
        for pair in pairs:
            key = tuple(sorted([pair.entity_id_1, pair.entity_id_2]))
            if key not in seen:
                seen.add(key)
                unique.append(pair)
        return unique


class TrainingDataSplitter:
    """
    Split labeled pairs into train/validation/test WITHOUT identity-cluster leakage.

    Why not a random pair-level split: several labeled pairs can describe the same
    real-world identity (records A, B, C of one person give positives A-B, A-C, B-C).
    A pair-level split puts A-B in train and A-C in test; because B and C are near
    duplicates, the two feature vectors are nearly identical and test metrics are
    inflated. Measured on the synthetic dev set: with the old pair-level split, 100%
    of test positives had their identity's positive pairs in train.

    Grouping: entities joined by POSITIVE pairs form a match cluster (union-find).
    This uses only the labels — no ground-truth identity field — so it works the same
    on production labels. Every positive pair lies inside one cluster, and each cluster
    is assigned to exactly one split, so no identity's matches are spread across splits.

    Negative pairs follow the cluster of their lexicographically smaller entity id
    (orientation-independent). Stricter "entity-disjoint" splitting was evaluated and
    rejected: negatives connect almost every cluster (the dev pair graph is one
    component per tenant), so an entity-disjoint split must drop every cross-split
    negative, and the small splits keep only ~f^2 of them — measured validation
    ratio 1:0.59 and test 1:0.78 instead of ~1:5, making their metrics meaningless.
    Consequence, stated honestly: an entity record CAN appear in a train negative and
    in a validation/test pair. `last_stats` reports how often.

    Stratification: clusters are allocated separately per (tenant, has-positive-pairs)
    stratum, filling train, then validation, then test by pair weight, so every tenant
    and both labels appear in each split in close to the target proportions.
    Deterministic for a given random_seed.
    """

    def __init__(
        self,
        test_ratio: float = 0.15,
        val_ratio: float = 0.10,
        random_seed: int = 42,
    ):
        if not (0 < test_ratio < 1 and 0 <= val_ratio < 1 and test_ratio + val_ratio < 1):
            raise ValueError(f"invalid split ratios test={test_ratio} val={val_ratio}")
        self._test_ratio = test_ratio
        self._val_ratio = val_ratio
        self._seed = random_seed
        self.last_stats: Dict[str, int] = {}

    def split(self, pairs: List[LabeledPair]) -> tuple:
        """Returns (train_indices, val_indices, test_indices); every pair in exactly one."""
        import random
        from collections import defaultdict

        self.last_stats = {}
        if not pairs:
            return [], [], []

        # Entities are keyed by (tenant, entity_id): ids are only unique per tenant.
        parent: Dict[tuple, tuple] = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def key(p, eid):
            return (p.tenant_id, eid)

        for p in pairs:
            a, b = find(key(p, p.entity_id_1)), find(key(p, p.entity_id_2))
            if p.label == 1 and a != b:
                parent[max(a, b)] = min(a, b)

        # Anchor every pair to one cluster; count pair weight per cluster.
        anchors = []
        weight: Dict[tuple, int] = defaultdict(int)
        has_positive: Dict[tuple, bool] = defaultdict(bool)
        for p in pairs:
            anchor_entity = min(p.entity_id_1, p.entity_id_2)
            cluster = find(key(p, anchor_entity))
            anchors.append(cluster)
            weight[cluster] += 1
            if p.label == 1:
                has_positive[cluster] = True

        # Allocate clusters per stratum: train first, then val, then test.
        rng = random.Random(self._seed)
        strata: Dict[tuple, List[tuple]] = defaultdict(list)
        for cluster in weight:
            strata[(cluster[0], has_positive[cluster])].append(cluster)

        train_share = 1.0 - self._test_ratio - self._val_ratio
        assignment: Dict[tuple, str] = {}
        for stratum_key in sorted(strata):
            clusters = sorted(strata[stratum_key])
            rng.shuffle(clusters)
            total = sum(weight[c] for c in clusters)
            train_bound = train_share * total
            val_bound = (train_share + self._val_ratio) * total
            cumulative = 0
            for c in clusters:
                if cumulative < train_bound:
                    assignment[c] = "train"
                elif cumulative < val_bound:
                    assignment[c] = "val"
                else:
                    assignment[c] = "test"
                cumulative += weight[c]

        splits = {"train": [], "val": [], "test": []}
        for i, cluster in enumerate(anchors):
            splits[assignment[cluster]].append(i)

        # Guarantee by construction, checked anyway: a positive pair never straddles.
        for p in pairs:
            if p.label == 1:
                c1, c2 = find(key(p, p.entity_id_1)), find(key(p, p.entity_id_2))
                if c1 != c2:
                    raise AssertionError("positive pair spans two match clusters")

        self.last_stats = self._stats(pairs, splits)
        # Anchored clusters (clusters that own at least one pair).
        self.last_stats["clusters"] = len(weight)
        logger.info(
            f"Dataset split (identity-cluster): {len(splits['train'])} train / "
            f"{len(splits['val'])} val / {len(splits['test'])} test; "
            f"{self.last_stats['clusters']} match clusters; val/test pairs sharing an "
            f"entity record with train: {self.last_stats['val_pairs_sharing_train_entity']}"
            f"/{self.last_stats['test_pairs_sharing_train_entity']}"
        )
        return splits["train"], splits["val"], splits["test"]

    @staticmethod
    def _stats(pairs: List[LabeledPair], splits: Dict[str, List[int]]) -> Dict[str, int]:
        train_entities = {
            (pairs[i].tenant_id, e)
            for i in splits["train"]
            for e in (pairs[i].entity_id_1, pairs[i].entity_id_2)
        }

        def sharing(idx):
            return sum(
                1 for i in idx
                if (pairs[i].tenant_id, pairs[i].entity_id_1) in train_entities
                or (pairs[i].tenant_id, pairs[i].entity_id_2) in train_entities
            )

        stats: Dict[str, int] = {}
        for name, idx in splits.items():
            stats[f"{name}_pairs"] = len(idx)
            stats[f"{name}_positive"] = sum(1 for i in idx if pairs[i].label == 1)
            stats[f"{name}_negative"] = sum(1 for i in idx if pairs[i].label == 0)
        stats["val_pairs_sharing_train_entity"] = sharing(splits["val"])
        stats["test_pairs_sharing_train_entity"] = sharing(splits["test"])
        return stats
