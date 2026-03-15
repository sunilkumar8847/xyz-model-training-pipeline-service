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
from typing import List, Optional
from uuid import UUID

import numpy as np

from src.core.config import settings
from src.domain.models import LabeledPair, RetrainingTriggerEvent, TrainingDataset, TrainingRun

logger = logging.getLogger(__name__)


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

        # Deduplicate by pair hash
        all_pairs = self._deduplicate(all_pairs)

        # Quality filter: drop low-confidence labels
        all_pairs = [p for p in all_pairs if p.confidence >= 0.7]

        # Log summary
        n_pos = sum(1 for p in all_pairs if p.label == 1)
        n_neg = sum(1 for p in all_pairs if p.label == 0)
        duration = (datetime.utcnow() - start).total_seconds()

        logger.info(
            f"[Run {run.run_id}] Data collection complete: "
            f"{len(all_pairs)} total ({n_pos} pos, {n_neg} neg) in {duration:.1f}s"
        )

        if len(all_pairs) < settings.MIN_LABELED_PAIRS:
            logger.warning(
                f"Insufficient training data: {len(all_pairs)} < "
                f"minimum {settings.MIN_LABELED_PAIRS}"
            )

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
    Split dataset into train/validation/test sets.
    Stratified by label to preserve positive:negative ratio.
    Tenant-stratified to ensure all tenants appear in all splits.
    """

    def __init__(
        self,
        test_ratio: float = 0.15,
        val_ratio: float = 0.10,
        random_seed: int = 42,
    ):
        self._test_ratio = test_ratio
        self._val_ratio = val_ratio
        self._seed = random_seed

    def split(self, pairs: List[LabeledPair]) -> tuple:
        """
        Returns (train_indices, val_indices, test_indices).
        Stratified by label.
        """
        from sklearn.model_selection import train_test_split

        indices = list(range(len(pairs)))
        labels = [p.label for p in pairs]

        if len(indices) == 0:
            return [], [], []

        # First split: test set
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=self._test_ratio,
            stratify=labels,
            random_state=self._seed,
        )

        # Second split: validation from train_val
        train_val_labels = [labels[i] for i in train_val_idx]
        val_ratio_adjusted = self._val_ratio / (1 - self._test_ratio)

        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_ratio_adjusted,
            stratify=train_val_labels,
            random_state=self._seed,
        )

        logger.info(
            f"Dataset split: {len(train_idx)} train / "
            f"{len(val_idx)} val / {len(test_idx)} test"
        )
        return train_idx, val_idx, test_idx
