"""
model-training-pipeline/src/pipeline/stages/feature_extraction.py

Stage 2: Extract features from Feature Store for all training pairs.

Uses the Feature Store's offline API for point-in-time correct features,
ensuring ZERO training-serving skew. The same 50-dim vector used at
serving time is used at training time.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.config import settings
from src.core.exceptions import FeatureCoverageError
from src.domain.models import LabeledPair, TrainingDataset, TrainingRun

logger = logging.getLogger(__name__)


class FeatureExtractionStage:
    """
    Retrieves 50-dim feature vectors from the Feature Store for all training pairs.
    Critical: uses point-in-time retrieval to prevent leakage.
    """

    # Chunking is handled by FeatureStoreClient using FEATURE_STORE_BATCH_SIZE,
    # which matches the Feature Store's documented per-request cap.
    FEATURE_BATCH_SIZE = settings.FEATURE_STORE_BATCH_SIZE

    def __init__(self, feature_store_client):
        self._client = feature_store_client

    async def execute(
        self,
        run: TrainingRun,
        pairs: List[LabeledPair],
        as_of_timestamp: Optional[datetime] = None,
    ) -> TrainingDataset:
        """
        Build the full feature matrix (N × 50) for all training pairs.
        Returns TrainingDataset with feature_matrix and labels populated.
        """
        logger.info(
            f"[Run {run.run_id}] Stage 2: Feature Extraction for {len(pairs)} pairs"
        )
        start = time.perf_counter()
        as_of = as_of_timestamp or datetime.utcnow()

        # Batch requests to Feature Store
        feature_rows = await self._fetch_features_batched(pairs, as_of)

        # Build numpy arrays
        feature_matrix, labels, valid_pairs = self._build_arrays(pairs, feature_rows)

        # Coverage gate: refuse to "succeed" on a dataset the Feature Store could not
        # serve. Without this, an unreachable Feature Store yields an empty dataset,
        # dummy models and f1=0.0 reported as a completed run.
        coverage = len(valid_pairs) / len(pairs) if pairs else 0.0
        if coverage < settings.FEATURE_COVERAGE_MIN_RATIO:
            raise FeatureCoverageError(
                requested=len(pairs),
                retrieved=len(valid_pairs),
                required_ratio=settings.FEATURE_COVERAGE_MIN_RATIO,
            )
        logger.info(
            f"[Run {run.run_id}] Feature coverage {coverage:.1%} "
            f"({len(valid_pairs)}/{len(pairs)} pairs)"
        )

        # Build text representations for Transformer
        texts_1, texts_2 = self._build_text_representations(valid_pairs)

        elapsed = time.perf_counter() - start
        logger.info(
            f"[Run {run.run_id}] Feature extraction complete: "
            f"{len(valid_pairs)}/{len(pairs)} pairs extracted in {elapsed:.1f}s"
        )

        # The Feature Store declares the column order in its offline response; keep it
        # with the matrix so importance and later consumers use real feature names.
        feature_names = getattr(self._client, "last_feature_names", None)
        if feature_names is not None and len(feature_names) != feature_matrix.shape[1]:
            raise ValueError(
                f"Feature Store declared {len(feature_names)} feature names but the "
                f"matrix has {feature_matrix.shape[1]} columns"
            )

        dataset = TrainingDataset(
            run_id=run.run_id,
            pairs=valid_pairs,
            feature_matrix=feature_matrix,
            labels=labels,
            entity_texts_1=texts_1,
            entity_texts_2=texts_2,
            feature_names=list(feature_names) if feature_names else None,
        )
        return dataset

    async def _fetch_features_batched(
        self,
        pairs: List[LabeledPair],
        as_of: datetime,
    ) -> Dict[Tuple[str, str], Optional[List[float]]]:
        """
        Fetch features from the Feature Store in batches, point-in-time as of `as_of`.

        Pairs are grouped BY TENANT before batching: the Feature Store resolves every
        request under a single tenant (the request's tenant header), so a batch that
        mixed tenants was answered only for the first pair's tenant. With the synthetic
        dev set (pairs arrive shuffled across 2 tenants) that silently lost 48.8% of the
        pairs. Results are keyed by (tenant_id, "e1:e2") because entity ids are only
        unique within a tenant.
        """
        results: Dict[Tuple[str, str], Optional[List[float]]] = {}

        by_tenant: Dict[str, List[LabeledPair]] = {}
        for p in pairs:
            by_tenant.setdefault(p.tenant_id, []).append(p)

        done = 0
        for tenant_id in sorted(by_tenant):
            tenant_pairs = by_tenant[tenant_id]
            for i in range(0, len(tenant_pairs), self.FEATURE_BATCH_SIZE):
                batch = tenant_pairs[i:i + self.FEATURE_BATCH_SIZE]

                try:
                    response = await self._client.get_offline_features(
                        entity_pairs=[(p.entity_id_1, p.entity_id_2) for p in batch],
                        tenant_id=tenant_id,
                        as_of_timestamp=as_of,
                    )
                    for pair_key, feature_vec in response.items():
                        results[(tenant_id, pair_key)] = feature_vec

                except Exception as e:
                    logger.error(
                        f"Feature Store batch request failed for tenant {tenant_id}: "
                        f"{type(e).__name__}: {e}"
                    )
                    # Mark batch as missing — the coverage gate decides what that means.
                    for p in batch:
                        results[(tenant_id, f"{p.entity_id_1}:{p.entity_id_2}")] = None

                done += len(batch)
                if (done // self.FEATURE_BATCH_SIZE) % 10 == 0:
                    logger.info(f"  Extracted features for {done}/{len(pairs)} pairs...")

        return results

    def _build_arrays(
        self,
        pairs: List[LabeledPair],
        feature_rows: Dict[Tuple[str, str], Optional[List[float]]],
    ) -> Tuple["np.ndarray", "np.ndarray", List[LabeledPair]]:
        """Build (feature_matrix, labels) numpy arrays, filtering missing features."""
        feature_vecs = []
        label_vecs = []
        valid_pairs = []

        for pair in pairs:
            key = f"{pair.entity_id_1}:{pair.entity_id_2}"
            alt_key = f"{pair.entity_id_2}:{pair.entity_id_1}"

            features = (
                feature_rows.get((pair.tenant_id, key))
                or feature_rows.get((pair.tenant_id, alt_key))
            )

            if features is not None and len(features) == 50:
                feature_vecs.append(features)
                label_vecs.append(pair.label)
                valid_pairs.append(pair)
            else:
                # Dropped, NOT zero-filled: a fabricated vector would silently
                # corrupt training. Coverage is checked by the caller.
                logger.debug(f"Feature miss for pair {key} — pair dropped")

        if not feature_vecs:
            return np.zeros((0, 50), dtype=np.float32), np.zeros(0, dtype=np.int32), []

        return (
            np.array(feature_vecs, dtype=np.float32),
            np.array(label_vecs, dtype=np.int32),
            valid_pairs,
        )

    def _build_text_representations(
        self,
        pairs: List[LabeledPair],
    ) -> Tuple[List[str], List[str]]:
        """
        Build text representations for Transformer training.
        Format: "{name} [SEP] {email} [SEP] {phone} [SEP] {address}"
        """
        texts_1 = []
        texts_2 = []

        for pair in pairs:
            # In production: fetch entity field values from MDM entity store
            # For now: use entity_id as placeholder
            texts_1.append(pair.entity_id_1)
            texts_2.append(pair.entity_id_2)

        return texts_1, texts_2
