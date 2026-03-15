"""
model-training-pipeline/tests/unit/test_pipeline.py
Unit tests for training pipeline components.
"""
from __future__ import annotations

import numpy as np
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.domain.models import (
    CanaryMetrics, LabeledPair, ModelEvaluation,
    RetrainingTrigger, RunStatus, TrainingDataset, TrainingRun
)
from src.pipeline.stages.data_collection import DataCollectionStage, TrainingDataSplitter


# ─── Training Data Models ─────────────────────────────────────────────────────

class TestLabeledPair:
    def test_creation(self):
        pair = LabeledPair(
            entity_id_1="e1", entity_id_2="e2",
            tenant_id="t1", label=1,
            confidence=0.95, source="hitl_merge",
        )
        assert pair.label == 1
        assert pair.confidence == 0.95

    def test_defaults(self):
        pair = LabeledPair(
            entity_id_1="e1", entity_id_2="e2",
            tenant_id="t1", label=0,
            confidence=0.8, source="synthetic",
        )
        assert pair.id is not None
        assert isinstance(pair.labeled_at, datetime)


class TestTrainingRun:
    def test_initial_state(self):
        run = TrainingRun(trigger=RetrainingTrigger.MANUAL)
        assert run.status == RunStatus.PENDING
        assert run.n_training_pairs == 0
        assert run.ensemble_f1 is None

    def test_duration(self):
        run = TrainingRun(trigger=RetrainingTrigger.SCHEDULED)
        run.started_at = datetime(2024, 1, 1, 10, 0, 0)
        end = datetime(2024, 1, 1, 12, 30, 0)
        assert run.duration_seconds(end) == 9000.0

    def test_duration_none_when_not_started(self):
        run = TrainingRun(trigger=RetrainingTrigger.MANUAL)
        assert run.duration_seconds() is None


class TestTrainingDataset:
    def _make_pairs(self, n: int) -> list:
        pairs = []
        for i in range(n):
            pairs.append(LabeledPair(
                entity_id_1=f"e{i}", entity_id_2=f"e{i+100}",
                tenant_id="t1", label=i % 3 == 0,
                confidence=0.9, source="hitl",
            ))
        return pairs

    def test_stats(self):
        pairs = self._make_pairs(12)
        dataset = TrainingDataset(run_id=uuid4(), pairs=pairs)
        assert dataset.n_samples == 12
        assert dataset.n_positive == 4   # every 3rd
        assert dataset.n_negative == 8
        assert abs(dataset.positive_rate - 4/12) < 0.01


# ─── Data Splitter ────────────────────────────────────────────────────────────

class TestTrainingDataSplitter:
    def _make_pairs(self, n_pos: int, n_neg: int) -> list:
        pairs = []
        for i in range(n_pos):
            pairs.append(LabeledPair(f"p{i}", f"p{i+1000}", "t1", 1, 0.9, "hitl"))
        for i in range(n_neg):
            pairs.append(LabeledPair(f"n{i}", f"n{i+1000}", "t1", 0, 0.9, "synthetic"))
        return pairs

    def test_split_sizes(self):
        splitter = TrainingDataSplitter(test_ratio=0.15, val_ratio=0.10)
        pairs = self._make_pairs(200, 600)
        train_idx, val_idx, test_idx = splitter.split(pairs)

        total = len(train_idx) + len(val_idx) + len(test_idx)
        assert total == len(pairs)
        assert len(test_idx) == pytest.approx(len(pairs) * 0.15, abs=5)
        assert len(val_idx) == pytest.approx(len(pairs) * 0.10, abs=5)

    def test_no_overlap(self):
        splitter = TrainingDataSplitter(test_ratio=0.15, val_ratio=0.10)
        pairs = self._make_pairs(100, 300)
        train_idx, val_idx, test_idx = splitter.split(pairs)

        all_sets = [set(train_idx), set(val_idx), set(test_idx)]
        assert set(train_idx) & set(val_idx) == set()
        assert set(train_idx) & set(test_idx) == set()
        assert set(val_idx) & set(test_idx) == set()

    def test_empty_dataset(self):
        splitter = TrainingDataSplitter()
        train_idx, val_idx, test_idx = splitter.split([])
        assert train_idx == []
        assert val_idx == []
        assert test_idx == []


# ─── Canary Metrics ──────────────────────────────────────────────────────────

class TestCanaryMetrics:
    def test_error_rate(self):
        metrics = CanaryMetrics(model_id=uuid4(), inference_count=1000, error_count=5)
        assert metrics.error_rate == 0.005

    def test_error_rate_zero_inferences(self):
        metrics = CanaryMetrics(model_id=uuid4())
        assert metrics.error_rate == 0.0

    def test_should_rollback_f1_drop(self):
        metrics = CanaryMetrics(
            model_id=uuid4(),
            inference_count=5000,
            rolling_f1=0.88,
        )
        should, reason = metrics.should_rollback(champion_f1=0.94, champion_p95_ms=50.0)
        assert should is True
        assert "F1" in reason

    def test_should_rollback_latency_spike(self):
        metrics = CanaryMetrics(
            model_id=uuid4(),
            inference_count=5000,
            p95_latency_ms=150.0,
        )
        should, reason = metrics.should_rollback(champion_f1=0.94, champion_p95_ms=50.0)
        assert should is True
        assert "latency" in reason.lower()

    def test_should_rollback_error_rate(self):
        metrics = CanaryMetrics(
            model_id=uuid4(),
            inference_count=1000,
            error_count=15,  # 1.5%
        )
        should, reason = metrics.should_rollback(champion_f1=0.94, champion_p95_ms=50.0)
        assert should is True
        assert "Error rate" in reason

    def test_healthy_canary_no_rollback(self):
        metrics = CanaryMetrics(
            model_id=uuid4(),
            inference_count=5000,
            error_count=2,
            p95_latency_ms=45.0,
            rolling_f1=0.95,
        )
        should, reason = metrics.should_rollback(champion_f1=0.94, champion_p95_ms=50.0)
        assert should is False
        assert reason == ""


# ─── Model Evaluation ────────────────────────────────────────────────────────

class TestModelEvaluation:
    def _make_eval(self, f1: float, champion_f1: float = None) -> ModelEvaluation:
        return ModelEvaluation(
            model_id=uuid4(),
            run_id=uuid4(),
            precision=0.95,
            recall=0.93,
            f1_score=f1,
            auc_roc=0.97,
            average_precision=0.96,
            confusion_matrix=[[8000, 200], [150, 1650]],
            champion_f1=champion_f1,
        )

    def test_passes_promotion_f1_threshold(self):
        eval = self._make_eval(f1=0.93, champion_f1=0.92)
        assert eval.passes_promotion_criteria is True

    def test_fails_promotion_below_minimum(self):
        eval = self._make_eval(f1=0.88)
        assert eval.passes_promotion_criteria is False

    def test_fails_promotion_no_improvement_over_champion(self):
        eval = self._make_eval(f1=0.93, champion_f1=0.93)  # not beating by 0.5%
        assert eval.passes_promotion_criteria is False

    def test_passes_promotion_no_champion(self):
        eval = self._make_eval(f1=0.92, champion_f1=None)
        assert eval.passes_promotion_criteria is True
