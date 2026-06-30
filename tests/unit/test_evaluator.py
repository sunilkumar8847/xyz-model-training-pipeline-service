"""Unit tests for ModelEvaluator."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4

from src.domain.models import ModelEvaluation, TrainingDataset, LabeledPair


class TestModelEvaluationPromotion:
    """Tests for ModelEvaluation.passes_promotion_criteria."""

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

    def test_passes_when_f1_above_minimum_no_champion(self):
        assert self._make_eval(0.92).passes_promotion_criteria is True

    def test_fails_when_f1_below_minimum(self):
        assert self._make_eval(0.90).passes_promotion_criteria is False

    def test_fails_when_not_beating_champion_by_margin(self):
        # Must exceed champion by >= 0.5%
        assert self._make_eval(0.930, champion_f1=0.930).passes_promotion_criteria is False

    def test_passes_when_beating_champion_by_sufficient_margin(self):
        assert self._make_eval(0.936, champion_f1=0.930).passes_promotion_criteria is True

    def test_passes_when_no_champion_and_f1_at_minimum(self):
        assert self._make_eval(0.91).passes_promotion_criteria is True


class TestCanaryMetricsShouldRollback:
    """Tests for CanaryMetrics rollback decision logic."""

    def _make_metrics(self, **kwargs):
        from src.domain.models import CanaryMetrics
        return CanaryMetrics(model_id=uuid4(), **kwargs)

    def test_healthy_canary_no_rollback(self):
        m = self._make_metrics(
            inference_count=5000, error_count=2,
            p95_latency_ms=45.0, rolling_f1=0.95,
        )
        should, reason = m.should_rollback(0.94, 50.0)
        assert should is False

    def test_rollback_on_f1_drop(self):
        m = self._make_metrics(inference_count=5000, rolling_f1=0.88)
        should, reason = m.should_rollback(0.94, 50.0)
        assert should is True
        assert "F1" in reason

    def test_rollback_on_latency_spike(self):
        m = self._make_metrics(inference_count=5000, p95_latency_ms=150.0)
        should, reason = m.should_rollback(0.94, 50.0)
        assert should is True
        assert "latency" in reason.lower()

    def test_rollback_on_high_error_rate(self):
        m = self._make_metrics(inference_count=1000, error_count=15)
        should, reason = m.should_rollback(0.94, 50.0)
        assert should is True
        assert "Error rate" in reason

    def test_error_rate_zero_when_no_inferences(self):
        m = self._make_metrics()
        assert m.error_rate == 0.0


class TestTrainingRunDuration:
    """Tests for TrainingRun.duration_seconds."""

    def test_duration_seconds_computed(self):
        from src.domain.models import TrainingRun, RetrainingTrigger
        from datetime import datetime
        run = TrainingRun(trigger=RetrainingTrigger.MANUAL)
        run.started_at = datetime(2024, 1, 1, 10, 0, 0)
        end = datetime(2024, 1, 1, 12, 30, 0)
        assert run.duration_seconds(end) == 9000.0

    def test_duration_none_when_not_started(self):
        from src.domain.models import TrainingRun, RetrainingTrigger
        run = TrainingRun(trigger=RetrainingTrigger.MANUAL)
        assert run.duration_seconds() is None


class TestTrainingPipelineShouldAutoPromote:
    """Tests for TrainingPipeline._should_auto_promote."""

    def _make_eval(self, f1: float) -> ModelEvaluation:
        return ModelEvaluation(
            model_id=uuid4(),
            run_id=uuid4(),
            precision=0.95,
            recall=0.93,
            f1_score=f1,
            auc_roc=0.97,
            average_precision=0.96,
            confusion_matrix=[[8000, 200], [150, 1650]],
        )

    def _make_pipeline(self):
        from src.pipeline.training_pipeline import TrainingPipeline
        mock_db = MagicMock()
        return TrainingPipeline(db_session=mock_db)

    def test_auto_promote_true_when_f1_exceeds_champion(self):
        pipeline = self._make_pipeline()
        eval_result = self._make_eval(f1=0.960)
        assert pipeline._should_auto_promote(eval_result, champion_f1=0.940) is True

    def test_auto_promote_false_when_f1_below_threshold(self):
        pipeline = self._make_pipeline()
        eval_result = self._make_eval(f1=0.880)
        assert pipeline._should_auto_promote(eval_result, champion_f1=None) is False

    def test_auto_promote_false_when_not_beating_champion(self):
        pipeline = self._make_pipeline()
        eval_result = self._make_eval(f1=0.941)
        # Champion is 0.940, delta required is 0.005 — 0.941 < 0.940 + 0.005
        assert pipeline._should_auto_promote(eval_result, champion_f1=0.940) is False

    def test_auto_promote_true_when_no_champion(self):
        pipeline = self._make_pipeline()
        eval_result = self._make_eval(f1=0.950)
        assert pipeline._should_auto_promote(eval_result, champion_f1=None) is True
