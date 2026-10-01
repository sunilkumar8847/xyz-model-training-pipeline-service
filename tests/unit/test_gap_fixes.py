"""
tests/unit/test_gap_fixes.py

Unit tests for all 6 gap fixes implemented:
  Fix 1: Transformer model saved to MLflow
  Fix 2: Structured API error envelopes
  Fix 3: Health endpoint split (tested via API in Level 2)
  Fix 4: Stage retry logic
  Fix 5: Bias check in model evaluation
  Fix 6: Drift detection (label + prediction)
"""
from __future__ import annotations

import asyncio
import math
import numpy as np
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.domain.models import (
    LabeledPair, ModelEvaluation, RetrainingTrigger,
    RunStatus, TrainingDataset, TrainingRun,
)

# ── Check if heavy ML deps are available ─────────────────────────────────────
try:
    import mlflow
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False

requires_ml = pytest.mark.skipif(not HAS_ML_DEPS, reason="Requires mlflow/torch/transformers (~3GB)")


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 1: Transformer model saved to MLflow
# ═══════════════════════════════════════════════════════════════════════════════

@requires_ml
class TestFix1TransformerModelSave:
    """Verify that the ensemble trainer returns a tokenizer alongside the model."""

    def test_ensemble_return_dict_has_tokenizer_key(self, monkeypatch):
        """
        The EnsembleTrainer.train_all() return dict must include
        models["transformer"]["tokenizer"] for MLflow to save it.

        Pins the full ensemble: this asserts transformer behaviour, so it must not
        depend on the machine's ENABLED_TRAINERS (the local profile trains XGBoost
        only). Partial-ensemble behaviour is covered in TestEnabledTrainers.
        """
        from src.core.config import settings
        from src.trainers.ensemble_trainer import EnsembleTrainer

        monkeypatch.setattr(settings, "ENABLED_TRAINERS", "transformer,gnn,xgboost")

        with (
            patch.object(EnsembleTrainer, "__init__", lambda self: None),
            patch("src.trainers.ensemble_trainer.TransformerTrainer") as MockTT,
            patch("src.trainers.ensemble_trainer.GNNTrainer") as MockGNN,
            patch("src.trainers.ensemble_trainer.XGBoostTrainer") as MockXGB,
            patch("src.trainers.ensemble_trainer.AutoTokenizer") as MockTokenizer,
        ):
            trainer = EnsembleTrainer.__new__(EnsembleTrainer)
            trainer._transformer_trainer = MockTT.return_value
            trainer._gnn_trainer = MockGNN.return_value
            trainer._xgb_trainer = MockXGB.return_value

            trainer._transformer_trainer.train.return_value = (MagicMock(), 0.92)
            trainer._gnn_trainer.train.return_value = (MagicMock(), 0.90)
            trainer._xgb_trainer.train.return_value = (MagicMock(), 0.88)

            mock_tokenizer = MagicMock()
            MockTokenizer.from_pretrained.return_value = mock_tokenizer

            dataset = MagicMock(spec=TrainingDataset)
            dataset.feature_matrix = None

            result = trainer.train_all(dataset=dataset, mlflow_run_id=None, tune_weights=False)

            assert "transformer" in result
            assert "tokenizer" in result["transformer"], (
                "Fix 1 FAILED: train_all() must return tokenizer in models['transformer']"
            )
            assert result["transformer"]["tokenizer"] is mock_tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 2: Structured API error envelopes  (NO ML DEPS NEEDED)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFix2StructuredErrors:
    """Verify custom exception classes have correct error codes."""

    def test_invalid_training_config_error(self):
        from src.core.exceptions import InvalidTrainingConfigError
        exc = InvalidTrainingConfigError("bad input")
        assert exc.code == "MT_1001"
        assert exc.status_code == 422
        assert "bad input" in exc.message

    def test_evaluation_gate_failed_error(self):
        from src.core.exceptions import EvaluationGateFailedError
        exc = EvaluationGateFailedError(f1=0.89, gate=0.94)
        assert exc.code == "MT_4001"
        assert exc.status_code == 422
        assert "0.89" in exc.message
        assert "0.94" in exc.message

    def test_infrastructure_unavailable_error(self):
        from src.core.exceptions import InfrastructureUnavailableError
        exc = InfrastructureUnavailableError("MLflow", "connection refused")
        assert exc.code == "MT_9001"
        assert exc.status_code == 503
        assert "MLflow" in exc.message
        assert "connection refused" in exc.message

    def test_resource_not_found_error(self):
        from src.core.exceptions import ResourceNotFoundError
        exc = ResourceNotFoundError("Training run", "abc-123")
        assert exc.code == "MT_4040"
        assert exc.status_code == 404
        assert "abc-123" in exc.message

    def test_all_exceptions_inherit_from_base(self):
        from src.core.exceptions import (
            TrainingPipelineError, InvalidTrainingConfigError,
            EvaluationGateFailedError, InfrastructureUnavailableError,
            ResourceNotFoundError,
        )
        assert issubclass(InvalidTrainingConfigError, TrainingPipelineError)
        assert issubclass(EvaluationGateFailedError, TrainingPipelineError)
        assert issubclass(InfrastructureUnavailableError, TrainingPipelineError)
        assert issubclass(ResourceNotFoundError, TrainingPipelineError)

    def test_exceptions_are_catchable_as_base(self):
        from src.core.exceptions import TrainingPipelineError, InvalidTrainingConfigError
        try:
            raise InvalidTrainingConfigError("test")
        except TrainingPipelineError as e:
            assert e.code == "MT_1001"


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 4: Stage retry logic
# ═══════════════════════════════════════════════════════════════════════════════

@requires_ml
class TestFix4StageRetryLogic:
    """Verify the _run_stage_with_retry helper."""

    def _make_pipeline(self):
        from src.pipeline.training_pipeline import TrainingPipeline
        mock_db = MagicMock()
        return TrainingPipeline(db_session=mock_db)

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self):
        pipeline = self._make_pipeline()
        func = AsyncMock(return_value="success")
        result = await pipeline._run_stage_with_retry("data_collection", func)
        assert result == "success"
        assert func.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure_then_succeeds(self):
        pipeline = self._make_pipeline()
        func = AsyncMock(side_effect=[ValueError("transient"), "recovered"])

        with patch("src.pipeline.training_pipeline.asyncio.sleep", new_callable=AsyncMock):
            result = await pipeline._run_stage_with_retry("data_collection", func)

        assert result == "recovered"
        assert func.call_count == 2

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries(self):
        pipeline = self._make_pipeline()
        func = AsyncMock(side_effect=ValueError("permanent failure"))

        with patch("src.pipeline.training_pipeline.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ValueError, match="permanent failure"):
                await pipeline._run_stage_with_retry("data_collection", func)

        assert func.call_count == 3  # 3 retries for data_collection

    def test_retry_count_matches_lld_per_stage(self):
        """Verify different stages have different retry limits (from LLD Table 5)."""
        from src.pipeline.training_pipeline import STAGE_RETRY_COUNTS

        assert STAGE_RETRY_COUNTS["data_collection"] == 3
        assert STAGE_RETRY_COUNTS["feature_extraction"] == 3
        assert STAGE_RETRY_COUNTS["train_transformer"] == 2
        assert STAGE_RETRY_COUNTS["train_gnn"] == 2
        assert STAGE_RETRY_COUNTS["train_xgboost"] == 3
        assert STAGE_RETRY_COUNTS["ensemble"] == 2
        assert STAGE_RETRY_COUNTS["evaluation"] == 2
        assert STAGE_RETRY_COUNTS["registration"] == 3

    @pytest.mark.asyncio
    async def test_handles_sync_functions(self):
        """Retry helper should work with both sync and async functions."""
        pipeline = self._make_pipeline()

        def sync_func():
            return "sync_result"

        result = await pipeline._run_stage_with_retry("evaluation", sync_func)
        assert result == "sync_result"


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 5: Bias check in model evaluation
# ═══════════════════════════════════════════════════════════════════════════════

class TestFix5BiasCheckPromotion:
    """Verify bias check integration with promotion criteria (NO ML DEPS)."""

    def test_promotion_fails_when_bias_check_fails(self):
        eval_result = ModelEvaluation(
            model_id=uuid4(), run_id=uuid4(),
            precision=0.95, recall=0.93, f1_score=0.96,
            auc_roc=0.97, average_precision=0.96,
            confusion_matrix=[[8000, 200], [150, 1650]],
            champion_f1=0.93,
            bias_passes=False,  # Fails bias!
        )
        assert eval_result.passes_promotion_criteria is False

    def test_promotion_passes_when_bias_check_passes(self):
        eval_result = ModelEvaluation(
            model_id=uuid4(), run_id=uuid4(),
            precision=0.95, recall=0.93, f1_score=0.96,
            auc_roc=0.97, average_precision=0.96,
            confusion_matrix=[[8000, 200], [150, 1650]],
            champion_f1=0.93,
            bias_passes=True,
        )
        assert eval_result.passes_promotion_criteria is True

    def test_bias_defaults_to_true(self):
        eval_result = ModelEvaluation(
            model_id=uuid4(), run_id=uuid4(),
            precision=0.95, recall=0.93, f1_score=0.92,
            auc_roc=0.97, average_precision=0.96,
            confusion_matrix=[[8000, 200], [150, 1650]],
        )
        assert eval_result.bias_passes is True
        assert eval_result.bias_f1_per_tenant == {}


@requires_ml
class TestFix5BiasCheckMethod:
    """Verify _check_bias() logic (requires mlflow for ModelEvaluator import)."""

    def test_bias_check_passes_when_variance_low(self):
        from src.evaluators.model_evaluator import ModelEvaluator
        evaluator = ModelEvaluator.__new__(ModelEvaluator)
        y_true = np.array([1, 0, 1, 0] * 30 + [1, 0, 1, 0] * 30)
        y_pred = np.array([1, 0, 1, 0] * 30 + [1, 0, 1, 0] * 30)
        tenant_ids = ["tenant_A"] * 120 + ["tenant_B"] * 120

        passes, f1_per_tenant = evaluator._check_bias(
            y_true, y_pred, tenant_ids, min_samples_per_tenant=50
        )
        assert passes is True
        assert "tenant_A" in f1_per_tenant
        assert "tenant_B" in f1_per_tenant

    def test_bias_check_fails_when_variance_high(self):
        from src.evaluators.model_evaluator import ModelEvaluator
        evaluator = ModelEvaluator.__new__(ModelEvaluator)

        y_true_a = np.array([1, 0] * 50)
        y_pred_a = np.array([1, 0] * 50)  # Perfect
        y_true_b = np.array([1, 0] * 50)
        y_pred_b = np.array([0, 0] * 50)  # Terrible

        y_true = np.concatenate([y_true_a, y_true_b])
        y_pred = np.concatenate([y_pred_a, y_pred_b])
        tenant_ids = ["tenant_A"] * 100 + ["tenant_B"] * 100

        passes, _ = evaluator._check_bias(
            y_true, y_pred, tenant_ids, min_samples_per_tenant=50
        )
        assert passes is False

    def test_bias_check_passes_with_single_tenant(self):
        from src.evaluators.model_evaluator import ModelEvaluator
        evaluator = ModelEvaluator.__new__(ModelEvaluator)
        y_true = np.array([1, 0] * 50)
        y_pred = np.array([1, 0] * 50)
        tenant_ids = ["only_tenant"] * 100

        passes, _ = evaluator._check_bias(
            y_true, y_pred, tenant_ids, min_samples_per_tenant=50
        )
        assert passes is True


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 6: Drift detection (label + prediction)
# ═══════════════════════════════════════════════════════════════════════════════

@requires_ml
class TestFix6DriftDetection:
    """Verify drift detection thresholds and methods exist."""

    def test_thresholds_defined(self):
        from src.workers.drift_detector import (
            LABEL_DRIFT_OVERRIDE_THRESHOLD,
            PREDICTION_DRIFT_JS_THRESHOLD,
        )
        assert LABEL_DRIFT_OVERRIDE_THRESHOLD == 0.15
        assert PREDICTION_DRIFT_JS_THRESHOLD == 0.05

    def test_drift_detector_has_label_drift_method(self):
        from src.workers.drift_detector import DriftDetector
        detector = DriftDetector()
        assert hasattr(detector, "_check_label_drift")
        assert asyncio.iscoroutinefunction(detector._check_label_drift)

    def test_drift_detector_has_prediction_drift_method(self):
        from src.workers.drift_detector import DriftDetector
        detector = DriftDetector()
        assert hasattr(detector, "_check_prediction_drift")
        assert asyncio.iscoroutinefunction(detector._check_prediction_drift)

    @pytest.mark.asyncio
    async def test_label_drift_returns_none_when_service_unavailable(self):
        from src.workers.drift_detector import DriftDetector
        detector = DriftDetector()
        detector._http = MagicMock()
        detector._http.get = AsyncMock(side_effect=Exception("connection refused"))
        result = await detector._check_label_drift()
        assert result is None

    @pytest.mark.asyncio
    async def test_prediction_drift_returns_none_when_service_unavailable(self):
        from src.workers.drift_detector import DriftDetector
        detector = DriftDetector()
        detector._http = MagicMock()
        detector._http.get = AsyncMock(side_effect=Exception("connection refused"))
        result = await detector._check_prediction_drift()
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# Prometheus Metrics  (NO ML DEPS NEEDED)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPrometheusMetrics:
    """Verify new Prometheus metrics are registered."""

    def test_stage_retries_counter_exists(self):
        from src.core.metrics import STAGE_RETRIES
        assert STAGE_RETRIES is not None
        STAGE_RETRIES.labels(stage="test_stage")

    def test_drift_score_gauge_accepts_new_labels(self):
        from src.core.metrics import DRIFT_SCORE
        DRIFT_SCORE.labels(drift_type="label")
        DRIFT_SCORE.labels(drift_type="prediction")
