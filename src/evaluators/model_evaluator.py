"""
model-training-pipeline/src/evaluators/model_evaluator.py

Comprehensive model evaluation:
  - Precision, Recall, F1, AUC-ROC
  - Per-threshold analysis (precision-recall curve)
  - SHAP feature importance
  - Statistical significance vs champion
  - Promotion criteria enforcement
  - ONNX export for Triton deployment
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import shap
from scipy.stats import mannwhitneyu
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.core.config import settings
from src.domain.models import ModelEvaluation, TrainedModel, TrainingDataset
from src.trainers.ensemble_trainer import EnsembleTrainer

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluation of trained ensemble model.
    Compares against champion model for promotion decision.
    """

    def __init__(self, ensemble_trainer: EnsembleTrainer):
        self._ensemble = ensemble_trainer

    def evaluate(
        self,
        models: Dict,
        dataset: TrainingDataset,
        model_id,
        run_id,
        champion_f1: Optional[float] = None,
        champion_predictions: Optional[np.ndarray] = None,
    ) -> ModelEvaluation:
        """
        Full evaluation on test set.
        Returns ModelEvaluation with all metrics + promotion decision.
        """
        logger.info("ModelEvaluator: running comprehensive evaluation")
        start = time.time()

        test_idx = dataset.test_indices or list(range(int(len(dataset.pairs) * 0.85), len(dataset.pairs)))

        if len(test_idx) == 0:
            logger.warning("Empty test set — using validation set for evaluation")
            test_idx = dataset.val_indices or list(range(int(len(dataset.pairs) * 0.75), len(dataset.pairs)))

        y_true = dataset.labels[test_idx] if dataset.labels is not None else np.zeros(len(test_idx))

        # Get ensemble predictions
        y_proba = self._ensemble.predict_ensemble(models, dataset, test_idx)
        y_pred = (y_proba >= 0.5).astype(int)

        # Core metrics
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        try:
            auc = float(roc_auc_score(y_true, y_proba))
            ap = float(average_precision_score(y_true, y_proba))
        except ValueError:
            auc, ap = 0.0, 0.0

        cm = confusion_matrix(y_true, y_pred).tolist()

        # Per-threshold analysis
        prec_curve, rec_curve, thresh_curve = precision_recall_curve(y_true, y_proba)

        # Statistical significance vs champion
        p_value = None
        is_significant = False
        if champion_predictions is not None and len(champion_predictions) == len(y_proba):
            try:
                _, p_value = mannwhitneyu(y_proba, champion_predictions, alternative="greater")
                is_significant = bool(p_value < settings.PROMOTION_PVALUE)
            except Exception:
                pass

        # Feature importance (XGBoost SHAP)
        feature_importance = {}
        try:
            feature_importance = self._compute_shap_importance(
                models["xgb"]["model"],
                dataset.feature_matrix[test_idx] if dataset.feature_matrix is not None else None,
                dataset.feature_names,
            )
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")

        # Log to MLflow
        mlflow.log_metrics({
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "test_auc_roc": auc,
            "test_avg_precision": ap,
        })

        elapsed = time.time() - start
        logger.info(
            f"Evaluation complete in {elapsed:.1f}s: "
            f"P={precision:.4f} R={recall:.4f} F1={f1:.4f} AUC={auc:.4f}"
        )

        # Bias check — ensure model doesn't degrade on specific tenants
        bias_passes = True
        bias_f1_per_tenant = {}
        tenant_ids = [dataset.pairs[i].tenant_id for i in test_idx]
        if tenant_ids and any(tid for tid in tenant_ids):
            bias_passes, bias_f1_per_tenant = self._check_bias(
                y_true, y_pred, tenant_ids
            )
            if not bias_passes:
                logger.warning(
                    "Bias check FAILED: F1 variance across tenants is too high. "
                    "Per-tenant F1: %s", bias_f1_per_tenant
                )
            mlflow.log_metric("bias_check_passed", 1.0 if bias_passes else 0.0)

        evaluation = ModelEvaluation(
            model_id=model_id,
            run_id=run_id,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc,
            average_precision=ap,
            confusion_matrix=cm,
            n_test_samples=len(test_idx),
            n_positive=int(y_true.sum()),
            n_negative=int(len(y_true) - y_true.sum()),
            thresholds=thresh_curve.tolist(),
            precision_at_threshold=prec_curve.tolist(),
            recall_at_threshold=rec_curve.tolist(),
            feature_importance=feature_importance,
            champion_f1=champion_f1,
            p_value=p_value,
            is_significantly_better=is_significant,
            bias_passes=bias_passes,
            bias_f1_per_tenant=bias_f1_per_tenant,
        )

        return evaluation

    def _compute_shap_importance(
        self,
        xgb_model,
        X_test: Optional[np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute SHAP feature importance from XGBoost model."""
        if X_test is None or len(X_test) == 0:
            return {}

        # Sample up to 1000 rows for efficiency
        sample_size = min(len(X_test), 1000)
        X_sample = X_test[:sample_size]

        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_sample)

        # Mean absolute SHAP value per feature
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class

        importance = np.abs(shap_values).mean(axis=0)
        if feature_names is not None and len(feature_names) == len(importance):
            names = list(feature_names)
        else:
            # Only when the feature source declared no names (e.g. unit-test data).
            logger.warning("Feature names unavailable; SHAP importance keyed by column index")
            names = [f"column_{i}" for i in range(len(importance))]

        return dict(zip(names, importance.tolist()))

    def _check_bias(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        tenant_ids: List[str],
        max_f1_variance: float = 0.10,
        min_samples_per_tenant: int = 50,
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check for bias across tenant groups (LLD PART VI §6.2).
        Ensures the model doesn't degrade on protected attributes.

        Returns (passes_bias_check, f1_per_tenant_dict).
        Fails if the F1 variance between any two tenants exceeds max_f1_variance.
        """
        tenant_data = defaultdict(lambda: {"true": [], "pred": []})

        for yt, yp, tid in zip(y_true, y_pred, tenant_ids):
            if tid:  # Skip None/empty tenant IDs
                tenant_data[tid]["true"].append(int(yt))
                tenant_data[tid]["pred"].append(int(yp))

        f1_per_tenant = {}
        for tid, data in tenant_data.items():
            if len(data["true"]) >= min_samples_per_tenant:
                tenant_f1 = float(f1_score(data["true"], data["pred"], zero_division=0))
                f1_per_tenant[tid] = round(tenant_f1, 4)

        if len(f1_per_tenant) < 2:
            # Not enough tenants with sufficient data to check bias
            return True, f1_per_tenant

        min_f1 = min(f1_per_tenant.values())
        max_f1 = max(f1_per_tenant.values())
        variance = max_f1 - min_f1

        passes = variance < max_f1_variance
        logger.info(
            "Bias check: %d tenants evaluated, F1 range=[%.4f, %.4f], "
            "variance=%.4f (threshold=%.2f) → %s",
            len(f1_per_tenant), min_f1, max_f1, variance,
            max_f1_variance, "PASS" if passes else "FAIL",
        )
        return passes, f1_per_tenant


    def export_to_onnx(
        self,
        models: Dict,
        dataset: TrainingDataset,
        output_path: str,
    ) -> str:
        """
        Export the XGBoost component to ONNX format for Triton deployment.
        Returns path to ONNX file.
        """
        import onnxmltools
        from onnxmltools.convert import convert_xgboost
        from skl2onnx.common.data_types import FloatTensorType

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Convert XGBoost to ONNX
        initial_type = [("float_input", FloatTensorType([None, 50]))]
        onnx_model = convert_xgboost(
            models["xgb"]["model"],
            initial_types=initial_type,
        )

        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        logger.info(f"ONNX model exported to {output_path}")
        return output_path

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        target_precision: float = 0.95,
    ) -> float:
        """
        Find classification threshold that achieves target precision
        while maximizing recall.
        """
        prec, rec, thresh = precision_recall_curve(y_true, y_proba)
        # Find thresholds where precision >= target
        valid = prec[:-1] >= target_precision
        if not valid.any():
            return 0.5
        best_idx = np.argmax(rec[:-1][valid])
        return float(thresh[valid][best_idx])
