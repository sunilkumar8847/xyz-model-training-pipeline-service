"""
model-training-pipeline/src/registry/onnx_export.py

Builds the SERVING artifact for the Model Inference Service: the trained XGBoost
component converted to ONNX for Triton's `onnxruntime_onnx` backend (the runtime the
SDS and the inference service's Triton design specify).

Every export is verified before it can be registered:
  * parity   — ONNX probabilities must match XGBoost.predict_proba on real rows
               (max |diff| <= PARITY_TOLERANCE); otherwise registration fails
  * contract — exactly N_FEATURES float inputs named "input", output "probabilities"
               shape [N, 2] with the match probability at index 1
  * probes   — a few real rows with their expected probabilities are recorded so the
               publisher and the inference service can re-verify the artifact later

Scope, stated plainly: this is the XGBoost COMPONENT. It equals "the model" only while
the ensemble is XGBoost-only (weights {xgboost: 1.0}). serving_signature.json records
which component it is; the Triton publisher refuses to publish it for a full ensemble.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

N_FEATURES = 50
INPUT_NAME = "input"
OUTPUT_NAME = "probabilities"
POSITIVE_CLASS_INDEX = 1
# Triton 23.10 bundles ONNX Runtime 1.16 (IR <= 9). The XGBoost converter emits IR 8
# with the ai.onnx.ml domain; this ceiling is asserted so a converter upgrade cannot
# silently produce a file the documented Triton release cannot load.
MAX_IR_VERSION = 9
PARITY_TOLERANCE = 1e-5
N_PROBES = 8
GRAPH_NAME = "xyz_mdm_matcher_xgboost"


class OnnxExportError(RuntimeError):
    """The ONNX serving artifact could not be produced or failed verification."""


@dataclass
class OnnxExport:
    onnx_bytes: bytes
    signature: Dict = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.onnx_bytes).hexdigest()


def feature_names_sha256(feature_names: List[str]) -> str:
    """Fingerprint of the ordered feature contract, shared by training and serving."""
    return hashlib.sha256("\n".join(feature_names).encode("utf-8")).hexdigest()


def export_xgboost_to_onnx(
    xgb_model,
    feature_names: List[str],
    verification_rows: np.ndarray,
    component: str = "xgboost",
) -> OnnxExport:
    """
    Convert a fitted XGBClassifier to ONNX and verify it. Raises OnnxExportError on
    any contract or parity violation — a model that cannot be served faithfully must
    not be registered as servable.
    """
    import onnxruntime as ort
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    if feature_names is None or len(feature_names) != N_FEATURES:
        raise OnnxExportError(
            f"feature_names must list exactly {N_FEATURES} names, got "
            f"{None if feature_names is None else len(feature_names)}"
        )
    if len(set(feature_names)) != N_FEATURES:
        raise OnnxExportError("feature_names contains duplicates")
    if getattr(xgb_model, "n_features_in_", None) != N_FEATURES:
        raise OnnxExportError(
            f"model expects {getattr(xgb_model, 'n_features_in_', None)} features, "
            f"not {N_FEATURES}"
        )
    X = np.asarray(verification_rows, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != N_FEATURES or len(X) == 0:
        raise OnnxExportError(f"verification_rows must be (n>0, {N_FEATURES}), got {X.shape}")

    try:
        onnx_model = convert_xgboost(
            xgb_model,
            initial_types=[(INPUT_NAME, FloatTensorType([None, N_FEATURES]))],
        )
    except Exception as exc:
        raise OnnxExportError(f"XGBoost -> ONNX conversion failed: {exc}") from exc

    # onnxmltools names the graph with a fresh random UUID on every conversion, so two
    # exports of the same model differed byte-for-byte. A fixed name makes the artifact
    # (and its sha256) reproducible from the same trained model.
    onnx_model.graph.name = GRAPH_NAME

    if onnx_model.ir_version > MAX_IR_VERSION:
        raise OnnxExportError(
            f"ONNX IR version {onnx_model.ir_version} exceeds {MAX_IR_VERSION}, the "
            f"maximum the documented Triton release (23.10) can load"
        )
    onnx_bytes = onnx_model.SerializeToString()

    sess = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    outputs = {o.name: o for o in sess.get_outputs()}
    if [i.name for i in inputs] != [INPUT_NAME] or inputs[0].shape[1] != N_FEATURES:
        raise OnnxExportError(f"unexpected ONNX inputs {[(i.name, i.shape) for i in inputs]}")
    if OUTPUT_NAME not in outputs:
        raise OnnxExportError(f"ONNX model has no '{OUTPUT_NAME}' output: {list(outputs)}")

    onnx_proba = sess.run([OUTPUT_NAME], {INPUT_NAME: X})[0][:, POSITIVE_CLASS_INDEX]
    ref_proba = xgb_model.predict_proba(X)[:, POSITIVE_CLASS_INDEX]
    max_diff = float(np.abs(onnx_proba - ref_proba).max())
    if not max_diff <= PARITY_TOLERANCE:
        raise OnnxExportError(
            f"ONNX/XGBoost parity failed: max |diff| = {max_diff:.3e} > {PARITY_TOLERANCE:.0e} "
            f"over {len(X)} rows"
        )

    probe_idx = np.linspace(0, len(X) - 1, num=min(N_PROBES, len(X)), dtype=int)
    signature = {
        "format": "onnx",
        "triton_platform": "onnxruntime_onnx",
        "component": component,
        "input": {"name": INPUT_NAME, "datatype": "FP32", "dims": [N_FEATURES]},
        "output": {
            "name": OUTPUT_NAME, "datatype": "FP32", "dims": [2],
            "match_probability_index": POSITIVE_CLASS_INDEX,
        },
        "n_features": N_FEATURES,
        "feature_names": list(feature_names),
        "feature_names_sha256": feature_names_sha256(list(feature_names)),
        "onnx_ir_version": int(onnx_model.ir_version),
        "onnx_opsets": {(d.domain or "ai.onnx"): int(d.version) for d in onnx_model.opset_import},
        "parity": {"rows_checked": int(len(X)), "max_abs_diff": max_diff,
                   "tolerance": PARITY_TOLERANCE},
        "probes": [
            {"features": X[i].tolist(), "match_probability": float(onnx_proba[i])}
            for i in probe_idx
        ],
    }
    export = OnnxExport(onnx_bytes=onnx_bytes, signature=signature)
    signature["onnx_sha256"] = export.sha256
    logger.info(
        f"ONNX serving artifact verified: {len(X)} rows, max |diff| {max_diff:.2e}, "
        f"IR {onnx_model.ir_version}, sha256 {export.sha256[:12]}"
    )
    return export
