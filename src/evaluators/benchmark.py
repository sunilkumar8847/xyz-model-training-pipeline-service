"""
model-training-pipeline/src/evaluators/benchmark.py

Frozen benchmark and decision-tier evaluation.

Why this exists
  * Each training run evaluates on its own test split, so two model versions were never
    scored on the same pairs.
  * Evaluation used a single 0.5 threshold. The product does not decide at 0.5: the
    inference service maps the score to five tiers and AUTO_MATCH (>= 0.95) merges
    records without a human. How often AUTO_MATCH is right had never been measured.

What it does
  * The benchmark is the seed corpus's HOLDOUT split (identities that are never in the
    train split and that training never reads), pinned by the dataset's content hash:
    benchmark_id changes if a single pair or label changes.
  * The model is scored through its published serving artifact (the ONNX file logged
    with the registry version) on features read from the Feature Store, i.e. the same
    artifact and the same features the serving path uses.
  * It reports, per tier, how many pairs land there and how many are true matches —
    in particular AUTO_MATCH precision (wrong merges) and recall.

It MEASURES. It does not tune thresholds, and nothing here changes the model.
The numbers describe behaviour on synthetic bootstrap data, not on customer data.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# The decision tiers of the inference service (InferenceService._classify_decision):
# a score is assigned to the first tier whose lower bound it reaches.
DECISION_TIERS = (
    ("AUTO_MATCH", 0.95),
    ("LIKELY_MATCH", 0.85),
    ("UNCERTAIN", 0.50),
    ("LIKELY_NON", 0.15),
    ("AUTO_NON_MATCH", 0.0),
)
REVIEW_TIERS = ("LIKELY_MATCH", "UNCERTAIN", "LIKELY_NON")


class BenchmarkError(RuntimeError):
    """The benchmark could not be evaluated faithfully."""


def tier_of(score: float) -> str:
    for name, lower in DECISION_TIERS:
        if score >= lower:
            return name
    return DECISION_TIERS[-1][0]


def _ratio(num: int, den: int) -> Optional[float]:
    """None (not 0.0) when there is nothing to divide by — "no pairs" is not "0%"."""
    return round(num / den, 6) if den else None


def tier_report(
    labels: Sequence[int],
    scores: Sequence[float],
    tenants: Optional[Sequence[str]] = None,
    negative_kinds: Optional[Sequence[Optional[str]]] = None,
) -> Dict:
    """
    Decision-tier evaluation of scored, labeled pairs.

    auto_match.precision      of the pairs the system would merge automatically, the
                              share that really are the same entity (1 - wrong-merge rate)
    auto_match.recall         share of all true matches that are merged automatically
    auto_non_match.precision  of the pairs rejected automatically, the share that really
                              are different entities
    review                    pairs sent to a human (the three middle tiers)
    """
    labels = [int(x) for x in labels]
    scores = [float(x) for x in scores]
    if len(labels) != len(scores):
        raise ValueError("labels and scores differ in length")
    n, n_pos = len(labels), sum(labels)
    n_neg = n - n_pos
    tiers_of = [tier_of(s) for s in scores]

    tiers: Dict[str, Dict] = {}
    for name, lower in DECISION_TIERS:
        idx = [i for i, t in enumerate(tiers_of) if t == name]
        pos = sum(labels[i] for i in idx)
        tiers[name] = {
            "min_score": lower,
            "pairs": len(idx),
            "true_matches": pos,
            "true_non_matches": len(idx) - pos,
            "share_of_pairs": _ratio(len(idx), n),
            "match_rate": _ratio(pos, len(idx)),
        }

    am, anm = tiers["AUTO_MATCH"], tiers["AUTO_NON_MATCH"]
    review_pairs = sum(tiers[t]["pairs"] for t in REVIEW_TIERS)
    review_matches = sum(tiers[t]["true_matches"] for t in REVIEW_TIERS)

    pred = [1 if s >= 0.5 else 0 for s in scores]
    tp = sum(1 for p, y in zip(pred, labels) if p and y)
    fp = sum(1 for p, y in zip(pred, labels) if p and not y)
    fn = sum(1 for p, y in zip(pred, labels) if not p and y)
    prec, rec = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = round(2 * prec * rec / (prec + rec), 6) if prec and rec else (0.0 if n_pos else None)

    report = {
        "pairs": n,
        "true_matches": n_pos,
        "true_non_matches": n_neg,
        "tiers": tiers,
        "auto_match": {
            "pairs": am["pairs"],
            "correct_merges": am["true_matches"],
            "wrong_merges": am["true_non_matches"],
            "precision": _ratio(am["true_matches"], am["pairs"]),
            "recall": _ratio(am["true_matches"], n_pos),
        },
        "auto_non_match": {
            "pairs": anm["pairs"],
            "correct_rejections": anm["true_non_matches"],
            "missed_matches": anm["true_matches"],
            "precision": _ratio(anm["true_non_matches"], anm["pairs"]),
            "recall": _ratio(anm["true_non_matches"], n_neg),
        },
        "review": {
            "pairs": review_pairs,
            "share_of_pairs": _ratio(review_pairs, n),
            "true_matches": review_matches,
        },
        "at_threshold_0_5": {"precision": prec, "recall": rec, "f1": f1,
                             "tp": tp, "fp": fp, "fn": fn},
    }

    if tenants is not None:
        by_tenant: Dict[str, Dict] = {}
        for tenant in sorted(set(tenants)):
            idx = [i for i, t in enumerate(tenants) if t == tenant]
            t_pos = sum(labels[i] for i in idx)
            a = [i for i in idx if tiers_of[i] == "AUTO_MATCH"]
            a_pos = sum(labels[i] for i in a)
            by_tenant[tenant] = {
                "pairs": len(idx), "true_matches": t_pos,
                "auto_match_pairs": len(a), "auto_match_wrong_merges": len(a) - a_pos,
                "auto_match_precision": _ratio(a_pos, len(a)),
                "auto_match_recall": _ratio(a_pos, t_pos),
            }
        report["by_tenant"] = by_tenant

    if negative_kinds is not None:
        by_kind: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for y, t, kind in zip(labels, tiers_of, negative_kinds):
            if y == 0:
                by_kind[kind or "unspecified"]["pairs"] += 1
                by_kind[kind or "unspecified"][t] += 1
        report["non_matches_by_kind"] = {
            k: {"pairs": v["pairs"], **{name: v.get(name, 0) for name, _ in DECISION_TIERS}}
            for k, v in sorted(by_kind.items())
        }
    return report


def benchmark_id(pairs: Sequence, dataset_id: str) -> str:
    """Identity of the benchmark: the dataset it comes from and every pair and label."""
    lines = sorted(
        f"{p.tenant_id}:{min(p.entity_id_1, p.entity_id_2)}:{max(p.entity_id_1, p.entity_id_2)}:{p.label}"
        for p in pairs
    )
    h = hashlib.sha256(f"dataset:{dataset_id}\n".encode())
    h.update("\n".join(lines).encode())
    return f"bench-{len(lines)}-{h.hexdigest()[:16]}"


def score_with_onnx(onnx_path: str, signature: Dict, features: np.ndarray) -> np.ndarray:
    """Match scores from the serving artifact, exactly as Triton computes them."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run([signature["output"]["name"]],
                   {signature["input"]["name"]: np.asarray(features, dtype=np.float32)})[0]
    return out[:, signature["output"]["match_probability_index"]].astype(float)


async def run_benchmark(
    version: str,
    data_dir: str,
    model_name: str,
    client=None,
    feature_client=None,
    as_of=None,
    log_to_registry: bool = True,
) -> Dict:
    """
    Evaluate registry version `version` on the frozen benchmark and (by default) record
    the result on that version: artifact benchmark/<benchmark_id>.json, benchmark_*
    metrics on its run, and benchmark tags on the model version.
    """
    import tempfile
    from datetime import datetime
    from pathlib import Path

    import mlflow
    from mlflow import MlflowClient

    from src.adapters.feature_store_client import FeatureStoreClient
    from src.adapters.synthetic_data import dataset_provenance, load_benchmark_pairs
    from src.registry.onnx_export import feature_names_sha256

    client = client or MlflowClient()
    provenance = dataset_provenance(data_dir)
    rows = load_benchmark_pairs(data_dir)
    if not rows:
        raise BenchmarkError(f"{data_dir} has no holdout pairs to benchmark on")
    pairs = [p for p, _ in rows]
    info = {(p.tenant_id, p.entity_id_1, p.entity_id_2): i for p, i in rows}
    bench_id = benchmark_id(pairs, provenance["dataset_id"])

    mv = client.get_model_version(model_name, str(version))
    with tempfile.TemporaryDirectory() as tmp:
        try:
            onnx_path = mlflow.artifacts.download_artifacts(
                run_id=mv.run_id, artifact_path="onnx_model/model.onnx", dst_path=tmp)
            sig_path = mlflow.artifacts.download_artifacts(
                run_id=mv.run_id, artifact_path="onnx_model/serving_signature.json", dst_path=tmp)
        except Exception as exc:
            raise BenchmarkError(
                f"{model_name} v{version} has no ONNX serving artifact to benchmark "
                f"({type(exc).__name__})") from exc
        signature = json.loads(Path(sig_path).read_text(encoding="utf-8"))
        onnx_sha = hashlib.sha256(Path(onnx_path).read_bytes()).hexdigest()

        feature_client = feature_client or FeatureStoreClient.from_settings()
        as_of = as_of or datetime.utcnow()
        by_tenant: Dict[str, List] = defaultdict(list)
        for p in pairs:
            by_tenant[p.tenant_id].append(p)

        X, y, tenants, kinds, missing = [], [], [], [], 0
        for tenant in sorted(by_tenant):
            got = await feature_client.get_offline_features(
                entity_pairs=[(p.entity_id_1, p.entity_id_2) for p in by_tenant[tenant]],
                tenant_id=tenant, as_of_timestamp=as_of)
            for p in by_tenant[tenant]:
                vec = got.get(f"{p.entity_id_1}:{p.entity_id_2}") or got.get(f"{p.entity_id_2}:{p.entity_id_1}")
                if vec is None:
                    missing += 1
                    continue
                X.append(vec)
                y.append(p.label)
                tenants.append(tenant)
                kinds.append(info[(p.tenant_id, p.entity_id_1, p.entity_id_2)].get("negative_kind"))
        if missing:
            # A benchmark on "the pairs that happened to have features" is a different
            # benchmark every time. All of it, or nothing.
            raise BenchmarkError(
                f"{missing}/{len(pairs)} benchmark pairs have no features in the Feature Store; "
                f"materialize the dataset first")
        names = getattr(feature_client, "last_feature_names", None)
        if not names or feature_names_sha256(names) != signature["feature_names_sha256"]:
            raise BenchmarkError(
                "the Feature Store's feature names/order do not match the model's serving contract")

        scores = score_with_onnx(onnx_path, signature, np.asarray(X, dtype=np.float32))

    report = tier_report(y, scores, tenants=tenants, negative_kinds=kinds)
    report.update({
        "benchmark_id": bench_id,
        "dataset_id": provenance["dataset_id"],
        "dataset_manifest_sha256": provenance["manifest_sha256"],
        "split": "holdout",
        "label_source": provenance["label_source"],
        "model_name": model_name,
        "model_version": str(version),
        "mlflow_run_id": mv.run_id,
        "onnx_sha256": onnx_sha,
        "feature_catalog_version": getattr(feature_client, "_feature_version", None),
        "features_as_of": as_of.isoformat(),
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "note": "Synthetic bootstrap data. Not an estimate of accuracy on customer data.",
    })

    if log_to_registry:
        client.log_dict(mv.run_id, report, f"benchmark/{bench_id}.json")
        metrics = {
            "benchmark_auto_match_precision": report["auto_match"]["precision"],
            "benchmark_auto_match_recall": report["auto_match"]["recall"],
            "benchmark_auto_match_wrong_merges": report["auto_match"]["wrong_merges"],
            "benchmark_auto_non_match_precision": report["auto_non_match"]["precision"],
            "benchmark_auto_non_match_missed_matches": report["auto_non_match"]["missed_matches"],
            "benchmark_review_share": report["review"]["share_of_pairs"],
            "benchmark_f1_at_0_5": report["at_threshold_0_5"]["f1"],
            "benchmark_pairs": report["pairs"],
        }
        for key, value in metrics.items():
            if value is not None:
                client.log_metric(mv.run_id, key, float(value))
        client.set_model_version_tag(model_name, str(version), "benchmark_id", bench_id)
        for key in ("precision", "recall", "wrong_merges"):
            client.set_model_version_tag(
                model_name, str(version), f"benchmark_auto_match_{key}", str(report["auto_match"][key]))
    return report
