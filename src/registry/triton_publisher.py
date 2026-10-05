"""
model-training-pipeline/src/registry/triton_publisher.py

Publishes a REGISTERED model version from the MLflow Model Registry into a Triton
model repository, following the inference service's Triton design:

    <repository>/customer_matcher_v{N}/
        config.pbtxt               platform onnxruntime_onnx, input "input" FP32 [50],
                                   output "probabilities" FP32 [2]; lineage in parameters
        1/model.onnx               the exact ONNX artifact logged with registry version N
        serving_manifest.json      full mapping registry <-> MLflow run <-> Triton

Identity mapping (explicit, never assumed equal):
    registry  : <MODEL_NAME> version N          (MLflow Model Registry)
    run       : the MLflow run that produced version N
    triton    : model "customer_matcher_vN", version slot "1"
                (one model per registry version, as the inference service's client
                 addresses `customer_matcher_{model_version}`; the slot is always 1)
    serving   : model_version string "vN" used by the inference service / API

Nothing is published unless every check passes. Publication is atomic: the model is
assembled outside the repository and moved in with a single rename, so Triton (which
polls the repository) never sees a half-written model.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TRITON_MODEL_PREFIX = "customer_matcher"
TRITON_VERSION_SLOT = "1"
TRITON_MAX_BATCH_SIZE = 128


class TritonPublishError(RuntimeError):
    """The registry version cannot be published as a faithful Triton model."""


@dataclass
class PublishResult:
    registry_model_name: str
    registry_version: str
    mlflow_run_id: str
    serving_model_version: str
    triton_model_name: str
    triton_version: str
    model_dir: str
    onnx_sha256: str
    feature_catalog_version: str
    dataset_version: str
    already_published: bool
    dataset_id: str = ""
    registry_stage: str = ""
    registry_uri: str = ""


def triton_model_name(registry_version: str) -> str:
    return f"{TRITON_MODEL_PREFIX}_v{registry_version}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_version(client, model_name: str, version: Optional[str], stage: Optional[str]) -> str:
    """Exactly one of version / stage. A stage resolves to its highest version."""
    if bool(version) == bool(stage):
        raise TritonPublishError("specify exactly one of --version or --stage")
    if version:
        if not str(version).isdigit():
            raise TritonPublishError(f"invalid registry version {version!r}")
        try:
            client.get_model_version(model_name, str(version))
        except Exception as exc:
            raise TritonPublishError(f"{model_name} version {version} not found in registry: {exc}") from exc
        return str(version)
    candidates = client.search_model_versions(f"name='{model_name}'")
    in_stage = [m for m in candidates if (m.current_stage or "").lower() == stage.lower()]
    if not in_stage:
        raise TritonPublishError(f"no {model_name} version is in stage {stage!r}")
    return str(max(int(m.version) for m in in_stage))


def render_config(name: str, signature: dict, lineage: dict) -> str:
    params = "\n".join(
        f'  {{ key: "{k}" value: {{ string_value: "{v}" }} }},' for k, v in lineage.items()
    ).rstrip(",")
    return f'''name: "{name}"
platform: "onnxruntime_onnx"
max_batch_size: {TRITON_MAX_BATCH_SIZE}
version_policy: {{ specific: {{ versions: [ {TRITON_VERSION_SLOT} ] }} }}
input [
  {{
    name: "{signature['input']['name']}"
    data_type: TYPE_FP32
    dims: [ {signature['n_features']} ]
  }}
]
output [
  {{
    name: "{signature['output']['name']}"
    data_type: TYPE_FP32
    dims: [ {signature['output']['dims'][0]} ]
  }}
]
instance_group [ {{ kind: KIND_CPU, count: 1 }} ]
parameters [
{params}
]
'''


def publish_registered_model(
    repository: str,
    model_name: str,
    version: Optional[str] = None,
    stage: Optional[str] = None,
    client=None,
    allow_scratch_registry: bool = False,
    require_stage: Optional[str] = None,
) -> PublishResult:
    """
    allow_scratch_registry  Only the MLflow SERVER registry is authoritative. A model in
                            a local file/sqlite registry is refused unless this is set
                            (unit tests, throw-away local experiments).
    require_stage           Refuse unless the version is in this registry stage (the
                            `release` command passes "Production", so what Triton
                            serves is what the registry says is in production).
    """
    import mlflow
    import numpy as np
    import onnxruntime as ort
    from mlflow import MlflowClient

    from src.registry.onnx_export import N_FEATURES, PARITY_TOLERANCE, feature_names_sha256

    if not repository:
        raise TritonPublishError("no Triton model repository given (--repo or TRITON_MODEL_REPOSITORY)")
    repo = Path(repository)
    if not repo.is_dir():
        raise TritonPublishError(f"Triton model repository {repo} does not exist")

    client = client or MlflowClient()
    reg_uri = mlflow.get_registry_uri() or mlflow.get_tracking_uri()
    scope = "server" if str(reg_uri).lower().startswith(("http://", "https://")) else "local-scratch"
    if scope != "server" and not allow_scratch_registry:
        raise TritonPublishError(
            f"registry {reg_uri!r} is a local scratch store, not the authoritative MLflow "
            f"server. Models are published to Triton only from the server registry "
            f"(MLFLOW_TRACKING_URI=http://...)."
        )
    reg_version = resolve_version(client, model_name, version, stage)
    mv = client.get_model_version(model_name, reg_version)
    run = client.get_run(mv.run_id)
    if require_stage and (mv.current_stage or "").lower() != require_stage.lower():
        raise TritonPublishError(
            f"{model_name} v{reg_version} is in stage {mv.current_stage!r}, not {require_stage!r}"
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        try:
            onnx_path = Path(mlflow.artifacts.download_artifacts(
                run_id=mv.run_id, artifact_path="onnx_model/model.onnx", dst_path=str(tmp)))
            sig_path = Path(mlflow.artifacts.download_artifacts(
                run_id=mv.run_id, artifact_path="onnx_model/serving_signature.json", dst_path=str(tmp)))
        except Exception as exc:
            raise TritonPublishError(
                f"{model_name} v{reg_version} (run {mv.run_id}) has no ONNX serving artifact. "
                f"It was registered before ONNX export existed or the export failed; retrain "
                f"to produce a servable version. ({type(exc).__name__})"
            ) from exc
        try:
            manifest = json.loads(mlflow.artifacts.load_text(f"runs:/{mv.run_id}/ensemble_manifest.json"))
        except Exception as exc:
            raise TritonPublishError(f"v{reg_version} has no ensemble_manifest.json: {exc}") from exc
        signature = json.loads(sig_path.read_text(encoding="utf-8"))

        # ── integrity: the bytes are the ones training verified ──
        actual_sha = _sha256(onnx_path)
        for label, expected in (("serving_signature", signature.get("onnx_sha256")),
                                ("ensemble_manifest", manifest.get("onnx_sha256"))):
            if expected != actual_sha:
                raise TritonPublishError(
                    f"ONNX artifact sha256 {actual_sha[:12]} does not match {label} "
                    f"({str(expected)[:12]}): artifact corrupt or mismatched")

        # ── identity: the manifest belongs to THIS registry version ──
        if manifest.get("registered_model_version") != reg_version:
            raise TritonPublishError(
                f"manifest registered_model_version={manifest.get('registered_model_version')!r} "
                f"!= registry version {reg_version}")

        # ── serving scope: the ONNX file is the XGBoost component only ──
        weights = manifest.get("weights", {})
        if not (weights.get("xgboost") == 1.0 and not weights.get("transformer") and not weights.get("gnn")):
            raise TritonPublishError(
                f"v{reg_version} is an ensemble with weights {weights}; its ONNX artifact holds "
                f"only the XGBoost component, so publishing it would serve the wrong model. "
                f"Full-ensemble Triton packaging is not implemented.")

        # ── feature contract ──
        names = signature.get("feature_names") or []
        if signature.get("n_features") != N_FEATURES or len(names) != N_FEATURES:
            raise TritonPublishError(f"serving signature declares {len(names)} features, expected {N_FEATURES}")
        if feature_names_sha256(names) != signature.get("feature_names_sha256"):
            raise TritonPublishError("feature_names_sha256 does not match the declared feature names")

        # ── lineage ──
        for key in ("dataset_version", "feature_catalog_version"):
            if not manifest.get(key) or manifest.get(key) == "unknown":
                raise TritonPublishError(f"v{reg_version} lacks lineage field {key!r}")

        # ── behaviour: the artifact reproduces the probabilities training recorded ──
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        probes = signature.get("probes") or []
        if not probes:
            raise TritonPublishError("serving signature carries no verification probes")
        X = np.array([p["features"] for p in probes], dtype=np.float32)
        got = sess.run([signature["output"]["name"]], {signature["input"]["name"]: X})[0][
            :, signature["output"]["match_probability_index"]]
        want = np.array([p["match_probability"] for p in probes], dtype=np.float32)
        worst = float(np.abs(got - want).max())
        if worst > PARITY_TOLERANCE:
            raise TritonPublishError(f"probe verification failed: max |diff| {worst:.3e}")

        name = triton_model_name(reg_version)
        lineage = {
            "registry_model_name": model_name,
            "registry_version": reg_version,
            "serving_model_version": f"v{reg_version}",
            "mlflow_run_id": mv.run_id,
            "feature_catalog_version": manifest["feature_catalog_version"],
            "dataset_version": manifest["dataset_version"],
            "git_sha": manifest.get("git_sha", "unknown"),
            "feature_names_sha256": signature["feature_names_sha256"],
            "n_features": str(N_FEATURES),
            "match_probability_index": str(signature["output"]["match_probability_index"]),
            "onnx_sha256": actual_sha,
            # Recorded test-split metric of THIS version (synthetic bootstrap data).
            "test_f1": str(manifest.get("metrics", {}).get("f1", "")),
        }
        # Dataset identity and registry provenance — copied from what training recorded
        # in the manifest, never derived here. Absent for versions registered before
        # dataset identity existed; those keys are then simply not written.
        dataset = manifest.get("dataset") or {}
        datasets = dataset.get("datasets") or []
        if datasets:
            lineage["dataset_id"] = ",".join(d["dataset_id"] for d in datasets)
            lineage["dataset_manifest_sha256"] = ",".join(d["manifest_sha256"] for d in datasets)
        if dataset:
            lineage["data_mode"] = str(dataset.get("data_mode", ""))
            lineage["label_sources"] = ",".join(
                f"{k}={v}" for k, v in sorted((dataset.get("label_sources") or {}).items()))
        if manifest.get("git_dirty") is not None:
            lineage["git_dirty"] = str(manifest["git_dirty"]).lower()
        lineage["registry_scope"] = scope
        lineage["registry_stage"] = mv.current_stage or "None"
        result = PublishResult(
            registry_model_name=model_name, registry_version=reg_version,
            mlflow_run_id=mv.run_id, serving_model_version=f"v{reg_version}",
            triton_model_name=name, triton_version=TRITON_VERSION_SLOT,
            model_dir=str(repo / name), onnx_sha256=actual_sha,
            feature_catalog_version=manifest["feature_catalog_version"],
            dataset_version=manifest["dataset_version"], already_published=False,
            dataset_id=lineage.get("dataset_id", ""), registry_stage=mv.current_stage or "None",
            registry_uri=str(reg_uri),
        )

        target = repo / name
        if target.exists():
            existing = target / TRITON_VERSION_SLOT / "model.onnx"
            if existing.is_file() and _sha256(existing) == actual_sha:
                logger.info(f"{name} already published with identical artifact; nothing to do")
                result.already_published = True
                _tag_registry(client, model_name, reg_version, name, actual_sha)
                return result
            raise TritonPublishError(
                f"{target} already exists with a DIFFERENT artifact; refusing to overwrite a "
                f"published model version")

        # A NEW publication must be traceable to its data: the manifest has to say what
        # the model was trained on, and synthetic labels must name their dataset.
        if not dataset:
            raise TritonPublishError(
                f"v{reg_version} has no dataset identity in its manifest (registered before "
                f"dataset lineage existed); retrain to produce a traceable version.")
        if "synthetic" in (dataset.get("label_sources") or {}) and not datasets:
            raise TritonPublishError(
                f"v{reg_version} was trained on synthetic labels but records no dataset_id")

        # Assemble outside the repository, then a single rename into it.
        staging_root = repo.parent / ".triton-staging"
        staging_root.mkdir(exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=staging_root))
        try:
            (staging / TRITON_VERSION_SLOT).mkdir()
            shutil.copyfile(onnx_path, staging / TRITON_VERSION_SLOT / "model.onnx")
            (staging / "config.pbtxt").write_text(render_config(name, signature, lineage), encoding="utf-8")
            (staging / "serving_manifest.json").write_text(json.dumps({
                **asdict(result),
                "model_dir": str(target),
                "feature_names": names,
                "signature": {k: signature[k] for k in ("input", "output", "onnx_ir_version", "onnx_opsets", "parity")},
                "published_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2), encoding="utf-8")
            os.replace(staging, target)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    _tag_registry(client, model_name, reg_version, name, actual_sha)
    logger.info(f"Published {model_name} v{reg_version} -> {target} (Triton {name}, slot {TRITON_VERSION_SLOT})")
    return result


def _tag_registry(client, model_name: str, version: str, triton_name: str, onnx_sha256: str) -> None:
    """
    Record the publication ON the registry version, so the link can be followed from
    the registry to Triton as well as from Triton (config parameters) to the registry.
    """
    try:
        client.set_model_version_tag(model_name, version, "triton_model", triton_name)
        client.set_model_version_tag(model_name, version, "triton_version_slot", TRITON_VERSION_SLOT)
        client.set_model_version_tag(model_name, version, "triton_onnx_sha256", onnx_sha256)
        client.set_model_version_tag(
            model_name, version, "triton_published_at", datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        raise TritonPublishError(
            f"{triton_name} is in the Triton repository, but the registry could not be "
            f"tagged with the publication ({type(exc).__name__}: {exc})"
        ) from exc
