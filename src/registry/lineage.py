"""
model-training-pipeline/src/registry/lineage.py

Walks — and CHECKS — the lineage of a model version, hop by hop:

    dataset manifest / hash        (files on disk, re-hashed)
        -> training run            (MLflow run tags)
        -> model version           (MLflow Model Registry + ensemble manifest)
        -> ONNX artifact hash      (artifact downloaded and re-hashed)
        -> Triton model version    (repository files re-hashed, config parameters;
                                    optionally the RUNNING Triton server)
        -> served inference model  (optionally the running inference service /ready)

Every value is read from its own system and compared with the previous hop. Nothing is
inferred: a value a system does not record is reported as "not recorded", and a value
that disagrees with the previous hop is a broken link.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

OK, BROKEN, NOT_RECORDED, SKIPPED = "ok", "BROKEN", "not recorded", "skipped"


@dataclass
class Hop:
    name: str
    status: str
    facts: Dict[str, str] = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)


@dataclass
class LineageTrace:
    model_name: str
    version: str
    hops: List[Hop] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Every hop was checked and agrees with the one before it."""
        return bool(self.hops) and all(h.status == OK for h in self.hops)

    @property
    def broken(self) -> bool:
        return any(h.status == BROKEN for h in self.hops)

    def as_dict(self) -> Dict:
        return {
            "model_name": self.model_name, "version": self.version,
            "complete": self.complete, "broken": self.broken,
            "hops": [{"name": h.name, "status": h.status, "facts": h.facts, "problems": h.problems}
                     for h in self.hops],
        }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_config_parameters(config_text: str) -> Dict[str, str]:
    """The string `parameters` of a Triton config.pbtxt written by the publisher."""
    return dict(re.findall(r'key:\s*"([^"]+)"\s*value:\s*\{\s*string_value:\s*"([^"]*)"\s*\}', config_text))


def _finish(hop: Hop) -> Hop:
    if hop.problems and hop.status == OK:
        hop.status = BROKEN
    return hop


def trace_lineage(
    version: str,
    model_name: str,
    client=None,
    data_dir: Optional[str] = None,
    triton_repository: Optional[str] = None,
    triton_url: Optional[str] = None,
    inference_url: Optional[str] = None,
) -> LineageTrace:
    import tempfile

    import mlflow
    from mlflow import MlflowClient

    client = client or MlflowClient()
    version = str(version)
    trace = LineageTrace(model_name=model_name, version=version)

    # ── registry version + run (needed by every other hop) ────────────────────
    mv = client.get_model_version(model_name, version)
    run = client.get_run(mv.run_id)
    tags = run.data.tags
    original_run_id = tags.get("migrated_from_run_id")      # set by registry migration

    # ── hop 1: dataset ───────────────────────────────────────────────────────
    hop = Hop("dataset", OK)
    run_dataset_id = tags.get("dataset_id")
    if not run_dataset_id:
        hop.status = NOT_RECORDED
        hop.problems.append("the training run records no dataset_id (trained before dataset identity existed)")
    elif not data_dir:
        hop.status = SKIPPED
        hop.facts["dataset_id (from run)"] = run_dataset_id
        hop.problems.append("no dataset directory given to verify against")
    else:
        from src.adapters.synthetic_data import SyntheticDataError, dataset_provenance
        try:
            prov = dataset_provenance(data_dir)     # re-hashes the files
            hop.facts.update({
                "directory": str(data_dir),
                "dataset_id": prov["dataset_id"],
                "manifest_sha256": prov["manifest_sha256"],
                "content_sha256.pairs": prov["content_sha256"].get("pairs", ""),
                "generator_version": str(prov["generator_version"]),
                "seed": str(prov["seed"]),
                "tenants": str(len(prov["tenants"])),
                "label_source": str(prov["label_source"]),
            })
            if prov["dataset_id"] != run_dataset_id:
                hop.problems.append(
                    f"dataset on disk is {prov['dataset_id']}, the run was trained on {run_dataset_id}")
            if tags.get("dataset_manifest_sha256") != prov["manifest_sha256"]:
                hop.problems.append("manifest sha256 on disk differs from the one the run recorded")
        except SyntheticDataError as exc:
            hop.problems.append(f"dataset does not verify: {exc}")
    trace.hops.append(_finish(hop))

    # ── hop 2: training run ──────────────────────────────────────────────────
    hop = Hop("training_run", OK, {
        "mlflow_run_id": mv.run_id,
        "status": run.info.status,
        "dataset_version": tags.get("dataset_version", ""),
        "dataset_id": tags.get("dataset_id", ""),
        "data_mode": tags.get("data_mode", ""),
        "label_sources": tags.get("label_sources", ""),
        "feature_catalog_version": tags.get("feature_catalog_version", ""),
        "feature_as_of": tags.get("feature_as_of", ""),
        "git_sha": tags.get("git_sha", ""),
        "git_dirty": tags.get("git_dirty", ""),
        "environment": tags.get("environment", ""),
    })
    if original_run_id:
        hop.facts["migrated_from_run_id"] = original_run_id
        hop.facts["migrated_from_registry"] = tags.get("migrated_from_registry", "")
    if run.info.status != "FINISHED":
        hop.problems.append(f"run status is {run.info.status}")
    if not tags.get("dataset_version"):
        hop.problems.append("run has no dataset_version tag")
    if tags.get("git_dirty") == "true":
        hop.facts["note"] = "built from a working tree with uncommitted changes: git_sha alone does not reproduce it"
    trace.hops.append(_finish(hop))

    # ── hop 3: model version + manifest, hop 4: ONNX artifact ────────────────
    hop3 = Hop("model_version", OK, {
        "registry": str(mlflow.get_registry_uri() or mlflow.get_tracking_uri()),
        "name": model_name, "version": version, "stage": mv.current_stage or "None",
    })
    hop4 = Hop("onnx_artifact", OK)
    manifest: Dict = {}
    onnx_sha = None
    try:
        manifest = json.loads(mlflow.artifacts.load_text(f"runs:/{mv.run_id}/ensemble_manifest.json"))
    except Exception as exc:
        hop3.problems.append(f"ensemble_manifest.json cannot be read: {type(exc).__name__}")
    if manifest:
        hop3.facts["manifest.registered_model_version"] = str(manifest.get("registered_model_version"))
        hop3.facts["manifest.dataset_version"] = str(manifest.get("dataset_version"))
        if str(manifest.get("registered_model_version")) != version:
            hop3.problems.append("manifest belongs to another registry version")
        if manifest.get("dataset_version") != tags.get("dataset_version"):
            hop3.problems.append("manifest dataset_version differs from the run's")
        manifest_ids = ",".join(d["dataset_id"] for d in (manifest.get("dataset") or {}).get("datasets", []))
        if run_dataset_id and manifest_ids != run_dataset_id:
            hop3.problems.append("manifest dataset_id differs from the run's")
        if manifest.get("mlflow_run_id") not in (mv.run_id, original_run_id):
            hop3.problems.append("manifest names a different MLflow run")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            onnx_path = Path(mlflow.artifacts.download_artifacts(
                run_id=mv.run_id, artifact_path="onnx_model/model.onnx", dst_path=tmp))
            onnx_sha = _sha256(onnx_path)
            hop4.facts["onnx_sha256 (artifact re-hashed)"] = onnx_sha
            hop4.facts["manifest.onnx_sha256"] = str(manifest.get("onnx_sha256"))
            if manifest.get("onnx_sha256") != onnx_sha:
                hop4.problems.append("artifact bytes do not match the hash training recorded")
            if mv.tags.get("onnx_sha256") and mv.tags["onnx_sha256"] != onnx_sha:
                hop4.problems.append("model-version tag onnx_sha256 does not match the artifact")
        except Exception as exc:
            hop4.status = NOT_RECORDED
            hop4.problems.append(f"no ONNX serving artifact ({type(exc).__name__})")
    trace.hops.append(_finish(hop3))
    trace.hops.append(_finish(hop4))

    # ── hop 5: Triton model ──────────────────────────────────────────────────
    triton_name = f"customer_matcher_v{version}"
    hop = Hop("triton_model", OK, {"triton_model": triton_name, "version_slot": "1"})
    params: Dict[str, str] = {}
    if not triton_repository:
        hop.status = SKIPPED
        hop.problems.append("no Triton model repository given")
    else:
        model_dir = Path(triton_repository) / triton_name
        model_file = model_dir / "1" / "model.onnx"
        if not model_file.is_file() or not (model_dir / "config.pbtxt").is_file():
            hop.status = NOT_RECORDED
            hop.problems.append(f"{model_dir} is not published")
        else:
            repo_sha = _sha256(model_file)
            params = parse_config_parameters((model_dir / "config.pbtxt").read_text(encoding="utf-8"))
            hop.facts.update({
                "repository": str(triton_repository),
                "onnx_sha256 (file re-hashed)": repo_sha,
                "config.registry_version": params.get("registry_version", ""),
                "config.mlflow_run_id": params.get("mlflow_run_id", ""),
                "config.dataset_id": params.get("dataset_id", ""),
                "config.dataset_version": params.get("dataset_version", ""),
                "config.registry_stage": params.get("registry_stage", ""),
            })
            if onnx_sha and repo_sha != onnx_sha:
                hop.problems.append("the file Triton serves is NOT the registry's artifact")
            if params.get("onnx_sha256") != repo_sha:
                hop.problems.append("config onnx_sha256 does not match the file next to it")
            if params.get("registry_version") != version:
                hop.problems.append("config registry_version differs")
            if params.get("mlflow_run_id") not in (mv.run_id, original_run_id):
                hop.problems.append("config mlflow_run_id is not this version's run")
            if run_dataset_id and params.get("dataset_id") != run_dataset_id:
                hop.problems.append("config dataset_id differs from the training run's")
            if mv.tags.get("triton_onnx_sha256") and mv.tags["triton_onnx_sha256"] != repo_sha:
                hop.problems.append("registry tag triton_onnx_sha256 differs from the published file")
            hop.facts["registry tag triton_model"] = mv.tags.get("triton_model", "(not tagged)")
    if triton_url and hop.status == OK:
        import httpx
        try:
            base = triton_url.rstrip("/")
            ready = httpx.get(f"{base}/v2/models/{triton_name}/ready", timeout=5).status_code == 200
            live = httpx.get(f"{base}/v2/models/{triton_name}/config", timeout=5).json()
            live_params = {k: v.get("string_value", "") for k, v in (live.get("parameters") or {}).items()}
            hop.facts["running server"] = f"{base} ready={ready}"
            hop.facts["running server onnx_sha256"] = live_params.get("onnx_sha256", "")
            if not ready:
                hop.problems.append("the running Triton server does not have this model READY")
            if live_params.get("onnx_sha256") != params.get("onnx_sha256"):
                hop.problems.append("the running server's model config differs from the repository")
        except Exception as exc:
            hop.problems.append(f"running Triton server could not be queried: {type(exc).__name__}")
    trace.hops.append(_finish(hop))

    # ── hop 6: served inference model ────────────────────────────────────────
    hop = Hop("inference_service", OK)
    if not inference_url:
        hop.status = SKIPPED
        hop.problems.append("no inference service URL given")
    else:
        import httpx
        try:
            r = httpx.get(f"{inference_url.rstrip('/')}/ready", timeout=10)
            body = r.json()
            hop.facts.update({
                "url": inference_url, "http_status": str(r.status_code),
                "serving_model_version": str(body.get("serving_model_version")),
                "triton_model": str(body.get("triton_model")),
                "registry_version": str(body.get("registry_version")),
                "mlflow_run_id": str(body.get("mlflow_run_id")),
                "feature_catalog_version": str(body.get("feature_catalog_version")),
            })
            if r.status_code != 200 or not body.get("ready"):
                hop.problems.append("inference service is not ready")
            if body.get("serving_model_version") != f"v{version}":
                hop.problems.append(f"it serves {body.get('serving_model_version')}, not v{version}")
            if body.get("triton_model") != triton_name:
                hop.problems.append("it addresses a different Triton model")
            if params and body.get("mlflow_run_id") != params.get("mlflow_run_id"):
                hop.problems.append("its mlflow_run_id differs from the Triton model's")
            if body.get("feature_catalog_version") != tags.get("feature_catalog_version"):
                hop.problems.append("its feature catalog version differs from the training run's")
        except Exception as exc:
            hop.problems.append(f"inference service could not be queried: {type(exc).__name__}")
    trace.hops.append(_finish(hop))
    return trace
