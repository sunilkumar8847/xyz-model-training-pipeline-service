"""
Unit tests for the Feature Store client's ONLINE retrieval mode and for Stage 2
matrix-building on top of it.

Dependency type: TEST DOUBLE (httpx.MockTransport). No Feature Store required.

  - FeatureStoreClient batches pairs, sends record fields, retries, and maps vectors
    back (retrieval_mode="online"; the point-in-time "offline" default is covered in
    test_offline_retrieval.py)
  - Stage 2 builds the (N, 50) feature matrix from the client's vectors
  - the feature-coverage gate fails loudly when too few pairs are served

Moved unchanged from test_demo_data_source.py, whose name did not describe these
tests: none of them exercise demo data. Only fixture provenance changed
("demo_synthetic" -> "synthetic").
"""
import json

import httpx
import numpy as np
import pytest

from src.adapters.feature_store_client import FeatureStoreClient
from src.core.config import settings
from src.core.exceptions import FeatureCoverageError
from src.domain.models import LabeledPair, RetrainingTrigger, TrainingRun
from src.pipeline.stages.feature_extraction import FeatureExtractionStage

TENANT = "00000000-0000-0000-0000-000000000001"


def _run() -> TrainingRun:
    return TrainingRun(trigger=RetrainingTrigger.MANUAL, triggered_by="unit-test")


def _mock_transport(seen, unserved_ids=(), status_code=200):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((request, body))
        if status_code != 200:
            return httpx.Response(status_code, json={"detail": "boom"})
        results = [
            None if p["entity_id_1"] in unserved_ids else {"feature_vector": [0.25] * 50}
            for p in body["pairs"]
        ]
        return httpx.Response(200, json={"results": results})

    return httpx.MockTransport(handler)


# ─── FeatureStoreClient ──────────────────────────────────────────────────────

async def test_client_batches_and_sends_fields():
    seen = []
    fields = {f"C{i}": {"name": f"Person {i}", "email": None} for i in range(500)}
    client = FeatureStoreClient(
        "http://fs", batch_size=100, entity_fields=fields.get, transport=_mock_transport(seen),
        retrieval_mode="online",
    )
    pairs = [(f"C{i}", f"C{i + 250}") for i in range(250)]

    result = await client.get_offline_features(pairs, tenant_id=TENANT)

    assert [len(body["pairs"]) for _, body in seen] == [100, 100, 50]
    request, body = seen[0]
    assert request.url.path == "/api/v1/features/batch"
    assert request.headers["x-verified-tenant-id"] == TENANT
    assert body["include_entity_data"] is True
    assert body["pairs"][0]["entity1_fields"] == {"name": "Person 0", "email": None}
    assert len(result) == 250 and result["C0:C250"] == [0.25] * 50


async def test_client_omits_pairs_the_store_cannot_serve():
    client = FeatureStoreClient("http://fs", transport=_mock_transport([], unserved_ids={"C1"}), retrieval_mode="online")

    result = await client.get_offline_features([("C1", "C2"), ("C3", "C4")], tenant_id=TENANT)

    assert list(result) == ["C3:C4"]


async def test_client_without_catalog_sends_no_fields():
    seen = []
    client = FeatureStoreClient("http://fs", transport=_mock_transport(seen), retrieval_mode="online")

    await client.get_offline_features([("C1", "C2")], tenant_id=TENANT)

    body = seen[0][1]
    assert body["include_entity_data"] is False
    assert body["pairs"] == [{"entity_id_1": "C1", "entity_id_2": "C2"}]


async def test_client_raises_after_retrying_server_errors():
    seen = []
    client = FeatureStoreClient(
        "http://fs", retry_backoff_seconds=0, transport=_mock_transport(seen, status_code=500),
        retrieval_mode="online",
    )

    with pytest.raises(httpx.HTTPStatusError):
        await client.get_offline_features([("C1", "C2")], tenant_id=TENANT)
    assert len(seen) == 3


async def test_client_retries_timeout_then_succeeds():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json={"results": [{"feature_vector": [0.5] * 50}]})

    client = FeatureStoreClient("http://fs", retry_backoff_seconds=0, transport=httpx.MockTransport(handler), retrieval_mode="online")

    result = await client.get_offline_features([("C1", "C2")], tenant_id=TENANT)

    assert len(calls) == 2 and result == {"C1:C2": [0.5] * 50}


async def test_client_does_not_retry_client_errors():
    seen = []
    client = FeatureStoreClient(
        "http://fs", retry_backoff_seconds=0, transport=_mock_transport(seen, status_code=422),
        retrieval_mode="online",
    )

    with pytest.raises(httpx.HTTPStatusError):
        await client.get_offline_features([("C1", "C2")], tenant_id=TENANT)
    assert len(seen) == 1


# ─── Stage 2 with the client ─────────────────────────────────────────────────

async def test_feature_extraction_builds_matrix_from_client(monkeypatch):
    """Unserved pairs are dropped from the matrix, not zero-filled."""
    # This scenario drops 1 of 3 pairs on purpose; relax the coverage gate so the
    # matrix-building behaviour is what is under test here. The gate itself is
    # covered by test_coverage_gate_* below.
    monkeypatch.setattr(settings, "FEATURE_COVERAGE_MIN_RATIO", 0.5)

    client = FeatureStoreClient("http://fs", transport=_mock_transport([], unserved_ids={"C5"}), retrieval_mode="online")
    pairs = [
        LabeledPair("C1", "C2", TENANT, label=1, confidence=1.0, source="synthetic"),
        LabeledPair("C3", "C4", TENANT, label=0, confidence=1.0, source="synthetic"),
        LabeledPair("C5", "C6", TENANT, label=0, confidence=1.0, source="synthetic"),
    ]

    dataset = await FeatureExtractionStage(client).execute(_run(), pairs)

    assert dataset.feature_matrix.shape == (2, 50)
    assert dataset.labels.tolist() == [1, 0]
    assert np.allclose(dataset.feature_matrix, 0.25)


# ─── Coverage gate ───────────────────────────────────────────────────────────

async def test_coverage_gate_fails_when_too_many_pairs_unserved(monkeypatch):
    """
    An unreachable/empty Feature Store must FAIL the run, not silently yield a
    tiny dataset that trains dummy models and reports f1=0.0 as success.
    """
    monkeypatch.setattr(settings, "FEATURE_COVERAGE_MIN_RATIO", 0.95)
    client = FeatureStoreClient(
        "http://fs", transport=_mock_transport([], unserved_ids={"C5"}), retrieval_mode="online",
    )
    pairs = [
        LabeledPair("C1", "C2", TENANT, label=1, confidence=1.0, source="synthetic"),
        LabeledPair("C5", "C6", TENANT, label=0, confidence=1.0, source="synthetic"),
    ]

    with pytest.raises(FeatureCoverageError) as exc:
        await FeatureExtractionStage(client).execute(_run(), pairs)

    assert exc.value.retrieved == 1
    assert exc.value.requested == 2
    assert exc.value.code == "MT_9001"


async def test_coverage_gate_passes_at_full_coverage(monkeypatch):
    monkeypatch.setattr(settings, "FEATURE_COVERAGE_MIN_RATIO", 0.95)
    client = FeatureStoreClient("http://fs", transport=_mock_transport([]), retrieval_mode="online")
    pairs = [
        LabeledPair("C1", "C2", TENANT, label=1, confidence=1.0, source="synthetic"),
        LabeledPair("C3", "C4", TENANT, label=0, confidence=1.0, source="synthetic"),
    ]

    dataset = await FeatureExtractionStage(client).execute(_run(), pairs)

    assert dataset.feature_matrix.shape == (2, 50)
