"""
Unit tests for point-in-time feature retrieval (retrieval_mode="offline", the default).

Dependency type: TEST DOUBLE (httpx.MockTransport). No Feature Store required.

Covers the contract in SERVICE_CONTRACTS.md §1:
  - training calls POST /api/v1/features/offline, not the online batch endpoint
  - as_of_timestamp is sent (it is what prevents label leakage) and is mandatory
  - feature ordering reported by the store is captured and checked for drift
"""
import json
from datetime import datetime

import httpx
import pytest

from src.adapters.feature_store_client import FeatureStoreClient

TENANT = "00000000-0000-0000-0000-000000000001"
AS_OF = datetime(2026, 9, 17, 0, 0, 0)
FEATURE_NAMES = [f"grp_feat_{i:02d}" for i in range(50)]


def _offline_transport(seen, rows=None, feature_names=None):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((request, body))
        payload = {
            "request_id": "00000000-0000-0000-0000-0000000000ff",
            "pairs_requested": len(body["entity_pairs"]),
            "pairs_found": len(rows or []),
            "as_of_timestamp": body["as_of_timestamp"],
            "feature_version": body["feature_version"],
            "feature_names": FEATURE_NAMES if feature_names is None else feature_names,
            "results": rows if rows is not None else [],
            "download_url": None,
            "message": "ok",
        }
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


def _row(e1, e2, value=0.25):
    return {
        "entity_id_1": e1,
        "entity_id_2": e2,
        "feature_vector": [value] * 50,
        "computed_at": "2026-09-16T12:00:00",
    }


async def test_offline_is_the_default_mode():
    seen = []
    client = FeatureStoreClient("http://fs", transport=_offline_transport(seen, [_row("C1", "C2")]))

    await client.get_offline_features([("C1", "C2")], tenant_id=TENANT, as_of_timestamp=AS_OF)

    request, body = seen[0]
    assert request.url.path == "/api/v1/features/offline"
    assert body["entity_pairs"] == [["C1", "C2"]]
    assert body["tenant_id"] == TENANT


async def test_as_of_timestamp_is_sent():
    """The point-in-time bound must reach the store, or labels can leak."""
    seen = []
    client = FeatureStoreClient("http://fs", transport=_offline_transport(seen, [_row("C1", "C2")]))

    await client.get_offline_features([("C1", "C2")], tenant_id=TENANT, as_of_timestamp=AS_OF)

    assert seen[0][1]["as_of_timestamp"] == AS_OF.isoformat()


async def test_missing_as_of_timestamp_is_rejected():
    """Silently defaulting to 'now' would quietly destroy point-in-time correctness."""
    client = FeatureStoreClient("http://fs", transport=_offline_transport([], []))

    with pytest.raises(ValueError, match="as_of_timestamp is required"):
        await client.get_offline_features([("C1", "C2")], tenant_id=TENANT)


async def test_vectors_are_keyed_by_pair():
    client = FeatureStoreClient(
        "http://fs",
        transport=_offline_transport([], [_row("C1", "C2", 0.1), _row("C3", "C4", 0.9)]),
    )

    result = await client.get_offline_features(
        [("C1", "C2"), ("C3", "C4")], tenant_id=TENANT, as_of_timestamp=AS_OF,
    )

    assert set(result) == {"C1:C2", "C3:C4"}
    assert result["C1:C2"] == [0.1] * 50
    assert len(result["C3:C4"]) == 50


async def test_unserved_pairs_are_omitted_not_zero_filled():
    client = FeatureStoreClient("http://fs", transport=_offline_transport([], [_row("C1", "C2")]))

    result = await client.get_offline_features(
        [("C1", "C2"), ("C9", "C10")], tenant_id=TENANT, as_of_timestamp=AS_OF,
    )

    assert list(result) == ["C1:C2"]


async def test_feature_names_are_captured_for_parity_checks():
    client = FeatureStoreClient("http://fs", transport=_offline_transport([], [_row("C1", "C2")]))

    await client.get_offline_features([("C1", "C2")], tenant_id=TENANT, as_of_timestamp=AS_OF)

    assert client.last_feature_names == FEATURE_NAMES


async def test_feature_ordering_change_mid_run_is_fatal():
    """
    If the store reports a different ordering between chunks, the assembled matrix
    would mix two layouts. That must fail loudly, not train on scrambled columns.
    """
    call = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call["n"] += 1
        names = FEATURE_NAMES if call["n"] == 1 else list(reversed(FEATURE_NAMES))
        body = json.loads(request.content)
        return httpx.Response(200, json={
            "request_id": "00000000-0000-0000-0000-0000000000ff",
            "pairs_requested": len(body["entity_pairs"]),
            "pairs_found": 1,
            "as_of_timestamp": body["as_of_timestamp"],
            "feature_version": body["feature_version"],
            "feature_names": names,
            "results": [_row("C1", "C2")],
            "download_url": None,
            "message": "ok",
        })

    client = FeatureStoreClient(
        "http://fs", batch_size=1, transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ValueError, match="changed feature ordering"):
        await client.get_offline_features(
            [("C1", "C2"), ("C3", "C4")], tenant_id=TENANT, as_of_timestamp=AS_OF,
        )


async def test_feature_version_is_sent_from_config():
    seen = []
    client = FeatureStoreClient(
        "http://fs", transport=_offline_transport(seen, [_row("C1", "C2")]),
        feature_version="v2.0.0",
    )

    await client.get_offline_features([("C1", "C2")], tenant_id=TENANT, as_of_timestamp=AS_OF)

    assert seen[0][1]["feature_version"] == "v2.0.0"


def test_invalid_retrieval_mode_is_rejected_at_construction():
    with pytest.raises(ValueError, match="retrieval_mode"):
        FeatureStoreClient("http://fs", retrieval_mode="sometimes")
