"""
model-training-pipeline/src/adapters/feature_store_client.py

HTTP client for the Feature Store service, used by Stage 2 (feature extraction).

Two retrieval modes, chosen explicitly (never silently):

  "offline" (default, contract-correct) — POST /api/v1/features/offline.
      Point-in-time correct: returns each pair's latest vector computed at or before
      `as_of_timestamp`, read from the offline store (S3 Parquet). This is the mode the
      Training Pipeline LLD requires, because it is what prevents label leakage.

  "online" — POST /api/v1/features/batch.
      Current (not point-in-time) features, computed on the fly for cache misses when the
      records' fields are supplied via an entity-field lookup (in local development,
      the synthetic dataset's entities.parquet via SYNTHETIC_DATA_DIR). Useful before
      features have been materialized.

Both modes return the same 50-dim vector ordered by sorted feature name, so training and
inference consume an identical layout. `feature_names` from the offline response is exposed
via `last_feature_names` so callers can assert that ordering rather than assume it.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import httpx

from src.core.config import settings

logger = logging.getLogger(__name__)

EntityFields = Dict[str, Optional[str]]
EntityFieldLookup = Callable[[str], Optional[EntityFields]]

BATCH_ENDPOINT = "/api/v1/features/batch"
OFFLINE_ENDPOINT = "/api/v1/features/offline"

RETRIEVAL_MODES = ("offline", "online")


class FeatureStoreClient:
    """Async client returning {"<entity_id_1>:<entity_id_2>": feature_vector}."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 120,
        batch_size: int = 100,
        entity_fields: Optional[EntityFieldLookup] = None,
        service_user: str = "svc-model-training",
        service_roles: str = "PLATFORM_ADMIN",
        max_attempts: int = 3,
        retry_backoff_seconds: float = 2.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        retrieval_mode: str = "offline",
        feature_version: str = "v2.0.0",
    ):
        if retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(f"retrieval_mode must be one of {RETRIEVAL_MODES}, got {retrieval_mode!r}")
        self._retrieval_mode = retrieval_mode
        self._feature_version = feature_version
        # Feature ordering reported by the last offline call, for parity assertions.
        self.last_feature_names: Optional[List[str]] = None
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._batch_size = batch_size
        self._entity_fields = entity_fields
        self._service_user = service_user
        self._service_roles = service_roles
        self._max_attempts = max_attempts
        self._retry_backoff = retry_backoff_seconds
        self._transport = transport

    @classmethod
    def from_settings(cls) -> "FeatureStoreClient":
        entity_fields = None
        if settings.SYNTHETIC_DATA_DIR:
            # Development/test only; the reader enforces the environment and
            # provenance gates before returning anything.
            from src.adapters.synthetic_data import load_synthetic_entity_fields

            catalog = load_synthetic_entity_fields(settings.SYNTHETIC_DATA_DIR)
            entity_fields = catalog.get
            logger.warning(
                f"Feature Store client: SYNTHETIC entity catalog with {len(catalog)} records"
            )
        return cls(
            base_url=settings.FEATURE_STORE_URL,
            timeout_seconds=settings.FEATURE_STORE_TIMEOUT_SECONDS,
            batch_size=settings.FEATURE_STORE_BATCH_SIZE,
            entity_fields=entity_fields,
            service_user=settings.FEATURE_STORE_SERVICE_USER,
            service_roles=settings.FEATURE_STORE_SERVICE_ROLES,
            retrieval_mode=settings.FEATURE_STORE_RETRIEVAL_MODE,
            feature_version=settings.FEATURE_CATALOG_VERSION,
        )

    def _headers(self, tenant_id: str) -> Dict[str, str]:
        # Local stand-in for gateway-injected identity (see libs/xyz_security)
        return {
            "x-verified-tenant-id": tenant_id,
            "x-verified-user-id": self._service_user,
            "x-verified-roles": self._service_roles,
        }

    def _pair_payload(self, entity_id_1: str, entity_id_2: str) -> Dict:
        payload: Dict = {"entity_id_1": entity_id_1, "entity_id_2": entity_id_2}
        if self._entity_fields is not None:
            fields_1 = self._entity_fields(entity_id_1)
            fields_2 = self._entity_fields(entity_id_2)
            if fields_1:
                payload["entity1_fields"] = fields_1
            if fields_2:
                payload["entity2_fields"] = fields_2
        return payload

    async def _post_with_retry(
        self, client: httpx.AsyncClient, body: Dict, tenant_id: str, endpoint: str = BATCH_ENDPOINT
    ) -> httpx.Response:
        """
        POST one chunk. Timeouts, connection errors and 5xx responses are retried with a
        growing pause (the first requests can be slow while the Feature Store warms up);
        4xx responses are not retried.
        """
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = await client.post(endpoint, json=body, headers=self._headers(tenant_id))
                if response.status_code < 500 or attempt == self._max_attempts:
                    response.raise_for_status()
                    return response
                reason = f"HTTP {response.status_code}"
            except httpx.TransportError as e:
                if attempt == self._max_attempts:
                    raise
                reason = type(e).__name__
            logger.warning(
                f"Feature Store {endpoint} attempt {attempt}/{self._max_attempts} failed ({reason}); retrying"
            )
            await asyncio.sleep(self._retry_backoff * attempt)
        raise AssertionError("unreachable")

    async def get_offline_features(
        self,
        entity_pairs: Sequence[Tuple[str, str]],
        tenant_id: str,
        as_of_timestamp: Optional[datetime] = None,
    ) -> Dict[str, List[float]]:
        """
        Fetch 50-dim feature vectors keyed "<entity_id_1>:<entity_id_2>".

        Dispatches on retrieval_mode. Pairs the Feature Store has no features for are
        omitted from the result — the caller is responsible for deciding whether the
        resulting coverage is acceptable (see FeatureExtractionStage).
        Raises httpx.HTTPError if a request still fails after retries.
        """
        if self._retrieval_mode == "offline":
            results = await self._fetch_offline(entity_pairs, tenant_id, as_of_timestamp)
        else:
            results = await self._fetch_online(entity_pairs, tenant_id)

        missing = len(entity_pairs) - len(results)
        if missing:
            logger.warning(
                f"Feature Store ({self._retrieval_mode}) returned no features for "
                f"{missing}/{len(entity_pairs)} pairs"
            )
        return results

    async def _fetch_offline(
        self,
        entity_pairs: Sequence[Tuple[str, str]],
        tenant_id: str,
        as_of_timestamp: Optional[datetime],
    ) -> Dict[str, List[float]]:
        """
        Point-in-time retrieval via POST /api/v1/features/offline.

        `as_of_timestamp` is the point-in-time bound and is REQUIRED: without it the
        store would return present-day features for historical labels, which is exactly
        the leakage this endpoint exists to prevent.
        """
        if as_of_timestamp is None:
            raise ValueError(
                "as_of_timestamp is required for point-in-time retrieval. "
                "Pass the training run's cutoff, or use retrieval_mode='online' "
                "to explicitly accept current (non-point-in-time) features."
            )

        results: Dict[str, List[float]] = {}
        async with httpx.AsyncClient(
            base_url=self._base_url, timeout=self._timeout, transport=self._transport,
        ) as client:
            for i in range(0, len(entity_pairs), self._batch_size):
                chunk = list(entity_pairs[i:i + self._batch_size])
                body = {
                    "entity_pairs": [[e1, e2] for e1, e2 in chunk],
                    "tenant_id": tenant_id,
                    "as_of_timestamp": as_of_timestamp.isoformat(),
                    "feature_version": self._feature_version,
                }
                response = await self._post_with_retry(client, body, tenant_id, OFFLINE_ENDPOINT)
                payload = response.json()

                names = payload.get("feature_names") or None
                if names:
                    if self.last_feature_names and names != self.last_feature_names:
                        raise ValueError(
                            "Feature Store changed feature ordering mid-run — "
                            "training/serving parity cannot be guaranteed."
                        )
                    self.last_feature_names = names

                for row in payload.get("results", []):
                    results[f'{row["entity_id_1"]}:{row["entity_id_2"]}'] = row["feature_vector"]

        return results

    async def _fetch_online(
        self,
        entity_pairs: Sequence[Tuple[str, str]],
        tenant_id: str,
    ) -> Dict[str, List[float]]:
        """
        Current-value retrieval via POST /api/v1/features/batch. NOT point-in-time:
        returns features as they are now, computing on the fly for cache misses when
        entity fields are available.
        """
        results: Dict[str, List[float]] = {}
        async with httpx.AsyncClient(
            base_url=self._base_url, timeout=self._timeout, transport=self._transport,
        ) as client:
            for i in range(0, len(entity_pairs), self._batch_size):
                chunk = list(entity_pairs[i:i + self._batch_size])
                body = {
                    "tenant_id": tenant_id,
                    "include_entity_data": self._entity_fields is not None,
                    "pairs": [self._pair_payload(e1, e2) for e1, e2 in chunk],
                }
                response = await self._post_with_retry(client, body, tenant_id, BATCH_ENDPOINT)
                for (e1, e2), item in zip(chunk, response.json()["results"]):
                    if item is not None:
                        results[f"{e1}:{e2}"] = item["feature_vector"]
        return results
