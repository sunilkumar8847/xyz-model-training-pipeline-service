"""
Drift detection worker — monitors feature distribution and model F1 degradation.

LLD §9 requirements:
  - KL divergence > 0.1 on any feature group → emit DRIFT_DETECTED trigger
  - Rolling F1 drops > 3% below champion → emit DRIFT_DETECTED trigger (EMERGENCY)
  - Check interval: DRIFT_CHECK_INTERVAL_SECONDS (default 3600s)

Architecture:
  Feature drift stats come from the Feature Store drift endpoint.
  F1 degradation comes from the Prometheus metrics API queried via httpx.
  On threshold breach, emit a Kafka message to mdm.retraining.triggers.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
from datetime import datetime
from typing import Optional

import httpx
from aiokafka import AIOKafkaProducer

from src.core.config import settings
from src.core.metrics import DRIFT_SCORE

logger = logging.getLogger(__name__)

# KL divergence threshold for triggering retraining
KL_THRESHOLD = 0.1
# F1 absolute drop from champion that triggers emergency retraining
F1_DROP_THRESHOLD = 0.03


class DriftDetector:
    """
    Periodically checks feature distribution drift (KL divergence) and
    model F1 degradation, and emits a DRIFT_DETECTED retraining trigger
    to Kafka when thresholds are breached.
    """

    def __init__(self):
        self._producer: Optional[AIOKafkaProducer] = None
        self._running = False
        self._http: Optional[httpx.AsyncClient] = None

    async def start(self) -> None:
        self._producer = AIOKafkaProducer(
            bootstrap_servers=settings.KAFKA_BROKERS_LIST,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        await self._producer.start()
        self._http = httpx.AsyncClient(timeout=10.0)
        self._running = True
        logger.info("DriftDetector started (interval=%ds)", settings.DRIFT_CHECK_INTERVAL_SECONDS)

    async def stop(self) -> None:
        self._running = False
        if self._producer:
            await self._producer.stop()
        if self._http:
            await self._http.aclose()

    async def run(self) -> None:
        """Main loop — runs until stop() is called."""
        if not self._running:
            await self.start()

        while self._running:
            try:
                await self._check_once()
            except Exception as exc:
                logger.error("Drift check failed: %s", exc, exc_info=True)

            await asyncio.sleep(settings.DRIFT_CHECK_INTERVAL_SECONDS)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _check_once(self) -> None:
        kl_score, drifted_features = await self._check_feature_drift()
        f1_drop = await self._check_f1_degradation()

        if kl_score is not None:
            DRIFT_SCORE.labels(feature_group="ensemble").set(kl_score)

        if kl_score is not None and kl_score > KL_THRESHOLD:
            logger.warning(
                "Feature drift detected: KL=%.4f > %.4f (features: %s)",
                kl_score, KL_THRESHOLD, drifted_features,
            )
            await self._emit_trigger(
                trigger_type="DRIFT_DETECTED",
                priority="HIGH",
                drift_score=kl_score,
                reason=(
                    f"Feature distribution drift: KL={kl_score:.4f} "
                    f"(threshold={KL_THRESHOLD}). "
                    f"Drifted features: {', '.join(drifted_features[:5])}"
                ),
            )

        if f1_drop is not None and f1_drop > F1_DROP_THRESHOLD:
            logger.warning(
                "F1 degradation detected: drop=%.4f > %.4f",
                f1_drop, F1_DROP_THRESHOLD,
            )
            await self._emit_trigger(
                trigger_type="DRIFT_DETECTED",
                priority="CRITICAL",
                drift_score=f1_drop,
                reason=(
                    f"Model F1 degradation: {f1_drop:.4f} absolute drop "
                    f"below champion (threshold={F1_DROP_THRESHOLD})"
                ),
            )

    async def _check_feature_drift(self) -> tuple[Optional[float], list[str]]:
        """
        Query Feature Store drift endpoint.
        Returns (max_kl_divergence, list_of_drifted_feature_names).
        """
        try:
            resp = await self._http.get(
                f"{settings.FEATURE_STORE_URL}/v1/drift/statistics",
                params={"lookback_hours": settings.DRIFT_CHECK_INTERVAL_SECONDS // 3600},
            )
            resp.raise_for_status()
            data = resp.json()

            feature_kl: dict[str, float] = data.get("kl_divergence", {})
            if not feature_kl:
                return None, []

            drifted = [name for name, kl in feature_kl.items() if kl > KL_THRESHOLD]
            max_kl = max(feature_kl.values())
            return max_kl, drifted

        except httpx.HTTPStatusError as exc:
            logger.debug("Feature Store drift endpoint returned %s", exc.response.status_code)
            return None, []
        except Exception as exc:
            logger.debug("Could not fetch drift statistics: %s", exc)
            return None, []

    async def _check_f1_degradation(self) -> Optional[float]:
        """
        Query Prometheus for the current rolling F1 of the production model
        and compare against the champion F1 recorded at promotion time.
        Returns the absolute F1 drop (positive value) or None if unavailable.
        """
        try:
            # Try to get current F1 from Prometheus via the Model Inference Service
            resp = await self._http.get(
                f"{settings.MODEL_INFERENCE_SERVICE_URL}/metrics",
            )
            resp.raise_for_status()

            # Parse Prometheus text format for the model_f1_score gauge
            current_f1: Optional[float] = None
            champion_f1: Optional[float] = None

            for line in resp.text.splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                if "model_f1_score" in line and "production" in line:
                    try:
                        current_f1 = float(line.split()[-1])
                    except ValueError:
                        pass
                if "model_f1_score" in line and "champion" in line:
                    try:
                        champion_f1 = float(line.split()[-1])
                    except ValueError:
                        pass

            if current_f1 is not None and champion_f1 is not None and champion_f1 > 0:
                drop = champion_f1 - current_f1
                return drop if drop > 0 else None

            return None

        except Exception as exc:
            logger.debug("Could not fetch F1 from inference service metrics: %s", exc)
            return None

    async def _emit_trigger(
        self,
        trigger_type: str,
        priority: str,
        drift_score: float,
        reason: str,
    ) -> None:
        """Publish a retraining trigger event to Kafka."""
        if self._producer is None:
            logger.warning("DriftDetector: Kafka producer not started, cannot emit trigger")
            return

        payload = {
            "trigger_type": trigger_type,
            "priority": priority,
            "drift_score": drift_score,
            "reason": reason,
            "emitted_at": datetime.utcnow().isoformat(),
        }

        try:
            await self._producer.send_and_wait(settings.KAFKA_RETRAINING_TOPIC, payload)
            logger.info(
                "Drift trigger emitted to %s: type=%s priority=%s score=%.4f",
                settings.KAFKA_RETRAINING_TOPIC, trigger_type, priority, drift_score,
            )
        except Exception as exc:
            logger.error("Failed to emit drift trigger to Kafka: %s", exc)


def _kl_divergence(p: list[float], q: list[float], epsilon: float = 1e-10) -> float:
    """
    Compute KL divergence KL(P || Q) between two discrete distributions.
    Epsilon prevents log(0). Both lists must have the same length and sum to 1.
    """
    total = 0.0
    for pi, qi in zip(p, q):
        pi = max(pi, epsilon)
        qi = max(qi, epsilon)
        total += pi * math.log(pi / qi)
    return total
