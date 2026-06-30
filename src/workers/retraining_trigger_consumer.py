"""
model-training-pipeline/src/workers/retraining_trigger_consumer.py

Kafka consumer: listens for retraining trigger events and launches
the training pipeline automatically.

Retraining Triggers:
  1. SCHEDULED     — Cron: every Sunday 2 AM UTC
  2. DRIFT_DETECTED — KL divergence > 0.1 from Feature Store drift worker
  3. NEW_LABELS    — > 10,000 new HITL labeled pairs accumulated
  4. MANUAL        — API call from ML team
  5. EMERGENCY     — F1 < 90% on production traffic
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaError

from src.core.config import settings
from src.domain.models import RetrainingTrigger, RetrainingTriggerEvent, TrainingRun

logger = logging.getLogger(__name__)


class RetrainingTriggerConsumer:
    """
    Listens on the mdm.retraining.triggers Kafka topic and
    launches a training pipeline run for each qualifying event.
    """

    DLQ_TOPIC_SUFFIX = ".dlq"
    MAX_RETRIES = 3

    def __init__(self, pipeline_launcher):
        self._launcher = pipeline_launcher
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._dlq_producer: Optional[AIOKafkaProducer] = None
        self._running = False
        self._active_run: Optional[asyncio.Task] = None

    async def start(self):
        self._consumer = AIOKafkaConsumer(
            settings.KAFKA_RETRAINING_TOPIC,
            bootstrap_servers=settings.KAFKA_BROKERS_LIST,
            group_id=settings.KAFKA_CONSUMER_GROUP,
            auto_offset_reset="latest",
            enable_auto_commit=False,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )
        await self._consumer.start()
        self._dlq_producer = AIOKafkaProducer(
            bootstrap_servers=settings.KAFKA_BROKERS_LIST,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        await self._dlq_producer.start()
        self._running = True
        logger.info(f"Retraining trigger consumer started on {settings.KAFKA_RETRAINING_TOPIC}")

    async def stop(self):
        self._running = False
        if self._consumer:
            await self._consumer.stop()
        if self._dlq_producer:
            await self._dlq_producer.stop()

    async def run(self):
        if not self._consumer:
            await self.start()

        async for message in self._consumer:
            if not self._running:
                break

            retry_count = 0
            last_exc = None
            try:
                event = self._parse_event(message.value)
                logger.info(
                    f"Received retraining trigger: {event.trigger_type.value} "
                    f"priority={event.priority}"
                )

                # Skip if a training run is already active (unless EMERGENCY)
                if self._active_run and not self._active_run.done():
                    if event.priority != "CRITICAL":
                        logger.info("Training run already active, skipping trigger")
                        await self._consumer.commit()
                        continue
                    else:
                        logger.warning("EMERGENCY trigger received — cancelling current run")
                        self._active_run.cancel()

                # Launch with retry (up to MAX_RETRIES)
                for attempt in range(1, self.MAX_RETRIES + 1):
                    try:
                        self._active_run = asyncio.create_task(
                            self._launcher.launch(event)
                        )
                        break
                    except Exception as exc:
                        last_exc = exc
                        wait = 2 ** attempt
                        logger.warning(
                            f"Trigger launch attempt {attempt}/{self.MAX_RETRIES} failed: {exc}. "
                            f"Retrying in {wait}s."
                        )
                        await asyncio.sleep(wait)
                else:
                    # All retries exhausted — send to DLQ
                    await self._send_to_dlq(message, last_exc)

                await self._consumer.commit()

            except Exception as e:
                logger.error(f"Error processing retraining trigger: {e}")
                await self._send_to_dlq(message, e)

    async def _send_to_dlq(self, message, exc: Exception):
        """Route unprocessable messages to the dead-letter topic."""
        dlq_topic = settings.KAFKA_RETRAINING_TOPIC + self.DLQ_TOPIC_SUFFIX
        try:
            payload = {
                "original_value": message.value,
                "error": str(exc),
                "topic": message.topic,
                "partition": message.partition,
                "offset": message.offset,
            }
            await self._dlq_producer.send_and_wait(dlq_topic, payload)
            logger.error(f"Message sent to DLQ {dlq_topic}: {exc}")
        except Exception as dlq_exc:
            logger.critical(f"DLQ send failed: {dlq_exc}")

    def _parse_event(self, data: dict) -> RetrainingTriggerEvent:
        try:
            trigger_type = RetrainingTrigger(data.get("trigger_type", "MANUAL"))
        except ValueError:
            trigger_type = RetrainingTrigger.MANUAL

        return RetrainingTriggerEvent(
            trigger_type=trigger_type,
            priority=data.get("priority", "NORMAL"),
            reason=data.get("reason", ""),
            drift_score=data.get("drift_score"),
            new_label_count=data.get("new_label_count"),
        )


class PipelineLauncher:
    """
    Launches training pipeline runs in response to triggers.
    Manages concurrency and deduplication.
    """

    def __init__(self, session_factory, feature_store_client=None):
        self._session_factory = session_factory
        self._feature_store_client = feature_store_client

    async def launch(self, event: RetrainingTriggerEvent) -> TrainingRun:
        """Start a training run for the given trigger event."""
        from src.pipeline.training_pipeline import TrainingPipeline

        async with self._session_factory() as session:
            pipeline = TrainingPipeline(
                db_session=session,
                feature_store_client=self._feature_store_client,
            )

            run = TrainingRun(
                trigger=event.trigger_type,
                triggered_by=f"{event.trigger_type.value}:{event.reason}",
            )

            logger.info(
                f"Launching training pipeline: run_id={run.run_id} "
                f"trigger={event.trigger_type.value}"
            )

            return await pipeline.run(run)


class ScheduledRetrainingWorker:
    """
    APScheduler-based cron worker for scheduled retraining.
    Emits a trigger event to Kafka on schedule.
    """

    def __init__(self, launcher: PipelineLauncher):
        self._launcher = launcher

    async def start(self):
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger

        scheduler = AsyncIOScheduler()

        # Weekly scheduled training (Sunday 2 AM UTC)
        scheduler.add_job(
            self._trigger_scheduled_training,
            CronTrigger.from_crontab(settings.RETRAINING_SCHEDULE),
            id="weekly_training",
            max_instances=1,
            coalesce=True,
        )

        scheduler.start()
        logger.info(
            f"Scheduled retraining worker started. "
            f"Schedule: {settings.RETRAINING_SCHEDULE}"
        )

        while True:
            await asyncio.sleep(3600)

    async def _trigger_scheduled_training(self):
        """Called by APScheduler on cron schedule."""
        logger.info("Scheduled training triggered")
        event = RetrainingTriggerEvent(
            trigger_type=RetrainingTrigger.SCHEDULED,
            priority="NORMAL",
            reason="weekly_scheduled_retraining",
        )
        await self._launcher.launch(event)
