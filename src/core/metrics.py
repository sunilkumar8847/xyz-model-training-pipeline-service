"""
model-training-pipeline/src/core/metrics.py
Prometheus metrics matching LLD Section 8.1.
"""
from prometheus_client import Counter, Gauge, Histogram

TRAINING_DURATION = Histogram(
    "training_duration_seconds",
    "Training pipeline duration in seconds",
    ["model", "version"],
    buckets=[300, 600, 1800, 3600, 7200, 14400],
)
MODEL_F1_SCORE = Gauge("model_f1_score", "Model F1 score", ["model", "version"])
DRIFT_SCORE = Gauge("drift_score", "Feature drift score", ["drift_type"])
RETRAINING_TRIGGERED = Counter("retraining_triggered_total", "Retraining triggers", ["trigger_type"])
ROLLBACK_EVENTS = Counter("rollback_events_total", "Rollback events", ["reason"])
LABELED_PAIRS = Counter("labeled_pairs_total", "Labeled training pairs", ["source"])
ACTIVE_TRAINING_RUNS = Gauge("active_training_runs", "Currently running training jobs")
