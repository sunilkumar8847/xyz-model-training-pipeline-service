"""
model-training-pipeline/src/core/exceptions.py

Structured error codes matching LLD PART IX.
All API errors use the platform envelope: {code, message, correlation_id}.

Error codes:
  MT_1001 — Invalid training configuration
  MT_4001 — Evaluation F1 below the promotion gate
  MT_9001 — Compute / MLflow / infrastructure unavailable
"""
from __future__ import annotations


class TrainingPipelineError(Exception):
    """Base exception for all training pipeline errors."""

    def __init__(self, code: str, message: str, status_code: int = 400):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class InvalidTrainingConfigError(TrainingPipelineError):
    """MT_1001 — Invalid training configuration or parameters."""

    def __init__(self, message: str):
        super().__init__("MT_1001", message, 422)


class EvaluationGateFailedError(TrainingPipelineError):
    """MT_4001 — Model evaluation F1 below the promotion gate."""

    def __init__(self, f1: float, gate: float):
        super().__init__(
            "MT_4001",
            f"Evaluation F1 {f1:.4f} below the {gate:.2f} promotion gate",
            422,
        )


class InfrastructureUnavailableError(TrainingPipelineError):
    """MT_9001 — Compute, MLflow, or other infrastructure unavailable."""

    def __init__(self, service: str, detail: str = ""):
        msg = f"Infrastructure unavailable: {service}"
        if detail:
            msg += f" ({detail})"
        super().__init__("MT_9001", msg, 503)


class FeatureCoverageError(TrainingPipelineError):
    """
    MT_9001 — the Feature Store served features for too few pairs to train on.

    Raised instead of silently proceeding with a shrunken dataset: an unreachable
    Feature Store would otherwise produce an empty dataset, dummy models and an
    f1 of 0.0, all reported as a successful run.
    """

    def __init__(self, requested: int, retrieved: int, required_ratio: float):
        ratio = retrieved / requested if requested else 0.0
        super().__init__(
            "MT_9001",
            f"Feature Store coverage too low: {retrieved}/{requested} pairs "
            f"({ratio:.1%}) below the required {required_ratio:.0%}. "
            f"Check that the Feature Store is reachable and that features have been "
            f"materialized for these pairs at the requested point in time.",
            503,
        )
        self.requested = requested
        self.retrieved = retrieved


class ResourceNotFoundError(TrainingPipelineError):
    """Training run or model version not found."""

    def __init__(self, resource: str, identifier: str):
        super().__init__(
            "MT_4040",
            f"{resource} '{identifier}' not found",
            404,
        )


class InsufficientTrainingDataError(TrainingPipelineError):
    """MT_1001 — fewer labeled pairs than the environment requires."""

    def __init__(self, n_pairs: int, minimum: int):
        super().__init__(
            "MT_1001",
            f"Insufficient training data: {n_pairs} labeled pairs < required minimum "
            f"{minimum} (MIN_LABELED_PAIRS).",
            422,
        )
        self.n_pairs = n_pairs
        self.minimum = minimum
