"""Backward-compat shim — re-exports everything from benchflow.evaluation.

``from benchflow.job import Job, JobConfig`` keeps working.
"""

from benchflow.evaluation import *  # noqa: F403
from benchflow.evaluation import (
    DEFAULT_AGENT,
    Evaluation,
    EvaluationConfig,
    EvaluationResult,
    Job,
    JobConfig,
    JobResult,
    RetryConfig,
    effective_model,
)

__all__ = [
    "Evaluation",
    "EvaluationConfig",
    "EvaluationResult",
    "Job",
    "JobConfig",
    "JobResult",
    "RetryConfig",
    "DEFAULT_AGENT",
    "effective_model",
]
