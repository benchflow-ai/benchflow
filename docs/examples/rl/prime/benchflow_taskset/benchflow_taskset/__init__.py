"""BenchFlow tasks as a Verifiers v1 taskset, with the env that runs them.

Verifiers loads a plugin by its id: ``--env.taskset.id benchflow-taskset`` imports
this package, whose exported ``Taskset`` and ``Env`` subclasses become the run's
taskset and its default env. (The id cannot be ``benchflow``: that would import
the BenchFlow SDK itself.)
"""

from benchflow_taskset.env import BenchFlowEnv, BenchFlowEnvConfig
from benchflow_taskset.taskset import (
    BenchFlowConfig,
    BenchFlowData,
    BenchFlowInfraError,
    BenchFlowTask,
    BenchFlowTaskConfig,
    BenchFlowTaskset,
)

__all__ = [
    "BenchFlowConfig",
    "BenchFlowData",
    "BenchFlowEnv",
    "BenchFlowEnvConfig",
    "BenchFlowInfraError",
    "BenchFlowTask",
    "BenchFlowTaskConfig",
    "BenchFlowTaskset",
]
