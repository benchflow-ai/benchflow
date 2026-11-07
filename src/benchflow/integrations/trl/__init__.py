"""TRL integration surface for online GRPO training."""

from benchflow.integrations.trl.spec import (
    BashHarnessConfig,
    BenchFlowOptionalDependencyError,
    BenchFlowRuntimeEnvironment,
    BenchFlowSpec,
    BenchFlowSpecConfig,
    benchflow_environment_reward,
)

__all__ = [
    "BashHarnessConfig",
    "BenchFlowOptionalDependencyError",
    "BenchFlowRuntimeEnvironment",
    "BenchFlowSpec",
    "BenchFlowSpecConfig",
    "benchflow_environment_reward",
]
