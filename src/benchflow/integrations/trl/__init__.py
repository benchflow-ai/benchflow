"""TRL integration surface for online GRPO training."""

from benchflow.integrations.trl.spec import (
    BashHarnessConfig,
    BenchFlowOptionalDependencyError,
    BenchFlowRuntimeEnvironment,
    BenchFlowSpec,
    BenchFlowSpecConfig,
    bash_tool_schemas,
    benchflow_environment_reward,
    finish_rollout,
    rollout_record,
    write_rollout_record,
)

__all__ = [
    "BashHarnessConfig",
    "BenchFlowOptionalDependencyError",
    "BenchFlowRuntimeEnvironment",
    "BenchFlowSpec",
    "BenchFlowSpecConfig",
    "bash_tool_schemas",
    "benchflow_environment_reward",
    "finish_rollout",
    "rollout_record",
    "write_rollout_record",
]
