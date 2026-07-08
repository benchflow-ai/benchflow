"""benchflow — the universal environment framework for agent benchmarks.

Public API surface:
- Sandbox protocol for isolated execution environments
- ACP client for multi-turn agent communication
- Trajectory capture (LiteLLM callbacks, OTel collector, ACP native)
- Rollout lifecycle for single-task execution
- Evaluation orchestration with retries and concurrency
- Rewards protocol (composable Rubric + RewardFunc)
- Metrics collection and aggregation
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("benchflow")
except PackageNotFoundError:
    __version__ = "0+unknown"

from benchflow._types import Role, Scene, Turn
from benchflow._utils.yaml_loader import rollout_config_from_yaml
from benchflow.acp.client import ACPClient
from benchflow.acp.session import ACPSession
from benchflow.adapters import (
    InspectAdapter,
    ORSAdapter,
    ors_tool_outputs_to_reward_events,
    to_inspect_task,
    to_ors_reward,
    write_ors_tool_outputs_jsonl,
)
from benchflow.agents.registry import (
    AGENTS,
    get_agent,
    infer_env_key_for_model,
    is_vertex_model,
    list_agents,
    register_agent,
)
from benchflow.contracts.user import (
    BaseUser,
    DocumentNudgeUser,
    FunctionUser,
    ModelDocumentNudgeUser,
    PassthroughUser,
    RoundResult,
)
from benchflow.evaluation import (
    Evaluation,
    EvaluationConfig,
    EvaluationResult,
    RetryConfig,
)
from benchflow.metrics import BenchmarkMetrics, collect_metrics
from benchflow.models import AgentInstallError, AgentTimeoutError, RolloutResult
from benchflow.monitor import (
    Monitor,
    MonitorConfig,
    MonitorNotImplementedError,
    MonitorResult,
)

# Rewards plane. Reward is the canonical node-based contract
# (``score(node) -> VerifyResult``); RewardFunc is the legacy path-based shape
# (``score(rollout_dir) -> float``) adapted into Reward via PathReward.
from benchflow.rewards import (
    CodeExecRewardFunc,
    Criterion,
    JudgeConfig,
    LLMJudgeRewardFunc,
    PathReward,
    Reward,
    RewardEvent,
    RewardFunc,
    Rubric,
    RubricConfig,
    ScoringConfig,
    StringMatchRewardFunc,
    TestRewardFunc,
    VerifyResult,
    load_rubric,
    load_rubric_json,
    load_rubric_toml,
)
from benchflow.rollout import (
    BashToolResult,
    Rollout,
    RolloutConfig,
    TaskRuntime,
    TaskRuntimeConfig,
    TaskRuntimeResult,
)
from benchflow.runtime import (
    Agent,
    Environment,
    Runtime,
    RuntimeConfig,
    RuntimeResult,
    run,
)  # bf.run() — supports Agent, RolloutConfig, and str calling conventions
from benchflow.sandbox import (
    SERVICES,
    ImageBuilder,
    ImageConfig,
    ImageRef,
    Sandbox,
    SandboxImage,
    SandboxSnapshotNotSupported,
    build_service_hooks,
    detect_services_from_dockerfile,
    register_service,
)

# Sandbox protocol (v0.4)
from benchflow.sandbox import ExecResult as SandboxExecResult
from benchflow.sandbox.protocol import ExecResult
from benchflow.sandbox.setup import stage_dockerfile_deps
from benchflow.sandbox.snapshot import (
    list_snapshots,
    list_workspace_snapshots,
    restore,
    snapshot,
    workspace_restore,
    workspace_snapshot,
)
from benchflow.scenes import compile_scenes_to_steps
from benchflow.sdk import SDK
from benchflow.skills import SkillInfo, discover_skills, install_skill, parse_skill
from benchflow.task import (
    TASK_DOCUMENT_FILENAME,
    Task,
    TaskConfig,
    TaskDocument,
    TaskDocumentParseError,
    Verifier,
    VerifierResult,
    render_task_md_from_legacy,
)
from benchflow.trajectories.types import Trajectory

# Public API surface. Anything not in this list is implementation detail and
# may change without notice.
__all__ = [
    "__version__",
    "Reward",
    "Rubric",
    "RewardFunc",
    "RewardEvent",
    "PathReward",
    "VerifyResult",
    "TestRewardFunc",
    "LLMJudgeRewardFunc",
    "StringMatchRewardFunc",
    "CodeExecRewardFunc",
    "Criterion",
    "JudgeConfig",
    "RubricConfig",
    "ScoringConfig",
    "load_rubric",
    "load_rubric_json",
    "load_rubric_toml",
    "Sandbox",
    "SandboxExecResult",
    "SandboxImage",
    "SandboxSnapshotNotSupported",
    "ImageBuilder",
    "ImageConfig",
    "ImageRef",
    "ExecResult",
    "Task",
    "TaskConfig",
    "TASK_DOCUMENT_FILENAME",
    "TaskDocument",
    "TaskDocumentParseError",
    "render_task_md_from_legacy",
    "Verifier",
    "VerifierResult",
    "ACPClient",
    "ACPSession",
    "AGENTS",
    "get_agent",
    "infer_env_key_for_model",
    "is_vertex_model",
    "list_agents",
    "register_agent",
    "Evaluation",
    "EvaluationConfig",
    "EvaluationResult",
    "RetryConfig",
    "BenchmarkMetrics",
    "collect_metrics",
    "AgentInstallError",
    "AgentTimeoutError",
    "RolloutResult",
    # Monitor mode — scaffolded API surface (#386)
    "Monitor",
    "MonitorConfig",
    "MonitorResult",
    "MonitorNotImplementedError",
    "Agent",
    "Environment",
    "Runtime",
    "RuntimeConfig",
    "RuntimeResult",
    "run",
    "Role",
    "Scene",
    "Turn",
    "compile_scenes_to_steps",
    # Workspace snapshots (filesystem helper — NOT the Sandbox primitive, #384)
    "workspace_snapshot",
    "workspace_restore",
    "list_workspace_snapshots",
    # Backward-compatible aliases for the above (pre-#384 names)
    "snapshot",
    "restore",
    "list_snapshots",
    "Rollout",
    "RolloutConfig",
    "BashToolResult",
    "TaskRuntime",
    "TaskRuntimeConfig",
    "TaskRuntimeResult",
    "rollout_config_from_yaml",
    "BaseUser",
    "DocumentNudgeUser",
    "FunctionUser",
    "ModelDocumentNudgeUser",
    "PassthroughUser",
    "RoundResult",
    "SDK",
    "SERVICES",
    "build_service_hooks",
    "detect_services_from_dockerfile",
    "register_service",
    "stage_dockerfile_deps",
    "SkillInfo",
    "discover_skills",
    "install_skill",
    "parse_skill",
    "Trajectory",
    "InspectAdapter",
    "ORSAdapter",
    "ors_tool_outputs_to_reward_events",
    "to_inspect_task",
    "to_ors_reward",
    "write_ors_tool_outputs_jsonl",
]


def __getattr__(name: str):
    """Lazy submodule resolution."""
    import importlib

    try:
        return importlib.import_module(f"benchflow.{name}")
    except ModuleNotFoundError as e:
        if e.name != f"benchflow.{name}":
            raise
    raise AttributeError(f"module 'benchflow' has no attribute {name!r}")
