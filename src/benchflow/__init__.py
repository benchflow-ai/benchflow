"""benchflow — ACP-native agent benchmarking framework.

Public API surface:
- Sandbox protocol for isolated execution environments
- ACP client for multi-turn agent communication
- Trajectory capture (HTTP proxy, OTel collector, ACP native)
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

# Core types
from benchflow._types import Role, Scene, Turn
from benchflow._utils.yaml_loader import rollout_config_from_yaml
from benchflow.acp.client import ACPClient
from benchflow.acp.session import ACPSession
from benchflow.adapters import (
    InspectAdapter,
    ORSAdapter,
    to_inspect_task,
    to_ors_reward,
)
from benchflow.agents.registry import (
    AGENTS,
    get_agent,
    infer_env_key_for_model,
    is_vertex_model,
    list_agents,
    register_agent,
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
from benchflow.rollout import Rollout, RolloutConfig
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
    build_service_hooks,
    detect_services_from_dockerfile,
    register_service,
)

# Sandbox protocol (v0.4)
from benchflow.sandbox import ExecResult as SandboxExecResult
from benchflow.sandbox.protocol import ExecResult
from benchflow.sandbox.setup import stage_dockerfile_deps
from benchflow.sandbox.snapshot import list_snapshots, restore, snapshot
from benchflow.sandbox.user import BaseUser, FunctionUser, PassthroughUser, RoundResult
from benchflow.scenes import MailboxTransport, Message, MessageTransport, SceneRole
from benchflow.scenes import Scene as SceneRuntime
from benchflow.sdk import SDK
from benchflow.skills import SkillInfo, discover_skills, install_skill, parse_skill
from benchflow.task import (
    Task,
    TaskConfig,
    Verifier,
    VerifierResult,
)
from benchflow.trajectories.otel import OTelCollector
from benchflow.trajectories.proxy import TrajectoryProxy
from benchflow.trajectories.types import Trajectory

# Public API surface. Anything not in this list is implementation detail and
# may change without notice.
__all__ = [
    "__version__",
    # Rewards plane
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
    # Sandbox protocol
    "Sandbox",
    "SandboxExecResult",
    "ImageBuilder",
    "ImageConfig",
    "ImageRef",
    "ExecResult",
    "Task",
    "TaskConfig",
    "Verifier",
    "VerifierResult",
    # ACP
    "ACPClient",
    "ACPSession",
    # Agent registry
    "AGENTS",
    "get_agent",
    "infer_env_key_for_model",
    "is_vertex_model",
    "list_agents",
    "register_agent",
    # Evaluation orchestration
    "Evaluation",
    "EvaluationConfig",
    "EvaluationResult",
    "RetryConfig",
    # Metrics
    "BenchmarkMetrics",
    "collect_metrics",
    # Models / errors
    "AgentInstallError",
    "AgentTimeoutError",
    "RolloutResult",
    # Monitor mode — scaffolded API surface (#386)
    "Monitor",
    "MonitorConfig",
    "MonitorResult",
    "MonitorNotImplementedError",
    # Runtime
    "Agent",
    "Environment",
    "Runtime",
    "RuntimeConfig",
    "RuntimeResult",
    # Single entry point
    "run",
    # Declarative types
    "Role",
    "Scene",
    "Turn",
    # Multi-agent scene runtime
    "SceneRole",
    "SceneRuntime",
    "Message",
    "MessageTransport",
    "MailboxTransport",
    # Env snapshots
    "snapshot",
    "restore",
    "list_snapshots",
    # Rollout
    "Rollout",
    "RolloutConfig",
    "rollout_config_from_yaml",
    # User abstraction (progressive disclosure)
    "BaseUser",
    "FunctionUser",
    "PassthroughUser",
    "RoundResult",
    # SDK
    "SDK",
    # Sandbox services
    "SERVICES",
    "build_service_hooks",
    "detect_services_from_dockerfile",
    "register_service",
    "stage_dockerfile_deps",
    # Skills
    "SkillInfo",
    "discover_skills",
    "install_skill",
    "parse_skill",
    # Trajectories
    "OTelCollector",
    "TrajectoryProxy",
    "Trajectory",
    # External adapters
    "InspectAdapter",
    "ORSAdapter",
    "to_inspect_task",
    "to_ors_reward",
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
