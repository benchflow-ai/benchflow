"""benchflow — ACP-native agent benchmarking framework.

Re-exports environment APIs and adds:
- ACP client for multi-turn agent communication
- Trajectory capture (HTTP proxy, OTel collector, ACP native)
- Rollout lifecycle for single-task execution
- Evaluation orchestration with retries and concurrency
- Metrics collection and aggregation
"""

from importlib.metadata import version as _version

__version__ = _version("benchflow")

# Re-export Harbor's core types for downstream task authors
from harbor import (
    BaseAgent,
    BaseEnvironment,
    ExecResult,
    Task,
    TaskConfig,
    Verifier,
    VerifierResult,
)

# benchflow's additions
from benchflow._env_setup import stage_dockerfile_deps
from benchflow._scene import MailboxTransport, Message, MessageTransport, SceneRole
from benchflow._scene import Scene as SceneRuntime
from benchflow._snapshot import list_snapshots, restore, snapshot
from benchflow._types import Role, Scene, Turn
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
from benchflow.environments import (
    SERVICES,
    build_service_hooks,
    detect_services_from_dockerfile,
    register_service,
)
from benchflow.evaluation import (
    Evaluation,
    EvaluationConfig,
    EvaluationResult,
    RetryConfig,
)
from benchflow.metrics import BenchmarkMetrics, collect_metrics
from benchflow.models import AgentInstallError, AgentTimeoutError, RolloutResult

# Rewards protocol (v0.4 — composable Rubric + RewardFunc)
from benchflow.rewards import (
    CodeExecRewardFunc,
    Criterion,
    JudgeConfig,
    LLMJudgeRewardFunc,
    RewardEvent,
    RewardFunc,
    Rubric,
    RubricConfig,
    ScoringConfig,
    StringMatchRewardFunc,
    TestRewardFunc,
    VerifyResult,
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
)  # run is imported above

# Sandbox protocol (v0.4 — parallel types, Harbor not yet removed)
from benchflow.sandbox import ExecResult as SandboxExecResult
from benchflow.sandbox import ImageBuilder, ImageConfig, ImageRef, Sandbox
from benchflow.sdk import SDK
from benchflow.skills import SkillInfo, discover_skills, install_skill, parse_skill
from benchflow.trajectories.otel import OTelCollector
from benchflow.trajectories.proxy import TrajectoryProxy
from benchflow.trajectories.types import Trajectory
from benchflow.trial_yaml import trial_config_from_yaml
from benchflow.user import BaseUser, FunctionUser, PassthroughUser, RoundResult

# Backward-compat aliases
Trial = Rollout
TrialConfig = RolloutConfig
TrialRole = Role
TrialScene = Scene
RunResult = RolloutResult
Job = Evaluation
JobConfig = EvaluationConfig
JobResult = EvaluationResult

# Public API surface. Anything not in this list is implementation detail and
# may change without notice. Names are grouped by source module to match the
# imports above and to make it obvious to a future agent which module owns
# what.
__all__ = [
    "__version__",
    # Rewards protocol (v0.4)
    "Rubric",
    "RewardFunc",
    "RewardEvent",
    "VerifyResult",
    "TestRewardFunc",
    "LLMJudgeRewardFunc",
    "StringMatchRewardFunc",
    "CodeExecRewardFunc",
    # Rubric config (ENG-55)
    "Criterion",
    "JudgeConfig",
    "RubricConfig",
    "ScoringConfig",
    "load_rubric_toml",
    # Sandbox protocol (v0.4)
    "Sandbox",
    "SandboxExecResult",
    "ImageBuilder",
    "ImageConfig",
    "ImageRef",
    # Harbor re-exports
    "BaseAgent",
    "BaseEnvironment",
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
    # Evaluation orchestration (new names)
    "Evaluation",
    "EvaluationConfig",
    "EvaluationResult",
    "RetryConfig",
    # Backward-compat aliases for Job
    "Job",
    "JobConfig",
    "JobResult",
    # Metrics
    "BenchmarkMetrics",
    "collect_metrics",
    # Models / errors
    "AgentInstallError",
    "AgentTimeoutError",
    "RolloutResult",
    "RunResult",
    # Runtime (0.3 compat)
    "Agent",
    "Environment",
    "Runtime",
    "RuntimeConfig",
    "RuntimeResult",
    # Single entry point
    "run",
    # Canonical declarative types (_types.py — ENG-47)
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
    # Rollout (single execution path — ENG-46)
    "Rollout",
    "RolloutConfig",
    # Backward-compat aliases for Trial
    "Trial",
    "TrialConfig",
    "TrialRole",
    "TrialScene",
    "trial_config_from_yaml",
    # User abstraction (progressive disclosure)
    "BaseUser",
    "FunctionUser",
    "PassthroughUser",
    "RoundResult",
    # SDK (backwards compat)
    "SDK",
    # Environments / dep staging
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
    # External adapters (ENG-51)
    "InspectAdapter",
    "ORSAdapter",
    "to_inspect_task",
    "to_ors_reward",
]


def __getattr__(name: str):
    """Fall through to harbor for names not explicitly re-exported."""
    # Let Python's normal submodule resolution handle subpackages first.
    import importlib

    try:
        return importlib.import_module(f"benchflow.{name}")
    except ModuleNotFoundError as e:
        if e.name != f"benchflow.{name}":
            raise

    import harbor

    if hasattr(harbor, name):
        import warnings

        warnings.warn(
            f"'{name}' is not directly re-exported by benchflow. Use 'from harbor import {name}' instead.",
            ImportWarning,
            stacklevel=2,
        )
        return getattr(harbor, name)
    raise AttributeError(f"module 'benchflow' has no attribute {name!r}")
