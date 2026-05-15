"""benchflow — ACP-native agent benchmarking framework.

Public API for rollout-based agent evaluation.
"""

from importlib.metadata import version as _version

__version__ = _version("benchflow")

from benchflow._env_setup import stage_dockerfile_deps
from benchflow._scene import MailboxTransport, Message, MessageTransport
from benchflow._snapshot import list_snapshots, restore, snapshot
from benchflow.acp.client import ACPClient
from benchflow.acp.session import ACPSession
from benchflow.agents.registry import (
    AGENTS,
    AgentCapability,
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
from benchflow.job import Job, JobConfig, JobResult, RetryConfig
from benchflow.metrics import BenchmarkMetrics, collect_metrics
from benchflow.models import AgentInstallError, AgentTimeoutError, RunResult
from benchflow.rollouts import Role, RolloutConfig, RolloutResult, Scene, Turn
from benchflow.rollouts.runner import run
from benchflow.rollouts.yaml import rollout_config_from_yaml
from benchflow.sandboxes import ExecResult, SandboxSpec
from benchflow.skills import SkillInfo, discover_skills, install_skill, parse_skill
from benchflow.tasks import Task, TaskConfig
from benchflow.trajectories.otel import OTelCollector
from benchflow.trajectories.proxy import TrajectoryProxy
from benchflow.trajectories.types import Trajectory
from benchflow.trial_yaml import trial_config_from_yaml
from benchflow.user import BaseUser, FunctionUser, PassthroughUser, RoundResult

# Public API surface. Anything not in this list is implementation detail and
# may change without notice. Names are grouped by source module to match the
# imports above and to make it obvious to a future agent which module owns
# what.
__all__ = [
    "__version__",
    "ExecResult",
    "Task",
    "TaskConfig",
    # ACP
    "ACPClient",
    "ACPSession",
    # Agent registry
    "AGENTS",
    "AgentCapability",
    "get_agent",
    "infer_env_key_for_model",
    "is_vertex_model",
    "list_agents",
    "register_agent",
    # Job orchestration
    "Job",
    "JobConfig",
    "JobResult",
    "RetryConfig",
    # Metrics
    "BenchmarkMetrics",
    "collect_metrics",
    # Models / errors
    "AgentInstallError",
    "AgentTimeoutError",
    "RunResult",
    "RolloutConfig",
    "RolloutResult",
    "SandboxSpec",
    "run",
    # Multi-agent scene
    "Scene",
    "Role",
    "Message",
    "MessageTransport",
    "MailboxTransport",
    # Env snapshots
    "snapshot",
    "restore",
    "list_snapshots",
    "Turn",
    "trial_config_from_yaml",
    "rollout_config_from_yaml",
    # User abstraction (progressive disclosure)
    "BaseUser",
    "FunctionUser",
    "PassthroughUser",
    "RoundResult",
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
]
