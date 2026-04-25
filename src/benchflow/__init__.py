"""benchflow — ACP-native agent benchmarking framework.

Re-exports environment APIs and adds:
- ACP client for multi-turn agent communication
- Trajectory capture (HTTP proxy, OTel collector, ACP native)
- SDK for programmatic usage
- Job orchestration with retries and concurrency
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
from benchflow._scene import MailboxTransport, Message, MessageTransport, Role, Scene
from benchflow._snapshot import list_snapshots, restore, snapshot
from benchflow.acp.client import ACPClient
from benchflow.acp.session import ACPSession
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
from benchflow.job import Job, JobConfig, JobResult, RetryConfig
from benchflow.metrics import BenchmarkMetrics, collect_metrics
from benchflow.models import AgentInstallError, AgentTimeoutError, RunResult
from benchflow.runtime import (
    Agent,
    Environment,
    Runtime,
    RuntimeConfig,
    RuntimeResult,
    run,  # bf.run(agent, env) — the primary 0.3 API
)
from benchflow.sdk import SDK
from benchflow.skills import SkillInfo, discover_skills, install_skill, parse_skill
from benchflow.trajectories.otel import OTelCollector
from benchflow.trajectories.proxy import TrajectoryProxy
from benchflow.trajectories.types import Trajectory
from benchflow.trial import Role as TrialRole
from benchflow.trial import Scene as TrialScene
from benchflow.trial import Trial, TrialConfig, Turn
from benchflow.trial_yaml import trial_config_from_yaml

# Public API surface. Anything not in this list is implementation detail and
# may change without notice. Names are grouped by source module to match the
# imports above and to make it obvious to a future agent which module owns
# what.
__all__ = [
    "__version__",
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
    # Runtime (0.3 primary API)
    "Agent",
    "Environment",
    "Runtime",
    "RuntimeConfig",
    "RuntimeResult",
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
    # Trial (decomposed lifecycle)
    "Trial",
    "TrialConfig",
    "TrialRole",
    "TrialScene",
    "Turn",
    "trial_config_from_yaml",
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
]


def __getattr__(name: str):
    """Fall through to harbor for names not explicitly re-exported."""
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
