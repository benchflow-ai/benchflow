"""benchflow — ACP-native agent benchmarking framework.

Re-exports environment APIs and adds:
- ACP client for multi-turn agent communication
- Trajectory capture (HTTP proxy, OTel collector, ACP native)
- SDK for programmatic usage
- Job orchestration with retries and concurrency
- Metrics collection and aggregation
"""

__version__ = "2.0.0"

# Re-export Harbor's core types for downstream task authors
from harbor import (  # noqa: F401
    BaseAgent,
    BaseEnvironment,
    ExecResult,
    Task,
    TaskConfig,
    Trial,
    Verifier,
    VerifierResult,
)

# benchflow's additions
from benchflow.acp.client import ACPClient  # noqa: F401
from benchflow.acp.session import ACPSession  # noqa: F401
from benchflow.agents.registry import (  # noqa: F401
    AGENTS,
    get_agent,
    infer_env_key_for_model,
    is_vertex_model,
    list_agents,
    register_agent,
)
from benchflow.job import Job, JobConfig, JobResult, RetryConfig  # noqa: F401
from benchflow.metrics import BenchmarkMetrics, collect_metrics  # noqa: F401
from benchflow._env_setup import stage_dockerfile_deps  # noqa: F401
from benchflow._models import AgentInstallError, AgentTimeoutError, RunResult  # noqa: F401
from benchflow.sdk import SDK  # noqa: F401
from benchflow.skills import discover_skills, install_skill, parse_skill, SkillInfo  # noqa: F401
from benchflow.environments import (  # noqa: F401
    SERVICES,
    build_service_hooks,
    detect_services_from_dockerfile,
    register_service,
)
from benchflow.trajectories.otel import OTelCollector  # noqa: F401
from benchflow.trajectories.proxy import TrajectoryProxy  # noqa: F401
from benchflow.trajectories.types import Trajectory  # noqa: F401


def __getattr__(name: str):
    """Fall through to harbor for names not explicitly re-exported."""
    import harbor  # noqa: F811

    if hasattr(harbor, name):
        import warnings

        warnings.warn(
            f"'{name}' is not directly re-exported by benchflow. Use 'from harbor import {name}' instead.",
            ImportWarning,
            stacklevel=2,
        )
        return getattr(harbor, name)
    raise AttributeError(f"module 'benchflow' has no attribute {name!r}")
