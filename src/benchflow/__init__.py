"""benchflow — ACP-native agent benchmarking framework.

Superset of Harbor. Re-exports Harbor's full API and adds:
- ACP client for multi-turn agent communication
- Trajectory capture (HTTP proxy, OTel collector, ACP native)
- SDK for programmatic usage
- Job orchestration with retries and concurrency
- Metrics collection and aggregation
"""

__version__ = "2.0.0"

# Re-export Harbor's public API
from harbor import *  # noqa: F401, F403

# benchflow's additions
from benchflow.acp.client import ACPClient  # noqa: F401
from benchflow.acp.session import ACPSession  # noqa: F401
from benchflow.agents.registry import AGENTS, get_agent, list_agents, register_agent  # noqa: F401
from benchflow.job import Job, JobConfig, JobResult, RetryConfig  # noqa: F401
from benchflow.metrics import BenchmarkMetrics, collect_metrics  # noqa: F401
from benchflow.sdk import SDK, RunResult, stage_dockerfile_deps  # noqa: F401
from benchflow.skills import discover_skills, install_skill, parse_skill, SkillInfo  # noqa: F401
from benchflow.environments import SERVICES, detect_services_from_dockerfile, build_service_hooks, register_service  # noqa: F401
from benchflow.trajectories.otel import OTelCollector  # noqa: F401
from benchflow.trajectories.proxy import TrajectoryProxy  # noqa: F401
from benchflow.trajectories.types import Trajectory  # noqa: F401
