"""benchflow — ACP-native agent benchmarking framework.

The v0.4 public API is intentionally small. ``__all__`` exposes 8 names:

  Verbs:        run, run_batch
  Orchestrators: Trial, Job
  Configs:      TrialConfig, JobConfig
  Results:      TrialResult, JobResult

Power users discover the rest via explicit submodule imports — see
PLAN_V2_shaping §3.2:

  from benchflow.multi_agent     import Scene, Turn, Role
  from benchflow.agents          import register_agent, AGENTS, AgentConfig
  from benchflow.skill_registry  import SkillInfo, discover_skills, parse_skill
  from benchflow.errors          import AgentInstallError, AgentTimeoutError
  from benchflow.trajectories    import OTelCollector, TrajectoryProxy
  from benchflow.sandbox         import snapshot, restore, list_snapshots, SERVICES
"""

from importlib.metadata import version as _version

__version__ = _version("benchflow")

# Re-export benchflow's core sandbox/task types for downstream task authors
from benchflow.sandbox.build import stage_dockerfile_deps
from benchflow.acp.client import ACPClient
from benchflow.acp.session import ACPSession
from benchflow.agents.registry import (
    AGENTS,
    get_agent,
    infer_env_key_for_model,
    is_vertex_model,
    register_agent,
)
from benchflow.sandbox import (
    SERVICES,
    build_service_hooks,
)
from benchflow.job import Job, JobConfig, JobResult, RetryConfig
from benchflow.metrics import BenchmarkMetrics, collect_metrics
from benchflow.errors import AgentInstallError, AgentTimeoutError
from benchflow.results import TrialResult
from benchflow.api import (
    Agent,
    Environment,
    run,  # bf.run(agent, env) — the primary 0.3 API
    run_batch,  # bf.run_batch(tasks, agent, ...) — many-trial verb (v0.4)
)
from benchflow.sandbox.snapshot import list_snapshots, restore, snapshot
from benchflow.trial import Trial, TrialConfig
from benchflow._utils.yaml_loader import trial_config_from_yaml
from benchflow.skill_registry import SkillInfo, discover_skills, parse_skill
from benchflow.trajectories.otel import OTelCollector
from benchflow.trajectories.proxy import TrajectoryProxy

# Public API surface — 8 names + __version__. Everything else is
# implementation or accessed via explicit submodule imports (see module
# docstring above). v0.4 trim per PLAN_V2_shaping §3.1.
__all__ = [
    "__version__",
    # 2 verbs
    "run",
    "run_batch",
    # 2 orchestrators
    "Trial",
    "Job",
    # 2 configs (public for serialization/replay)
    "TrialConfig",
    "JobConfig",
    # 2 result types (Trial→TrialResult, Job→JobResult)
    "TrialResult",
    "JobResult",
]
