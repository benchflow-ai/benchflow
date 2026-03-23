"""benchflow — ACP-native agent benchmarking framework.

Superset of Harbor. Re-exports Harbor's full API and adds:
- ACP client for multi-turn agent communication
- Trajectory capture (HTTP proxy, OTel collector, ACP native)
- SDK for programmatic usage
"""

__version__ = "2.0.0"

# Re-export Harbor's public API
from harbor import *  # noqa: F401, F403

# benchflow's additions
from benchflow.acp.client import ACPClient  # noqa: F401
from benchflow.acp.session import ACPSession  # noqa: F401
from benchflow.trajectories.otel import OTelCollector  # noqa: F401
from benchflow.trajectories.proxy import TrajectoryProxy  # noqa: F401
from benchflow.trajectories.types import Trajectory  # noqa: F401
