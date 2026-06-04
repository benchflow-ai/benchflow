"""Trajectory capture and exchange schemas.

BenchFlow persists ACP-native trajectories plus LiteLLM callback-derived LLM
request/response exchanges. ``OTelCollector`` remains available for external
OpenTelemetry captures.

Files
-----
- ``otel.py``         ``OTelCollector`` — minimal OTLP receiver that
                      decodes LLM-call spans into ``LLMExchange``.
- ``types.py``        ``LLMRequest`` / ``LLMResponse`` / ``LLMExchange`` /
                      ``Trajectory`` pydantic models — the shared schema
                      LiteLLM callbacks and OTel write into.
ACP-native capture itself lives in ``benchflow/_trajectory.py`` (sibling
module, not in this package), since it consumes ACP session updates
rather than HTTP / OTLP traffic.
"""

from .otel import OTelCollector
from .tree import RolloutNode, RolloutTree, Step, branch_points, trajectory
from .types import LLMExchange, LLMRequest, LLMResponse, Trajectory

__all__ = [
    # raw LLM-traffic capture schema
    "LLMExchange",
    "LLMRequest",
    "LLMResponse",
    "OTelCollector",
    "Trajectory",
    # tree-native Rollout data model
    "RolloutNode",
    "RolloutTree",
    "Step",
    "branch_points",
    "trajectory",
]
