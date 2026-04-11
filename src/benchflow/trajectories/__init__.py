"""Trajectory capture — intercept and record agent ↔ LLM traffic.

Three capture strategies (pick one per run):
  - TrajectoryProxy: HTTP reverse-proxy, records request/response pairs.
  - OTelCollector: OpenTelemetry OTLP receiver, captures LLM call spans.
  - ACP-native: session updates from the ACP client (no extra infra).

See also: _trajectory.py for post-hoc trajectory assembly, and
trajectories/atif.py for the ATIF interchange format (backlog).
"""

from .otel import OTelCollector
from .proxy import TrajectoryProxy
from .types import LLMExchange, LLMRequest, LLMResponse, Trajectory

__all__ = [
    "LLMExchange",
    "LLMRequest",
    "LLMResponse",
    "OTelCollector",
    "Trajectory",
    "TrajectoryProxy",
]
