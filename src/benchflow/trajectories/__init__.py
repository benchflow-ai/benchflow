"""Trajectory capture — intercept and record agent ↔ LLM traffic.

Three capture strategies (pick one per run):
  - TrajectoryProxy: HTTP reverse-proxy, records request/response pairs.
  - OTelCollector: OpenTelemetry OTLP receiver, captures LLM call spans.
  - ACP-native: session updates from the ACP client (no extra infra).

The SDK currently uses the ACP-native path; the proxy and OTel paths
exist for agents that don't speak ACP or for cross-tool comparison.

Files
-----
- ``proxy.py``        ``TrajectoryProxy`` — async HTTP reverse-proxy that
                      records request/response pairs (streaming + non-
                      streaming). Used when the agent talks plain HTTP.
- ``otel.py``         ``OTelCollector`` — minimal OTLP receiver that
                      decodes LLM-call spans into ``LLMExchange``.
- ``types.py``        ``LLMRequest`` / ``LLMResponse`` / ``LLMExchange`` /
                      ``Trajectory`` pydantic models — the shared schema
                      both proxy and otel write into.
- ``atif.py``         **Backlog — not wired.** Agent-agnostic trajectory
                      interchange format. See CLAUDE.md "Later" section.
- ``claude_code.py``  **Backlog — not wired.** Converts Claude Code
                      stream-json output → ATIF. Depends on ``atif.py``.

ACP-native capture itself lives in ``benchflow/_trajectory.py`` (sibling
module, not in this package), since it consumes ACP session updates
rather than HTTP / OTLP traffic.
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
