"""Framework adapters for BenchFlow — the edges (architecture.md, capability #8).

Two directions, both pure format translators (no external SDK required):

**Outbound** — translate BenchFlow's ``Scene`` / ``Rubric`` / ``VerifyResult``
/ ``RewardEvent`` into the conventions of external eval frameworks:

* **Inspect AI** — ``InspectAdapter`` / ``to_inspect_task``
* **ORS (OpenReward)** — ``ORSAdapter`` / ``to_ors_reward``

**Inbound** — translate a foreign benchmark's task directory into BenchFlow's
native task format so the foreign benchmark runs natively:

* **Harbor** — ``HarborAdapter`` / ``from_harbor_task``
* **Terminal-Bench** — ``TerminalBenchAdapter`` / ``from_terminal_bench_task``
  (Terminal-Bench is also backward-compatible through the Harbor adapter,
  since Harbor is itself terminal-bench-derived).

``detect_adapter`` sniffs a task directory and returns the matching inbound
adapter; every inbound adapter returns an :class:`InboundTask`.

To add a new adapter, create a module under ``benchflow.adapters`` and
re-export its public symbols here.
"""

from benchflow.adapters.harbor import HarborAdapter, from_harbor_task
from benchflow.adapters.inbound import InboundTask, detect_adapter
from benchflow.adapters.inspect_ai import InspectAdapter, to_inspect_task
from benchflow.adapters.ors import ORSAdapter, to_ors_reward
from benchflow.adapters.terminal_bench import (
    TerminalBenchAdapter,
    from_terminal_bench_task,
)

__all__ = [
    "HarborAdapter",
    "InboundTask",
    "InspectAdapter",
    "ORSAdapter",
    "TerminalBenchAdapter",
    "detect_adapter",
    "from_harbor_task",
    "from_terminal_bench_task",
    "to_inspect_task",
    "to_ors_reward",
]
