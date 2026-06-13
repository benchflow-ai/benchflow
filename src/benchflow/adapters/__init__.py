"""Framework adapters for BenchFlow — the edges (architecture.md, capability #8).

Two directions, both pure format translators (no external SDK required):

**Outbound** — translate BenchFlow's ``Scene`` / ``Rubric`` / ``VerifyResult``
/ ``RewardEvent`` into the conventions of external eval frameworks:

* **Inspect AI** — ``InspectAdapter`` / ``to_inspect_task``
* **ORS (OpenReward)** — ``ORSAdapter`` / ``to_ors_reward`` /
  ``write_ors_tool_outputs_jsonl``

**Inbound** — translate a foreign benchmark's task directory into BenchFlow's
native task format so the foreign benchmark runs natively:

* **Native split format** — ``HarborAdapter`` / ``from_harbor_task``
* **Terminal-Bench** — ``TerminalBenchAdapter`` / ``from_terminal_bench_task``
  (Terminal-Bench is also backward-compatible through the native split-format
  adapter, since the split format is itself terminal-bench-derived).

``detect_adapter`` sniffs a task directory and returns the matching inbound
adapter; every inbound adapter returns an :class:`InboundTask`.

To add a new adapter, create a module under ``benchflow.adapters`` and
re-export its public symbols here.
"""

from benchflow.adapters.browser_use import BrowserUseAdapter, from_browser_use_task
from benchflow.adapters.computer_use import (
    ComputerUseAdapter,
    from_computer_use_task,
)
from benchflow.adapters.harbor import HarborAdapter, from_harbor_task
from benchflow.adapters.inbound import (
    InboundCompatibility,
    InboundSupportReport,
    InboundTask,
    UnsupportedInboundTaskError,
    detect_adapter,
    manifest_from_task_config,
    materialize_inbound_task_md,
)
from benchflow.adapters.inspect_ai import InspectAdapter, to_inspect_task
from benchflow.adapters.iosworld import IOSWorldAdapter, from_iosworld_task
from benchflow.adapters.ors import (
    ORSAdapter,
    ors_tool_outputs_to_reward_events,
    to_ors_reward,
    write_ors_tool_outputs_jsonl,
)
from benchflow.adapters.stagehand import StagehandEvalAdapter, from_stagehand_task
from benchflow.adapters.terminal_bench import (
    TerminalBenchAdapter,
    from_terminal_bench_task,
)
from benchflow.adapters.use_computer_cookbook import (
    UseComputerCookbookAdapter,
    from_use_computer_cookbook_task,
)

__all__ = [
    "HarborAdapter",
    "BrowserUseAdapter",
    "ComputerUseAdapter",
    "InboundCompatibility",
    "InboundSupportReport",
    "InboundTask",
    "InspectAdapter",
    "IOSWorldAdapter",
    "ORSAdapter",
    "StagehandEvalAdapter",
    "TerminalBenchAdapter",
    "UnsupportedInboundTaskError",
    "UseComputerCookbookAdapter",
    "detect_adapter",
    "from_browser_use_task",
    "from_computer_use_task",
    "from_harbor_task",
    "from_iosworld_task",
    "from_stagehand_task",
    "from_terminal_bench_task",
    "from_use_computer_cookbook_task",
    "materialize_inbound_task_md",
    "manifest_from_task_config",
    "ors_tool_outputs_to_reward_events",
    "to_inspect_task",
    "to_ors_reward",
    "write_ors_tool_outputs_jsonl",
]
