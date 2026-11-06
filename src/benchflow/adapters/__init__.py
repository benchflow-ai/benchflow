"""Framework adapters for BenchFlow — the edges (architecture.md, capability #8).

Two directions, both pure format translators (no external SDK required):

**Outbound** — translate BenchFlow's ``Scene`` / ``Rubric`` / ``VerifyResult``
/ ``RewardEvent`` into the conventions of external eval frameworks:

* **Inspect AI** — ``InspectAdapter`` / ``to_inspect_task``
* **ORS (OpenReward)** — ``ORSAdapter`` / ``to_ors_reward`` /
  ``write_ors_tool_outputs_jsonl``

**Inbound** — translate a foreign benchmark's task directory into BenchFlow's
native task format so the foreign benchmark runs natively:

* **task.toml folders** — ``TaskTomlAdapter`` / ``from_task_toml_dir``

``detect_adapter`` sniffs a task directory and returns the matching inbound
adapter; every inbound adapter returns an :class:`InboundTask`.

To add a new adapter, create a module under ``benchflow.adapters`` and
re-export its public symbols here.
"""

from benchflow.adapters.inbound import (
    InboundCompatibility,
    InboundTask,
    detect_adapter,
    manifest_from_task_config,
    materialize_inbound_task_md,
)
from benchflow.adapters.inspect_ai import InspectAdapter, to_inspect_task
from benchflow.adapters.ors import (
    ORSAdapter,
    ors_tool_outputs_to_reward_events,
    to_ors_reward,
    write_ors_tool_outputs_jsonl,
)
from benchflow.adapters.source import adapt_resolved_source_if_needed
from benchflow.adapters.task_toml import TaskTomlAdapter, from_task_toml_dir

__all__ = [
    "InboundCompatibility",
    "InboundTask",
    "InspectAdapter",
    "ORSAdapter",
    "TaskTomlAdapter",
    "detect_adapter",
    "adapt_resolved_source_if_needed",
    "from_task_toml_dir",
    "materialize_inbound_task_md",
    "manifest_from_task_config",
    "ors_tool_outputs_to_reward_events",
    "to_inspect_task",
    "to_ors_reward",
    "write_ors_tool_outputs_jsonl",
]
