"""External framework adapters for BenchFlow types.

These are pure format converters — they translate BenchFlow's ``Scene``,
``Rubric``, ``VerifyResult``, and ``RewardEvent`` types into dicts that
follow the conventions of external eval frameworks.  No external SDK is
required.

Supported targets:

* **Inspect AI** — ``InspectAdapter`` / ``to_inspect_task``
* **ORS (OpenReward)** — ``ORSAdapter`` / ``to_ors_reward``

To add a new adapter, create a module under ``benchflow.adapters`` and
re-export its public symbols here.
"""

from benchflow.adapters.inspect_ai import InspectAdapter, to_inspect_task
from benchflow.adapters.ors import ORSAdapter, to_ors_reward

__all__ = [
    "InspectAdapter",
    "ORSAdapter",
    "to_inspect_task",
    "to_ors_reward",
]
