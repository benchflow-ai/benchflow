"""benchflow.multi_agent — multi-agent composition primitives.

Re-exports :class:`Scene`, :class:`Turn`, and :class:`Role` from
:mod:`benchflow.contracts.trial_config`. Today these are also re-exported
from :mod:`benchflow.trial` for v0.3 back-compat; the trial.py path emits
:class:`DeprecationWarning` and is removed in v0.5.

Use::

    from benchflow.multi_agent import Scene, Turn, Role

This is the explicit-import discovery path for power users — multi-agent
composition is not in the simple-case top-level namespace. See
PLAN_V2_shaping §3.2.
"""

from benchflow.contracts.trial_config import Role, Scene, Turn

__all__ = ["Role", "Scene", "Turn"]
