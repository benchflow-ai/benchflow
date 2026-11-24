"""What an attempt that ended with no record of its own spent.

A trial writes ``result.json`` when it ends, and a stopped one a
cancellation receipt (``budget-cancelled.json``). A run that stops mid-turn
with no chance to write either (a VM reset, a power cut, a SIGKILL or OOM
kill of the bench process) leaves neither: the folder keeps its ``config.json`` and what was written while the
trial ran. The model proxy writes each finished call to
``trajectory/llm_trajectory.jsonl`` as it ends (its live capture), and the
agent's session log is mirrored while the turn runs, so the spend is on disk.

:func:`salvaged_attempt` counts it from those logs, by the same steps
``bench eval backfill-cost`` and a trial's own teardown take
(:mod:`benchflow.cost_backfill`): the proxy's log when there is one (the
provider's own usage), else the agent's logs. The count is a floor: a call
still in flight when the run stopped has no line, so the row says
``usage_incomplete``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: ``usage_incomplete_reason`` of a salvaged attempt.
SALVAGED_REASON = (
    "the attempt ended without a result or a receipt (its run stopped "
    "mid-turn: a host reset or a killed process); counted from the logs it "
    "left, so a call still running then is not in it"
)

#: The key a salvaged attempt's row carries, naming what it was counted from
#: (``usage_source``: proxy, session_log, stream_log).
SALVAGED_KEY = "_usage_salvaged"


def salvaged_attempt(folder: Path) -> dict[str, Any] | None:
    """A stopped-attempt row, shaped as a cancellation receipt's
    (``_budget_cancelled``, its spend under ``agent_result``), for the
    attempt folder ``folder`` that left no result.json, solver snapshot or
    receipt; None when its logs hold no model call. Never raises: an
    unreadable log is None, with a warning."""
    from benchflow import cost_backfill as backfill
    from benchflow.budget import _SPEND_FIELDS
    from benchflow.trajectories.model_calls import PROXY_LOG
    from benchflow.usage_tracking import USAGE_PROXY

    config = backfill._read_json(folder / "config.json")
    config = config if isinstance(config, dict) else {}
    stub = {"agent": config.get("agent"), "model": config.get("model")}
    # The proxy's record of the provider's answers comes first, as a
    # proxied trial's teardown counts it; else the agent's own logs.
    proxied = {"usage_source": USAGE_PROXY} if (folder / PROXY_LOG).is_file() else {}
    try:
        count = backfill._derive(folder, stub, proxied).count
        if count is None and proxied:
            count = backfill._derive(folder, stub, {}).count
        if count is None or count.total() <= 0:
            return None
        metrics = backfill._counted_metrics(count, {})
        if count.source == USAGE_PROXY:
            priced = backfill._proxy_priced(count, metrics)
        else:
            priced = backfill._priced_metrics(count, metrics, str(stub["model"] or ""))
    except Exception as exc:
        logger.warning("Could not read the logs of %s: %s", folder, exc)
        return None
    spent = {key: priced.get(key) for key in _SPEND_FIELDS if key in priced}
    spent.update(
        usage_source=count.source,
        usage_incomplete=True,
        usage_incomplete_reason=SALVAGED_REASON,
    )
    return {
        "_budget_cancelled": True,
        SALVAGED_KEY: count.source,
        "agent": stub["agent"],
        "model": stub["model"],
        "agent_result": spent,
        "timing": {},
    }
