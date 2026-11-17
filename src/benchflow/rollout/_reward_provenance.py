"""Whether BenchFlow can confirm test.sh wrote the reward.

One verdict per trial with an agent's turn, in integrity off and audit, in
the trial's ``integrity/verifier-inputs.json`` (``provenance``). It fails
closed: ``verified`` only when every check that
applies ran and passed; a check that could not run makes it
``unverifiable`` with that check named, as does one that failed. The record
lists every check, whether it ran and whether it passed. The checks:

- ``outputs-closed``: the verifier's output folder was closed to the agent
  for the whole of grading (not asked when no test.sh output folder was);
- ``links-before-test``: every link the turn made into the folder was found
  and removed before test.sh (:mod:`benchflow.sandbox._verifier_links`);
- ``links-after-grading``: none was there after grading;
- ``root-agent-processes``: for a root agent's turn (not the oracle's), no
  process it started was alive during grading;
- ``frozen-after-test``: for a root agent's turn, the sandbox was held still
  from test.sh's exit until its outputs were copied
  (:mod:`benchflow.task.verifier_freeze`). A non-root agent can't write the
  folder, so its freeze is recorded (``freeze``) but decides nothing.

It never changes the reward. Strict ends the agent's processes before
grading and is not assessed.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

VERIFIED = "verified"
UNVERIFIABLE = "unverifiable"
ROOT_PROCESSES = (
    "processes a root agent left running were alive during grading; "
    "BenchFlow can't confirm test.sh wrote this reward"
)
FROZEN = (
    "the sandbox was not held still from test.sh's exit until its outputs "
    "were copied ({why}); BenchFlow can't confirm test.sh wrote this reward"
)
NETWORK_ALIVE = (
    "; it was refused while agent processes were alive; whose traffic it was is unknown"
)
_SAID: set[tuple[str, str]] = set()


def run_start_line(mode: str) -> str:
    return (
        f"This agent runs as root in integrity {mode}: a process it leaves "
        "running can rewrite the reward while test.sh runs. Each trial's "
        "integrity/verifier-inputs.json records the reward's provenance, "
        "unverifiable when such a process was alive during grading. Use the "
        "default agent user, or --integrity strict, to rule this out."
    )


def assessed(integrity: str | None) -> bool:
    """Off and audit; strict ends the agent's processes before grading."""
    from benchflow.integrity.constants import hardens_verifier

    return not hardens_verifier(integrity)


def say_run_start(
    agent: str | None, mode: str | None, sandbox_user: str | None
) -> None:
    """The run-start line, once per agent and mode in this process, for an
    agent that runs as root in off or audit (not the oracle)."""
    if sandbox_user is not None or agent == "oracle" or not assessed(mode):
        return
    key = (str(agent), str(mode or "audit"))
    if key in _SAID:
        return
    _SAID.add(key)
    logger.warning(run_start_line(key[1]))


def processes(record: Any) -> list[dict[str, Any]]:
    """The detector's inventory of processes born in the turn and alive when
    grading started (``where: "processes"``)."""
    if not isinstance(record, dict):
        return []
    rows = []
    for key in ("flags", "changes", "processes_at_grading"):
        for row in record.get(key) or ():
            if isinstance(row, dict) and row.get("where") == "processes":
                after = row.get("after") if isinstance(row.get("after"), dict) else {}
                rows.append(
                    {
                        "pid": after.get("pid"),
                        "argv": after.get("argv"),
                        "uid": after.get("uid"),
                        "category": row.get("category"),
                    }
                )
    rows.sort(key=lambda r: (r["pid"] is None, r["pid"] or 0))
    return rows


def processes_listed(record: Any) -> bool:
    """Whether the processes alive as grading started were listed: the
    detector's walk after the turn (audit), or off's own."""
    if not isinstance(record, dict):
        return False
    return record.get("after") is not None or isinstance(
        record.get("processes_at_grading"), list
    )


def _cannot(name: str, why: str | None) -> str:
    return (
        f"the {name} check could not run ({why or 'unknown why'}); BenchFlow "
        "can't confirm test.sh wrote this reward"
    )


def _link_check(
    found: dict[str, Any] | None,
) -> tuple[bool, str | None, list[dict[str, Any]]]:
    """A link check's record as (ran, why, its links); a check not made did
    not run."""
    if not isinstance(found, dict):
        return False, "the check was not made", []
    why = found.get("why")
    links = found.get("links")
    rows = [r for r in links if isinstance(r, dict)] if isinstance(links, list) else []
    return bool(found.get("ran")), why if isinstance(why, str) else None, rows


def verdict(
    *,
    agent: str | None,
    sandbox_user: str | None,
    record: Any,
    outputs_closed: bool | None,
    outputs_open_why: str | None,
    links_before: dict[str, Any] | None,
    links_after: dict[str, Any] | None,
    freeze: dict[str, Any] | None = None,
    outputs_close_applies: bool = True,
) -> dict[str, Any]:
    """The trial's provenance (module docstring). *links_before* /
    *links_after*: the link checks' records (``ran``, ``why``, ``links``);
    *freeze*: the freeze's record, None when the verify made none (strict,
    a verifier without the stop). *outputs_close_applies*: False for a verify
    that runs no test command (llm-judge, agent-judge, an ORS episode), whose
    reward test.sh never wrote, so the output folder's close is no check of
    it."""
    checks: list[dict[str, Any]] = []
    reasons: list[str] = []

    def check(
        name: str, ran: bool, passed: bool, why: str | None, reason: str | None
    ) -> None:
        checks.append({"check": name, "ran": ran, "passed": ran and passed, "why": why})
        if not ran:
            reasons.append(_cannot(name, why))
        elif not passed and reason:
            reasons.append(reason)

    # An unset close (the kept-services path returns before it sets one) is
    # a check that did not run, never one left out (F34); a verify that runs
    # no test command was never due one.
    if outputs_close_applies:
        check(
            "outputs-closed",
            outputs_closed is not None,
            bool(outputs_closed),
            outputs_open_why
            if outputs_closed is not None
            else outputs_open_why or "the output folder's close did not report",
            "the verifier's output folder was not closed to the agent during "
            f"grading ({outputs_open_why or 'unknown why'}); BenchFlow can't "
            "confirm test.sh wrote this reward",
        )
    before_ran, before_why, removed_rows = _link_check(links_before)
    unremoved = [r for r in removed_rows if not r.get("removed")]
    check(
        "links-before-test",
        before_ran,
        not unremoved,
        before_why,
        "a link the agent made into /logs/verifier could not be removed before "
        "test.sh; BenchFlow can't confirm test.sh wrote this reward",
    )
    after_ran, after_why, made = _link_check(links_after)
    from benchflow.sandbox._verifier_links import made_during_grading_reason

    check(
        "links-after-grading",
        after_ran,
        not made,
        after_why,
        made_during_grading_reason(made) if made else None,
    )
    root_agent = sandbox_user is None and agent not in (None, "oracle")
    alive: list[dict[str, Any]] = []
    if root_agent:
        listed = processes_listed(record)
        alive = processes(record) if listed else []
        check(
            "root-agent-processes",
            listed,
            not alive,
            None
            if listed
            else "the processes alive during grading could not be listed",
            ROOT_PROCESSES,
        )
        if freeze is not None:
            check(
                "frozen-after-test",
                True,
                freeze.get("ok") is True,
                freeze.get("why"),
                FROZEN.format(why=freeze.get("why") or "unknown why"),
            )
    found: dict[str, Any] = {
        "verdict": UNVERIFIABLE if reasons else VERIFIED,
        "reasons": reasons,
        "checks": checks,
        "root_agent": root_agent,
        "processes": alive,
        "links_removed_before_test": removed_rows,
        "links_after_grading": made,
        "links_checked": before_ran and after_ran,
    }
    if freeze is not None:
        found["freeze"] = freeze
    return found


def warning(found: dict[str, Any]) -> str | None:
    """The trial warning for an unverifiable reward, else None."""
    if found.get("verdict") != UNVERIFIABLE:
        return None
    text = "Reward provenance unverifiable: " + " | ".join(found.get("reasons") or ())
    pids = [str(p.get("pid")) for p in found.get("processes") or ()]
    if pids:
        text += f" (pids {', '.join(pids[:10])}{' ...' if len(pids) > 10 else ''})"
    return text
