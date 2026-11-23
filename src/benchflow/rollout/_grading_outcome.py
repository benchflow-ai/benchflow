"""The one decision of a verify's ending.

:func:`final_outcome` is the last step of the verify: nothing after it sets
the verifier's error, the rewards or the diagnostic. In order:

1. An earlier ending is kept as it is, with its error, diagnostic, category
   and retry decision: a timeout, a breach, a refusal, a setup failure, a
   crash, a wedge, and BenchFlow's own 0 (a reward test.sh did not write).
   Main's retry and recovery run from it.
2. Otherwise the reward is the one test.sh wrote. For a non-root agent, any
   check of its provenance that did not complete, or whose action did not
   take effect, ends the attempt with reward 0, counted, with the reason
   (``provenance_incomplete``). A root agent's reward is scored as written,
   ``unverifiable``; the oracle's is not marked.
3. Otherwise the reward is scored, ``verified`` or ``unverifiable``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: The reasons a non-root agent's attempt scores 0 (the provenance record's
#: ``outcome_reason``, and the trial warning after its head).
NOT_COMPLETE = "the link check did not complete"
NOT_REMOVED = "a link into the verifier's output folder could not be removed"
APPEARED = "a link into the verifier's output folder appeared during grading"
CLOSE_UNREPORTED = "the output folder's close did not report"
NOT_CLOSED = "the output folder was not closed to the agent during grading"
FAULT = "the reward's provenance could not be decided"


@dataclass
class Attempt:
    """A verify's ending as the verifier and BenchFlow's checks left it."""

    rewards: dict | None
    verifier_error: str | None
    diagnostic: Any
    #: Whether *rewards* is test.sh's own, read from its outputs and still
    #: standing (never BenchFlow's own 0).
    test_sh_reward: bool
    #: The provenance record (``_reward_provenance.verdict``), or None when
    #: the verify was not assessed (strict, or no agent's turn).
    provenance: dict | None
    agent: str | None
    sandbox_user: str | None


def incomplete_reason(provenance: dict[str, Any]) -> str | None:
    """Why a non-root agent's reward cannot be shown to be test.sh's own, or
    None when every check ran and its action took effect."""
    fault = provenance.get("fault")
    if isinstance(fault, str) and fault:
        return f"{FAULT} ({fault})"
    checks = {
        c.get("check"): c for c in provenance.get("checks") or () if isinstance(c, dict)
    }

    def why(check: dict) -> str:
        said = check.get("why")
        return f" ({said})" if isinstance(said, str) and said else ""

    for name in ("links-before-test", "links-after-grading"):
        check = checks.get(name)
        if check is None or not check.get("ran"):
            return NOT_COMPLETE + (why(check) if check else "")
    if not checks["links-before-test"].get("passed"):
        return NOT_REMOVED
    if not checks["links-after-grading"].get("passed"):
        return APPEARED
    closed = checks.get("outputs-closed")
    if closed is not None and closed.get("applies") is False:
        closed = None  # a verify never due a close (no test command): no check
    if closed is not None and not closed.get("ran"):
        return CLOSE_UNREPORTED
    if closed is not None and not closed.get("passed"):
        return NOT_CLOSED + why(closed)
    return None


class IntegrityUnavailableError(RuntimeError):
    """BenchFlow cannot check a reward's provenance in this sandbox (bfx
    cannot run there): the trial is refused before the agent's turn,
    unscored, never retried. Its message is the trial's error as it is."""


def benchflow_gap(provenance: dict[str, Any]) -> str | None:
    """Why a link check did not run through BenchFlow's own gap (no record
    from before the turn, or one that is not complete; bfx cannot run), or
    None."""
    from benchflow.sandbox._verifier_links import (
        BFX_UNAVAILABLE,
        INCOMPLETE_RECORD,
        NO_RECORD,
    )

    for c in provenance.get("checks") or ():
        why = c.get("why") if isinstance(c, dict) else None
        if (
            not c.get("ran")
            and c.get("applies") is not False
            and isinstance(why, str)
            and (
                why == NO_RECORD
                or why.startswith(BFX_UNAVAILABLE)
                or why.startswith(INCOMPLETE_RECORD)
            )
        ):
            return why
    return None


def pre_turn_refusal(record: Any, *, primary_agent: str | None) -> str | None:
    """Why the trial is refused before the agent's turn because of its record
    of the links before the turn, or None.

    A record that is not complete (a walk bfx failed on twice, a walk over a
    cap, bfx that could not run after all) can't tell the image's own links
    from the turn's. That is BenchFlow's gap, known before any agent code
    runs, so it is a refusal before the turn (Q4), unscored, for every model
    agent, non-root or root; never the agent's 0 after it (Q6 asks for a
    check defeated during the run). The oracle is exempt and runs. No record
    at all keeps its own rule (unscored after the turn, F39)."""
    from benchflow._utils.scoring import INTEGRITY_UNAVAILABLE_HEADER
    from benchflow.sandbox._verifier_links import (
        BFX_UNAVAILABLE,
        INCOMPLETE_RECORD,
        Walked,
    )

    if primary_agent == "oracle" or not isinstance(record, Walked) or record.complete:
        return None
    why = record.why or "unknown why"
    if why.startswith(BFX_UNAVAILABLE):
        return INTEGRITY_UNAVAILABLE_HEADER + why
    return f"{INTEGRITY_UNAVAILABLE_HEADER}{INCOMPLETE_RECORD}: {why}"


async def rb2_refusal(
    verifier_user: str,
    agent_user: str,
    read_uid: Callable[[str], Awaitable[str | None]],
) -> str | None:
    """RB2 by uid, before the turn: why the trial is refused, or None.
    Another name for the agent's uid is the agent's own user, whose
    processes could reach the verifier's output folder. *read_uid* reads a
    user's uid as that user (None: it could not be read). Where either uid
    can't be read nothing shows they differ, so the trial is refused with
    that reason (REVIEW-FINAL F7: fail closed)."""
    uid = await read_uid(verifier_user)
    if uid is None:
        return (
            f"the uid of the task's verifier user ({verifier_user}) could not "
            "be read, so it can't be told from the agent's"
        )
    agent_uid = await read_uid(agent_user)
    if agent_uid is None:
        return (
            f"the uid of the agent's user ({agent_user}) could not be read, so "
            f"the verifier user's ({verifier_user}, uid {uid}) can't be told "
            "from it"
        )
    if uid == agent_uid:
        return (
            f"the task's verifier user ({verifier_user}) has the agent's own "
            f"uid ({uid}, {agent_user}), so its output folder cannot be kept "
            "from the agent's processes"
        )
    return None


def final_outcome(attempt: Attempt) -> tuple[dict | None, str | None, Any]:
    """``(rewards, verifier_error, diagnostic)``: the attempt's ending."""
    from benchflow._utils.scoring import (
        PROVENANCE_INCOMPLETE,
        PROVENANCE_INCOMPLETE_ZERO,
    )
    from benchflow._utils.trial_warnings import report_trial_warning

    kept = (attempt.rewards, attempt.verifier_error, attempt.diagnostic)
    if (
        not attempt.test_sh_reward
        or attempt.verifier_error is not None
        or attempt.rewards is None
    ):
        return kept  # an earlier ending, as it is
    provenance = attempt.provenance
    if (
        provenance is None
        or attempt.agent in (None, "oracle")
        or attempt.sandbox_user is None
    ):
        # Not assessed; the oracle, not marked; a root agent, scored as
        # written with its verdict (unverifiable when a check failed).
        return kept
    gap = benchflow_gap(provenance)
    if gap is not None:
        from benchflow._utils.scoring import INTEGRITY_UNAVAILABLE_VERIFIER

        # BenchFlow's own gap: unscored with its reason, never the agent's 0.
        error = INTEGRITY_UNAVAILABLE_VERIFIER + gap
        logger.error(error)
        return None, error, None
    reason = incomplete_reason(provenance)
    if reason is None:
        return kept
    provenance["outcome"] = PROVENANCE_INCOMPLETE
    provenance["outcome_reason"] = reason
    warning = PROVENANCE_INCOMPLETE_ZERO + reason
    logger.warning(warning)
    report_trial_warning(warning)
    return {"reward": 0.0}, None, None
