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
    from before the turn; bfx cannot run), or None."""
    from benchflow.sandbox._verifier_links import BFX_UNAVAILABLE, NO_RECORD

    for c in provenance.get("checks") or ():
        why = c.get("why") if isinstance(c, dict) else None
        if (
            not c.get("ran")
            and isinstance(why, str)
            and (why == NO_RECORD or why.startswith(BFX_UNAVAILABLE))
        ):
            return why
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
