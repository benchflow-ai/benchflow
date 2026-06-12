"""Ownership labelling and the age-based auto-reaper for Daytona sandboxes.

Extracted from ``benchflow.sandbox.daytona`` as a cohesion seam (the ownership
scope gate plus the TTL reaper that depends on it). The names here are
re-exported from ``benchflow.sandbox.daytona`` so existing imports such as
``from benchflow.sandbox.daytona import reap_stale_sandboxes`` keep working
unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("benchflow")


_REAP_DEFAULT_MAX_AGE_MIN = 1440
_REAP_FAILED_MAX_AGE_MIN = 120
_REAP_FAILED_STATE_MARKERS = ("FAILED", "ERROR")
# Minimum idle window before the general (non-failed) tier may reap an owned
# sandbox that is old by creation time. The SDK returns ``last_activity_at`` on
# list results, so a genuinely live, recently-active run older than the TTL is
# protected without an extra API call. Well under the 1440-min creation TTL.
_REAP_MIN_IDLE_MIN = 30

# Ownership scoping for the auto-reaper. Every sandbox benchflow creates is
# stamped with this label so :func:`reap_stale_sandboxes` can restrict its
# age-based deletion to benchflow's *own* sandboxes. Without it, a reap run
# against a ``DAYTONA_API_KEY`` shared across an org (or with other tools) would
# delete unrelated sandboxes by age alone — irreversible, cross-tenant data
# loss. Foreign / unlabeled sandboxes are therefore never touched, regardless
# of age. The dotted key follows the Docker/Daytona label convention.
_BENCHFLOW_MANAGED_LABEL = "benchflow.managed"
_BENCHFLOW_MANAGED_VALUE = "1"
# The dotted prefix every benchflow-stamped label key shares. Used only by the
# read-only orphan-leak guard below to recognize a sandbox that *looks* like
# benchflow created it (carries the namespace) yet fails the strict ownership
# check — i.e. its ownership label drifted off the exact key/value the reaper
# keys on. ``_BENCHFLOW_MANAGED_LABEL`` itself lives under this namespace.
_BENCHFLOW_LABEL_NAMESPACE = "benchflow."


def _benchflow_owned_labels() -> dict[str, str]:
    """Return a fresh ownership-label dict for one sandbox-creation call.

    A new dict per call is required: the Daytona SDK mutates ``params.labels``
    in place (it injects the language label), so a shared dict would leak that
    mutation across creation sites.
    """
    return {_BENCHFLOW_MANAGED_LABEL: _BENCHFLOW_MANAGED_VALUE}


def _is_benchflow_owned(sb: Any) -> bool:
    """Return whether *sb* carries benchflow's exact ownership label.

    This scope check is the only thing standing between the age-based reaper
    and other people's sandboxes when the API key is shared. Anything missing
    the exact key/value pair — including sandboxes with no labels at all, or a
    ``labels`` attribute that is not a mapping — is treated as foreign and left
    untouched.
    """
    labels = getattr(sb, "labels", None)
    if not isinstance(labels, dict):
        return False
    return labels.get(_BENCHFLOW_MANAGED_LABEL) == _BENCHFLOW_MANAGED_VALUE


def _is_benchflow_label_orphan(sb: Any) -> bool:
    """Return whether *sb* looks benchflow-created but lacks the ownership label.

    True only when the sandbox carries at least one ``benchflow.``-namespaced
    label key yet fails :func:`_is_benchflow_owned` (the exact
    ``benchflow.managed=1`` pair is absent or has drifted to another value).
    Such a sandbox is almost certainly one benchflow created whose ownership
    label was lost — the age-based reaper's scope gate will now skip it forever,
    so it leaks. This is a *detection-only* predicate: the missing/altered label
    means ownership cannot be proven strongly enough to delete on a shared API
    key, so the reaper only warns. A correctly-labeled (owned) sandbox is never
    an orphan, and a purely foreign sandbox (no benchflow namespace) is ignored.
    """
    if _is_benchflow_owned(sb):
        return False
    labels = getattr(sb, "labels", None)
    if not isinstance(labels, dict):
        return False
    return any(
        isinstance(key, str) and key.startswith(_BENCHFLOW_LABEL_NAMESPACE)
        for key in labels
    )


def _parse_sandbox_timestamp(raw: Any) -> Any | None:
    """Parse a Daytona timestamp string to a tz-aware datetime, or ``None``.

    A naive (no ``Z``/offset) value is coerced to UTC rather than dropped, so a
    server that omits the offset doesn't silently leak a stale sandbox. Returns
    ``None`` for a non-string or unparseable value; the caller decides whether
    absence is protective (activity guard) or skip-worthy (created_at).
    """
    from datetime import UTC, datetime

    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def reap_stale_sandboxes(
    client: Any | None = None,
    *,
    max_age_minutes: int = _REAP_DEFAULT_MAX_AGE_MIN,
    failed_max_age_minutes: int = _REAP_FAILED_MAX_AGE_MIN,
    min_idle_minutes: int = _REAP_MIN_IDLE_MIN,
    dry_run: bool = False,
    on_decision: Any | None = None,
) -> dict[str, int]:
    """Delete orphaned Daytona sandboxes past their TTL.

    Ownership-scoped: only sandboxes benchflow created — those carrying the
    ``benchflow.managed`` label (see :func:`_is_benchflow_owned`) — are ever
    considered. Foreign / unlabeled sandboxes are skipped before any age check,
    so a ``DAYTONA_API_KEY`` shared across an org or with other tools cannot be
    used to destroy unrelated sandboxes by age alone.

    Two tiers: sandboxes whose state contains a failure marker (e.g.
    ``BUILD_FAILED``) are reaped after *failed_max_age_minutes*; everything
    else after *max_age_minutes*. The general (non-failed) tier additionally
    honours an activity guard: an owned sandbox whose ``last_activity_at`` is
    within *min_idle_minutes* is skipped even when old by creation time, so a
    genuinely live run is never reaped. Absent/unparseable activity is not
    protective (it falls through to age-only), or nothing old would ever reap.
    The failed tier ignores the activity guard so a crashed sandbox still reaps
    on its short TTL. Defaults are deliberately conservative so concurrent live
    runs are never touched — only multi-hour idle orphans from crashed or
    interrupted sessions.

    *on_decision* (sandbox, age_minutes, will_delete) is called per *owned*
    sandbox when provided — the CLI uses it for per-row display; foreign
    sandboxes are never surfaced as reap candidates. Returns counts:
    ``{"found", "deleted", "skipped", "failed"}`` (``found`` counts every
    sandbox listed; foreign ones fall into ``skipped``).
    """
    from datetime import UTC, datetime

    if client is None:
        from benchflow.sandbox.daytona import build_sync_client

        client = build_sync_client()
    now = datetime.now(UTC)
    counts = {"found": 0, "deleted": 0, "skipped": 0, "failed": 0}
    for sb in client.list():
        counts["found"] += 1
        if not _is_benchflow_owned(sb):
            # Scope guard: never touch a sandbox benchflow did not create.
            # This is the load-bearing safety check on a shared API key.
            if _is_benchflow_label_orphan(sb):
                # Read-only orphan-leak guard: a sandbox carrying the benchflow
                # namespace but missing the exact ownership label will never be
                # reaped by age. Surface it so an operator can reclaim it by
                # hand; we deliberately do not delete unlabeled sandboxes.
                logger.warning(
                    "Daytona sandbox %s carries a benchflow label namespace but "
                    "is missing the %s=%s ownership label; the age-based reaper "
                    "will never reclaim it (possible orphan leak). Not deleting — "
                    "verify and remove it manually if it is stale.",
                    getattr(sb, "id", "?"),
                    _BENCHFLOW_MANAGED_LABEL,
                    _BENCHFLOW_MANAGED_VALUE,
                )
            counts["skipped"] += 1
            continue
        if not getattr(sb, "created_at", None):
            counts["skipped"] += 1
            continue
        # Guard the age parse per-sandbox (mirror the delete guard below): one
        # odd timestamp must warn + skip, never abort the sweep and leak every
        # sandbox after it.
        created_at = _parse_sandbox_timestamp(sb.created_at)
        if created_at is None:
            logger.warning(
                "Daytona sandbox %s has unparseable created_at %r; skipping "
                "(possible leak)",
                getattr(sb, "id", "?"),
                sb.created_at,
            )
            counts["skipped"] += 1
            continue
        age_minutes = (now - created_at).total_seconds() / 60
        state = str(getattr(sb, "state", "") or "").upper()
        is_failed = any(marker in state for marker in _REAP_FAILED_STATE_MARKERS)
        ttl = failed_max_age_minutes if is_failed else max_age_minutes
        will_delete = age_minutes >= ttl
        # Activity guard (general tier only): protect a genuinely live run whose
        # last activity is within the idle window, even when old by creation.
        # Failed sandboxes ignore the guard so they still reap on the short TTL.
        if will_delete and not is_failed:
            last_activity = _parse_sandbox_timestamp(
                getattr(sb, "last_activity_at", None) or getattr(sb, "updated_at", None)
            )
            if last_activity is not None:
                idle_minutes = (now - last_activity).total_seconds() / 60
                if idle_minutes < min_idle_minutes:
                    will_delete = False
        if on_decision is not None:
            on_decision(sb, age_minutes, will_delete)
        if not will_delete:
            counts["skipped"] += 1
            continue
        if dry_run:
            counts["deleted"] += 1
            continue
        try:
            client.delete(sb)
            counts["deleted"] += 1
        except Exception:
            logger.warning("Failed to delete sandbox %s", getattr(sb, "id", "?"))
            counts["failed"] += 1
    return counts
