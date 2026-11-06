"""Ownership labelling, run leases and the reapers for Daytona sandboxes.

Extracted from ``benchflow.sandbox.daytona`` as a cohesion seam (the ownership
scope gate plus the TTL reaper that depends on it). The names here are
re-exported from ``benchflow.sandbox.daytona`` so existing imports such as
``from benchflow.sandbox.daytona import reap_stale_sandboxes`` keep working
unchanged.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
import uuid
from typing import Any

from benchflow.sandbox.daytona_delete import (
    delete_sandbox_sync,
    sandbox_state,
    still_building,
)

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
_BENCHFLOW_OWNER_LABEL = "benchflow.owner"
_BENCHFLOW_OWNER_ENV = "BENCHFLOW_DAYTONA_OWNER"
_BENCHFLOW_OWNER_MAX_LEN = 48
# The dotted prefix every benchflow-stamped label key shares. Used only by the
# read-only orphan-leak guard below to recognize a sandbox that *looks* like
# benchflow created it (carries the namespace) yet fails the strict ownership
# check — i.e. its ownership label drifted off the exact key/value the reaper
# keys on. ``_BENCHFLOW_MANAGED_LABEL`` itself lives under this namespace.
_BENCHFLOW_LABEL_NAMESPACE = "benchflow."


# Liveness is judged by Daytona, never by a clock outside it .
# Every sandbox is created with Daytona's inactivity
# stop, AUTO_STOP_MIN: Daytona stops a sandbox no one has called for that
# long, on its own clock. A living controller sends refresh_activity() every
# _HEARTBEAT_SEC besides its exec polls (both count as activity; set_labels
# does not: measured, results/DAYTONA-START.md R3.3), so only a paused,
# cut-off or dead controller's sandbox is stopped. 15 min is Daytona's own
# default and over 10 times the longest gap a living controller leaves.
AUTO_STOP_MIN = 15
_HEARTBEAT_SEC = 60.0
# The states in which Daytona has taken a sandbox out of service. The
# eval-start auto-reap deletes only these, never a STARTED one.
_OUT_OF_SERVICE_STATES = frozenset({"STOPPED", "ARCHIVED", "ERROR", "BUILD_FAILED"})

# Run leases. Every sandbox carries its creating run's id and a lease, a Unix
# time until which that run promises to renew it; the run renews it every
# _LEASE_RENEW_SEC while it holds the sandbox. A lapsed lease alone proves
# nothing (a paused controller's command still runs): the auto-reap also
# needs Daytona to have stopped the sandbox. The lease only keeps a stopped
# sandbox of a run that still holds it from being deleted early.
_BENCHFLOW_RUN_LABEL = "benchflow.run"
_BENCHFLOW_LEASE_LABEL = "benchflow.lease"
# A sandbox kept with --keep-sandbox (rollout/_keep_sandbox): no reaper deletes
# it by age, lease or --all; `bench sandbox release` (or --id) does.
KEPT_LABEL = "benchflow.kept"
_LEASE_SEC = 3600
_LEASE_RENEW_SEC = 600
_LEASE_GRACE_SEC = 600
# One id per process: the run that creates (and renews) its sandboxes.
RUN_ID = uuid.uuid4().hex[:16]


def lease_labels(extra_sec: float = 0.0) -> dict[str, str]:
    """This run's id and a lease from now, plus *extra_sec* (a create's own
    timeout, so a long build cannot outlive the lease)."""
    until = int(time.time() + _LEASE_SEC + max(extra_sec, 0.0))
    return {_BENCHFLOW_RUN_LABEL: RUN_ID, _BENCHFLOW_LEASE_LABEL: str(until)}


def lease_state(sb: Any, now: float | None = None) -> str:
    """``live``, ``lapsed`` (renewal stopped; not proof by itself that the
    run is gone) or ``none`` (no lease, or one that cannot be read)."""
    labels = getattr(sb, "labels", None)
    raw = labels.get(_BENCHFLOW_LEASE_LABEL) if isinstance(labels, dict) else None
    if not isinstance(raw, str) or not raw.isdigit():
        return "none"
    now = time.time() if now is None else now
    return "lapsed" if now > int(raw) + _LEASE_GRACE_SEC else "live"


def out_of_service(sb: Any) -> bool:
    """Daytona has stopped, archived or errored it: no command runs in it."""
    return sandbox_state(sb) in _OUT_OF_SERVICE_STATES


def _benchflow_owner_scope() -> str | None:
    raw = os.environ.get(_BENCHFLOW_OWNER_ENV, "")
    if not raw:
        return None
    # This value authorizes deletion. Never collapse distinct operator
    # identities by replacing characters, trimming whitespace or truncating.
    if (
        len(raw) > _BENCHFLOW_OWNER_MAX_LEN
        or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", raw) is None
        or raw.endswith((".", "-"))
    ):
        raise ValueError(
            "BENCHFLOW_DAYTONA_OWNER must be 1-48 ASCII letters, digits, '_', "
            "'.' or '-', without a leading or trailing '.' or '-'"
        )
    return raw


def _benchflow_managed_value() -> str:
    owner = _benchflow_owner_scope()
    return _BENCHFLOW_MANAGED_VALUE if owner is None else f"1:{owner}"


def _benchflow_owned_labels() -> dict[str, str]:
    """Return a fresh ownership-label dict for one sandbox-creation call.

    A new dict per call is required: the Daytona SDK mutates ``params.labels``
    in place (it injects the language label), so a shared dict would leak that
    mutation across creation sites.
    """
    owner = _benchflow_owner_scope()
    labels = {_BENCHFLOW_MANAGED_LABEL: _benchflow_managed_value()}
    if owner is not None:
        labels[_BENCHFLOW_OWNER_LABEL] = owner
    return labels


# Branch snapshots (``DaytonaSandbox.snapshot``). Daytona snapshots carry no
# labels, so ownership lives in a versioned, fixed-width digest field before
# variable text. Putting the owner text first lets another accepted owner
# extend that text through the digest delimiter and match the shorter prefix.
# Legacy names cannot prove ownership and are left for manual cleanup.
_SNAPSHOT_PREFIX = "bf-snap-"
_OWNED_SNAPSHOT_PREFIX = "bf-snap2-"
_SNAPSHOT_ENV_PART_MAX_LEN = 32


def _snapshot_name_part(text: str) -> str:
    return re.sub(r"[^a-z0-9.-]+", "-", text.lower()).strip(".-")


def owner_snapshot_prefix() -> str | None:
    """Name prefix of this owner's branch snapshots, or None when unscoped."""
    owner = _benchflow_owner_scope()
    if owner is None:
        return None
    tag = hashlib.sha256(owner.encode()).hexdigest()[:32]
    return f"{_OWNED_SNAPSHOT_PREFIX}{tag}-"


def benchflow_snapshot_name(environment_name: str) -> str:
    """A fresh branch snapshot name, owner-scoped when an owner is set."""
    prefix = owner_snapshot_prefix() or _SNAPSHOT_PREFIX
    env_part = (
        _snapshot_name_part(environment_name)[:_SNAPSHOT_ENV_PART_MAX_LEN].strip(".-")
        or "sandbox"
    )
    return f"{prefix}{env_part}-{uuid.uuid4().hex[:12]}"


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
    return labels.get(_BENCHFLOW_MANAGED_LABEL) == _benchflow_managed_value()


def _is_benchflow_label_orphan(sb: Any) -> bool:
    """Return whether *sb* looks benchflow-created but lacks the ownership label.

    True only when the sandbox carries at least one ``benchflow.``-namespaced
    label key yet lacks any valid ownership marker. A sandbox with a scoped
    ``benchflow.managed=1:<owner>`` value, or the unscoped ``1`` while this run
    is scoped, belongs to another BenchFlow operator, so it is foreign rather
    than orphaned and must be silently skipped.
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
    managed = labels.get(_BENCHFLOW_MANAGED_LABEL)
    # Any valid marker names an owner: unscoped "1" belongs to runs without
    # BENCHFLOW_DAYTONA_OWNER, "1:<owner>" to that owner. Either is foreign
    # when it is not ours, never an orphan.
    if managed == _BENCHFLOW_MANAGED_VALUE or (
        isinstance(managed, str) and managed.startswith("1:")
    ):
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
    Snapshot DTOs already carry ``datetime`` values; those pass through.
    """
    from datetime import UTC, datetime

    if isinstance(raw, datetime):
        return raw if raw.tzinfo is not None else raw.replace(tzinfo=UTC)
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _list_query(filters: dict[str, Any]) -> Any:
    try:
        from daytona import ListSandboxesQuery
    except ImportError:  # a faked client (tests without the Daytona extra)
        import types

        return types.SimpleNamespace(**filters)
    return ListSandboxesQuery(**filters)


def reap_stale_sandboxes(
    client: Any | None = None,
    *,
    max_age_minutes: int = _REAP_DEFAULT_MAX_AGE_MIN,
    failed_max_age_minutes: int = _REAP_FAILED_MAX_AGE_MIN,
    min_idle_minutes: int = _REAP_MIN_IDLE_MIN,
    dry_run: bool = False,
    on_decision: Any | None = None,
    ignore_age: bool = False,
    on_skip: Any | None = None,
    names: frozenset[str] | set[str] = frozenset(),
) -> dict[str, int]:
    """Delete orphaned Daytona sandboxes past their TTL (``bench sandbox
    cleanup``, operator-invoked; the eval-start auto-reap is
    :func:`reap_abandoned_sandboxes`).

    A run lease decides first: a sandbox whose lease is live (its run still
    renews it) is never deleted here, ``--all`` included, unless its id is in
    *names* (``--id``). One whose lease lapsed and that Daytona has stopped,
    archived or errored is deleted whatever its age. Any other sandbox (no
    lease, or a lapsed lease on a STARTED one) is selected by the age and
    activity rules below, which cannot tell a quiet live run from an
    abandoned one; the CLI warns about that.

    ``ignore_age=True`` deletes every sandbox this operator owns that is not
    STARTED and has no live lease, whatever its age or activity (the ``bench
    sandbox cleanup --all`` teardown step for CI; the ownership scope still
    applies). A sandbox whose id is in *names* is deleted whatever its state,
    age, activity or lease, but for one state: a sandbox whose image Daytona
    is still building is never deleted (``daytona_delete.still_building``),
    since the delete would make Daytona keep its build as failed for every
    later create of the image. *on_skip* (sandbox, reason) is called for each
    sandbox left alone, with reason ``foreign`` (not owned), ``building``
    (still building), ``leased`` (a live lease), ``started`` (STARTED, under
    ``ignore_age``), ``no_created_at``, ``young`` (under its TTL) or
    ``active`` (the activity guard).

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
    for name, value in (
        ("max_age_minutes", max_age_minutes),
        ("failed_max_age_minutes", failed_max_age_minutes),
        ("min_idle_minutes", min_idle_minutes),
    ):
        if value < 0:
            # A negative age moves the cutoff into the future, so every
            # sandbox — including one that is STARTED and serving a run —
            # looks past its TTL and is selected for deletion.
            raise ValueError(
                f"{name} must be >= 0, got {value}. A negative age would "
                "select every sandbox, including running ones."
            )

    from datetime import UTC, datetime

    if client is None:
        from benchflow.sandbox.daytona import build_sync_client

        client = build_sync_client()
    now = datetime.now(UTC)
    counts = {"found": 0, "deleted": 0, "skipped": 0, "failed": 0}
    orphans: list[str] = []
    for sb in client.list():
        counts["found"] += 1
        if not _is_benchflow_owned(sb):
            # Scope guard: never touch a sandbox benchflow did not create.
            # This is the load-bearing safety check on a shared API key.
            if _is_benchflow_label_orphan(sb):
                # Read-only orphan-leak guard: a sandbox carrying the benchflow
                # namespace but missing the exact ownership label will never be
                # reaped by age. It is reported after the sweep, so an operator
                # can reclaim it by hand; we deliberately do not delete
                # unlabeled sandboxes.
                orphans.append(str(getattr(sb, "id", "?")))
            counts["skipped"] += 1
            if on_skip is not None:
                on_skip(sb, "foreign")
            continue
        if still_building(sb):
            # Left to Daytona, whose auto-stop and auto-delete remove it once
            # it has started; a later cleanup can delete it then.
            logger.warning(
                "Not deleting Daytona sandbox %s: it is still building (%s), "
                "and a delete now would make Daytona keep its build as failed "
                '("context canceled") for every later create of the image',
                getattr(sb, "id", "?"),
                sandbox_state(sb),
            )
            counts["skipped"] += 1
            if on_skip is not None:
                on_skip(sb, "building")
            continue
        lease = lease_state(sb, now.timestamp())
        if getattr(sb, "id", None) in names:
            # Asked for by name: the operator's decision, whatever the lease.
            if on_decision is not None:
                on_decision(sb, _age_minutes(sb, now), True)
            _delete_counted(client, sb, counts, dry_run)
            continue
        labels = getattr(sb, "labels", None)
        if isinstance(labels, dict) and labels.get(KEPT_LABEL) == "1":
            counts["skipped"] += 1
            if on_skip is not None:
                on_skip(sb, "kept")
            continue
        if lease == "live":
            # Its run still renews it: never deleted by age, nor by --all.
            counts["skipped"] += 1
            if on_skip is not None:
                on_skip(sb, "leased")
            continue
        if ignore_age:
            if sandbox_state(sb) == "STARTED":
                # Running, so possibly a live run's (a paused controller, a
                # lease written by a clock behind): --all keeps it, lapsed
                # lease or not. Daytona stops an abandoned one within
                # AUTO_STOP_MIN; --id deletes it now.
                counts["skipped"] += 1
                if on_skip is not None:
                    on_skip(sb, "started")
                continue
            if on_decision is not None:
                on_decision(sb, 0.0, True)
            _delete_counted(client, sb, counts, dry_run)
            continue
        if lease == "lapsed" and out_of_service(sb):
            # Daytona stopped it and its run stopped renewing: whatever its age.
            if on_decision is not None:
                on_decision(sb, _age_minutes(sb, now), True)
            _delete_counted(client, sb, counts, dry_run)
            continue
        if not getattr(sb, "created_at", None):
            counts["skipped"] += 1
            if on_skip is not None:
                on_skip(sb, "no_created_at")
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
        skip_reason = None if will_delete else "young"
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
                    skip_reason = "active"
        if on_decision is not None:
            on_decision(sb, age_minutes, will_delete)
        if not will_delete:
            counts["skipped"] += 1
            if on_skip is not None:
                on_skip(sb, skip_reason or "young")
            continue
        if dry_run:
            counts["deleted"] += 1
            continue
        try:
            delete_sandbox_sync(client, sb)
            counts["deleted"] += 1
        except Exception:
            logger.warning("Failed to delete sandbox %s", getattr(sb, "id", "?"))
            counts["failed"] += 1
    _warn_label_orphans(orphans)
    return counts


_ORPHAN_IDS_SHOWN = 5


def _warn_label_orphans(ids: list[str]) -> None:
    """One warning for every sandbox the sweep found with benchflow labels but
    no ownership label. A shared account can hold hundreds of them, and one
    line each buried the sweep's summary. On a shared API key they may be
    other users' sandboxes, so the warning must not invite deleting them."""
    if not ids:
        return
    shown = ", ".join(ids[:_ORPHAN_IDS_SHOWN])
    if len(ids) > _ORPHAN_IDS_SHOWN:
        shown += f" and {len(ids) - _ORPHAN_IDS_SHOWN} more"
    logger.warning(
        "%d Daytona sandbox(es) carry a benchflow label namespace but not the "
        "%s=%s ownership label, so the age-based reaper will never reclaim them "
        "(possible orphan leak): %s. Not deleting. They may belong to other "
        "users of this API key: delete one only if you started it, with "
        "`bench sandbox cleanup --id <sandbox id>`.",
        len(ids),
        _BENCHFLOW_MANAGED_LABEL,
        _benchflow_managed_value(),
        shown,
    )


def _age_minutes(sb: Any, now: Any) -> float:
    created_at = _parse_sandbox_timestamp(getattr(sb, "created_at", None))
    return (now - created_at).total_seconds() / 60 if created_at else 0.0


def _delete_counted(client: Any, sb: Any, counts: dict[str, int], dry_run: bool):
    if dry_run:
        counts["deleted"] += 1
        return
    try:
        delete_sandbox_sync(client, sb)
        counts["deleted"] += 1
    except Exception:
        logger.warning("Failed to delete sandbox %s", getattr(sb, "id", "?"))
        counts["failed"] += 1


def reap_abandoned_sandboxes(
    client: Any | None = None, *, report_after_minutes: int = _REAP_DEFAULT_MAX_AGE_MIN
) -> dict[str, int]:
    """The eval-start auto-reap (DAYTONA-START round 3, rule A).

    It deletes a sandbox only when Daytona has already taken it out of
    service (STOPPED, ARCHIVED, ERROR or BUILD_FAILED), its run lease lapsed,
    and it is this operator's. A STARTED sandbox is never deleted, whatever
    its lease, age or activity: a paused or cut-off controller's command may
    still run in it, and Daytona's own inactivity stop (``AUTO_STOP_MIN``)
    is what decides it is idle. A STARTED one with a lapsed lease is counted
    in ``started_lapsed`` (its ids in the log) and left. One with no lease
    older than *report_after_minutes* is counted in ``unleased``.

    Daytona is asked only for this operator's sandboxes created before a
    lease could have lapsed (O(1) pages, not the org). Returns counts
    ``{"found", "deleted", "skipped", "failed", "unleased", "started_lapsed"}``.
    """
    from datetime import UTC, datetime, timedelta

    if client is None:
        from benchflow.sandbox.daytona import build_sync_client

        client = build_sync_client()
    now = datetime.now(UTC)
    counts = {
        "found": 0,
        "deleted": 0,
        "skipped": 0,
        "failed": 0,
        "unleased": 0,
        "started_lapsed": 0,
    }
    # A lease is set at create to at least created + _LEASE_SEC, so nothing
    # younger can have lapsed; the grace covers clock skew.
    query = _list_query(
        {
            "labels": {_BENCHFLOW_MANAGED_LABEL: _benchflow_managed_value()},
            "created_at_before": now - timedelta(seconds=_LEASE_SEC),
        }
    )
    for sb in client.list(query):
        counts["found"] += 1
        if not _is_benchflow_owned(sb):
            counts["skipped"] += 1
            continue
        lease = lease_state(sb, now.timestamp())
        if lease == "lapsed" and out_of_service(sb):
            _delete_counted(client, sb, counts, dry_run=False)
            continue
        counts["skipped"] += 1
        if lease == "lapsed" and sandbox_state(sb) == "STARTED":
            counts["started_lapsed"] += 1
            logger.info(
                "Daytona sandbox %s is STARTED with a lapsed run lease; not "
                "deleted (Daytona stops it once it is idle for %s min)",
                getattr(sb, "id", "?"),
                AUTO_STOP_MIN,
            )
        if lease == "none" and _age_minutes(sb, now) >= report_after_minutes:
            counts["unleased"] += 1
    return counts


def reap_stale_snapshots(
    client: Any | None = None,
    *,
    max_age_minutes: int = _REAP_DEFAULT_MAX_AGE_MIN,
    dry_run: bool = False,
    on_decision: Any | None = None,
) -> dict[str, int]:
    """Delete this owner's branch snapshots older than *max_age_minutes*.

    The branch engine deletes each fork's snapshot when the fork finishes; this
    is the backstop for a process that died mid-fork. Only snapshots named
    with this owner's prefix (:func:`owner_snapshot_prefix`) are considered.
    Without ``BENCHFLOW_DAYTONA_OWNER`` nothing proves a snapshot is ours, so
    no snapshot is listed or deleted. *on_decision* (snapshot, age_minutes,
    will_delete) is called per owned snapshot. Returns counts
    ``{"found", "deleted", "skipped", "failed"}``; ``found`` counts every
    snapshot listed.
    """
    if max_age_minutes < 0:
        raise ValueError(
            f"max_age_minutes must be >= 0, got {max_age_minutes}. A negative "
            "age would select every snapshot, including ones in use."
        )
    counts = {"found": 0, "deleted": 0, "skipped": 0, "failed": 0}
    prefix = owner_snapshot_prefix()
    if prefix is None:
        return counts

    from datetime import UTC, datetime

    if client is None:
        from benchflow.sandbox.daytona import build_sync_client

        client = build_sync_client()
    now = datetime.now(UTC)
    # Collect first: deleting while paginating would shift later pages.
    candidates: list[tuple[Any, float]] = []
    page = 1
    while True:
        result = client.snapshot.list(page=page, limit=100)
        for snap in result.items:
            counts["found"] += 1
            name = getattr(snap, "name", None)
            created_at = _parse_sandbox_timestamp(getattr(snap, "created_at", None))
            if not isinstance(name, str) or not name.startswith(prefix):
                counts["skipped"] += 1
                continue
            if created_at is None:
                logger.warning(
                    "Daytona snapshot %s has no parseable created_at; skipping", name
                )
                counts["skipped"] += 1
                continue
            candidates.append((snap, (now - created_at).total_seconds() / 60))
        if page >= (getattr(result, "total_pages", None) or 1):
            break
        page += 1
    for snap, age_minutes in candidates:
        will_delete = age_minutes >= max_age_minutes
        if on_decision is not None:
            on_decision(snap, age_minutes, will_delete)
        if not will_delete:
            counts["skipped"] += 1
            continue
        if dry_run:
            counts["deleted"] += 1
            continue
        try:
            client.snapshot.delete(snap)
            counts["deleted"] += 1
        except Exception:
            logger.warning("Failed to delete snapshot %s", snap.name)
            counts["failed"] += 1
    return counts
