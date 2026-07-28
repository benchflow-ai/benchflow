"""Reclaim stale BenchFlow-managed AgentCore runtimes.

Runtimes are shared across rollouts and deliberately outlive the run that
created them, so nothing deletes them on the hot path. They are also a scarce
resource: *Total Agents per Account* defaults to 100, and a full SkillsBench
matrix (one image per task per skill arm) can exceed that. Left alone, a few
large runs would exhaust the quota and every later ``CreateAgentRuntime``
would fail.

This mirrors ``daytona_reaper``: only resources carrying BenchFlow's managed
tag are ever considered, so a runtime someone else created in the same account
is never touched.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from benchflow.sandbox.agentcore_provisioning import (
    LEASE_TAG,
    MANAGED_TAG,
    MANAGED_VALUE,
)

logger = logging.getLogger("benchflow").getChild("agentcore-reaper")

REAP_DEFAULT_MAX_AGE_MIN = 1440


@dataclass
class ReapReport:
    """What a reap pass considered, skipped, and deleted."""

    scanned: int = 0
    deleted: list[str] = field(default_factory=list)
    skipped_unmanaged: int = 0
    skipped_recent: int = 0
    skipped_active: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"scanned={self.scanned} deleted={len(self.deleted)} "
            f"unmanaged={self.skipped_unmanaged} recent={self.skipped_recent} "
            f"active={self.skipped_active} errors={len(self.errors)}"
        )


def _runtime_timestamp(runtime: Mapping[str, Any]) -> datetime | None:
    """Best available age signal for a runtime, or None if there is none.

    ``ListAgentRuntimes`` returns **only** ``lastUpdatedAt`` — there is no
    ``createdAt`` in the list shape, though ``GetAgentRuntime`` has both.
    Reading the wrong field silently yielded ``None`` for every runtime, which
    skipped the age comparison entirely and made a one-day cleanup delete
    minutes-old runtimes out from under a running matrix.

    Returning None here means "age unknown", and the caller must treat that as
    not-stale rather than as stale.
    """
    for field_name in ("createdAt", "lastUpdatedAt"):
        value = runtime.get(field_name)
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=UTC)
    return None


def _read_tags(control: Any, arn: str) -> dict[str, str] | None:
    """Tags for *arn*, or None when they cannot be read.

    None means "cannot prove anything about this runtime", which every caller
    must treat as a reason to leave it alone.
    """
    try:
        return control.list_tags_for_resource(resourceArn=arn).get("tags", {})
    except Exception:
        logger.debug("Could not read tags for %s; leaving it alone", arn)
        return None


def _lease_is_active(tags: Mapping[str, str], now: datetime) -> bool:
    """Whether a runtime is still leased, and so must not be deleted.

    A malformed lease counts as active. Cleanup is destructive and there is no
    API to enumerate active sessions, so an unparseable lease is a reason to
    stop, not to proceed.
    """
    raw = tags.get(LEASE_TAG)
    if not raw:
        # Every runtime BenchFlow provisions is leased before any session runs,
        # and that write is fatal if it fails. A managed runtime with no lease
        # is therefore unexplained, not idle — treat it as in use.
        logger.warning(
            "BenchFlow-managed AgentCore runtime has no lease tag; "
            "treating it as active rather than deleting it"
        )
        return True
    try:
        until = datetime.fromisoformat(raw)
    except ValueError:
        logger.warning("Unparseable AgentCore lease %r; treating as active", raw)
        return True
    if until.tzinfo is None:
        until = until.replace(tzinfo=UTC)
    return until > now


def reap_stale_runtimes(
    control: Any,
    *,
    max_age_minutes: int = REAP_DEFAULT_MAX_AGE_MIN,
    dry_run: bool = False,
    now: datetime | None = None,
) -> ReapReport:
    """Delete BenchFlow-managed runtimes older than *max_age_minutes*.

    Two independent gates must both pass: the runtime's lease must have
    expired, and its control-plane age must exceed *max_age_minutes*. The lease
    exists because session traffic does not move ``lastUpdatedAt``, so age
    alone cannot tell an idle runtime from one serving a matrix right now.
    """
    if max_age_minutes < 0:
        # A negative age puts the cutoff in the future, which makes every
        # runtime — including one created a second ago — look stale.
        raise ValueError(
            f"max_age_minutes must be >= 0, got {max_age_minutes}. A negative "
            "age would select every runtime, including active ones."
        )
    report = ReapReport()
    moment = now or datetime.now(UTC)
    cutoff = moment - timedelta(minutes=max_age_minutes)

    for page in control.get_paginator("list_agent_runtimes").paginate():
        for runtime in page.get("agentRuntimes", []):
            report.scanned += 1
            arn = runtime.get("agentRuntimeArn")
            runtime_id = runtime.get("agentRuntimeId")
            if not arn or not runtime_id:
                continue
            tags = _read_tags(control, arn)
            if tags is None or tags.get(MANAGED_TAG) != MANAGED_VALUE:
                report.skipped_unmanaged += 1
                continue
            if _lease_is_active(tags, moment):
                report.skipped_active += 1
                continue

            stamp = _runtime_timestamp(runtime)
            if stamp is None or stamp > cutoff:
                # No usable age means we cannot prove the runtime is stale, and
                # a runtime in use by a live matrix looks exactly like one that
                # is idle. Fail closed.
                report.skipped_recent += 1
                continue

            if dry_run:
                report.deleted.append(runtime_id)
                continue
            try:
                control.delete_agent_runtime(agentRuntimeId=runtime_id)
                report.deleted.append(runtime_id)
                logger.info("Reaped AgentCore runtime %s", runtime_id)
            except Exception as exc:
                report.errors.append(f"{runtime_id}: {exc}")
                logger.warning(
                    "Failed to reap AgentCore runtime %s: %s", runtime_id, exc
                )

    return report
