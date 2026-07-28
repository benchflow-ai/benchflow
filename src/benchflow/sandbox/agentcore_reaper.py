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

from benchflow.sandbox.agentcore_provisioning import MANAGED_TAG, MANAGED_VALUE

logger = logging.getLogger("benchflow").getChild("agentcore-reaper")

REAP_DEFAULT_MAX_AGE_MIN = 1440


@dataclass
class ReapReport:
    """What a reap pass considered, skipped, and deleted."""

    scanned: int = 0
    deleted: list[str] = field(default_factory=list)
    skipped_unmanaged: int = 0
    skipped_recent: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"scanned={self.scanned} deleted={len(self.deleted)} "
            f"unmanaged={self.skipped_unmanaged} recent={self.skipped_recent} "
            f"errors={len(self.errors)}"
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


def _is_benchflow_managed(control: Any, arn: str) -> bool:
    """True only when the runtime carries BenchFlow's managed tag.

    Fails closed: if the tags cannot be read, the runtime is treated as
    someone else's and left alone.
    """
    try:
        tags = control.list_tags_for_resource(resourceArn=arn).get("tags", {})
    except Exception:
        logger.debug("Could not read tags for %s; leaving it alone", arn)
        return False
    return tags.get(MANAGED_TAG) == MANAGED_VALUE


def reap_stale_runtimes(
    control: Any,
    *,
    max_age_minutes: int = REAP_DEFAULT_MAX_AGE_MIN,
    dry_run: bool = False,
    now: datetime | None = None,
) -> ReapReport:
    """Delete BenchFlow-managed runtimes older than *max_age_minutes*."""
    report = ReapReport()
    cutoff = (now or datetime.now(UTC)) - timedelta(minutes=max_age_minutes)

    for page in control.get_paginator("list_agent_runtimes").paginate():
        for runtime in page.get("agentRuntimes", []):
            report.scanned += 1
            arn = runtime.get("agentRuntimeArn")
            runtime_id = runtime.get("agentRuntimeId")
            if not arn or not runtime_id:
                continue
            if not _is_benchflow_managed(control, arn):
                report.skipped_unmanaged += 1
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
