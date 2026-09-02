"""Idle-watchdog state and pending-tool grace policy for ACP prompts."""

from __future__ import annotations

from collections.abc import Sized
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from benchflow.diagnostics import IdleTimeoutDiagnostic, IdleTimeoutError

# A pending tool call defers the idle watchdog for at most this many idle
# budgets while staying silent. Streaming updates restart the grace for their
# own pending call; unrelated or terminal updates do not.
PENDING_GRACE_MULTIPLIER = 3


class WatchdogSession(Protocol):
    """The small ACPSession surface consumed by the idle watchdog."""

    tool_calls: Sized
    message_chunks: Sized
    thought_chunks: Sized
    tool_call_update_count: int

    def pending_tool_call_state(self) -> tuple[tuple[str, int], ...]: ...


def _activity_count(session: WatchdogSession) -> int:
    return (
        len(session.tool_calls)
        + len(session.message_chunks)
        + len(session.thought_chunks)
    )


@dataclass
class IdleWatchdog:
    """Track prompt activity and enforce bounded pending-tool deferral."""

    idle_timeout_sec: int
    wall_timeout_sec: int
    started_at: float
    last_progress_at: float
    last_activity_at: datetime
    last_activity_count: int
    pending_state: tuple[tuple[str, int], ...]
    pending_last_progress_at: dict[str, float]
    pending_last_update_at: dict[str, datetime]
    pending_set_last_changed_at: datetime | None

    @classmethod
    def start(
        cls,
        session: WatchdogSession,
        *,
        idle_timeout_sec: int,
        wall_timeout_sec: int,
        now: float,
        wall_now: datetime | None = None,
    ) -> IdleWatchdog:
        """Create a watchdog snapshot at prompt start."""
        observed_at = wall_now or datetime.now(UTC)
        pending_state = session.pending_tool_call_state()
        return cls(
            idle_timeout_sec=idle_timeout_sec,
            wall_timeout_sec=wall_timeout_sec,
            started_at=now,
            last_progress_at=now,
            last_activity_at=observed_at,
            last_activity_count=_activity_count(session),
            pending_state=pending_state,
            pending_last_progress_at={call_id: now for call_id, _ in pending_state},
            pending_last_update_at={},
            pending_set_last_changed_at=observed_at if pending_state else None,
        )

    @property
    def pending_grace_sec(self) -> int:
        return self.idle_timeout_sec * PENDING_GRACE_MULTIPLIER

    @property
    def poll_interval_sec(self) -> int:
        return max(
            1,
            min(
                30,
                self.idle_timeout_sec // 4,
                max(1, self.wall_timeout_sec // 4),
            ),
        )

    @property
    def deadline(self) -> float:
        return self.started_at + self.wall_timeout_sec

    @property
    def pending_ids(self) -> tuple[str, ...]:
        return tuple(call_id for call_id, _version in self.pending_state)

    @property
    def n_pending_tool_updates(self) -> int:
        return sum(version for _call_id, version in self.pending_state)

    def observe(
        self,
        session: WatchdogSession,
        *,
        now: float,
        wall_now: datetime | None = None,
    ) -> None:
        """Observe one poll and advance only the relevant grace clocks."""
        observed_at = wall_now or datetime.now(UTC)
        current_state = session.pending_tool_call_state()
        previous_versions = dict(self.pending_state)
        current_ids = tuple(call_id for call_id, _version in current_state)

        if current_ids != self.pending_ids:
            self.pending_set_last_changed_at = observed_at

        for call_id, version in current_state:
            if call_id not in previous_versions:
                self.pending_last_progress_at[call_id] = now
            if version > previous_versions.get(call_id, 0):
                self.pending_last_progress_at[call_id] = now
                self.pending_last_update_at[call_id] = observed_at

        current_id_set = set(current_ids)
        self.pending_last_progress_at = {
            call_id: progress_at
            for call_id, progress_at in self.pending_last_progress_at.items()
            if call_id in current_id_set
        }
        self.pending_last_update_at = {
            call_id: update_at
            for call_id, update_at in self.pending_last_update_at.items()
            if call_id in current_id_set
        }

        self.pending_state = current_state
        current_activity_count = _activity_count(session)
        if current_activity_count > self.last_activity_count:
            self.last_progress_at = now
            self.last_activity_at = observed_at
            self.last_activity_count = current_activity_count
        elif current_ids and not self.expired_pending_ids(now):
            # Silent tools may legitimately run longer than idle_timeout. The
            # grace, not unrelated session noise, owns this deferral.
            self.last_progress_at = now
            self.last_activity_at = observed_at

    def expired_pending_ids(self, now: float) -> tuple[str, ...]:
        """Pending calls whose own relevant progress grace has expired."""
        return tuple(
            call_id
            for call_id in self.pending_ids
            if now - self.pending_last_progress_at[call_id] >= self.pending_grace_sec
        )

    def idle_expired(self, now: float) -> bool:
        return bool(self.expired_pending_ids(now)) or (
            now - self.last_progress_at >= self.idle_timeout_sec
        )

    def wall_clock_expired(self, now: float) -> bool:
        return now > self.deadline

    def timeout_error(
        self,
        session: WatchdogSession,
        *,
        now: float,
        wall_now: datetime | None = None,
    ) -> IdleTimeoutError:
        """Build a truthful idle error for generic or pending-grace expiry."""
        fired_at = wall_now or datetime.now(UTC)
        pending_ids = list(self.pending_ids)
        expired_pending_ids = list(self.expired_pending_ids(now))
        pending_versions = dict(self.pending_state)
        n_expired_pending_updates = sum(
            pending_versions[call_id] for call_id in expired_pending_ids
        )
        idle_duration = (
            max(
                now - self.pending_last_progress_at[call_id]
                for call_id in expired_pending_ids
            )
            if expired_pending_ids
            else now - self.last_progress_at
        )
        expired_update_times = [
            self.pending_last_update_at[call_id]
            for call_id in expired_pending_ids
            if call_id in self.pending_last_update_at
        ]
        last_expired_update_at = (
            max(expired_update_times) if expired_update_times else None
        )
        diagnostic = IdleTimeoutDiagnostic(
            idle_timeout_sec=self.idle_timeout_sec,
            idle_duration_sec=int(idle_duration),
            wall_clock_elapsed_sec=int(now - self.started_at),
            n_tool_calls=len(session.tool_calls),
            n_message_chunks=len(session.message_chunks),
            n_thought_chunks=len(session.thought_chunks),
            last_activity_at=self.last_activity_at.isoformat(),
            pending_tool_call_ids=pending_ids,
            expired_pending_tool_call_ids=expired_pending_ids,
            pending_grace_sec=self.pending_grace_sec,
            pending_set_last_changed_at=(
                self.pending_set_last_changed_at.isoformat()
                if self.pending_set_last_changed_at is not None
                else None
            ),
            last_tool_update_at=(
                last_expired_update_at.isoformat()
                if last_expired_update_at is not None
                else None
            ),
            n_tool_call_updates=session.tool_call_update_count,
            n_pending_tool_call_updates=self.n_pending_tool_updates,
            n_expired_pending_tool_call_updates=n_expired_pending_updates,
        )
        if expired_pending_ids:
            set_age = _age(fired_at, self.pending_set_last_changed_at)
            update_age = _age(fired_at, last_expired_update_at)
            return IdleTimeoutError(
                f"Agent idle for {self.idle_timeout_sec}s: "
                f"{len(expired_pending_ids)} pending tool call(s) exceeded the "
                f"{self.pending_grace_sec}s pending grace "
                f"({', '.join(expired_pending_ids)}; pending set last changed "
                f"{set_age}, last pending-call update {update_age}, "
                f"{n_expired_pending_updates} relevant updates seen, "
                f"{len(session.tool_calls)} tool calls so far)",
                diagnostic,
            )
        return IdleTimeoutError(
            f"Agent idle for {self.idle_timeout_sec}s with no new tool call, "
            f"message, or thought "
            f"(last activity {int(now - self.last_progress_at)}s ago, "
            f"{len(session.tool_calls)} tool calls so far)",
            diagnostic,
        )


def _age(now: datetime, then: datetime | None) -> str:
    if then is None:
        return "never"
    return f"{int((now - then).total_seconds())}s ago"
