"""Structured diagnostics for rollout failures (issues #503, #504).

The Single Source of Truth for diagnostic events that need to surface in
``result.json``, the job summary, and the e2e check_results auditor. Each
diagnostic kind owns its own dataclass; the dataclass owns its JSON field
name, the error_category it pairs with, the summary warning template, and
the per-task issue line check_results renders.

Adding a new diagnostic is therefore a single edit: declare a new subclass
of :class:`Diagnostic` with ``field``/``category``/``summary_description``/
``format_issue`` set. The result builder, summary, and check_results all
discover it through :data:`DIAGNOSTIC_REGISTRY`.

The diagnostics are emitted at the source as typed exceptions
(:class:`IdleTimeoutError`, :class:`TransportClosedError`,
:class:`SandboxStartupError`) carrying a structured ``.diagnostic`` attribute,
so downstream consumers never reverse-engineer fields from
human-readable error strings. This replaces the regex-based
``rollout._parse_transport_error`` shim that issue #504 flagged as brittle.

The module has no optional-dep imports — base installs can reach the typed
exceptions without pulling Daytona/Modal SDKs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Literal

DIAGNOSTIC_REASON_IDLE_TIMEOUT = "idle_timeout"
DIAGNOSTIC_REASON_WALL_CLOCK_TIMEOUT = "wall_clock_timeout"
DIAGNOSTIC_REASON_SANDBOX_STARTUP_FAILED = "sandbox_startup_failed"
DIAGNOSTIC_REASON_TRANSPORT_CLOSED = "transport_closed"

DiagnosticReason = Literal[
    "idle_timeout",
    "wall_clock_timeout",
    "sandbox_startup_failed",
    "transport_closed",
]

# Diagnostic value objects


class AgentPromptTimeoutError(TimeoutError):
    """BenchFlow-owned prompt wall-clock timeout with captured ACP state.

    This is distinct from provider/client ``TimeoutError`` exceptions. It is
    raised only when BenchFlow's prompt budget expires and the ACP prompt task
    can be cancelled/drained cleanly enough to snapshot the session. The
    attached diagnostic says whether that snapshot is a complete terminal
    timeout trajectory or still has pending tool calls and must remain partial.
    """

    def __init__(
        self,
        message: str,
        *,
        trajectory: list[dict],
        diagnostic: AgentPromptTimeoutDiagnostic,
        executed_prompts: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.trajectory = trajectory
        self.diagnostic = diagnostic
        self.n_tool_calls = diagnostic.n_tool_calls
        self.terminal_trajectory_complete = diagnostic.terminal_trajectory_complete
        self.executed_prompts = executed_prompts or []


@dataclass
class Diagnostic:
    """Base class for structured diagnostic events.

    Subclasses declare the JSON field name they serialize to, the
    ``error_category`` from :mod:`benchflow._utils.scoring` that produces
    them, and how they render into summary warnings and check_results
    invalidation lines.
    """

    # Class-level metadata (overridden by subclasses)
    # Field name under which to_dict() lands in result.json.
    field: ClassVar[str] = ""
    # error_category (from _utils.scoring) that this diagnostic backs.
    # ``None`` means the diagnostic doesn't surface a top-level error
    # category (e.g. verifier_timeout's category lives on verifier_error,
    # not on the agent's ``error`` channel).
    category: ClassVar[str | None] = None
    # Channel the category lives on — "error" or "verifier_error".
    channel: ClassVar[str] = "error"
    # Human description for the summary warning ("hit idle timeout",
    # "lost transport (pipe closed / rc=255)" …).
    summary_description: ClassVar[str] = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize for result.json, omitting unpopulated optional fields."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def format_issue(self, task_name: str) -> str:
        """Render the per-task line check_results emits for this diagnostic."""
        raise NotImplementedError

    @classmethod
    def format_issue_from_dict(cls, task_name: str, info: dict[str, Any]) -> str:
        """Render a per-task issue line from a deserialized dict.

        check_results works off the round-tripped result.json, not the live
        dataclass; this lets it reuse the dataclass's formatting rules
        without having to reconstruct the instance.
        """
        return cls(
            **{k: v for k, v in info.items() if k in cls._init_fields()}
        ).format_issue(task_name)

    @classmethod
    def _init_fields(cls) -> set[str]:
        """Names of fields the dataclass actually takes — drops legacy/extra keys."""
        from dataclasses import fields

        return {f.name for f in fields(cls)}


@dataclass
class IdleTimeoutDiagnostic(Diagnostic):
    """Agent went silent — no tool call, message, or thought arrived in time."""

    reason: Literal["idle_timeout"] = DIAGNOSTIC_REASON_IDLE_TIMEOUT
    idle_timeout_sec: int = 0
    idle_duration_sec: int = 0
    wall_clock_elapsed_sec: int = 0
    n_tool_calls: int = 0
    n_message_chunks: int = 0
    n_thought_chunks: int = 0
    last_activity_at: str = ""

    field: ClassVar[str] = "idle_timeout_info"
    category: ClassVar[str | None] = "idle_timeout"
    summary_description: ClassVar[str] = "hit idle timeout"

    def format_issue(self, task_name: str) -> str:
        return (
            f"{task_name}: idle timeout after "
            f"{self.idle_duration_sec}s idle "
            f"({self.n_tool_calls} tool calls, "
            f"{self.wall_clock_elapsed_sec}s wall)"
        )


@dataclass
class AgentPromptTimeoutDiagnostic(Diagnostic):
    """BenchFlow hit the prompt wall-clock budget and wrote timeout evidence."""

    reason: Literal["wall_clock_timeout"] = DIAGNOSTIC_REASON_WALL_CLOCK_TIMEOUT
    timeout_sec: float = 0.0
    n_tool_calls: int = 0
    pending_tool_call_ids: list[str] = field(default_factory=list)
    terminal_event_recorded: bool = False
    terminal_trajectory_complete: bool = False

    field: ClassVar[str] = "agent_timeout_info"
    category: ClassVar[str | None] = "timeout"
    summary_description: ClassVar[str] = "hit agent wall-clock timeout"

    def format_issue(self, task_name: str) -> str:
        pending = len(self.pending_tool_call_ids)
        complete = "complete" if self.terminal_trajectory_complete else "partial"
        return (
            f"{task_name}: agent wall-clock timeout after "
            f"{self.timeout_sec}s ({complete}, {self.n_tool_calls} tool calls, "
            f"{pending} pending)"
        )


@dataclass
class SandboxStartupDiagnostic(Diagnostic):
    """Sandbox creation failed before the rollout ever ran."""

    reason: Literal["sandbox_startup_failed"] = DIAGNOSTIC_REASON_SANDBOX_STARTUP_FAILED
    sandbox_id: str | None = None
    sandbox_state: str | None = None
    attempts: int = 0
    build_timeout_sec: float | None = None
    raw_message: str = ""

    field: ClassVar[str] = "sandbox_startup_info"
    category: ClassVar[str | None] = "sandbox_setup"
    summary_description: ClassVar[str] = "failed during sandbox startup"

    def format_issue(self, task_name: str) -> str:
        return (
            f"{task_name}: sandbox startup failed (sandbox_id={self.sandbox_id or '?'}, "
            f"state={self.sandbox_state or '?'}, attempts={self.attempts}, "
            f"build_timeout_sec={self.build_timeout_sec if self.build_timeout_sec is not None else '?'})"
        )


@dataclass
class TransportClosedDiagnostic(Diagnostic):
    """ACP transport pipe closed — process died or remote session was killed.

    Replaces the regex-parsed dict that ``_parse_transport_error`` used to
    reconstruct from the stringified ``ConnectionError`` (issue #504).
    """

    reason: Literal["transport_closed"] = DIAGNOSTIC_REASON_TRANSPORT_CLOSED
    raw_message: str = ""
    process_exit_code: int | None = None
    process_pid: int | None = None
    transport_diagnosis: str = "unknown"
    stderr_snippet: str | None = None
    # Populated by Rollout._probe_sandbox_health() — sandbox liveness
    # checks happen after the exception lands so they live alongside the
    # source-emitted fields here.
    sandbox_reachable: bool | None = None
    sandbox_probe_rc: int | None = None
    sandbox_probe_stdout: str | None = None
    sandbox_probe_error: str | None = None
    sandbox_probe_error_type: str | None = None
    sandbox_probe_traceback: str | None = None

    field: ClassVar[str] = "transport_error_info"
    category: ClassVar[str | None] = "pipe_closed"
    summary_description: ClassVar[str] = "lost transport (pipe closed / rc=255)"

    def format_issue(self, task_name: str) -> str:
        rc = self.process_exit_code if self.process_exit_code is not None else "?"
        reachable = "?" if self.sandbox_reachable is None else self.sandbox_reachable
        return (
            f"{task_name}: transport closed (rc={rc}, "
            f"diagnosis={self.transport_diagnosis}, sandbox_reachable={reachable})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize, dropping fields that were never populated.

        The legacy regex-parser only included fields it could match — drop
        the same way so result.json stays compact and tests that compare
        explicit dicts continue to work.
        """
        return {k: v for k, v in super().to_dict().items() if k != "raw_message" or v}


@dataclass
class VerifierTimeoutDiagnostic(Diagnostic):
    """Verifier exceeded its timeout budget."""

    timeout_budget_sec: float = 0.0
    elapsed_sec: float = 0.0
    task_name: str = ""

    field: ClassVar[str] = "verifier_timeout_info"
    category: ClassVar[str | None] = "verifier_timeout"
    channel: ClassVar[str] = "verifier_error"
    summary_description: ClassVar[str] = "had verifier timeouts"

    def format_issue(self, task_name: str) -> str:
        return (
            f"{task_name}: verifier timed out "
            f"(budget={self.timeout_budget_sec}s, elapsed={self.elapsed_sec}s) — "
            f"measurement invalid (verifier never produced reward)"
        )


@dataclass
class ProviderApiErrorDiagnostic(Diagnostic):
    """Every captured provider API request failed — no model response ever
    reached the agent (rate limit, auth rejection, quota, model-not-found,
    5xx). Proxy-proven: built from the usage proxy's captured exchange status
    codes only (#546/#564 — bodies/headers are never read)."""

    subcategory: str = "provider_error"
    transient: bool = False
    dominant_status: int | None = None
    status_counts: dict[str, int] | None = None
    total_requests: int = 0
    failed_requests: int = 0
    fingerprint: str = ""

    field: ClassVar[str] = "api_error_info"
    category: ClassVar[str | None] = "api_error"
    summary_description: ClassVar[str] = "failed on provider API errors"

    def format_issue(self, task_name: str) -> str:
        kind = "transient" if self.transient else "permanent"
        return (
            f"{task_name}: provider api error [{self.subcategory}/{kind}] "
            f"HTTP {self.dominant_status} on "
            f"{self.failed_requests}/{self.total_requests} requests — "
            f"measurement invalid (agent never got a model response)"
        )


@dataclass
class SuspectedApiErrorDiagnostic(Diagnostic):
    """Zero-signal rollout: the agent ended its turn with zero tokens AND
    zero tool calls and no error — the signature of a provider API failure
    swallowed inside the agent (e.g. a model id rejected against the agent's
    own catalog before any request is issued)."""

    total_tokens: int = 0
    n_tool_calls: int = 0
    total_requests: int = 0
    failed_requests: int = 0

    field: ClassVar[str] = "suspected_api_error_info"
    category: ClassVar[str | None] = "suspected_api_error"
    summary_description: ClassVar[str] = (
        "ended with zero model/tool activity (suspected provider api error)"
    )

    def format_issue(self, task_name: str) -> str:
        return (
            f"{task_name}: suspected provider api error — agent ended with "
            f"{self.total_tokens} tokens and {self.n_tool_calls} tool calls "
            f"({self.failed_requests}/{self.total_requests} captured requests "
            f"failed) — measurement suspect"
        )


# Public registry — every diagnostic kind goes here exactly once.
DIAGNOSTIC_REGISTRY: tuple[type[Diagnostic], ...] = (
    IdleTimeoutDiagnostic,
    AgentPromptTimeoutDiagnostic,
    SandboxStartupDiagnostic,
    TransportClosedDiagnostic,
    VerifierTimeoutDiagnostic,
    ProviderApiErrorDiagnostic,
    SuspectedApiErrorDiagnostic,
)

# field_name → Diagnostic class, for check_results lookup.
DIAGNOSTIC_BY_FIELD: dict[str, type[Diagnostic]] = {
    d.field: d for d in DIAGNOSTIC_REGISTRY
}


# Diagnostic-carrying exceptions


class IdleTimeoutError(TimeoutError):
    """Raised by the idle watchdog when the agent stops producing activity.

    Carries an :class:`IdleTimeoutDiagnostic` instance via ``.diagnostic`` —
    serialize via ``exc.diagnostic.to_dict()`` to recover the result.json
    payload (issue #503).
    """

    def __init__(self, message: str, diagnostic: IdleTimeoutDiagnostic) -> None:
        super().__init__(message)
        self.diagnostic: IdleTimeoutDiagnostic = diagnostic


class TransportClosedError(ConnectionError):
    """Raised at the source (``sandbox/process.py``) when an ACP transport dies.

    Carries a structured :class:`TransportClosedDiagnostic`, so downstream
    code never has to regex-parse the human-readable error string to
    recover ``rc``, ``pid``, ``diagnosis``, or ``stderr_snippet``
    (issue #504).
    """

    def __init__(self, message: str, diagnostic: TransportClosedDiagnostic) -> None:
        super().__init__(message)
        self.diagnostic: TransportClosedDiagnostic = diagnostic


# SandboxStartupError lives in ``benchflow.sandbox.protocol`` (not here)
# so a base install can import it without pulling Daytona/Modal SDKs —
# see ``tests/test_base_install_imports.py``. It carries a
# :class:`SandboxStartupDiagnostic` via its ``.diagnostic`` attribute.


# Collector — replaces 4 parallel _*_info slots on Rollout


class RolloutDiagnostics:
    """Collects diagnostics observed during one rollout.

    Replaces the ``_idle_timeout_info`` / ``_sandbox_startup_info`` /
    ``_transport_error_info`` / ``_verifier_timeout_info`` quadruple on
    ``Rollout`` with a single keyed bag. Serialization to result.json
    keeps the legacy flat field names so existing tooling does not break
    (issue #503).
    """

    def __init__(self) -> None:
        self._events: dict[str, Diagnostic] = {}

    def set(self, diagnostic: Diagnostic) -> None:
        """Record a diagnostic, keyed by its result.json field name."""
        self._events[diagnostic.field] = diagnostic

    def get(self, field_name: str) -> Diagnostic | None:
        return self._events.get(field_name)

    def capture_timeout(self, exc: BaseException) -> None:
        """Extract a structured timeout diagnostic from a TimeoutError, if present.

        Idle-watchdog and BenchFlow-owned wall-clock prompt timeouts both carry
        typed diagnostics. Provider/client ``TimeoutError`` exceptions do not.
        """
        diag = getattr(exc, "diagnostic", None)
        if isinstance(diag, (IdleTimeoutDiagnostic, AgentPromptTimeoutDiagnostic)):
            self.set(diag)

    def capture_idle(self, exc: BaseException) -> None:
        """Backward-compatible alias for older call sites."""
        self.capture_timeout(exc)

    def capture_transport(self, exc: ConnectionError) -> None:
        """Record the transport-closed diagnostic.

        Typed :class:`TransportClosedError` raised by
        ``sandbox/process.py`` carries the structured fields directly
        (issue #504). Bare ``ConnectionError`` s (third-party SDK paths)
        fall back to a minimal diagnostic so result.json still has a
        populated ``transport_error_info`` block.
        """
        diag = getattr(exc, "diagnostic", None)
        if isinstance(diag, TransportClosedDiagnostic):
            self.set(diag)
            return
        self.set(
            TransportClosedDiagnostic(
                raw_message=str(exc)[:500], transport_diagnosis="unknown"
            )
        )

    def to_result_fields(self) -> dict[str, dict[str, Any] | None]:
        """Return the flat ``{field_name: dict|None}`` view for result.json.

        Includes every registered diagnostic field — absent ones serialize
        to ``None`` so the result schema stays stable.
        """
        return {
            d.field: self._events[d.field].to_dict()
            if d.field in self._events
            else None
            for d in DIAGNOSTIC_REGISTRY
        }

    def category_for_channel(self, channel: str) -> str | None:
        """Return the structured error category for a result channel, if any."""
        for diagnostic_cls in DIAGNOSTIC_REGISTRY:
            diagnostic = self._events.get(diagnostic_cls.field)
            if (
                diagnostic is not None
                and diagnostic_cls.channel == channel
                and diagnostic_cls.category
            ):
                return diagnostic_cls.category
        return None

    # Convenience accessors for callers that need to enrich an in-flight
    # diagnostic (e.g. probe_sandbox_health adds sandbox_reachable to
    # the transport diagnostic after the exception lands).
    @property
    def transport_closed(self) -> TransportClosedDiagnostic | None:
        d = self._events.get(TransportClosedDiagnostic.field)
        return d if isinstance(d, TransportClosedDiagnostic) else None


# Summary / check_results helpers driven by the registry


def summary_warning(diagnostic_cls: type[Diagnostic], count: int, total: int) -> str:
    """Format the one-line warning the job summary emits for a diagnostic."""
    pct = (count / total * 100) if total else 0.0
    return (
        f"{count} tasks ({pct:.0f}%) {diagnostic_cls.summary_description} — "
        f"check {diagnostic_cls.field} in result.json for diagnostics"
    )


def format_issue_for_field(
    field_name: str, task_name: str, info: dict[str, Any] | None
) -> str:
    """Render the per-task check_results invalidation line for a diagnostic.

    Falls back to a generic format when the diagnostic dict is absent —
    matches the legacy ``infra_errors.append(f"{task}: {err}")`` branch.
    """
    cls = DIAGNOSTIC_BY_FIELD.get(field_name)
    if cls is None or not info:
        return f"{task_name}: (no {field_name})"
    return cls.format_issue_from_dict(task_name, info)


__all__ = [
    "Diagnostic",
    "IdleTimeoutDiagnostic",
    "SandboxStartupDiagnostic",
    "TransportClosedDiagnostic",
    "VerifierTimeoutDiagnostic",
    "DIAGNOSTIC_REGISTRY",
    "DIAGNOSTIC_BY_FIELD",
    "IdleTimeoutError",
    "TransportClosedError",
    "RolloutDiagnostics",
    "summary_warning",
    "format_issue_for_field",
]
