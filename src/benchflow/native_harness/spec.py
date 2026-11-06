"""The native-harness extension point: a command builder and a parser per CLI.

A native harness runs an agent's own command-line program in its headless JSON
mode instead of through an ACP adapter. Everything around the process is shared
with the ACP path: the sandbox, the model proxy, credentials, skills, the idle
watchdog and the trajectory (the parser emits the same ``session/update``
payloads an ACP adapter sends, and an :class:`~benchflow.acp.session.ACPSession`
records them). What differs per CLI is exactly two things, and this module
declares both:

* **the command builder** (:attr:`NativeHarness.build_launch`): the CLI's
  arguments for one turn, given a :class:`NativeTurn` and the launch
  environment (read-only; Codex reads its CODEX_CONFIG there);
* **the parser** (:attr:`NativeHarness.new_parser`, called with the working
  directory and the turn number): a :class:`NativeParser` that turns the CLI's
  JSON lines into ACP session updates and, at the end of the turn, a
  :class:`NativeTurnOutcome`.

Adding another CLI (Gemini CLI, OpenCode) means writing those two and one
:class:`NativeHarness` entry in :mod:`benchflow.native_harness.registry`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from benchflow.acp.types import McpServerSpec, StopReason
from benchflow.agents.errors import AgentProtocolError
from benchflow.agents.protocol import AgentCapabilities

HARNESS_ACP = "acp"
HARNESS_NATIVE = "native"
HARNESSES = (HARNESS_ACP, HARNESS_NATIVE)


class NativeHarnessError(AgentProtocolError):
    """The CLI reported a failed turn, or exited before it reported a result.

    A subclass of :class:`AgentProtocolError`, like the ACP client's
    ``ACPError``, so the rollout classifies it the same way: the message starts
    with ``Native harness error`` (``benchflow._utils.scoring.classify_error``
    files it with ACP errors, and a provider failure the proxy saw is appended
    to it as for an ACP error).
    """

    def __init__(
        self,
        cli: str,
        message: str,
        *,
        exit_code: int | None = None,
        agent_text: str | None = None,
        rate_limit: dict[str, Any] | None = None,
    ):
        self.cli = cli
        self.exit_code = exit_code
        # The CLI's own words, unwrapped and undecorated: a caller that
        # classifies a failure by its wording (a subscription usage limit)
        # must read this, not ``message``, which is prefixed and suffixed.
        self.agent_text = agent_text if agent_text is not None else message
        # The CLI's record of a rejected usage limit, when it reported one
        # (see NativeTurnOutcome.rate_limit).
        self.rate_limit = rate_limit
        self.message = f"Native harness error ({cli}): {message}"
        super().__init__(self.message)


@dataclass(frozen=True)
class NativeTurn:
    """What a command builder needs to launch one turn.

    ``resume_id`` continues an existing CLI session (every turn after the
    first, and a branch child resuming its parent). ``new_session_id`` names
    the session a first turn creates, for CLIs that accept a caller-chosen id
    (Claude Code's ``--session-id``); CLIs that pick their own id report it in
    their first event instead. ``model`` is set only when the launch
    environment does not already select the model (see
    ``connect_native``).
    """

    cwd: str
    resume_id: str | None = None
    new_session_id: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    mcp_servers: tuple[McpServerSpec, ...] = ()


@dataclass(frozen=True)
class NativeLaunch:
    """One turn's command line (after the executable) and extra environment."""

    argv: tuple[str, ...]
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass
class NativeTurnOutcome:
    """How a turn ended, as the CLI's own final event reported it.

    ``usage`` is this turn's token usage in the ACP ``PromptResponse.usage``
    field names (``input_tokens``, ``output_tokens``, ``cached_read_tokens``,
    ``cached_write_tokens``, ``thought_tokens``, ``total_tokens``), or None
    when the CLI reported none. A CLI that reports a running total over its
    session instead (Codex: the thread's, resumed turns included) sets
    ``usage_total`` and leaves ``usage`` None; the client takes the turn's
    share from consecutive totals. ``cost_usd`` is the CLI's own list-price
    estimate; it is kept for evidence, never used as the run's cost.
    ``error`` is set when the CLI reported a failed turn. ``agent_text`` is
    the same failure in the CLI's own words, before this package adds anything
    to it (Claude Code's ``(HTTP 429)``), for a reader that matches on what the
    agent said. ``rate_limit`` is the CLI's own record of a rejected usage
    limit when it reports one (Claude Code's ``rate_limit_event``:
    ``{"status": "rejected", "rateLimitType": "seven_day", "resetsAt":
    <unix time>}``), which names the window and the reset without reading the
    message (``You've hit your weekly limit · resets ...``).
    """

    stop_reason: StopReason | None = None
    usage: dict[str, int] | None = None
    usage_total: dict[str, int] | None = None
    cost_usd: float | None = None
    error: str | None = None
    agent_text: str | None = None
    rate_limit: dict[str, Any] | None = None
    session_id: str | None = None
    completed: bool = False


class NativeParser(Protocol):
    """Turns one turn's JSON lines into ACP session updates.

    ``feed`` receives each decoded JSON object in order and returns the ACP
    ``session/update`` payloads (the ``update`` object, with ``sessionUpdate``)
    it implies; the client applies them to the session. ``outcome`` is read
    once the process has exited.
    """

    @property
    def session_id(self) -> str | None: ...

    def feed(self, event: dict[str, Any]) -> list[dict[str, Any]]: ...

    def outcome(self) -> NativeTurnOutcome: ...


@dataclass(frozen=True)
class NativeHarness:
    """One agent's headless CLI: where it lives, how to launch it, how to read it.

    ``agent`` is the registry entry this harness belongs to (``harness`` is a
    run option of that entry, not another agent). ``package`` is the exact npm
    pin and ``version_output`` what ``<executable> --version`` must print, so
    a run never drives a CLI other than the pinned one. ``install_cmd`` is
    what the harness adds to the agent's own install, empty when the agent's
    install already brings the CLI. ``owns_model_via_env`` names the variable
    that selects the model when BenchFlow's proxy routes the run (the launch
    then passes no model flag).
    """

    agent: str
    cli: str
    executable: str
    package: str
    version_output: str
    install_cmd: str
    build_launch: Callable[[NativeTurn, Mapping[str, str]], NativeLaunch]
    new_parser: Callable[[str, int], NativeParser]
    capabilities: AgentCapabilities
    efforts: frozenset[str] = frozenset()
    owns_model_via_env: str = ""
    # Whether the first turn can create the session under an id BenchFlow
    # chooses (Claude Code's --session-id); otherwise the CLI reports its id.
    accepts_session_id: bool = False

    @property
    def version(self) -> str:
        return self.package.rpartition("@")[2]


def acp_stop_reason(value: str | None, default: StopReason) -> StopReason:
    """The ACP ``StopReason`` member for a value, or ``default``."""
    try:
        return StopReason(value) if value else default
    except ValueError:
        return default
