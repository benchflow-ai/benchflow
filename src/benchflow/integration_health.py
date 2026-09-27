"""Detect agent integrations that broke silently.

A trial "did nothing useful" when, after the prompt, the agent made no tool
call, sent no message with text, produced no genuine thought (a harness shim's
diagnostic such as ``[openclaw stderr] …`` does not count) and reported no
output tokens. When that happens the reward the verifier gave the untouched
workspace says nothing about the agent: the trial is an execution failure of
the harness (``error_category = "agent_integration"``), unscored, with a named
cause:

- ``agent_auth``: an auth or billing failure in a diagnostic thought or an
  agent log (credit balance too low, invalid API key, 401, not logged in).
- ``agent_install``: an install or runtime failure (``Node.js vX+ is
  required``, command not found, Cannot find module, npm ERR!).
- ``truncated_trajectory``: the trajectory file has unparseable lines.
- ``empty_trajectory``: nothing at all after the prompt.
- ``immediate_exit``: the agent phase lasted under 10 seconds.
- ``no_activity``: the agent ran and produced nothing.

Runs that captured no events at all (not even the prompt) are not judged: a
flat-telemetry agent has that shape when healthy (PR #886). Control runs
(oracle, nop) are never judged.

Used at run time by ``Rollout._maybe_classify_api_error`` and at read time by
``bf.load_job`` / ``bf.load_trial`` for results written before it existed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from benchflow.diagnostics import AGENT_INTEGRATION, IntegrationFailureDiagnostic

__all__ = [
    "AGENT_INTEGRATION",
    "CAUSES",
    "IntegrationFailure",
    "IntegrationFailureDiagnostic",
    "diagnose",
    "diagnose_trial_dir",
]
IntegrationCause = Literal[
    "agent_auth",
    "agent_install",
    "truncated_trajectory",
    "empty_trajectory",
    "immediate_exit",
    "no_activity",
]
CAUSES: tuple[str, ...] = (
    "agent_auth",
    "agent_install",
    "truncated_trajectory",
    "empty_trajectory",
    "immediate_exit",
    "no_activity",
)
#: Causes that will recur on every trial of the batch (the circuit breaker
#: counts them); the others can be transient.
PERMANENT_CAUSES = frozenset({"agent_auth", "agent_install"})
IMMEDIATE_EXIT_SEC = 10.0
_CONTROL_AGENTS = frozenset({"oracle", "nop", "empty"})
_CONTROL_EVENTS = frozenset({"oracle", "nop"})
_EVIDENCE_LIMIT = 300
_LOG_LIMIT = 200_000

# Shim diagnostics that reach the trajectory as thoughts: "[openclaw stderr]",
# "[openclaw-acp-shim] …", "[agent stderr]" and the like.
_DIAGNOSTIC_THOUGHT = re.compile(r"^\s*\[[\w .-]*(?:stderr|shim)[\w .-]*\]", re.I)

_AUTH_MARKERS = re.compile(
    r"credit balance is too low|invalid[ _-]?x-api-key|invalid[ _]api[ _]key|"
    r"incorrect api key|api key not valid|authentication_error|"
    r"\b401\b|unauthori[sz]ed|not logged in|please run /login|"
    r"oauth token (?:has )?expired|token (?:has )?expired|insufficient_quota|"
    r"api key is missing|no api key",
    re.I,
)
_INSTALL_MARKERS = re.compile(
    r"node\.js v[\d.]+\+? is required|command not found|cannot find module|"
    r"modulenotfounderror|exec format error|npm err!|err_module_not_found|"
    r"no such file or directory.*(?:node|npx|python|bin/)|"
    r"error while loading shared libraries|glibc_[\d.]+' not found",
    re.I,
)


@dataclass(frozen=True)
class IntegrationFailure:
    """What the detector found."""

    cause: str
    evidence: str
    evidence_source: str
    activity: dict[str, Any] = field(default_factory=dict)
    agent_seconds: float | None = None

    def error_text(self) -> str:
        return f"agent integration failure [{self.cause}]: {self.evidence}"

    def diagnostic(
        self, *, reward_withheld: dict[str, Any] | None = None
    ) -> IntegrationFailureDiagnostic:
        return IntegrationFailureDiagnostic(
            cause=self.cause,
            evidence=self.evidence,
            evidence_source=self.evidence_source,
            tool_calls=int(self.activity.get("tool_calls") or 0),
            agent_messages=int(self.activity.get("agent_messages") or 0),
            agent_thoughts=int(self.activity.get("agent_thoughts") or 0),
            output_tokens=self.activity.get("output_tokens"),
            agent_seconds=self.agent_seconds,
            reward_withheld=reward_withheld,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.diagnostic().to_dict()


def _text(event: Mapping[str, Any]) -> str:
    text = event.get("text")
    if isinstance(text, str):
        return text
    content = event.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping) and isinstance(content.get("text"), str):
        return content["text"]
    return ""


def _is_prompt(event: Mapping[str, Any]) -> bool:
    return event.get("type") in ("user_message", "user_message_chunk")


def _marker_line(text: str, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(text)
    if match is None:
        return None
    start = text.rfind("\n", 0, match.start()) + 1
    end = text.find("\n", match.end())
    line = text[start : end if end != -1 else len(text)].strip()
    # Prefer the readable message over a JSON dump around it.
    return line[:_EVIDENCE_LIMIT]


def diagnose(
    events: Sequence[Mapping[str, Any]],
    *,
    agent: str | None,
    n_tool_calls: int,
    output_tokens: int | None,
    logs: Mapping[str, str],
    agent_seconds: float | None,
    truncated: bool = False,
    prompt_sent: bool = False,
) -> IntegrationFailure | None:
    """Name why an agent did nothing useful, or None when it did something.

    ``events`` are the ACP trajectory events; ``logs`` maps a log's name
    (``agent/openclaw.txt``) to its text. ``prompt_sent`` says a prompt was
    sent even if the trajectory does not record it (older results).
    """
    if (agent or "").strip().lower() in _CONTROL_AGENTS:
        return None
    if not events and not truncated:
        return None
    # Old results may lack the agent name; the control's own trajectory
    # event still says what ran.
    if any(isinstance(e, Mapping) and e.get("type") in _CONTROL_EVENTS for e in events):
        return None
    prompt_seen = False
    after_prompt: list[Mapping[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        if _is_prompt(event):
            prompt_seen = True
            continue
        after_prompt.append(event)
    if not (prompt_seen or prompt_sent or truncated):
        return None
    tool_calls = int(n_tool_calls or 0) + sum(
        1 for e in after_prompt if e.get("type") in ("tool_call", "tool_call_update")
    )
    messages = [
        e
        for e in after_prompt
        if e.get("type") in ("agent_message", "agent_message_chunk")
        and _text(e).strip()
    ]
    thoughts = [
        e
        for e in after_prompt
        if e.get("type") in ("agent_thought", "agent_thought_chunk")
        and _text(e).strip()
    ]
    genuine = [t for t in thoughts if not _DIAGNOSTIC_THOUGHT.match(_text(t))]
    activity = {
        "tool_calls": tool_calls,
        "agent_messages": len(messages),
        "agent_thoughts": len(genuine),
        "output_tokens": output_tokens,
    }
    if tool_calls or messages or genuine or (output_tokens or 0) > 0:
        return None

    def found(cause: str, evidence: str, source: str) -> IntegrationFailure:
        return IntegrationFailure(
            cause=cause,
            evidence=evidence,
            evidence_source=source,
            activity=activity,
            agent_seconds=agent_seconds,
        )

    sources: list[tuple[str, str]] = [
        ("trajectory agent_thought", _text(t)) for t in thoughts
    ]
    sources += [(name, text[-_LOG_LIMIT:]) for name, text in logs.items()]
    for cause, pattern in (
        ("agent_auth", _AUTH_MARKERS),
        ("agent_install", _INSTALL_MARKERS),
    ):
        for source, text in sources:
            line = _marker_line(text, pattern)
            if line:
                return found(cause, line, source)
    seconds = f"{agent_seconds:.1f} s" if agent_seconds is not None else "unknown time"
    nothing = (
        f"0 tool calls, 0 messages, 0 thoughts"
        f"{', 0 output tokens' if output_tokens == 0 else ''} after the prompt "
        f"in {seconds}"
    )
    if truncated:
        return found(
            "truncated_trajectory",
            f"trajectory has unparseable lines; {nothing}",
            "trajectory",
        )
    if not after_prompt:
        return found(
            "empty_trajectory", f"no event after the prompt ({seconds})", "trajectory"
        )
    if agent_seconds is not None and agent_seconds < IMMEDIATE_EXIT_SEC:
        return found("immediate_exit", nothing, "timing")
    return found("no_activity", nothing, "trajectory")


def read_agent_logs(trial_dir: Path) -> dict[str, str]:
    """The agent's log files (``agent/*.txt``), last 200 kB of each."""
    logs: dict[str, str] = {}
    folder = Path(trial_dir) / "agent"
    if not folder.is_dir():
        return logs
    for path in sorted(folder.glob("*.txt")):
        try:
            with path.open("rb") as handle:
                size = path.stat().st_size
                handle.seek(max(0, size - _LOG_LIMIT))
                logs[f"agent/{path.name}"] = handle.read().decode(
                    "utf-8", errors="replace"
                )
        except OSError:
            continue
    return logs


def _read_events(path: Path) -> tuple[list[dict[str, Any]], bool] | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    events: list[dict[str, Any]] = []
    truncated = False
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            truncated = True
            continue
        if isinstance(event, dict):
            events.append(event)
    return events, truncated


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)


def _agent_seconds(trial_dir: Path, result: Mapping[str, Any]) -> float | None:
    timing = result.get("timing")
    if not isinstance(timing, Mapping):
        try:
            timing = json.loads((trial_dir / "timing.json").read_text())
        except (OSError, ValueError):
            timing = {}
    value = timing.get("agent_execution") if isinstance(timing, Mapping) else None
    return float(value) if isinstance(value, int | float) else None


def stored_integration_failure(result: Mapping[str, Any]) -> dict[str, Any] | None:
    info = result.get("integration_failure_info")
    return dict(info) if isinstance(info, Mapping) else None


def diagnose_trial_dir(
    trial_dir: str | Path, result: Mapping[str, Any]
) -> IntegrationFailure | None:
    """Read-time detection for a finished trial folder.

    Only trials the scorer would otherwise count are judged: a trial with a
    stored finding is returned as is by :func:`stored_integration_failure`,
    and a trial that already failed with an error (other than a timeout) is
    left to that error. A folder without its trajectory is left alone (a
    partial copy of a job, not an empty run). Never raises.
    """
    try:
        return _diagnose_trial_dir(Path(trial_dir), result)
    except Exception:  # the detector must never break reading a job
        return None


def _diagnose_trial_dir(
    trial_dir: Path, result: Mapping[str, Any]
) -> IntegrationFailure | None:
    if result.get("purpose") not in (None, "task"):
        return None
    if result.get("error") and result.get("error_category") not in (
        None,
        "timeout",
        "idle_timeout",
    ):
        return None
    if result.get("trajectory_source") not in (None, "acp", "partial_acp"):
        return None
    candidates = (
        trial_dir / "trajectory" / "acp_trajectory.jsonl",
        trial_dir / "agent" / "acp_trajectory.jsonl",
    )
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return None
    read = _read_events(path)
    if read is None:
        return None
    events, truncated = read
    agent_result = result.get("agent_result")
    agent_result = agent_result if isinstance(agent_result, Mapping) else {}
    output_tokens = _int(agent_result.get("n_output_tokens"))
    if output_tokens is None:
        output_tokens = _int(result.get("n_output_tokens"))
    return diagnose(
        events,
        agent=str(result.get("agent") or ""),
        n_tool_calls=_int(result.get("n_tool_calls")) or 0,
        output_tokens=output_tokens,
        logs=read_agent_logs(trial_dir),
        agent_seconds=_agent_seconds(trial_dir, result),
        truncated=truncated,
        prompt_sent=(_int(result.get("n_prompts")) or 0) > 0,
    )
