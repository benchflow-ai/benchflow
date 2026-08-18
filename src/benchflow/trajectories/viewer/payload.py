"""ACP payload assembly: normalization, diagnostics, sidecar loading.

Builds the typed :class:`~.models.ViewerPayload` the interactive template
renders client-side. All trajectory content is untrusted input — it stays
data throughout.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import (
    ErrorBanner,
    Meta,
    Step,
    StepCounts,
    TimeoutInfo,
    ToolCall,
    VerifierArtifacts,
    ViewerPayload,
    tool_hue,
)


def _parse_jsonl(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


# result.json diagnostic blocks surfaced in the header banner. The live list
# derives from DIAGNOSTIC_REGISTRY so new diagnostics (e.g. behavior flags)
# appear without this module drifting; the fallback pins the 0.7.4 set.
_DIAGNOSTIC_KEYS_FALLBACK = (
    "idle_timeout_info",
    "agent_timeout_info",
    "sandbox_startup_info",
    "transport_error_info",
    "verifier_timeout_info",
    "api_error_info",
    "suspected_api_error_info",
)


def _diagnostic_keys() -> tuple[str, ...]:
    """Diagnostic block keys, in registry display order."""
    try:
        from benchflow.diagnostics import DIAGNOSTIC_REGISTRY
    except Exception:  # pragma: no cover — registry is core; stay lenient
        return _DIAGNOSTIC_KEYS_FALLBACK
    return tuple(d.field for d in DIAGNOSTIC_REGISTRY)


def _load_json(path: Path) -> Any:
    # errors="replace": binary/mis-encoded sidecars degrade to a JSON parse
    # failure (→ None) instead of an unhandled UnicodeDecodeError.
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _tool_content_texts(content: Any) -> list[str]:
    """Flatten ACP tool_call content blocks to plain strings.

    Three shapes occur in captures: the canonical nested block
    ``{"type": "content", "content": {"type": "text", "text": ...}}``, a flat
    ``{"text": ...}``, and edit-tool diff blocks
    ``{"type": "diff", "path": ..., "oldText": ..., "newText": ...}``.
    """
    texts: list[str] = []
    if not isinstance(content, list):
        return texts
    for item in content:
        if not isinstance(item, dict):
            texts.append(str(item))
            continue
        inner = item.get("content")
        if isinstance(inner, dict) and "text" in inner:
            texts.append(str(inner.get("text", "")))
        elif "text" in item:
            texts.append(str(item.get("text", "")))
        elif item.get("type") == "diff":
            texts.append(
                f"diff {item.get('path', '')}\n"
                f"--- old\n{item.get('oldText') or ''}\n"
                f"+++ new\n{item.get('newText') or ''}"
            )
        else:
            texts.append(json.dumps(item, ensure_ascii=False))
    return [t for t in texts if t]


def _parse_ts(value: Any) -> float | None:
    """Lenient timestamp → epoch seconds; None for absent/unparseable.

    Accepts numeric epochs and anything ``datetime.fromisoformat`` reads
    (ISO-8601, and the space-separated ``str(datetime.now())`` form used by
    result.json's started_at).
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return None
    return None


def _normalize_steps(events: list[dict], prompts: list[str] | None) -> list[Step]:
    """Project raw ACP events onto renderer steps with server-side numbering.

    Prompt labels ("PROMPT 1") are computed here, not client-side, and the
    ``prompts`` list is embedded only when the trajectory has no inline
    user_message events — embedding both would duplicate prompt text in the
    emitted page (pinned by TestViewerCompatibility).

    Timestamps are forward-compatible passthrough: today's captures carry
    none (verified across the HF ground-truth uploads), but when the
    capture-side proposal lands (``ts`` on text events, ``started_at`` /
    ``finished_at`` on tool calls — benchflow#1033) steps gain ``t`` (epoch)
    and tools ``dur`` (seconds), and the renderer shows a timeline with no
    further changes.
    """
    steps: list[Step] = []
    prompt_counter = 0

    def add(step: Step, event: dict | None = None) -> None:
        step.i = len(steps) + 1
        if event is not None:
            if step.kind == "tool":
                started = _parse_ts(event.get("started_at") or event.get("ts"))
                finished = _parse_ts(event.get("finished_at"))
                if started is not None:
                    step.t = started
                    if finished is not None and finished >= started:
                        step.dur = finished - started
            else:
                step.t = _parse_ts(event.get("ts"))
        steps.append(step)

    has_inline_prompts = any(e.get("type") == "user_message" for e in events)
    if not has_inline_prompts:
        for text in prompts or []:
            prompt_counter += 1
            add(
                Step(
                    i=0,
                    kind="prompt",
                    label=f"PROMPT {prompt_counter}",
                    text=str(text),
                )
            )

    for event in events:
        etype = event.get("type", "")
        if etype == "user_message":
            prompt_counter += 1
            add(
                Step(
                    i=0,
                    kind="prompt",
                    label=f"PROMPT {prompt_counter}",
                    text=str(event.get("text", "")),
                ),
                event,
            )
        elif etype == "agent_message":
            add(Step(i=0, kind="message", text=str(event.get("text", ""))), event)
        elif etype == "agent_thought":
            add(Step(i=0, kind="thought", text=str(event.get("text", ""))), event)
        elif etype == "tool_call":
            kind = str(event.get("kind", "other"))
            title = str(event.get("title", ""))
            add(
                Step(
                    i=0,
                    kind="tool",
                    tool=ToolCall(
                        id=str(event.get("tool_call_id", "")),
                        kind=kind,
                        title=title,
                        status=str(event.get("status", "")),
                        content=_tool_content_texts(event.get("content")),
                        hue=tool_hue(kind, title),
                    ),
                ),
                event,
            )
        elif etype == "agent_timeout":
            # Coerce to a list of strings — a crafted non-list value would
            # otherwise reach the client and break the renderer.
            pending_raw = event.get("pending_tool_call_ids")
            add(
                Step(
                    i=0,
                    kind="timeout",
                    timeout=TimeoutInfo(
                        reason=str(event.get("reason", "")),
                        timeout_sec=event.get("timeout_sec"),
                        pending=[str(x) for x in pending_raw]
                        if isinstance(pending_raw, list)
                        else [],
                        complete=event.get("terminal_trajectory_complete"),
                    ),
                ),
                event,
            )
        else:
            # Unknown/future event types render as a generic card, never
            # crash — and ship complete (the template collapses long bodies;
            # silent truncation would betray the full-event-stream promise).
            add(
                Step(
                    i=0,
                    kind="unknown",
                    type=str(etype),
                    text=json.dumps(event, ensure_ascii=False),
                ),
                event,
            )
    return steps


def _build_meta(result_data: dict, timing: dict | None, steps: list[Step]) -> Meta:
    rewards = result_data.get("rewards")
    reward = rewards.get("reward") if isinstance(rewards, dict) else None
    agent_result = result_data.get("agent_result")
    usage = agent_result if isinstance(agent_result, dict) else {}

    errors: list[ErrorBanner] = []
    if result_data.get("error"):
        errors.append(
            ErrorBanner(
                label=str(result_data.get("error_category") or "error"),
                text=str(result_data["error"]),
            )
        )
    if result_data.get("verifier_error"):
        errors.append(
            ErrorBanner(
                label=str(
                    result_data.get("verifier_error_category") or "verifier error"
                ),
                text=str(result_data["verifier_error"]),
            )
        )
    if result_data.get("export_error"):
        errors.append(
            ErrorBanner(label="export error", text=str(result_data["export_error"]))
        )
    # Diagnostics that ride along an actual error render as error banners;
    # on an otherwise-clean rollout they are behavior flags (e.g. #1025's
    # chat-only completion) and render neutrally. All three error channels
    # count as "actual error". Text ships complete — the template collapses
    # long banners instead of this side truncating them.
    has_error = bool(
        result_data.get("error")
        or result_data.get("verifier_error")
        or result_data.get("export_error")
    )
    for key in _diagnostic_keys():
        value = result_data.get(key)
        if value:
            errors.append(
                ErrorBanner(
                    label=key.removesuffix("_info").replace("_", " "),
                    text=json.dumps(value, ensure_ascii=False, default=str),
                    level="error" if has_error else "info",
                )
            )

    return Meta(
        task_name=result_data.get("task_name"),
        agent_name=result_data.get("agent_name") or result_data.get("agent"),
        model=result_data.get("model"),
        skill_mode=result_data.get("skill_mode"),
        reward=reward,
        usage=usage,
        counts=StepCounts(
            prompts=sum(1 for s in steps if s.kind == "prompt"),
            messages=sum(1 for s in steps if s.kind == "message"),
            thoughts=sum(1 for s in steps if s.kind == "thought"),
            tools=sum(1 for s in steps if s.kind == "tool"),
        ),
        timing=timing,
        duration_sec=(timing or {}).get("total"),
        errors=errors,
        trajectory_source=result_data.get("trajectory_source"),
        partial_trajectory=result_data.get("partial_trajectory"),
        started_at=result_data.get("started_at"),
        finished_at=result_data.get("finished_at"),
    )


# The exact verifier sidecars the viewer renders. Also the verifier/ portion
# of the hf:// download allowlist (imported by .sources), so the fetch
# surface can never drift wider than what this module actually reads.
VERIFIER_SIDECARS = ("reward.txt", "test-stdout.txt", "test-stderr.txt", "ctrf.json")
(
    _VERIFIER_REWARD,
    _VERIFIER_STDOUT,
    _VERIFIER_STDERR,
    _VERIFIER_CTRF,
) = VERIFIER_SIDECARS


def _load_verifier(rollout_dir: Path) -> VerifierArtifacts:
    vdir = rollout_dir / "verifier"
    reward = _read_text(vdir / _VERIFIER_REWARD)
    ctrf_tests = None
    ctrf = _load_json(vdir / _VERIFIER_CTRF)
    if isinstance(ctrf, dict):
        raw_tests = (ctrf.get("results") or {}).get("tests")
        if isinstance(raw_tests, list):
            ctrf_tests = [
                {
                    "name": str(t.get("name", "")),
                    "status": str(t.get("status", "")),
                    "duration": t.get("duration"),
                }
                for t in raw_tests
                if isinstance(t, dict)
            ]
    return VerifierArtifacts(
        reward=reward.strip() if reward else None,
        stdout=_read_text(vdir / _VERIFIER_STDOUT),
        stderr=_read_text(vdir / _VERIFIER_STDERR),
        ctrf=ctrf_tests,
    )


def _safe_json(obj: Any) -> str:
    """Serialize untrusted payload data to UTF-8-safe JSON text.

    Lone surrogates (possible in crafted trajectory text) survive
    ``json.dumps(ensure_ascii=False)`` but crash utf-8 encoding later in
    ``serve()``; replace them here.
    """
    data = json.dumps(obj, ensure_ascii=False, default=str)
    return data.encode("utf-8", errors="replace").decode("utf-8")


def _build_acp_payload(rollout_dir: Path, prompts: list[str] | None) -> ViewerPayload:
    """Assemble the typed renderer payload for one canonical ACP rollout dir."""
    acp_path = rollout_dir / "trajectory" / "acp_trajectory.jsonl"
    try:
        text = acp_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    events = _parse_jsonl(text)

    if prompts is None:
        loaded = _load_json(rollout_dir / "prompts.json")
        prompts = loaded if isinstance(loaded, list) else None

    result_data = _load_result_json(rollout_dir)
    timing = _load_json(rollout_dir / "timing.json")
    if not isinstance(timing, dict):
        embedded = result_data.get("timing")
        timing = embedded if isinstance(embedded, dict) else None

    steps = _normalize_steps(events, prompts)
    return ViewerPayload(
        rollout_name=rollout_dir.name,
        meta=_build_meta(result_data, timing, steps),
        steps=steps,
        verifier=_load_verifier(rollout_dir),
    )


def _is_acp_rollout_dir(path: Path) -> bool:
    return (path / "trajectory" / "acp_trajectory.jsonl").exists()


def _load_result_json(rollout_dir: Path) -> dict:
    parsed = _load_json(rollout_dir / "result.json")
    return parsed if isinstance(parsed, dict) else {}
