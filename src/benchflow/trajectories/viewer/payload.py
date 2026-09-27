"""Normalize untrusted rollout artifacts into the typed viewer payload."""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from benchflow._utils.json_safe import dumps_finite, scrub_non_finite
from benchflow._utils.scoring import (
    assessment_status,
    assessment_withholds_score,
    classify_error,
    classify_verifier_error,
)
from benchflow.review.outcome import scoring_from_result

from .models import (
    AssessmentStatus,
    BranchChild,
    BranchFork,
    BranchOrigin,
    ErrorBanner,
    ExecutionStatus,
    JsonObject,
    JsonValue,
    Lineage,
    MessageStep,
    Meta,
    PromptStep,
    RecoveryAttempt,
    RolloutMetadata,
    RubricCriterion,
    RubricReview,
    RunStatus,
    Step,
    StepCounts,
    SubagentGroupStep,
    ThoughtStep,
    TimeoutInfo,
    TimeoutStep,
    Timing,
    ToolCall,
    ToolStep,
    UnknownStep,
    Usage,
    VerifierArtifacts,
    VerifierRecovery,
    VerifierTest,
    ViewerPayload,
    iter_event_steps,
    normalize_tool_status,
    normalize_verifier_status,
    tool_hue,
    tool_kind,
)
from .subagents import count_subagents, nest_subagent_steps

_MAX_JSON_DEPTH = 64
_MAX_JSON_OUTPUT_DEPTH = _MAX_JSON_DEPTH + 32


def _within_json_depth(value: Any, *, max_depth: int = _MAX_JSON_DEPTH) -> bool:
    """Bound container nesting without using Python recursion.

    JSON produced by BenchFlow is shallow. Rejecting unusually deep artifacts
    keeps the recursive finite-number scrubber and encoder comfortably below
    Python's recursion limit while still allowing wide, otherwise-valid data.
    """
    pending: list[tuple[Any, int]] = [(value, 0)]
    seen_containers: set[int] = set()
    while pending:
        current, depth = pending.pop()
        if isinstance(current, dict):
            identity = id(current)
            if identity in seen_containers:
                return False
            seen_containers.add(identity)
            if depth >= max_depth:
                return False
            pending.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, (list, tuple)):
            identity = id(current)
            if identity in seen_containers:
                return False
            seen_containers.add(identity)
            if depth >= max_depth:
                return False
            pending.extend((child, depth + 1) for child in current)
    return True


def _parse_json(text: str) -> Any:
    """Parse one bounded JSON value; malformed or over-deep input is absent."""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        return None
    return parsed if _within_json_depth(parsed) else None


def _parse_jsonl(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parsed = _parse_json(line)
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


# Keep a fallback so the viewer remains useful if diagnostics cannot import.
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
    try:
        from benchflow.diagnostics import DIAGNOSTIC_REGISTRY
    except Exception:  # pragma: no cover - diagnostics is a core module
        return _DIAGNOSTIC_KEYS_FALLBACK
    return tuple(diagnostic.field for diagnostic in DIAGNOSTIC_REGISTRY)


def _load_json(path: Path) -> Any:
    """Read a JSON sidecar, returning ``None`` for any malformed artifact."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return _parse_json(text)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _json_value(value: Any) -> JsonValue:
    """Project a scrubbed value onto the recursive JSON value type."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _json_object(value: Any) -> JsonObject:
    if not isinstance(value, dict) or not _within_json_depth(value):
        return {}
    value = scrub_non_finite(value)
    return {str(key): _json_value(item) for key, item in value.items()}


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _nonnegative_int(value: Any) -> int | None:
    number = _finite_float(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _optional_text(value: Any) -> str | None:
    """Coerce scalar identity fields; reject container-shaped identities."""
    if value is None or isinstance(value, (dict, list, tuple)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return str(value)


def _display_text(value: Any) -> str:
    """Losslessly display JSON-shaped content without leaking invalid JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        if not _within_json_depth(value):
            return ""
        return dumps_finite(value, ensure_ascii=False, default=str)
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return str(value)


def _load_prompts(rollout_dir: Path) -> list[str] | None:
    """Load prompts.json through the shared bounded JSON/text boundary.

    The artifact contract is a list, but older or hand-edited captures may
    contain non-string entries. Coerce entries exactly as trajectory text is
    coerced; a non-list top level is not a prompt collection.
    """
    parsed = _load_json(rollout_dir / "prompts.json")
    if not isinstance(parsed, list):
        return None
    return [_display_text(prompt) for prompt in parsed]


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _tool_content_texts(content: Any) -> list[str]:
    """Flatten canonical, flat-text, and diff ACP tool content blocks."""
    if not isinstance(content, list):
        return []
    texts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            text = _display_text(item)
        else:
            inner = item.get("content")
            if isinstance(inner, dict) and "text" in inner:
                text = _display_text(inner.get("text"))
            elif "text" in item:
                text = _display_text(item.get("text"))
            elif item.get("type") == "diff":
                text = (
                    f"diff {_display_text(item.get('path'))}\n"
                    f"--- old\n{_display_text(item.get('oldText'))}\n"
                    f"+++ new\n{_display_text(item.get('newText'))}"
                )
            else:
                text = dumps_finite(item, ensure_ascii=False, default=str)
        if text:
            texts.append(text)
    return texts


def _raw_io_texts(kind: str, raw_input: Any, raw_output: Any) -> list[str]:
    """Render ACP ``rawInput`` / ``rawOutput`` for a call without content blocks.

    codex-acp reports a command as ``rawInput.command`` and its result as
    ``rawOutput.formatted_output`` plus ``exit_code``; anything else is shown
    as JSON so no recorded detail is hidden.
    """
    texts: list[str] = []
    if isinstance(raw_input, dict) and kind == "execute" and "command" in raw_input:
        command = raw_input["command"]
        texts.append(
            " ".join(str(part) for part in command)
            if isinstance(command, list)
            else _display_text(command)
        )
    elif raw_input is not None:
        texts.append(dumps_finite(raw_input, ensure_ascii=False, default=str))
    if isinstance(raw_output, dict) and "formatted_output" in raw_output:
        output = _display_text(raw_output.get("formatted_output"))
        exit_code = raw_output.get("exit_code")
        if exit_code not in (None, 0):
            output = f"{output.rstrip()}\n[exit code {exit_code}]"
        texts.append(output)
    elif raw_output is not None:
        texts.append(dumps_finite(raw_output, ensure_ascii=False, default=str))
    return [text for text in texts if text]


# claude-agent-acp titles a call before its input has streamed in, then sends
# the real title ("<verb> <path>") in a later tool_call_update. The capture
# now records that final title (acp/session.py); trajectories captured by
# earlier releases kept the first one, so these provisional titles, keyed by
# (kind, title), are rebuilt from the recorded input the adapter used.
_PROVISIONAL_TITLES: dict[tuple[str, str], str] = {
    ("edit", "Preparing file…"): "Write",
    ("edit", "Write"): "Write",
    ("edit", "Edit"): "Edit",
    ("read", "Read File"): "Read",
}


def _final_tool_title(kind: str, title: str, raw_input: Any, content: Any) -> str:
    """The adapter's final title for a provisional one, when a path is known.

    The path is the input's ``file_path``, else the path of the call's diff
    blocks when they all name one file; otherwise the title is kept.
    """
    verb = _PROVISIONAL_TITLES.get((kind, title.strip()))
    if verb is None:
        return title
    path = raw_input.get("file_path") if isinstance(raw_input, dict) else None
    if not isinstance(path, str) or not path:
        diff_paths = {
            item.get("path")
            for item in (content if isinstance(content, list) else [])
            if isinstance(item, dict) and item.get("type") == "diff"
        }
        path = diff_paths.pop() if len(diff_paths) == 1 else None
    return f"{verb} {path}" if isinstance(path, str) and path else title


def _parse_ts(value: Any) -> float | None:
    """Parse finite epoch or ISO-8601 timestamps into epoch seconds."""
    numeric = _finite_float(value)
    if numeric is not None and not isinstance(value, str):
        return numeric
    if isinstance(value, str):
        try:
            timestamp = datetime.fromisoformat(value).timestamp()
        except (ValueError, OverflowError, OSError):
            return None
        return timestamp if math.isfinite(timestamp) else None
    return None


def _tool_timing(event: dict[str, Any]) -> tuple[float | None, float | None]:
    started = _parse_ts(event.get("started_at"))
    if started is None:
        started = _parse_ts(event.get("ts"))
    finished = _parse_ts(event.get("finished_at"))
    duration = (
        finished - started
        if started is not None and finished is not None and finished >= started
        else None
    )
    return started, duration


def _normalize_steps(
    events: list[dict[str, Any]], prompts: list[str] | None
) -> list[Step]:
    """Project raw ACP event variants onto explicit renderer step variants.

    Steps are numbered in capture order (``i``). Events a capture attributed
    to a subagent (``parent_tool_call_id``) are then nested under the tool
    call that spawned them (see :mod:`.subagents`).
    """
    steps: list[Step] = []
    sources: list[dict[str, Any] | None] = []
    prompt_counter = 0

    def next_index() -> int:
        return len(steps) + 1

    has_inline_prompts = any(event.get("type") == "user_message" for event in events)
    if not has_inline_prompts:
        for text in prompts or []:
            prompt_counter += 1
            steps.append(
                PromptStep(
                    i=next_index(),
                    label=f"PROMPT {prompt_counter}",
                    text=_display_text(text),
                )
            )
            sources.append(None)

    for event in events:
        if not _within_json_depth(event):
            continue
        # Every branch below appends exactly one step for this event.
        sources.append(event)
        event_type = _optional_text(event.get("type")) or ""
        event_time = _parse_ts(event.get("ts"))
        if event_type == "user_message":
            prompt_counter += 1
            steps.append(
                PromptStep(
                    i=next_index(),
                    label=f"PROMPT {prompt_counter}",
                    text=_display_text(event.get("text")),
                    t=event_time,
                )
            )
        elif event_type == "agent_message":
            steps.append(
                MessageStep(
                    i=next_index(), text=_display_text(event.get("text")), t=event_time
                )
            )
        elif event_type == "agent_thought":
            steps.append(
                ThoughtStep(
                    i=next_index(), text=_display_text(event.get("text")), t=event_time
                )
            )
        elif event_type == "tool_call":
            title = _display_text(event.get("title"))
            kind = tool_kind(
                _display_text(event.get("kind")) or "other",
                title,
                event.get("raw_input"),
            )
            title = _final_tool_title(
                kind, title, event.get("raw_input"), event.get("content")
            )
            started, duration = _tool_timing(event)
            steps.append(
                ToolStep(
                    i=next_index(),
                    tool=ToolCall(
                        id=_display_text(event.get("tool_call_id")),
                        kind=kind,
                        title=title,
                        status=normalize_tool_status(event.get("status")),
                        content=_tool_content_texts(event.get("content"))
                        or _raw_io_texts(
                            kind, event.get("raw_input"), event.get("raw_output")
                        ),
                        hue=tool_hue(kind, title),
                    ),
                    t=started,
                    dur=duration,
                )
            )
        elif event_type == "agent_timeout":
            pending = event.get("pending_tool_call_ids")
            steps.append(
                TimeoutStep(
                    i=next_index(),
                    timeout=TimeoutInfo(
                        reason=_display_text(event.get("reason")),
                        timeout_sec=_finite_float(event.get("timeout_sec")),
                        pending=[_display_text(item) for item in pending]
                        if isinstance(pending, list)
                        else [],
                        complete=_optional_bool(
                            event.get("terminal_trajectory_complete")
                        ),
                    ),
                    t=event_time,
                )
            )
        else:
            steps.append(
                UnknownStep(
                    i=next_index(),
                    type=event_type,
                    text=dumps_finite(event, ensure_ascii=False, default=str),
                    t=event_time,
                )
            )
    return nest_subagent_steps(steps, sources)


_USAGE_INT_FIELDS = (
    "n_tool_calls",
    "n_skill_invocations",
    "n_prompts",
    "n_input_tokens",
    "n_output_tokens",
    "n_cache_read_tokens",
    "n_cache_creation_tokens",
    "total_tokens",
)


def _normalize_usage(raw: Any) -> Usage:
    values = _json_object(raw)
    integers: dict[str, int | None] = {}
    for field_name in _USAGE_INT_FIELDS:
        normalized = _nonnegative_int(values.get(field_name))
        integers[field_name] = normalized
        if field_name in values:
            values[field_name] = normalized

    cost_usd = _finite_float(values.get("cost_usd"))
    if "cost_usd" in values:
        values["cost_usd"] = cost_usd
    usage_source = _optional_text(values.get("usage_source"))
    price_source = _optional_text(values.get("price_source"))
    if "usage_source" in values:
        values["usage_source"] = usage_source
    if "price_source" in values:
        values["price_source"] = price_source

    return Usage(
        values=values,
        n_tool_calls=integers["n_tool_calls"],
        n_skill_invocations=integers["n_skill_invocations"],
        n_prompts=integers["n_prompts"],
        n_input_tokens=integers["n_input_tokens"],
        n_output_tokens=integers["n_output_tokens"],
        n_cache_read_tokens=integers["n_cache_read_tokens"],
        n_cache_creation_tokens=integers["n_cache_creation_tokens"],
        total_tokens=integers["total_tokens"],
        cost_usd=cost_usd,
        usage_source=usage_source,
        price_source=price_source,
    )


def _normalize_timing(raw: Any) -> Timing | None:
    if not isinstance(raw, dict) or not _within_json_depth(raw):
        return None
    values = {str(key): _finite_float(value) for key, value in raw.items()}
    return Timing(values=values, total=values.get("total"))


def _error_text(value: Any) -> str | None:
    return _display_text(value) if value else None


_EXECUTION_LABELS: dict[ExecutionStatus, str] = {
    "completed": "completed",
    "errored": "errored",
    "timed_out": "timed out",
}


def _category_text(category: str | None) -> str | None:
    return category.replace("_", " ") if category else None


def _run_status(result_data: dict[str, Any], reward: float | None) -> RunStatus:
    """Split one ``result.json`` into execution and assessment status.

    Execution reads only the agent ``error`` (classified like the score
    accounting in ``_utils/scoring.py``). A declared outcome assessment that
    is pending or unassessable (embodied trials) is unscored whatever sits
    beside it. Otherwise assessment reads the integrated ``scoring`` block
    when present -- a malformed or ``error`` block is unscored even beside a
    stale reward -- else the reward, else the verifier error.
    """
    error = _error_text(result_data.get("error"))
    execution: ExecutionStatus = "completed"
    execution_detail: str | None = None
    if error is not None:
        category = _optional_text(result_data.get("error_category")) or (
            classify_error(error)
        )
        execution = "timed_out" if category and "timeout" in category else "errored"
        execution_detail = _category_text(category)

    assessment: AssessmentStatus = "unscored"
    assessment_detail: str | None
    scoring_block: JsonObject | None = None
    if assessment_withholds_score(result_data):
        assessment_detail = f"assessment {assessment_status(result_data)}"
        declared = result_data.get("assessment")
        reason = declared.get("reason") if isinstance(declared, dict) else None
        if isinstance(reason, str) and reason:
            assessment_detail += f" ({_category_text(reason)})"
    elif result_data.get("scoring") is not None:
        scoring_block = _json_object(result_data.get("scoring")) or None
        try:
            scoring = scoring_from_result(result_data)
        except Exception:  # untrusted artifact: any malformed block is unscored
            scoring = None
        if scoring is None:
            assessment_detail = "malformed scoring block"
        elif scoring.status == "complete":
            assessment = "scored"
            assessment_detail = "tests + rubric, " + (
                "passed" if scoring.passed else "not passed"
            )
        else:
            assessment_detail = f"scoring error: {scoring.error}"
    elif reward is not None:
        assessment = "scored"
        assessment_detail = None
    elif verifier_error := _error_text(result_data.get("verifier_error")):
        category = _optional_text(result_data.get("verifier_error_category")) or (
            classify_verifier_error(verifier_error)
        )
        assessment_detail = _category_text(category) or "verifier error"
    elif error is not None:
        assessment_detail = "no verdict after the execution error"
    else:
        assessment_detail = "no reward recorded"

    label = _EXECUTION_LABELS[execution]
    if assessment == "scored":
        summary = "completed and scored" if error is None else f"{label} but scored"
    else:
        summary = "completed but unscored" if error is None else f"{label}, unscored"
    return RunStatus(
        execution=execution,
        execution_detail=execution_detail,
        assessment=assessment,
        assessment_detail=assessment_detail,
        summary=summary,
        scoring=scoring_block,
    )


def _normalize_metadata(
    result_data: dict[str, Any], timing_data: dict[str, Any] | None
) -> RolloutMetadata:
    """Canonical result/timing boundary shared by detail and catalog views."""
    if not _within_json_depth(result_data):
        result_data = {}
    rewards = result_data.get("rewards")
    reward = _finite_float(rewards.get("reward")) if isinstance(rewards, dict) else None
    if assessment_withholds_score(result_data):
        reward = None  # a pending/unassessable assessment has no reward yet
    usage = _normalize_usage(result_data.get("agent_result"))
    timing = _normalize_timing(timing_data)

    agent_error = _error_text(result_data.get("error"))
    verifier_error = _error_text(result_data.get("verifier_error"))
    export_error = _error_text(result_data.get("export_error"))
    has_error = any((agent_error, verifier_error, export_error))
    errors: list[ErrorBanner] = []
    if agent_error is not None:
        errors.append(
            ErrorBanner(
                label=_optional_text(result_data.get("error_category")) or "error",
                text=agent_error,
            )
        )
    if verifier_error is not None:
        errors.append(
            ErrorBanner(
                label=_optional_text(result_data.get("verifier_error_category"))
                or "verifier error",
                text=verifier_error,
            )
        )
    if export_error is not None:
        errors.append(ErrorBanner(label="export error", text=export_error))

    for key in _diagnostic_keys():
        value = result_data.get(key)
        if value:
            errors.append(
                ErrorBanner(
                    label=key.removesuffix("_info").replace("_", " "),
                    text=dumps_finite(value, ensure_ascii=False, default=str),
                    level="error" if has_error else "info",
                )
            )

    top_level_tool_calls = _nonnegative_int(result_data.get("n_tool_calls"))
    agent_name = _optional_text(result_data.get("agent_name")) or _optional_text(
        result_data.get("agent")
    )
    return RolloutMetadata(
        task_name=_optional_text(result_data.get("task_name")),
        agent_name=agent_name,
        model=_optional_text(result_data.get("model")),
        skill_mode=_optional_text(result_data.get("skill_mode")),
        reward=reward,
        usage=usage,
        timing=timing,
        n_tool_calls=top_level_tool_calls
        if top_level_tool_calls is not None
        else usage.n_tool_calls,
        errors=tuple(errors),
        has_error=has_error,
        trajectory_source=_optional_text(result_data.get("trajectory_source")),
        partial_trajectory=_optional_bool(result_data.get("partial_trajectory")),
        started_at=_optional_text(result_data.get("started_at")),
        finished_at=_optional_text(result_data.get("finished_at")),
        status=_run_status(result_data, reward) if result_data else None,
    )


def _load_rollout_metadata(rollout_dir: Path) -> RolloutMetadata:
    """Load and normalize both metadata sidecars exactly once per projection."""
    result_data = _load_result_json(rollout_dir)
    preferred_timing = _load_json(rollout_dir / "timing.json")
    if not isinstance(preferred_timing, dict):
        embedded_timing = result_data.get("timing")
        preferred_timing = (
            embedded_timing if isinstance(embedded_timing, dict) else None
        )
    return _normalize_metadata(result_data, preferred_timing)


def _build_meta(
    metadata: RolloutMetadata,
    steps: list[Step],
    *,
    extra_errors: tuple[ErrorBanner, ...] = (),
    branch: BranchOrigin | None = None,
) -> Meta:
    events = iter_event_steps(steps)
    root_events = sum(not isinstance(step, SubagentGroupStep) for step in steps)
    counts = StepCounts(
        prompts=sum(step.kind == "prompt" for step in events),
        messages=sum(step.kind == "message" for step in events),
        thoughts=sum(step.kind == "thought" for step in events),
        tools=sum(step.kind == "tool" for step in events),
        subagents=count_subagents(steps),
        subagent_events=len(events) - root_events,
    )
    return Meta(
        task_name=metadata.task_name,
        agent_name=metadata.agent_name,
        model=metadata.model,
        skill_mode=metadata.skill_mode,
        reward=metadata.reward,
        usage=metadata.usage,
        counts=counts,
        timing=metadata.timing,
        duration_sec=metadata.timing.total if metadata.timing is not None else None,
        errors=metadata.errors + extra_errors,
        trajectory_source=metadata.trajectory_source,
        partial_trajectory=metadata.partial_trajectory,
        started_at=metadata.started_at,
        finished_at=metadata.finished_at,
        status=metadata.status,
        branch=branch,
    )


# Exact verifier sidecars rendered by the viewer and downloaded for hf://.
VERIFIER_SIDECARS = ("reward.txt", "test-stdout.txt", "test-stderr.txt", "ctrf.json")
(
    _VERIFIER_REWARD,
    _VERIFIER_STDOUT,
    _VERIFIER_STDERR,
    _VERIFIER_CTRF,
) = VERIFIER_SIDECARS


def _load_verifier(rollout_dir: Path) -> VerifierArtifacts:
    verifier_dir = rollout_dir / "verifier"
    reward = _read_text(verifier_dir / _VERIFIER_REWARD)
    ctrf_tests: list[VerifierTest] | None = None
    ctrf = _load_json(verifier_dir / _VERIFIER_CTRF)
    if isinstance(ctrf, dict):
        results = ctrf.get("results")
        raw_tests = results.get("tests") if isinstance(results, dict) else None
        if isinstance(raw_tests, list):
            ctrf_tests = [
                VerifierTest(
                    name=_display_text(test.get("name")),
                    status=normalize_verifier_status(test.get("status")),
                    duration=_finite_float(test.get("duration")),
                )
                for test in raw_tests
                if isinstance(test, dict)
            ]
    return VerifierArtifacts(
        reward=reward.strip() if reward else None,
        stdout=_read_text(verifier_dir / _VERIFIER_STDOUT),
        stderr=_read_text(verifier_dir / _VERIFIER_STDERR),
        ctrf=ctrf_tests,
    )


# Verifier-only recovery artifacts (docs/verifier-recovery.md): the pointer to
# the admitted attempt and one receipt per attempt directory.
_VERIFICATION_POINTER = "verification.json"
_RECOVERY_DIR = "verifier-recovery"
_RECOVERY_RECEIPT = "recovery.json"
_MAX_RECOVERY_ATTEMPTS = 50


def _valid_attempt_ref(value: Any) -> bool:
    """The pointer shape ``_verifier_recovery.verification_source`` accepts."""
    if not isinstance(value, str):
        return False
    parts = value.split("/")
    return len(parts) == 2 and parts[0] == _RECOVERY_DIR and parts[1].isalnum()


def _recovery_attempt(ref: str, record: Any, *, admitted: bool) -> RecoveryAttempt:
    if not isinstance(record, dict):
        return RecoveryAttempt(attempt=ref, admitted=admitted)
    rewards = record.get("rewards")
    timing = record.get("timing")
    return RecoveryAttempt(
        attempt=ref,
        admitted=admitted,
        status=_optional_text(record.get("status")),
        original_error=_error_text(record.get("original_error")),
        error=_error_text(record.get("error")),
        evidence=_optional_text(record.get("evidence")),
        solver_replayed=_optional_bool(record.get("solver_replayed")),
        publication_error=_error_text(record.get("publication_error")),
        cleanup_error=_error_text(record.get("cleanup_error")),
        admission=_optional_text(record.get("admission")),
        reward=_finite_float(rewards.get("reward"))
        if isinstance(rewards, dict)
        else None,
        verifier_sec=_finite_float(timing.get("verifier"))
        if isinstance(timing, dict)
        else None,
    )


def _load_recovery(rollout_dir: Path) -> VerifierRecovery | None:
    """Read the verification pointer and every recovery receipt, if any.

    The admitted attempt (``verification.json``) comes first; other receipts
    follow oldest first. A pointer whose receipt is absent (e.g. a dataset
    slice that fetched only the pointer) still yields an admitted row.
    """
    pointer: str | None = None
    pointer_error: str | None = None
    pointer_path = rollout_dir / _VERIFICATION_POINTER
    if pointer_path.is_file():
        raw = _load_json(pointer_path)
        attempt = raw.get("attempt") if isinstance(raw, dict) else None
        if _valid_attempt_ref(attempt):
            pointer = attempt
        else:
            pointer_error = "does not reference verifier-recovery/<attempt id>"
    receipts: list[tuple[float, str, Path]] = []
    try:
        entries = sorted((rollout_dir / _RECOVERY_DIR).iterdir())
    except OSError:
        entries = []
    for entry in entries[:_MAX_RECOVERY_ATTEMPTS]:
        receipt = entry / _RECOVERY_RECEIPT
        if not entry.name.isalnum():
            continue
        try:
            modified = receipt.stat().st_mtime
        except OSError:
            continue
        receipts.append((modified, f"{_RECOVERY_DIR}/{entry.name}", receipt))
    if pointer is None and pointer_error is None and not receipts:
        return None
    receipts.sort()
    attempts = [
        _recovery_attempt(ref, _load_json(path), admitted=ref == pointer)
        for _, ref, path in receipts
    ]
    if pointer is not None and all(item.attempt != pointer for item in attempts):
        attempts.append(RecoveryAttempt(attempt=pointer, admitted=True))
    attempts.sort(key=lambda item: not item.admitted)
    return VerifierRecovery(
        pointer=pointer, pointer_error=pointer_error, attempts=attempts
    )


def _recovery_banners(recovery: VerifierRecovery | None) -> tuple[ErrorBanner, ...]:
    """Surface a failed publication: ``verifier/`` then predates the score."""
    if recovery is None:
        return ()
    return tuple(
        ErrorBanner(
            label="verifier publication error",
            text=(
                f"{attempt.publication_error}\nThe admitted score comes from "
                f"{attempt.attempt}; verifier/ still holds the outputs from "
                f"before recovery, and the recovered outputs are in "
                f"{attempt.attempt}/verifier/."
            ),
            level="info",
        )
        for attempt in recovery.attempts
        if attempt.admitted and attempt.publication_error
    )


# Rollout-branch lineage (branch_lineage.ForkRecord.persist).
_TREE_FILE = "tree.json"
_TREE_KIND = "benchflow-branch-tree"
_BRANCH_OBSERVATION = "observation.json"
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _error_record_text(value: Any) -> str | None:
    """``{"type", "code"}`` records (never messages) as ``Type (code)``."""
    if isinstance(value, dict):
        kind = _optional_text(value.get("type"))
        code = _optional_text(value.get("code"))
        text = " ".join(part for part in (kind, f"({code})" if code else None) if part)
        return text or None
    return _error_text(value)


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := _optional_text(item))]


def _branch_child_dir(rollout_dir: Path, relative: str | None) -> Path | None:
    """The child's archive, only when it stays inside the rollout directory."""
    if not relative:
        return None
    root = rollout_dir.resolve()
    try:
        candidate = (rollout_dir / relative).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if candidate == root or not candidate.is_relative_to(root):
        return None
    return candidate if (candidate / _BRANCH_OBSERVATION).is_file() else None


def _branch_child(
    rollout_dir: Path, fork_id: str | None, raw: dict[str, Any]
) -> BranchChild:
    intervention = raw.get("intervention")
    intervention = intervention if isinstance(intervention, dict) else {}
    artifacts = raw.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, dict) else {}
    node_id = _optional_text(raw.get("node_id"))
    artifacts_status = _optional_text(artifacts.get("status"))
    artifacts_path = _optional_text(artifacts.get("path"))
    ref = None
    if (
        fork_id
        and node_id
        and _SAFE_ID_RE.match(fork_id)
        and _SAFE_ID_RE.match(node_id)
        and artifacts_status == "available"
        and _branch_child_dir(rollout_dir, artifacts_path) is not None
    ):
        ref = f"{fork_id}/{node_id}"
    index = raw.get("index")
    return BranchChild(
        index=index if isinstance(index, int) and not isinstance(index, bool) else None,
        node_id=node_id,
        status=_optional_text(raw.get("status")),
        reward=_finite_float(raw.get("reward")),
        reward_source=_optional_text(raw.get("reward_source")),
        error=_error_record_text(raw.get("error")),
        cleanup_error=_error_record_text(raw.get("cleanup_error")),
        intervention={
            key: _display_text(intervention.get(key)) or None
            for key in ("label", "requested", "execution", "evidence")
        },
        artifacts_status=artifacts_status,
        artifacts_path=artifacts_path,
        ref=ref,
    )


def _load_lineage(rollout_dir: Path) -> Lineage | None:
    """Parse ``tree.json`` into fork rows; ``None`` when the run never branched."""
    path = rollout_dir / _TREE_FILE
    if not path.is_file():
        return None
    raw = _load_json(path)
    if not isinstance(raw, dict) or raw.get("kind") != _TREE_KIND:
        return Lineage(nodes=0, forks=[], error="not a benchflow-branch-tree document")
    if raw.get("schema_version") != 1:
        return Lineage(
            nodes=0,
            forks=[],
            error=f"unsupported schema_version {_display_text(raw.get('schema_version'))}",
        )
    nodes = raw.get("nodes")
    forks: list[BranchFork] = []
    raw_forks = raw.get("forks")
    for fork in raw_forks if isinstance(raw_forks, list) else []:
        if not isinstance(fork, dict):
            continue
        fork_id = _optional_text(fork.get("id")) or ""
        snapshot = fork.get("snapshot")
        snapshot = snapshot if isinstance(snapshot, dict) else {}
        requested = fork.get("requested_children")
        children = fork.get("children")
        forks.append(
            BranchFork(
                id=fork_id,
                parent_node=_optional_text(fork.get("parent_node")),
                status=_optional_text(fork.get("status")),
                value=_finite_float(fork.get("value")),
                requested_children=requested
                if isinstance(requested, int) and not isinstance(requested, bool)
                else None,
                requested_layers=_text_list(snapshot.get("requested_layers")),
                captured_layers=_text_list(snapshot.get("captured_layers")),
                parent_restore=_optional_text(fork.get("parent_restore")),
                error=_error_record_text(fork.get("error")),
                parent_restore_error=_error_record_text(
                    fork.get("parent_restore_error")
                ),
                artifact_error=_error_record_text(fork.get("artifact_error")),
                children=[
                    _branch_child(rollout_dir, fork_id, child)
                    for child in (children if isinstance(children, list) else [])
                    if isinstance(child, dict)
                ],
            )
        )
    return Lineage(nodes=len(nodes) if isinstance(nodes, list) else 0, forks=forks)


def _build_branch_child_payload(rollout_dir: Path, ref: str) -> ViewerPayload | None:
    """A branch child's trajectory, resolved only through the parent's tree.

    ``ref`` must name a child the freshly parsed ``tree.json`` lists with
    published evidence, so crafted refs never reach the filesystem. The
    child's ``observation.json`` holds its continuation events (the steps
    after the fork point); its verifier sidecars were handed off under
    ``mounted/verifier``.
    """
    lineage = _load_lineage(rollout_dir)
    for fork in lineage.forks if lineage is not None else []:
        for child in fork.children:
            if child.ref is None or child.ref != ref:
                continue
            child_dir = _branch_child_dir(rollout_dir, child.artifacts_path)
            observation = (
                _load_json(child_dir / _BRANCH_OBSERVATION) if child_dir else None
            )
            if child_dir is None or not isinstance(observation, dict):
                return None
            raw_events = observation.get("trajectory")
            events = [
                event
                for event in (raw_events if isinstance(raw_events, list) else [])
                if isinstance(event, dict)
            ]
            parent = _load_rollout_metadata(rollout_dir)
            error = observation.get("error")
            verifier_error = observation.get("verifier_error")
            if child.status == "unscored":
                # The recorded failure is the missing verdict, not the agent.
                verifier_error, error = verifier_error or error, None
            metadata = _normalize_metadata(
                {
                    "task_name": parent.task_name,
                    "agent_name": parent.agent_name,
                    "model": parent.model,
                    "skill_mode": parent.skill_mode,
                    "rewards": {"reward": child.reward}
                    if child.reward is not None
                    else None,
                    "error": error,
                    "verifier_error": verifier_error,
                    "export_error": observation.get("export_error"),
                    "agent_result": observation.get("native_usage"),
                    "trajectory_source": "branch observation",
                },
                observation.get("timing")
                if isinstance(observation.get("timing"), dict)
                else None,
            )
            steps = _normalize_steps(events, None)
            origin = BranchOrigin(
                parent_rollout=rollout_dir.name,
                fork_id=fork.id,
                node_id=child.node_id or "",
                index=child.index,
                parent_node=fork.parent_node,
                status=child.status,
                intervention=child.intervention.get("label"),
            )
            return ViewerPayload(
                rollout_name=f"{rollout_dir.name} / branch {child.node_id}",
                meta=_build_meta(metadata, steps, branch=origin),
                steps=steps,
                # In-place children hand off the mounted sidecars; isolated
                # children are full trial folders with their own verifier/.
                verifier=_load_verifier(
                    child_dir / "mounted"
                    if (child_dir / "mounted").is_dir()
                    else child_dir
                ),
            )
    return None


def _branch_child_payloads(
    rollout_dir: Path, lineage: Lineage | None
) -> dict[str, dict[str, Any]]:
    """Every openable child payload, embedded by single-trajectory pages."""
    payloads: dict[str, dict[str, Any]] = {}
    for fork in lineage.forks if lineage is not None else []:
        for child in fork.children:
            if child.ref is None or child.ref in payloads:
                continue
            payload = _build_branch_child_payload(rollout_dir, child.ref)
            if payload is not None:
                payloads[child.ref] = payload.to_payload()
    return payloads


# How far above a rollout directory review reports are looked for. Covers
# ``jobs/<run>/<rollout>`` next to ``jobs/review-<stamp>/`` (bench review's
# default) and deeper per-trial layouts.
_REVIEW_SEARCH_DEPTH = 4
_REVIEW_REPORT = "review_report.json"


def _criteria(trial: dict[str, Any]) -> list[RubricCriterion]:
    checks = trial.get("checks")
    checks = checks if isinstance(checks, dict) else {}
    metadata = trial.get("criterion_metadata")
    rows: list[RubricCriterion] = []
    for meta in metadata if isinstance(metadata, list) else []:
        if not isinstance(meta, dict):
            continue
        name = _display_text(meta.get("name"))
        check = checks.get(name)
        check = check if isinstance(check, dict) else {}
        score = check.get("score")
        weight = meta.get("weight")
        blocker = meta.get("blocker")
        rows.append(
            RubricCriterion(
                name=name,
                blocker=None if blocker is None else bool(blocker),
                weight=weight
                if isinstance(weight, int) and not isinstance(weight, bool)
                else None,
                outcome=_optional_text(check.get("outcome")),
                score=score
                if isinstance(score, int) and not isinstance(score, bool)
                else None,
                explanation=_display_text(check.get("explanation")),
            )
        )
    return rows


def _rubric_from_report(path: Path, rollout_name: str) -> RubricReview | None:
    report = _load_json(path)
    if not isinstance(report, dict):
        return None
    reviewer = report.get("reviewer")
    model = (
        _optional_text(reviewer.get("model")) if isinstance(reviewer, dict) else None
    )
    found: RubricReview | None = None
    trials = report.get("trials")
    for trial in trials if isinstance(trials, list) else []:
        if not isinstance(trial, dict) or trial.get("trial_name") != rollout_name:
            continue
        scoring = trial.get("scoring")
        found = RubricReview(
            reviewer_model=model,
            review_valid=bool(trial.get("review_valid")),
            scoring=dict(scoring) if isinstance(scoring, dict) else {},
            summary=_display_text(trial.get("summary")),
            criteria=_criteria(trial),
            source=str(path),
        )
    return found


def _load_rubric(rollout_dir: Path) -> RubricReview | None:
    """Find the review of this rollout in the nearest ``review*`` directory.

    The search walks up ``_REVIEW_SEARCH_DEPTH`` ancestors; at each level every
    ``review*/**/review_report.json`` is read and the last valid entry for
    this rollout in sorted order wins, the rule the leaderboard consumers use.
    An invalid review (no scoring) never shadows a valid one.
    """
    name = rollout_dir.name
    ancestor = rollout_dir
    for _ in range(_REVIEW_SEARCH_DEPTH):
        ancestor = ancestor.parent
        found: RubricReview | None = None
        for review_dir in sorted(ancestor.glob("review*")):
            if not review_dir.is_dir():
                continue
            for path in sorted(review_dir.rglob(_REVIEW_REPORT)):
                rubric = _rubric_from_report(path, name)
                if rubric is None:
                    continue
                if rubric.scoring or found is None:
                    found = rubric
        if found is not None:
            return found
        if ancestor == ancestor.parent:
            break
    return None


def _safe_json(obj: Any) -> str:
    """Emit strict, finite JSON and replace lone UTF-16 surrogates."""
    if not _within_json_depth(obj, max_depth=_MAX_JSON_OUTPUT_DEPTH):
        return "null"
    data = dumps_finite(obj, ensure_ascii=False, default=str)
    return data.encode("utf-8", errors="replace").decode("utf-8")


def _build_acp_payload(rollout_dir: Path, prompts: list[str] | None) -> ViewerPayload:
    acp_path = rollout_dir / "trajectory" / "acp_trajectory.jsonl"
    try:
        text = acp_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    events = _parse_jsonl(text)

    if prompts is None:
        prompts = _load_prompts(rollout_dir)

    steps = _normalize_steps(events, prompts)
    recovery = _load_recovery(rollout_dir)
    return ViewerPayload(
        rollout_name=rollout_dir.name,
        meta=_build_meta(
            _load_rollout_metadata(rollout_dir),
            steps,
            extra_errors=_recovery_banners(recovery),
        ),
        steps=steps,
        verifier=replace(_load_verifier(rollout_dir), recovery=recovery),
        rubric=_load_rubric(rollout_dir),
        lineage=_load_lineage(rollout_dir),
    )


def _is_acp_rollout_dir(path: Path) -> bool:
    return (path / "trajectory" / "acp_trajectory.jsonl").exists()


def _load_result_json(rollout_dir: Path) -> dict[str, Any]:
    parsed = _load_json(rollout_dir / "result.json")
    return parsed if isinstance(parsed, dict) else {}
