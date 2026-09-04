"""Normalize untrusted rollout artifacts into the typed viewer payload."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from benchflow._utils.json_safe import dumps_finite, scrub_non_finite

from .models import (
    ErrorBanner,
    JsonObject,
    JsonValue,
    MessageStep,
    Meta,
    PromptStep,
    RolloutMetadata,
    RubricCriterion,
    RubricReview,
    Step,
    StepCounts,
    ThoughtStep,
    TimeoutInfo,
    TimeoutStep,
    Timing,
    ToolCall,
    ToolStep,
    UnknownStep,
    Usage,
    VerifierArtifacts,
    VerifierTest,
    ViewerPayload,
    normalize_tool_status,
    normalize_verifier_status,
    tool_hue,
)

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
    """Project raw ACP event variants onto explicit renderer step variants."""
    steps: list[Step] = []
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

    for event in events:
        if not _within_json_depth(event):
            continue
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
            kind = _display_text(event.get("kind")) or "other"
            title = _display_text(event.get("title"))
            started, duration = _tool_timing(event)
            steps.append(
                ToolStep(
                    i=next_index(),
                    tool=ToolCall(
                        id=_display_text(event.get("tool_call_id")),
                        kind=kind,
                        title=title,
                        status=normalize_tool_status(event.get("status")),
                        content=_tool_content_texts(event.get("content")),
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
    return steps


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


def _normalize_metadata(
    result_data: dict[str, Any], timing_data: dict[str, Any] | None
) -> RolloutMetadata:
    """Canonical result/timing boundary shared by detail and catalog views."""
    if not _within_json_depth(result_data):
        result_data = {}
    rewards = result_data.get("rewards")
    reward = _finite_float(rewards.get("reward")) if isinstance(rewards, dict) else None
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


def _build_meta(metadata: RolloutMetadata, steps: list[Step]) -> Meta:
    counts = StepCounts(
        prompts=sum(step.kind == "prompt" for step in steps),
        messages=sum(step.kind == "message" for step in steps),
        thoughts=sum(step.kind == "thought" for step in steps),
        tools=sum(step.kind == "tool" for step in steps),
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
        errors=metadata.errors,
        trajectory_source=metadata.trajectory_source,
        partial_trajectory=metadata.partial_trajectory,
        started_at=metadata.started_at,
        finished_at=metadata.finished_at,
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
        rows.append(
            RubricCriterion(
                name=name,
                blocker=bool(meta.get("blocker")),
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
    for trial in report.get("trials") or []:
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
    return ViewerPayload(
        rollout_name=rollout_dir.name,
        meta=_build_meta(_load_rollout_metadata(rollout_dir), steps),
        steps=steps,
        verifier=_load_verifier(rollout_dir),
        rubric=_load_rubric(rollout_dir),
    )


def _is_acp_rollout_dir(path: Path) -> bool:
    return (path / "trajectory" / "acp_trajectory.jsonl").exists()


def _load_result_json(rollout_dir: Path) -> dict[str, Any]:
    parsed = _load_json(rollout_dir / "result.json")
    return parsed if isinstance(parsed, dict) else {}
