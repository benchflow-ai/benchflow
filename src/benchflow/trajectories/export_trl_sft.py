"""Convert BenchFlow LLM trajectories into native TRL SFT JSONL."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from benchflow._utils.json_safe import dumps_finite, scrub_non_finite
from benchflow.trajectories.call_purpose import infer_call_purpose
from benchflow.trajectories.export_prime_sft import (
    _benchflow_row_training_skip_reason,
    _iter_rollout_dirs,
    _iter_selected_rollout_dirs,
    _load_json,
    _result_training_skip_reason,
    _reward_from_result,
    _row_reward,
    load_llm_trajectory_jsonl,
    normalize_prime_sft_exchange,
    validate_prime_sft_row,
)
from benchflow.trajectories.types import redact_trajectory_obj

TrlSftRowMode = Literal["rollout", "exchange"]


@dataclass
class TrlSftExportStats:
    rollouts_seen: int = 0
    exchanges_seen: int = 0
    rows_written: int = 0
    rows_with_tool_calls: int = 0
    skipped_no_result: int = 0
    skipped_no_trajectory: int = 0
    skipped_reward: int = 0
    skipped_provider_error: int = 0
    skipped_no_assistant: int = 0
    skipped_missing_tools: int = 0
    skipped_terminal_error: int = 0
    skipped_helper_calls: int = 0
    skipped_invalid: int = 0
    sources: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": "trl-sft",
            "rollouts_seen": self.rollouts_seen,
            "exchanges_seen": self.exchanges_seen,
            "rows_written": self.rows_written,
            "rows_with_tool_calls": self.rows_with_tool_calls,
            "skipped_no_result": self.skipped_no_result,
            "skipped_no_trajectory": self.skipped_no_trajectory,
            "skipped_reward": self.skipped_reward,
            "skipped_provider_error": self.skipped_provider_error,
            "skipped_no_assistant": self.skipped_no_assistant,
            "skipped_missing_tools": self.skipped_missing_tools,
            "skipped_terminal_error": self.skipped_terminal_error,
            "skipped_helper_calls": self.skipped_helper_calls,
            "skipped_invalid": self.skipped_invalid,
            "sources": self.sources,
        }


def _json_line(record: dict[str, Any], *, redact: bool) -> str:
    clean = scrub_non_finite(record)
    if redact:
        clean = redact_trajectory_obj(clean)
    return dumps_finite(clean, default=str)


def _call_purpose(exchange: dict[str, Any], *, agent: str | None) -> str:
    metadata = exchange.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("call_purpose"), str):
        return cast(str, metadata["call_purpose"])
    request = exchange.get("request")
    body = request.get("body") if isinstance(request, dict) else {}
    if not isinstance(body, dict):
        return "helper"
    return infer_call_purpose(agent_name=agent or "", request_body=body)


def _arguments_as_object(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"_malformed_json_arguments": arguments}
        if isinstance(parsed, dict):
            return parsed
        return {"_non_object_json_arguments": parsed}
    if arguments is None:
        return {}
    return {"_non_object_arguments": arguments}


def _trl_message(message: dict[str, Any]) -> dict[str, Any]:
    out = dict(message)
    tool_calls = out.get("tool_calls")
    if isinstance(tool_calls, list):
        normalized = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            call = dict(tool_call)
            function = call.get("function")
            if isinstance(function, dict):
                function = dict(function)
                function["arguments"] = _arguments_as_object(function.get("arguments"))
                call["function"] = function
            normalized.append(call)
        out["tool_calls"] = normalized
    return out


def _has_tool_calls(messages: list[dict[str, Any]]) -> bool:
    return any(bool(message.get("tool_calls")) for message in messages)


def _load_tokenizer(tokenizer_id: str, revision: str | None) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ValueError(
            "tokenizer validation requires the benchflow train or trl extra"
        ) from exc
    kwargs = {"revision": revision} if revision else {}
    return AutoTokenizer.from_pretrained(tokenizer_id, **kwargs)


def _training_chat_template(tokenizer: Any) -> str | None:
    try:
        from trl.chat_template_utils import get_training_chat_template
    except ImportError as exc:
        raise ValueError(
            "assistant-mask validation requires the benchflow trl extra"
        ) from exc
    return get_training_chat_template(tokenizer)


def _input_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError("tokenizer apply_chat_template did not return input_ids")
    return value


def _assistant_masks(value: Any) -> list[int]:
    if not isinstance(value, Mapping):
        raise ValueError("tokenizer did not return assistant token masks")
    masks = value.get("assistant_masks")
    if isinstance(masks, list) and masks and isinstance(masks[0], list):
        masks = masks[0]
    if not isinstance(masks, list) or not all(
        isinstance(item, int | bool) for item in masks
    ):
        raise ValueError("chat template does not provide assistant token masks")
    return [int(item) for item in masks]


def _validate_tokenized_row(
    row: dict[str, Any],
    *,
    row_num: int,
    tokenizer: Any,
    chat_template: str | None,
    max_length: int | None,
) -> tuple[int, int]:
    prompt = cast(list[dict[str, Any]], row["prompt"])
    completion = cast(list[dict[str, Any]], row["completion"])
    tools = cast(list[dict[str, Any]], row.get("tools") or [])
    template_kwargs: dict[str, Any] = {"tools": tools or None}
    if chat_template is not None:
        template_kwargs["chat_template"] = chat_template
    prompt_output = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        **template_kwargs,
    )
    full_output = tokenizer.apply_chat_template(
        prompt + completion,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        **template_kwargs,
    )
    prompt_ids = _input_ids(prompt_output)
    full_ids = _input_ids(full_output)
    assistant_masks = _assistant_masks(full_output)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"row {row_num}: tokenized prompt is not a prefix of prompt+completion"
        )
    if len(assistant_masks) != len(full_ids):
        raise ValueError(
            f"row {row_num}: assistant mask length does not match input_ids"
        )
    trainable = sum(assistant_masks[len(prompt_ids) :])
    if trainable < 1:
        raise ValueError(f"row {row_num}: no trainable assistant completion tokens")
    if max_length is not None and len(full_ids) > max_length:
        raise ValueError(
            f"row {row_num}: tokenized length {len(full_ids)} exceeds "
            f"max_length {max_length}"
        )
    return len(full_ids), trainable


def _step_call_purpose(
    step: dict[str, Any],
    *,
    agent: str | None,
    tools: list[dict[str, Any]],
) -> str:
    extras = step.get("extras")
    if isinstance(extras, dict) and isinstance(extras.get("call_purpose"), str):
        return cast(str, extras["call_purpose"])
    return _call_purpose(
        {
            "request": {
                "body": {
                    "messages": step.get("prompt"),
                    "tools": tools,
                }
            }
        },
        agent=agent,
    )


def validate_trl_sft_row(row: dict[str, Any], row_num: int = 1) -> None:
    prompt = row.get("prompt")
    completion = row.get("completion")
    tools = row.get("tools")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"row {row_num}: prompt must be a non-empty message list")
    if (
        not isinstance(completion, list)
        or len(completion) != 1
        or not isinstance(completion[0], dict)
        or completion[0].get("role") != "assistant"
    ):
        raise ValueError(
            f"row {row_num}: completion must contain exactly one assistant message"
        )
    if "tool_defs" in row:
        raise ValueError(f"row {row_num}: TRL rows must use tools, not tool_defs")
    validate_prime_sft_row(
        {"prompt": prompt, "completion": completion, "tools": tools},
        row_num,
    )
    for message_idx, message in enumerate(prompt + completion):
        if not isinstance(message, dict):
            continue
        for call_idx, tool_call in enumerate(message.get("tool_calls") or []):
            function = (
                tool_call.get("function") if isinstance(tool_call, dict) else None
            )
            arguments = (
                function.get("arguments") if isinstance(function, dict) else None
            )
            if not isinstance(arguments, dict):
                raise ValueError(
                    f"row {row_num}: messages[{message_idx}].tool_calls"
                    f"[{call_idx}].function.arguments must be an object"
                )


def _row_from_exchange(
    *,
    exchange: dict[str, Any],
    rollout_dir: Path,
    result: dict[str, Any],
    reward: float | None,
    exchange_idx: int,
    redact: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    normalized, skip_reason = normalize_prime_sft_exchange(
        exchange,
        redact=redact,
    )
    if skip_reason:
        return None, skip_reason
    if normalized is None:
        return None, "invalid_trl_sft_row"
    prompt = [_trl_message(message) for message in normalized.messages[:-1]]
    completion = [_trl_message(normalized.messages[-1])]
    row = {
        "prompt": prompt,
        "completion": completion,
        "tools": normalized.tool_defs,
        "reward": reward,
        "task_id": result.get("task_name") or rollout_dir.name,
        "agent": result.get("agent"),
        "model": ((exchange.get("request") or {}).get("body") or {}).get("model"),
        "exchange_index": exchange_idx,
        "call_purpose": "agent",
        "source": "benchflow-llm-trajectory",
        "source_path": str(rollout_dir / "trajectory" / "llm_trajectory.jsonl"),
        "source_rollout_dir": str(rollout_dir),
    }
    agent_result = result.get("agent_result")
    if isinstance(agent_result, dict):
        row["token_usage"] = agent_result
    return {key: value for key, value in row.items() if value is not None}, None


def convert_benchflow_rollouts_to_trl_sft_rows(
    jobs_dir: str | Path,
    *,
    min_reward: float | None = None,
    row_mode: TrlSftRowMode = "exchange",
    canonical_selection: str | Path | None = None,
    redact: bool = True,
) -> tuple[list[dict[str, Any]], TrlSftExportStats]:
    if row_mode not in {"rollout", "exchange"}:
        raise ValueError("row_mode must be rollout or exchange")
    stats = TrlSftExportStats()
    rows: list[dict[str, Any]] = []
    rollout_dirs = (
        _iter_selected_rollout_dirs(canonical_selection)
        if canonical_selection is not None
        else _iter_rollout_dirs(jobs_dir)
    )
    for rollout_dir in rollout_dirs:
        stats.rollouts_seen += 1
        result = _load_json(rollout_dir / "result.json")
        if result is None:
            stats.skipped_no_result += 1
            continue
        if _result_training_skip_reason(result) is not None:
            stats.skipped_terminal_error += 1
            continue
        reward = _reward_from_result(result)
        if min_reward is not None and (reward is None or reward < min_reward):
            stats.skipped_reward += 1
            continue
        trajectory_path = rollout_dir / "trajectory" / "llm_trajectory.jsonl"
        exchanges = load_llm_trajectory_jsonl(trajectory_path, strict=True)
        if not exchanges:
            stats.skipped_no_trajectory += 1
            continue
        stats.exchanges_seen += len(exchanges)
        successful = [
            (idx, exchange)
            for idx, exchange in enumerate(exchanges)
            if ((exchange.get("response") or {}).get("status_code") == 200)
        ]
        if not successful:
            stats.skipped_provider_error += 1
            continue
        agent = result.get("agent")
        primary = []
        for idx, exchange in successful:
            if _call_purpose(exchange, agent=agent) != "agent":
                stats.skipped_helper_calls += 1
                continue
            primary.append((idx, exchange))
        candidates = primary if row_mode == "exchange" else primary[-1:]
        for exchange_idx, exchange in candidates:
            row, skip_reason = _row_from_exchange(
                exchange=exchange,
                rollout_dir=rollout_dir,
                result=result,
                reward=reward,
                exchange_idx=exchange_idx,
                redact=redact,
            )
            if skip_reason == "no_assistant":
                stats.skipped_no_assistant += 1
                continue
            if skip_reason == "missing_tool_defs":
                stats.skipped_missing_tools += 1
                continue
            if row is None:
                stats.skipped_invalid += 1
                continue
            try:
                validate_trl_sft_row(row, len(rows) + 1)
            except ValueError:
                stats.skipped_invalid += 1
                continue
            rows.append(row)
            stats.rows_written += 1
            if _has_tool_calls(row["prompt"] + row["completion"]):
                stats.rows_with_tool_calls += 1
            stats.sources.append(str(trajectory_path))
    return rows, stats


def _convert_results_jsonl_to_trl_sft_rows(
    path: Path,
    *,
    min_reward: float | None,
    row_mode: TrlSftRowMode,
) -> tuple[list[dict[str, Any]], TrlSftExportStats]:
    stats = TrlSftExportStats(sources=[str(path)])
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                source_row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: line {source_index + 1}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(source_row, dict):
                raise ValueError(
                    f"{path}: line {source_index + 1}: row must be an object"
                )
            stats.rollouts_seen += 1
            if all(key in source_row for key in ("prompt", "completion", "tools")):
                reward = _row_reward(source_row)
                if min_reward is not None and (reward is None or reward < min_reward):
                    stats.skipped_reward += 1
                    continue
                native_row = dict(source_row)
                native_row["prompt"] = [
                    _trl_message(cast(dict[str, Any], message))
                    for message in source_row["prompt"]
                    if isinstance(message, dict)
                ]
                native_row["completion"] = [
                    _trl_message(cast(dict[str, Any], message))
                    for message in source_row["completion"]
                    if isinstance(message, dict)
                ]
                native_row["source_path"] = str(path)
                native_row["source_format"] = "trl-sft"
                native_row["source_index"] = source_index
                validate_trl_sft_row(native_row, len(rows) + 1)
                rows.append(native_row)
                stats.rows_written += 1
                if _has_tool_calls(native_row["prompt"] + native_row["completion"]):
                    stats.rows_with_tool_calls += 1
                continue
            if _benchflow_row_training_skip_reason(source_row) is not None:
                stats.skipped_terminal_error += 1
                continue
            reward = _row_reward(source_row)
            if min_reward is not None and (reward is None or reward < min_reward):
                stats.skipped_reward += 1
                continue
            trajectory = source_row.get("trajectory")
            if not isinstance(trajectory, list) or not trajectory:
                stats.skipped_no_trajectory += 1
                continue
            raw_tools = source_row.get("tools", source_row.get("tool_defs"))
            tools = (
                [
                    cast(dict[str, Any], tool)
                    for tool in raw_tools
                    if isinstance(tool, dict)
                ]
                if isinstance(raw_tools, list)
                else []
            )
            info = source_row.get("info")
            info = cast(dict[str, Any], info) if isinstance(info, dict) else {}
            agent = source_row.get("agent") or info.get("agent")
            model = source_row.get("model") or info.get("model")
            task_id = (
                source_row.get("task_id")
                or source_row.get("task_name")
                or info.get("task_id")
                or info.get("task_name")
            )
            source_rollout_dir = info.get("rollout_dir")
            successful: list[tuple[int, dict[str, Any]]] = []
            for fallback_index, raw_step in enumerate(trajectory):
                if not isinstance(raw_step, dict):
                    stats.skipped_invalid += 1
                    continue
                step = cast(dict[str, Any], raw_step)
                stats.exchanges_seen += 1
                if _step_call_purpose(step, agent=agent, tools=tools) != "agent":
                    stats.skipped_helper_calls += 1
                    continue
                successful.append((fallback_index, step))
            candidates = successful if row_mode == "exchange" else successful[-1:]
            for fallback_index, step in candidates:
                prompt = step.get("prompt")
                completion = step.get("completion")
                if not isinstance(prompt, list) or not isinstance(completion, list):
                    stats.skipped_invalid += 1
                    continue
                extras = step.get("extras")
                extras = (
                    cast(dict[str, Any], extras) if isinstance(extras, dict) else {}
                )
                row = {
                    "prompt": [
                        _trl_message(cast(dict[str, Any], message))
                        for message in prompt
                        if isinstance(message, dict)
                    ],
                    "completion": [
                        _trl_message(cast(dict[str, Any], message))
                        for message in completion
                        if isinstance(message, dict)
                    ],
                    "tools": tools,
                    "reward": reward,
                    "task_id": task_id,
                    "agent": agent,
                    "model": model,
                    "exchange_index": extras.get("exchange_index", fallback_index),
                    "call_purpose": "agent",
                    "source": "benchflow-results-jsonl",
                    "source_path": str(path),
                    "source_rollout_dir": source_rollout_dir,
                    "source_format": "benchflow-results-jsonl",
                    "source_index": source_index,
                }
                row = {key: value for key, value in row.items() if value is not None}
                try:
                    validate_trl_sft_row(row, len(rows) + 1)
                except ValueError:
                    stats.skipped_invalid += 1
                    continue
                rows.append(row)
                stats.rows_written += 1
                row_prompt = cast(list[dict[str, Any]], row["prompt"])
                row_completion = cast(list[dict[str, Any]], row["completion"])
                if _has_tool_calls(row_prompt + row_completion):
                    stats.rows_with_tool_calls += 1
    return rows, stats


def export_trl_sft_jsonl(
    jobs_dir: str | Path,
    out: str | Path,
    *,
    min_reward: float | None = None,
    row_mode: TrlSftRowMode = "exchange",
    expected_rows: int | None = None,
    manifest: str | Path | None = None,
    canonical_selection: str | Path | None = None,
    redact: bool = True,
) -> TrlSftExportStats:
    source_path = Path(jobs_dir)
    if source_path.is_file() and source_path.suffix == ".jsonl":
        if source_path.resolve() == Path(out).resolve():
            raise ValueError("--out must differ from the source JSONL path")
        if canonical_selection is not None:
            raise ValueError("--canonical-selection requires a jobs directory")
        rows, stats = _convert_results_jsonl_to_trl_sft_rows(
            source_path,
            min_reward=min_reward,
            row_mode=row_mode,
        )
    else:
        rows, stats = convert_benchflow_rollouts_to_trl_sft_rows(
            jobs_dir,
            min_reward=min_reward,
            row_mode=row_mode,
            canonical_selection=canonical_selection,
            redact=redact,
        )
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"row count {len(rows)} != expected {expected_rows}")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_json_line(row, redact=redact) + "\n")
    if manifest is not None:
        manifest_path = Path(manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(stats.as_dict(), indent=2, sort_keys=True) + "\n"
        )
    return stats


def validate_trl_sft_jsonl(
    jsonl: str | Path,
    *,
    expected_rows: int | None = None,
    tokenizer_id: str | None = None,
    tokenizer_revision: str | None = None,
    max_length: int | None = None,
) -> dict[str, Any]:
    if max_length is not None and max_length < 1:
        raise ValueError("max_length must be positive")
    if max_length is not None and tokenizer_id is None:
        raise ValueError("max_length requires --tokenizer")
    path = Path(jsonl)
    rows = 0
    rows_with_tool_calls = 0
    tokenizer = (
        _load_tokenizer(tokenizer_id, tokenizer_revision) if tokenizer_id else None
    )
    chat_template = (
        _training_chat_template(tokenizer) if tokenizer is not None else None
    )
    token_lengths: list[int] = []
    trainable_counts: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_num, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"row {row_num}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"row {row_num}: top-level row must be object")
            validate_trl_sft_row(row, row_num)
            messages = [
                message
                for message in row["prompt"] + row["completion"]
                if isinstance(message, dict)
            ]
            if _has_tool_calls(messages):
                rows_with_tool_calls += 1
            if tokenizer is not None:
                token_length, trainable = _validate_tokenized_row(
                    row,
                    row_num=row_num,
                    tokenizer=tokenizer,
                    chat_template=chat_template,
                    max_length=max_length,
                )
                token_lengths.append(token_length)
                trainable_counts.append(trainable)
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"row count {rows} != expected {expected_rows}")
    result: dict[str, Any] = {
        "ok": True,
        "format": "trl-sft",
        "rows": rows,
        "rows_with_tool_calls": rows_with_tool_calls,
    }
    if tokenizer_id is not None and token_lengths:
        ordered = sorted(token_lengths)
        p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
        median = statistics.median(ordered)
        median_value: int | float = (
            int(median) if isinstance(median, float) and median.is_integer() else median
        )
        result["tokenization"] = {
            "tokenizer": tokenizer_id,
            "tokenizer_revision": tokenizer_revision,
            "max_length": max_length,
            "min_tokens": ordered[0],
            "median_tokens": median_value,
            "p95_tokens": ordered[p95_index],
            "max_tokens": ordered[-1],
            "min_trainable_assistant_tokens": min(trainable_counts),
        }
    return result
