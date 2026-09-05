"""Convert BenchFlow LLM trajectories into Prime-RL SFT JSONL.

Prime-RL's local SFT path consumes a Hugging Face-style dataset whose rows carry
OpenAI-compatible ``messages`` plus optional ``tool_defs`` / ``tools``. BenchFlow
already emits lower-level provider traffic as
``trajectory/llm_trajectory.jsonl``; this module reconstructs trainer rows from
those request/response exchanges without touching the existing Verifiers/ADP
exporters.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from benchflow._utils.json_safe import dumps_finite, scrub_non_finite
from benchflow.trajectories.message_contract import (
    PrimeSftExchangeData as PrimeSftExchangeData,
)
from benchflow.trajectories.message_contract import (
    PrimeSftTrajectoryJsonlError as PrimeSftTrajectoryJsonlError,
)
from benchflow.trajectories.message_contract import (
    _align_legacy_tool_call_ids,
    _has_tool_calls,
    _normalize_tool_call,
    _row_messages,
)
from benchflow.trajectories.message_contract import (
    load_llm_trajectory_jsonl as load_llm_trajectory_jsonl,
)
from benchflow.trajectories.message_contract import (
    normalize_prime_sft_exchange as normalize_prime_sft_exchange,
)
from benchflow.trajectories.message_contract import (
    prime_sft_last_user_training_window as prime_sft_last_user_training_window,
)
from benchflow.trajectories.message_contract import (
    validate_prime_sft_row as validate_prime_sft_row,
)
from benchflow.trajectories.types import redact_trajectory_obj

PrimeSftRowMode = Literal["rollout", "exchange"]


@dataclass
class PrimeSftExportStats:
    rollouts_seen: int = 0
    exchanges_seen: int = 0
    rows_written: int = 0
    rows_with_tool_calls: int = 0
    skipped_no_result: int = 0
    skipped_no_trajectory: int = 0
    skipped_reward: int = 0
    # Rollout-level: number of rollouts skipped because *every* captured exchange
    # was a provider error (non-200). Counts rollouts (+= 1), like the other
    # skipped_* fields, so the manifest stays comparable across granularities.
    skipped_provider_error: int = 0
    # Exchange-level companion: total non-200 exchanges across those skipped
    # rollouts. A single rollout with 20 failed calls adds 20 here but 1 above.
    skipped_exchanges_provider_error: int = 0
    skipped_no_assistant: int = 0
    skipped_missing_tool_defs: int = 0
    skipped_terminal_error: int = 0
    skipped_invalid: int = 0
    tool_call_ids_rewritten: int = 0
    tool_messages_merged: int = 0
    sources: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rollouts_seen": self.rollouts_seen,
            "exchanges_seen": self.exchanges_seen,
            "rows_written": self.rows_written,
            "rows_with_tool_calls": self.rows_with_tool_calls,
            "skipped_no_result": self.skipped_no_result,
            "skipped_no_trajectory": self.skipped_no_trajectory,
            "skipped_reward": self.skipped_reward,
            "skipped_provider_error": self.skipped_provider_error,
            "skipped_exchanges_provider_error": self.skipped_exchanges_provider_error,
            "skipped_no_assistant": self.skipped_no_assistant,
            "skipped_missing_tool_defs": self.skipped_missing_tool_defs,
            "skipped_terminal_error": self.skipped_terminal_error,
            "skipped_invalid": self.skipped_invalid,
            "tool_call_ids_rewritten": self.tool_call_ids_rewritten,
            "tool_messages_merged": self.tool_messages_merged,
            "sources": self.sources,
        }


def _json_line(record: dict[str, Any], *, redact: bool = True) -> str:
    # Redact secrets in the record's string values BEFORE serializing so the
    # emitted SFT row is always valid JSON; redacting the serialized text could
    # split a backslash escape next to a secret and corrupt the line.
    clean = scrub_non_finite(record)
    if redact:
        clean = redact_trajectory_obj(clean)
    clean = _sanitize_prime_sft_row_tool_call_arguments(clean, redact=redact)
    return dumps_finite(clean, default=str)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _iter_rollout_dirs(root: str | Path) -> list[Path]:
    path = Path(root)
    if (path / "result.json").is_file():
        return [path]
    if not path.is_dir():
        return []
    return sorted({p.parent for p in path.rglob("result.json")})


def _iter_selected_rollout_dirs(selection_path: str | Path) -> list[Path]:
    path = Path(selection_path)
    try:
        selection = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid canonical selection JSON: {path}: {exc}") from exc
    if not isinstance(selection, dict):
        raise ValueError(f"canonical selection must be an object: {path}")
    job_dir = Path(str(selection.get("job_dir") or ""))
    selected = selection.get("selected", selection.get("selection"))
    if not isinstance(selected, list):
        raise ValueError(
            f"canonical selection {path} must contain selected or selection list"
        )
    rollout_dirs = []
    for idx, row in enumerate(selected, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"canonical selection row {idx} must be an object")
        row = cast(dict[str, Any], row)
        raw_dir = row.get("rollout_dir")
        if not isinstance(raw_dir, str) or not raw_dir:
            raise ValueError(f"canonical selection row {idx} missing rollout_dir")
        rollout_dir = Path(raw_dir)
        if (
            not (rollout_dir / "result.json").is_file()
            and not rollout_dir.is_absolute()
        ):
            rollout_dir = job_dir / rollout_dir
        if not (rollout_dir / "result.json").is_file() and isinstance(
            row.get("result_json"), str
        ):
            result_json = Path(row["result_json"])
            if result_json.is_absolute() and not result_json.is_file():
                marker = f"/{path.parent.name}/"
                _, sep, suffix = str(result_json).partition(marker)
                if sep:
                    result_json = path.parent / suffix
            rollout_dir = result_json.parent
        if not (rollout_dir / "result.json").is_file():
            raise ValueError(f"selected rollout has no result.json: {rollout_dir}")
        rollout_dirs.append(rollout_dir)
    return rollout_dirs


def _reward_from_result(result: dict[str, Any] | None) -> float | None:
    if not isinstance(result, dict):
        return None
    rewards = result.get("rewards")
    if isinstance(rewards, dict):
        reward = rewards.get("reward")
        if isinstance(reward, (int, float)) and not isinstance(reward, bool):
            return float(reward)
    reward = result.get("reward")
    if isinstance(reward, (int, float)) and not isinstance(reward, bool):
        return float(reward)
    return None


def _sanitize_message_tool_call_arguments(
    messages: list[Any], *, redact: bool = True
) -> list[Any]:
    sanitized: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            sanitized.append(message)
            continue
        out = dict(message)
        tool_calls = out.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            out["tool_calls"] = [
                _normalize_tool_call(
                    cast(dict[str, Any], tool_call), idx, redact=redact
                )
                for idx, tool_call in enumerate(tool_calls)
                if isinstance(tool_call, dict)
            ]
        sanitized.append(out)
    return sanitized


def _sanitize_prime_sft_row_tool_call_arguments(
    row: dict[str, Any], *, redact: bool = True
) -> dict[str, Any]:
    out = dict(row)
    messages = out.get("messages")
    if isinstance(messages, list):
        out["messages"] = _sanitize_message_tool_call_arguments(messages, redact=redact)
    prompt = out.get("prompt")
    if isinstance(prompt, list):
        out["prompt"] = _sanitize_message_tool_call_arguments(prompt, redact=redact)
    completion = out.get("completion")
    if isinstance(completion, list):
        out["completion"] = _sanitize_message_tool_call_arguments(
            completion, redact=redact
        )
    return out


def _row_message_segments(
    row: dict[str, Any],
) -> list[tuple[Literal["messages", "prompt", "completion"], Any]]:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        return [("messages", message) for message in messages]
    prompt = row.get("prompt")
    completion = row.get("completion")
    if isinstance(prompt, list) and isinstance(completion, list):
        return [("prompt", message) for message in prompt] + [
            ("completion", message) for message in completion
        ]
    return []


def _canonicalize_existing_prime_sft_row(
    row: dict[str, Any],
    row_num: int,
    *,
    sanitize_tool_call_arguments: bool = False,
    redact: bool = True,
) -> tuple[dict[str, Any], dict[str, int]]:
    out = (
        _sanitize_prime_sft_row_tool_call_arguments(row, redact=redact)
        if sanitize_tool_call_arguments
        else dict(row)
    )
    segments = _row_message_segments(out)
    if not segments:
        validate_prime_sft_row(out, row_num)
        return out, {"tool_call_ids_rewritten": 0, "tool_messages_merged": 0}

    repaired, stats = _align_legacy_tool_call_ids(segments)
    segment_names = {segment for segment, _ in repaired}
    if segment_names == {"messages"}:
        out["messages"] = [message for _, message in repaired]
    else:
        out.pop("messages", None)
        out["prompt"] = [
            message for segment, message in repaired if segment == "prompt"
        ]
        out["completion"] = [
            message for segment, message in repaired if segment == "completion"
        ]
    validate_prime_sft_row(out, row_num)
    return out, stats


def validate_prime_sft_jsonl(
    jsonl: str | Path,
    *,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    path = Path(jsonl)
    rows = 0
    rows_with_tool_calls = 0
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
            validate_prime_sft_row(row, row_num)
            row_messages = _row_messages(row, row_num)
            typed_messages = [m for m in row_messages if isinstance(m, dict)]
            if _has_tool_calls(typed_messages):
                rows_with_tool_calls += 1
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"row count {rows} != expected {expected_rows}")
    return {"ok": True, "rows": rows, "rows_with_tool_calls": rows_with_tool_calls}


def _iter_prime_sft_jsonl_rows(
    path: Path,
    *,
    sanitize_tool_call_arguments: bool = False,
    redact: bool = True,
) -> Iterator[tuple[int, dict[str, Any], dict[str, int]]]:
    with path.open("r", encoding="utf-8") as handle:
        for row_num, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"row {row_num}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"row {row_num}: top-level row must be object")
            row, repair_stats = _canonicalize_existing_prime_sft_row(
                row,
                row_num,
                sanitize_tool_call_arguments=sanitize_tool_call_arguments,
                redact=redact,
            )
            yield row_num, row, repair_stats


def _row_reward(row: dict[str, Any]) -> float | None:
    reward = row.get("reward", row.get("score"))
    if isinstance(reward, (int, float)) and not isinstance(reward, bool):
        return float(reward)
    return None


def _compact_existing_prime_sft_row(
    row: dict[str, Any],
    *,
    row_num: int,
    source: Path,
) -> dict[str, Any]:
    """Emit a compact trainer row from a validated existing JSONL row."""
    messages = [
        message for message in _row_messages(row, row_num) if isinstance(message, dict)
    ]
    if _is_benchflow_results_row(row):
        window = prime_sft_last_user_training_window(messages)
        if window is not None:
            prompt, completion = window
            messages = prompt + completion
    out: dict[str, Any] = {"messages": messages}

    tools = row.get("tool_defs", row.get("tools"))
    if tools is not None:
        out["tool_defs"] = tools

    reward = _row_reward(row)
    if reward is not None:
        out["reward"] = reward

    raw_info = row.get("info")
    info: dict[str, Any] = (
        cast(dict[str, Any], raw_info) if isinstance(raw_info, dict) else {}
    )
    task_id = (
        row.get("task_id")
        or row.get("task_name")
        or info.get("task_id")
        or info.get("task_name")
    )
    if task_id:
        out["task_id"] = str(task_id)
    task_name = row.get("task_name") or info.get("task_name") or task_id
    if task_name:
        out["task_name"] = str(task_name)
    if isinstance(row.get("model"), str):
        out["model"] = row["model"]
    elif isinstance(info.get("model"), str):
        out["model"] = info["model"]
    if isinstance(row.get("agent"), str):
        out["agent"] = row["agent"]
    elif isinstance(info.get("agent"), str):
        out["agent"] = info["agent"]
    if isinstance(row.get("example_id"), int):
        out["source_example_id"] = row["example_id"]
    if isinstance(info.get("rollout_dir"), str):
        out["source_rollout_dir"] = info["rollout_dir"]
    if row.get("chat_template_kwargs") is not None:
        out["chat_template_kwargs"] = row["chat_template_kwargs"]
    out["source_path"] = str(source)
    out["source_index"] = row_num - 1
    out["source_format"] = "benchflow-results-jsonl"
    return out


def _is_benchflow_results_row(row: dict[str, Any]) -> bool:
    info = row.get("info")
    if isinstance(info, dict) and info.get("source") == "benchflow":
        return True
    return any(
        key in row
        for key in (
            "trajectory",
            "stop_condition",
            "token_usage",
            "is_completed",
            "is_truncated",
        )
    )


def _benchflow_row_training_skip_reason(row: dict[str, Any]) -> str | None:
    if not _is_benchflow_results_row(row):
        return None
    info = row.get("info")
    if isinstance(info, dict) and info.get("training_ready") is False:
        return "not_training_ready"
    if row.get("error"):
        return "terminal_error"
    if row.get("is_truncated") is True:
        return "partial_trajectory"
    stop_condition = row.get("stop_condition")
    if isinstance(stop_condition, str) and stop_condition not in {
        "",
        "agent_completed",
    }:
        return stop_condition
    return None


def _existing_prime_sft_jsonl_stats(
    path: Path,
    *,
    min_reward: float | None,
    redact: bool = True,
) -> PrimeSftExportStats:
    stats = PrimeSftExportStats(sources=[str(path)])
    for row_num, row, repair_stats in _iter_prime_sft_jsonl_rows(
        path, sanitize_tool_call_arguments=True, redact=redact
    ):
        stats.tool_call_ids_rewritten += repair_stats["tool_call_ids_rewritten"]
        stats.tool_messages_merged += repair_stats["tool_messages_merged"]
        stats.rollouts_seen += 1
        if _benchflow_row_training_skip_reason(row) is not None:
            stats.skipped_terminal_error += 1
            continue
        reward = _row_reward(row)
        if min_reward is not None and (reward is None or reward < min_reward):
            stats.skipped_reward += 1
            continue
        messages = _row_messages(row, row_num)
        typed_messages = [m for m in messages if isinstance(m, dict)]
        if _has_tool_calls(typed_messages):
            stats.rows_with_tool_calls += 1
        stats.rows_written += 1
    return stats


def _copy_existing_prime_sft_jsonl(
    source: Path,
    out: Path,
    *,
    min_reward: float | None,
    redact: bool = True,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row_num, row, _ in _iter_prime_sft_jsonl_rows(
            source, sanitize_tool_call_arguments=True, redact=redact
        ):
            if _benchflow_row_training_skip_reason(row) is not None:
                continue
            if min_reward is not None:
                reward = _row_reward(row)
                if reward is None or reward < min_reward:
                    continue
            compact = _compact_existing_prime_sft_row(
                row,
                row_num=row_num,
                source=source,
            )
            validate_prime_sft_row(compact, row_num)
            handle.write(_json_line(compact, redact=redact) + "\n")


def _row_from_exchange(
    *,
    exchange: dict[str, Any],
    rollout_dir: Path,
    result: dict[str, Any] | None,
    reward: float | None,
    exchange_idx: int,
    redact: bool = True,
) -> tuple[dict[str, Any] | None, str | None]:
    normalized, skip_reason = normalize_prime_sft_exchange(exchange, redact=redact)
    if skip_reason:
        return None, skip_reason
    if normalized is None:
        return None, "invalid_prime_sft_row"
    messages = normalized.messages
    window = prime_sft_last_user_training_window(messages)
    if window is not None:
        prompt, completion = window
        messages = prompt + completion

    agent_result = result.get("agent_result") if isinstance(result, dict) else None
    row = {
        "messages": messages,
        "tool_defs": normalized.tool_defs,
        "task_name": (result or {}).get("task_name") or rollout_dir.name,
        "source": "benchflow-llm-trajectory",
        "source_path": str(rollout_dir / "trajectory" / "llm_trajectory.jsonl"),
        "exchange_index": exchange_idx,
        "reward": reward,
        "score": reward,
        "model": ((exchange.get("request") or {}).get("body") or {}).get("model"),
        "agent": (result or {}).get("agent"),
        "token_usage": agent_result if isinstance(agent_result, dict) else None,
    }
    return {key: value for key, value in row.items() if value is not None}, None


def convert_benchflow_rollouts_to_prime_sft_rows(
    jobs_dir: str | Path,
    *,
    min_reward: float | None = None,
    row_mode: PrimeSftRowMode = "rollout",
    canonical_selection: str | Path | None = None,
    redact: bool = True,
) -> tuple[list[dict[str, Any]], PrimeSftExportStats]:
    stats = PrimeSftExportStats()
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
            (idx, ex)
            for idx, ex in enumerate(exchanges)
            if ((ex.get("response") or {}).get("status_code") == 200)
        ]
        if not successful:
            stats.skipped_provider_error += 1
            stats.skipped_exchanges_provider_error += len(exchanges)
            continue

        candidates = successful if row_mode == "exchange" else [successful[-1]]
        for idx, exchange in candidates:
            row, skip_reason = _row_from_exchange(
                exchange=exchange,
                rollout_dir=rollout_dir,
                result=result,
                reward=reward,
                exchange_idx=idx,
                redact=redact,
            )
            if skip_reason == "no_assistant":
                stats.skipped_no_assistant += 1
                continue
            if skip_reason == "missing_tool_defs":
                stats.skipped_missing_tool_defs += 1
                continue
            if row is None:
                stats.skipped_invalid += 1
                continue
            try:
                validate_prime_sft_row(row, len(rows) + 1)
            except ValueError:
                stats.skipped_invalid += 1
                continue
            rows.append(row)
            stats.rows_written += 1
            if _has_tool_calls(row["messages"]):
                stats.rows_with_tool_calls += 1
            stats.sources.append(str(trajectory_path))

    return rows, stats


def _result_training_skip_reason(result: dict[str, Any]) -> str | None:
    if result.get("error"):
        return "agent_error"
    if result.get("verifier_error"):
        return "verifier_error"
    if result.get("partial_trajectory") is True:
        return "partial_trajectory"
    return None


def export_prime_sft_jsonl(
    jobs_dir: str | Path,
    out: str | Path,
    *,
    min_reward: float | None = None,
    row_mode: PrimeSftRowMode = "rollout",
    expected_rows: int | None = None,
    manifest: str | Path | None = None,
    canonical_selection: str | Path | None = None,
    redact: bool = True,
) -> PrimeSftExportStats:
    """Export BenchFlow rollouts to a Prime-RL SFT JSONL file.

    When ``expected_rows`` is set, the row-count assertion fires *before* the
    output file (or manifest) is opened — so a mismatch raises ``ValueError`` and
    writes nothing, rather than leaving a partial file. Callers should not expect
    ``out`` to exist on failure.
    """
    source_path = Path(jobs_dir)
    out_path = Path(out)
    manifest_path = Path(manifest) if manifest is not None else None

    if source_path.is_file() and source_path.suffix == ".jsonl":
        if source_path.resolve() == out_path.resolve():
            raise ValueError("--out must differ from the source JSONL path")
        stats = _existing_prime_sft_jsonl_stats(
            source_path, min_reward=min_reward, redact=redact
        )
        if expected_rows is not None and stats.rows_written != expected_rows:
            raise ValueError(
                f"row count {stats.rows_written} != expected {expected_rows}"
            )
        _copy_existing_prime_sft_jsonl(
            source_path,
            out_path,
            min_reward=min_reward,
            redact=redact,
        )
        if manifest_path is not None:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(stats.as_dict(), indent=2, sort_keys=True) + "\n"
            )
        return stats

    rows, stats = convert_benchflow_rollouts_to_prime_sft_rows(
        jobs_dir,
        min_reward=min_reward,
        row_mode=row_mode,
        canonical_selection=canonical_selection,
        redact=redact,
    )
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"row count {len(rows)} != expected {expected_rows}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_json_line(row, redact=redact) + "\n")

    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(stats.as_dict(), indent=2, sort_keys=True) + "\n"
        )

    return stats
