"""A rollout's model calls as whole conversations, for the verl, SkyRL and OpenEnv exports.

TRL's per-call rows (``--row-mode exchange``) hold exactly the model calls BenchFlow trains: a
finished call of the agent whose tool input parsed, each with the conversation before it as its
prompt. verl's ``MultiTurnSFTDataset`` trains every assistant message of a row, SkyRL's
``tokenize_chat_example`` the last one or all of them, and an OpenEnv episode is a whole
transcript. Those formats need the calls as conversations and need to know which assistant
turns are trained.

A call joins the conversation its prompt extends (same tools); its completion is a trained turn.
Every other assistant turn of a conversation is context, and its reason is kept: a call cut at
the output cap, one whose tool input was not a JSON object, one trained in another row.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchflow._utils.json_safe import dumps_finite, scrub_non_finite
from benchflow.trajectories.export_prime_sft import load_llm_trajectory_jsonl
from benchflow.trajectories.export_trl_sft import (
    TrlSftExportStats,
    TrlSftRowMode,
    _calls_tool_with_unparsed_input,
    _trl_message,
    convert_benchflow_rollouts_to_trl_sft_rows,
)
from benchflow.trajectories.results import (
    _response_is_training_success,
    _response_tool_calls,
)
from benchflow.trajectories.sft_subagents import SubagentConversionCounts
from benchflow.trajectories.training_signal import GroupAdvantage
from benchflow.trajectories.types import redact_trajectory_obj

# Why an assistant turn of a conversation is context, not a training target.
CONTEXT_REASONS = {
    "output_cap": "cut at the output cap",
    "unfinished": "not a finished call",
    "unparsed_tool_input": "with tool input that was not a JSON object",
    "trained_in_another_row": "trained in another row",
    "other": "not a call BenchFlow trains",
}


@dataclass
class Conversation:
    """One conversation of a rollout: messages, tools, and which assistant turns are trained."""

    rollout: str | None
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    trained: list[int]
    # The per-call rows whose completions are the trained turns, in order.
    calls: list[dict[str, Any]]
    # Index of every other assistant turn -> a CONTEXT_REASONS key.
    context: dict[int, str] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        """Row fields of the last call (reward, task, model, signals), with every call index."""
        meta = {
            key: value
            for key, value in self.calls[-1].items()
            if key not in {"prompt", "completion", "tools", "exchange_index"}
        }
        meta["exchange_indices"] = [call["exchange_index"] for call in self.calls]
        return meta


@dataclass
class RlExportStats:
    format: str
    row_mode: str
    # The per-call conversion these rows are built from.
    per_call: TrlSftExportStats
    rows_written: int = 0
    rows_with_tool_calls: int = 0
    trained_turns: int = 0
    context_turns: Counter[str] = field(default_factory=Counter)
    # skyrl-sft rollout rows end before the first turn that is context.
    rows_cut: int = 0
    skipped_no_trained_turn: int = 0
    trained_turns_left_out: int = 0
    # openenv-episodes: an episode needs a reward; None is never written as 0.
    skipped_unscored: int = 0
    skipped_no_transcript: int = 0

    # What the `bench train convert` summary reads from every format's stats.
    @property
    def rollouts_seen(self) -> int:
        return self.per_call.rollouts_seen

    @property
    def skipped_control(self) -> int:
        return self.per_call.skipped_control

    @property
    def subagents(self) -> SubagentConversionCounts:
        return self.per_call.subagents

    @property
    def training_signal(self) -> dict[str, Any] | None:
        return self.per_call.training_signal

    def count(self, conversations: Sequence[Conversation]) -> None:
        for conversation in conversations:
            self.rows_written += 1
            self.trained_turns += len(conversation.trained)
            self.context_turns.update(conversation.context.values())
            if any(m.get("tool_calls") for m in conversation.messages):
                self.rows_with_tool_calls += 1

    def as_dict(self) -> dict[str, Any]:
        per_call = self.per_call.as_dict()
        per_call.pop("format", None)
        return {
            "format": self.format,
            "row_mode": self.row_mode,
            "rows_written": self.rows_written,
            "rows_with_tool_calls": self.rows_with_tool_calls,
            "trained_turns": self.trained_turns,
            "context_turns": dict(sorted(self.context_turns.items())),
            "rows_cut": self.rows_cut,
            "skipped_no_trained_turn": self.skipped_no_trained_turn,
            "trained_turns_left_out": self.trained_turns_left_out,
            "skipped_unscored": self.skipped_unscored,
            "skipped_no_transcript": self.skipped_no_transcript,
            "per_call": per_call,
        }


def write_manifest(stats: RlExportStats, manifest: str | Path | None) -> None:
    if manifest is None:
        return
    path = Path(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats.as_dict(), indent=2, sort_keys=True) + "\n")


def context_summary(stats: RlExportStats, *, trained_label: str = "trained") -> str:
    """``18 assistant turn(s) trained, 12 kept as context (10 cut at the output cap, ...)``."""
    kept = sum(stats.context_turns.values())
    text = f"{stats.trained_turns} assistant turn(s) {trained_label}, {kept} kept as context"
    if kept:
        parts = [
            f"{count} {CONTEXT_REASONS.get(reason, reason)}"
            for reason, count in stats.context_turns.most_common()
        ]
        text += " (" + ", ".join(parts) + ")"
    if stats.rows_cut or stats.skipped_no_trained_turn:
        text += (
            f"; {stats.rows_cut} row(s) cut short and {stats.skipped_no_trained_turn} "
            "conversation(s) left out at a turn BenchFlow does not train "
            f"({stats.trained_turns_left_out} trained call(s) not written: "
            "--row-mode exchange writes them all)"
        )
    return text


def clean(value: Any, *, redact: bool) -> Any:
    """What every trainer export writes: finite numbers, secrets masked unless --no-redact."""
    value = scrub_non_finite(value)
    return redact_trajectory_obj(value) if redact else value


def wire_message(message: dict[str, Any]) -> dict[str, Any]:
    """A row message in OpenAI wire format: tool-call arguments as a JSON string."""
    out = dict(message)
    calls = out.get("tool_calls")
    if isinstance(calls, list):
        wire = []
        for call in calls:
            call = dict(call)
            function = dict(call.get("function") or {})
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                function["arguments"] = dumps_finite(arguments, default=str)
            call["function"] = function
            wire.append(call)
        out["tool_calls"] = wire
    return out


def require_job_directory(jobs_dir: str | Path, format_name: str) -> None:
    if Path(jobs_dir).is_file():
        raise ValueError(
            f"--format {format_name} reads a rollout or jobs directory (each trial's "
            f"trajectory/llm_trajectory.jsonl), not the file {Path(jobs_dir).name}"
        )


def build_conversations(
    jobs_dir: str | Path,
    *,
    format_name: str,
    row_mode: TrlSftRowMode,
    min_reward: float | None = None,
    canonical_selection: str | Path | None = None,
    redact: bool = True,
    subagent_rows: bool = False,
    reward_vector: bool = False,
    group_advantage: GroupAdvantage | None = None,
    group_by: str | Sequence[str] | None = None,
) -> tuple[list[Conversation], RlExportStats]:
    """Each rollout's trained calls as conversations (``rollout``) or one per call (``exchange``).

    ``exchange`` conversations train their last turn only, as TRL's per-call rows do;
    ``rollout`` conversations train every call of the conversation once.
    """
    require_job_directory(jobs_dir, format_name)
    if row_mode not in {"rollout", "exchange"}:
        raise ValueError("row_mode must be rollout or exchange")
    rows, per_call = convert_benchflow_rollouts_to_trl_sft_rows(
        jobs_dir,
        min_reward=min_reward,
        row_mode="exchange",
        canonical_selection=canonical_selection,
        redact=redact,
        subagent_rows=subagent_rows,
        reward_vector=reward_vector,
        group_advantage=group_advantage,
        group_by=group_by,
    )
    stats = RlExportStats(format=format_name, row_mode=row_mode, per_call=per_call)
    if row_mode == "rollout":
        conversations = chain_calls(rows)
    else:
        conversations = [
            Conversation(
                rollout=row.get("source_rollout_dir"),
                messages=row["prompt"] + row["completion"],
                tools=row["tools"],
                trained=[len(row["prompt"])],
                calls=[row],
            )
            for row in rows
        ]
    label_context_turns(conversations, rows)
    return conversations, stats


def chain_calls(rows: Sequence[dict[str, Any]]) -> list[Conversation]:
    """Join each per-call row to the longest conversation of its rollout that its prompt extends.

    A row whose prompt extends none (the first call, a compacted context, a subagent, other
    tools) starts a conversation. Rows are in call order within each conversation.
    """
    by_rollout: dict[str | None, list[Conversation]] = {}
    ordered: list[Conversation] = []
    for row in rows:
        rollout = row.get("source_rollout_dir")
        prompt = row["prompt"]
        best: Conversation | None = None
        for conversation in by_rollout.get(rollout, []):
            size = len(conversation.messages)
            if (
                conversation.tools == row["tools"]
                and size <= len(prompt)
                and prompt[:size] == conversation.messages
                and (best is None or size > len(best.messages))
            ):
                best = conversation
        if best is None:
            best = Conversation(
                rollout=rollout, messages=[], tools=row["tools"], trained=[], calls=[]
            )
            by_rollout.setdefault(rollout, []).append(best)
            ordered.append(best)
        best.messages = prompt + row["completion"]
        best.trained.append(len(prompt))
        best.calls.append(row)
    return ordered


def _hit_output_cap(body: dict[str, Any]) -> bool:
    if body.get("stop_reason") == "max_tokens":
        return True
    details = body.get("incomplete_details")
    if isinstance(details, dict) and details.get("reason") == "max_output_tokens":
        return True
    choices = body.get("choices")
    return isinstance(choices, list) and any(
        isinstance(choice, dict) and choice.get("finish_reason") == "length"
        for choice in choices
    )


def call_reasons(exchanges: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Tool-call id -> why that call is not trained, for every call a response made."""
    reasons: dict[str, str] = {}
    for exchange in exchanges:
        response = exchange.get("response")
        body = response.get("body") if isinstance(response, dict) else None
        if not isinstance(body, dict):
            continue
        calls = _response_tool_calls(body)
        if not calls:
            continue
        if not _response_is_training_success(response):
            reason = "output_cap" if _hit_output_cap(body) else "unfinished"
        elif _calls_tool_with_unparsed_input(
            [_trl_message({"role": "assistant", "tool_calls": calls})]
        ):
            reason = "unparsed_tool_input"
        else:
            reason = "other"
        for call in calls:
            if call.get("id"):
                reasons.setdefault(call["id"], reason)
    return reasons


def _call_ids(message: dict[str, Any]) -> list[str]:
    return [
        call["id"]
        for call in message.get("tool_calls") or []
        if isinstance(call, dict) and isinstance(call.get("id"), str) and call["id"]
    ]


def context_reason(
    message: dict[str, Any], reasons: dict[str, str], trained_ids: set[str]
) -> str:
    ids = _call_ids(message)
    if ids and all(call_id in trained_ids for call_id in ids):
        return "trained_in_another_row"
    return next((reasons[call_id] for call_id in ids if call_id in reasons), "other")


def label_context_turns(
    conversations: Sequence[Conversation], rows: Sequence[dict[str, Any]]
) -> None:
    """Give every assistant turn that is not trained its reason, from the rollout's calls."""
    trained_ids = {
        call_id
        for row in rows
        for message in row["completion"]
        for call_id in _call_ids(message)
    }
    reasons_by_rollout: dict[str | None, dict[str, str]] = {}
    for conversation in conversations:
        rollout = conversation.rollout
        if rollout not in reasons_by_rollout:
            path = Path(rollout or "") / "trajectory" / "llm_trajectory.jsonl"
            exchanges = load_llm_trajectory_jsonl(path, strict=True) if rollout else []
            reasons_by_rollout[rollout] = call_reasons(exchanges)
        reasons = reasons_by_rollout[rollout]
        trained = set(conversation.trained)
        conversation.context = {
            index: context_reason(message, reasons, trained_ids)
            for index, message in enumerate(conversation.messages)
            if message.get("role") == "assistant" and index not in trained
        }
