"""Convert BenchFlow rollouts into JSONL rows for SkyRL's SFT trainer.

SkyRL's ``tokenize_chat_example`` reads OpenAI-format ``messages`` and a ``tools`` column (a list
or a JSON string). It trains the last assistant message (``train_on_what=last_assistant_message``,
its default) or every one (``all_assistant_messages``). SkyRL loads the file with Hugging Face
``datasets``, which gives nested JSON one schema per column, so tool-call arguments are JSON
strings, as in the OpenAI wire format (SkyRL parses them back to objects), and ``tools`` is a
JSON string.

SkyRL cannot keep an assistant turn as context, so each row is shaped for one setting:

- ``--row-mode exchange``: one row per call BenchFlow trains, ending on that call, for the
  default ``last_assistant_message``. Earlier turns are context, whatever they were.
- ``--row-mode rollout`` (default): one row per conversation, for ``all_assistant_messages``.
  The row ends before the first assistant turn BenchFlow does not train (a call cut at the
  output cap, one whose tool input was not a JSON object); a conversation that starts with one
  is left out.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from benchflow._utils.json_safe import dumps_finite
from benchflow.trajectories.export_trl_sft import TrlSftRowMode
from benchflow.trajectories.rl_conversations import (
    Conversation,
    RlExportStats,
    build_conversations,
    clean,
    wire_message,
    write_manifest,
)
from benchflow.trajectories.training_signal import GroupAdvantage


def skyrl_train_on_what(row_mode: TrlSftRowMode) -> str:
    """The ``train_on_what`` value the rows of ``row_mode`` are written for."""
    return (
        "all_assistant_messages" if row_mode == "rollout" else "last_assistant_message"
    )


def end_before_context(conversation: Conversation) -> Conversation | None:
    """The conversation up to its last trained turn before any turn that is context."""
    if not conversation.context:
        return conversation
    first = min(conversation.context)
    trained = [index for index in conversation.trained if index < first]
    if not trained:
        return None
    return Conversation(
        rollout=conversation.rollout,
        messages=conversation.messages[: trained[-1] + 1],
        tools=conversation.tools,
        trained=trained,
        calls=conversation.calls[: len(trained)],
    )


def skyrl_row(conversation: Conversation, *, redact: bool = True) -> dict[str, Any]:
    messages = clean(conversation.messages, redact=redact)
    return {
        "messages": [wire_message(message) for message in messages],
        "tools": dumps_finite(clean(conversation.tools, redact=redact), default=str),
        **clean(conversation.metadata(), redact=redact),
    }


def export_skyrl_sft_jsonl(
    jobs_dir: str | Path,
    out: str | Path,
    *,
    min_reward: float | None = None,
    row_mode: TrlSftRowMode = "rollout",
    expected_rows: int | None = None,
    manifest: str | Path | None = None,
    canonical_selection: str | Path | None = None,
    redact: bool = True,
    subagent_rows: bool = False,
    reward_vector: bool = False,
    group_advantage: GroupAdvantage | None = None,
    group_by: str | Sequence[str] | None = None,
) -> RlExportStats:
    """Write SkyRL SFT JSONL from a rollout or jobs directory; return the conversion stats."""
    conversations, stats = build_conversations(
        jobs_dir,
        format_name="skyrl-sft",
        row_mode=row_mode,
        min_reward=min_reward,
        canonical_selection=canonical_selection,
        redact=redact,
        subagent_rows=subagent_rows,
        reward_vector=reward_vector,
        group_advantage=group_advantage,
        group_by=group_by,
    )
    if row_mode == "rollout":
        kept = []
        for conversation in conversations:
            ended = end_before_context(conversation)
            stats.trained_turns_left_out += len(conversation.trained) - (
                len(ended.trained) if ended else 0
            )
            if ended is None:
                stats.skipped_no_trained_turn += 1
                continue
            if ended is not conversation:
                stats.rows_cut += 1
            kept.append(ended)
        conversations = kept
    rows = [skyrl_row(conversation, redact=redact) for conversation in conversations]
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"row count {len(rows)} != expected {expected_rows}")
    stats.count(conversations)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(dumps_finite(row, default=str) + "\n")
    write_manifest(stats, manifest)
    return stats
