"""Convert BenchFlow rollouts into parquet for verl's multi-turn SFT trainer.

verl's ``MultiTurnSFTDataset`` renders each row's ``messages`` with the model's chat template and
``tools``, and trains assistant messages. Two things a plain parquet export gets wrong:

- Nested parquet values share one schema per column. Tool calls and tool schemas of different
  shapes (``Bash {command, description}`` next to ``Write {file_path, content}``) each gain the
  others' keys as nulls, and the template renders them, so the training text changes. The
  ``messages`` and ``tools`` columns are JSON strings instead, which keeps every row exact; tool
  arguments are objects inside them, as chat templates expect.
- verl trains every assistant message, while BenchFlow trains only finished calls whose tool
  input parsed (the calls TRL's per-call rows hold). ``train_message_indices`` lists them.

:mod:`benchflow.trajectories.verl_sft_dataset` is the verl dataset class that reads both; verl
loads it through ``data.custom_cls``. ``--row-mode rollout`` (default) writes one row per
conversation and trains each call once; ``exchange`` writes one row per call and trains its last
turn, like TRL.
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
    write_manifest,
)
from benchflow.trajectories.training_signal import GroupAdvantage

VERL_DATASET_CLASS = "BenchFlowSFTDataset"


def verl_dataset_path() -> Path:
    """The file verl's ``data.custom_cls.path`` takes (stdlib and verl imports only)."""
    return Path(__file__).with_name("verl_sft_dataset.py")


def _pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow
        import pyarrow.parquet
    except ImportError as exc:
        raise ValueError(
            "--format verl-sft writes parquet, which needs pyarrow: pip install pyarrow "
            "(the benchflow trl extra includes it)"
        ) from exc
    return pyarrow, pyarrow.parquet


def _schema(pa: Any) -> Any:
    return pa.schema(
        [
            pa.field("messages", pa.string(), nullable=False),
            pa.field("tools", pa.string(), nullable=False),
            pa.field("train_message_indices", pa.list_(pa.int64()), nullable=False),
            pa.field("reward", pa.float64()),
            pa.field("task_id", pa.string()),
            pa.field("agent", pa.string()),
            pa.field("model", pa.string()),
            pa.field("source_rollout_dir", pa.string()),
            pa.field("exchange_indices", pa.list_(pa.int64())),
            pa.field("metadata", pa.string()),
        ]
    )


def verl_record(conversation: Conversation, *, redact: bool = True) -> dict[str, Any]:
    meta = clean(conversation.metadata(), redact=redact)
    reward = meta.get("reward")
    return {
        "messages": dumps_finite(
            clean(conversation.messages, redact=redact), default=str
        ),
        "tools": dumps_finite(clean(conversation.tools, redact=redact), default=str),
        "train_message_indices": list(conversation.trained),
        "reward": float(reward) if isinstance(reward, (int, float)) else None,
        "task_id": meta.get("task_id"),
        "agent": meta.get("agent"),
        "model": meta.get("model"),
        "source_rollout_dir": meta.get("source_rollout_dir"),
        "exchange_indices": meta["exchange_indices"],
        "metadata": dumps_finite(meta, default=str),
    }


def export_verl_sft_parquet(
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
    """Write verl SFT parquet from a rollout or jobs directory; return the conversion stats."""
    pa, pq = _pyarrow()
    conversations, stats = build_conversations(
        jobs_dir,
        format_name="verl-sft",
        row_mode=row_mode,
        min_reward=min_reward,
        canonical_selection=canonical_selection,
        redact=redact,
        subagent_rows=subagent_rows,
        reward_vector=reward_vector,
        group_advantage=group_advantage,
        group_by=group_by,
    )
    records = [
        verl_record(conversation, redact=redact) for conversation in conversations
    ]
    if expected_rows is not None and len(records) != expected_rows:
        raise ValueError(f"row count {len(records)} != expected {expected_rows}")
    stats.count(conversations)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records, schema=_schema(pa)), str(out_path))
    write_manifest(stats, manifest)
    return stats
