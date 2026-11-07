"""Convert BenchFlow rollouts into OpenEnv episode records.

Each line of the output is one rollout with the fields of OpenEnv's ``EpisodeRecord``
(``openenv.core.harness.collect``): ``episode_id``, ``messages``, ``reward``, ``done``,
``tool_trace``, ``metrics``, ``verify_metrics``, ``artifacts``, ``task`` and ``extra``.
OpenEnv's ``RolloutSerializer(<folder>)`` reads ``<folder>/results.jsonl``.

- ``messages`` is the main agent's last model call, its request and its reply, in OpenAI format
  as OpenEnv's harness writes it: tool-call arguments as JSON strings, and each tool result
  with the tool's name.
- ``tool_trace`` has one entry per tool call: its name, its arguments as an object, and the
  result text.
- ``reward`` is the verifier's. OpenEnv never counts a missing reward as 0, so an unscored
  rollout is left out and counted.
- ``extra.train_message_indices`` lists the assistant turns BenchFlow trains, as in the other
  trainer formats; ``extra.context_turns`` gives the reason for each of the others (a call cut
  at the output cap, tool input that was not a JSON object). They are what happened, not
  targets.

OpenEnv's token-level ``TrainingTrace`` is not written: it needs the sampling policy's log
probability for every completion token, which a rollout recorded through a provider API does
not have, and a placeholder would train against probabilities nobody sampled.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from benchflow._utils.json_safe import dumps_finite
from benchflow.trajectories.export_prime_sft import (
    _iter_rollout_dirs,
    _iter_selected_rollout_dirs,
    _load_json,
    _result_training_skip_reason,
    _reward_from_result,
    is_control_run,
    load_llm_trajectory_jsonl,
    normalize_prime_sft_exchange,
)
from benchflow.trajectories.export_trl_sft import (
    TrlSftRowMode,
    _call_purpose,
    _trl_message,
    convert_benchflow_rollouts_to_trl_sft_rows,
)
from benchflow.trajectories.rl_conversations import (
    Conversation,
    RlExportStats,
    clean,
    label_context_turns,
    require_job_directory,
    wire_message,
    write_manifest,
)
from benchflow.trajectories.sft_subagents import (
    SubagentConversionCounts,
    split_rollout_exchanges,
)
from benchflow.trajectories.training_signal import GroupAdvantage

EPISODE_FIELDS = (
    "episode_id",
    "messages",
    "reward",
    "done",
    "tool_trace",
    "metrics",
    "verify_metrics",
    "artifacts",
    "task",
    "extra",
)


def refuse_episode_options(
    *,
    row_mode: str,
    subagent_rows: bool,
    reward_vector: bool,
    group_advantage: Any,
) -> None:
    for flag, used, why in (
        ("--row-mode exchange", row_mode != "rollout", "one episode per rollout"),
        ("--subagent-rows", subagent_rows, "the main agent's transcript"),
        ("--reward-vector", reward_vector, "every verifier score in verify_metrics"),
        (
            "--group-advantage",
            group_advantage is not None,
            "the reward, not advantages",
        ),
    ):
        if used:
            raise ValueError(f"--format openenv-episodes keeps {why}: drop {flag}")


def _transcript(
    rollout_dir: Path,
    result: dict[str, Any],
    exchanges: list[dict[str, Any]],
    *,
    redact: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int] | None:
    """The main agent's last model call that normalizes: messages, tools, call index."""
    agent = result.get("agent")
    primary = [
        (index, exchange)
        for index, exchange in enumerate(exchanges)
        if (exchange.get("response") or {}).get("status_code") == 200
        and _call_purpose(exchange, agent=agent) == "agent"
    ]
    split = split_rollout_exchanges(
        rollout_dir,
        exchanges,
        primary,
        counts=SubagentConversionCounts(),
        subagent_rows=False,
    )
    for index, exchange in reversed(split.parent):
        normalized, skip_reason = normalize_prime_sft_exchange(exchange, redact=redact)
        if skip_reason and skip_reason.startswith("unsupported_tool_protocol:"):
            raise ValueError(
                f"{rollout_dir / 'trajectory' / 'llm_trajectory.jsonl'}: "
                f"exchange {index}: {skip_reason}"
            )
        if normalized is not None:
            messages = [_trl_message(message) for message in normalized.messages]
            return messages, normalized.tool_defs, index
    return None


def episode_record(
    conversation: Conversation,
    *,
    rollout_dir: Path,
    result: dict[str, Any],
    reward: float,
    model_calls: int,
    last_call: int,
    redact: bool = True,
) -> dict[str, Any]:
    messages = clean(conversation.messages, redact=redact)
    names = {
        call["id"]: call["function"]["name"]
        for message in messages
        for call in message.get("tool_calls") or []
    }
    outputs = {
        message["tool_call_id"]: message.get("content")
        for message in messages
        if message.get("role") == "tool" and message.get("tool_call_id")
    }
    wire = []
    for message in messages:
        message = wire_message(message)
        if message.get("role") == "tool" and message.get("tool_call_id") in names:
            message["name"] = names[message["tool_call_id"]]
        wire.append(message)
    tool_trace = [
        {
            "tool_name": call["function"]["name"],
            "arguments": call["function"]["arguments"],
            "result": {
                "data": outputs.get(call["id"]),
                "done": False,
                "metadata": {"tool_call_id": call["id"]},
                "error": None,
            },
        }
        for message in messages
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
    ]
    agent_result = result.get("agent_result")
    metrics: dict[str, Any] = {
        "model_calls": model_calls,
        "assistant_turns": sum(m.get("role") == "assistant" for m in messages),
        "tool_calls": len(tool_trace),
    }
    if isinstance(agent_result, dict):
        metrics["token_usage"] = agent_result
    record = {
        "episode_id": f"{rollout_dir.parent.name}/{rollout_dir.name}",
        "messages": wire,
        "reward": float(reward),
        "done": True,
        "tool_trace": tool_trace,
        "metrics": metrics,
        "verify_metrics": {
            "rewards": result.get("rewards"),
            "scoring": result.get("scoring"),
        },
        "artifacts": {
            "llm_trajectory": str(rollout_dir / "trajectory" / "llm_trajectory.jsonl"),
            "result": str(rollout_dir / "result.json"),
        },
        "task": {
            "task_id": result.get("task_name") or rollout_dir.name,
            "task_digest": result.get("task_digest"),
        },
        "extra": {
            "agent": result.get("agent"),
            "model": result.get("model"),
            "tools": conversation.tools,
            "train_message_indices": conversation.trained,
            "context_turns": {
                str(index): reason
                for index, reason in sorted(conversation.context.items())
            },
            "exchange_index": last_call,
            "source": "benchflow-llm-trajectory",
        },
    }
    return clean(record, redact=redact)


def export_openenv_episodes_jsonl(
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
    """Write OpenEnv episode records, one per scored rollout; return the stats."""
    del group_by  # no group advantages in an episode
    require_job_directory(jobs_dir, "openenv-episodes")
    refuse_episode_options(
        row_mode=row_mode,
        subagent_rows=subagent_rows,
        reward_vector=reward_vector,
        group_advantage=group_advantage,
    )
    rows, per_call = convert_benchflow_rollouts_to_trl_sft_rows(
        jobs_dir,
        min_reward=min_reward,
        row_mode="exchange",
        canonical_selection=canonical_selection,
        redact=redact,
    )
    stats = RlExportStats(
        format="openenv-episodes", row_mode="rollout", per_call=per_call
    )
    rows_by_rollout: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_rollout[row["source_rollout_dir"]].append(row)
    rollout_dirs = (
        _iter_selected_rollout_dirs(canonical_selection)
        if canonical_selection is not None
        else _iter_rollout_dirs(jobs_dir)
    )
    conversations: list[Conversation] = []
    records: list[dict[str, Any]] = []
    for rollout_dir in rollout_dirs:
        # The per-call conversion counted every rollout left out here but the unscored.
        result = _load_json(rollout_dir / "result.json")
        if result is None or is_control_run(rollout_dir):
            continue
        if _result_training_skip_reason(result) is not None:
            continue
        reward = _reward_from_result(result)
        if min_reward is not None and (reward is None or reward < min_reward):
            continue
        if reward is None:
            stats.skipped_unscored += 1
            continue
        exchanges = load_llm_trajectory_jsonl(
            rollout_dir / "trajectory" / "llm_trajectory.jsonl", strict=True
        )
        transcript = _transcript(rollout_dir, result, exchanges, redact=redact)
        if transcript is None:
            stats.skipped_no_transcript += 1
            continue
        messages, tools, last_call = transcript
        calls = [
            row
            for row in rows_by_rollout.get(str(rollout_dir), [])
            if messages[: len(row["prompt"]) + 1] == row["prompt"] + row["completion"]
        ]
        conversation = Conversation(
            rollout=str(rollout_dir),
            messages=messages,
            tools=tools,
            trained=[len(row["prompt"]) for row in calls],
            calls=calls,
        )
        label_context_turns([conversation], rows)
        conversations.append(conversation)
        records.append(
            episode_record(
                conversation,
                rollout_dir=rollout_dir,
                result=result,
                reward=reward,
                model_calls=len(exchanges),
                last_call=last_call,
                redact=redact,
            )
        )
    if expected_rows is not None and len(records) != expected_rows:
        raise ValueError(f"row count {len(records)} != expected {expected_rows}")
    stats.count(conversations)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(dumps_finite(record, default=str) + "\n")
    write_manifest(stats, manifest)
    return stats
