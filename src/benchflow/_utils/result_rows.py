"""The reward of a results.jsonl or trainer/verifiers.jsonl row, one rule for
every reader: ``bf.load_job`` reading an export and ``bench train
convert`` choosing training rows agree on which rows were scored."""

from __future__ import annotations

from typing import Any


def row_reward(row: dict[str, Any]) -> float | None:
    """The row's reward; None when its trial was not scored.

    A row from this release writes an unscored trial's ``reward`` as null.
    Older rows wrote 0.0: a BenchFlow results.jsonl row before
    ``info.schema_version`` 2 (a scored one also has its reward in
    ``metrics``), and a verifiers.jsonl row beside ``info.reward_valid``
    false. Unscored is never 0.
    """
    reward = row.get("reward")
    if not isinstance(reward, int | float) or isinstance(reward, bool):
        return None
    raw_info, raw_metrics = row.get("info"), row.get("metrics")
    info: dict[str, Any] = raw_info if isinstance(raw_info, dict) else {}
    metrics: dict[str, Any] = raw_metrics if isinstance(raw_metrics, dict) else {}
    if reward == 0.0 and (
        (
            info.get("source") == "benchflow"
            and "schema_version" not in info
            and "reward" not in metrics
        )
        or info.get("reward_valid") is False
    ):
        return None
    return float(reward)
