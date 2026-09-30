"""Miles prompt data for a folder of BenchFlow tasks.

Each row is ``{"prompt": [...], "metadata": {"instance_id": ...}}``; launch Miles
with ``--input-key prompt --metadata-key metadata`` and without
``--apply-chat-template`` (the session server renders the messages). The
prompt is the one the RL cookbooks' evaluator sends: the task's prompt with
the shared harness message appended.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchflow.integrations.miles.episode import HARNESS_MESSAGE
from benchflow.integrations.trl.spec import BenchFlowSpec

# Manifest fields copied into each row's metadata (the RL task family writes them).
_MANIFEST_FIELDS = ("kind", "level", "seed", "tags")


def dataset_rows(
    tasks_dir: str | Path,
    *,
    reset_message: str = HARNESS_MESSAGE,
    split: str | None = None,
    include_tasks: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """One Miles row per task prompt under ``tasks_dir``."""

    tasks_dir = Path(tasks_dir)
    spec = BenchFlowSpec(tasks_dir=tasks_dir, include_tasks=include_tasks)
    manifest = _manifest(tasks_dir)
    rows: list[dict[str, Any]] = []
    for row in spec.train_dataset_rows:
        messages = [dict(message) for message in row["prompt"]]
        messages[-1]["content"] += reset_message
        task_id = row["benchflow_task_id"]
        info = manifest.get(task_id, {})
        metadata: dict[str, Any] = {"instance_id": task_id}
        metadata.update({key: info[key] for key in _MANIFEST_FIELDS if key in info})
        if split:
            metadata["split"] = split
        rows.append({"prompt": messages, "metadata": metadata})
    return rows


def write_dataset(rows: list[dict[str, Any]], out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return out


def _manifest(tasks_dir: Path) -> dict[str, dict[str, Any]]:
    path = tasks_dir / "manifest.jsonl"
    if not path.is_file():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {row["task"]: row for row in rows if "task" in row}


__all__ = ["dataset_rows", "write_dataset"]
