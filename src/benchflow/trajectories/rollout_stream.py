"""Stream finished rollouts of a running job to a trainer.

``stream_rollouts(job_dir)`` (and ``astream_rollouts``) yields one
:class:`StreamedRollout` per rollout as soon as its ``result.json`` exists,
in the order rollouts finish, until the job ends. ``bench train stream
<job_dir> --format jsonl`` prints the same records, one JSON line each. A
trainer can read them while the job is still running (online RL): each record
carries the rollout's reward and group id and, when the gateway captured
them, the token ids and logprobs of every model call
(``BENCHFLOW_CAPTURE_TOKEN_LOGPROBS=1``, see ``docs/reference/token-capture.md``).

The record is ``benchflow.rollout-stream.v1``; its JSON Schema is
:data:`SCHEMA`, committed as
``docs/reference/schemas/benchflow-rollout-stream.v1.schema.json``
(``python -m benchflow.trajectories.rollout_stream docs/reference/schemas``).
See ``docs/reference/rollout-stream.md``.

A rollout counts as finished when its ``result.json`` exists: BenchFlow
writes it atomically and last, after the trajectory and the gateway log. The
job's state comes from its folder: ``.evaluation.lock`` is held while it runs
and ``summary.json`` is written when it ends.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import time
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from benchflow._utils.result_paths import iter_task_result_paths
from benchflow._utils.scoring import classify_score_outcome, extract_reward
from benchflow.trajectories.token_capture import (
    TOKEN_CAPTURE_METADATA_KEY,
    TOKEN_CAPTURE_SCHEMA_VERSION,
    summarize_token_capture,
)
from benchflow.trajectories.training_signal import (
    _advantages,
    _group_id,
    _key_value,
    parse_group_by,
)

ROLLOUT_STREAM_SCHEMA_VERSION = "benchflow.rollout-stream.v1"
JOB_LOCK = ".evaluation.lock"
EVALUATION_RECORD = "evaluation.json"
JobState = Literal["waiting", "running", "finished", "stopped"]


class StreamEnded(Exception):
    """Base for the ways a followed stream ends without the job finishing."""


class StreamTimeout(StreamEnded):
    """``timeout`` seconds passed before the job finished."""


class JobProcessGone(StreamEnded):
    """The job's lock names a process on this host that no longer exists."""


class JobNotFound(StreamEnded):
    """The job folder does not exist (and the stream does not wait for it)."""


# --- job folder state -------------------------------------------------------


def resolve_job_dir(path: str | Path) -> Path:
    """The job folder for ``path``: itself, or its only child job folder.

    ``bench eval run --jobs-dir DIR`` writes the job to ``DIR/<job name>``;
    passing ``DIR`` works when it holds exactly one job.
    """
    root = Path(path)
    if not root.is_dir() or _is_job_dir(root):
        return root
    children = [c for c in root.iterdir() if c.is_dir() and _is_job_dir(c)]
    return children[0] if len(children) == 1 else root


def _is_job_dir(path: Path) -> bool:
    return (path / JOB_LOCK).exists() or (path / EVALUATION_RECORD).exists()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OverflowError):
        return True
    return True


def job_state(job_dir: str | Path) -> tuple[JobState, str]:
    """``(state, detail)`` of a job folder.

    ``running``: the job lock is held by a live process (or by another host,
    which cannot be checked). ``finished``: no lock and ``summary.json``
    exists. ``stopped``: the lock names a process on this host that is gone.
    ``waiting``: no lock and no summary yet (not started, or a run that does
    not take the lock).
    """
    root = Path(job_dir)
    lock = root / JOB_LOCK
    if lock.exists():
        try:
            holder = json.loads(lock.read_text())
        except (OSError, ValueError):
            holder = {}
        pid, host = holder.get("pid"), holder.get("host")
        if (
            host == socket.gethostname()
            and isinstance(pid, int)
            and not _pid_alive(pid)
        ):
            return "stopped", f"the job's process {pid} is gone (lock {lock})"
        return "running", f"job lock held by process {pid} on {host}"
    if (root / "summary.json").exists():
        return "finished", "summary.json written"
    if not root.exists():
        return "waiting", f"{root} does not exist yet"
    return "waiting", "no job lock and no summary.json yet"


# --- one record -------------------------------------------------------------


@dataclass(frozen=True)
class StreamedRollout:
    """One finished rollout, as a trainer reads it (``benchflow.rollout-stream.v1``)."""

    job: str
    rollout: str
    rollout_path: str
    task: str | None
    agent: str | None
    model: str | None
    group_id: str
    group: dict[str, Any]
    reward: float | None
    outcome: str
    error: str | None
    finished_at: str | None
    token_capture: dict[str, Any]
    calls: list[dict[str, Any]]
    sequences: list[dict[str, Any]]
    advantage: float | None = None
    group_complete: bool | None = None
    rollout_dir: Path = field(default=Path(), compare=False, repr=False)

    @property
    def scored(self) -> bool:
        return self.reward is not None

    @property
    def training_grade(self) -> bool:
        return bool(self.token_capture.get("training_grade"))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ROLLOUT_STREAM_SCHEMA_VERSION,
            "job": self.job,
            "rollout": self.rollout,
            "rollout_path": self.rollout_path,
            "task": self.task,
            "agent": self.agent,
            "model": self.model,
            "group_id": self.group_id,
            "group": dict(self.group),
            "reward": self.reward,
            "scored": self.scored,
            "outcome": self.outcome,
            "error": self.error,
            "finished_at": self.finished_at,
            "token_capture": dict(self.token_capture),
            "calls": list(self.calls),
            "sequences": list(self.sequences),
            "advantage": self.advantage,
            "group_complete": self.group_complete,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), separators=(",", ":"))


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _exchanges(rollout_dir: Path) -> list[dict[str, Any]] | None:
    path = rollout_dir / "trajectory" / "llm_trajectory.jsonl"
    if not path.is_file():
        return None
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _capture(exchange: dict[str, Any]) -> dict[str, Any] | None:
    metadata = exchange.get("metadata")
    capture = (
        metadata.get(TOKEN_CAPTURE_METADATA_KEY) if isinstance(metadata, dict) else None
    )
    if (
        isinstance(capture, dict)
        and capture.get("schema_version") == TOKEN_CAPTURE_SCHEMA_VERSION
    ):
        return capture
    return None


def _calls(exchanges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls = []
    for index, exchange in enumerate(exchanges):
        capture = _capture(exchange)
        if capture is None:
            continue
        calls.append(
            {
                "index": index,
                "provider": capture.get("provider"),
                "prompt_token_ids": capture.get("prompt_token_ids"),
                "completions": [
                    {
                        "index": c.get("index", 0),
                        "token_ids": c.get("token_ids"),
                        "logprobs": c.get("logprobs"),
                    }
                    for c in capture.get("completions") or []
                    if isinstance(c, dict)
                ],
                "unavailable": capture.get("unavailable") or {},
            }
        )
    return calls


def merged_sequence(calls: list[dict[str, Any]]) -> dict[str, Any] | None:
    """One token sequence for a token-in/token-out rollout, or None.

    ``prompt_ids`` is the first call's prompt. ``completion_ids`` is
    everything after it: each call's sampled tokens (mask 1, their logprobs)
    and the tokens the next prompt adds after them, such as tool results
    (mask 0, logprob 0.0). None when a call lacks ids or logprobs, or a
    prompt does not extend the previous prompt and sampled tokens.
    """
    if not calls:
        return None
    first = calls[0]["prompt_token_ids"]
    if not isinstance(first, list):
        return None
    prefix = list(first)
    completion_ids: list[int] = []
    mask: list[int] = []
    logprobs: list[float] = []
    for position, call in enumerate(calls):
        prompt = call["prompt_token_ids"]
        completions = call["completions"]
        if not isinstance(prompt, list) or not completions:
            return None
        sampled, sampled_logprobs = (
            completions[0].get("token_ids"),
            completions[0].get("logprobs"),
        )
        if (
            not isinstance(sampled, list)
            or not isinstance(sampled_logprobs, list)
            or len(sampled) != len(sampled_logprobs)
        ):
            return None
        if prompt[: len(prefix)] != prefix:
            return None
        if position:
            added = prompt[len(prefix) :]
            completion_ids += added
            mask += [0] * len(added)
            logprobs += [0.0] * len(added)
        completion_ids += sampled
        mask += [1] * len(sampled)
        logprobs += [float(v) for v in sampled_logprobs]
        prefix = prompt + sampled
    return {
        "prompt_ids": list(first),
        "completion_ids": completion_ids,
        "completion_mask": mask,
        "completion_logprobs": logprobs,
    }


def _sequences(
    calls: list[dict[str, Any]], summary: dict[str, Any]
) -> list[dict[str, Any]]:
    """One merged sequence per conversation (see ``conversation_threads``)."""
    by_index = {c["index"]: c for c in calls}
    out = []
    for thread in summary.get("threads") or []:
        members = [by_index[i] for i in thread["calls"] if i in by_index]
        merged = merged_sequence(members)
        if merged is not None:
            out.append(
                {
                    "thread": thread["thread"],
                    "kind": thread["kind"],
                    "calls": list(thread["calls"]),
                    **merged,
                }
            )
    return out


def _capture_path(calls: list[dict[str, Any]]) -> str | None:
    providers = sorted({str(c["provider"]) for c in calls if c.get("provider")})
    return ",".join(providers) if providers else None


def read_rollout(
    rollout_dir: str | Path,
    *,
    job_dir: str | Path | None = None,
    group_by: str | Sequence[str] | None = None,
    result: dict[str, Any] | None = None,
) -> StreamedRollout | None:
    """The stream record for one finished rollout; None without a readable result."""
    root = Path(rollout_dir)
    if result is None:
        result = _read_json(root / "result.json")
    if result is None:
        return None
    job_root = Path(job_dir) if job_dir is not None else root.parent
    keys = parse_group_by(group_by)
    key = {k: _key_value(result, root, k) for k in keys}
    exchanges = _exchanges(root)
    if exchanges is None:
        usage = result.get("usage_tracking")
        kind = usage.get("endpoint_kind") if isinstance(usage, dict) else None
        summary: dict[str, Any] = {
            "status": "no_gateway_capture",
            "reason": (
                "the agent's model calls did not go through BenchFlow's gateway "
                f"(usage_tracking.endpoint_kind={kind!r})"
                if kind
                else "no llm_trajectory.jsonl in this rollout"
            ),
            "training_grade": False,
        }
        calls: list[dict[str, Any]] = []
    else:
        summary = summarize_token_capture(exchanges)
        calls = _calls(exchanges)
        summary["status"] = "captured" if calls else "capture_off"
    summary["path"] = _capture_path(calls)
    sequences = _sequences(calls, summary) if summary.get("training_grade") else []
    if summary.get("training_grade") and len(sequences) != len(
        summary.get("threads") or []
    ):
        # A conversation that cannot be merged into one token stream is never
        # dropped silently: the rollout is not training-grade, and says why.
        summary["training_grade"] = False
        summary["reason"] = "a conversation's calls could not be merged into one sequence"
        sequences = []
    try:
        rollout_path = str(root.relative_to(job_root))
    except ValueError:
        rollout_path = root.name
    error = result.get("error")
    finished = result.get("finished_at")
    return StreamedRollout(
        job=job_root.name,
        rollout=root.name,
        rollout_path=rollout_path,
        task=key.get("task") if "task" in key else _key_value(result, root, "task"),
        agent=result.get("agent"),
        model=result.get("model"),
        group_id=_group_id(key),
        group=key,
        reward=extract_reward(result),
        outcome=classify_score_outcome(result),
        error=None if error is None else str(error),
        finished_at=None if finished is None else str(finished),
        token_capture=summary,
        calls=calls,
        sequences=sequences,
        rollout_dir=root,
    )


# --- the stream -------------------------------------------------------------


def _with_advantages(
    members: list[StreamedRollout], complete: bool
) -> list[StreamedRollout]:
    import dataclasses

    values = _advantages([m.reward for m in members], "grpo")
    return [
        dataclasses.replace(m, advantage=v, group_complete=complete)
        for m, v in zip(members, values, strict=True)
    ]


class _Grouper:
    def __init__(self, size: int | None) -> None:
        self.size = size
        self.pending: dict[str, list[StreamedRollout]] = {}

    def add(self, record: StreamedRollout) -> list[StreamedRollout]:
        if self.size is None:
            return [record]
        members = self.pending.setdefault(record.group_id, [])
        members.append(record)
        if len(members) < self.size:
            return []
        del self.pending[record.group_id]
        return _with_advantages(members, True)

    def flush(self) -> list[StreamedRollout]:
        out: list[StreamedRollout] = []
        for members in self.pending.values():
            out += [_replace(m, advantage=None, group_complete=False) for m in members]
        self.pending.clear()
        return out


def _replace(record: StreamedRollout, **changes: Any) -> StreamedRollout:
    import dataclasses

    return dataclasses.replace(record, **changes)


class _Scanner:
    def __init__(
        self,
        job_dir: Path,
        group_by: str | Sequence[str] | None,
        warn: Callable[[str], None],
    ) -> None:
        self.job_dir = job_dir
        self.group_by = parse_group_by(group_by)
        self.warn = warn
        self.seen: set[Path] = set()
        self.bad: set[Path] = set()

    def scan(self) -> list[StreamedRollout]:
        if not self.job_dir.is_dir():
            return []
        fresh: list[tuple[float, StreamedRollout]] = []
        for path in iter_task_result_paths(self.job_dir):
            root = path.parent
            if root in self.seen:
                continue
            result = _read_json(path)
            if result is None:
                if root not in self.bad:
                    self.bad.add(root)
                    self.warn(f"skipping {path}: not a readable JSON object")
                continue
            record = read_rollout(
                root, job_dir=self.job_dir, group_by=self.group_by, result=result
            )
            if record is None:
                continue
            self.seen.add(root)
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0
            fresh.append((mtime, record))
        fresh.sort(key=lambda pair: (pair[0], pair[1].rollout_path))
        return [record for _, record in fresh]


def _stderr(message: str) -> None:
    print(f"bench train stream: {message}", file=sys.stderr, flush=True)


class _Stream:
    """One followed job: each :meth:`poll` scans once and says whether to stop."""

    def __init__(
        self,
        job_dir: str | Path,
        *,
        follow: bool,
        timeout: float | None,
        group_by: str | Sequence[str] | None,
        group_size: int | None,
        on_warning: Callable[[str], None] | None,
    ) -> None:
        if group_size is not None and group_size < 1:
            raise ValueError("group_size must be at least 1")
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must not be negative")
        self.root = Path(job_dir)
        if not follow and not self.root.exists():
            raise JobNotFound(f"no such job folder: {self.root}")
        self.follow = follow
        self.timeout = timeout
        self.deadline = None if timeout is None else time.monotonic() + timeout
        self.grouper = _Grouper(group_size)
        self.scanner = _Scanner(
            resolve_job_dir(self.root), group_by, on_warning or _stderr
        )

    def poll(self) -> tuple[list[StreamedRollout], bool, StreamEnded | None]:
        """``(records, stop, error)``: yield the records, then stop or raise."""
        if self.scanner.job_dir == self.root:
            # The job folder may appear under a --jobs-dir after the stream starts.
            self.scanner.job_dir = resolve_job_dir(self.root)
        state, detail = job_state(self.scanner.job_dir)
        records: list[StreamedRollout] = []
        for record in self.scanner.scan():
            records += self.grouper.add(record)
        if not self.follow or state == "finished":
            return records + self.grouper.flush(), True, None
        if state == "stopped":
            return records + self.grouper.flush(), True, JobProcessGone(detail)
        if self.deadline is not None and time.monotonic() >= self.deadline:
            error = StreamTimeout(
                f"job not finished after {self.timeout:g}s ({detail}); "
                f"{len(self.scanner.seen)} rollouts streamed"
            )
            return records + self.grouper.flush(), True, error
        return records, False, None


def stream_rollouts(
    job_dir: str | Path,
    *,
    follow: bool = True,
    poll_interval: float = 1.0,
    timeout: float | None = None,
    group_by: str | Sequence[str] | None = None,
    group_size: int | None = None,
    on_warning: Callable[[str], None] | None = None,
) -> Iterator[StreamedRollout]:
    """Yield each finished rollout of a job as it finishes.

    With ``follow`` (default) the folder is polled every ``poll_interval``
    seconds until the job finishes (``summary.json`` written, lock
    released); rollouts already finished are yielded first. Without it, one
    scan. ``group_size=N`` holds records until N rollouts of a group
    (``group_by``, default task, agent, model) have finished, then yields
    them with a GRPO advantage; groups still short when the stream ends are
    yielded with ``advantage=None`` and ``group_complete=False``.

    Raises :class:`JobNotFound` without ``follow`` when the folder does not
    exist, :class:`StreamTimeout` after ``timeout`` seconds, and
    :class:`JobProcessGone` when the job's process died without finishing,
    each after yielding every rollout that finished.
    """
    stream = _Stream(
        job_dir,
        follow=follow,
        timeout=timeout,
        group_by=group_by,
        group_size=group_size,
        on_warning=on_warning,
    )
    while True:
        records, stop, error = stream.poll()
        yield from records
        if error is not None:
            raise error
        if stop:
            return
        time.sleep(poll_interval)


async def astream_rollouts(
    job_dir: str | Path,
    *,
    follow: bool = True,
    poll_interval: float = 1.0,
    timeout: float | None = None,
    group_by: str | Sequence[str] | None = None,
    group_size: int | None = None,
    on_warning: Callable[[str], None] | None = None,
) -> AsyncIterator[StreamedRollout]:
    """Async form of :func:`stream_rollouts`; folder scans run off the event loop.

    Run a job and train on it in one process::

        evaluation = bf.Evaluation(tasks_dir, jobs_dir, config=config)
        job = asyncio.create_task(evaluation.run())
        async for rollout in bf.astream_rollouts(evaluation.job_dir):
            trainer.add(rollout)
    """
    stream = _Stream(
        job_dir,
        follow=follow,
        timeout=timeout,
        group_by=group_by,
        group_size=group_size,
        on_warning=on_warning,
    )
    while True:
        records, stop, error = await asyncio.to_thread(stream.poll)
        for record in records:
            yield record
        if error is not None:
            raise error
        if stop:
            return
        await asyncio.sleep(poll_interval)


# --- JSON Schema (docs/reference/schemas) -----------------------------------

_INTS = {"type": "array", "items": {"type": "integer"}}
_OPT_INTS = {"type": ["array", "null"], "items": {"type": "integer"}}
_OPT_NUMS = {"type": ["array", "null"], "items": {"type": "number"}}
_OPT_STR = {"type": ["string", "null"]}

SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://benchflow.ai/schemas/benchflow-rollout-stream.v1.schema.json",
    "title": "One finished rollout from bench train stream / stream_rollouts",
    "description": "One JSON line per finished rollout. reward is null for an "
    "unscored rollout, never 0. calls holds the gateway's token capture "
    "(benchflow.token_capture.v1) per model call; sequences (one per "
    "conversation) are present only when every call is complete and each "
    "prompt extends the previous prompt and sampled tokens of its "
    "conversation (token-in/token-out).",
    "type": "object",
    "required": [
        "schema_version",
        "job",
        "rollout",
        "rollout_path",
        "task",
        "agent",
        "model",
        "group_id",
        "group",
        "reward",
        "scored",
        "outcome",
        "error",
        "finished_at",
        "token_capture",
        "calls",
        "sequences",
        "advantage",
        "group_complete",
    ],
    "additionalProperties": False,
    "properties": {
        "schema_version": {"const": ROLLOUT_STREAM_SCHEMA_VERSION},
        "job": {"type": "string", "description": "Job folder name."},
        "rollout": {"type": "string", "description": "Rollout folder name."},
        "rollout_path": {
            "type": "string",
            "description": "Rollout folder relative to the job folder.",
        },
        "task": _OPT_STR,
        "agent": _OPT_STR,
        "model": _OPT_STR,
        "group_id": {
            "type": "string",
            "description": "Stable id of the group, e.g. "
            "'task=hello|agent=claude-agent-acp|model=vllm/policy'.",
        },
        "group": {
            "type": "object",
            "description": "The grouping key (task, agent, model by default).",
        },
        "reward": {"type": ["number", "null"]},
        "scored": {"type": "boolean"},
        "outcome": {"enum": ["passed", "failed", "errored", "verifier_errored"]},
        "error": _OPT_STR,
        "finished_at": _OPT_STR,
        "token_capture": {
            "type": "object",
            "required": ["status", "training_grade", "path"],
            "properties": {
                "status": {"enum": ["captured", "capture_off", "no_gateway_capture"]},
                "training_grade": {"type": "boolean"},
                "path": {
                    **_OPT_STR,
                    "description": "Route provider(s) of the captured calls, "
                    "e.g. vllm or sglang.",
                },
                "reason": {"type": "string"},
                "calls": {"type": "integer"},
                "captured_calls": {"type": "integer"},
                "complete_calls": {"type": "integer"},
                "unavailable": {"type": "object"},
                "prefix": {"type": "object"},
            },
        },
        "calls": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "index",
                    "provider",
                    "prompt_token_ids",
                    "completions",
                    "unavailable",
                ],
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "Line of trajectory/llm_trajectory.jsonl.",
                    },
                    "provider": _OPT_STR,
                    "prompt_token_ids": _OPT_INTS,
                    "completions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["index", "token_ids", "logprobs"],
                            "properties": {
                                "index": {"type": "integer"},
                                "token_ids": _OPT_INTS,
                                "logprobs": _OPT_NUMS,
                            },
                        },
                    },
                    "unavailable": {"type": "object"},
                },
            },
        },
        "sequences": {
            "type": "array",
            "description": "One merged token stream per conversation of the "
            "rollout (agent loop, helper calls, subagents), only when the "
            "rollout is training-grade; empty otherwise.",
            "items": {
                "type": "object",
                "required": [
                    "thread",
                    "kind",
                    "calls",
                    "prompt_ids",
                    "completion_ids",
                    "completion_mask",
                    "completion_logprobs",
                ],
                "additionalProperties": False,
                "properties": {
                    "thread": {"type": "integer"},
                    "kind": {"enum": ["agent", "helper", "chat"]},
                    "calls": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "The calls merged, by index in calls[].index.",
                    },
                    "prompt_ids": _INTS,
                    "completion_ids": _INTS,
                    "completion_mask": {
                        "type": "array",
                        "items": {"enum": [0, 1]},
                        "description": "1 for sampled tokens, 0 for tokens "
                        "the environment added (tool results, next user "
                        "turn).",
                    },
                    "completion_logprobs": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Sampled-token logprob; 0.0 where "
                        "the mask is 0.",
                    },
                },
            },
        },
        "advantage": {
            "type": ["number", "null"],
            "description": "GRPO advantage over the group's scored rollouts "
            "(group_size only); null otherwise.",
        },
        "group_complete": {
            "type": ["boolean", "null"],
            "description": "group_size only: false when the stream ended "
            "before the group filled.",
        },
    },
}

SCHEMA_FILE = "benchflow-rollout-stream.v1.schema.json"


def write_schema(directory: str | Path) -> Path:
    path = Path(directory) / SCHEMA_FILE
    path.write_text(json.dumps(SCHEMA, indent=2) + "\n")
    return path


if __name__ == "__main__":
    print(write_schema(sys.argv[1] if len(sys.argv) > 1 else "."))
