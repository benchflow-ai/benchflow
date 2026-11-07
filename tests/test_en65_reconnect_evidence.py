"""Guard reconnect evidence retention at candidate 3387df47f/ac3bedaa1."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow._utils.reachable_write import pin_destination
from benchflow.integrity.trial import audit_trial, evidence_from_rollout
from benchflow.rollout import Rollout
from benchflow.sandbox.egress_denylist import EgressDenylist, stop_egress_denylist
from benchflow.task import TaskConfig


def _row(number=1):
    return {
        "ts": f"2026-10-26T01:00:0{number}+00:00",
        "action": "blocked",
        "method": "GET",
        "url": "https://blocked.example/paper",
        "rule": "host:blocked.example",
    }


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _archive(trial, epoch):
    return (
        trial / "network-transports" / (epoch * 32) / "trajectory/egress_denylist.jsonl"
    )


def _trial(tmp_path):
    trial = tmp_path / "trial"
    trial.mkdir()
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text(
        '[sandbox]\nnetwork_mode="denylist"\nblocked_hosts=["blocked.example"]\n'
    )
    (task / "instruction.md").write_text("Solve locally.")
    (trial / "result.json").write_text(
        json.dumps({"agent": "codex-acp", "rewards": {"reward": 1.0}})
    )
    rollout = Rollout.__new__(Rollout)
    rollout._rollout_dir = trial
    rollout._rollout_name = "trial"
    rollout._task = SimpleNamespace(
        name="task",
        config=TaskConfig.model_validate(
            {
                "sandbox": {
                    "network_mode": "denylist",
                    "blocked_hosts": ["blocked.example"],
                }
            }
        ),
    )
    rollout._config = SimpleNamespace(
        task_path=task,
        primary_agent="codex-acp",
        sandbox_user="agent",
        environment="daytona",
    )
    result = SimpleNamespace(rewards={"reward": 1.0}, trajectory=[])
    return trial, task, rollout, result


def _assert_readers(trial, task, rollout, result, expected):
    live = evidence_from_rollout(rollout, result, mode="audit")
    assert live.egress_denials == expected
    before = (trial / "result.json").read_bytes()
    audit_trial(trial, task_path=task)
    conformance = json.loads((trial / "integrity/conformance.json").read_text())
    assert conformance["host_lane"]["egress_denials"] == len(expected)
    assert (trial / "result.json").read_bytes() == before


@pytest.mark.asyncio
async def test_real_reconnect_denial_remains_in_live_and_offline_audit(tmp_path):
    """Guards ac3bedaa1: retained old transport evidence remains visible after reconnect."""
    trial, task, rollout, result = _trial(tmp_path)
    held_root = pin_destination(trial)
    original = json.dumps(_row()) + "\n"
    env = SimpleNamespace(log=original)

    async def download(_source, target):
        target.write_text(env.log)

    async def execute(command, **kwargs):
        if "rm -rf" in command:
            env.log = ""
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    env.download_file = download
    env.exec = execute
    rollout._env = env
    rollout._egress_denylist = EgressDenylist((), ("blocked.example",))
    rollout._active_egress_policy = rollout._egress_denylist
    rollout._usage_runtime = None
    rollout._planes = SimpleNamespace(
        stop_egress_denylist=stop_egress_denylist,
        start_egress_denylist=AsyncMock(),
    )
    await rollout._start_egress_denylist(rollout._egress_denylist)
    await rollout._teardown_egress()
    archives = list(trial.glob("network-transports/*/trajectory/egress_denylist.jsonl"))
    assert len(archives) == 1
    assert archives[0].read_text() == original
    assert (trial / "trajectory/egress_denylist.jsonl").read_text() == ""
    _assert_readers(trial, task, rollout, result, [_row()])
    assert archives[0].read_text() == original
    assert (held_root / "trajectory/egress_denylist.jsonl").read_text() == ""


def test_current_and_distinct_epochs_keep_repeated_requests(tmp_path):
    """Guards 3387df47f: same-second text is not a request identity across transports."""
    trial, task, rollout, result = _trial(tmp_path)
    _write(trial / "trajectory/egress_denylist.jsonl", [_row(2)])
    _write(_archive(trial, "a"), [_row(), _row()])
    _write(_archive(trial, "b"), [_row()])
    _assert_readers(trial, task, rollout, result, [_row(2), _row(), _row(), _row()])
    # Re-auditing enumerates each source again; it never appends to a combined log.
    _assert_readers(trial, task, rollout, result, [_row(2), _row(), _row(), _row()])


def test_hardlinked_current_and_archived_file_is_counted_once(tmp_path):
    """Guards 3387df47f: two paths to one log file cannot double-count its requests."""
    trial, task, rollout, result = _trial(tmp_path)
    current = _write(trial / "trajectory/egress_denylist.jsonl", [_row(), _row()])
    archive = _archive(trial, "a")
    archive.parent.mkdir(parents=True)
    os.link(current, archive)
    _assert_readers(trial, task, rollout, result, [_row(), _row()])


@pytest.mark.parametrize(
    "component", ["root", "epoch", "trajectory", "file", "current"]
)
def test_symlinked_archive_components_cannot_import_outside_evidence(
    tmp_path, component
):
    """Guards 3387df47f: no reconnect/canonical log component may escape the trial root."""
    trial, task, rollout, result = _trial(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    current = trial / "trajectory/egress_denylist.jsonl"
    _write(current, [_row()])
    archive = _archive(trial, "a")
    if component == "root":
        _write(outside / ("a" * 32) / "trajectory/egress_denylist.jsonl", [_row(9)])
        (trial / "network-transports").symlink_to(outside, target_is_directory=True)
    elif component == "epoch":
        _write(outside / "trajectory/egress_denylist.jsonl", [_row(9)])
        archive.parents[1].parent.mkdir(parents=True)
        archive.parents[1].symlink_to(outside, target_is_directory=True)
    elif component == "trajectory":
        _write(outside / "egress_denylist.jsonl", [_row(9)])
        archive.parent.parent.mkdir(parents=True)
        archive.parent.symlink_to(outside, target_is_directory=True)
    elif component == "file":
        target = _write(outside / "egress_denylist.jsonl", [_row(9)])
        archive.parent.mkdir(parents=True)
        archive.symlink_to(target)
    else:
        target = _write(outside / "egress_denylist.jsonl", [_row(9)])
        current.unlink()
        current.symlink_to(target)
    _assert_readers(
        trial, task, rollout, result, [] if component == "current" else [_row()]
    )


def test_only_recorded_epoch_shape_and_regular_jsonl_are_read(tmp_path):
    """Guards 3387df47f: unexpected folders, special files and malformed rows are ignored."""
    trial, task, rollout, result = _trial(tmp_path)
    current = _write(trial / "trajectory/egress_denylist.jsonl", [_row()])
    _write(
        trial / "network-transports/other/trajectory/egress_denylist.jsonl", [_row(9)]
    )
    archive = _write(_archive(trial, "a"), [_row(2)])
    archive.write_text(archive.read_text() + "invalid\n[]\nnull\n")
    fifo = _archive(trial, "b")
    fifo.parent.mkdir(parents=True)
    os.mkfifo(fifo)
    _assert_readers(trial, task, rollout, result, [_row(), _row(2)])
    assert current.read_text() == json.dumps(_row()) + "\n"
