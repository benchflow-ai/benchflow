"""Fixtures: a fake BenchFlow venv whose `python` runs fake_bridge.py.

The package's own code runs for real (the Bridge process management, the session,
the in-process MCP server, the env, the task's hooks); only the BenchFlow side is
scripted. Nothing here needs the network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent

DEFAULT_SCENARIO = {
    "tasks": [
        {
            "name": "t1",
            "id": "fam/t1",
            "task_dir": "/tasks/t1",
            "prompt": "Solve t1.",
            "metadata": {"kind": "sql"},
            "verifier_timeout_sec": 60,
        }
    ],
    "start": {"ok": True, "workspace": "/workdir", "rollout_dir": "/jobs/r1"},
    "bash": [
        {
            "ok": True,
            "return_code": 0,
            "stdout": "out\n",
            "stderr": "",
            "timed_out": False,
        }
    ],
    "write": {"ok": True, "return_code": 0, "stdout": "", "stderr": ""},
    "verify": {
        "ok": True,
        "decision": {
            "reward": 1.0,
            "dropped": False,
            "reason": "scored",
            "detail": None,
            "flagged": False,
        },
        "result": {"reward": 1.0},
    },
}


class World:
    """The fake BenchFlow side: its python shim, its scenario, its event log."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.python = tmp / "fake-benchflow-python"
        self.python.write_text(
            f'#!/bin/sh\nshift\nexec {sys.executable} {HERE / "fake_bridge.py"} "$@"\n'
        )
        self.python.chmod(0o755)
        self.scenario_path = tmp / "scenario.json"
        self.log = tmp / "bridge-events.jsonl"
        self.set()

    def set(self, **changes) -> None:
        scenario = {**DEFAULT_SCENARIO, "log": str(self.log), **changes}
        self.scenario_path.write_text(json.dumps(scenario))

    def events(self, name: str | None = None) -> list[dict]:
        if not self.log.exists():
            return []
        rows = [
            json.loads(line)
            for line in self.log.read_text().splitlines()
            if line.strip()
        ]
        return [row for row in rows if name is None or row["event"] == name]

    def ops(self) -> list[str]:
        return [row["op"] for row in self.events("request")]

    def closed(self) -> list[str]:
        return [row["reason"] for row in self.events("sandbox_closed")]


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    monkeypatch.setenv("FAKE_BRIDGE_SCENARIO", str(tmp_path / "scenario.json"))
    return World(tmp_path)


def make_session(world: World, **overrides):
    from benchflow_taskset.session import BenchFlowSession, Bridge

    bridge = Bridge(
        str(world.python),
        idle_timeout_sec=60,
        log_path=world.tmp / "bridge.log",
        env_passthrough=("FAKE_BRIDGE_SCENARIO",),
    )
    settings = dict(
        task_dir="/tasks/t1",
        rollout_name="r1",
        environment="daytona",
        sandbox_user="agent",
        jobs_dir=str(world.tmp / "jobs"),
        bash_timeout_sec=5,
        max_output_chars=300,
        submit_path="/workdir/answer.txt",
        agent_budget_sec=None,
        sandbox_setup_timeout_sec=30,
        verify_timeout_sec=5,
    )
    settings.update(overrides)
    return BenchFlowSession(bridge, **settings)


def env_config(world: World, **task_overrides) -> dict:
    task = {
        "benchflow_python": str(world.python),
        "jobs_dir": str(world.tmp / "jobs"),
        "slots_dir": str(world.tmp / "slots"),
        "bridge_env": ["FAKE_BRIDGE_SCENARIO"],
        "verify_timeout_sec": 5,
        **task_overrides,
    }
    return {
        "taskset": {
            "id": "benchflow-taskset",
            "tasks_dir": str(world.tmp / "tasks"),
            "task": task,
        }
    }
