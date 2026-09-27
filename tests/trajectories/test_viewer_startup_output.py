"""The viewer's startup lines reach a pipe while it is still serving.

``bench eval view`` prints the URL and then serves until stopped. With stdout
on a pipe (an agent running it in the background, as the trajectory upload
skill does, or ``| tee``), Python block-buffers stdout, so the URL never
appeared before the process exited.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest


def _rollout(base: Path) -> Path:
    rollout = base / "task__abcd1234"
    (rollout / "trajectory").mkdir(parents=True)
    (rollout / "trajectory" / "acp_trajectory.jsonl").write_text(
        json.dumps({"type": "agent_message", "text": "hello"}) + "\n"
    )
    (rollout / "result.json").write_text(
        json.dumps({"task_name": "task", "rewards": {"reward": 1.0}})
    )
    return rollout


def _first_line(path: Path, timeout: float = 20.0) -> str:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "benchflow.trajectories.viewer", str(path), "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        env=env,
        text=True,
    )
    lines: list[str] = []
    reader = threading.Thread(
        target=lambda: lines.append(proc.stdout.readline()), daemon=True
    )
    try:
        reader.start()
        reader.join(timeout)
        return lines[0] if lines else ""
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.mark.parametrize("mode", ["single", "browse"])
def test_viewer_url_reaches_a_pipe_while_serving(tmp_path, mode):
    rollout = _rollout(tmp_path / "jobs")
    target = rollout if mode == "single" else tmp_path / "jobs"
    line = _first_line(target)
    assert "http://localhost:" in line


@pytest.mark.parametrize("layout", ["trial", "parent"])
def test_physical_trials_get_a_pointer_instead_of_a_dead_end(tmp_path, capsys, layout):
    """`bench eval view <physical trial>` said only "No trajectories found":
    the viewer does not render trial records yet, and
    nothing pointed at the robotics commands that read them."""
    from benchflow.trajectories.viewer import serve

    trial = tmp_path / "physical-trials" / "20260101T000000Z-0123456789"
    trial.mkdir(parents=True)
    (trial / "manifest.json").write_text(
        json.dumps({"trial_id": trial.name, "kind": "physical_trial"})
    )
    (trial / "trial-record.json").write_text(
        json.dumps({"kind": "benchflow-embodied-trial"})
    )
    target = trial if layout == "trial" else tmp_path
    with pytest.raises(SystemExit) as exc:
        serve(str(target), 0)
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "No trajectories found" in out
    assert "physical robot trial" in out
    assert "python -m benchflow.robotics index" in out
    assert "python -m benchflow.robotics report" in out


def _interrupt(self):
    raise KeyboardInterrupt


def test_counts_of_one_run_read_in_the_singular(tmp_path, capsys, monkeypatch):
    """Regression test: the viewer printed "1 runs". Every count
    it prints before serving (startup line, cap note, --confirm refusal,
    job-directory page) now agrees with its number."""
    from http.server import HTTPServer

    from benchflow.trajectories.viewer import render_rollout, serve

    monkeypatch.setattr(HTTPServer, "serve_forever", _interrupt)
    jobs = tmp_path / "jobs"
    _rollout(jobs)
    serve(str(jobs), 0)
    assert f"Scanning: {jobs} (1 run)\n" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        serve(str(jobs), 0, confirm=True)
    assert "is a directory of 1 run\n" in capsys.readouterr().out

    second = jobs / "task__abcd5678"
    (second / "trajectory").mkdir(parents=True)
    (second / "trajectory" / "acp_trajectory.jsonl").write_text("")
    monkeypatch.setenv("BENCHFLOW_VIEWER_MAX_RUNS", "1")
    serve(str(jobs), 0)
    assert "(first 1 run (capped" in capsys.readouterr().out

    (second / "trajectory" / "acp_trajectory.jsonl").unlink()
    (second / "trajectory").rmdir()
    second.rmdir()
    assert "a job directory with 1 rollout." in render_rollout(jobs)
