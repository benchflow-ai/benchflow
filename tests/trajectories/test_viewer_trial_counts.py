"""The viewer's catalog counts the trials ``bf.load_job`` counts.

Guards P4-R1-B-1 (round PY8): the catalog's group pass rate counts every
run, and browse mode lists every folder with a trajectory as a run, so a
superseded errored attempt of a retried task, or a run with no result.json
(still going, or killed), counted as a failed run. In an Evaluation job
whose task errored once and then passed, ``bench eval metrics`` said 100%
(1 of 1 trials) while the viewer said 50% of 2 runs. Each catalog row now
says when the job does not count it, and why.
"""

import json
import os
import re
import socket
import threading
import time
import urllib.request
from pathlib import Path

import benchflow as bf
from benchflow.trajectories.viewer.server import _serve_browse


def _run(
    job: Path,
    name: str,
    *,
    task: str,
    reward: float | None,
    mtime: float | None = None,
    result: bool = True,
    **extra,
) -> Path:
    d = job / name
    (d / "trajectory").mkdir(parents=True)
    (d / "trajectory" / "acp_trajectory.jsonl").write_text(
        json.dumps({"type": "agent_message", "text": "hi"})
    )
    if result:
        data = {
            "task_name": task,
            "agent_name": "claude-agent-acp",
            "model": "sonnet-x",
            "rewards": {"reward": reward} if reward is not None else None,
            **extra,
        }
        (d / "result.json").write_text(json.dumps(data))
        if mtime is not None:
            os.utime(d / "result.json", (mtime, mtime))
    else:
        (d / "config.json").write_text(json.dumps({"agent": "claude-agent-acp"}))
    return d


def _rows(base: Path) -> dict[str, dict]:
    """The catalog rows the browse server embeds in its page, by run id."""
    with socket.socket() as probe:
        probe.bind(("localhost", 0))
        port = probe.getsockname()[1]
    threading.Thread(target=_serve_browse, args=(base, port, 1), daemon=True).start()
    for _ in range(50):
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/", timeout=1) as r:
                page = r.read().decode("utf-8")
            break
        except OSError:
            time.sleep(0.1)
    else:  # pragma: no cover
        raise AssertionError("viewer server did not come up")
    found = re.search(
        r"<script id=\"bf-payload\" type=\"application/json\">(.*?)</script>",
        page,
        re.S,
    )
    assert found is not None
    boot = json.loads(found.group(1))
    return {row["id"]: row for row in boot["rollouts"]}


def test_catalog_counts_the_trials_load_job_counts(tmp_path: Path) -> None:
    job = tmp_path / "eval-job"
    now = time.time()
    _run(
        job,
        "t__00000001",
        task="t",
        reward=None,
        error="sandbox create failed",
        mtime=now - 100,
    )
    _run(job, "t__00000002", task="t", reward=1.0, mtime=now - 50)
    _run(job, "u__00000003", task="u", reward=None, result=False)
    _run(job, "v__00000004", task="v", reward=0.0, mtime=now - 10)
    (job / "evaluation.json").write_text("{}")

    rows = _rows(tmp_path)
    assert {rid: row.get("not_counted") for rid, row in rows.items()} == {
        "eval-job/t__00000001": "superseded",
        "eval-job/t__00000002": None,
        "eval-job/u__00000003": "unfinished",
        "eval-job/v__00000004": None,
    }
    counted = {rid for rid, row in rows.items() if row.get("not_counted") is None}
    loaded = bf.load_job(job)
    assert counted == {f"eval-job/{t.path.name}" for t in loaded.trials}
    assert [p.name for p in loaded.interrupted] == ["u__00000003"]


def test_every_run_counts_outside_an_evaluation_job(tmp_path: Path) -> None:
    """A folder without evaluation.json or summary.json holds independent
    samples (``bf.run_batch``): an errored run there is a trial."""
    job = tmp_path / "batch"
    now = time.time()
    _run(job, "t__00000001", task="t", reward=None, error="boom", mtime=now - 100)
    _run(job, "t__00000002", task="t", reward=1.0, mtime=now - 50)

    rows = _rows(tmp_path)
    assert [row.get("not_counted") for row in rows.values()] == [None, None]
    assert len(bf.load_job(job).trials) == 2


def test_a_reviewer_run_is_not_a_trial(tmp_path: Path) -> None:
    job = tmp_path / "job"
    _run(job, "t__00000001", task="t", reward=1.0)
    _run(job, "review__00000002", task="t", reward=None, purpose="reviewer")

    rows = _rows(tmp_path)
    assert rows["job/review__00000002"].get("not_counted") == "reviewer"
    assert rows["job/t__00000001"].get("not_counted") is None
    assert [t.path.name for t in bf.load_job(job).trials] == ["t__00000001"]


def test_verifier_output_that_is_not_utf8_keeps_the_trial(tmp_path: Path) -> None:
    """P4-R2-A-1 (round PY8): verifier/test-stdout.txt is a raw redirection
    of what the tests printed in the sandbox. A byte that is not UTF-8 there
    made ``load_trial`` raise, so ``bf.load_job`` dropped a finished, scored
    trial and the viewer called it "no result yet": the solve rate said 0% of
    1 trial beside Score 1/2."""
    job = tmp_path / "eval-job"
    passed = _run(job, "t__00000001", task="t", reward=1.0)
    (passed / "verifier").mkdir()
    (passed / "verifier" / "test-stdout.txt").write_bytes(
        b"PASSED \xff\xfe binary diff\n"
    )
    (passed / "verifier" / "test-stderr.txt").write_bytes(b"cut \xe2\x82\n")
    _run(job, "v__00000002", task="v", reward=0.0)
    (job / "evaluation.json").write_text("{}")

    rows = _rows(tmp_path)
    assert {rid: row.get("not_counted") for rid, row in rows.items()} == {
        "eval-job/t__00000001": None,
        "eval-job/v__00000002": None,
    }
    loaded = bf.load_job(job)
    assert sorted(t.path.name for t in loaded.trials) == ["t__00000001", "v__00000002"]
    trial = next(t for t in loaded.trials if t.path.name == "t__00000001")
    assert trial.verifier.stdout == "PASSED �� binary diff\n"
    assert trial.verifier.stderr is not None and trial.verifier.stderr.startswith(
        "cut "
    )
    rates = loaded.solve_rates()
    assert rates.trials == 2
    assert rates.get(1).pass_at_k == 0.5
