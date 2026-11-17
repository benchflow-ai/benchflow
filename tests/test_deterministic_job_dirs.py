"""The deterministic tier's session jobs never share a copytree target
between xdist workers (MG's pass D on #234 failed with FileExistsError
when two workers built the same branch job into one det_root), and every job
test sits in its fixture's xdist group. No Docker needed."""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import pytest

from tests.integration.deterministic import harness as h
from tests.test_deterministic_integration import _job_dirs


def test_two_workers_get_their_own_tasks_and_jobs(tmp_path, monkeypatch):
    seen = set()
    for worker in ("gw0", "gw1", None):
        if worker is None:
            monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
        else:
            monkeypatch.setenv("PYTEST_XDIST_WORKER", worker)
        tasks, jobs = _job_dirs(tmp_path, "branch-parallel")
        assert tasks.parent == jobs.parent == tmp_path / "branch-parallel"
        seen.add(tasks)
        seen.add(jobs)
    assert len(seen) == 6


def _materialize_as(worker: str, root: str, queue) -> None:
    os.environ["PYTEST_XDIST_WORKER"] = worker
    tasks, _ = _job_dirs(Path(root), "branch-parallel")
    tasks.mkdir(parents=True, exist_ok=True)
    try:
        h.materialize_task(h.TaskVariant("branch-parallel", "hello-pass"), tasks)
        queue.put((worker, "ok"))
    except Exception as exc:
        queue.put((worker, f"{type(exc).__name__}: {exc}"))


def test_two_workers_materialize_one_job_at_once(tmp_path):
    """The collision itself: two processes materializing the same session
    job's task into one shared det_root. Each lands in its own folder."""
    queue = multiprocessing.get_context("spawn").Queue()
    procs = [
        multiprocessing.get_context("spawn").Process(
            target=_materialize_as, args=(worker, str(tmp_path), queue)
        )
        for worker in ("gw0", "gw1")
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=120)
    results = dict(queue.get(timeout=10) for _ in procs)
    assert results == {"gw0": "ok", "gw1": "ok"}, results


def test_the_old_shared_folder_collided(tmp_path):
    """The control: one shared tasks folder, as before, refuses the second
    copy (what MG's pass D met)."""
    shared = tmp_path / "branch-parallel" / "tasks"
    shared.mkdir(parents=True)
    h.materialize_task(h.TaskVariant("branch-parallel", "hello-pass"), shared)
    with pytest.raises(FileExistsError, match="branch-parallel"):
        h.materialize_task(h.TaskVariant("branch-parallel", "hello-pass"), shared)


def test_every_branch_job_test_names_its_fixture():
    """conftest groups a deterministic test by the session jobs it names; a
    test that fetches its job by name at run time gets no group and builds
    the job again on another worker."""
    import inspect

    import tests.test_deterministic_integration as module

    for name, func in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        source = inspect.getsource(func)
        assert "getfixturevalue(" not in source, name
