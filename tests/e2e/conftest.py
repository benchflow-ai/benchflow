"""Fixtures for the end-to-end tier. See ``tests/e2e/README.md``."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest

from tests.e2e import harness as h


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    here = Path(__file__).resolve().parent
    for item in items:
        if here in Path(str(item.fspath)).resolve().parents:
            item.add_marker(pytest.mark.e2e)


@pytest.fixture(scope="session")
def sandbox() -> str:
    backend, reason = h.select_sandbox()
    if backend is None:
        pytest.skip(reason)
    return backend


@pytest.fixture(scope="session")
def e2e_out(sandbox: str) -> Path:
    configured = os.environ.get(h.OUT_ENV)
    out = Path(configured) if configured else Path(tempfile.mkdtemp(prefix="bf-e2e-"))
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture(scope="session")
def ledger(e2e_out: Path) -> h.Ledger:
    return h.Ledger(e2e_out / "ledger.jsonl")


@pytest.fixture(scope="session")
def tasks_root(e2e_out: Path) -> Path:
    root = e2e_out / "tasks"
    root.mkdir(exist_ok=True)
    return root


@pytest.fixture(scope="session")
def jobs_root(e2e_out: Path) -> Path:
    root = e2e_out / "jobs"
    root.mkdir(exist_ok=True)
    return root


@pytest.fixture(scope="session")
def batch_tasks(tasks_root: Path) -> Path:
    """Three tasks: one the oracle passes, one it fails, one with a broken verifier."""
    root = tasks_root / "batch"
    h.write_task(root, "e2e-pass")
    h.write_task(root, "e2e-fail", solve=h.WRONG_SOLVE)
    h.write_task(root, "e2e-verifier-error", test=h.BROKEN_TEST)
    return root


@pytest.fixture(scope="session")
def batch_job(
    sandbox: str, batch_tasks: Path, jobs_root: Path, ledger: h.Ledger
) -> Path:
    """One oracle batch job over ``batch_tasks``, frozen workspaces kept."""
    job = jobs_root / "batch-oracle"
    if not h.needs_run(job):
        return job
    run = h.bench(
        "eval", "run",
        "--tasks-dir", batch_tasks,
        "--agent", "oracle",
        "--sandbox", sandbox,
        "--jobs-dir", jobs_root,
        "--job-name", job.name,
        "--concurrency", "3",
        "--freeze-workspace",
        "--max-sandbox-seconds", str(h.cap_seconds()),
        "--summary-out", jobs_root / "batch-oracle.run-summary.json",
        "--fail-on", "verifier-error",
        "--quiet",
        log=jobs_root / "batch-oracle.log",
    )  # fmt: skip
    ledger.record(
        "eval run batch (oracle, 3 tasks, --freeze-workspace, --fail-on)",
        surface="CLI",
        seconds=run.seconds,
        job_dir=job,
        result="pass" if run.returncode == 1 else f"exit {run.returncode}",
    )
    # --fail-on verifier-error must trip on the broken verifier.
    h.assert_exit(run, 1)
    return job


@pytest.fixture(scope="session", autouse=True)
def _teardown_owner_sandboxes(request):
    """Delete every sandbox of this owner after the session and count what is left."""
    yield
    backend, _ = h.select_sandbox()
    if backend is None:
        return
    out = os.environ.get(h.OUT_ENV)
    log = Path(out) / "teardown.log" if out else None
    # --max-age 0 also deletes this owner's kept branch/checkpoint snapshots.
    run = h.bench("sandbox", "cleanup", "--all", "--max-age", "0", log=log, timeout=600)
    # Sandbox and snapshot deletion finish asynchronously on Daytona.
    deadline = time.monotonic() + 300
    left, snaps = h.owner_sandboxes(), h.owner_snapshots()
    while (left or snaps) and time.monotonic() < deadline:
        time.sleep(10)
        left, snaps = h.owner_sandboxes(), h.owner_snapshots()
    if log is not None:
        with log.open("a") as fh:
            fh.write(
                f"\n# sandboxes left for this owner: {len(left)}"
                f"\n# snapshots left for this owner: {len(snaps)}\n"
            )
    assert run.returncode == 0, run.tail()
    assert left == [], f"sandboxes left after cleanup: {left}"
    assert snaps == [], f"snapshots left after cleanup: {snaps}"
