"""Item 6 on a real Docker daemon: a task whose image ENTRYPOINT exits
right after main's release (TasksMiner's strict network_lock.sh, exit 4
without NET_ADMIN) ends its trial as sandbox_setup, unscored and never
retried, naming the ENTRYPOINT, its exit status and its last output. Before,
such a trial ended as whatever step met the stopped main first (an install
failure, the protected runtime's sealing) and was retried."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_cloud_metadata_failclosed_docker import _docker_ready

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
    pytest.mark.skipif(not _docker_ready(), reason="no Docker daemon answers"),
]

ENTRY_SH = """#!/bin/sh
echo "network lock: cannot add the agent's rules (no NET_ADMIN)" >&2
exit 4
"""


def _task(root: Path) -> Path:
    task = root / "early-exit"
    env_dir = task / "environment"
    env_dir.mkdir(parents=True)
    (env_dir / "entry.sh").write_text(ENTRY_SH)
    (env_dir / "Dockerfile").write_text(
        "FROM ubuntu:24.04\nCOPY entry.sh /entry.sh\nRUN chmod 755 /entry.sh\n"
        'WORKDIR /app\nENTRYPOINT ["/entry.sh"]\n'
    )
    (task / "solution").mkdir()
    (task / "solution" / "solve.sh").write_text("#!/bin/bash\ntrue\n")
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
    )
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[agent]\ntimeout_sec = 120\n\n'
        "[verifier]\ntimeout_sec = 120\n"
    )
    (task / "instruction.md").write_text("Nothing to do.\n")
    return task


@pytest.mark.parametrize("direct", ["1", "0"], ids=["direct", "compose"])
def test_main_that_exits_right_after_its_release_ends_the_start(tmp_path, direct):
    """Both Docker paths: the Engine API's and Compose's
    (BENCHFLOW_DIRECT_DOCKER=0), whose start tears main down when the
    release's readiness wait finds it gone."""
    bench = Path(sys.executable).parent / "bench"
    if not bench.exists():
        pytest.skip("no bench script beside this Python")
    env = {k: v for k, v in os.environ.items() if k in ("USER", "LANG", "HOME")}
    env.update(
        PATH=os.environ.get("PATH", "/usr/bin:/bin"),
        HARBOR_TELEMETRY="off",
        PYTHONUNBUFFERED="1",
        BENCHFLOW_DIRECT_DOCKER=direct,
    )
    if "PYTHONPATH" in os.environ:
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    jobs = tmp_path / "jobs"
    done = subprocess.run(
        [
            str(bench),
            "eval",
            "run",
            "--tasks-dir",
            str(_task(tmp_path)),
            "--agent",
            "oracle",
            "--sandbox",
            "docker",
            "--concurrency",
            "1",
            "--retry-attempts",
            "2",
            "--jobs-dir",
            str(jobs),
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=900,
    )
    results = [p for p in jobs.rglob("result.json") if "__" in p.parent.name]
    assert results, done.stderr[-3000:]
    result = json.loads(results[0].read_text())
    error = result.get("error") or ""
    assert "sandbox startup failed: main stopped right after its release" in error, (
        result.get("error_category"),
        error[-1500:],
    )
    assert "ENTRYPOINT (/entry.sh) exited with status 4" in error, error
    assert "cannot add the agent's rules (no NET_ADMIN)" in error, error
    assert result.get("error_category") == "sandbox_setup"
    assert result.get("rewards") in (None, {})
    # Never retried (--retry-attempts 2): one attempt's trial folder only.
    assert len(results) == 1, [str(p) for p in results]
