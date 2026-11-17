"""A BusyBox image (Alpine, no bash) scores as its test.sh says in a full
rollout, in every integrity mode, on Docker's direct and compose paths; and
the outputs folder is root's, mode 0755, while test.sh runs (its owner
rule). Before the fix the rule's command failed on BusyBox's find and the
verifier crashed with PermissionError in audit, off and strict.

Plain Alpine has neither util-linux setpriv nor a Python. Where the line's
bounded launch needs one of them (0.9.x; rc has bfx), the trial ends with
that refusal, as on the line's base, never a crash; with setpriv added
(find stays BusyBox's) it scores on every line."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tests.integration.deterministic import harness as h
from tests.test_verifier_script_shell_docker import ALPINE, _task

pytestmark = pytest.mark.deterministic

ALPINE_SETPRIV = ALPINE.replace(
    "WORKDIR /app\n", "RUN apk add --no-cache setpriv\nWORKDIR /app\n"
)
IMAGES = {"alpine": ALPINE, "alpine-setpriv": ALPINE_SETPRIV}
# The bounded launch's refusal on an image with neither (0.9.x).
LAUNCH_REFUSED = (
    "verifier launch refused: the bounded launch refused: test.sh needs "
    "util-linux setpriv"
)

SANDBOX, SKIP_REASON = h.select_sandbox()

# 1 only under a shell other than bash in an image with no bash, with the
# outputs folder root's and 0755 while it runs.
OWNER_TEST_SH = """set -eu
owner=$(stat -c %u:%a /logs/verifier)
echo "$owner" > /logs/verifier/owner.txt
if [ -z "${BASH_VERSION:-}" ] && ! command -v bash >/dev/null 2>&1 \\
  && [ "$owner" = "0:755" ]; then
  r=1
else
  r=0
fi
echo "$r" > /logs/verifier/reward.txt
"""


def _run(tasks: Path, jobs: Path, integrity: str, direct: str) -> str:
    env = {
        **os.environ,
        "BENCHFLOW_SKIP_UPDATE_CHECK": "1",
        "BENCHFLOW_DIRECT_DOCKER": direct,
    }
    proc = subprocess.run(
        [
            h.bench_executable(),
            "eval",
            "run",
            "--tasks-dir",
            str(tasks),
            "--agent",
            "oracle",
            "--sandbox",
            "docker",
            "--integrity",
            integrity,
            "--retry-attempts",
            "0",
            "--jobs-dir",
            str(jobs),
        ],
        cwd=h.REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=1800,
    )
    return proc.stdout


@pytest.mark.parametrize("direct", ["1", "0"], ids=["direct", "compose"])
@pytest.mark.parametrize("integrity", ["audit", "off", "strict"])
@pytest.mark.parametrize("image", sorted(IMAGES))
def test_a_no_bash_alpine_task_scores_in_a_full_rollout(
    tmp_path, image, integrity, direct
):
    if SANDBOX != "docker":
        pytest.skip(SKIP_REASON if SANDBOX is None else "Docker only")
    # A name of its own per case: Docker tags a task's image bf__<name>.
    path = "direct" if direct == "1" else "compose"
    name = f"busybox-owner-{image}-{integrity}-{path}"
    tasks = tmp_path / "tasks"
    _task(tasks, name, OWNER_TEST_SH, IMAGES[image])
    jobs = tmp_path / "jobs"
    output = _run(tasks, jobs, integrity, direct)
    (result,) = jobs.glob(f"*/{name}__*/result.json")
    record = json.loads(result.read_text())
    owner = result.parent / "verifier" / "owner.txt"
    shown = owner.read_text().strip() if owner.exists() else None
    error = record.get("verifier_error")
    assert "verifier crashed" not in str(error), (error, output[-3000:])
    if image == "alpine" and str(error).startswith(LAUNCH_REFUSED):
        assert record.get("rewards") is None, record.get("rewards")
        return
    assert error is None, (error, output[-3000:])
    assert record.get("rewards") == {"reward": 1.0}, (
        record.get("rewards"),
        shown,
        output[-3000:],
    )
    assert shown == "0:755", shown
