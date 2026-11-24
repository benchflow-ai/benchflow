"""Row D1 on Daytona (#1247): after sandbox setup, the agent's uid can't create a name in /logs/verifier.

One oracle trial of the deterministic task through the real ``Rollout`` and ``DaytonaSandbox`` against the fake
Daytona API of ``tests/test_daytona_fake_trial_exec_counts.py`` (each command runs in a Docker container of the
task's image). The task's solve.sh, run as root after setup and before the clear that precedes test.sh, shows the
folder's owner and mode and tries to create a name there as a uid that is neither root nor the host's.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from tests.integration.deterministic import harness as h
from tests.test_daytona_fake_trial_exec_counts import (  # noqa: F401
    _Counts,
    _FakeClient,
    _manager_for,
    fake_daytona_image,
)

pytestmark = [
    pytest.mark.deterministic,
    pytest.mark.xdist_group("fake_daytona_image"),
]

_SOLVE = (
    "#!/bin/bash\n"
    "echo \"D1 $(stat -c '%u %a' /logs/verifier) "
    "$(setpriv --reuid=4242 --regid=4242 --clear-groups touch /logs/verifier/planted 2>/dev/null "
    '&& echo planted || echo refused)"\n'
    "rm -f /logs/verifier/planted\n"
    "printf 'Hello, world!\\n' > /app/hello.txt\n"
)


@pytest.mark.usefixtures("fake_daytona_image")
async def test_daytona_setup_leaves_the_verifier_folder_roots(
    monkeypatch, tmp_path
) -> None:
    """Row D1, Daytona (fake API). Mutant: put back the ``chmod 777``."""
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.sandbox import daytona as daytona_module

    client = _FakeClient(_Counts())
    monkeypatch.setattr(daytona_module, "DaytonaClientManager", _manager_for(client))
    task = tmp_path / "deterministic"
    shutil.copytree(h.TEMPLATE_TASK, task)
    (task / "solution" / "solve.sh").write_text(_SOLVE)
    try:
        rollout = Rollout(
            RolloutConfig(
                task_path=task,
                agent="oracle",
                environment="daytona",
                jobs_dir=tmp_path / "jobs",
                job_name="d1-daytona",
                rollout_name="deterministic__d1",
            )
        )
        await rollout.run()
    finally:
        for sandbox in client.sandboxes:
            await sandbox.delete()
    # solve.sh's output is in the oracle step of the trajectory.
    (log,) = list(Path(tmp_path / "jobs").rglob("trajectory/acp_trajectory.jsonl"))
    found = re.search(r"D1 (\d+) (\d+) (\w+)", log.read_text())
    assert found, log.read_text()[-2000:]
    line = found.group(0)
    owner, mode, outcome = found.groups()
    assert outcome == "refused", line
    assert owner == "0" and not int(mode, 8) & 0o022, line
