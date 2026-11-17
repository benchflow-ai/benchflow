"""A removed start receipt retries the test command on real Docker (EN-548).

Deterministic tier (``tests/integration/README.md``): a whole oracle trial
on Docker, no kept services. Docker's exec answers a command that printed
nothing with ``stdout=None``, and the start-receipt probe prints nothing when
the receipt is missing (it marks the attempt abandoned instead). The probe
called ``.strip()`` on that None: every probe failed ("probe failed;
retrying"), the retry in place never ran, and the verifier waited out its
whole budget and scored 0.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from benchflow.rollout import _setup
from tests import test_keep_services_docker as kept

pytestmark = pytest.mark.deterministic
# Whole trials on real Docker: the exec channel and the side-by-side steps
# before the verifier stay on, as in a run (tests/conftest.py).
DOCKER_RUN_DEFAULTS = True

# The first attempt removes its own start receipt (the only one in the
# receipt folder then) and hangs; the second scores 1.
TEST_SH = """#!/bin/bash
if [ ! -e /tmp/receipt-removed ]; then
    touch /tmp/receipt-removed
    rm -f /run/benchflow/verifier-started-*
    echo first attempt
    sleep 300
fi
echo second attempt
echo 1 > /logs/verifier/reward.txt
"""


@pytest.fixture
def require_docker() -> None:
    if kept.SANDBOX != "docker":
        reason = (
            kept.SKIP_REASON
            if kept.SANDBOX is None
            else f"Docker-only scenario ({kept.SANDBOX})"
        )
        pytest.skip(reason)


@pytest.mark.usefixtures("require_docker")
async def test_a_removed_start_receipt_retries_the_test_command(
    tmp_path: Path, monkeypatch, caplog
):
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.sdk import SDK

    assert DockerSandbox.exec_channel is True
    assert DockerSandbox.concurrent_steps is True
    monkeypatch.setattr(_setup, "_VERIFIER_START_GRACE_SEC", 5.0)
    task = kept._task(tmp_path / "tasks", "receipt-removed", "#!/bin/bash\ntrue\n")
    toml = task / "task.toml"
    text = toml.read_text()
    keep = 'x-keep-services = [{ port = 5000, name = "fixture service" }]\n'
    budget = "[verifier]\ntimeout_sec = 120.0\n"
    assert keep in text and budget in text
    toml.write_text(
        text.replace(keep, "").replace(budget, "[verifier]\ntimeout_sec = 60.0\n")
    )
    (task / "tests" / "test.sh").write_text(TEST_SH)
    caplog.set_level(logging.INFO)
    jobs = tmp_path / "jobs"
    result = await SDK().run(
        task_path=task, agent="oracle", jobs_dir=jobs, environment="docker"
    )
    (record,) = jobs.glob("*/*/result.json")
    data = json.loads(record.read_text())
    stdout = (record.parent / "verifier" / "test-stdout.txt").read_text()
    assert result.rewards == {"reward": 1.0}, (data.get("verifier_error"), stdout)
    assert data.get("verifier_error") is None, data.get("verifier_error")
    assert stdout == "second attempt\n", stdout
    assert "start-receipt probe failed" not in caplog.text
    assert "no start receipt after 5s); retrying it once" in caplog.text
