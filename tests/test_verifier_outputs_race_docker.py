"""A process of another user still writing in /logs/verifier when the owner
rule closes it (the agent's, alive through grading in integrity audit and
off) cannot keep the close from finishing, on GNU (ubuntu) and BusyBox
(Alpine) images; a close that still fails, or does not finish, is the
verifier's failure, never infrastructure (R330D, 10/10 before the fix)."""

from __future__ import annotations

import pytest

import benchflow.task.verifier_core as verifier_core
from benchflow.sandbox.docker import DockerSandbox
from benchflow.task import RolloutPaths, Task
from benchflow.task.config import SandboxConfig
from benchflow.task.verifier_core import Verifier, VerifierOutputsHeldError
from tests.integration.deterministic import harness as h
from tests.test_verifier_owner_rule_busybox_docker import ALPINE_SETPRIV
from tests.test_verifier_script_shell_docker import POSIX_TEST_SH, UBUNTU, _task

pytestmark = pytest.mark.deterministic

SANDBOX, SKIP_REASON = h.select_sandbox()
IMAGES = {"ubuntu": UBUNTU, "alpine-setpriv": ALPINE_SETPRIV}
# uid 1000 owns a subfolder and keeps making files and folders in it.
WRITER = (
    "mkdir -p /logs/verifier/sub && chmod 777 /logs/verifier && "
    "chown 1000:1000 /logs/verifier/sub && "
    "setsid setpriv --reuid=1000 --regid=1000 --clear-groups sh -c "
    "'cd /logs/verifier/sub && i=0; while :; do : > f$i 2>/dev/null; "
    "mkdir -p d$i/e 2>/dev/null; i=$((i+1)); done' >/dev/null 2>&1 < /dev/null & "
    "sleep 1"
)


async def _closed(tmp_path, image: str, case: str):
    """The close run against a live writer; returns (verifier, sandbox, error)."""
    if SANDBOX != "docker":
        pytest.skip(SKIP_REASON if SANDBOX is None else "Docker only")
    name = f"outputs-race-{case}-{image}"
    task_dir = _task(tmp_path / "tasks", name, POSIX_TEST_SH, IMAGES[image])
    paths = RolloutPaths(rollout_dir=tmp_path / "run")
    paths.mkdir()
    sandbox = DockerSandbox(
        environment_dir=task_dir / "environment",
        environment_name=name,
        session_id=name,
        rollout_paths=paths,
        task_env_config=SandboxConfig(cpus=1, memory_mb=512),
    )
    await sandbox.start(force_build=False)
    verifier = Verifier(task=Task(task_dir), rollout_paths=paths, sandbox=sandbox)
    error: BaseException | None = None
    try:
        await sandbox.exec(WRITER, user="root")
        try:
            await verifier._close_output_dir("main", mounted=sandbox.is_mounted)
        except VerifierOutputsHeldError as exc:
            error = exc
        shown = await sandbox.exec(
            "stat -c %u:%a /logs/verifier; find /logs/verifier -mindepth 1 ! -user 0 | wc -l",
            user="root",
        )
        state = (getattr(shown, "stdout", "") or "").split()
        await verifier._open_output_dir_to_host("main")
    finally:
        await sandbox.exec("pkill -u 1000 || true", user="root")
        await sandbox.stop(delete=True)
    return verifier, state, error


@pytest.mark.parametrize("image", sorted(IMAGES))
async def test_a_writer_left_in_the_folder_cannot_stop_the_close(tmp_path, image):
    _, state, error = await _closed(tmp_path, image, "race")
    assert error is None, error
    assert state == ["0:755", "0"], state


@pytest.mark.parametrize("image", sorted(IMAGES))
async def test_a_sweep_that_still_fails_is_the_verifiers_failure(
    tmp_path, monkeypatch, image
):
    monkeypatch.setattr(verifier_core, "_SWEEP_FOREIGN", "exit 1")
    _, _, error = await _closed(tmp_path, image, "fail")
    assert isinstance(error, VerifierOutputsHeldError), error


@pytest.mark.parametrize("image", sorted(IMAGES))
async def test_a_close_that_does_not_finish_is_the_verifiers_failure(
    tmp_path, monkeypatch, image
):
    monkeypatch.setattr(verifier_core, "_SETUP_EXEC_TIMEOUT_SEC", 3)
    monkeypatch.setattr(verifier_core, "_SWEEP_FOREIGN", "sleep 30")
    _, _, error = await _closed(tmp_path, image, "flood")
    assert isinstance(error, VerifierOutputsHeldError), error
    assert "did not finish" in str(error), error
