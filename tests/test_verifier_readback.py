"""What root reads back after grading is what test.sh left (read-back #3, #5, #6, #7).

#3: taskmd's test criteria come from ctrf.json as test.sh left it at its exit
(``verifier-at-exit/``), never from the live folder, which goes to the host
user after test.sh, a uid the agent's may share.
#5: the output folder's close removes a root-owned name with another link (a
hard link to a root file the agent may write), not only names not root's.
#6: a close that fails fails closed (VerifierOutputsHeldError, the verifier
failed): test.sh never runs with the folder open to the agent.
#7: a verifier-network refusal while the agent's processes were alive says
whose traffic it was is unknown.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.task.paths import RolloutPaths
from benchflow.task.verifier import Verifier
from benchflow.task.verifier_core import _OUTPUTS_CLOSED


def _verifier(tmp_path: Path, *, close_rc: int = 0) -> tuple[Verifier, list[str]]:
    rollout_paths = RolloutPaths(tmp_path)
    rollout_paths.mkdir()
    task = MagicMock()
    task.config.verifier.user = "root"
    sandbox = MagicMock()
    sandbox.is_mounted = False
    seen: list[str] = []

    async def fake_exec(*args, **kwargs):
        command = args[0] if args else kwargs.get("command", "")
        seen.append(command)
        rc = close_rc if "chown 0:0" in command else 0
        return SimpleNamespace(return_code=rc, exit_code=rc, stdout="", stderr="")

    sandbox.exec = AsyncMock(side_effect=fake_exec)
    return Verifier(task=task, rollout_paths=rollout_paths, sandbox=sandbox), seen


# -- #6 -------------------------------------------------------------------------


async def test_a_close_that_fails_fails_closed(tmp_path):
    """The folder never became root's (no marker): test.sh does not run with
    it open, the verifier failed, never retried (VerifierOutputsHeldError,
    FXRANGE's ending), and the reward's provenance records why."""
    from benchflow.task.verifier_core import VerifierOutputsHeldError

    verifier, _ = _verifier(tmp_path, close_rc=1)
    with pytest.raises(VerifierOutputsHeldError, match="could not be made root's"):
        await verifier._close_output_dir("main", mounted=False)
    assert verifier.outputs_closed is False
    assert verifier.outputs_open_why == "the close failed (rc=1)"


async def test_a_close_that_holds_is_recorded(tmp_path):
    verifier, _ = _verifier(tmp_path)
    await verifier._close_output_dir("main", mounted=False)
    assert verifier.outputs_closed is True


async def test_a_verifier_of_its_own_user_gets_the_folder_as_its_own(tmp_path):
    """Read-back #2 (RB2): a verifier that runs as a user of its own gets the
    output folder as that user's, 0755, swept of every other user's names."""
    verifier, seen = _verifier(tmp_path)
    verifier._task.config.verifier.user = "tester"
    verifier._verifier_uid_read = "1234"
    await verifier._close_output_dir("main", mounted=False)
    assert verifier.outputs_closed is True
    close = next(c for c in seen if "chmod 755" in c)
    assert close.startswith("chown 1234 ") and "! -user 1234" in close, close
    assert not any("chown 0:0" in c for c in seen)


# -- #5 -------------------------------------------------------------------------


#: BusyBox's find (no -uid) and GNU's: the close runs in both.
ALPINE = (
    "alpine@sha256:d9e853e87e55526f6b2917df91a2115c36dd7c696a35be12163d44e6e2a4b6bc"
)


@pytest.mark.deterministic
@pytest.mark.parametrize("image", ["debian", "busybox"])
async def test_the_close_removes_a_hard_link_to_a_root_file(tmp_path, image):
    """In a container, as root: a hard link in the folder to a root-owned file
    any user may write goes, as does a name not root's; a root file of the
    folder's own stays."""
    from tests.exec_diet_fakes import PYTHON, docker_available

    if not docker_available():
        pytest.skip("needs a Docker daemon")
    verifier, seen = _verifier(tmp_path)
    await verifier._close_output_dir("main", mounted=False)
    (close,) = [c for c in seen if "chown 0:0" in c]
    script = (
        "set -e; mkdir -p /logs/verifier /opt/shared; "
        "echo planted > /opt/shared/f; chmod 666 /opt/shared/f; "
        "ln /opt/shared/f /logs/verifier/reward.txt; "
        "echo agent > /logs/verifier/agent.txt; chown 1000 /logs/verifier/agent.txt; "
        "echo own > /logs/verifier/own.txt; "
        f"{close}; ls -A /logs/verifier"
    )
    out = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            PYTHON if image == "debian" else ALPINE,
            "sh",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    left = [n for n in out.stdout.split() if n != _OUTPUTS_CLOSED]
    assert left == ["own.txt"], out.stdout


@pytest.mark.deterministic
async def test_an_image_without_find_is_swept_by_the_shell(tmp_path):
    """No find in the image: the shell sweeps each top-level name not root's
    and each link; a root file of the folder's own stays. (A root name with
    another link needs find's -links: the shell cannot see it.)"""
    from tests.exec_diet_fakes import PYTHON, docker_available

    if not docker_available():
        pytest.skip("needs a Docker daemon")
    verifier, seen = _verifier(tmp_path)
    await verifier._close_output_dir("main", mounted=False)
    (close,) = [c for c in seen if "chown 0:0" in c]
    script = (
        'set -e; rm -f "$(command -v find)"; ! command -v find; '
        "mkdir -p /logs/verifier/agent-dir; chown -R 1000 /logs/verifier/agent-dir; "
        "echo agent > /logs/verifier/.agent; chown 1000 /logs/verifier/.agent; "
        "ln -s /etc/hostname /logs/verifier/link; "
        "echo own > /logs/verifier/own.txt; "
        f"{close}; ls -A /logs/verifier"
    )
    out = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", PYTHON, "sh", "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    left = [n for n in out.stdout.split() if n != _OUTPUTS_CLOSED]
    assert left == ["own.txt"], out.stdout


# -- #3 -------------------------------------------------------------------------


def test_an_output_is_read_from_the_copy_at_exit(tmp_path):
    from benchflow.task.verifier_outputs import AT_EXIT

    verifier, _ = _verifier(tmp_path)
    (tmp_path / "verifier" / "ctrf.json").write_text("live")
    (tmp_path / "verifier" / "other.json").write_text("live")
    (tmp_path / AT_EXIT).mkdir()
    (tmp_path / AT_EXIT / "ctrf.json").write_text("at exit")
    # No copy taken: the live folder is test.sh's.
    assert verifier.output_at_exit("ctrf.json").read_text() == "live"
    verifier.outputs_at_exit = {
        "files": {"ctrf.json": {"sha256": "0" * 64, "size": 7, "copied": True}},
        "special": [],
    }
    assert verifier.output_at_exit("ctrf.json").read_text() == "at exit"
    # Not in the copy: test.sh left none, whatever the live folder holds now.
    assert not verifier.output_at_exit("other.json").is_file()
    # Too large to copy: read where it is.
    verifier.outputs_at_exit["files"]["ctrf.json"]["copied"] = False
    assert verifier.output_at_exit("ctrf.json").read_text() == "live"


async def test_taskmd_reads_the_ctrf_test_sh_left(tmp_path):
    """test.sh's report fails the test; the live folder's, written after it
    exited, passes it: the trial fails."""
    from benchflow.task.formats import materialize_task_dir
    from benchflow.taskmd import verify
    from tests import test_taskmd_verify as tv
    from tests._taskmd_helpers import EXAMPLES

    native = materialize_task_dir(EXAMPLES / "hello-world")
    verifier = tv._Verifier(
        native,
        tmp_path / "trial",
        tv._Sandbox(tmp_path / "c"),
        report=tv._ctrf(("test_hello", "failed")),
    )
    at_exit = tmp_path / "trial" / "verifier-at-exit"
    run = verifier._verify_test_script

    async def then_rewritten(**kwargs):
        result = await run(**kwargs)
        live = verifier._rollout_paths.verifier_dir / "ctrf.json"
        at_exit.mkdir(parents=True, exist_ok=True)
        (at_exit / "ctrf.json").write_text(live.read_text())
        live.write_text(json.dumps(tv._ctrf(("test_hello", "passed"))))
        return result

    verifier._verify_test_script = then_rewritten
    verifier.output_at_exit = lambda name: at_exit / name
    result = await verify.verify_taskmd(verifier, tv._strategy(native))
    assert result.rewards["reward"] == 0.0


# -- #7 -------------------------------------------------------------------------


def test_a_network_refusal_beside_the_agents_processes_says_so():
    """The refusal's read flag and record say whose packets are unknown when
    the agent's processes were alive during test.sh; the score is not
    touched."""
    from types import SimpleNamespace

    from benchflow._utils.trial_warnings import recording_trial_warnings
    from benchflow.rollout import _setup

    refusal = {
        "phase": "test.sh",
        "where": "in-container",
        "mode": "no-network",
        "refused": 3,
        "refused_words": "3 packets",
        "exit_code": 1,
        "passed": False,
    }
    for alive in (True, False):
        rows = [{"where": "processes", "after": {"pid": 7, "argv": "sleep 99"}}]
        env = SimpleNamespace(
            _benchflow_verifier_inputs={"flags": [], "changes": rows if alive else []}
        )
        verifier = SimpleNamespace(network_refusal=dict(refusal))
        with recording_trial_warnings() as warnings:
            _setup._verifier_network_refused(env, verifier)
        record = warnings.records["verifier_network_refused"]
        assert record.get("agent_processes_alive") is (True if alive else None)
        (line,) = warnings
        assert ("whose traffic it was is unknown" in line) is alive, line
        assert "refused test.sh 3 packets" in line, line
        assert "scored as test.sh says" in line, line
