"""The soft verifier's pins through the real Docker exec wrapper (b127).

On 3a54f24a the pin made the soft verifier's exec env non-empty,
so ``DockerSandbox.exec`` wrapped the command in its env-file wrapper, which
runs ``base64 -d`` from the image's PATH as root and sources what it prints.
An agent-written ``base64`` earlier on PATH could then name its own pip
configuration. Separately (pre-existing), an image BASH_ENV the agent could
rewrite ran before test.sh and replaced the pin.

The sandbox here is ``DockerSandbox`` itself, with only ``docker exec`` swapped
for a local ``/bin/sh -c``: the command it builds, wrapper included, runs as
built, under an image environment whose PATH starts with the agent's folder.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.docker import DockerSandbox
from benchflow.task import RolloutPaths, Verifier
from benchflow.task.config import TaskConfig
from tests._receipt_fakes import inner_command, receipt_answer

TEST_SH = """#!/bin/bash
{pip} download --no-deps --retries 0 --timeout 5 -d "$OUT/got" bfprobe \\
  >"$OUT/pip.log" 2>&1
if ls "$OUT/got"/*.whl >/dev/null 2>&1; then echo 0; else echo 1; fi \\
  > "$VERIFIER_DIR/reward.txt"
"""


def _docker(tmp_path: Path, paths: RolloutPaths, image_env: dict) -> DockerSandbox:
    """A DockerSandbox whose ``docker exec`` runs the built ``sh -c`` here.

    The env exec feeds on stdin is fed to it; the verifier's uid probe
    answers root; the root-only env folder is a folder of the test's own.
    """
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._persistent_env = {}
    sandbox.default_user = None
    sandbox._main_container = "main-container"
    tests = tmp_path / "sandbox-tests"
    out = tmp_path / "sandbox-out"
    out.mkdir()
    root_env = tmp_path / "root-env"
    sandbox.ran = []  # type: ignore[attr-defined]
    sandbox.stdin = []  # type: ignore[attr-defined]

    def local(command: str) -> str:
        return (
            command.replace("/logs/verifier", str(paths.rollout_dir / "verifier"))
            .replace("/tests/", f"{tests}/")
            .replace("/var/lib/benchflow/verifier-env", str(root_env))
        )

    async def run(
        argv, check=False, timeout_sec=None, raw_output=False, stdin_data=None
    ):
        assert argv[0] == "exec" and argv[-3] in ("sh", "/bin/sh") and argv[-2] == "-c"
        command = argv[-1]
        # integ reads housekeeping through exec receipts: answer the command a
        # receipt wraps and print its receipt, as the sandbox shell would.
        inner = inner_command(command)
        if stdin_data is not None:
            sandbox.stdin.append(stdin_data)  # type: ignore[attr-defined]
        if "Uid:" in inner and "/proc/self/status" in inner:  # the uid probe
            return receipt_answer(
                command, ExecResult(stdout="0\n", stderr="", return_code=0)
            )
        if inner.startswith("chmod +x ") or "chmod +x" in inner:
            subprocess.run(["/bin/sh", "-c", local(inner)], check=False)
        if "test-stdout.txt" not in inner:
            return receipt_answer(
                command, ExecResult(stdout="", stderr="", return_code=0)
            )
        sandbox.ran.append(command)  # type: ignore[attr-defined]
        result = subprocess.run(
            ["/bin/sh", "-c", local(command)],
            env={
                **image_env,
                "OUT": str(out),
                "VERIFIER_DIR": str(paths.rollout_dir / "verifier"),
            },
            input=stdin_data,
            capture_output=True,
            timeout=120,
        )
        return ExecResult(
            stdout=result.stdout.decode(),
            stderr=result.stderr.decode(),
            return_code=result.returncode,
        )

    async def upload_dir(source_dir, target_dir, **kwargs):
        subprocess.run(["cp", "-R", str(source_dir), str(tests)], check=True)

    async def upload_file(source_path, target_path, **kwargs):
        if not target_path.startswith("/var/lib/benchflow/verifier-env/"):
            return  # the trajectory published for the verifier: not read here
        target = Path(local(target_path))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)

    sandbox._run_docker_container_command = run  # type: ignore[method-assign]
    sandbox.upload_dir = upload_dir  # type: ignore[method-assign]
    sandbox.upload_file = upload_file  # type: ignore[method-assign]
    # /logs/verifier is the rollout's folder here already (local() above).
    sandbox.download_dir = AsyncMock()  # type: ignore[method-assign]
    sandbox.out = out  # type: ignore[attr-defined]
    return sandbox


def _setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A clean grader venv, the agent's wheelhouse and its pip.conf naming it."""
    from tests.test_verifier_user_config import _venv_with_pip, _wheelhouse

    python = _venv_with_pip(tmp_path / "scratch", "grader")
    if python is None:
        pytest.skip("no venv with pip")
    agent = _wheelhouse(tmp_path / "agent-wheels", "6.6")
    planted = tmp_path / "app" / "extra.conf"
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.write_text(f"[global]\nfind-links = {agent}\n")
    return Path(python).parent / "pip", planted, tmp_path / "app"


def _task(tmp_path: Path, pip: Path) -> MagicMock:
    tests_dir = tmp_path / "task" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test.sh").write_text(TEST_SH.format(pip=pip))
    task = MagicMock()
    task.name = "soft-transport"
    task.task_dir = tests_dir.parent
    task.paths.task_dir = tests_dir.parent
    task.paths.tests_dir = tests_dir
    task.paths.test_path = tests_dir / "test.sh"
    task.paths.uses_native_verifier_dir = False
    task.config = TaskConfig.model_validate_toml('version = "1.0"\n[verifier]\n')
    return task


async def _soft(tmp_path: Path, sandbox: DockerSandbox, task: MagicMock, paths):
    from benchflow.contracts.planes import default_rollout_planes
    from benchflow.sandbox.user import PassthroughUser
    from tests.test_user import _make_user_trial

    trial = _make_user_trial(PassthroughUser(), tmp_path=tmp_path / "trial")
    trial._task = task
    trial._rollout_paths = paths
    trial._env = sandbox
    trial._planes = default_rollout_planes()
    trial._config.sandbox_user = None
    trial._trajectory = [{"type": "agent_message", "text": "done"}]
    with (
        patch("benchflow.sandbox.lockdown._read_hardening_config", return_value={}),
        patch("benchflow.sandbox.lockdown._build_cleanup_cmd", return_value="true"),
        patch(
            "benchflow.sandbox.lockdown._blocked_before_freeze",
            AsyncMock(side_effect=lambda env, blocked: blocked),
            create=True,
        ),
    ):
        return await trial.soft_verify()


@pytest.mark.skipif(shutil.which("base64") is None, reason="needs base64")
async def test_soft_verify_never_runs_a_base64_the_agent_put_on_path(tmp_path):
    """Guards the b127 fix (fails on 3a54f24a: the wrapper ran it as root)."""
    pip, planted, app = _setup(tmp_path)
    marker = tmp_path / "planted-base64-ran"
    agent_bin = app / "bin"
    agent_bin.mkdir()
    (agent_bin / "base64").write_text(
        f"#!/bin/sh\ncat >/dev/null\ntouch {marker}\n"
        f"echo 'export PIP_CONFIG_FILE={planted}'\n"
    )
    (agent_bin / "base64").chmod(0o755)
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    image_env = {
        "PATH": f"{agent_bin}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "HOME": str(tmp_path),
        "PIP_NO_INDEX": "1",
    }
    sandbox = _docker(tmp_path, paths, image_env)
    rewards, _, error = await _soft(tmp_path, sandbox, _task(tmp_path, pip), paths)

    assert error is None, error
    assert not marker.exists(), "the agent's base64 ran as root"
    assert rewards == {"reward": 1.0}, (sandbox.out / "pip.log").read_text()
    (command,) = sandbox.ran
    assert "base64" not in command and "/tmp/.benchflow_exec_env_" not in command
    assert "export PIP_CONFIG_FILE=/dev/null" in command


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
async def test_soft_verify_ignores_an_agent_written_bash_env(tmp_path):
    """Guards the b127 fix (fails on 3a54f24a and integ: bash ran it)."""
    pip, planted, app = _setup(tmp_path)
    activate = app / "activate.sh"
    activate.write_text(f"export PIP_CONFIG_FILE={planted}\n")
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    image_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp_path),
        "PIP_NO_INDEX": "1",
        "BASH_ENV": str(activate),
    }
    sandbox = _docker(tmp_path, paths, image_env)
    rewards, _, error = await _soft(tmp_path, sandbox, _task(tmp_path, pip), paths)

    assert error is None, error
    assert rewards == {"reward": 1.0}, (sandbox.out / "pip.log").read_text()


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
async def test_a_direct_verifier_ignores_an_agent_written_bash_env(tmp_path):
    """A Verifier nobody hardened gets no BASH_ENV file either."""
    pip, planted, app = _setup(tmp_path)
    activate = app / "activate.sh"
    activate.write_text(f"export PIP_CONFIG_FILE={planted}\n")
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    image_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp_path),
        "PIP_NO_INDEX": "1",
        "BASH_ENV": str(activate),
        "ENV": str(activate),
    }
    sandbox = _docker(tmp_path, paths, image_env)
    result = await Verifier(
        task=_task(tmp_path, pip), rollout_paths=paths, sandbox=sandbox
    ).verify()
    assert result.rewards == {"reward": 1.0}, (sandbox.out / "pip.log").read_text()


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
async def test_the_planted_bash_env_is_what_bash_would_run(tmp_path):
    """The control: the same command with BASH_ENV left alone reads the file."""
    pip, planted, app = _setup(tmp_path)
    activate = app / "activate.sh"
    activate.write_text(f"export PIP_CONFIG_FILE={planted}\n")
    out = tmp_path / "out"
    out.mkdir()
    script = tmp_path / "test.sh"
    script.write_text(TEST_SH.format(pip=pip))
    subprocess.run(
        ["bash", str(script)],
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(tmp_path),
            "PIP_NO_INDEX": "1",
            "PIP_CONFIG_FILE": os.devnull,
            "BASH_ENV": str(activate),
            "OUT": str(out),
            "VERIFIER_DIR": str(out),
        },
        check=True,
    )
    assert (out / "reward.txt").read_text().strip() == "0"


@pytest.mark.skipif(shutil.which("base64") is None, reason="needs base64")
async def test_the_env_file_wrapper_never_runs_a_base64_on_the_image_path(tmp_path):
    """Pre-existing on integ (round 8 A/B): a verifier with an env of its own.

    Hardening always gives the verifier a non-empty env, so Docker's env-file
    wrapper ran ``base64 -d`` (and its trap's ``rm``) from the image's PATH as
    root, and sourced what the agent's ``base64`` printed. Both now come from
    the system folders.
    """
    pip, planted, app = _setup(tmp_path)
    marker = tmp_path / "planted-base64-ran"
    agent_bin = app / "bin"
    agent_bin.mkdir()
    for tool in ("base64", "rm"):
        (agent_bin / tool).write_text(
            f"#!/bin/sh\ncat >/dev/null\ntouch {marker}\n"
            f"echo 'export PIP_CONFIG_FILE={planted}'\n"
        )
        (agent_bin / tool).chmod(0o755)
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    image_env = {
        "PATH": f"{agent_bin}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "HOME": str(tmp_path),
        "PIP_NO_INDEX": "1",
    }
    sandbox = _docker(tmp_path, paths, image_env)
    task = _task(tmp_path, pip)
    task.config.verifier.env = {"JUDGE_API_KEY": "bf-fake-not-a-secret"}
    result = await Verifier(task=task, rollout_paths=paths, sandbox=sandbox).verify()

    (command,) = sandbox.ran
    assert ". /dev/stdin" in command  # the env went on stdin (b156)
    assert not marker.exists(), "the agent's base64 or rm ran as root"
    assert result.rewards == {"reward": 1.0}, (sandbox.out / "pip.log").read_text()


# --- b143: no decoder at all, so a trusted /usr/local/bin one works ---

MODE_TEST_SH = """#!/bin/sh
if [ "$MODE" = grade ]; then echo 1; else echo 0; fi > "$VERIFIER_DIR/reward.txt"
"""


async def test_an_image_whose_only_decoder_is_in_usr_local_bin_still_verifies(
    tmp_path, monkeypatch
):
    """Guards the b143 fix (fails on 09d0e332: ``base64: not found``).

    The image's system folders here hold no ``base64``; its own PATH has a
    trusted one in a ``/usr/local/bin`` of its own. A verifier with an env of
    its own must still get it, whatever folders hold which tools.
    """
    real = shutil.which("base64")
    if real is None:
        pytest.skip("needs base64")
    system = tmp_path / "system-bin"  # the image's /usr/bin and /bin: no base64
    system.mkdir()
    for tool in ("rm", "sh", "cat", "chmod"):
        found = shutil.which(tool)
        if found:
            (system / tool).symlink_to(found)
    local = tmp_path / "usr-local-bin"
    local.mkdir()
    (local / "base64").symlink_to(real)
    monkeypatch.setattr("benchflow.sandbox._base.SYSTEM_PATH", str(system))
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    image_env = {"PATH": f"{local}:{system}", "HOME": str(tmp_path)}
    sandbox = _docker(tmp_path, paths, image_env)
    tests_dir = tmp_path / "task" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test.sh").write_text(MODE_TEST_SH)
    task = _task_for(tests_dir)
    task.config.verifier.env = {"MODE": "grade"}
    result = await Verifier(task=task, rollout_paths=paths, sandbox=sandbox).verify()
    assert result.rewards == {"reward": 1.0}
    (command,) = sandbox.ran
    assert "base64" not in command and "/tmp/" not in command


def _task_for(tests_dir: Path) -> MagicMock:
    task = MagicMock()
    task.name = "decoder"
    task.task_dir = tests_dir.parent
    task.paths.task_dir = tests_dir.parent
    task.paths.tests_dir = tests_dir
    task.paths.test_path = tests_dir / "test.sh"
    task.paths.uses_native_verifier_dir = False
    task.config = TaskConfig.model_validate_toml('version = "1.0"\n[verifier]\n')
    return task


_AWKWARD = {
    "PLAIN": "value",
    "SECRET": "sk-very-secret-7f3a91",
    "QUOTES": 'it\'s "quoted" `tick` $(echo no) $HOME',
    "LINES": "first\nsecond\n",
    "PERCENT": "100% \\n %s \\0101",
    "UTF8": "é—🙂",
}


@pytest.mark.parametrize("shell", ["dash", "bash", "busybox", "sh"])
def test_the_wrapper_sets_every_value_with_the_shell_alone(tmp_path, shell):
    """Every value round-trips through printf and eval, no program on PATH."""
    from benchflow.sandbox._base import wrap_command_with_env_file

    program = shutil.which("busybox" if shell == "busybox" else shell)
    if program is None:
        pytest.skip(f"no {shell}")
    argv = [program, "sh"] if shell == "busybox" else [program]
    dump = tmp_path / "dump.py"
    dump.write_text(
        "import json, os, sys\n"
        f"keys = {sorted(_AWKWARD)!r}\n"
        "json.dump({k: os.environ.get(k) for k in keys}, open(sys.argv[1], 'w'))\n"
    )
    out = tmp_path / "out.json"
    import sys as _sys

    command = wrap_command_with_env_file(
        _AWKWARD, f"exec {_sys.executable} {dump} {out}", env_path_prefix="/tmp/x"
    )
    assert "sk-very-secret" not in command and "value" not in command
    empty = tmp_path / "empty-path"
    empty.mkdir()
    ran = subprocess.run(
        [*argv, "-c", command], env={"PATH": str(empty)}, capture_output=True, text=True
    )
    assert ran.returncode == 0, ran.stderr
    import json as _json

    assert _json.loads(out.read_text()) == _AWKWARD


# --- b143: a task's own BASH_ENV goes through the trust check ------


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("path", ["soft", "direct"])
async def test_a_task_bash_env_naming_an_agent_file_is_not_run(tmp_path, path):
    """Guards the b143 fix (fails on 09d0e332 and integ: bash ran it)."""
    pip, planted, app = _setup(tmp_path)
    activate = app / "activate.sh"
    activate.write_text(f"export PIP_CONFIG_FILE={planted}\n")
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    image_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp_path),
        "PIP_NO_INDEX": "1",
    }
    sandbox = _docker(tmp_path, paths, image_env)
    task = _task(tmp_path, pip)
    task.config.verifier.env = {"BASH_ENV": str(activate)}
    if path == "soft":
        rewards, _, error = await _soft(tmp_path, sandbox, task, paths)
        assert error is None, error
    else:
        result = await Verifier(
            task=task, rollout_paths=paths, sandbox=sandbox
        ).verify()
        rewards = result.rewards
    assert rewards == {"reward": 1.0}, (sandbox.out / "pip.log").read_text()
