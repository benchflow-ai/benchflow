"""Every verifier entry point gets the verifier's pinned environment.

Guards the b89 finding: ``soft_verify()`` ran the
verifier without hardening, so its pip read a ``pip.conf`` the agent planted in
a venv the verifier used, as root. The fix is by construction: one function,
``benchflow._verifier_env.pin_verifier_env``, sets the invariants, and it
runs both when hardening assembles the environment and at each place the
verifier hands an environment to the sandbox. These tests enumerate those
places and every entry point that reaches them, so a new one fails here until
it is listed, and show the behaviour on the soft path, the one that never
hardens.
"""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.task import RolloutPaths, Verifier
from benchflow.task.config import TaskConfig
from tests._package_files import package_path
from tests._receipt_fakes import inner_command, receipt_answer

SRC = package_path("")

# Every function that constructs the verifier (``planes.verifier(...)``, or
# ``Verifier(...)`` directly), with the entry points that reach it.
VERIFIER_CONSTRUCTORS = {
    # V3 builds it in _verify_hardened, inside _verify_rollout's root route.
    ("rollout/_setup.py", "_verify_hardened"): (
        "hard verification (Rollout.verify), its in-place retry (_await_verifier),"
        " separate-verifier sandboxes, verifier recovery, regrade, sdk, task"
        " acceptance and equivalence checks"
    ),
    ("rollout/__init__.py", "_soft_verify"): (
        "soft verification between user rounds (soft_verify)"
    ),
    ("rollout_planes.py", "verifier"): "the plane factory both of those call",
}

# Every caller of _verify_rollout: each one is hard verification on some
# sandbox (Docker or Daytona alike: the sandbox object is the only difference).
VERIFY_ROLLOUT_CALLERS = {
    ("eval_regrade.py", "run_verifier_on_frozen_trial"),
    ("sdk.py", "_verify"),
    ("rollout/_separate_verifier.py", "verify"),
    ("rollout/_verifier_recovery.py", "_recover_verifier"),
    # Rollout.verify scores through _score_finished_solver (SV3).
    ("rollout/__init__.py", "_score_finished_solver"),
    ("task/acceptance_live.py", "_run_single_case"),
    ("task/equivalence.py", "grade"),
}

# The functions in verifier_core that run a command as the verifier's user in
# the sandbox: each takes its env from sandbox_verifier_env.
VERIFIER_EXEC_FUNCTIONS = {"_verify_test_script", "_verify_criteria_runner"}


def _functions(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _calls(node: ast.AST):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            yield sub


def _name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _innermost(path: Path, predicate) -> set[tuple[str, str]]:
    """(file, innermost function) for every call matching *predicate*."""
    found: set[tuple[str, str]] = set()
    rel = path.relative_to(SRC).as_posix()
    functions = list(_functions(path))
    for function in functions:
        inner = [f for f in functions if f is not function and f in ast.walk(function)]
        nested = {id(c) for f in inner for c in _calls(f)}
        for call in _calls(function):
            if id(call) not in nested and predicate(call):
                found.add((rel, function.name))
    return found


def _sources():
    return sorted(p for p in SRC.rglob("*.py") if "trajectories/viewer" not in str(p))


def _constructs_verifier(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id in ("Verifier", "_Verifier")
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "verifier"
        and ast.unparse(func.value).endswith("planes")
    )


def test_every_verifier_constructor_is_listed():
    found: set[tuple[str, str]] = set()
    for path in _sources():
        found |= _innermost(path, _constructs_verifier)
    assert found == set(VERIFIER_CONSTRUCTORS)


def test_every_hard_verification_caller_is_listed():
    found: set[tuple[str, str]] = set()
    for path in _sources():
        found |= _innermost(path, lambda c: _name(c) == "_verify_rollout")
    assert found == VERIFY_ROLLOUT_CALLERS


def _is_verifier_user(node: ast.AST) -> bool:
    return ast.unparse(node).endswith("config.verifier.user")


def test_every_verifier_exec_takes_the_pinned_env():
    """Each exec as the verifier's user passes an env built by sandbox_verifier_env."""
    core = SRC / "task" / "verifier_core.py"
    seen = set()
    for function in _functions(core):
        execs = [
            c
            for c in _calls(function)
            if _name(c) in ("exec", "exec_raw")
            and any(k.arg == "user" and _is_verifier_user(k.value) for k in c.keywords)
        ]
        if not execs:
            continue
        seen.add(function.name)
        names = {_name(c) for c in _calls(function)}
        # The env is the task's; the pins are builtins in the command.
        assert {"sandbox_verifier_env", "pinned_shell_prefix"} <= names, function.name
        for call in execs:
            (env,) = [k.value for k in call.keywords if k.arg == "env"]
            assert isinstance(env, ast.Name) and env.id in ("env", "run_env")
    assert seen == VERIFIER_EXEC_FUNCTIONS
    # Nowhere else runs a command as the verifier's user.
    elsewhere = set()
    for path in _sources():
        if path == core:
            continue
        elsewhere |= _innermost(
            path,
            lambda c: (
                _name(c) in ("exec", "exec_raw")
                and any(
                    k.arg == "user" and _is_verifier_user(k.value) for k in c.keywords
                )
            ),
        )
    assert elsewhere == set()


def test_hardening_and_the_taskmd_runner_use_the_same_pins():
    from benchflow._verifier_env import VERIFIER_PINNED_ENV

    lockdown = (SRC / "sandbox" / "lockdown.py").read_text()
    assert "verifier_env = pin_verifier_env(verifier_env)" in lockdown
    from benchflow.taskmd.verify import SandboxRunner

    runner = SimpleNamespace(
        drop=("setpriv", "/usr/bin/setpriv", None),
        kept=(),
        path="/usr/bin:/bin",
        safepath=None,
    )
    script = SandboxRunner.command_script(runner, "true", 5)
    for key, value in VERIFIER_PINNED_ENV.items():
        assert f" {key}={value}" in script


def test_the_pins_win_over_the_task_env():
    from benchflow._verifier_env import pin_verifier_env

    assert pin_verifier_env({"PIP_CONFIG_FILE": "/task/pip.conf", "A": "b"}) == {
        "PIP_CONFIG_FILE": os.devnull,
        "A": "b",
    }
    assert pin_verifier_env(None) == {"PIP_CONFIG_FILE": os.devnull}


def test_image_pip_config_is_refused_with_the_fix():
    """A task.toml still naming the cut opt-in fails to load, naming the fix."""
    from benchflow._verifier_env import PIP_CONFIG_FIX
    from benchflow.task.imports import TaskConfigKeyError, load_task_config_toml

    with pytest.raises(TaskConfigKeyError) as refused:
        load_task_config_toml(
            'version = "1.0"\n[verifier]\nimage_pip_config = true\n',
            source="task.toml",
        )
    assert "'verifier.image_pip_config' is not supported" in str(refused.value)
    assert PIP_CONFIG_FIX in str(refused.value)


def test_restore_workspace_is_refused_with_what_to_do_instead():
    """SkillsBench tasks reached for restore_workspace to keep inputs
    the agent could edit honest; it was refused as a bare unknown key."""
    from benchflow.task.imports import TaskConfigKeyError, load_task_config_toml

    with pytest.raises(TaskConfigKeyError) as refused:
        load_task_config_toml(
            'version = "1.0"\n[verifier]\nrestore_workspace = true\n',
            source="task.toml",
        )
    message = str(refused.value)
    assert "'verifier.restore_workspace' is not a task setting" in message
    assert "sha256sum -c" in message


# --- the behaviour: a planted venv pip.conf on each verifier exec path --------

TEST_SH = """#!/bin/sh
{pip} download --no-deps --retries 0 --timeout 5 -d "$OUT/got" bfprobe \
  >"$OUT/pip.log" 2>&1
if ls "$OUT/got"/*.whl >/dev/null 2>&1; then echo 0; else echo 1; fi \
  > "$VERIFIER_DIR/reward.txt"
"""


class LocalSandbox:
    """The verifier's real command, through /bin/sh here, with the env it passed.

    Other commands (the soft path's housekeeping as root) succeed without
    running: only what the verifier runs as its user matters here.
    """

    is_mounted = True
    # Root's log read on soft verification takes the system shell (PT, C8).
    supports_system_shell = True

    def __init__(self, tmp_path: Path, paths: RolloutPaths, home: Path, unpinned=False):
        self.unpinned = unpinned  # the control: drop the pin before running
        self.tests = tmp_path / "sandbox-tests"
        self.out = tmp_path / "sandbox-out"
        self.out.mkdir()
        self.paths = paths
        self.home = home
        self.envs: list[dict | None] = []
        self.commands: list[str] = []
        self.ran: list[tuple[str, str]] = []

    async def upload_dir(self, source_dir, target_dir, **kwargs):
        subprocess.run(["cp", "-R", str(source_dir), str(self.tests)], check=True)

    async def upload_file(self, *args, **kwargs):
        return None

    def local(self, command: str) -> str:
        return command.replace(
            "/logs/verifier", str(self.paths.rollout_dir / "verifier")
        ).replace("/tests/", f"{self.tests}/")

    async def exec(self, command, env=None, **kwargs):
        # integ reads housekeeping through exec receipts: answer the command a
        # receipt wraps and print its receipt, as the sandbox shell would.
        inner = inner_command(command)
        if inner.startswith("chmod +x "):
            subprocess.run(["/bin/sh", "-c", self.local(inner)], check=True)
        if "test-stdout.txt" not in inner:
            return receipt_answer(
                command, ExecResult(stdout="", stderr="", return_code=0)
            )
        self.envs.append(env)
        self.commands.append(command)
        if self.unpinned:
            command = command.replace("export PIP_CONFIG_FILE=/dev/null; ", "")
        result = subprocess.run(
            ["/bin/sh", "-c", self.local(command)],
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": str(self.home),
                "OUT": str(self.out),
                "VERIFIER_DIR": str(self.paths.rollout_dir / "verifier"),
                "PIP_NO_INDEX": "1",
                **(env or {}),
            },
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.ran.append((self.local(command), result.stderr))
        return ExecResult(
            stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
        )


def _planted(tmp_path: Path) -> tuple[Path, Path]:
    """A venv the verifier uses, whose site pip.conf the agent wrote."""
    from tests.test_verifier_user_config import _venv_with_pip, _wheelhouse

    python = _venv_with_pip(tmp_path / "scratch", "grader")
    if python is None:
        pytest.skip("no venv with pip")
    venv = Path(python).parent.parent
    agent = _wheelhouse(tmp_path / "agent-wheels", "6.6")
    (venv / "pip.conf").write_text(f"[global]\nfind-links = {agent}\n")
    return venv, Path(python)


def _task(tmp_path: Path, pip: Path) -> MagicMock:
    tests_dir = tmp_path / "task" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test.sh").write_text(TEST_SH.format(pip=pip))
    task = MagicMock()
    task.name = "planted-venv-pip-conf"
    task.task_dir = tests_dir.parent
    task.paths.task_dir = tests_dir.parent
    task.paths.tests_dir = tests_dir
    task.paths.test_path = tests_dir / "test.sh"
    task.paths.uses_native_verifier_dir = False
    task.config = TaskConfig.model_validate_toml('version = "1.0"\n[verifier]\n')
    return task


async def test_a_planted_venv_pip_conf_is_not_read_on_the_soft_path(tmp_path):
    """Failing first on edfe9a4e: soft_verify passed the task's env unpinned."""
    from benchflow.contracts.planes import default_rollout_planes
    from benchflow.sandbox.user import PassthroughUser
    from tests.test_user import _make_user_trial

    venv, _ = _planted(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    trial = _make_user_trial(PassthroughUser(), tmp_path=tmp_path / "trial")
    trial._task = _task(tmp_path, venv / "bin" / "pip")
    trial._rollout_paths = paths
    trial._env = LocalSandbox(tmp_path, paths, home)
    trial._planes = default_rollout_planes()
    trial._config.sandbox_user = None
    trial._trajectory = [{"type": "agent_message", "text": "done"}]
    with (
        patch("benchflow.sandbox.lockdown._read_hardening_config", return_value={}),
        patch("benchflow.sandbox.lockdown._build_cleanup_cmd", return_value="true"),
        # Root's housekeeping before the purge (mount aliases): not the env.
        patch(
            "benchflow.sandbox.lockdown._blocked_before_freeze",
            AsyncMock(side_effect=lambda env, blocked: blocked),
            create=True,
        ),
    ):
        rewards, _, error = await trial.soft_verify()

    assert error is None, error
    # 0.0: pip read the planted pip.conf and fetched the agent's wheel.
    assert rewards == {"reward": 1.0}, (trial._env.out / "pip.log").read_text()
    # The pin rides in the command, the exec env stays the task's (none).
    (env,) = trial._env.envs
    assert env is None
    (command,) = trial._env.commands
    assert "export PIP_CONFIG_FILE=/dev/null; " in command


async def test_the_planted_venv_pip_conf_is_what_the_soft_path_would_read(tmp_path):
    """The control: the same command without the pin fetches the agent's wheel."""
    venv, _ = _planted(tmp_path)
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    sandbox = LocalSandbox(tmp_path, paths, tmp_path, unpinned=True)
    task = _task(tmp_path, venv / "bin" / "pip")
    result = await Verifier(task=task, rollout_paths=paths, sandbox=sandbox).verify()
    assert result.rewards == {"reward": 0.0}


@pytest.mark.parametrize("task_env", [None, {"PIP_CONFIG_FILE": "/task/pip.conf"}])
async def test_a_planted_venv_pip_conf_is_not_read_by_a_direct_verifier(
    tmp_path, task_env
):
    """A Verifier nobody hardened (an SDK caller's own): still pinned."""
    venv, _ = _planted(tmp_path)
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    sandbox = LocalSandbox(tmp_path, paths, tmp_path)
    task = _task(tmp_path, venv / "bin" / "pip")
    task.config.verifier.env = task_env or {}
    result = await Verifier(task=task, rollout_paths=paths, sandbox=sandbox).verify()
    assert result.rewards == {"reward": 1.0}, (sandbox.out / "pip.log").read_text()
    (command,) = sandbox.commands
    assert "export PIP_CONFIG_FILE=/dev/null; " in command


async def test_a_non_root_soft_verifier_without_env_is_not_refused(tmp_path):
    """V3-INTEG-2: BenchFlow's TEST_DIR default is no env to hand over.

    A sandbox whose exec env goes on a command line (Daytona's inline form;
    this fake is not private either) refuses a non-root verifier only for an
    env the agent must not read. A task with ``[verifier] user`` and no env
    has none, so soft verification runs, and test.sh still gets TEST_DIR.
    """
    tests_dir = tmp_path / "task" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test.sh").write_text(
        '#!/bin/sh\nprintf %s "$TEST_DIR" > "$OUT/test_dir"\n'
        'echo 1 > "$VERIFIER_DIR/reward.txt"\n'
    )
    task = MagicMock()
    task.name = "non-root-no-env"
    task.task_dir = tests_dir.parent
    task.paths.task_dir = tests_dir.parent
    task.paths.tests_dir = tests_dir
    task.paths.test_path = tests_dir / "test.sh"
    task.paths.uses_native_verifier_dir = False
    task.config = TaskConfig.model_validate_toml(
        'version = "1.0"\n[verifier]\nuser = "tester"\n'
    )
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    sandbox = LocalSandbox(tmp_path, paths, tmp_path)
    assert getattr(sandbox, "exec_env_is_private", False) is not True
    verifier = Verifier(task=task, rollout_paths=paths, sandbox=sandbox)
    verifier.agent_processes_alive = True
    result = await verifier.verify()
    assert result.rewards == {"reward": 1.0}
    (env,) = sandbox.envs
    assert env is None
    assert (sandbox.out / "test_dir").read_text() == "/tests"
