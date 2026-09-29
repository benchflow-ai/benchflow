"""BenchFlow's own verifier state must not trip a task's environment checks.

Task verifiers often check, before or inside pytest, that the grading
environment carries no Python startup hooks and that the verifier output
directory holds only what test.sh wrote. The synthetic verifiers below cover
the usual forms:

- a script run as ``python3 -I -S`` before pytest that fails when
  ``PYTHONPATH``, ``PYTHONHOME``, ``PYTHONSTARTUP``, ``PYTHONUSERBASE`` or
  ``PYTEST_PLUGINS`` is set, or when ``/logs/verifier`` already holds a file
  other than ``test-stdout.txt``;
- pytest tests that assert ``PYTHONPATH`` is unset or empty;
- a grading script that test.sh runs under ``env -i``.

The pytest plugin guard put its directory on the PYTHONPATH of every
``main``-service verifier, and solver-evidence preservation wrote a start
receipt into /logs/verifier before every test.sh. Either one made a correct
solution fail checks like these.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.rollout._setup import _verifier_started, _verify_rollout
from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.lockdown import (
    _build_verifier_env,
    harden_before_verify,
    pytest_plugin_guard_name,
)
from benchflow.task import RolloutPaths, Verifier
from benchflow.task.config import TaskConfig

CHECK_ENV = """\
import os
import sys
from pathlib import Path

HOOK_VARIABLES = (
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
    "PYTEST_PLUGINS",
)

errors = [f"hook variable set: {name}" for name in HOOK_VARIABLES if os.environ.get(name)]
output_dir = Path(sys.argv[1])
errors += [
    f"unexpected file: {entry.name}"
    for entry in sorted(output_dir.iterdir())
    if entry.name != "test-stdout.txt"
]
for error in errors:
    print("check_env:", error)
sys.exit(1 if errors else 0)
"""

CLEAN_ENV_TESTS = """\
import os


def test_no_hook_variables():
    for name in ("PYTHONPATH", "PYTHONSTARTUP", "PYTEST_PLUGINS"):
        assert os.environ.get(name, "") == "", name


def test_pythonpath_is_empty():
    assert os.environ.get("PYTHONPATH") in (None, "")
"""

HOSTILE_PLUGIN = (
    "def pytest_collection_modifyitems(items):\n"
    "    for item in items:\n"
    "        item.obj = lambda: None\n"
)

# Verifier shapes where every pytest runs in an interpreter that is on the
# verifier PATH at hardening, so the guard loads from its copy.
CHECK_THEN_PYTEST = """\
#!/bin/bash
if ! /usr/local/bin/python3 -I -S /verifier/check_env.py /logs/verifier; then
    echo 0 > /logs/verifier/reward.txt
    exit 0
fi
/usr/local/bin/python3 -I -m pytest /verifier/test_outputs.py -rA
"""
PYTEST_ENV_RESET = """\
#!/bin/bash
python3 -I -S /verifier/check_env.py /logs/verifier || exit 0
env -u PYTHONPATH -u PYTHONSTARTUP PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \\
    python3 -I -m pytest -p no:cacheprovider /verifier/test_outputs.py -rA
"""
BARE_PYTHON3 = """\
#!/bin/bash
python3 -I -m pytest -p no:cacheprovider /verifier/test_outputs.py -rA
"""
GRADE_UNDER_ENV_I = """\
#!/bin/bash
env -i PATH=/usr/bin:/bin HOME=/tmp python3 -I /tests/grade.py
"""
# pytest runs in a Python that uvx creates after hardening.
UVX = """\
#!/bin/bash
uvx --python 3.12 --with pytest pytest /tests/test_outputs.py -rA
"""


def _task(tmp_path: Path, scripts: dict[str, str]) -> MagicMock:
    verifier_dir = tmp_path / "task" / "verifier"
    verifier_dir.mkdir(parents=True)
    for name, text in scripts.items():
        (verifier_dir / name).write_text(text)
    task = MagicMock()
    task.task_dir = verifier_dir.parent
    task.paths.tests_dir = verifier_dir
    task.paths.uses_native_verifier_dir = True
    task.config.verifier.env = None
    task.config.verifier.user = None
    task.config.verifier.pytest_plugins = []
    task.config.verifier.service = "main"
    return task


def _sandbox(*, guarded=("/usr/local/bin/python3",), image_pythonpath=""):
    """Hardening's view of an image; ``guarded`` Pythons take a guard copy."""

    def execute(command, **kwargs):
        stdout = ""
        if command == "printenv PATH":
            stdout = "/usr/local/bin:/usr/bin:/bin\n"
        elif command.startswith("printenv PYTHONPATH"):
            stdout = image_pythonpath
        elif "install_guard" in command:
            stdout = "".join(f"guarded {path}\n" for path in guarded)
        elif "from importlib.metadata import entry_points" in command:
            stdout = json.dumps({"plugins": [], "rejected": []})
        elif command.startswith("python3 -c"):
            entries = image_pythonpath.split(":") if image_pythonpath else []
            stdout = json.dumps(entries if entries and entries[0] in command else [])
        return MagicMock(stdout=stdout, stderr="", exit_code=0)

    env = MagicMock()
    env.exec = AsyncMock(side_effect=execute)
    return env


async def _hardened_env(tmp_path, scripts, **sandbox) -> dict[str, str]:
    task = _task(tmp_path, scripts)
    await harden_before_verify(
        _sandbox(**sandbox), task, sandbox_user="agent", workspace="/root"
    )
    return task.config.verifier.env


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scripts",
    [
        {"test.sh": CHECK_THEN_PYTEST, "check_env.py": CHECK_ENV},
        {"test.sh": PYTEST_ENV_RESET},
        {"test.sh": BARE_PYTHON3, "test_outputs.py": CLEAN_ENV_TESTS},
        {"test.sh": GRADE_UNDER_ENV_I},
        {"test.sh": BARE_PYTHON3 + "# The image ships pytest; no uvx or venv here.\n"},
    ],
    ids=[
        "check-before-pytest",
        "pytest-env-reset",
        "bare-python3",
        "env-i-grade",
        "tools-in-comments",
    ],
)
@pytest.mark.parametrize("image_pythonpath", ["", "/opt/lib"])
async def test_guard_stays_off_pythonpath_when_every_python_has_a_copy(
    tmp_path, scripts, image_pythonpath
):
    """Guards against the PYTHONPATH regression introduced by the pytest plugin guard.

    These verifiers run pytest only in Pythons that took a guard copy, so the
    guard directory on PYTHONPATH loads nothing and an environment check reads it
    as a startup-injection hook. PYTHONPATH must be what it was before the guard:
    the image's trusted entries, empty when the image defines none.
    """
    env = await _hardened_env(tmp_path, scripts, image_pythonpath=image_pythonpath)

    assert env["PYTHONPATH"] == image_pythonpath
    guard = pytest_plugin_guard_name(env["PYTEST_ADDOPTS"])
    assert guard is not None and guard.startswith("_benchflow_guard_")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scripts",
    [
        {"test.sh": UVX},
        {"test.sh": "#!/bin/bash\ncd /app && uv run pytest tests/ -rA\n"},
        {
            "test.sh": (
                "#!/bin/bash\npython3 -m venv /tmp/v\n"
                "/tmp/v/bin/pip install pytest\n/tmp/v/bin/pytest /tests\n"
            )
        },
        {"test.sh": "#!/bin/bash\n/opt/venv/bin/python -m pytest /tests -rA\n"},
        {"test.sh": '#!/bin/bash\nexport PATH="/opt/venv/bin:$PATH"\npytest /tests\n'},
        {"test.sh": '#!/bin/bash\n"$WORKDIR/.venv/bin/pytest" /tests -rA\n'},
        {
            "test.sh": "#!/bin/bash\napt-get install -y python3.11\npython3.11 -m pytest\n"
        },
        {
            "test.sh": "#!/bin/bash\npython3 -m pytest /tests/test_setup.py\n",
            "testing_utils.py": 'cmd = ["uv", "run", "--with", "demo-package", "pytest"]\n',
        },
    ],
    ids=[
        "uvx",
        "uv-run",
        "fresh-venv",
        "venv-off-path",
        "venv-put-on-path",
        "relative-venv",
        "distro-python",
        "uv-from-python",
    ],
)
@pytest.mark.parametrize("image_pythonpath", ["", "/opt/lib"])
async def test_guard_stays_on_pythonpath_for_pythons_created_later(
    tmp_path, scripts, image_pythonpath
):
    """Guards the pytest-guard PYTHONPATH fallback for Pythons hardening cannot reach.

    A Python that uvx, ``uv run``, a venv or a package manager provides, or one
    off the verifier PATH, has no guard copy; without the PYTHONPATH entry its
    pytest aborts on ``-p <guard>`` and the run is a verifier error.
    """
    env = await _hardened_env(tmp_path, scripts, image_pythonpath=image_pythonpath)

    guard = pytest_plugin_guard_name(env["PYTEST_ADDOPTS"])
    assert guard is not None
    assert env["PYTHONPATH"] == ":".join(filter(None, ("/" + guard, image_pythonpath)))


@pytest.mark.asyncio
async def test_guard_stays_on_pythonpath_when_no_python_took_a_copy(tmp_path):
    """Guards the pytest-guard PYTHONPATH fallback for images without Python 3.

    Any pytest such a verifier runs comes from a Python installed after
    hardening, which only the PYTHONPATH entry reaches.
    """
    env = await _hardened_env(
        tmp_path, {"test.sh": BARE_PYTHON3}, guarded=(), image_pythonpath=""
    )

    guard = pytest_plugin_guard_name(env["PYTEST_ADDOPTS"])
    assert env["PYTHONPATH"] == "/" + guard


@pytest.mark.asyncio
async def test_guard_stays_on_pythonpath_when_verifier_scripts_are_unknown():
    """Guards the pytest-guard PYTHONPATH fallback when scripts cannot be read.

    Only scripts BenchFlow has read can show that every pytest has a guard
    copy; without them the entry stays.
    """
    task = MagicMock()
    task.task_dir = None
    task.paths.uses_native_verifier_dir = False
    task.config.verifier.env = None
    task.config.verifier.pytest_plugins = []
    task.config.verifier.service = "main"
    await harden_before_verify(_sandbox(), task, sandbox_user=None)

    env = task.config.verifier.env
    guard = pytest_plugin_guard_name(env["PYTEST_ADDOPTS"])
    assert env["PYTHONPATH"] == "/" + guard


# Real guard and interpreters, synthetic environment checks.


class _GuardInstallEnv:
    """Hardening's probes answered for an image whose PATH is one bin dir.

    Only the guard installer runs, with the real POSIX shell, so it writes into
    the test's own interpreter and never into a shared Python.
    """

    def __init__(self, bin_dir: Path) -> None:
        self.bin_dir = bin_dir

    async def exec(self, command, **kwargs):
        stdout = ""
        if "install_guard" in command:
            result = subprocess.run(
                ["/bin/sh", "-c", command],
                env={"PATH": "/usr/bin:/bin"},
                capture_output=True,
                text=True,
                timeout=120,
            )
            return SimpleNamespace(
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        if command == "printenv PATH":
            stdout = f"{self.bin_dir}\n"
        elif "from importlib.metadata import entry_points" in command:
            stdout = json.dumps({"plugins": [], "rejected": []})
        elif command.startswith("python3 -c") and str(self.bin_dir) in command:
            stdout = json.dumps([str(self.bin_dir)])
        return SimpleNamespace(return_code=0, stdout=stdout, stderr="")


def _venv(path: Path) -> SimpleNamespace:
    """A venv whose ``-I`` interpreter imports this environment's pytest."""
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(path)],
        check=True,
        capture_output=True,
        timeout=120,
    )
    python = path / "bin" / "python3"
    purelib = Path(
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    )
    (purelib / "outer_pytest.pth").write_text(
        str(Path(pytest.__file__).resolve().parent.parent) + "\n"
    )
    return SimpleNamespace(bin=path / "bin", python=python, purelib=purelib)


def _run(python, *args, env, cwd, isolated=False):
    return subprocess.run(
        [str(python), *(["-I"] if isolated else []), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.asyncio
async def test_environment_checks_pass_and_guard_still_refuses_planted_plugins(
    tmp_path, monkeypatch
):
    """Guards against the PYTHONPATH regression introduced by the pytest plugin guard.

    With the verifier environment hardening builds, the environment check
    (``python3 -I -S``) and the in-pytest PYTHONPATH checks pass, while
    ``python3 -I -m pytest`` still loads the guard from its copy and refuses a
    plugin planted in site-packages.
    """
    image = _venv(tmp_path / "image")
    monkeypatch.setattr(lockdown, "_PYTEST_PLUGIN_GUARD_PARENT", str(tmp_path))
    monkeypatch.setattr(lockdown, "_SAFE_VERIFIER_PATH_PARTS", ())
    workspace = tmp_path / "root"
    workspace.mkdir()
    logs = tmp_path / "logs-verifier"
    logs.mkdir()
    task = _task(
        tmp_path,
        {
            # This image's only PATH entry is the venv's bin dir.
            "test.sh": CHECK_THEN_PYTEST.replace("/usr/local/bin/python3", "python3"),
            "check_env.py": CHECK_ENV,
            "test_outputs.py": CLEAN_ENV_TESTS,
            "test_fail.py": "def test_fail():\n    assert False\n",
        },
    )
    verifier_dir = task.paths.tests_dir
    (image.purelib / "planted_plugin.py").write_text(HOSTILE_PLUGIN)

    env = await _build_verifier_env(
        _GuardInstallEnv(image.bin), task, "agent", str(workspace)
    )
    env["PYTEST_ADDOPTS"] = env["PYTEST_ADDOPTS"].replace(
        "--confcutdir=/verifier", f"--confcutdir={verifier_dir}"
    )
    guard = pytest_plugin_guard_name(env["PYTEST_ADDOPTS"])
    assert guard is not None

    env_check = _run(
        image.python,
        "-S",
        str(verifier_dir / "check_env.py"),
        str(logs),
        env=env,
        cwd=workspace,
        isolated=True,
    )
    assert env_check.returncode == 0, env_check.stdout + env_check.stderr

    checks = _run(
        image.python,
        "-m",
        "pytest",
        str(verifier_dir / "test_outputs.py"),
        env=env,
        cwd=workspace,
        isolated=True,
    )
    assert checks.returncode == 0, checks.stdout + checks.stderr

    for addopts in (env["PYTEST_ADDOPTS"], "-p no:cacheprovider"):
        planted = _run(
            image.python,
            "-m",
            "pytest",
            "-p",
            "planted_plugin",
            str(verifier_dir / "test_fail.py"),
            env={**env, "PYTEST_ADDOPTS": addopts},
            cwd=workspace,
            isolated=True,
        )
        if guard in addopts:
            assert planted.returncode != 0
            assert "Verifier plugin trust rejected: planted_plugin" in planted.stderr, (
                planted.stdout + planted.stderr
            )
        else:
            # Unguarded, the planted plugin really does turn a failure into a pass.
            assert planted.returncode == 0, planted.stdout + planted.stderr


@pytest.mark.asyncio
async def test_python_created_after_hardening_still_loads_the_guard(
    tmp_path, monkeypatch
):
    """Guards the pytest-guard PYTHONPATH fallback against the regression fix.

    A verifier that makes its own Python (uvx here, a venv made after
    hardening in the test) keeps the guard directory on PYTHONPATH, so its
    pytest imports the guard and still refuses a planted plugin.
    """
    image = _venv(tmp_path / "image")
    monkeypatch.setattr(lockdown, "_PYTEST_PLUGIN_GUARD_PARENT", str(tmp_path))
    monkeypatch.setattr(lockdown, "_SAFE_VERIFIER_PATH_PARTS", ())
    workspace = tmp_path / "root"
    workspace.mkdir()
    task = _task(
        tmp_path,
        {
            "test.sh": UVX,
            "test_pass.py": "def test_pass():\n    assert True\n",
            "test_fail.py": "def test_fail():\n    assert False\n",
        },
    )
    verifier_dir = task.paths.tests_dir

    env = await _build_verifier_env(
        _GuardInstallEnv(image.bin), task, "agent", str(workspace)
    )
    env["PYTEST_ADDOPTS"] = env["PYTEST_ADDOPTS"].replace(
        "--confcutdir=/verifier", f"--confcutdir={verifier_dir}"
    )
    late = _venv(tmp_path / "uvx-env")
    (late.purelib / "planted_plugin.py").write_text(HOSTILE_PLUGIN)

    passing = _run(
        late.python,
        "-m",
        "pytest",
        str(verifier_dir / "test_pass.py"),
        env=env,
        cwd=workspace,
    )
    assert passing.returncode == 0, passing.stdout + passing.stderr
    planted = _run(
        late.python,
        "-m",
        "pytest",
        "-p",
        "planted_plugin",
        str(verifier_dir / "test_fail.py"),
        env=env,
        cwd=workspace,
    )
    assert planted.returncode != 0
    assert "Verifier plugin trust rejected: planted_plugin" in planted.stderr


# The start receipt (solver-evidence preservation) must not be in the verifier output directory.


class _MappedSandbox:
    """A mounted sandbox whose commands run in the real shell.

    Sandbox paths in commands and uploaded scripts are mapped into the test's
    directories in one pass, so ``/logs/verifier`` is the rollout's own
    verifier directory, as on a bind-mounted Docker sandbox.
    """

    is_mounted = True

    def __init__(self, mapping: dict[str, Path]) -> None:
        self._mapping = {key: str(value) for key, value in mapping.items()}
        self._pattern = re.compile(
            "|".join(re.escape(key) for key in sorted(mapping, key=len, reverse=True))
        )
        self.commands: list[str] = []

    def _map(self, text: str) -> str:
        return self._pattern.sub(lambda match: self._mapping[match.group(0)], text)

    async def upload_dir(self, source_dir, target_dir, service="main"):
        target = Path(self._map(str(target_dir)))
        for source in Path(source_dir).rglob("*"):
            if source.is_file():
                destination = target / source.relative_to(source_dir)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(self._map(source.read_text()))

    async def download_dir(self, *args, **kwargs):
        raise AssertionError("mounted verifier outputs are never downloaded")

    async def exec(self, command, env=None, **kwargs):
        self.commands.append(command)
        result = subprocess.run(
            ["/bin/sh", "-c", self._map(command)],
            env={"PATH": "/usr/bin:/bin", **(env or {})},
            capture_output=True,
            text=True,
            timeout=120,
        )
        return ExecResult(
            stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
        )


def _receipt_task(tmp_path: Path) -> MagicMock:
    tests_dir = tmp_path / "task" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "check_env.py").write_text(CHECK_ENV)
    (tests_dir / "test.sh").write_text(
        "#!/bin/sh\n"
        f"if {sys.executable} -I -S "
        "/tests/check_env.py /logs/verifier; then\n"
        "    echo 1 > /logs/verifier/reward.txt\n"
        "else\n"
        "    echo 0 > /logs/verifier/reward.txt\n"
        "fi\n"
    )
    task = MagicMock()
    task.task_dir = tests_dir.parent
    task.paths.task_dir = tests_dir.parent
    task.paths.tests_dir = tests_dir
    task.paths.test_path = tests_dir / "test.sh"
    task.paths.uses_native_verifier_dir = False
    task.config = TaskConfig.model_validate_toml(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
    )
    return task


def _receipt_sandbox(tmp_path: Path, rollout_paths: RolloutPaths) -> _MappedSandbox:
    return _MappedSandbox(
        {
            "/logs/verifier": rollout_paths.verifier_dir,
            "/tests": tmp_path / "sandbox-tests",
            "/run/benchflow": tmp_path / "sandbox-run-benchflow",
        }
    )


@pytest.mark.asyncio
async def test_verifier_output_dir_is_empty_when_test_sh_starts(tmp_path):
    """Guards against the start-receipt regression introduced by solver-evidence preservation.

    A check that lists /logs/verifier before scoring would find a receipt
    written there and score a correct solution 0.
    """
    task = _receipt_task(tmp_path)
    rollout_paths = RolloutPaths(tmp_path / "rollout")
    rollout_paths.mkdir()
    sandbox = _receipt_sandbox(tmp_path, rollout_paths)

    verifier = Verifier(task, rollout_paths, sandbox)
    result = await verifier.verify()

    assert result.rewards == {"reward": 1.0}, rollout_paths.test_stdout_path.read_text()
    assert verifier.execution_receipt is None
    assert not any("started" in command for command in sandbox.commands)


@pytest.mark.asyncio
async def test_recovery_start_receipt_lives_outside_the_output_dir(tmp_path):
    """Guards #1136's start receipt against the solver-evidence preservation output-dir placement.

    On the recovery path the command still proves it started, and
    ``_verifier_started`` still reads that proof, but the task's environment check
    sees only what test.sh itself wrote.
    """
    task = _receipt_task(tmp_path)
    rollout_paths = RolloutPaths(tmp_path / "rollout")
    rollout_paths.mkdir()
    sandbox = _receipt_sandbox(tmp_path, rollout_paths)

    verifier = Verifier(task, rollout_paths, sandbox, execution_receipt=True)
    result = await verifier.verify()

    assert result.rewards == {"reward": 1.0}, rollout_paths.test_stdout_path.read_text()
    assert verifier.execution_receipt is not None
    receipt, service = verifier.execution_receipt
    assert service == "main"
    assert receipt.startswith("/run/benchflow/")
    assert not receipt.startswith("/logs/")
    assert sorted(p.name for p in rollout_paths.verifier_dir.iterdir()) == [
        "reward.txt",
        "test-stdout.txt",
    ]
    assert await _verifier_started(sandbox, verifier) is True


@pytest.mark.asyncio
async def test_unwritable_receipt_dir_leaves_start_unknown(tmp_path):
    """Guards #1136: a receipt BenchFlow cannot place is unknown, not a wedge."""
    task = _receipt_task(tmp_path)
    rollout_paths = RolloutPaths(tmp_path / "rollout")
    rollout_paths.mkdir()
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("")
    sandbox = _MappedSandbox(
        {
            "/logs/verifier": rollout_paths.verifier_dir,
            "/tests": tmp_path / "sandbox-tests",
            "/run/benchflow": blocker / "run",
        }
    )

    verifier = Verifier(task, rollout_paths, sandbox, execution_receipt=True)
    result = await verifier.verify()

    assert result.rewards == {"reward": 1.0}
    assert verifier.execution_receipt is None
    assert await _verifier_started(sandbox, verifier) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("recovery_eligible", [False, True])
async def test_every_verifier_asks_for_a_start_receipt(tmp_path, recovery_eligible):
    """#1136: every script verifier's command writes a start receipt, so a
    command the exec layer lost is found in 30 s instead of a whole budget
    (tests/test_verifier_start_receipts.py). The regression solver-evidence
    preservation introduced, a receipt in ``/logs/verifier``, stays guarded
    by test_recovery_start_receipt_lives_outside_the_output_dir.
    """
    requested: dict = {}

    def verifier(**kwargs):
        requested.update(kwargs)
        return SimpleNamespace(
            verify=AsyncMock(return_value=SimpleNamespace(rewards={"reward": 1.0}))
        )

    planes = SimpleNamespace(harden_before_verify=AsyncMock(), verifier=verifier)
    task = SimpleNamespace(
        name="receipt-task",
        config=SimpleNamespace(
            verifier=SimpleNamespace(timeout_sec=5, reward_range=None, env={})
        ),
    )
    rewards, error, _ = await _verify_rollout(
        SimpleNamespace(exec=AsyncMock()),
        task,
        SimpleNamespace(verifier_dir=tmp_path / "verifier"),
        {},
        planes,
        recovery_eligible=recovery_eligible,
    )

    assert error is None and rewards == {"reward": 1.0}
    assert requested["execution_receipt"] is True
