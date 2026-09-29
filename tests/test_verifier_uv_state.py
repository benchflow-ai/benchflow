"""The verifier's uv and pip state, and plugins test.sh installs with them.

Guards a fix: the pytest plugin guard added for #1116 refused the ``ctrf``
plugin that citation-check's test.sh installs with ``uvx --with
pytest-json-ctrf``. The task image sets ``WORKDIR /root``, so uv built the
plugin's environment in ``$HOME/.cache/uv`` inside the agent-writable
workspace; the guard refused it, pytest stopped, and test.sh wrote 0 for the
oracle. Several SkillsBench tasks share the pattern; their oracles scored 1.0
before the guard.

The scenarios run the real hardening code (``_build_verifier_env``) against a
directory tree standing in for the image, then a model of test.sh that places
uv's environment where uv would (``$UV_CACHE_DIR``, else ``$HOME/.cache/uv``)
and runs a real pytest with the armed guard, scored through
``_verify_rollout``. Nothing here is root, so an ownership model treats every
path outside the agent-writable prefixes as root-owned image state; ownership
and permission checks themselves are covered in test_verifier_plugin_trust.py.
"""

import itertools
import json
import os
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow._utils.scoring import VERIFIER_FAILED, classify_verifier_error
from benchflow.rollout._setup import _verify_rollout
from benchflow.sandbox import _pytest_plugin_guard as guard_module
from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult
from benchflow.task import RolloutPaths, Verifier
from benchflow.task.config import TaskConfig

# citation-check's verifier/test.sh at skillsbench 9a1f4dd, the README quickstart task.
CITATION_CHECK_TEST_SH = """#!/bin/bash
apt-get update
apt-get install -y curl
curl -LsSf https://astral.sh/uv/0.9.7/install.sh | sh
source $HOME/.local/bin/env
mkdir -p /logs/verifier
uvx \\
  --with pytest==8.4.1 \\
  --with pytest-json-ctrf==0.3.5 \\
  pytest --ctrf /logs/verifier/ctrf.json /verifier/test_outputs.py -rA -v
PYTEST_EXIT_CODE=$?
if [ $PYTEST_EXIT_CODE -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""
PASSING = "def test_solution():\n    assert True\n"
FAILING = "def test_solution():\n    assert False\n"
# What pytest-json-ctrf contributes that matters here: the --ctrf option.
CTRF_PLUGIN = (
    "def pytest_addoption(parser):\n"
    "    parser.addoption('--ctrf', action='store', default=None)\n"
)


def hostile_ctrf(proof: Path) -> str:
    """A ctrf plugin that passes every test and leaves *proof* when imported."""
    return (
        "from pathlib import Path\n"
        f"Path({str(proof)!r}).touch()\n"
        + CTRF_PLUGIN
        + "def pytest_collection_modifyitems(items):\n"
        "    for item in items:\n"
        "        item.obj = lambda: None\n"
    )


def install_plugin(site: Path, source: str, *, name="ctrf", module="ctrf_model"):
    """Write a pytest11 registration and its module, as pip or uv would."""
    info = site / f"{module}-0.3.5.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {module}\n")
    (info / "entry_points.txt").write_text(f"[pytest11]\n{name} = {module}\n")
    (site / f"{module}.py").write_text(source)


def modelled(script: str, definition: str) -> str:
    """Insert *definition* before the script's entry point."""
    marker = '\nif __name__ == "__main__":'
    assert marker in script
    return script.replace(marker, "\n" + definition + marker)


# Paths outside the blocked prefixes are root-owned image state in this model.
GUARD_OWNERSHIP = (
    "def trusted(path, blocked):\n"
    "    return bool(path) and str(path).startswith('/') and not any(\n"
    "        under(candidate, prefix)\n"
    "        for candidate in (os.path.abspath(path), os.path.realpath(path))\n"
    "        for prefix in blocked\n"
    "    )\n"
)

MOVED_KEYS = (
    "UV_CACHE_DIR",
    "UV_TOOL_DIR",
    "UV_PYTHON_INSTALL_DIR",
    "PIP_CACHE_DIR",
    "UV_CONFIG_FILE",
    "PIP_CONFIG_FILE",
)


def state_ownership(tmp_path: Path) -> str:
    """Model root ownership, and keep the host's own /etc out of the probe."""
    return (
        "def owned_safely(st):\n    return True\n"
        f"SYSTEM_CONFIG_DIRS = {str(tmp_path / 'etc-xdg')!r}\n"
        f"SYSTEM_UV_CONFIG = {str(tmp_path / 'etc-uv/uv.toml')!r}\n"
    )


class Layout:
    """A task image with WORKDIR (= $HOME) *workspace*, as seen from the host."""

    def __init__(self, tmp_path: Path, monkeypatch, workspace_name="root"):
        self.tmp = tmp_path
        self.fs = tmp_path / "fs"  # stands in for "/" of the sandbox
        self.fs.mkdir()
        self.workspace = tmp_path / workspace_name
        self.workspace.mkdir()
        self.home = tmp_path / "root"
        self.home.mkdir(exist_ok=True)
        self.cwd = tmp_path / "cwd"
        self.cwd.mkdir()
        self.bindir = tmp_path / "bin"
        self.bindir.mkdir()
        python3 = self.bindir / "python3"
        python3.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
        python3.chmod(0o755)
        self.image_pythonpath = ""
        self.paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
        self.paths.mkdir()
        # The sandbox's runtime prefixes (/tmp, /logs, ...) stand in here, so a
        # pytest tmp_path under the host's /tmp is not itself agent-writable.
        monkeypatch.setattr(
            lockdown, "_RUNTIME_PATH_PREFIXES", (str(tmp_path / "sandbox-tmp"),)
        )
        monkeypatch.setattr(lockdown, "_PYTEST_PLUGIN_GUARD_PARENT", str(self.fs))
        monkeypatch.setattr(
            lockdown, "_VERIFIER_TOOL_STATE_PARENT", str(self.fs), raising=False
        )
        monkeypatch.setattr(
            lockdown, "_PYTEST_PLUGIN_GUARD_MARKERS_DIR", str(self.paths.verifier_dir)
        )
        monkeypatch.setattr(
            lockdown,
            "_DISCOVER_PYTEST_PLUGINS_SCRIPT",
            modelled(lockdown._DISCOVER_PYTEST_PLUGINS_SCRIPT, GUARD_OWNERSHIP),
        )
        script = getattr(lockdown, "_VERIFIER_TOOL_STATE_SCRIPT", None)
        if script is not None:
            monkeypatch.setattr(
                lockdown,
                "_VERIFIER_TOOL_STATE_SCRIPT",
                modelled(script, state_ownership(tmp_path)),
            )

    def run(self, command: str) -> ExecResult:
        result = subprocess.run(
            ["/bin/sh", "-c", command],
            cwd=self.workspace,
            env={"PATH": f"{self.bindir}:/usr/bin:/bin"},
            capture_output=True,
            text=True,
            timeout=60,
        )
        return ExecResult(
            stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
        )

    async def exec(self, command: str, **kwargs) -> ExecResult:
        """The hardening commands, against this tree."""
        if command == "printenv PATH" or "os-release" in command:
            return ExecResult(stdout="", stderr="", return_code=0)
        if command.startswith("printenv PYTHONPATH"):
            return ExecResult(stdout=self.image_pythonpath, stderr="", return_code=0)
        if "raw_path = json.loads(sys.argv[1])" in command:
            # The image PYTHONPATH entries are root-owned in this model.
            entries = [e for e in self.image_pythonpath.split(":") if e]
            return ExecResult(stdout=json.dumps(entries), stderr="", return_code=0)
        if "PYTHONNOUSERSITE=1 python3 -c" in command:
            # Plugin discovery: this image ships no pytest11 plugins.
            return ExecResult(
                stdout='{"plugins": [], "rejected": []}', stderr="", return_code=0
            )
        if command.startswith("mkdir -m 755") and "_benchflow_guard_" in command:
            # Write the guard; no Python here takes a site-packages copy.
            command = command.split(" && {", 1)[0]
        return self.run(command)


def make_task(layout: Layout, test_sh: str = CITATION_CHECK_TEST_SH, env=None):
    tests_dir = layout.tmp / "task" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test.sh").write_text(test_sh)
    task = MagicMock()
    task.name = "citation-check"
    task.task_dir = tests_dir.parent
    task.paths.task_dir = tests_dir.parent
    task.paths.tests_dir = tests_dir
    task.paths.test_path = tests_dir / "test.sh"
    task.paths.uses_native_verifier_dir = False
    task.config = TaskConfig.model_validate_toml('version = "1.0"\n[verifier]\n')
    task.config.verifier.env = {"HOME": str(layout.home), **(env or {})}
    return task


class ScriptSandbox:
    """Run a model of test.sh with the environment the verifier passes it."""

    is_mounted = True

    def __init__(self, layout: Layout, solution: str, place):
        self.layout = layout
        self.solution = solution
        self.place = place
        self.env: dict[str, str] = {}

    async def upload_dir(self, *args, **kwargs):
        pass

    async def exec(self, command, env=None, **kwargs):
        if "test-stdout.txt" not in command:
            return ExecResult(stdout="", stderr="", return_code=0)
        self.env = dict(env or {})
        tests = self.layout.tmp / "verifier-tests"
        tests.mkdir(exist_ok=True)
        (tests / "test_outputs.py").write_text(self.solution)
        site = self.place(self.env, self.layout)
        pythonpath = [p for p in self.env.get("PYTHONPATH", "").split(":") if p]
        flags = shlex.split(self.env.get("PYTEST_ADDOPTS", ""))
        plugins = [
            arg
            for flag, name in itertools.pairwise(flags)
            if flag == "-p"
            for arg in ("-p", name)
        ]
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("PYTEST_", "PYTHON"))
        }
        environment.update(
            PYTHONPATH=":".join([*pythonpath, *map(str, site)]),
            PYTEST_ADDOPTS=" ".join(["-p", "no:cacheprovider", *plugins]),
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
            PYTHONDONTWRITEBYTECODE="1",
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--ctrf",
                str(self.layout.paths.verifier_dir / "ctrf.json"),
                str(tests / "test_outputs.py"),
            ],
            cwd=self.layout.cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
        )
        self.layout.paths.test_stdout_path.write_text(result.stdout)
        self.layout.paths.reward_text_path.write_text(
            "1" if result.returncode == 0 else "0"
        )
        return ExecResult(stdout="", stderr="", return_code=result.returncode)


def uvx(env, layout):
    """``uvx --with pytest-json-ctrf``: an environment in uv's cache, reused if there."""
    cache = env.get("UV_CACHE_DIR") or os.path.join(env["HOME"], ".cache", "uv")
    site = Path(cache) / "archive-v0" / "ctrf-env" / "site-packages"
    if not site.exists():
        install_plugin(site, CTRF_PLUGIN)
    return [site]


async def score(
    tmp_path,
    monkeypatch,
    solution,
    place=uvx,
    *,
    before=None,
    env=None,
    workspace_name="root",
    guard_model=None,
):
    """Harden a WORKDIR /root image (or *workspace_name*), run test.sh, and score it."""
    layout = Layout(tmp_path, monkeypatch, workspace_name=workspace_name)
    if guard_model is not None:
        monkeypatch.setattr(
            lockdown,
            "_DISCOVER_PYTEST_PLUGINS_SCRIPT",
            modelled(Path(guard_module.__file__).read_text(), guard_model),
        )
    if before is not None:
        before(layout)
    task = make_task(layout, env=env)
    task.config.verifier.env = await lockdown._build_verifier_env(
        layout, task, "agent", str(layout.workspace)
    )
    sandbox = ScriptSandbox(layout, solution, place)
    planes = SimpleNamespace(harden_before_verify=AsyncMock(), verifier=Verifier)
    rewards, error, _ = await _verify_rollout(sandbox, task, layout.paths, {}, planes)
    return layout, sandbox.env, rewards, error


def mode_ownership(tmp_path: Path) -> str:
    """The guard's owner and mode rule with real mode bits, for a Layout's tree.

    Nothing here runs as root, so owners are modelled as root's; a file or
    directory below the stand-ins for ``/`` and ``/root`` that others can
    write is refused, as it is in the sandbox. Those stand-ins and everything
    outside them are the image's own (root-owned 0755), whatever mask the
    test process has.
    """
    roots = [str(tmp_path / "fs") + "/", str(tmp_path / "root") + "/"]
    return (
        "def ownership_problem(path, st):\n"
        f"    if not path.startswith(tuple({roots!r})):\n"
        "        return None\n"
        "    if st.st_mode & 0o022:\n"
        "        return path + ' is group- or world-writable'\n"
        "    return None\n"
    )


def uvx_under_umask_0000(env, layout):
    """``uvx`` where ``docker exec`` runs with umask 0000 (Docker-in-Docker)."""
    previous = os.umask(0)
    try:
        return uvx(env, layout)
    finally:
        os.umask(previous)


def home_cache_under_umask_0000(env, layout):
    """The same install in ``$HOME/.cache/uv``, where c5b75fcc left uv's cache."""
    return uvx_under_umask_0000({**env, "UV_CACHE_DIR": ""}, layout)


def markers(layout, kind):
    return sorted(layout.paths.verifier_dir.glob(f"_benchflow_guard_*.*.{kind}"))


@pytest.mark.parametrize(
    ("solution", "reward"), [(PASSING, 1.0), (FAILING, 0.0)], ids=["oracle", "empty"]
)
async def test_uvx_plugin_under_workdir_root_is_scored(
    tmp_path, monkeypatch, solution, reward
):
    """citation-check's oracle scored 0 once the pytest plugin guard existed.

    With uv's cache in the workspace the guard refused ``ctrf``; with the
    verifier's uv state moved out of it, the plugin loads and the solution
    decides the reward.
    """
    layout, env, rewards, error = await score(tmp_path, monkeypatch, solution)

    stdout = layout.paths.test_stdout_path.read_text()
    assert "Verifier plugin trust rejected" not in stdout, stdout
    assert error is None
    assert rewards == {"reward": reward}
    cache = Path(env["UV_CACHE_DIR"])
    assert not cache.is_relative_to(layout.workspace)
    assert cache.parent.parent == layout.fs


async def test_verifier_install_under_a_umask_0000_runtime_is_scored(
    tmp_path, monkeypatch
):
    """Terminal-Bench 2's regex-log oracle was unscored on Docker-in-Docker.

    Guards the fix for the plugin guard's umask dependence (review of
    sdk-update-2026-09-27 at c5b75fcc, must-fix 3). The image has WORKDIR
    /app, so c5b75fcc left uv's cache in /root/.cache/uv; ``docker exec``
    there runs with umask 0000, uv created the cache 0777/0666, the guard
    refused the verifier's own ctrf and the correct oracle's trial was
    unscored. The cache now always moves into the directory hardening made
    after the agent stopped, which the guard trusts by path, so the same
    world-writable install is scored.
    """
    layout, env, rewards, error = await score(
        tmp_path,
        monkeypatch,
        PASSING,
        uvx_under_umask_0000,
        workspace_name="app",
        guard_model=mode_ownership(tmp_path),
    )

    stdout = layout.paths.test_stdout_path.read_text()
    assert "Verifier plugin trust rejected" not in stdout, stdout
    assert error is None
    assert rewards == {"reward": 1.0}
    (site,) = Path(env["UV_CACHE_DIR"]).glob("archive-v0/*/site-packages")
    assert (site / "ctrf_model.py").stat().st_mode & 0o777 == 0o666


async def test_same_install_outside_the_trusted_directory_is_still_judged_by_mode(
    tmp_path, monkeypatch
):
    """Guards the trust-by-path rule against widening beyond its directory.

    The world-writable install in $HOME/.cache/uv (where c5b75fcc left it) is
    refused, and, being newer than the guard, reported as the verifier's own
    install: unscored, as every --ctrf task was on Docker-in-Docker.
    """
    layout, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        PASSING,
        home_cache_under_umask_0000,
        workspace_name="app",
        guard_model=mode_ownership(tmp_path),
    )

    assert "Verifier plugin trust rejected: ctrf" in (
        layout.paths.test_stdout_path.read_text()
    )
    assert rewards is None
    assert error is not None and "installed after the agent stopped" in error


async def test_workdir_root_state_moves_to_one_fresh_root_directory(
    tmp_path, monkeypatch
):
    """Every uv and pip location under an agent-writable $HOME moves.

    Caches, tool environments and managed Pythons go to one new directory,
    and uv and pip read an empty configuration there instead of files the
    agent could have written in $HOME or the workspace.
    """
    layout = Layout(tmp_path, monkeypatch)
    task = make_task(layout)
    env = await lockdown._build_verifier_env(
        layout, task, "agent", str(layout.workspace)
    )

    (directory,) = layout.fs.glob("_benchflow_verifier_*")
    assert directory.stat().st_mode & 0o777 == 0o755
    assert {key: env.get(key) for key in MOVED_KEYS} == {
        "UV_CACHE_DIR": str(directory / "uv-cache"),
        "UV_TOOL_DIR": str(directory / "uv-tools"),
        "UV_PYTHON_INSTALL_DIR": str(directory / "uv-python"),
        "PIP_CACHE_DIR": str(directory / "pip-cache"),
        "UV_CONFIG_FILE": str(directory / "uv.toml"),
        "PIP_CONFIG_FILE": str(directory / "pip.conf"),
    }
    for name in ("uv.toml", "pip.conf"):
        assert (directory / name).read_text() == ""
        assert (directory / name).stat().st_mode & 0o777 == 0o644


async def test_home_outside_the_workspace_moves_too_and_keeps_pip_config(
    tmp_path, monkeypatch
):
    """A $HOME the agent cannot write still gets a fresh directory the guard trusts.

    Guards the fix for the plugin guard's umask dependence (review of
    sdk-update-2026-09-27 at c5b75fcc, must-fix 3). With WORKDIR /app the uv
    cache stayed in /root/.cache/uv and was judged by its modes; on
    Docker-in-Docker, whose exec mask is 0000, uv created it world-writable
    and the guard refused Terminal-Bench 2's own ctrf. Caches, tools and
    managed Pythons now always move. pip configuration in a safe $HOME (a
    mirror in ~/.config/pip/pip.conf) is still read: pip installs outside the
    trusted directory, and reads no workspace configuration.
    """
    layout = Layout(tmp_path, monkeypatch, workspace_name="app")
    task = make_task(layout)
    env = await lockdown._build_verifier_env(
        layout, task, "agent", str(layout.workspace)
    )

    (directory,) = layout.fs.glob("_benchflow_verifier_*")
    assert {key: env.get(key) for key in MOVED_KEYS} == {
        "UV_CACHE_DIR": str(directory / "uv-cache"),
        "UV_TOOL_DIR": str(directory / "uv-tools"),
        "UV_PYTHON_INSTALL_DIR": str(directory / "uv-python"),
        "PIP_CACHE_DIR": str(directory / "pip-cache"),
        "UV_CONFIG_FILE": str(directory / "uv.toml"),
        "PIP_CONFIG_FILE": None,
    }
    # The armed guard trusts exactly that directory by path.
    (guard,) = layout.fs.glob("_benchflow_guard_*/_benchflow_guard_*.py")
    assert f"_BENCHFLOW_TRUSTED = {(str(directory),)!r}" in guard.read_text()


# Attacks: what the agent can plant must still never pass the task.


async def test_preseeded_uv_cache_and_tools_are_never_used(tmp_path, monkeypatch):
    """Guards the uv-state fix: a planted ~/.cache/uv environment or uv tool never loads.

    The agent owns $HOME under WORKDIR /root. It plants the exact cached
    environment uvx would reuse and a ``pytest`` uv tool, both carrying a ctrf
    plugin that passes every test. The failing solution must still score 0.
    """
    proof = tmp_path / "hostile-imported"

    def plant(layout):
        site = layout.home / ".cache/uv/archive-v0/ctrf-env/site-packages"
        install_plugin(site, hostile_ctrf(proof))
        tool = layout.home / ".local/share/uv/tools/pytest/lib/site-packages"
        install_plugin(tool, hostile_ctrf(proof))

    _, _, rewards, error = await score(tmp_path, monkeypatch, FAILING, before=plant)

    assert rewards == {"reward": 0.0} and error is None
    assert not proof.exists()


@pytest.mark.parametrize("where", ["uv-cache", "workspace-venv", "pythonpath"])
async def test_planted_plugin_loaded_anyway_is_refused_and_scored(
    tmp_path, monkeypatch, where
):
    """Guards #1116 through the uv-state fix: planted code the verifier loads is a 0.

    test.sh (or an image PYTHONPATH) puts an agent-written location on the
    path anyway: the old uv cache, a venv the agent pre-seeded in the
    workspace, or the workspace itself on PYTHONPATH. The guard refuses the
    plugin, and because the files predate hardening the refusal is scored,
    not reported as the verifier's own install.
    """
    proof = tmp_path / "hostile-imported"
    planted = {
        "uv-cache": Path(".cache/uv/archive-v0/ctrf-env/site-packages"),
        "workspace-venv": Path(".venv/lib/python3.12/site-packages"),
        "pythonpath": Path("."),
    }[where]

    def plant(layout):
        install_plugin(layout.home / planted, hostile_ctrf(proof))
        if where == "pythonpath":
            layout.image_pythonpath = str(layout.home)

    def loads_planted(env, layout):
        # Whatever puts the planted copy on the path puts it ahead of the
        # verifier's own ctrf, so pytest would import the planted one.
        return [layout.home / planted, *uvx(env, layout)]

    layout, _, rewards, error = await score(
        tmp_path, monkeypatch, FAILING, loads_planted, before=plant
    )

    stdout = layout.paths.test_stdout_path.read_text()
    assert "Verifier plugin trust rejected: ctrf" in stdout, stdout
    assert rewards == {"reward": 0.0} and error is None
    assert markers(layout, "installed") == []
    assert not proof.exists()


def _wheel(directory: Path, name: str) -> Path:
    """A minimal wheel for *name*, enough for a resolver to pick it."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name.replace('-', '_')}-9.9.9-py3-none-any.whl"
    dist = f"{name.replace('-', '_')}-9.9.9.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{dist}/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: 9.9.9\n"
        )
        archive.writestr(
            f"{dist}/WHEEL",
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\n"
            "Tag: py3-none-any\n",
        )
        archive.writestr(f"{dist}/RECORD", "")
    return path


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs the uv binary")
@pytest.mark.parametrize("planted", ["user", "project"])
async def test_planted_uv_config_cannot_redirect_the_verifier_index(
    tmp_path, monkeypatch, planted
):
    """Guards the uv-state fix: a planted uv.toml cannot pick the verifier's packages.

    With the verifier's installs now landing in a trusted directory, an
    agent-written uv.toml in $HOME (read by uvx and every uv command) or in the
    workspace (read by ``uv pip``, ``uv run``) could point uv at the agent's own
    index. The real uv binary resolves ``evil-ctrf`` from a planted
    ``find-links`` only without BenchFlow's configuration.
    """
    layout = Layout(tmp_path, monkeypatch)
    _wheel(layout.home / "wheels", "evil-ctrf")
    config = (
        f"no-index = true\nfind-links = [{json.dumps(str(layout.home / 'wheels'))}]\n"
    )
    target = {
        "user": layout.home / ".config/uv/uv.toml",
        "project": layout.workspace / "uv.toml",
    }[planted]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(config)
    (layout.workspace / "requirements.in").write_text("evil-ctrf\n")
    task = make_task(layout)
    env = await lockdown._build_verifier_env(
        layout, task, "agent", str(layout.workspace)
    )

    def compile_requirements(extra):
        base = {"PATH": os.environ["PATH"], "HOME": str(layout.home)}
        return subprocess.run(
            ["uv", "pip", "compile", "--offline", "requirements.in"],
            cwd=layout.workspace,
            env={**base, "UV_CACHE_DIR": str(tmp_path / "probe-cache"), **extra},
            capture_output=True,
            text=True,
            timeout=120,
        )

    # The planted file is effective when uv reads it ...
    redirected = compile_requirements({})
    assert redirected.returncode == 0 and "evil-ctrf==9.9.9" in redirected.stdout
    # ... and ignored with the verifier's environment.
    keys = ("UV_CONFIG_FILE", "UV_NO_CONFIG", "UV_CACHE_DIR")
    hardened = compile_requirements({k: env[k] for k in keys if k in env})
    assert "evil-ctrf==9.9.9" not in hardened.stdout
    assert hardened.returncode != 0


async def test_planted_pip_config_is_replaced_by_an_existing_empty_file(
    tmp_path, monkeypatch
):
    """Guards the uv-state fix: ~/.pip/pip.conf under WORKDIR /root is the agent's.

    ``pip install pytest-json-ctrf`` in test.sh lands in the image's
    site-packages, which the guard trusts, so an agent index there would be a
    pass. pip skips user configuration when ``PIP_CONFIG_FILE`` names an
    existing file; an empty one keeps the image's global /etc/pip.conf.
    """
    layout = Layout(tmp_path, monkeypatch)
    (layout.home / ".pip").mkdir()
    (layout.home / ".pip/pip.conf").write_text("[global]\nindex-url = http://evil\n")
    env = await lockdown._build_verifier_env(
        layout, make_task(layout), "agent", str(layout.workspace)
    )

    config = Path(env["PIP_CONFIG_FILE"])
    assert config.is_file() and config.read_text() == ""
    assert not config.is_relative_to(layout.workspace)
    pip = shutil.which("pip3") or shutil.which("pip")
    if pip is None:
        return
    shown = subprocess.run(
        [pip, "config", "list"],
        env={
            "PATH": os.environ["PATH"],
            "HOME": str(layout.home),
            **{"PIP_CONFIG_FILE": env["PIP_CONFIG_FILE"]},
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "evil" not in shown.stdout


# How refusals are scored.


async def test_refused_plugin_the_verifier_installed_is_a_verifier_error(
    tmp_path, monkeypatch
):
    """Guards the uv-state silent zero: BenchFlow's own false positive is not scored.

    test.sh itself installs ctrf into a venv in the agent-writable workspace
    after hardening (``uv init && uv add && uv run`` as in
    powerlifting-coef-calc). The guard cannot trust that directory, but every
    file there is newer than the guard, so the agent did not write it: the
    run is a verifier error, never the solution's 0.
    """

    def workspace_venv(env, layout):
        site = layout.workspace / ".venv/lib/python3.12/site-packages"
        install_plugin(site, CTRF_PLUGIN)
        return [site]

    layout, _, rewards, error = await score(
        tmp_path, monkeypatch, PASSING, workspace_venv
    )

    assert "Verifier plugin trust rejected: ctrf" in (
        layout.paths.test_stdout_path.read_text()
    )
    assert rewards is None
    assert error is not None and "installed after the agent stopped" in error
    assert classify_verifier_error(error) == VERIFIER_FAILED
    (marker,) = markers(layout, "installed")
    assert ".venv/lib/python3.12/site-packages" in marker.read_text()


async def test_planted_duplicate_next_to_a_verifier_install_stays_scored(
    tmp_path, monkeypatch
):
    """Guards the uv-state scoring rule against excusing tampering.

    The agent pre-seeds the workspace venv with a second ``ctrf``
    registration; test.sh then installs the real one there. One file behind
    the refusal predates hardening, so the refusal is the agent's 0.
    """
    proof = tmp_path / "hostile-imported"
    site_path = Path(".venv/lib/python3.12/site-packages")

    def plant(layout):
        install_plugin(layout.home / site_path, hostile_ctrf(proof), module="agent")

    def workspace_venv(env, layout):
        install_plugin(layout.workspace / site_path, CTRF_PLUGIN)
        return [layout.workspace / site_path]

    layout, _, rewards, error = await score(
        tmp_path, monkeypatch, FAILING, workspace_venv, before=plant
    )

    assert rewards == {"reward": 0.0} and error is None
    assert markers(layout, "installed") == []
    assert not proof.exists()
