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

import hashlib
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

from benchflow._utils.trusted_path import PATH_QUERY_CMD, PYTHONPATH_QUERY_CMD
from benchflow.rollout._setup import _verify_rollout
from benchflow.sandbox import _pytest_plugin_guard as guard_module
from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult, with_system_path
from benchflow.sandbox.lockdown import _keep_route_path
from benchflow.task import RolloutPaths, Verifier
from benchflow.task.config import TaskConfig
from tests._receipt_fakes import receipt_aware

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


def install_plugin(
    site: Path,
    source: str,
    *,
    name="ctrf",
    module="ctrf_model",
    entry=None,
    version=None,
):
    """Write a pytest11 registration and its module, as pip or uv would.

    *entry* is the entry point's object reference (default: the module);
    *version* adds a Version field to METADATA."""
    info = site / f"{module}-0.3.5.dist-info"
    info.mkdir(parents=True)
    metadata = f"Metadata-Version: 2.1\nName: {module}\n"
    if version is not None:
        metadata += f"Version: {version}\n"
    (info / "METADATA").write_text(metadata)
    (info / "entry_points.txt").write_text(f"[pytest11]\n{name} = {entry or module}\n")
    (site / f"{module}.py").write_text(source)


def published_table(source=CTRF_PLUGIN, *, name="ctrf", module="ctrf_model"):
    """The release :func:`install_plugin` writes, as a vouched published wheel.

    Stands in for pytest-json-ctrf's PyPI releases: the guard trusts
    these bytes wherever they are installed, and nothing else.
    """
    info = f"{module}-0.3.5.dist-info"
    files = {
        f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {module}\n",
        f"{info}/entry_points.txt": f"[pytest11]\n{name} = {module}\n",
        f"{module}.py": source,
    }
    return {
        info: [
            {
                path: hashlib.sha256(text.encode()).hexdigest()
                for path, text in files.items()
            }
        ]
    }


def modelled(script: str, definition: str) -> str:
    """Insert *definition* before the script's entry point."""
    marker = '\nif __name__ == "__main__":'
    assert marker in script
    return script.replace(marker, "\n" + definition + marker)


# Every path is root's alone in this model: the blocked prefixes, not owners,
# decide (a file the verifier installed there after hardening aside, #1170).
GUARD_OWNERSHIP = "def ownership_problem(path, st):\n    return None\n"

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
        "def owned_link(st):\n    return True\n"
        # The pip entry points the verifier PATH finds: the test's own list
        # (pip-entry-points.json), never the host's.
        "def pip_entry_points(env):\n"
        f"    path = {str(tmp_path / 'pip-entry-points.json')!r}\n"
        "    return json.load(open(path)) if os.path.exists(path) else {}\n"
        f"SYSTEM_CONFIG_DIRS = {str(tmp_path / 'etc-xdg')!r}\n"
        f"SYSTEM_UV_CONFIG = {str(tmp_path / 'etc-uv/uv.toml')!r}\n"
        f"SYSTEM_PIP_CONFIG = {str(tmp_path / 'etc/pip.conf')!r}\n"
    )


class Layout:
    """A task image with WORKDIR (= $HOME) *workspace*, as seen from the host."""

    def __init__(
        self, tmp_path: Path, monkeypatch, workspace_name="root", python=sys.executable
    ):
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
        python3.write_text(f'#!/bin/sh\nexec {shlex.quote(python)} "$@"\n')
        python3.chmod(0o755)
        self.image_pythonpath = ""
        # The image's own environment beyond PATH (XDG_CONFIG_DIRS, ...).
        self.image_env: dict[str, str] = {}
        # What plugin discovery finds in the image's Python.
        self.discovered: list[str] = []
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
            env={"PATH": f"{self.bindir}:/usr/bin:/bin", **self.image_env},
            capture_output=True,
            text=True,
            timeout=60,
        )
        return ExecResult(
            stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
        )

    @receipt_aware
    async def exec(self, command: str, **kwargs) -> ExecResult:
        """The hardening commands, against this tree."""
        # Hardening sets its own PATH first; this model's
        # image keeps its Python in bindir, a folder the PATH check would keep.
        command = command.removeprefix(_keep_route_path("")).removeprefix(
            with_system_path("")
        )
        if (
            PATH_QUERY_CMD in command
            or "os-release" in command
            # An unset PATH in this model, not a set but empty one.
            or "${PATH+set}" in command
        ):
            return ExecResult(stdout="", stderr="", return_code=0)
        if command == "id -u":
            # test.sh runs as the image's user, root in task images.
            return ExecResult(stdout="0\n", stderr="", return_code=0)
        if PYTHONPATH_QUERY_CMD in command:
            return ExecResult(stdout=self.image_pythonpath, stderr="", return_code=0)
        if "raw_path = json.loads(sys.argv[1])" in command:
            # The image PYTHONPATH entries are root-owned in this model.
            entries = [e for e in self.image_pythonpath.split(":") if e]
            return ExecResult(stdout=json.dumps(entries), stderr="", return_code=0)
        if (
            "IMAGE_PYTHON_PATHS.json" in command
            and "from importlib.metadata import entry_points" in command
        ):
            # Plugin discovery: by default this image ships no pytest11 plugins.
            found = {"plugins": self.discovered, "rejected": []}
            return ExecResult(stdout=json.dumps(found), stderr="", return_code=0)
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

    @receipt_aware
    async def exec(self, command, env=None, **kwargs):
        if "test-stdout.txt" not in command:
            if "/.markers" in command:
                # The verifier reading the guard's markers back.
                return self.layout.run(command)
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
        # What test.sh sets for its pytest (PYTHONPYCACHEPREFIX, say).
        environment.update(getattr(self.layout, "pytest_env", {}))
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                *getattr(self.layout, "pytest_args", ()),
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
    agent_paths=None,
    discovered=(),
    published=True,
    pinned=None,
):
    """Harden a WORKDIR /root image (or *workspace_name*), run test.sh, and score it.

    *agent_paths* hardens a separate verifier sandbox that received those paths;
    *discovered* are the plugins discovery finds in the image's Python.
    *published* puts the model ctrf (:data:`CTRF_PLUGIN`) in BenchFlow's
    protected verifier runtime, as pytest-json-ctrf is when test.sh asks for
    --ctrf; False builds none. *pinned* maps a task's pinned
    release (``pytest_plugin_releases``) to the plugins it adds there
    (``{name: (module, source)}``). The runtime is a folder of the test's,
    owned as the model says (root's).
    """
    served = {"ctrf": ("ctrf_model", CTRF_PLUGIN)} if published else {}

    async def runtime_manifest(env_, task_, sandbox_user, *, build, service="main"):
        plugins = dict(served)
        verifier = task_.config.verifier
        for pin in getattr(verifier, "pytest_plugin_releases", None) or []:
            plugins.update((pinned or {}).get(pin, {}))
        if not plugins:
            return None
        site = tmp_path / "protected-runtime" / "site"
        # A spec is (module, source), optionally with the entry point's
        # object reference and a METADATA version: (module, source, entry, version).
        specs = {name: (*spec, None, None)[:4] for name, spec in plugins.items()}
        for name, (module, source, entry, version) in specs.items():
            if not (site / f"{module}.py").exists():
                install_plugin(
                    site, source, name=name, module=module, entry=entry, version=version
                )
        return {
            "site": str(site),
            "plugins": {
                name: entry or module for name, (module, _, entry, _) in specs.items()
            },
            "plugin_top": sorted(module for module, _, _, _ in specs.values()),
            "runner": None,
        }

    monkeypatch.setattr(lockdown, "_verifier_runtime_manifest", runtime_manifest)
    layout = Layout(tmp_path, monkeypatch, workspace_name=workspace_name)
    layout.discovered = list(discovered)
    if guard_model is not None:
        monkeypatch.setattr(
            lockdown,
            "_DISCOVER_PYTEST_PLUGINS_SCRIPT",
            modelled(Path(guard_module.__file__).read_text(), guard_model),
        )
    if before is not None:
        before(layout)
    task = make_task(layout, env=env)
    if agent_paths is not None:
        agent_paths = tuple(p.format(workspace=layout.workspace) for p in agent_paths)
    task.config.verifier.env = await lockdown._build_verifier_env(
        layout,
        task,
        "agent" if agent_paths is None else None,
        str(layout.workspace),
        agent_paths=agent_paths,
    )
    sandbox = ScriptSandbox(layout, solution, place)
    # One sandbox hardens and verifies in a rollout; here two stand-ins do.
    sandbox._benchflow_verifier_runtime = getattr(
        layout, "_benchflow_verifier_runtime", None
    )
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
    """The guard's signed markers of *kind*, as the verifier copied them to the trial."""
    return sorted(layout.paths.verifier_dir.glob(f"_benchflow_guard_*.*.{kind}"))


def plugin_never_ran(rewards, error) -> bool:
    """An agent's ctrf in test.sh's environment never ran: ctrf came
    from BenchFlow's protected runtime, so the failing solution scored 0."""
    return rewards == {"reward": 0.0} and error is None


def refused_scored_zero(rewards, error) -> bool:
    """Verifier-controlled guard evidence lowers the score, never drops it."""
    return rewards == {"reward": 0.0} and error is None


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
    tmp_path, monkeypatch, caplog
):
    """The guard evidence prevents a pass, with a neutral warning and score 0.

    The verifier user can write these signals; they cannot discard the trial.
    Source and marker assertions below retain the original refusal check."""
    layout, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        PASSING,
        home_cache_under_umask_0000,
        workspace_name="app",
        guard_model=mode_ownership(tmp_path),
        published=False,
    )

    assert "Verifier plugin trust rejected: ctrf" in (
        layout.paths.test_stdout_path.read_text()
    )
    assert rewards == {"reward": 0.0}
    assert "pytest plugin guard not confirmed" in caplog.text
    assert error is None


def image_plugin_others_can_write(layout):
    """An image plugin in a world-writable venv (``chmod -R 777 /opt/venv``)."""
    previous = os.umask(0)
    try:
        install_plugin(layout.home / "opt-venv/site-packages", CTRF_PLUGIN)
    finally:
        os.umask(previous)


def image_venv(env, layout):
    return [layout.home / "opt-venv/site-packages"]


@pytest.mark.parametrize(
    ("mode", "reward"),
    [("separate", 1.0), ("shared", None)],
    ids=["separate", "shared"],
)
async def test_separate_verifier_distrusts_only_what_the_transfer_wrote(
    tmp_path, monkeypatch, mode, reward
):
    """Guards the separate-mode trust rule (review of sdk-update-2026-09-27, must-fix 3).

    The separate verifier passed ``sandbox_user=None`` but kept the owner and
    mode rule meant for the agent's own sandbox, although no agent process
    ever ran in the fresh one. There only the transferred paths (and /logs)
    are the agent's, so an image plugin others could write is the image's
    own. In the agent's sandbox the same plugin could have been rewritten by
    the agent, and refusing it is not scored.
    """
    layout, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        PASSING,
        image_venv,
        before=image_plugin_others_can_write,
        workspace_name="app",
        guard_model=mode_ownership(tmp_path),
        agent_paths=("{workspace}", "/logs/artifacts") if mode == "separate" else None,
        published=False,
    )

    stdout = layout.paths.test_stdout_path.read_text()
    assert rewards == {"reward": 0.0 if reward is None else reward}
    assert error is None
    assert ("Verifier plugin trust rejected: ctrf" in stdout) == (mode == "shared")


async def test_separate_verifier_still_refuses_a_plugin_planted_in_the_workspace(
    tmp_path, monkeypatch, caplog
):
    """Guards #1116 in separate mode: the transferred workspace is still the agent's.

    A hostile ctrf the agent left in its workspace venv crosses with the
    workspace. The verifier's pytest sees it first, the guard refuses it, and
    the failing solution scores 0: the planted hook never runs.
    """
    proof = tmp_path / "hostile-imported"
    site_path = Path(".venv/lib/python3.12/site-packages")

    def plant(layout):
        install_plugin(layout.workspace / site_path, hostile_ctrf(proof))

    def loads_planted(env, layout):
        return [layout.workspace / site_path, *uvx(env, layout)]

    layout, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        FAILING,
        loads_planted,
        before=plant,
        workspace_name="app",
        guard_model=mode_ownership(tmp_path),
        agent_paths=("{workspace}", "/logs/artifacts"),
        published=False,
    )

    stdout = layout.paths.test_stdout_path.read_text()
    assert "Verifier plugin trust rejected: ctrf" in stdout, stdout
    assert refused_scored_zero(rewards, error), (rewards, error)
    assert "pytest plugin guard not confirmed" in caplog.text
    assert markers(layout, "installed") == []
    assert not proof.exists()


async def test_separate_verifier_guard_blocks_the_transfer_not_the_runtime_dirs(
    tmp_path, monkeypatch
):
    """The armed guard in a separate verifier sandbox blocks the transfer's paths."""
    layout = Layout(tmp_path, monkeypatch, workspace_name="app")
    task = make_task(layout)
    await lockdown._build_verifier_env(
        layout,
        task,
        None,
        str(layout.workspace),
        agent_paths=(str(layout.workspace), "/out/report.txt", "/logs/artifacts"),
    )

    (guard,) = layout.fs.glob("_benchflow_guard_*/_benchflow_guard_*.py")
    source = guard.read_text()
    blocked = (str(layout.workspace), "/out/report.txt", "/logs/artifacts", "/logs")
    assert f"_BENCHFLOW_BLOCKED = {blocked!r}" in source
    assert "_BENCHFLOW_OWNERSHIP = False" in source


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
        "PIP_CONFIG_FILE": os.devnull,
    }
    for name in ("uv.toml", "pip.conf"):
        assert (directory / name).read_text() == ""
        assert (directory / name).stat().st_mode & 0o777 == 0o644
    assert not (directory / "pip").exists()


async def test_home_outside_the_workspace_moves_too_with_fixed_pip_policy(
    tmp_path, monkeypatch
):
    """A $HOME the agent cannot write still gets a fresh directory the guard trusts.

    Guards the fix for the plugin guard's umask dependence (review of
    sdk-update-2026-09-27 at c5b75fcc, must-fix 3). With WORKDIR /app the uv
    cache stayed in /root/.cache/uv and was judged by its modes; on
    Docker-in-Docker, whose exec mask is 0000, uv created it world-writable
    and the guard refused Terminal-Bench 2's own ctrf. Caches, tools and
    managed Pythons now always move. CONFIRM-lockdown-2 also pins pip file
    configuration off across interpreters; trusted mirrors use environment
    settings instead of ~/.config/pip/pip.conf.
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
        "PIP_CONFIG_FILE": os.devnull,
    }
    # Now the guard trusts nothing below it by path: a plugin
    # test.sh installs there comes from the protected runtime instead.
    assert len(list(layout.fs.glob("_benchflow_guard_*/_benchflow_guard_*.py"))) == 1


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
async def test_planted_plugin_loaded_anyway_is_refused_scored_zero(
    tmp_path, monkeypatch, where, caplog
):
    """The guard evidence prevents a pass, with a neutral warning and score 0.

    The verifier user can write these signals; they cannot discard the trial.
    Source and marker assertions below retain the original refusal check."""
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
        tmp_path,
        monkeypatch,
        FAILING,
        loads_planted,
        before=plant,
        published=False,
    )

    stdout = layout.paths.test_stdout_path.read_text()
    assert "Verifier plugin trust rejected: ctrf" in stdout, stdout
    assert refused_scored_zero(rewards, error), (rewards, error)
    assert "pytest plugin guard not confirmed" in caplog.text
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


async def test_planted_pip_config_is_disabled_for_every_interpreter(
    tmp_path, monkeypatch
):
    """Guards the uv-state fix: ~/.pip/pip.conf under WORKDIR /root is the agent's.

    ``pip install pytest-json-ctrf`` in test.sh lands in the image's
    site-packages, which the guard trusts, so an agent index there would be a
    pass. CONFIRM-lockdown-1/2 requires /dev/null, which suppresses every
    file source, including another interpreter's site config.
    """
    layout = Layout(tmp_path, monkeypatch)
    (layout.home / ".pip").mkdir()
    (layout.home / ".pip/pip.conf").write_text("[global]\nindex-url = http://evil\n")
    env = await lockdown._build_verifier_env(
        layout, make_task(layout), "agent", str(layout.workspace)
    )

    config = Path(env["PIP_CONFIG_FILE"])
    assert str(config) == os.devnull
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


def _pip_venv(tmp_path: Path) -> Path:
    """A venv with pip standing in for the image's system interpreter."""
    from tests.test_verifier_user_config import _venv_with_pip

    python = _venv_with_pip(tmp_path / "scratch", "image-python")
    if python is None:
        pytest.skip("no venv with pip")
    return Path(python)


async def test_an_image_wheelhouse_is_not_read_and_the_warning_names_the_fix(
    tmp_path, monkeypatch, caplog
):
    """No pip configuration, and no key to turn it on.

    The image's root-owned global pip.conf (``no-index``, ``find-links``) would
    let test.sh ``pip install`` a wheel only the image has. The verifier's pip
    reads no configuration file, and the warning tells the task to pass those
    options in its own verifier script.
    """
    from tests.test_verifier_user_config import _download, _wheelhouse

    python = _pip_venv(tmp_path)
    image = _wheelhouse(tmp_path / "image-wheels", "1.0")
    xdg = tmp_path / "etc-xdg"
    (xdg / "pip").mkdir(parents=True)
    (xdg / "pip/pip.conf").write_text(
        f"[global]\nno-index = true\nfind-links = {image}\n"
    )
    layout = Layout(tmp_path, monkeypatch, python=str(python))
    layout.image_env = {"XDG_CONFIG_DIRS": str(xdg)}
    with caplog.at_level("WARNING"):
        env = await lockdown._build_verifier_env(
            layout, make_task(layout), "agent", str(layout.workspace)
        )
    assert env["PIP_CONFIG_FILE"] == os.devnull
    assert str(xdg / "pip/pip.conf") in caplog.text
    assert "pass pip options explicitly in its verifier script" in caplog.text
    assert "image_pip_config" not in caplog.text
    run = {
        "PATH": f"{env['PATH']}:{python.parent}",
        "HOME": str(layout.home),
        "XDG_CONFIG_DIRS": str(xdg),
        "PIP_CONFIG_FILE": env["PIP_CONFIG_FILE"],
        "PIP_NO_INDEX": "1",
    }
    argv = [str(python), "-m", "pip"]
    assert _download(argv, run, tmp_path / "got") == []
    # The fix the warning names: the verifier script passes the options itself.
    got = tmp_path / "got-explicit"
    subprocess.run(
        [
            *argv,
            "download",
            "--no-deps",
            "--no-index",
            "--find-links",
            str(image),
            "-d",
            str(got),
            "bfprobe",
        ],
        env=run,
        capture_output=True,
    )
    assert sorted(p.name for p in got.glob("*.whl")) == ["bfprobe-1.0-py3-none-any.whl"]


@pytest.mark.parametrize("how", ["pip --python", "pip"])
async def test_the_pip_on_the_verifier_path_reads_no_configuration(
    tmp_path, monkeypatch, how
):
    """test.sh's `pip`, as hardening leaves PATH.

    The image's `pip` is on the verifier PATH (the model's entry-point list
    names it, as round 5's binding looked for). Run as `pip --python` into a
    venv whose site file the agent wrote, or plainly with the agent's file as
    the image's site file, it must find nothing: no pip configuration at all.
    """
    from tests.test_verifier_user_config import _download, _venv_with_pip, _wheelhouse

    python = _pip_venv(tmp_path)
    other = _venv_with_pip(tmp_path / "scratch", "agent-venv")
    if other is None:
        pytest.skip("no venv with pip")
    agent = _wheelhouse(tmp_path / "agent-wheels", "6.6")
    planted = f"[global]\nfind-links = {agent}\n"
    if how == "pip --python":
        (Path(other).parent.parent / "pip.conf").write_text(planted)
    else:
        (python.parent.parent / "pip.conf").write_text(planted)
    (tmp_path / "pip-entry-points.json").write_text(
        json.dumps({"pip": str(python.parent / "pip")})
    )
    layout = Layout(tmp_path, monkeypatch, python=str(python))
    layout.image_env = {"XDG_CONFIG_DIRS": str(tmp_path / "no-xdg")}
    env = await lockdown._build_verifier_env(
        layout, make_task(layout), "agent", str(layout.workspace)
    )
    search = f"{env['PATH']}:{python.parent}"
    pip = shutil.which("pip", path=search)
    argv = [pip, "--python", other] if how == "pip --python" else [pip]
    run = {
        "PATH": search,
        "HOME": str(layout.home),
        "PIP_CONFIG_FILE": env["PIP_CONFIG_FILE"],
        "PIP_NO_INDEX": "1",
    }
    assert _download(argv, run, tmp_path / "got") == []
    assert env["PIP_CONFIG_FILE"] == os.devnull


async def test_a_task_env_pip_config_file_is_pinned_to_devnull(
    tmp_path, monkeypatch, caplog
):
    """Guards the fix (the refusal path left the task's file on).

    A task's ``[verifier] env`` naming ``PIP_CONFIG_FILE`` would make every pip,
    including a venv test.sh creates, read its own live site file beside it.
    The verifier keeps /dev/null and says to pass pip options in the script.
    """
    layout = Layout(tmp_path, monkeypatch)
    named = tmp_path / "task-pip.conf"
    named.write_text("[global]\nindex-url = http://elsewhere\n")
    with caplog.at_level("WARNING"):
        env = await lockdown._build_verifier_env(
            layout,
            make_task(layout, env={"PIP_CONFIG_FILE": str(named)}),
            "agent",
            str(layout.workspace),
        )
    assert env["PIP_CONFIG_FILE"] == os.devnull
    assert "pass pip options explicitly in its verifier script" in caplog.text


async def test_a_root_only_pip_config_never_reaches_a_file_others_can_read(
    tmp_path, monkeypatch
):
    """A 0600 global pip.conf holding an index token.

    The verifier copies no pip configuration anywhere, so
    the token stays in the image's own file and in no verifier variable.
    """
    python = _pip_venv(tmp_path)
    xdg = tmp_path / "etc-xdg" / "pip"
    xdg.mkdir(parents=True)
    (xdg / "pip.conf").write_text(
        "[global]\nindex-url = https://user:SECRET-TOKEN@index.example/\n"
    )
    (xdg / "pip.conf").chmod(0o600)
    layout = Layout(tmp_path, monkeypatch, python=str(python))
    layout.image_env = {"XDG_CONFIG_DIRS": str(xdg.parent)}
    env = await lockdown._build_verifier_env(
        layout, make_task(layout), "agent", str(layout.workspace)
    )

    holders = [
        path
        for path in layout.fs.rglob("*")
        if path.is_file() and "SECRET-TOKEN" in path.read_text(errors="replace")
    ]
    assert holders == []
    assert not any("SECRET-TOKEN" in value for value in env.values())


async def test_a_global_pip_config_planted_after_hardening_is_never_read(
    tmp_path, monkeypatch
):
    """A kept service writes pip config after the check.

    The verifier's pip reads no file at all, so a file planted after the
    check is never read either.
    """
    python = _pip_venv(tmp_path)
    runtime = tmp_path / "sandbox-tmp" / "xdg"
    runtime.mkdir(parents=True)
    layout = Layout(tmp_path, monkeypatch, python=str(python))
    layout.image_env = {"XDG_CONFIG_DIRS": str(runtime)}
    env = await lockdown._build_verifier_env(
        layout, make_task(layout), "agent", str(layout.workspace)
    )
    (runtime / "pip").mkdir()
    (runtime / "pip/pip.conf").write_text("[global]\nindex-url = http://planted\n")

    assert env["PIP_CONFIG_FILE"] == os.devnull
    assert not (layout.fs / "_benchflow_pip_config").exists()


async def test_agent_writable_global_pip_config_leaves_pip_reading_no_file(
    tmp_path, monkeypatch
):
    """Only a global file root alone decided is kept.

    An image ``XDG_CONFIG_DIRS`` naming the workspace makes its
    ``pip/pip.conf`` a global file the agent wrote: pip reads no file at all.
    """
    layout = Layout(tmp_path, monkeypatch)
    planted = layout.workspace / "xdg"
    (planted / "pip").mkdir(parents=True)
    (planted / "pip/pip.conf").write_text("[global]\nindex-url = http://evil\n")
    env = await lockdown._build_verifier_env(
        layout,
        make_task(layout, env={"XDG_CONFIG_DIRS": str(planted)}),
        "agent",
        str(layout.workspace),
    )
    assert env["PIP_CONFIG_FILE"] == os.devnull


# How refusals are scored.


def workspace_venv(env, layout):
    """``uv init && uv add pytest-json-ctrf`` in the workspace: a venv there."""
    (layout.workspace / "pyproject.toml").write_text("[project]\nname = 'app'\n")
    (layout.workspace / "uv.lock").write_text("version = 1\n")
    site = layout.workspace / ".venv/lib/python3.12/site-packages"
    install_plugin(site, CTRF_PLUGIN)
    (layout.workspace / ".venv/pyvenv.cfg").write_text("home = /usr/bin\n")
    return [site]


@pytest.mark.parametrize(
    ("solution", "reward"), [(PASSING, 1.0), (FAILING, 0.0)], ids=["oracle", "empty"]
)
async def test_plugin_the_verifier_installs_in_a_workspace_venv_is_scored(
    tmp_path, monkeypatch, solution, reward
):
    """Guards #1170: USACO's oracle was unscored on every trial.

    test.sh adds pytest-json-ctrf to a ``uv init`` project in the workspace
    (/app/.venv) after hardening and runs ``uv run pytest --ctrf``. Every
    file of it is in a site directory and newer than the guard, so the agent
    did not write it: the guard loads it and the solution decides the reward.
    """
    layout, _, rewards, error = await score(
        tmp_path, monkeypatch, solution, workspace_venv, workspace_name="app"
    )

    stdout = layout.paths.test_stdout_path.read_text()
    assert "Verifier plugin trust rejected" not in stdout, stdout
    assert error is None
    assert rewards == {"reward": reward}
    assert markers(layout, "installed") == []


async def test_stale_plugin_module_under_a_fresh_registration_is_refused(
    tmp_path, monkeypatch, caplog
):
    """The guard evidence prevents a pass, with a neutral warning and score 0.

    The verifier user can write these signals; they cannot discard the trial.
    Source and marker assertions below retain the original refusal check."""
    proof = tmp_path / "hostile-imported"
    site_path = Path(".venv/lib/python3.12/site-packages")

    def plant(layout):
        site = layout.workspace / site_path
        site.mkdir(parents=True)
        (site / "ctrf_model.py").write_text(hostile_ctrf(proof))

    def registration_only(env, layout):
        site = layout.workspace / site_path
        info = site / "ctrf_model-0.3.5.dist-info"
        info.mkdir()
        (info / "METADATA").write_text("Metadata-Version: 2.1\nName: ctrf_model\n")
        (info / "entry_points.txt").write_text("[pytest11]\nctrf = ctrf_model\n")
        return [site]

    layout, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        FAILING,
        registration_only,
        before=plant,
        workspace_name="app",
        published=False,
    )

    stdout = layout.paths.test_stdout_path.read_text()
    assert "Verifier plugin trust rejected: ctrf" in stdout
    assert refused_scored_zero(rewards, error), (rewards, error)
    assert "pytest plugin guard not confirmed" in caplog.text
    assert not proof.exists()


async def test_refused_plugin_the_verifier_installed_is_a_verifier_error(
    tmp_path, monkeypatch, caplog
):
    """The guard evidence prevents a pass, with a neutral warning and score 0.

    The verifier user can write these signals; they cannot discard the trial.
    Source and marker assertions below retain the original refusal check."""

    def workspace_target(env, layout):
        target = layout.workspace / "verifier-plugins"
        install_plugin(target, CTRF_PLUGIN)
        return [target]

    layout, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        PASSING,
        workspace_target,
        published=False,
    )

    assert "Verifier plugin trust rejected: ctrf" in (
        layout.paths.test_stdout_path.read_text()
    )
    assert rewards == {"reward": 0.0}
    assert "pytest plugin guard not confirmed" in caplog.text
    assert error is None
    # Keep the underlying refusal evidence without assigning fault.
    (marker,) = markers(layout, "installed")
    assert "verifier-plugins" in marker.read_text()


async def test_planted_duplicate_next_to_a_verifier_install_is_not_excused(
    tmp_path, monkeypatch, caplog
):
    """The guard evidence prevents a pass, with a neutral warning and score 0.

    The verifier user can write these signals; they cannot discard the trial.
    Source and marker assertions below retain the original refusal check."""
    proof = tmp_path / "hostile-imported"
    site_path = Path(".venv/lib/python3.12/site-packages")

    def plant(layout):
        install_plugin(layout.home / site_path, hostile_ctrf(proof), module="agent")

    def workspace_venv(env, layout):
        install_plugin(layout.workspace / site_path, CTRF_PLUGIN)
        return [layout.workspace / site_path]

    layout, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        FAILING,
        workspace_venv,
        before=plant,
        published=False,
    )

    assert refused_scored_zero(rewards, error), (rewards, error)
    assert "pytest plugin guard not confirmed" in caplog.text
    assert markers(layout, "installed") == []
    assert not proof.exists()
