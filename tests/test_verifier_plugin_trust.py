"""Guards GH #1116 with selective PR #1117 adaptation and stricter trust edges.

The ownership-model tests use real files/import resolution with controlled POSIX
metadata, so trusted-path checks run on non-root developer machines. The exploit
test uses an actual subprocess and pytest; it does not fake imports or results.
"""

import ast
import importlib.metadata
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import _pytest_plugin_guard as guard
from benchflow.sandbox import lockdown
from benchflow.sandbox.lockdown import (
    _DISCOVER_PYTEST_PLUGINS_SCRIPT,
    _discover_pytest_plugin_flags,
)


def registration(root, distribution, name, module):
    info = root / f"{distribution}-1.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {distribution}\nVersion: 1.0\n"
    )
    (info / "entry_points.txt").write_text(f"[pytest11]\n{name} = {module}\n")
    return info


def discover(
    monkeypatch, capsys, roots, *, blocked=(), writable=(), nonroot=(), requested=()
):
    """Run real discovery with a deterministic root-owned image filesystem model."""
    eps = [
        ep
        for root in roots
        for dist in importlib.metadata.distributions(path=[str(root)])
        for ep in dist.entry_points
        if ep.group == "pytest11"
    ]
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **_: eps)
    original_stat = os.stat
    writable = {str(p) for p in writable}
    nonroot = {str(p) for p in nonroot}

    def image_stat(path, *args, **kwargs):
        result = list(original_stat(path, *args, **kwargs))
        result[4] = 1000 if str(path) in nonroot else 0
        result[0] &= ~(stat.S_IWGRP | stat.S_IWOTH)
        if str(path) in writable:
            result[0] |= stat.S_IWOTH
        return os.stat_result(result)

    monkeypatch.setattr(os, "stat", image_stat)
    monkeypatch.setattr(
        sys,
        "argv",
        ["discovery", json.dumps([str(p) for p in blocked]), json.dumps(requested)],
    )
    monkeypatch.setattr(sys, "path", [str(p) for p in roots] + sys.path)
    exec(
        compile(_DISCOVER_PYTEST_PLUGINS_SCRIPT, "discovery", "exec"),
        {"__name__": "__main__"},
    )
    return json.loads(capsys.readouterr().out)["plugins"]


def test_workspace_plugin_cannot_turn_failing_verifier_into_pass(tmp_path):
    """Guards #1116/PR #1117: actual pytest must not load an agent registration."""
    registration(tmp_path, "planted", "agent_planted", "agent_plugin")
    (tmp_path / "agent_plugin.py").write_text(
        "def pytest_collection_modifyitems(items):\n"
        "    for item in items:\n        item.obj = lambda: None\n"
    )
    (tmp_path / "test_failing.py").write_text("def test_failure():\n    assert False\n")
    env = dict(os.environ, PYTHONPATH=str(tmp_path), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    found = subprocess.run(
        [
            sys.executable,
            "-c",
            _DISCOVER_PYTEST_PLUGINS_SCRIPT,
            json.dumps([str(tmp_path)]),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    plugins = json.loads(found.stdout)["plugins"]
    assert "agent_planted" not in plugins
    base = [sys.executable, "-m", "pytest", "-q", "test_failing.py"]
    hostile = subprocess.run(
        [*base, "-p", "agent_planted"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    # Prove the planted hook is effective.
    assert hostile.returncode == 0, hostile.stderr
    safe = subprocess.run(
        base + [flag for name in plugins for flag in ("-p", name)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert safe.returncode == 1
    assert "1 failed" in safe.stdout


def test_trusted_image_plugins_remain_available_without_importing_them(
    tmp_path, monkeypatch, capsys
):
    """Guards PR #1117: discovery retains image plugins without running their code."""
    registration(tmp_path, "trusted", "image_plugin", "trusted_package.plugin")
    package = tmp_path / "trusted_package"
    package.mkdir()
    (package / "__init__.py").write_text("raise AssertionError('must not import')\n")
    (package / "plugin.py").write_text("raise AssertionError('must not import')\n")
    assert discover(monkeypatch, capsys, [tmp_path]) == ["image_plugin"]


@pytest.mark.parametrize("case", ["registration", "code", "ancestor", "nonroot_code"])
def test_writable_or_nonroot_provenance_is_rejected(
    tmp_path, monkeypatch, capsys, case
):
    """Guards #1116: protected code alone is insufficient when its path is replaceable."""
    info = registration(tmp_path, "package", "plugin", "image_module")
    code = tmp_path / "image_module.py"
    code.write_text("# no hooks\n")
    writable = {
        "registration": [info / "entry_points.txt"],
        "code": [code],
        "ancestor": [tmp_path],
        "nonroot_code": [],
    }[case]
    nonroot = [code] if case == "nonroot_code" else []
    assert (
        discover(monkeypatch, capsys, [tmp_path], writable=writable, nonroot=nonroot)
        == []
    )


def test_editable_source_and_workspace_shadow_are_rejected(
    tmp_path, monkeypatch, capsys
):
    """Guards PR #1117: resolve the import Python will use, including workspace shadowing."""
    system = tmp_path / "image"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registration(system, "imagepkg", "plugin", "module")
    (system / "module.py").write_text("# trusted copy\n")
    (workspace / "module.py").write_text("# shadowing copy\n")
    assert discover(monkeypatch, capsys, [workspace, system], blocked=[workspace]) == []


def test_duplicate_names_are_not_selected_by_metadata_iteration_order(
    tmp_path, monkeypatch, capsys
):
    """Guards #1116: pytest resolves a name, not the precise entry point inspected."""
    registration(tmp_path, "one", "duplicate", "one")
    registration(tmp_path, "two", "duplicate", "two")
    (tmp_path / "one.py").write_text("# module one\n")
    (tmp_path / "two.py").write_text("# module two\n")
    assert discover(monkeypatch, capsys, [tmp_path]) == []


def test_root_owned_workspace_is_still_blocked(tmp_path, monkeypatch, capsys):
    """Guards PR #1117: verifier freeze/chown does not make agent-produced code trustworthy."""
    registration(tmp_path, "workspace", "workspace_plugin", "module")
    (tmp_path / "module.py").write_text("# agent produced\n")
    assert discover(monkeypatch, capsys, [tmp_path], blocked=[tmp_path]) == []


@pytest.mark.asyncio
async def test_discovery_wiring_and_explicit_task_declarations(tmp_path):
    """Guards PR #1117: pass writable prefixes and preserve operator task declarations."""
    env = SimpleNamespace(
        exec=AsyncMock(
            return_value=SimpleNamespace(
                stdout='{"plugins": ["image_plugin", "declared_plugin"], "rejected": []}',
                stderr="",
                return_code=0,
            )
        )
    )
    task = SimpleNamespace(
        task_dir=None,
        config=SimpleNamespace(
            verifier=SimpleNamespace(pytest_plugins=["declared_plugin"])
        ),
    )
    flags = await _discover_pytest_plugin_flags(env, task, "agent", "/workspace")
    assert flags == "-p image_plugin -p declared_plugin"
    command = env.exec.call_args.args[0]
    assert "/workspace" in command
    assert "/home/agent" in command
    assert env.exec.call_args.kwargs["user"] == "root"


@pytest.mark.asyncio
@pytest.mark.parametrize("declared", [True, False])
async def test_rejected_registration_cannot_be_reenabled_by_task_or_ctrf_inference(
    tmp_path, declared
):
    """Guards #1116: fallback declarations cannot bypass a known failed trust check."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test.sh").write_text("pytest --ctrf /logs/verifier/ctrf.json\n")
    task = SimpleNamespace(
        task_dir=tmp_path,
        paths=SimpleNamespace(uses_native_verifier_dir=False),
        config=SimpleNamespace(
            verifier=SimpleNamespace(pytest_plugins=["ctrf"] if declared else [])
        ),
    )
    env = SimpleNamespace(
        exec=AsyncMock(
            return_value=SimpleNamespace(
                stdout='{"plugins": [], "rejected": ["ctrf"]}', stderr="", return_code=0
            )
        )
    )
    with pytest.raises(RuntimeError, match="untrusted, ambiguous"):
        await _discover_pytest_plugin_flags(env, task, "agent", "/workspace")


def test_registration_cannot_point_outside_blocked_tree_after_symlink_resolution(
    tmp_path, monkeypatch, capsys
):
    """Guards PR #1117: a trusted-looking link cannot hide agent-controlled code."""
    image = tmp_path / "image"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registration(image, "package", "plugin", "module")
    (workspace / "module.py").write_text("# agent code\n")
    (image / "module.py").symlink_to(workspace / "module.py")
    assert discover(monkeypatch, capsys, [image], blocked=[workspace]) == []


@pytest.mark.parametrize("name", ["declared_plugin", "ctrf"])
@pytest.mark.parametrize("python_available_at_setup", [True, False])
@pytest.mark.asyncio
async def test_unregistered_workspace_module_is_rejected(
    tmp_path, name, python_available_at_setup
):
    """GH1116 review: real discovery must reject plain modules, not only dist-info."""
    (tmp_path / f"{name}.py").write_text(
        "def pytest_collection_modifyitems(items):\n"
        "    for item in items:\n        item.obj = lambda: None\n"
    )
    (tmp_path / "test_failure.py").write_text("def test_failure():\n    assert False\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test.sh").write_text(
        "pytest --ctrf out.json" if name == "ctrf" else "pytest"
    )
    process_env = dict(
        os.environ, PYTHONPATH=str(tmp_path), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
    )
    hostile = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", name, "test_failure.py"],
        cwd=tmp_path,
        env=process_env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert hostile.returncode == 0, hostile.stderr

    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "python3").write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n'
    )
    (bindir / "python3").chmod(0o755)

    async def execute(command, **kwargs):
        result = subprocess.run(
            ["/bin/sh", "-c", command],
            cwd=tmp_path,
            env=dict(process_env, PATH=f"{bindir}:/usr/bin:/bin")
            if python_available_at_setup
            else {"PATH": str(tmp_path / "no-python")},
            capture_output=True,
            text=True,
            timeout=15,
        )
        return SimpleNamespace(
            stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
        )

    task = SimpleNamespace(
        task_dir=tmp_path,
        config=SimpleNamespace(
            verifier=SimpleNamespace(pytest_plugins=[name] if name != "ctrf" else [])
        ),
    )
    flags = await _discover_pytest_plugin_flags(
        SimpleNamespace(exec=execute), task, "agent", str(tmp_path)
    )
    guard_name = "_test_guard"
    (tmp_path / (guard_name + ".py")).write_text(
        _DISCOVER_PYTEST_PLUGINS_SCRIPT
        + "\n_BENCHFLOW_BLOCKED = "
        + repr((str(tmp_path),))
        + "\n"
        + "_BENCHFLOW_REQUESTED = "
        + repr([name])
        + "\n"
    )
    process_env["PYTEST_ADDOPTS"] = f"-p {guard_name} {flags}"
    protected = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "test_failure.py"],
        cwd=tmp_path,
        env=process_env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert protected.returncode != 0
    assert "Verifier plugin trust rejected: " + name in protected.stderr


def test_trusted_unregistered_module_and_missing_plugin(tmp_path, monkeypatch, capsys):
    """GH1116 review: protected plain modules work; absent future plugins do not."""
    (tmp_path / "trusted_plugin.py").write_text("raise AssertionError('do not import')")
    assert discover(
        monkeypatch, capsys, [tmp_path], requested=["trusted_plugin", "future_missing"]
    ) == ["trusted_plugin"]


@pytest.mark.parametrize(
    "stdout, code",
    [
        ("{}", 0),
        ("[]", 0),
        ("bad json", 0),
        ('{"plugins": [], "rejected": []}', 1),
    ],
)
@pytest.mark.asyncio
async def test_discovery_failure_or_missing_declared_plugin_fails_closed(stdout, code):
    """GH1116 review: neither discovery failures nor future installs authorize -p."""
    task = SimpleNamespace(
        task_dir=None,
        config=SimpleNamespace(
            verifier=SimpleNamespace(pytest_plugins=["future_missing"])
        ),
    )
    env = SimpleNamespace(
        exec=AsyncMock(
            return_value=SimpleNamespace(stdout=stdout, stderr="", return_code=code)
        )
    )
    with pytest.raises(RuntimeError, match="pytest plugin trust discovery"):
        await _discover_pytest_plugin_flags(env, task)


@pytest.mark.parametrize("through_environment", [False, True])
def test_nested_plugin_module_cannot_borrow_trusted_entrypoint(
    tmp_path, through_environment
):
    """GH1116 review: pytest_plugins/PYTEST_PLUGINS resolve modules, not aliases."""
    image = tmp_path / "image"
    workspace = tmp_path / "workspace"
    image.mkdir()
    workspace.mkdir()
    registration(image, "trusted_dependency", "dependency", "protected_dep")
    (image / "protected_dep.py").write_text("# trusted entry point target\n")
    (image / "parent_plugin.py").write_text("pytest_plugins = ['dependency']\n")
    (workspace / "dependency.py").write_text(
        "from pathlib import Path\nPath('hostile_imported').touch()\n"
        "def pytest_collection_modifyitems(items):\n"
        "    for item in items:\n        item.obj = lambda: None\n"
    )
    (workspace / "test_failure.py").write_text(
        "def test_failure():\n    assert False\n"
    )
    guard_name = "_test_guard"
    (image / (guard_name + ".py")).write_text(
        _DISCOVER_PYTEST_PLUGINS_SCRIPT
        + "\n_BENCHFLOW_BLOCKED = "
        + repr((str(workspace),))
        + "\n_BENCHFLOW_REQUESTED = []\n"
        # Deterministic image ownership model; actual pytest/import/entrypoint
        # resolution remains real, as in the filesystem discovery tests above.
        + "def trusted(path, blocked):\n    return bool(path) and path.startswith("
        + repr(str(image) + "/")
        + ")\n"
    )
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(image), str(workspace)]),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_PLUGINS="dependency" if through_environment else "",
        PYTEST_ADDOPTS="" if through_environment else "-p parent_plugin",
    )
    command = [sys.executable, "-m", "pytest", "-q", "test_failure.py"]
    unsafe = subprocess.run(
        command, cwd=workspace, env=env, capture_output=True, text=True, timeout=15
    )
    assert unsafe.returncode == 0, unsafe.stdout + unsafe.stderr
    (workspace / "hostile_imported").unlink()
    env["PYTEST_ADDOPTS"] = f"-p {guard_name} " + env["PYTEST_ADDOPTS"]
    safe = subprocess.run(
        command, cwd=workspace, env=env, capture_output=True, text=True, timeout=15
    )
    assert safe.returncode != 0
    assert "Verifier plugin trust rejected: dependency" in safe.stderr
    assert not (workspace / "hostile_imported").exists()


def test_runtime_guard_blocks_resolved_workspace_symlink(tmp_path, monkeypatch):
    """GH1116 review: runtime and discovery must normalize the same blocked roots."""
    actual = tmp_path / "actual_workspace"
    actual.mkdir()
    link = tmp_path / "workspace_link"
    link.symlink_to(actual, target_is_directory=True)
    (actual / "workspace_plugin.py").write_text(
        "raise AssertionError('must not import')"
    )
    original_stat = os.stat

    def root_owned(path, *args, **kwargs):
        result = list(original_stat(path, *args, **kwargs))
        result[4] = 0
        result[0] &= ~(stat.S_IWGRP | stat.S_IWOTH)
        return os.stat_result(result)

    monkeypatch.setattr(os, "stat", root_owned)
    monkeypatch.setattr(sys, "path", [str(actual), *sys.path])
    monkeypatch.setattr(guard, "_BENCHFLOW_BLOCKED", (str(link),))
    assert guard.trusted_module("workspace_plugin", ())
    with pytest.raises(
        RuntimeError, match="Verifier plugin trust rejected: workspace_plugin"
    ):
        guard._validate(["workspace_plugin"], entry_points=False)


def test_cli_builtin_name_cannot_borrow_builtin_trust_for_hostile_alias(tmp_path):
    """GH1116 review: -p pytester resolves a registered alias before its builtin."""
    image = tmp_path / "image"
    workspace = tmp_path / "workspace"
    image.mkdir()
    workspace.mkdir()
    metadata = registration(workspace, "planted_builtin", "pytester", "hostile")
    (workspace / "hostile.py").write_text(
        "from pathlib import Path\nPath('hostile_imported').touch()\n"
        "def pytest_collection_modifyitems(items):\n"
        "    for item in items:\n        item.obj = lambda: None\n"
    )
    (workspace / "test_failure.py").write_text(
        "def test_failure():\n    assert False\n"
    )
    (image / "_test_guard.py").write_text(
        _DISCOVER_PYTEST_PLUGINS_SCRIPT
        + "\n_BENCHFLOW_BLOCKED = "
        + repr((str(workspace),))
        + "\n_BENCHFLOW_REQUESTED = []\n"
        # Root-image ownership model, retaining real import/entrypoint behavior.
        + "def trusted(path, blocked):\n    return bool(path) and not path.startswith("
        + repr(str(workspace))
        + ")\n"
    )
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(image), str(workspace)]),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_ADDOPTS="-p pytester",
    )
    command = [sys.executable, "-m", "pytest", "-q", "test_failure.py"]
    unsafe = subprocess.run(
        command, cwd=workspace, env=env, capture_output=True, text=True, timeout=15
    )
    assert unsafe.returncode == 0, unsafe.stderr
    (workspace / "hostile_imported").unlink()
    env["PYTEST_ADDOPTS"] = "-p _test_guard -p pytester"
    safe = subprocess.run(
        command, cwd=workspace, env=env, capture_output=True, text=True, timeout=15
    )
    assert safe.returncode != 0
    assert "Verifier plugin trust rejected: pytester" in safe.stderr
    assert not (workspace / "hostile_imported").exists()
    # With no alias registered, the same -p name can load the genuine builtin.
    (metadata / "entry_points.txt").unlink()
    builtin = subprocess.run(
        command, cwd=workspace, env=env, capture_output=True, text=True, timeout=15
    )
    assert builtin.returncode == 1, builtin.stderr
    assert "1 failed" in builtin.stdout
    assert "Verifier plugin trust rejected" not in builtin.stderr


@pytest.mark.asyncio
async def test_shell_only_verifier_does_not_require_python_discovery(tmp_path):
    """GH1116 smoke regression: shell verifiers on minimal images need no Python."""

    async def execute(command, **kwargs):
        result = subprocess.run(
            ["/bin/sh", "-c", command],
            env={"PATH": str(tmp_path / "no-python")},
            capture_output=True,
            text=True,
            timeout=5,
        )
        return SimpleNamespace(
            stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
        )

    task = SimpleNamespace(
        task_dir=None,
        config=SimpleNamespace(verifier=SimpleNamespace(pytest_plugins=[])),
    )
    assert (
        await _discover_pytest_plugin_flags(SimpleNamespace(exec=execute), task) == ""
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("probe_code", [0, 2])
async def test_exit127_is_not_treated_as_missing_python_without_absence_proof(
    probe_code,
):
    """GH1116: a broken discovery process or failed availability probe stays closed."""
    task = SimpleNamespace(
        task_dir=None,
        config=SimpleNamespace(verifier=SimpleNamespace(pytest_plugins=[])),
    )
    env = SimpleNamespace(
        exec=AsyncMock(
            side_effect=[
                SimpleNamespace(return_code=127, stdout="", stderr="failed"),
                SimpleNamespace(return_code=probe_code, stdout="", stderr=""),
            ]
        )
    )
    with pytest.raises(RuntimeError, match="pytest plugin trust discovery"):
        await _discover_pytest_plugin_flags(env, task)


# Arguments pytest 7.0 passes to the hooks the guard implements. pytest 8.1
# added ``plugin_name`` to pytest_plugin_registered; pluggy refuses a whole
# plugin whose hookimpl names an argument its hookspec lacks.
_PYTEST_7_HOOK_ARGUMENTS = {
    "pytest_addhooks": {"pluginmanager"},
    "pytest_plugin_registered": {"plugin", "manager"},
}
_BUILTIN_GENERICS = {"dict", "frozenset", "list", "set", "tuple", "type"}


def test_guard_source_imports_on_python_37_and_registers_on_pytest_7():
    """Guards the fix for the pytest plugin guard on Python 3.8 + pytest 7.4.4.

    Import-time annotations such as ``tuple[str, ...]`` raise TypeError before
    Python 3.9, and a ``plugin_name`` hook argument is a pluggy validation
    error before pytest 8.1; either one aborts pytest, so a correct solution
    silently scores 0 on older verifier images.
    """
    tree = ast.parse(_DISCOVER_PYTEST_PLUGINS_SCRIPT, feature_version=(3, 7))
    lazy_annotations = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )
    annotations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = node.args
            annotations += [
                argument.annotation
                for argument in [
                    *arguments.posonlyargs,
                    *arguments.args,
                    *arguments.kwonlyargs,
                    *filter(None, (arguments.vararg, arguments.kwarg)),
                ]
                if argument.annotation is not None
            ]
            if node.returns is not None:
                annotations.append(node.returns)
    for annotation in [] if lazy_annotations else annotations:
        for part in ast.walk(annotation):
            assert not (
                isinstance(part, ast.Subscript)
                and isinstance(part.value, ast.Name)
                and part.value.id in _BUILTIN_GENERICS
            ), ast.unparse(annotation)
            assert not (
                isinstance(part, ast.BinOp) and isinstance(part.op, ast.BitOr)
            ), ast.unparse(annotation)
    hooks = {
        node.name: {argument.arg for argument in node.args.args}
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("pytest_")
    }
    assert hooks.keys() == _PYTEST_7_HOOK_ARGUMENTS.keys()
    for name, arguments in hooks.items():
        assert arguments <= _PYTEST_7_HOOK_ARGUMENTS[name], name


def _python_38():
    found = shutil.which("python3.8")
    if found:
        return found
    uv = shutil.which("uv")
    if uv is None:
        return None
    result = subprocess.run(
        [uv, "python", "find", "--no-python-downloads", "3.8"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


@pytest.mark.skipif(_python_38() is None, reason="no Python 3.8 interpreter")
def test_guard_imports_and_finds_registrations_on_real_python_38(tmp_path):
    """Guards the fix for the pytest plugin guard on a real Python 3.8.

    Before Python 3.10 an entry point does not name its distribution, so the
    registration path must come from the distribution that declares it;
    otherwise every protected plugin is refused and the verifier aborts.
    """
    image = tmp_path / "image"
    registration(image, "trusted", "image_plugin", "image_module")
    (image / "image_module.py").write_text("raise AssertionError('must not import')\n")
    (tmp_path / "guard38.py").write_text(
        _DISCOVER_PYTEST_PLUGINS_SCRIPT
        # Root-image ownership model; metadata and import resolution stay real.
        + "\ndef trusted(path, blocked):\n    return str(path).startswith("
        + repr(str(image) + "/")
        + ")\n"
    )
    result = subprocess.run(
        [
            _python_38(),
            "-I",
            "-c",
            "import json, sys; sys.path[:0] = sys.argv[1:3]; import guard38; "
            "print(json.dumps(guard38.discover((), ['image_plugin'])))",
            str(image),
            str(tmp_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "image_plugin" in json.loads(result.stdout)["plugins"]


def _ownership_model(script):
    """Model an image where everything outside the blocked trees is root-owned.

    Inserted before the discovery entry point so both the discovery command and
    the runtime guard use it; metadata and import resolution stay real.
    """
    model = (
        "\ndef trusted(path, blocked):\n"
        "    return bool(path) and str(path).startswith('/') and not any(\n"
        "        under(candidate, prefix)\n"
        "        for candidate in (os.path.abspath(path), os.path.realpath(path))\n"
        "        for prefix in blocked\n"
        "    )\n"
    )
    marker = '\nif __name__ == "__main__":'
    assert marker in script
    return script.replace(marker, model + marker)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "planted,workspace_on_image_pythonpath",
    [
        ("registration", False),
        ("registration", True),
        # A workspace that is on the trusted verifier PYTHONPATH really does
        # shadow the plugin module there, so only the WORKDIR case applies.
        ("shadow_module", False),
    ],
)
@pytest.mark.parametrize("declared", [False, True])
async def test_workspace_registration_cannot_drop_or_veto_an_image_plugin(
    tmp_path, monkeypatch, planted, workspace_on_image_pythonpath, declared
):
    """Guards the fix for the pytest plugin guard's discovery reading the image WORKDIR.

    Discovery ran ``python3 -c`` in the agent-writable workspace, so a planted
    dist-info reusing a protected plugin's name, or a module shadowing its
    code, silently dropped that plugin from ``-p`` or, when the task declares
    it, forced a hardening error. The runtime guard, which sees the interpreter
    pytest really uses, must still refuse the plugin whenever the planted file
    is visible there.
    """
    image = tmp_path / "image"
    workspace = tmp_path / "workspace"
    registration(image, "imagepkg", "image_plugin", "image_module")
    (image / "image_module.py").write_text("# protected plugin\n")
    workspace.mkdir()
    if planted == "registration":
        registration(workspace, "planted", "image_plugin", "hostile")
    (
        workspace / ("hostile.py" if planted == "registration" else "image_module.py")
    ).write_text(
        "from pathlib import Path\nPath('hostile_imported').touch()\n"
        "def pytest_collection_modifyitems(items):\n"
        "    for item in items:\n        item.obj = lambda: None\n"
    )
    (workspace / "test_failure.py").write_text(
        "def test_failure():\n    assert False\n"
    )
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "python3").write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    (bindir / "python3").chmod(0o755)
    image_pythonpath = os.pathsep.join(
        [str(workspace), str(image)] if workspace_on_image_pythonpath else [str(image)]
    )
    # The sandbox's runtime prefixes (/tmp, /logs, ...) stand in here: this
    # image lives under pytest's tmp_path, which is under /tmp on Linux, and a
    # blocked /tmp would refuse the image plugin itself instead of the planted
    # workspace file this test is about.
    monkeypatch.setattr(
        lockdown, "_RUNTIME_PATH_PREFIXES", (str(tmp_path / "sandbox-tmp"),)
    )
    monkeypatch.setattr(
        lockdown,
        "_DISCOVER_PYTEST_PLUGINS_SCRIPT",
        _ownership_model(_DISCOVER_PYTEST_PLUGINS_SCRIPT),
    )

    async def execute(command, **kwargs):
        # The image's WORKDIR and PYTHONPATH, as a plain ``exec`` inherits them.
        result = subprocess.run(
            ["/bin/sh", "-c", command],
            cwd=workspace,
            env={"PATH": f"{bindir}:/usr/bin:/bin", "PYTHONPATH": image_pythonpath},
            capture_output=True,
            text=True,
            timeout=30,
        )
        return SimpleNamespace(
            stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
        )

    task = SimpleNamespace(
        task_dir=None,
        config=SimpleNamespace(
            verifier=SimpleNamespace(
                pytest_plugins=["image_plugin"] if declared else []
            )
        ),
    )
    flags = await _discover_pytest_plugin_flags(
        SimpleNamespace(exec=execute), task, "agent", str(workspace)
    )
    assert shlex.split(flags)[1::2].count("image_plugin") == 1

    # The -p list holds discovered as well as declared plugins, and the guard
    # re-checks all of them where pytest runs: here the planted registration
    # is on sys.path (cwd under ``python -m``), so pytest would load it.
    (image / "_test_guard.py").write_text(
        _ownership_model(_DISCOVER_PYTEST_PLUGINS_SCRIPT)
        + "\n_BENCHFLOW_BLOCKED = "
        + repr((str(workspace),))
        + "\n_BENCHFLOW_REQUESTED = "
        + repr(shlex.split(flags)[1::2])
        + "\n"
    )
    env = dict(
        os.environ,
        PYTHONPATH=str(image),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_ADDOPTS=flags,
    )
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    unsafe = subprocess.run(
        [*command, "test_failure.py"],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert unsafe.returncode == 0, unsafe.stdout + unsafe.stderr
    (workspace / "hostile_imported").unlink()
    env["PYTEST_ADDOPTS"] = "-p _test_guard " + flags
    safe = subprocess.run(
        [*command, "test_failure.py"],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert safe.returncode != 0
    assert "Verifier plugin trust rejected: image_plugin" in safe.stderr
    assert not (workspace / "hostile_imported").exists()


def test_guard_reads_the_command_line_on_pytest_before_invocation_params(
    monkeypatch,
):
    """Guards the fix for the pytest plugin guard on pytest < 5.1 (4.6.x aside).

    ``Config.invocation_params`` does not exist there, so ``pytest_addhooks``
    raised AttributeError and aborted every verifier pytest (reproduced with
    pytest 3.10.1, 4.5.0 and 5.0.1). Those versions keep the command line in
    ``Config._origargs`` before importing any ``-p`` plugin; its ``-p`` names
    must still be validated.
    """
    validated = []
    monkeypatch.setattr(
        guard, "_validate", lambda names, **kwargs: validated.append(sorted(names))
    )
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    monkeypatch.delenv("PYTEST_PLUGINS", raising=False)
    config = SimpleNamespace(_origargs=["-q", "-p", "task_plugin", "-pother", "t.py"])

    guard.pytest_addhooks(SimpleNamespace(get_plugin=lambda name: config))

    assert validated[0] == ["other", "task_plugin"]


def _armed_markers(monkeypatch, tmp_path):
    prefix = tmp_path / "markers" / "_benchflow_guard_test"
    prefix.parent.mkdir()
    key = tmp_path / "key"
    key.write_text(lockdown.pytest_plugin_guard_key("_benchflow_guard_test"))
    monkeypatch.setattr(guard, "_BENCHFLOW_MARKERS", str(prefix))
    monkeypatch.setattr(guard, "_BENCHFLOW_KEY_FILE", str(key))
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    monkeypatch.delenv("PYTEST_PLUGINS", raising=False)
    return prefix.parent


def test_guard_crash_is_recorded_for_the_verifier(monkeypatch, tmp_path):
    """Guards the fix for guard crashes being scored since the pytest plugin guard.

    Reading a pytest API that is not there (one such case was fixed earlier) is
    the guard failing, not a plugin being refused: the hook records a
    ``crashed`` marker with the traceback and still stops pytest.
    """
    markers = _armed_markers(monkeypatch, tmp_path)

    with pytest.raises(AttributeError):
        guard.pytest_addhooks(SimpleNamespace(get_plugin=lambda name: None))

    (marker,) = markers.iterdir()
    assert marker.name.endswith(".crashed")
    assert "_origargs" in marker.read_text()


def test_unreadable_plugin_metadata_is_a_scored_rejection(monkeypatch, tmp_path):
    """Guards the crash classification added for the pytest plugin guard.

    Distribution metadata on sys.path can be planted by the agent. Failing to
    parse it must refuse the plugin (the run is scored), never count as a
    guard crash, or a planted file would turn any failure into an unscored
    verifier error. CPython 3.12 raises TypeError, not ValueError, for this
    malformed line, so the rule cannot depend on the exception type.
    """
    markers = _armed_markers(monkeypatch, tmp_path)
    planted = tmp_path / "workspace"
    info = registration(planted, "planted", "agent_plugin", "agent_plugin")
    (info / "entry_points.txt").write_text("[pytest11]\nnot an entry point\n")
    monkeypatch.setattr(sys, "path", [str(planted), *sys.path])
    config = SimpleNamespace(_origargs=["-p", "agent_plugin"])

    with pytest.raises(guard.Rejected, match="agent_plugin"):
        guard.pytest_addhooks(SimpleNamespace(get_plugin=lambda name: config))

    assert list(markers.iterdir()) == []


def _stat_model(monkeypatch, root_only):
    """Model an image where only *root_only* paths are root's alone.

    Every other path reports uid 1000 and mode 0777, as uv's cache did on a
    runtime whose exec mask is 0000 (plus an owner a non-root test can have).
    """
    original_stat = os.stat
    root_only = {str(p) for p in root_only}

    def model(path, *args, **kwargs):
        result = original_stat(path, *args, **kwargs)
        fields = list(result)
        if str(path) in root_only:
            fields[4] = 0
            fields[0] = (fields[0] & ~0o7777) | 0o755
        else:
            fields[4] = 1000
            fields[0] = (fields[0] & ~0o7777) | 0o777
        return os.stat_result(fields, {"st_ctime_ns": result.st_ctime_ns})

    monkeypatch.setattr(os, "stat", model)


def _ancestors(path):
    path = str(path)
    found = [path]
    while os.path.dirname(path) != path:
        path = os.path.dirname(path)
        found.append(path)
    return found


def test_code_below_the_verifier_tool_state_directory_is_trusted_by_path(
    tmp_path, monkeypatch
):
    """Guards the fix for the plugin guard's umask dependence (sdk-update review, must-fix 3).

    Hardening creates the verifier's uv and pip directory root-owned 0755
    after the agent stopped, so what uv writes inside is the verifier's own
    whatever modes the runtime's mask gave it. The same files anywhere else
    are still judged by owner and mode.
    """
    state = tmp_path / "_benchflow_verifier_x"
    site = state / "uv-cache/archive-v0/env/lib/python3.13/site-packages"
    site.mkdir(parents=True)
    (site / "ctrf").mkdir()
    (site / "ctrf/__init__.py").write_text("# the verifier's ctrf\n")
    elsewhere = tmp_path / "root/.cache/uv/site-packages/ctrf"
    elsewhere.mkdir(parents=True)
    (elsewhere / "__init__.py").write_text("# same files, no trusted directory\n")
    _stat_model(monkeypatch, _ancestors(state))
    monkeypatch.setattr(guard, "_BENCHFLOW_TRUSTED", (str(state),))

    assert guard.trusted(str(site / "ctrf/__init__.py"), ())
    assert guard.untrusted_reason(str(elsewhere / "__init__.py"), ()) == (
        str(elsewhere / "__init__.py") + " is owned by uid 1000, not root"
    )
    # Blocked prefixes still win inside it.
    assert not guard.trusted(str(site / "ctrf/__init__.py"), (str(site),))


@pytest.mark.parametrize("flaw", ["writable", "foreign", "parent"])
def test_trusted_directory_itself_must_be_roots_alone(tmp_path, monkeypatch, flaw):
    """Guards the trust-by-path rule: it covers what is below the directory only.

    A directory other users could write, or one below such a parent, could
    have been filled by someone other than the verifier.
    """
    state = tmp_path / "state"
    (state / "pkg").mkdir(parents=True)
    (state / "pkg/plugin.py").write_text("# plugin\n")
    root_only = set(_ancestors(state))
    root_only.discard(str(state if flaw != "parent" else tmp_path))
    _stat_model(monkeypatch, root_only)
    if flaw == "foreign":
        original = os.stat

        def foreign(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            fields = list(result)
            if str(path) == str(state):
                fields[0] = (fields[0] & ~0o7777) | 0o755
            return os.stat_result(fields, {"st_ctime_ns": result.st_ctime_ns})

        monkeypatch.setattr(os, "stat", foreign)
    monkeypatch.setattr(guard, "_BENCHFLOW_TRUSTED", (str(state),))

    assert not guard.trusted(str(state / "pkg/plugin.py"), ())


def test_symlink_out_of_the_trusted_directory_is_judged_where_it_points(
    tmp_path, monkeypatch
):
    """Guards the trust-by-path rule against a link that leaves the directory.

    The resolved path is checked on its own merits, so a link to the
    workspace is refused however trusted the directory holding it.
    """
    state = tmp_path / "state"
    state.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "plugin.py").write_text("# agent code\n")
    (state / "plugin.py").symlink_to(workspace / "plugin.py")
    _stat_model(monkeypatch, [*_ancestors(state), *_ancestors(workspace / "plugin.py")])
    monkeypatch.setattr(guard, "_BENCHFLOW_TRUSTED", (str(state),))

    assert guard.trusted(str(workspace / "plugin.py"), ())
    assert not guard.trusted(str(state / "plugin.py"), (str(workspace),))


def test_separate_sandbox_distrusts_only_the_transferred_paths(tmp_path, monkeypatch):
    """Guards the separate-mode trust rule (sdk-update review, must-fix 3).

    No agent process ever ran in a separate verifier sandbox, so owners and
    modes say nothing about the agent there: only the paths the transfer
    wrote are refused, and the refusal says why.
    """
    workspace = tmp_path / "app"
    image = tmp_path / "image"
    workspace.mkdir()
    image.mkdir()
    (workspace / "plugin.py").write_text("# transferred from the agent\n")
    (image / "plugin.py").write_text("# the verifier image's own\n")
    _stat_model(monkeypatch, ())  # nothing is root's alone
    monkeypatch.setattr(guard, "_BENCHFLOW_OWNERSHIP", False)
    blocked = (str(workspace),)

    assert guard.trusted(str(image / "plugin.py"), blocked)
    assert guard.untrusted_reason(str(workspace / "plugin.py"), blocked) == (
        f"{workspace}/plugin.py is under {workspace}, which holds files copied "
        "from the agent's sandbox"
    )
    # In the agent's own sandbox the same image file is not root's alone.
    monkeypatch.setattr(guard, "_BENCHFLOW_OWNERSHIP", True)
    assert not guard.trusted(str(image / "plugin.py"), blocked)
    assert guard.untrusted_reason(str(workspace / "plugin.py"), blocked).endswith(
        "which the agent could write"
    )


def test_separate_sandbox_discovery_proposes_image_plugins_by_path_alone(tmp_path):
    """Guards the separate-mode trust rule in discovery, with real files and Python.

    The files here belong to the test's user, as an image's might not be
    root's alone; discovery told the sandbox is separate still proposes the
    image plugin, and never the one registered in the transferred workspace.
    """
    image = tmp_path / "image"
    workspace = tmp_path / "app"
    registration(image, "imagepkg", "image_plugin", "image_module")
    (image / "image_module.py").write_text("# image plugin\n")
    registration(workspace, "planted", "agent_plugin", "agent_module")
    (workspace / "agent_module.py").write_text("# agent plugin\n")
    command = lockdown._discover_pytest_plugins_cmd(
        (str(workspace),), pythonpath=f"{image}:{workspace}", ownership=False
    )

    result = subprocess.run(
        ["/bin/sh", "-c", command.replace("python3 -c", f"{sys.executable} -c", 1)],
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    plugins = json.loads(result.stdout)["plugins"]
    assert "image_plugin" in plugins and "agent_plugin" not in plugins


def _root_owned_everywhere(monkeypatch):
    """Every path is root's alone, as an image's own files are."""
    original_stat = os.stat

    def model(path, *args, **kwargs):
        result = original_stat(path, *args, **kwargs)
        fields = list(result)
        fields[4] = 0
        fields[0] &= ~(stat.S_IWGRP | stat.S_IWOTH)
        return os.stat_result(fields, {"st_ctime_ns": result.st_ctime_ns})

    monkeypatch.setattr(os, "stat", model)


def test_refusal_names_each_plugin_and_what_the_guard_could_not_trust(
    tmp_path, monkeypatch
):
    """Guards the refusal message fix (review of sdk-update-2026-09-27, must-fix 3).

    "Where the agent could write" appeared even when nothing was
    agent-writable. The message now gives each refused plugin its reason: the
    untrusted path and why, a name registered twice, or a plugin installed
    nowhere.
    """
    workspace = tmp_path / "workspace"
    image = tmp_path / "image"
    info = registration(workspace, "planted", "agent_plugin", "agent_module")
    (workspace / "agent_module.py").write_text("# agent\n")
    registration(image, "one", "twice", "mod_one")
    registration(image, "two", "twice", "mod_two")
    (image / "mod_one.py").write_text("# one\n")
    (image / "mod_two.py").write_text("# two\n")
    monkeypatch.setattr(sys, "path", [str(workspace), str(image), *sys.path])
    _root_owned_everywhere(monkeypatch)
    monkeypatch.setattr(guard, "_BENCHFLOW_BLOCKED", (str(workspace),))
    monkeypatch.setattr(guard, "_BENCHFLOW_MARKERS", "")

    with pytest.raises(guard.Rejected) as refused:
        guard._validate(["agent_plugin", "twice", "missing_plugin"])

    message = str(refused.value)
    assert message.startswith(
        "Verifier plugin trust rejected: agent_plugin, missing_plugin, twice ("
    )
    assert (
        f"agent_plugin: {info} is under {workspace}, which the agent could write"
        in message
    )
    assert "twice: registered 2 times" in message
    assert "missing_plugin: not installed where this pytest looks" in message
    # Nothing untrusted explains "twice" or "missing_plugin": not an install.
    assert "not scored" not in message


def test_install_evidence_needs_an_untrusted_file(tmp_path, monkeypatch):
    """Guards the "installed" evidence fix (sdk-update review, must-fix 3).

    A refusal counted as the verifier's own install, and so unscored, even
    when no file behind it was untrusted: two trusted registrations of one
    name, all newer than the guard, were "installed where the agent could
    write". Now only untrusted, newer files are evidence.
    """
    image = tmp_path / "image"
    workspace = tmp_path / "workspace"
    registration(image, "one", "twice", "mod_one")
    registration(image, "two", "twice", "mod_two")
    (image / "mod_one.py").write_text("# one\n")
    (image / "mod_two.py").write_text("# two\n")
    info = registration(workspace, "venv", "venv_plugin", "venv_module")
    (workspace / "venv_module.py").write_text("# the verifier's install\n")
    monkeypatch.setattr(sys, "path", [str(image), str(workspace), *sys.path])
    _root_owned_everywhere(monkeypatch)
    monkeypatch.setattr(guard, "_armed_ns", lambda: 0)  # every file is newer
    blocked = (str(workspace),)

    assert guard._installed_during_verification(["twice"], True, blocked, ()) == []
    assert guard._installed_during_verification(["venv_plugin"], True, blocked, ()) == [
        str(info),
        str(info / "entry_points.txt"),
        str(workspace / "venv_module.py"),
    ]
    # One refused name without untrusted evidence keeps the whole refusal scored.
    assert (
        guard._installed_during_verification(
            ["twice", "venv_plugin"], True, blocked, ()
        )
        == []
    )


def _signed_line(name, detail, key_guard):
    import hashlib
    import hmac

    body = json.dumps(detail)
    key = lockdown.pytest_plugin_guard_key(key_guard).encode()
    mac = hmac.new(key, f"{name}\n{body}".encode(), hashlib.sha256).hexdigest()
    return f"{name}\t{mac}\t{body}"


def test_verifier_keeps_only_markers_this_guard_signed():
    """Guards the marker fix (sdk-update review, must-fix 3): a marker must be signed.

    The key derives from a secret that never leaves the host process, per
    guard name; a line signed for another guard, unsigned, of an unknown kind
    or with a body that is not a string is dropped.
    """
    guard = "_benchflow_guard_" + "a" * 32
    other = "_benchflow_guard_" + "b" * 32
    crashed = f"{guard}.12-0123abcd.crashed"
    output = "\n".join(
        [
            _signed_line(crashed, "Traceback: boom", guard),
            _signed_line(f"{guard}.12-0123abcd.installed", "x", other),
            _signed_line(f"{guard}.12-0123abcd.forged", "x", guard),
            _signed_line(f"{other}.12-0123abcd.crashed", "x", guard),
            f'{guard}.13-0123abcd.loading\t{"0" * 64}\t""',
            _signed_line(f"{guard}.14-0123abcd.loading", ["not", "a", "string"], guard),
            "garbage",
            "",
        ]
    )

    assert lockdown.parse_pytest_plugin_guard_markers(output, guard) == [
        lockdown.GuardMarker(crashed, "crashed", "Traceback: boom")
    ]
    assert lockdown.pytest_plugin_guard_key(guard) != lockdown.pytest_plugin_guard_key(
        other
    )


def test_marker_read_back_skips_links_and_fifos_and_is_bounded(tmp_path, monkeypatch):
    """Guards the marker read-back against files that could hang or flood it."""
    monkeypatch.setattr(lockdown, "_PYTEST_PLUGIN_GUARD_PARENT", str(tmp_path))
    guard = "_benchflow_guard_" + "c" * 32
    markers = tmp_path / guard / ".markers"
    markers.mkdir(parents=True)
    # A marker file holds its signature and body; its name is the file name.
    genuine = _signed_line(f"{guard}.1-0123abcd.crashed", "boom", guard)
    content = genuine.split("\t", 1)[1] + "\n"
    (markers / f"{guard}.1-0123abcd.crashed").write_text(content)
    os.mkfifo(markers / f"{guard}.2-0123abcd.loading")
    (tmp_path / "elsewhere").write_text(content)
    (markers / f"{guard}.3-0123abcd.crashed").symlink_to(tmp_path / "elsewhere")
    for index in range(100):
        (markers / f"z{index:03d}").write_text("x" * 200_000)

    result = subprocess.run(
        ["/bin/sh", "-c", lockdown.pytest_plugin_guard_markers_cmd(guard)],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    names = [line.split("\t", 1)[0] for line in result.stdout.splitlines() if line]
    assert f"{guard}.1-0123abcd.crashed" in names
    assert not any(".2-0123abcd." in name or ".3-0123abcd." in name for name in names)
    assert len(result.stdout) < 65 * 64 * 1024
    assert [
        m.kind for m in lockdown.parse_pytest_plugin_guard_markers(result.stdout, guard)
    ] == ["crashed"]


def test_guard_writes_its_loading_marker_only_under_pytests_plugin_loader(tmp_path):
    """Guards the marker fix: code that imports the guard by name leaves no marker.

    Otherwise a solution the tests run could ``import <guard>`` in a fresh
    Python, leave a ``loading`` marker that no pytest ever removes, and make
    its run unscored.
    """
    guard_name = "_benchflow_guard_" + "d" * 32
    guard_dir = tmp_path / guard_name
    markers = guard_dir / ".markers"
    markers.mkdir(parents=True)
    key = guard_dir / ".key"
    key.write_text(lockdown.pytest_plugin_guard_key(guard_name))
    (guard_dir / f"{guard_name}.py").write_text(
        lockdown._pytest_plugin_guard_source(
            guard_name, (), [], str(markers), key_file=str(key)
        )
    )
    (tmp_path / "test_pass.py").write_text("def test_pass():\n    assert True\n")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTEST_", "PYTHON"))
    }
    env.update(PYTHONPATH=str(guard_dir), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")

    subprocess.run(
        [sys.executable, "-c", f"import {guard_name}"], env=env, check=True, timeout=30
    )
    assert list(markers.iterdir()) == []

    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", guard_name, "test_pass.py"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    (marker,) = markers.iterdir()
    assert marker.name.endswith(".registered")
    (signed,) = lockdown.parse_pytest_plugin_guard_markers(
        f"{marker.name}\t{marker.read_text()}", guard_name
    )
    assert signed.kind == "registered"
