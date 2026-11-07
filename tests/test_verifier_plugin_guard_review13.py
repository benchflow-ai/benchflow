"""Guard V2-R12-CONF-1/2/3/4/5 against the inputs missed at c05d01a3."""

import json
import shlex
import shutil
from types import SimpleNamespace

import pytest

from benchflow.sandbox import _pytest_plugin_guard as guard
from benchflow.sandbox import lockdown
from tests import test_verifier_uv_state as uv_state
from tests.test_verifier_plugin_guard_review12 import proven
from tests.test_verifier_uv_state import (
    CTRF_PLUGIN,
    FAILING,
    PASSING,
    hostile_ctrf,
    install_plugin,
    markers,
    plugin_never_ran,
    score,
)

ARCHIVES = [
    "pytest_json_ctrf-0.3.5-py3-none-any.whl",
    *[
        f"plugin-0.3.5{suffix}"
        for suffix in (
            ".zip",
            ".tar",
            ".tar.gz",
            ".tgz",
            ".tar.bz2",
            ".tbz",
            ".tar.xz",
            ".txz",
            ".tar.Z",
        )
    ],
    "plugin.future-format",
    "plugin==0.3.5.tar.gz",
]


@pytest.mark.parametrize(
    "value",
    [
        "pytest",
        "pytest-json-ctrf",
        "pytest_json_ctrf",
        "zope.interface==6.0",
        "pytest[testing]",
        "pytest>=8.0rc1",
        "pytest!=8.0.*",
        "pytest~=1!8.0.post1",
        "pytest<9",
    ],
)
def test_registry_subset_preserves_honest_operands(tmp_path, value):
    """Guard the supported registry subset alongside CONF-1's local sources."""
    assert proven(tmp_path, "uv pip install -r /tests/deps.txt", value + "\n") == [
        "/tests/deps.txt"
    ]


@pytest.mark.parametrize("operand", ARCHIVES)
@pytest.mark.parametrize("tool", ["uv pip install", "uv add", "pip install"])
@pytest.mark.parametrize("transport", ["shell", "python", "requirements"])
def test_archive_names_never_certify_registry_inputs(
    tmp_path, operand, tool, transport
):
    """CONF-1 at c05d01a3: a package-name regex also accepted local archives."""
    argv = [*shlex.split(tool), operand]
    requirements = "pytest-json-ctrf==0.3.5\n"
    if transport == "requirements":
        argv = [*shlex.split(tool), "-r", "/tests/deps.txt"]
        requirements = operand + "\n"
    source = shlex.join(argv)
    if transport == "python":
        source = (
            "#!/usr/bin/python3\nimport subprocess\nsubprocess.run("
            + repr(argv)
            + ")\n"
        )
    assert proven(tmp_path, source, requirements) is None


@pytest.mark.parametrize(
    "tool",
    [
        "pip-sync",
        "pip-compile",
        "python3 -m piptools",
        "python3 -m piptools.sync",
        "pipenv",
        "pipx",
    ],
)
@pytest.mark.parametrize(
    "arguments", ["", "--quiet", "/tests/deps.txt", "-r /tests/deps.txt"]
)
def test_unmodeled_installers_include_their_implicit_inputs(tmp_path, tool, arguments):
    """CONF-2 at c05d01a3: no explicit file did not mean no input files."""
    assert proven(tmp_path, f"{tool} {arguments}") is None


@pytest.mark.parametrize(
    "kind",
    ["script", "criteria-runner", "llm-judge", "agent-judge", "ors-episode", "taskmd"],
)
@pytest.mark.parametrize(
    "command",
    [
        "./test.sh",
        "uv pip install -r /app/deps.txt && ./test.sh",
        "time { ./test.sh; }",
        "helper",
    ],
)
def test_selected_strategy_is_never_replaced_by_unselected_scripts(
    tmp_path, kind, command
):
    """CONF-3 at c05d01a3: verifier.md selects the execution entrypoint."""
    assert proven(tmp_path, "uv pip install pytest-json-ctrf") == []
    document = {
        "verifier": {
            "default_strategy": "selected",
            "strategies": {"selected": {"type": kind, "command": command}},
        },
    }
    (tmp_path / "tests/verifier.md").write_text(
        "---\n" + json.dumps(document) + "\n---\n"
    )
    task = SimpleNamespace(paths=SimpleNamespace(tests_dir=tmp_path / "tests"))
    assert lockdown._verifier_dependency_files(task) is None


@pytest.mark.parametrize("contents", ["malformed", "dangling", "directory"])
def test_unreadable_strategy_document_never_grants_script_precision(tmp_path, contents):
    """CONF-3: a failed strategy load is never proof of the fallback script."""
    assert proven(tmp_path, "uv pip install pytest-json-ctrf") == []
    path = tmp_path / "tests/verifier.md"
    if contents == "dangling":
        path.symlink_to("missing.md")
    elif contents == "directory":
        path.mkdir()
    else:
        path.write_text("DUMMY malformed")
    assert (
        lockdown._verifier_dependency_files(
            SimpleNamespace(paths=SimpleNamespace(tests_dir=path.parent))
        )
        is None
    )


@pytest.mark.parametrize(
    "executable",
    [
        "pip",
        "/app/tools/bin/pip",
        "/app/tools/bin/python -m pip",
        "python3 -m pip",
        "python3 -m pip.__main__",
        "pip --python /app/tools/bin/python",
    ],
)
@pytest.mark.parametrize(
    "arguments", ["install pytest-json-ctrf", "install -r /tests/deps.txt"]
)
def test_pip_execution_environment_requires_conservative_proof(
    tmp_path, executable, arguments
):
    """CONF-4 at c05d01a3: bootstrap sys.prefix cannot certify another pip."""
    assert proven(tmp_path, f"{executable} {arguments}") is None


@pytest.mark.parametrize(
    "command",
    [
        "uv add pytest",
        "uv add -r /tests/deps.txt",
        "uv run pytest",
        "python3 -m uv add pytest",
    ],
)
def test_project_commands_never_certify_only_explicit_operands(tmp_path, command):
    """CONF-4 sibling: project dependencies and sources are implicit inputs."""
    assert proven(tmp_path, command) is None


@pytest.mark.parametrize(
    "directory", ["/app", "/opt/project", "/srv/nested", "/tmp/linked"]
)
def test_literal_cwd_carries_its_configuration_into_the_proof(tmp_path, directory):
    """CONF-4 sibling: a supported cd can select configuration outside /app."""
    assert proven(tmp_path, f"cd {directory} && uv pip install pytest") == [
        f"{directory}/pyproject.toml",
        f"{directory}/uv.toml",
    ]


@pytest.mark.parametrize("linked", [False, True])
def test_selected_cwd_configuration_is_not_limited_to_blocked_trees(
    tmp_path, monkeypatch, linked
):
    """Configuration outside the agent's trees is the image's or the verifier's.

    The proof is the agent's files alone: an external uv.toml a
    literal cd selects neither cleans nor taints it, whatever it says; what
    it may point at in the agent's trees is walked like every other file.
    """
    workspace = tmp_path / "app"
    workspace.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (external / "uv.toml").write_text("[pip]\nfind-links = ['/app/wheels']\n")
    cwd = external / "project"
    cwd.mkdir()
    if linked:
        alias = tmp_path / "alias"
        alias.symlink_to(cwd, target_is_directory=True)
        cwd = alias
    proof = tmp_path / "input-proof"
    monkeypatch.setattr(guard, "_BENCHFLOW_BLOCKED", (str(workspace),))
    monkeypatch.setattr(
        guard,
        "_BENCHFLOW_NAMED_INPUTS",
        (str(cwd / "uv.toml"), str(cwd / "pyproject.toml")),
    )
    monkeypatch.setattr(guard, "_BENCHFLOW_INPUT_PROOF", str(proof))
    monkeypatch.setattr(guard, "_record", lambda: {})
    monkeypatch.setattr(guard, "_new_since_record", lambda path, record: False)
    monkeypatch.setattr(guard, "_image_bytes", lambda path: False)
    guard._write_input_proof()
    assert proof.read_text() == "clean\n"
    (workspace / "wheels").mkdir()
    (workspace / "wheels/plugin.whl").write_text("DUMMY agent wheel")
    proof.unlink()
    guard._write_input_proof()
    assert proof.read_text() == "tainted\n"


@pytest.mark.parametrize(
    "name",
    [
        "uv.toml",
        "pyproject.toml",
        "nested/uv.toml",
        "nested/pyproject.toml",
        "../uv.toml",
    ],
)
@pytest.mark.parametrize(
    "contents",
    [
        "[tool.uv.pip]\nfind-links = ['/app/wheels']\n",
        "DUMMY unsupported configuration",
    ],
)
def test_project_configuration_requires_all_file_proof(
    tmp_path, monkeypatch, name, contents
):
    """CONF-4 sibling: unchanged config bytes can name changed source bytes.

    The configuration keeps the image's bytes; the source it names is the
    agent's, and the walk finds it.
    """
    workspace = tmp_path / "app"
    workspace.mkdir()
    config = workspace / name
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(contents)
    (workspace / "wheels").mkdir()
    (workspace / "wheels/plugin.whl").write_text("DUMMY agent wheel")
    proof = tmp_path / "input-proof"
    image = {str(config)}
    monkeypatch.setattr(guard, "_BENCHFLOW_BLOCKED", (str(workspace),))
    monkeypatch.setattr(guard, "_BENCHFLOW_NAMED_INPUTS", ())
    monkeypatch.setattr(guard, "_BENCHFLOW_INPUT_PROOF", str(proof))
    monkeypatch.setattr(guard, "_record", lambda: {})
    monkeypatch.setattr(guard, "_new_since_record", lambda path, record: False)
    monkeypatch.setattr(guard, "_image_bytes", lambda path: path in image)
    guard._write_input_proof()
    assert proof.read_text() == "tainted\n"


@pytest.mark.parametrize("workspace", ["app", "testbed", "root"])
@pytest.mark.parametrize("named", [(), None])
async def test_real_build_snapshot_does_not_taint_supported_install(
    tmp_path, monkeypatch, workspace, named
):
    """CONF-5 at c05d01a3: include the actual pre-agent snapshot producer."""
    fs = tmp_path / "fs"
    live = fs / workspace
    live.mkdir(parents=True)
    (live / "pyproject.toml").write_text(
        '[project]\nname = "dummy"\nversion = "0.0.0"\n'
    )
    # Map the production snapshot location into this fixture's sandbox root.
    snapshot = fs / lockdown._SNAPSHOT_DIR.lstrip("/")
    monkeypatch.setattr(lockdown, "_SNAPSHOT_DIR", str(snapshot))
    monkeypatch.setattr(lockdown, "_SNAPSHOT_MANIFEST", str(snapshot / "manifest.json"))

    async def execute(command, **kwargs):
        import subprocess

        result = subprocess.run(
            ["/bin/sh", "-c", command], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        return SimpleNamespace(stdout=result.stdout)

    # The bash-env inspector integ/py-0.8 adds to this step writes to /etc,
    # which the test user cannot; it is not what this test checks.
    sandbox = SimpleNamespace(exec=execute, _benchflow_bash_env_capture_allowed=False)
    await lockdown._snapshot_build_config(sandbox, str(live))
    assert (snapshot / "pyproject.toml").read_bytes() == (
        live / "pyproject.toml"
    ).read_bytes()
    baseline = fs / "testbed_verify"
    shutil.copytree(live, baseline)
    monkeypatch.setattr(guard, "_BENCHFLOW_BLOCKED", (str(live), str(fs / "tmp")))
    monkeypatch.setattr(guard, "_BENCHFLOW_WORKSPACE", str(live))
    monkeypatch.setattr(guard, "_BENCHFLOW_SNAPSHOT", str(baseline))
    monkeypatch.setattr(guard, "_BENCHFLOW_NAMED_INPUTS", named)
    monkeypatch.setattr(guard, "_private_to_root", lambda path: path == str(baseline))
    monkeypatch.setattr(guard, "_new_since_record", lambda path, record: False)
    assert guard._agent_files_unchanged({})
    # A frozen root-owned agent file still has no image-byte provenance.
    (live / "pyproject.toml").write_text("DUMMY agent change")
    assert not guard._agent_files_unchanged({})


VARIANTS = ["archive", "protected-archive", "pip-sync", "strategy", "site-config"]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("case", ["oracle", "nop", "hostile", "hostile-deleted"])
@pytest.mark.parametrize("workspace", ["app", "testbed", "root"])
@pytest.mark.parametrize("plane", ["same", "separate"])
async def test_confirmation13_through_real_guard(
    tmp_path, monkeypatch, variant, case, workspace, plane
):
    """c05d01a3 regression: real guard/pytest/scoring, modeled install/owners."""
    proof = tmp_path / "hostile-imported"
    original = uv_state.make_task

    def make_task(layout, env=None):
        interpreter = f"{layout.workspace}/.venv/bin/python"
        command = {
            "archive": "uv pip install pytest_json_ctrf-0.3.5-py3-none-any.whl",
            "protected-archive": "uv pip install -r /tests/deps.txt",
            "pip-sync": "pip-sync",
            "strategy": "true",
            "site-config": f"{layout.workspace}/tools/bin/python -m pip install pytest-json-ctrf",
        }[variant]
        task = original(
            layout,
            test_sh=f"{command}\n{interpreter} -m pytest --ctrf /logs/verifier/ctrf.json /opt/checks/test_outputs.py\n",
            env=env,
        )
        # Trusted tests live outside this input-discovery fixture, so their
        # definitions cannot accidentally force the conservative path.
        if variant == "protected-archive":
            (task.paths.tests_dir / "deps.txt").write_text(
                "pytest_json_ctrf-0.3.5-py3-none-any.whl\n"
            )
        if variant == "strategy":
            document = {
                "verifier": {
                    "default_strategy": "selected",
                    "strategies": {
                        "selected": {
                            "type": "script",
                            "command": "uv pip install -r /app/deps.txt && ./test.sh",
                        }
                    },
                }
            }
            (task.paths.tests_dir / "verifier.md").write_text(
                "---\n" + json.dumps(document) + "\n---\n"
            )
        task.config.verifier.env = {}
        return task

    monkeypatch.setattr(uv_state, "make_task", make_task)

    def edit(layout):
        names = {
            "archive": ["pytest_json_ctrf-0.3.5-py3-none-any.whl"],
            "protected-archive": ["pytest_json_ctrf-0.3.5-py3-none-any.whl"],
            "pip-sync": ["requirements.txt", "wheels/plugin.whl"],
            "strategy": ["deps.txt", "wheels/plugin.whl"],
            "site-config": ["tools/pip.conf", "wheels/plugin.whl"],
        }[variant]
        for name in names:
            path = layout.workspace / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("DUMMY agent input")

    def install(env, layout):
        site = layout.workspace / ".venv/lib/python3.12/site-packages"
        install_plugin(
            site, hostile_ctrf(proof) if case.startswith("hostile") else CTRF_PLUGIN
        )
        (layout.workspace / ".venv/pyvenv.cfg").write_text("home = /usr/bin\n")
        if case == "hostile-deleted":
            for path in layout.workspace.rglob("*"):
                if path.is_file() and ".venv" not in path.parts:
                    path.unlink()
        return [site]

    # The strategy is used for discovery; ScriptSandbox models the trusted
    # command's install. Strategy parsing/execution has its own production tests.
    layout, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        PASSING if case == "oracle" else FAILING,
        install,
        before=edit if case.startswith("hostile") else None,
        workspace_name=workspace,
        agent_paths=("{workspace}",) if plane == "separate" else None,
    )
    assert not proof.exists()
    if case.startswith("hostile"):
        assert plugin_never_ran(rewards, error), (rewards, error)
        assert markers(layout, "installed") == []
    else:
        assert rewards == {"reward": float(case == "oracle")} and error is None, (
            rewards,
            error,
        )
