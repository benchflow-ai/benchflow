"""Fail closed for V2-R11-CONF-1/2/3 and unmodeled dependency commands."""

from types import SimpleNamespace

import pytest

from benchflow.sandbox import _pytest_plugin_guard as guard
from benchflow.sandbox import lockdown
from tests import test_verifier_uv_state as uv_state
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

EXPRESSIONS = [
    '"-r{path}".format(**{"path": PATH})',
    '"-r{}".format(*(PATH,))',
    '"-r{path}".format_map({**{"path": PATH}})',
    '"-r%.13s" % (PATH + ".extra")',
    '"-r{:.13}".format(PATH + ".extra")',
    '"-r%*.*s" % (13, 13, PATH + ".extra")',
    '"-r{path:.13}".format_map({"path": PATH + ".extra"})',
    '"-r" + PATH',
    'f"-r{PATH}"',
    '"".join(("-r", PATH))',
    "get_argument()",
]
SHELL = [
    "time {{ {command}; }}",
    "time -p {{ {command}; }}",
    "time -p {command}",
    "{{ {command}; }}",
    "helper --deps {path}",
    "eval 'pip install -r {path}'",
    "pip install --future-option -r {path}",
]


def proven(tmp_path, source, requirements="pytest-json-ctrf==0.3.5\n"):
    tests = tmp_path / "tests"
    tests.mkdir(exist_ok=True)
    (tests / "test.sh").write_text(source)
    (tests / "deps.txt").write_text(requirements)
    return lockdown._verifier_dependency_files(
        SimpleNamespace(paths=SimpleNamespace(tests_dir=tests))
    )


@pytest.mark.parametrize("expression", EXPRESSIONS)
def test_computed_arguments_have_no_provenance(tmp_path, expression):
    source = (
        "#!/usr/bin/python3\nimport subprocess, sys\n"
        f"subprocess.run([sys.executable, '-mpip', 'install', {expression}])\n"
    )
    assert proven(tmp_path, source) is None


@pytest.mark.parametrize("shell", SHELL)
@pytest.mark.parametrize(
    "tool", ["pip install", "python3 -mpip install", "uv pip install"]
)
def test_unmodeled_shell_taints_all_inputs(tmp_path, shell, tool):
    command = f"{tool} -r /app/deps.txt"
    assert proven(tmp_path, shell.format(command=command, path="/app/deps.txt")) is None


@pytest.mark.parametrize(
    "tool", ["pip install", "python3 -mpip install", "uv pip install", "uv add"]
)
@pytest.mark.parametrize(
    "option", ["-r /tests/deps.txt", "-r/tests/deps.txt", "--requirem=/tests/deps.txt"]
)
@pytest.mark.parametrize("language", ["shell", "python"])
def test_supported_literal_commands_remain_precise(tmp_path, tool, option, language):
    import shlex

    argv = shlex.split(f"{tool} {option}")
    source = shlex.join(argv)
    if language == "python":
        source = (
            "#!/usr/bin/python3\nimport subprocess\nsubprocess.run("
            + repr(argv)
            + ", check=True)\n"
        )
    # R12-CONF-4: pip's executing interpreter/configuration cannot be
    # established from these source bytes. Only uv pip retains precise inputs.
    assert proven(tmp_path, source) == (
        ["/tests/deps.txt"] if tool == "uv pip install" else None
    )


@pytest.mark.parametrize(
    "requirements",
    [
        "-r /app/nested/arbitrary.data\n",
        "ctrf @ file:///app/plugin.whl\n",
        "--find-links /app/wheels\n",
        "./local-project\n",
        "${INPUT}\n",
    ],
)
def test_transitive_file_inputs_are_unknown(tmp_path, requirements):
    assert proven(tmp_path, "uv pip install -r /tests/deps.txt", requirements) is None


@pytest.mark.parametrize(
    "command",
    [
        "pip install --require=/tests/deps.txt",
        "pip install --future=/tests/deps.txt",
        "pip install -Zr/tests/deps.txt",
        "pip install -r",
        "pip install --requirement",
        "pip install --find-links /app/wheels pytest-json-ctrf",
        "uv --project /app pip install pytest-json-ctrf",
        "uv run --with-requirements /tests/deps.txt pytest",
        "uv pip install --config-file /app/custom.toml pytest-json-ctrf",
        "pip install ./local-project",
        "pip install ctrf@file:///app/dummy.whl",
        "env pip install -r /tests/deps.txt",
        "command pip install -r /tests/deps.txt",
        'pip install $"-r/tests/deps.txt"',
        "pip install $'-r/tests/deps.txt'",
        "pip install -r /app/custom.data",
        "pip install -r ../tests/deps.txt",
        "python3 -c 'import pip'",
        "bash -c 'pip install -r /tests/deps.txt'",
    ],
)
def test_every_unmodeled_option_source_and_transport_is_unknown(tmp_path, command):
    assert proven(tmp_path, command) is None


@pytest.mark.parametrize("source", ["binary", "oversized", "symlink"])
def test_unreadable_or_skipped_sources_cannot_prove_inputs(tmp_path, source):
    assert proven(tmp_path, "uv pip install pytest-json-ctrf") == []
    path = tmp_path / "tests/helper.py"
    if source == "binary":
        path.write_bytes(b"\0")
    elif source == "oversized":
        path.write_text("#" * (lockdown._VERIFIER_SCRIPT_MAX_BYTES + 1))
    else:
        path.symlink_to("missing-source.py")
    task = SimpleNamespace(paths=SimpleNamespace(tests_dir=path.parent))
    assert lockdown._verifier_dependency_files(task) is None


@pytest.mark.parametrize(
    "name", ["deps.txt", ".hidden", "arbitrary.data", "limits.in", "answer.py"]
)
def test_unknown_means_every_filename(tmp_path, monkeypatch, name):
    """Fails on 297f151c: None previously matched conventional names alone."""
    workspace = tmp_path / "app"
    workspace.mkdir()
    (workspace / name).write_text("DUMMY")
    monkeypatch.setattr(guard, "_BENCHFLOW_BLOCKED", (str(workspace),))
    monkeypatch.setattr(guard, "_new_since_record", lambda path, record: False)
    monkeypatch.setattr(guard, "_image_bytes", lambda path: False)
    assert not guard._agent_files_unchanged({})


@pytest.mark.parametrize("tree", ["workspace", "artifact", "temporary"])
@pytest.mark.parametrize("name", ["deps.txt", "nested/.data", "nested/deep/limits.bin"])
def test_unknown_checks_every_tree_recursively(tmp_path, monkeypatch, tree, name):
    roots = [tmp_path / value for value in ["workspace", "artifact", "temporary"]]
    for root in roots:
        root.mkdir()
    path = tmp_path / tree / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("DUMMY")
    monkeypatch.setattr(guard, "_BENCHFLOW_BLOCKED", tuple(map(str, roots)))
    monkeypatch.setattr(guard, "_BENCHFLOW_NAMED_INPUTS", None)
    monkeypatch.setattr(guard, "_new_since_record", lambda path, record: False)
    monkeypatch.setattr(guard, "_image_bytes", lambda path: False)
    assert not guard._agent_files_unchanged({})


@pytest.mark.parametrize("failure", ["missing-record", "walk-error"])
def test_prearm_proof_fails_closed(tmp_path, monkeypatch, failure):
    proof = tmp_path / "input-proof"
    monkeypatch.setattr(guard, "_BENCHFLOW_INPUT_PROOF", str(proof))
    monkeypatch.setattr(
        guard, "_record", lambda: None if failure == "missing-record" else {}
    )

    def unreadable(record):
        raise OSError("DUMMY unreadable tree")

    monkeypatch.setattr(guard, "_agent_files_unchanged", unreadable)
    guard._write_input_proof()
    assert proof.read_text() == "tainted\n"


@pytest.mark.parametrize(
    "key", ["PIP_REQUIREMENT", "PIP_CONSTRAINT", "UV_CONSTRAINT", "UV_FUTURE_INPUT"]
)
@pytest.mark.parametrize("changed", [True, False])
def test_environment_inputs_cannot_keep_literal_precision(
    tmp_path, monkeypatch, key, changed
):
    """The proof is the agent's files alone: an installer variable
    naming one can neither clean a changed file nor taint an unchanged one."""
    proof = tmp_path / "input-proof"
    monkeypatch.setattr(guard, "_BENCHFLOW_INPUT_PROOF", str(proof))
    monkeypatch.setattr(guard, "_BENCHFLOW_NAMED_INPUTS", ())
    monkeypatch.setattr(guard, "_record", lambda: {})
    monkeypatch.setenv(key, "/app/deps.txt")
    monkeypatch.setattr(guard, "_agent_files_unchanged", lambda record: not changed)
    guard._write_input_proof()
    assert proof.read_text() == ("tainted\n" if changed else "clean\n")


def test_installer_configuration_has_unknown_transitive_inputs(tmp_path, monkeypatch):
    """A configuration file naming an agent file decides nothing; the agent's
    changed file does."""
    workspace = tmp_path / "app"
    workspace.mkdir()
    (workspace / "deps.txt").write_text("DUMMY agent change\n")
    config = tmp_path / "pip.conf"
    config.write_text(f"[install]\nrequirement = {workspace}/deps.txt\n")
    proof = tmp_path / "input-proof"
    monkeypatch.setattr(guard, "_BENCHFLOW_BLOCKED", (str(workspace),))
    monkeypatch.setattr(guard, "_BENCHFLOW_INPUT_PROOF", str(proof))
    monkeypatch.setattr(guard, "_BENCHFLOW_NAMED_INPUTS", ())
    monkeypatch.setattr(guard, "_record", lambda: {})
    monkeypatch.setattr(guard, "_new_since_record", lambda path, record: False)
    monkeypatch.setattr(guard, "_image_bytes", lambda path: False)
    monkeypatch.setenv("PIP_CONFIG_FILE", str(config))
    guard._write_input_proof()
    assert proof.read_text() == "tainted\n"


@pytest.mark.parametrize(
    "key",
    [
        "HOME",
        "XDG_CONFIG_HOME",
        "XDG_CONFIG_DIRS",
        "PIP_REQUIREMENT",
        "UV_FUTURE_INPUT",
    ],
)
def test_task_environment_overlays_are_outside_literal_model(tmp_path, key):
    assert proven(tmp_path, "uv pip install pytest-json-ctrf") == []
    task = SimpleNamespace(
        paths=SimpleNamespace(tests_dir=tmp_path / "tests"),
        config=SimpleNamespace(verifier=SimpleNamespace(env={key: "/app/inputs"})),
    )
    assert lockdown._verifier_dependency_files(task) is None


@pytest.mark.parametrize(
    "contents", [None, "", "clean", "clean\ntruncated", "tainted\n"]
)
def test_only_complete_clean_proof_grants_trust(tmp_path, monkeypatch, contents):
    proof = tmp_path / "input-proof"
    if contents is not None:
        proof.write_text(contents)
    monkeypatch.setattr(guard, "_BENCHFLOW_INPUT_PROOF", str(proof))
    monkeypatch.setattr(guard, "ownership_problem", lambda path, st: None)
    monkeypatch.setattr(guard, "_root_alone", lambda directory: True)
    assert not guard._input_proof()


VARIANTS = [
    "time",
    "time-p",
    "unpack-keyword",
    "unpack-positional",
    "unpack-map",
    "percent-precision",
    "format-precision",
    "unknown-helper",
]


@pytest.mark.parametrize("workspace", ["app", "testbed", "root"])
@pytest.mark.parametrize("plane", ["same", "separate"])
@pytest.mark.parametrize("case", ["oracle", "nop"])
async def test_supported_install_ignores_unrelated_answer(
    tmp_path, monkeypatch, workspace, plane, case
):
    original = uv_state.make_task

    def make_task(layout, env=None):
        interpreter = f"{layout.workspace}/.venv/bin/python"
        task = original(
            layout,
            test_sh=f"uv venv {layout.workspace}/.venv\nuv pip install --python {interpreter} -r /tests/deps.txt\n{interpreter} -m pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py\n",
            env=env,
        )
        (task.paths.tests_dir / "deps.txt").write_text("pytest-json-ctrf==0.3.5\n")
        # The shared helper supplies HOME; precise provenance supports an
        # empty task overlay and uses the harness's fixed verifier HOME.
        task.config.verifier.env = {}
        return task

    monkeypatch.setattr(uv_state, "make_task", make_task)

    def edit(layout):
        (layout.workspace / "answer.py").write_text("DUMMY agent answer")

    def install(env, layout):
        site = layout.workspace / ".venv/lib/python3.12/site-packages"
        install_plugin(site, CTRF_PLUGIN)
        (layout.workspace / ".venv/pyvenv.cfg").write_text("home = /usr/bin\n")
        return [site]

    _, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        PASSING if case == "oracle" else FAILING,
        install,
        before=edit,
        workspace_name=workspace,
        agent_paths=("{workspace}",) if plane == "separate" else None,
    )
    assert rewards == {"reward": float(case == "oracle")} and error is None, (
        rewards,
        error,
    )


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("case", ["oracle", "nop", "hostile", "hostile-deleted"])
@pytest.mark.parametrize("workspace", ["app", "testbed", "root"])
@pytest.mark.parametrize("plane", ["same", "separate"])
async def test_confirmation12_through_real_guard(
    tmp_path, monkeypatch, variant, case, workspace, plane
):
    """Real hardening, guard, pytest and scoring; dummy installation/ownership."""
    proof = tmp_path / "hostile-imported"
    original = uv_state.make_task

    def make_task(layout, env=None):
        interpreter = f"{layout.workspace}/.venv/bin/python"
        path = f"{layout.workspace}/inputs/deps.txt"
        if variant in {"time", "time-p"}:
            body = (
                "time -p" if variant == "time-p" else "time"
            ) + f" {{ uv pip install --python {interpreter} -r {path}; }}\n"
        elif variant == "unknown-helper":
            body = f"helper --deps {path}\n"
        else:
            expression = {
                "unpack-keyword": f'"-r{{path}}".format(**{{"path": {path!r}}})',
                "unpack-positional": f'"-r{{}}".format(*({path!r},))',
                "unpack-map": f'"-r{{path}}".format_map({{**{{"path": {path!r}}}}})',
                "percent-precision": f'"-r%.{len(path)}s" % {path + ".extra"!r}',
                "format-precision": f'"-r{{:.{len(path)}}}".format({path + ".extra"!r})',
            }[variant]
            # A Python file is supported as a transport; its computed argv is
            # deliberately unknown, independent of shell/heredoc recovery.
            body = f"{interpreter} /tests/install.py\n"
            installer = f"import subprocess, sys\nsubprocess.run([sys.executable, '-mpip', 'install', {expression}])\n"
        task = original(
            layout,
            test_sh=f"uv venv {layout.workspace}/.venv\n"
            + body
            + f"{interpreter} -m pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py\n",
            env=env,
        )
        if variant not in {"time", "time-p", "unknown-helper"}:
            (task.paths.tests_dir / "install.py").write_text(installer)
        return task

    monkeypatch.setattr(uv_state, "make_task", make_task)

    def edit(layout):
        inputs = layout.workspace / "inputs"
        inputs.mkdir()
        (inputs / "deps.txt").write_text("ctrf @ file:///dummy-wheel\n")

    def install(env, layout):
        site = layout.workspace / ".venv/lib/python3.12/site-packages"
        install_plugin(
            site, hostile_ctrf(proof) if case.startswith("hostile") else CTRF_PLUGIN
        )
        (layout.workspace / ".venv/pyvenv.cfg").write_text("home = /usr/bin\n")
        if case == "hostile-deleted":
            (layout.workspace / "inputs/deps.txt").unlink()
        return [site]

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
