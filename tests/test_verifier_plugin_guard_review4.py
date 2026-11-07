"""Guard V2-R3-CONF-1..4 against the parser regressions at 2860f8f6."""

import shlex
from types import SimpleNamespace

import pytest

from benchflow.sandbox import lockdown
from tests import test_verifier_uv_state as uv_state
from tests.test_verifier_plugin_guard_review1 import named
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


@pytest.mark.parametrize("shell", ["sh", "bash"])
@pytest.mark.parametrize("flag", ["-r", "-c"])
def test_wrapped_quoted_interpreter_keeps_pip_inputs(tmp_path, shell, flag):
    """Guards CONF-1 at 2860f8f6: an incomplete prefix cannot discard pip inputs."""
    assert named(
        tmp_path,
        f"{shell} -c 'uv pip install --python \"/app/with space/bin/python\" {flag} /app/deps.txt pytest'\n",
    ) == ["/app/deps.txt"]


@pytest.mark.parametrize(
    "expression",
    [
        'APP + "/requirements.txt"',
        '"%s/requirements.txt" % APP',
        '"{}/requirements.txt".format(APP)',
        '"".join([APP, "/requirements.txt"])',
        'f"{APP}/requirements.txt"',
    ],
)
@pytest.mark.parametrize(
    "wrapper",
    [
        "heredoc",
        "command-string",
        "isolated",
        "unbuffered",
        "combined",
        "attached",
        "warning-option",
        "x-option",
        "nested-shell",
    ],
)
def test_embedded_python_paths_follow_expression_semantics(
    tmp_path, expression, wrapper
):
    """Guards CONF-2 at 2860f8f6 and the flag/wrapper gaps left at e8e897f8."""
    source = (
        f'APP = "/app"\nsubprocess.run(["uv", "pip", "install", "-r", {expression}])\n'
    )
    if wrapper == "heredoc":
        script = "#!/bin/sh\npython3 - <<'PY'\n" + source + "PY\n"
    else:
        command = {
            "command-string": "python3 -c",
            "isolated": "python3 -I -c",
            "unbuffered": "python3 -u -c",
            "combined": "python3 -Ic",
            "attached": "python3 -c",
            "warning-option": "python3 -W error -c",
            "x-option": "python3 -X dev -c",
            "nested-shell": "python3 -I -c",
        }[wrapper]
        script = command + ("" if wrapper == "attached" else " ") + shlex.quote(source)
        if wrapper == "nested-shell":
            script = "sh -ec " + shlex.quote(script)
        script = "#!/bin/sh\n" + script + "\n"
    assert named(tmp_path, script) == ["requirements.txt"]


@pytest.mark.parametrize("filename", ["verify.py", "verify"])
@pytest.mark.parametrize("flag", ["-r", "-c", "--requirement", "--constraint"])
@pytest.mark.parametrize("argument", ["", "args="])
def test_complete_python_literal_keeps_its_directory(
    tmp_path, filename, flag, argument
):
    """Guards CONF-3 at 2860f8f6: unrelated workspace files must keep their pass."""
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / filename).write_text(
        "#!/usr/bin/env python3\n"
        f'subprocess.run({argument}["uv", "pip", "install", "{flag}", "/tests/requirements.txt"])\n'
    )
    task = SimpleNamespace(paths=SimpleNamespace(tests_dir=tests))
    assert lockdown._verifier_named_dependency_files(task) == [
        "/tests/requirements.txt"
    ]


@pytest.mark.parametrize(
    "command", ["python3 -m pip.__main__", "/app/.venv/bin/python -m pip.__main__"]
)
@pytest.mark.parametrize("flag", ["-r", "-c"])
def test_pip_main_module_keeps_custom_inputs(tmp_path, command, flag):
    """Guards CONF-4 at 2860f8f6 without restoring the pipeline substring match."""
    assert named(tmp_path, command + f" install {flag} /app/deps.txt\n") == [
        "/app/deps.txt"
    ]


@pytest.mark.parametrize(
    "wrapper",
    ["python-file", "heredoc", "tabbed-heredoc", "command-string", "isolated"],
)
@pytest.mark.parametrize(
    "expression,expected",
    [
        ('"/tests/deps.txt"', "/tests/deps.txt"),
        ('APP + "/deps.txt"', "deps.txt"),
    ],
)
def test_python_custom_inputs_keep_literal_and_expression_distinction(
    tmp_path, wrapper, expression, expected
):
    """Guards CONF-2..4's class for custom names, module flags and Python wrappers."""
    source = f'subprocess.run(["python3", "-I", "-m", "pip.__main__", "install", "-r", {expression}])\n'
    if wrapper == "python-file":
        script = "#!/usr/bin/env python3\n" + source
    elif wrapper == "command-string":
        script = "python3 -c " + shlex.quote(source) + "\n"
    elif wrapper == "isolated":
        script = "python3 -I -c " + shlex.quote(source) + "\n"
    elif wrapper == "tabbed-heredoc":
        script = "python3 - <<-'CODE'\n\t" + source + "\tCODE\n"
    else:
        script = "python3 - <<'CODE'\n" + source + "CODE\n"
    assert named(tmp_path, script) == [expected]


@pytest.mark.parametrize(
    "variant",
    [
        "quoted-shell",
        "heredoc",
        "command-string-flags",
        "literal-python",
        "literal-python-keyword",
        "pip-main",
    ],
)
@pytest.mark.parametrize("case", ["oracle", "nop", "hostile"])
async def test_confirmed_inputs_through_real_guard(
    tmp_path, monkeypatch, variant, case
):
    """Guards CONF-1..4 at 2860f8f6 with real hardening, pytest and dummy markers."""
    proof = tmp_path / "hostile-imported"
    filename = (
        "requirements.txt"
        if variant
        in (
            "heredoc",
            "command-string-flags",
            "literal-python",
            "literal-python-keyword",
        )
        else "deps.txt"
    )
    original_make_task = uv_state.make_task

    def make_task(layout, env=None):
        interpreter = f"{layout.workspace}/.venv/bin/python"
        dependency = f"{layout.workspace}/{filename}"
        argument = "args=" if variant == "literal-python-keyword" else ""
        python_script = (
            f"APP = {str(layout.workspace)!r}\n"
            f'subprocess.run({argument}["uv", "pip", "install", "--python", '
            f'{interpreter!r}, "-r", '
        )
        if variant == "quoted-shell":
            script = f"sh -c 'uv pip install --python \"{layout.workspace}/with space/bin/python\" -c {dependency} pytest pytest-json-ctrf'\n"
        elif variant == "heredoc":
            script = (
                "python3 - <<'PY'\n"
                + python_script
                + 'APP + "/requirements.txt"], check=True)\nPY\n'
            )
        elif variant in ("literal-python", "literal-python-keyword"):
            script = "python3 /tests/verify.py\n"
        elif variant == "command-string-flags":
            script = (
                "python3 -I -c "
                + shlex.quote(python_script + 'APP + "/requirements.txt"], check=True)')
                + "\n"
            )
        else:
            script = f"{interpreter} -m pip.__main__ install -r {dependency}\n"
        task = original_make_task(
            layout,
            test_sh=f"uv venv {layout.workspace}/.venv\n"
            + script
            + f"{interpreter} -m pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py\n",
            env=env,
        )
        if variant in ("literal-python", "literal-python-keyword"):
            path = dependency if case == "hostile" else "/tests/requirements.txt"
            (task.paths.tests_dir / "verify.py").write_text(
                python_script + f"{path!r}], check=True)\n"
            )
        return task

    monkeypatch.setattr(uv_state, "make_task", make_task)

    def edit_workspace(layout):
        # The literal-path honest controls deliberately have a colliding answer file.
        (layout.workspace / filename).write_text("ctrf @ file:///dummy-wheel\n")

    def place_plugin(env, layout):
        site = layout.workspace / ".venv/lib/python3.12/site-packages"
        install_plugin(site, hostile_ctrf(proof) if case == "hostile" else CTRF_PLUGIN)
        (layout.workspace / ".venv/pyvenv.cfg").write_text("home = /usr/bin\n")
        return [site]

    layout, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        PASSING if case == "oracle" else FAILING,
        place_plugin,
        before=edit_workspace
        if case == "hostile" or variant in ("literal-python", "literal-python-keyword")
        else None,
        workspace_name="app",
    )
    assert not proof.exists()
    # Script helpers/assignments are outside the closed provenance grammar;
    # their colliding workspace file is therefore a possible input too.
    # Honest bytes load however the install was spelled.
    if case == "hostile":
        assert plugin_never_ran(rewards, error), (rewards, error)
        assert markers(layout, "installed") == []
    else:
        assert rewards == {"reward": float(case == "oracle")} and error is None, (
            rewards,
            error,
        )


@pytest.mark.parametrize(
    "wrapper",
    [
        "argv = {argv}\nsubprocess.run(transform(argv))",
        'subprocess.run([arg.replace("/tests", APP) for arg in {argv}])',
        "subprocess.run(transform({argv}))",
    ],
)
def test_transformed_python_argv_does_not_grant_literal_path_trust(tmp_path, wrapper):
    """Guards CONF-2/3's precision boundary: a surrounding expression can change argv."""
    argv = '["uv", "pip", "install", "-r", "/tests/requirements.txt"]'
    assert named(
        tmp_path, "#!/usr/bin/env python3\n" + wrapper.format(argv=argv) + "\n"
    ) == ["requirements.txt"]
