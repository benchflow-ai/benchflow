"""Guard the named-input regressions in 89b3ac3e."""

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


@pytest.mark.parametrize(
    "selector",
    [
        "--python /app/.venv/bin/python",
        "-p /app/.venv/bin/python3",
        "--python=/app/.venv/bin/python3.12",
        '--python "/app/with space/bin/python"',
    ],
)
@pytest.mark.parametrize("flag", ["-r", "-c"])
def test_pip_inputs_after_an_interpreter_selection_are_named(tmp_path, selector, flag):
    """Guards V2-R3-A-1 from 89b3ac3e, including constraints and quoted paths."""
    assert named(tmp_path, f"uv pip install {selector} {flag} /app/deps.txt\n") == [
        "/app/deps.txt"
    ]


@pytest.mark.parametrize(
    "expression",
    [
        'APP + "/requirements.txt"',
        'APP+"/requirements.txt"',
        '"%s/requirements.txt" % APP',
        '"{}/requirements.txt".format(APP)',
        '"".join([APP, "/requirements.txt"])',
        'f"{APP}/requirements.txt"',
        '"/app/requirements.txt"',
    ],
)
@pytest.mark.parametrize("filename", ["verify.py", "verify"])
def test_python_paths_are_conservatively_named_by_basename(
    tmp_path, expression, filename
):
    """Guards V2-R3-B-1 and CONF-3: expressions are conservative, literals precise."""
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / filename).write_text(
        '#!/usr/bin/env python3\nAPP = "/app"\n'
        f'subprocess.run(["uv", "pip", "install", "-r", {expression}])\n'
    )
    task = SimpleNamespace(paths=SimpleNamespace(tests_dir=tests))
    expected = (
        "/app/requirements.txt"
        if expression == '"/app/requirements.txt"'
        else "requirements.txt"
    )
    assert lockdown._verifier_named_dependency_files(task) == [expected]


@pytest.mark.parametrize("variant", ["interpreter", "python-expression"])
@pytest.mark.parametrize("case", ["oracle", "nop", "hostile"])
async def test_named_input_blocks_a_hostile_install_but_keeps_controls(
    tmp_path, monkeypatch, variant, case
):
    """Guards both round-3 findings end to end through hardening and real pytest."""
    proof = tmp_path / "hostile-imported"
    filename = (
        "test-requirements.txt" if variant == "interpreter" else "requirements.txt"
    )
    original_make_task = uv_state.make_task

    def make_task(layout, env=None):
        if variant == "interpreter":
            script = f"uv venv {layout.workspace}/.venv\nuv pip install --python {layout.workspace}/.venv/bin/python -r {layout.workspace}/{filename}\nuv run pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py\n"
            return original_make_task(layout, test_sh=script, env=env)
        task = original_make_task(
            layout,
            test_sh="python3 /tests/verify.py\nuv run pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py\n",
            env=env,
        )
        (task.paths.tests_dir / "verify.py").write_text(
            f'APP = {str(layout.workspace)!r}\nsubprocess.run(["uv", "pip", "install", "-r", APP + "/requirements.txt"])\n'
        )
        return task

    monkeypatch.setattr(uv_state, "make_task", make_task)

    def edit_requirements(layout):
        (layout.workspace / filename).write_text("ctrf @ file:///evil\n")

    def install_hostile(env, layout):
        site = layout.workspace / ".venv/lib/python3.12/site-packages"
        install_plugin(site, hostile_ctrf(proof) if case == "hostile" else CTRF_PLUGIN)
        (layout.workspace / ".venv/pyvenv.cfg").write_text("home = /usr/bin\n")
        return [site]

    layout, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        PASSING if case == "oracle" else FAILING,
        install_hostile,
        before=edit_requirements if case == "hostile" else None,
        workspace_name="app",
    )
    assert not proof.exists()
    if case == "hostile":
        assert plugin_never_ran(rewards, error), (rewards, error)
        assert markers(layout, "installed") == []
    else:
        assert rewards == {"reward": float(case == "oracle")} and error is None, (
            rewards,
            error,
        )


@pytest.mark.parametrize(
    "command",
    [
        "python3 /tests/check_pipeline.py -c config.yaml",
        "pytest tests/test_pipeline.py -c pytest.ini",
        'python3 -c "import pip"',
        'sh -c "pip --version"',
    ],
)
def test_non_dependency_command_words_do_not_name_files(tmp_path, command):
    """Guards V2-R3-B-2: 'pipeline' is not a pip command word."""
    assert named(tmp_path, command + "\n") == []


@pytest.mark.parametrize(
    "command",
    ["pip3.12", "/usr/bin/pip", "python3 -m pip", "uv pip", "pip-sync", "pipenv"],
)
def test_dependency_command_siblings_still_name_custom_inputs(tmp_path, command):
    """Guards the dependency-command closure while narrowing 'pip'."""
    assert named(tmp_path, command + " install -r /app/deps.txt\n") == ["/app/deps.txt"]
