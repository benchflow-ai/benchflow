"""Guard V2-R4-CONF-1..3 against the scanner regressions at a4c87246."""

import shlex

import pytest

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


@pytest.mark.parametrize("separator", [" && ", ";", " || ", " | "])
@pytest.mark.parametrize("flag", ["-r", "-c", "--requirement"])
def test_installer_before_python_heredoc_is_not_removed(tmp_path, separator, flag):
    """Guards V2-R4-CONF-1 at a4c87246, including adjacent shell separators."""
    script = (
        f"uv pip install --python /app/.venv/bin/python {flag} /app/deps.txt"
        + separator
        + "python3 - <<'PY'\nprint('ready')\nPY\n"
    )
    assert named(tmp_path, script) == ["/app/deps.txt"]


@pytest.mark.parametrize(
    "command",
    [
        "python3 \\\n  -c ",
        "python3 -I \\\n  -c ",
        "# here's the verifier\npython3 \\\n  -c ",
        "cd /app;python3 -c ",
        "true&&python3 -c ",
        "false||python3 -c ",
        "echo ready|python3 -c ",
        "(python3 -c ",
    ],
)
@pytest.mark.parametrize("operand", ['APP + "/requirements.txt"', 'APP + "/deps.txt"'])
def test_shell_grammar_does_not_turn_python_fragments_into_absolute_paths(
    tmp_path, command, operand
):
    """Guards V2-R4-CONF-2 at a4c87246 for continuations and punctuation."""
    source = f'subprocess.run(["uv", "pip", "install", "-r", {operand}])'
    script = command + shlex.quote(source) + (")" if command.startswith("(") else "")
    assert named(tmp_path, script) == [
        "requirements.txt" if "requirements" in operand else "deps.txt"
    ]


@pytest.mark.parametrize("separator", [";", "&&", "||", "|", "\n"])
def test_python_payload_mask_keeps_the_next_shell_command(tmp_path, separator):
    """Guards V2-R4-CONF-2's lookahead boundary at a4c87246."""
    source = 'subprocess.run(["uv", "pip", "install", "-r", APP + "/deps.txt"])'
    assert named(
        tmp_path,
        "python3 -c "
        + shlex.quote(source)
        + separator
        + "uv pip install -r /app/other.txt\n",
    ) == ["/app/other.txt", "deps.txt"]


@pytest.mark.parametrize(
    "imports,call,interpreter",
    [
        ("import subprocess, sys", "subprocess.run", "sys.executable"),
        ("import subprocess as sp, sys as runtime", "sp.run", "runtime.executable"),
        (
            "from subprocess import run as invoke\nfrom sys import executable as python",
            "invoke",
            "python",
        ),
        ("import subprocess as sp", "sp.check_call", '"python3"'),
        ("from subprocess import Popen", "Popen", '"python3"'),
    ],
)
@pytest.mark.parametrize("argument", ["", "args="])
@pytest.mark.parametrize("module", ["pip", "pip.__main__"])
def test_static_python_installer_aliases_keep_complete_protected_operands(
    tmp_path, imports, call, interpreter, argument, module
):
    """Guards V2-R4-CONF-3 at a4c87246 without evaluating verifier expressions."""
    source = (
        f"#!/usr/bin/env python3\n{imports}\n"
        f'{call}({argument}[{interpreter}, "-I", "-m", "{module}", '
        '"install", "-r", "/tests/requirements.txt"])\n'
    )
    assert named(tmp_path, source) == ["/tests/requirements.txt"]


@pytest.mark.parametrize(
    "source",
    [
        "import subprocess as sp\nsp = transform(sp)\nsp.run({argv})",
        "from subprocess import run as invoke\ninvoke = transform(invoke)\ninvoke({argv})",
        "import subprocess as sp\ndef f(sp):\n    sp.run({argv})",
        "import subprocess as sp\nsp.run = transform\nsp.run({argv})",
        "import subprocess as sp\nsp.run(transform({argv}))",
        'import subprocess as sp\nsp.run([arg.replace("/tests", APP) for arg in {argv}])',
    ],
)
def test_shadowed_aliases_and_transformed_argv_remain_conservative(tmp_path, source):
    """Guards V2-R4-CONF-3's boundary: an import is not authority after rebinding."""
    argv = '["python3", "-m", "pip", "install", "-r", "/tests/requirements.txt"]'
    assert named(tmp_path, "#!/usr/bin/env python3\n" + source.format(argv=argv)) == [
        "requirements.txt"
    ]


@pytest.mark.parametrize(
    "variant",
    ["heredoc-prefix", "continuation", "separator", "sys-executable", "alias"],
)
@pytest.mark.parametrize("case", ["oracle", "nop", "hostile"])
async def test_confirmation2_inputs_through_real_guard(
    tmp_path, monkeypatch, variant, case
):
    """Guards V2-R4-CONF-1..3 at a4c87246 with hardening, pytest and dummy markers."""
    proof = tmp_path / "hostile-imported"
    literal = variant in ("sys-executable", "alias")
    filename = "deps.txt" if variant == "heredoc-prefix" else "requirements.txt"
    original_make_task = uv_state.make_task

    def make_task(layout, env=None):
        interpreter = f"{layout.workspace}/.venv/bin/python"
        dependency = f"{layout.workspace}/{filename}"
        source = (
            f"import subprocess, sys\nAPP = {str(layout.workspace)!r}\n"
            'subprocess.run(["uv", "pip", "install", "--python", '
            f'{interpreter!r}, "-r", APP + "/requirements.txt"], check=True)'
        )
        if variant == "heredoc-prefix":
            script = (
                f"uv pip install --python {interpreter} -r {dependency} && "
                "python3 - <<'PY'\nprint('ready')\nPY\n"
            )
        elif variant == "continuation":
            script = "python3 \\\n  -c " + shlex.quote(source) + "\n"
        elif variant == "separator":
            script = "true;python3 -c " + shlex.quote(source) + "\n"
        else:
            script = f"{interpreter} /tests/verify.py\n"
        task = original_make_task(
            layout,
            test_sh=f"uv venv {layout.workspace}/.venv\n"
            + script
            + f"{interpreter} -m pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py\n",
            env=env,
        )
        if literal:
            path = dependency if case == "hostile" else "/tests/requirements.txt"
            imports, call, python = (
                ("import subprocess, sys", "subprocess.run", "sys.executable")
                if variant == "sys-executable"
                else ("import subprocess as sp", "sp.run", repr(interpreter))
            )
            (task.paths.tests_dir / "verify.py").write_text(
                f'{imports}\n{call}([{python}, "-m", "pip", "install", "-r", {path!r}], check=True)\n'
            )
        return task

    monkeypatch.setattr(uv_state, "make_task", make_task)

    def edit_workspace(layout):
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
        before=edit_workspace if case == "hostile" or literal else None,
        workspace_name="app",
    )
    assert not proof.exists()
    # Helper-script invocation/aliases have unknown dependency inputs.
    # Honest bytes load however the install was spelled.
    if case == "hostile":
        assert plugin_never_ran(rewards, error), (rewards, error)
        assert markers(layout, "installed") == []
    else:
        assert rewards == {"reward": float(case == "oracle")} and error is None, (
            rewards,
            error,
        )
