"""Guard V2-R5-CONF-1/2 against dependency scanner regressions at 25c2de2a."""

import pytest

from benchflow.sandbox.lockdown import _dependency_script_bodies, _shell_continuations
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


@pytest.mark.parametrize("ending", ["\n", "\r\n"])
@pytest.mark.parametrize("prefix", ["# install", "  # it's ready", "true # install"])
@pytest.mark.parametrize("flag", ["-r", "-c", "--requirement"])
def test_comment_backslash_keeps_the_next_installer(tmp_path, ending, prefix, flag):
    """Guards V2-R5-CONF-1 at 25c2de2a for full-line and trailing comments."""
    script = prefix + "\\" + ending + f"uv pip install {flag} /app/deps.txt\n"
    assert named(tmp_path, script) == ["/app/deps.txt"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("# comment\\\nnext\n", "# comment\\\nnext\n"),
        ("# quote '\\\nnext\n", "# quote '\\\nnext\n"),
        ("echo x \\\nnext\n", "echo x next\n"),
        ('echo "x\\\nnext"\n', 'echo "xnext"\n'),
        ("echo 'x\\\nnext'\n", "echo 'x\\\nnext'\n"),
        ("echo \\#x \\\nnext\n", "echo \\#x next\n"),
    ],
)
def test_shell_continuations_respect_comment_and_quote_boundaries(text, expected):
    """Guards V2-R5-CONF-1 at 25c2de2a without changing quoted continuations."""
    assert _shell_continuations(text) == expected


@pytest.mark.parametrize(
    "opening",
    [
        "python3 \\\n  - <<'PY'",
        'python3 -I \\\n  - <<"PY"',
        "python3 \\\n  - \\\n  <<PY",
        "python3 \\\n  - <<-'PY'",
        "python3 \\\r\n  - <<'PY'",
        "py\\\nthon3 - <<'PY'",
        "/usr/bin/py\\\nthon3 \\\n  - <<'PY'",
        "cd /app; python3 \\\n  - <<'PY'",
        "uv pip install -r /app/first.txt && python3 \\\n  - <<'PY'",
    ],
)
@pytest.mark.parametrize(
    "operand",
    ['APP + "/requirements.txt"', 'APP + "/deps.txt"', '"/tests/requirements.txt"'],
)
def test_continued_heredoc_retains_python_path_semantics(tmp_path, opening, operand):
    """Guards V2-R5-CONF-2 at 25c2de2a, including flags, tabs and separators."""
    source = f'import subprocess\nsubprocess.run(["uv", "pip", "install", "-r", {operand}])\n'
    tabs = "\t" if "<<-" in opening else ""
    script = opening + "\n" + "".join(tabs + line for line in source.splitlines(True))
    script += tabs + "PY\nuv pip install -r /app/last.txt\n"
    name = (
        "/tests/requirements.txt"
        if operand.startswith('"/tests')
        else ("requirements.txt" if "requirements" in operand else "deps.txt")
    )
    expected = ["/app/last.txt", name]
    if "first.txt" in opening:
        expected.insert(0, "/app/first.txt")
    assert named(tmp_path, script) == expected


@pytest.mark.parametrize("opening", ["python3 - <<'PY'", "python3 \\\n - <<'PY'"])
def test_heredoc_body_is_not_normalized_as_shell(opening):
    """Guards V2-R5-CONF-2 at 25c2de2a: preserve Python escapes and comments."""
    source = '# Python comment \\\nprint("ready")\npath = """/tests/require\\\nments.txt"""\n'
    bodies = _dependency_script_bodies(opening + "\n" + source + "PY\n", ".sh")
    assert (source, True) in bodies


def test_commented_shell_heredoc_does_not_hide_python(tmp_path):
    """Guards V2-R5-CONF-2 at 25c2de2a when comments show shell syntax."""
    script = (
        "# Example redirection: <<END\n"
        "python3 \\\n - <<'PY'\n"
        'subprocess.run(["uv", "pip", "install", "-r", APP + "/requirements.txt"])\n'
        "PY\nEND\n"
    )
    assert named(tmp_path, script) == ["requirements.txt"]


@pytest.mark.parametrize("variant", ["comment", "heredoc"])
@pytest.mark.parametrize("case", ["oracle", "nop", "hostile"])
async def test_confirmation3_inputs_through_real_guard(
    tmp_path, monkeypatch, variant, case
):
    """Guards V2-R5-CONF-1/2 at 25c2de2a with real pytest and dummy markers."""
    proof = tmp_path / "hostile-imported"
    filename = "deps.txt" if variant == "comment" else "requirements.txt"
    original_make_task = uv_state.make_task

    def make_task(layout, env=None):
        interpreter = f"{layout.workspace}/.venv/bin/python"
        if variant == "comment":
            script = (
                "# Install verifier dependencies \\\n"
                f"uv pip install --python {interpreter} -r {layout.workspace}/{filename}\n"
            )
        else:
            script = (
                "python3 \\\n  - <<'PY'\n"
                f"import subprocess\nAPP = {str(layout.workspace)!r}\n"
                'subprocess.run(["uv", "pip", "install", "--python", '
                f'{interpreter!r}, "-r", APP + "/requirements.txt"], check=True)\nPY\n'
            )
        return original_make_task(
            layout,
            test_sh=f"uv venv {layout.workspace}/.venv\n"
            + script
            + f"{interpreter} -m pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py\n",
            env=env,
        )

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
        before=edit_workspace if case == "hostile" else None,
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
