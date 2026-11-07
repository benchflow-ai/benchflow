"""Guard V2-R6-CONF-1/2 against the heredoc regressions at fb7307ac."""

import shlex

import pytest

from benchflow.sandbox.lockdown import _dependency_script_bodies
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
    "comment",
    [
        "# Example: python3 - <<'PY'",
        "  # Example: python3 - <<'PY'",
        "true # Example: python3 - <<'PY'",
        "# Example: python3 - <<'PY' \\",
        "# Example: python3 - <<'OTHER'",
        "# Example: python3 - <<'PY'\n# Example: python3 - <<'PY'",
    ],
)
@pytest.mark.parametrize("wrapper", ["direct", "sh-c", "bash-c"])
@pytest.mark.parametrize(
    "operand", ['APP + "/requirements.txt"', '"/tests/requirements.txt"']
)
def test_commented_heredoc_cannot_consume_a_real_body(
    tmp_path, comment, wrapper, operand
):
    """Guards V2-R6-CONF-1 at fb7307ac, including nested shell payloads."""
    script = (
        comment + "\npython3 - <<'PY'\n"
        f'import subprocess\nsubprocess.run(["uv", "pip", "install", "-r", {operand}])\n'
        "PY\nuv pip install -r /app/last.txt\n"
    )
    if wrapper != "direct":
        script = wrapper.split("-")[0] + " -c " + shlex.quote(script)
    expected = (
        "requirements.txt" if operand.startswith("APP") else "/tests/requirements.txt"
    )
    assert named(tmp_path, script) == sorted(["/app/last.txt", expected])


@pytest.mark.parametrize(
    "opening,delimiter,tabs",
    [
        ("python3 - <<\\\n'PY'", "PY", False),
        ('python3 - <<\\\n"PY"', "PY", False),
        ("python3 - <<\\\nPY", "PY", False),
        ("python3 - <\\\n<\\\n'PY'", "PY", False),
        ("python3 - <<-\\\n'PY'", "PY", True),
        ("python3 - <<\\\n-\\\n'PY'", "PY", True),
        ('python3 - <<P\\\n"Y"', "PY", False),
        ("python3 - <<'P'Y", "PY", False),
        ("python3 - <<\\PY", "PY", False),
        ("python3 - <<'END-PY'", "END-PY", False),
        ("python3 - <<'END PY'", "END PY", False),
        ("python3 - <<''", "", False),
        ("/usr/bin/py\\\nthon3 - 0<<\\\n'PY'", "PY", False),
        ("python3 - <<\\\r\n'PY'", "PY", False),
    ],
)
@pytest.mark.parametrize(
    "operand",
    ['APP + "/requirements.txt"', 'APP + "/deps.txt"', '"/tests/requirements.txt"'],
)
def test_heredoc_words_are_parsed_before_consuming_body(
    tmp_path, opening, delimiter, tabs, operand
):
    """Guards V2-R6-CONF-2 at fb7307ac across redirection and word spellings."""
    source = f'import subprocess\nsubprocess.run(["uv", "pip", "install", "-r", {operand}])\n'
    body = "".join(("\t" if tabs else "") + line for line in source.splitlines(True))
    script = opening + "\n" + body + ("\t" if tabs else "") + delimiter + "\n"
    script += "uv pip install -r /app/last.txt\n"
    expected = (
        operand.strip('"')
        if operand.startswith('"')
        else operand.rsplit("/", 1)[-1].rstrip('"')
    )
    assert named(tmp_path, script) == sorted(["/app/last.txt", expected])
    bodies = _dependency_script_bodies(script, ".sh")
    assert (source, True) in bodies


@pytest.mark.parametrize("opening", ["python3 - <<\\\n'PY'", "python3 - <<'P'Y"])
def test_heredoc_body_bytes_remain_python(opening):
    """Guards V2-R6-CONF-2 at fb7307ac: never apply shell escapes to Python."""
    body = '# comment \\\nvalue = """/tests/require\\\nments.txt"""\n'
    assert (body, True) in _dependency_script_bodies(
        opening + "\n" + body + "PY\n", ".sh"
    )


@pytest.mark.parametrize("separator", [";", "&&", "||", "|", "\n"])
def test_headers_and_multiple_bodies_keep_their_own_language(tmp_path, separator):
    """Guards V2-R6-CONF-1/2 at fb7307ac for FIFO bodies and adjacent installers."""
    script = (
        "uv pip install -r /app/first.txt "
        + separator
        + " python3 - <<'DATA' <<\\\n'PY'; uv pip install -r /app/middle.txt\n"
        "# Documentation: python3 - <<'PY'\nDATA\n"
        'import subprocess\nsubprocess.run(["uv", "pip", "install", "-r", APP + "/deps.txt"])\n'
        "PY\nuv pip install -r /app/last.txt\n"
    )
    assert named(tmp_path, script) == [
        "/app/first.txt",
        "/app/last.txt",
        "/app/middle.txt",
        "deps.txt",
    ]


@pytest.mark.parametrize("shell", ["sh", "/bin/bash", "dash", "zsh", "ksh", "ash"])
def test_explicit_shell_heredocs_recurse_without_losing_protected_paths(
    tmp_path, shell
):
    """Guards V2-R6-CONF-2 at fb7307ac: keep shell operands and nested Python distinct."""
    script = (
        f"{shell} -s <<\\\n'SH'\nuv pip install -r /tests/requirements.txt\n"
        "# Example: python3 - <<'PY'\npython3 - <<\\\n'PY'\n"
        'import subprocess\nsubprocess.run(["uv", "pip", "install", "-r", APP + "/deps.txt"])\n'
        "PY\nSH\n"
    )
    assert named(tmp_path, script) == ["/tests/requirements.txt", "deps.txt"]


@pytest.mark.parametrize(
    "prefix", ['echo "python3 - <<PY"', "printf '%s' '<<PY'", "cat <<< '<<PY'"]
)
def test_quoted_redirections_do_not_consume_the_next_body(tmp_path, prefix):
    """Guards V2-R6-CONF-1 at fb7307ac for shell words that only look like openers."""
    script = (
        prefix + "\npython3 - <<'PY'\n"
        'import subprocess\nsubprocess.run(["uv", "pip", "install", "-r", APP + "/deps.txt"])\n'
        "PY\n"
    )
    assert named(tmp_path, script) == ["deps.txt"]


@pytest.mark.parametrize(
    "opening", ["python3 - <<'MISSING'", "python3 - <<", "python3 - <<$'PY'"]
)
def test_unparsed_heredocs_never_grant_precision_to_fragments(tmp_path, opening):
    """Guards V2-R6-CONF-2 at fb7307ac: unsupported boundaries stay conservative."""
    script = (
        opening
        + '\nsubprocess.run(["uv", "pip", "install", "-r", APP + "/requirements.txt"])\nPY\n'
    )
    assert named(tmp_path, script) == ["requirements.txt"]


@pytest.mark.parametrize("variant", ["comment", "delimiter"])
@pytest.mark.parametrize("case", ["oracle", "nop", "hostile"])
@pytest.mark.parametrize("workspace", ["app", "testbed", "root"])
@pytest.mark.parametrize("plane", ["same", "separate"])
async def test_confirmation7_inputs_through_real_guard(
    tmp_path, monkeypatch, variant, case, workspace, plane
):
    """Guards V2-R6-CONF-1/2 at fb7307ac with real hardening and pytest."""
    proof = tmp_path / "hostile-imported"
    original_make_task = uv_state.make_task

    def make_task(layout, env=None):
        interpreter = f"{layout.workspace}/.venv/bin/python"
        opening = (
            "# Example: python3 - <<'PY'\npython3 - <<'PY'"
            if variant == "comment"
            else "python3 - <<\\\n'PY'"
        )
        return original_make_task(
            layout,
            test_sh=f"uv venv {layout.workspace}/.venv\n" + opening + "\n"
            f"import subprocess\nAPP = {str(layout.workspace)!r}\n"
            'subprocess.run(["uv", "pip", "install", "--python", '
            f'{interpreter!r}, "-r", APP + "/requirements.txt"], check=True)\nPY\n'
            f"{interpreter} -m pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py\n",
            env=env,
        )

    monkeypatch.setattr(uv_state, "make_task", make_task)

    def edit_workspace(layout):
        (layout.workspace / "requirements.txt").write_text(
            "ctrf @ file:///dummy-wheel\n"
        )

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
        workspace_name=workspace,
        agent_paths=("{workspace}",) if plane == "separate" else None,
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
