"""Guard V2-R7-CONF-1/2/3 against heredoc misclassification at aed7ce5a."""

import subprocess

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
    "command",
    [
        "sh -c 'python3 -'",
        "bash -ec 'python3 -'",
        "bash -sc 'python3 -'",
        "bash -o errexit -c 'python3 -'",
        "sh -c 'python3 /dev/fd/3' 3",
        "sh -c 'python3 - <&3' 3",
        "bash /dev/stdin",
        "sh /dev/fd/3 3",
    ],
)
@pytest.mark.parametrize("flag", ["-r", "-c", "--requirement", "--constraint"])
@pytest.mark.parametrize("filename", ["requirements.txt", "deps.txt", "limits.txt"])
def test_forwarded_heredoc_is_not_the_shell_script(tmp_path, command, flag, filename):
    """Guards V2-R7-CONF-1 at aed7ce5a for command strings and descriptor inputs."""
    # Script-path operands consume shell scripts; use those as literal controls.
    shell = command.startswith(("bash /dev", "sh /dev"))
    body = (
        f"uv pip install {flag} /tests/{filename}\n"
        if shell
        else f'subprocess.run(["uv", "pip", "install", "{flag}", APP + "/{filename}"])\n'
    )
    assert named(tmp_path, command + "<<'END'\n" + body + "END\n") == [
        f"/tests/{filename}" if shell else filename
    ]


@pytest.mark.parametrize(
    "word,delimiter",
    [
        ("$'PY'", "PY"),
        (r"$'P\x59'", "PY"),
        (r"$'\120Y'", "PY"),
        (r"$'P\u0059'", "PY"),
        ("P$'Y'", "PY"),
        ("$'P'\"Y\"", "PY"),
        (r'"P\q"', r"P\q"),
        ("'P'Y", "PY"),
        (r"$'P\'Y'", "P'Y"),
        (r"$'P\\Y'", r"P\Y"),
        (r"$'P\tY'", "P\tY"),
    ],
)
@pytest.mark.parametrize("flag", ["-r", "-c", "--requirement", "--constraint"])
@pytest.mark.parametrize("operand", ['APP + "/deps.txt"', '"/tests/limits.txt"'])
def test_delimiter_quote_removal_preserves_python_operands(
    tmp_path, word, delimiter, flag, operand
):
    """Guards V2-R7-CONF-2 at aed7ce5a for custom names and literal precision."""
    consumer = subprocess.run(
        ["bash", "-c", f"cat <<{word}\nDUMMY\n{delimiter}\n"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert consumer.stdout == "DUMMY\n" and consumer.stderr == ""
    script = (
        f"python3 - <<{word}\n"
        f'subprocess.run(["uv", "pip", "install", "{flag}", {operand}])\n'
        f"{delimiter}\nuv pip install -r /app/last.txt\n"
    )
    assert named(tmp_path, script) == sorted(
        [
            "/app/last.txt",
            "deps.txt" if operand.startswith("APP") else "/tests/limits.txt",
        ]
    )


@pytest.mark.parametrize(
    "opening", ["python3 - <<MISSING", "python3 - <<", "python3 - <<$'\\x'"]
)
@pytest.mark.parametrize("flag", ["-r", "-c", "--requirement", "--constraint"])
@pytest.mark.parametrize("operand", ['APP + "/deps.txt"', '"/tests/limits.txt"'])
def test_unknown_boundary_keeps_quoted_custom_operands(
    tmp_path, opening, flag, operand
):
    """Guards V2-R7-CONF-2 at aed7ce5a when no AST or delimiter is available."""
    assert named(
        tmp_path,
        opening
        + f'\nsubprocess.run(["uv", "pip", "install", "{flag}", {operand}])\nPY\n',
    ) == ["deps.txt" if operand.startswith("APP") else "limits.txt"]


@pytest.mark.parametrize(
    "argv",
    [
        '["uv", "pip", "install", "-r", "/app" + "/deps.txt"]',
        '["uv", "pip", "install", "--requirement=/app/deps.txt"]',
        '("uv", "pip", "install", "-c", f"{APP}/deps.txt")',
        '["python3", "-m", "pip", "install", "-c",\n APP + "/deps.txt"]',
    ],
)
def test_boundary_fallback_reuses_python_argv_rules(tmp_path, argv):
    """Guards V2-R7-CONF-2 at aed7ce5a: recover expressions without path precision."""
    assert named(
        tmp_path, "python3 - <<MISSING\nsubprocess.run(" + argv + ")\nPY\n"
    ) == ["deps.txt"]


@pytest.mark.parametrize(
    "header,shell",
    [
        ("bash -s 3<<'SH'", False),
        ("bash -s 3<<'SH' <&3", True),
        ("bash -s <&3 3<<'SH'", False),
        ("bash -s <<'SH' </dev/null", False),
        ("bash -s </dev/null <<'SH'", True),
        ("<<'SH' env bash -s", True),
        ("3<<'SH' command bash /proc/self/fd/3", True),
        ("bash script.sh <<'SH'", False),
        ("bash -sc 'python3 -' <<'SH'", False),
    ],
)
def test_only_the_script_descriptor_grants_shell_semantics(header, shell):
    """Guards V2-R7-CONF-1/3 at aed7ce5a: ordered redirections select the input."""
    body = "uv pip install -r /tests/requirements.txt\n"
    assert (body, not shell) in _dependency_script_bodies(
        header + "\n" + body + "SH\n", ".sh"
    )


@pytest.mark.parametrize("options", ["-c", "-ec", "-sc", "-s -c", "-o errexit -c"])
def test_command_payload_and_forwarded_body_are_separate(tmp_path, options):
    """Guards V2-R7-CONF-1 at aed7ce5a: each source retains its own language."""
    script = (
        f"bash {options} 'uv pip install -r /tests/first.txt; python3 -' <<'PY'\n"
        'subprocess.run(["uv", "pip", "install", "-r", APP + "/deps.txt"])\nPY\n'
    )
    assert named(tmp_path, script) == ["/tests/first.txt", "deps.txt"]


@pytest.mark.parametrize(
    "command",
    [
        "env bash -s",
        "env -i FLAG=value /bin/bash -s",
        "command bash -s",
        "command -p bash -s",
        "FLAG=value bash -s",
        "exec bash -s",
        "2>/dev/null bash -s",
        "bash -s 2>/dev/null",
        "env command bash -s",
        "env -u FLAG bash -es",
        "bash --noprofile --norc -s",
        "bash -e",
        "bash -s -- argument",
        "bash -o errexit -s",
    ],
)
@pytest.mark.parametrize(
    "path", ["/tests/requirements.txt", "/verifier/deps.txt", "/app/limits.txt"]
)
def test_wrapped_shell_consumes_stdin_precisely(tmp_path, command, path):
    """Guards V2-R7-CONF-3 at aed7ce5a for wrappers, options and redirections."""
    assert named(tmp_path, command + f" <<'SH'\nuv pip install -r {path}\nSH\n") == [
        path
    ]


@pytest.mark.parametrize("variant", ["forwarded", "delimiter", "wrapper"])
@pytest.mark.parametrize("case", ["oracle", "nop", "hostile"])
@pytest.mark.parametrize("workspace", ["app", "testbed", "root"])
@pytest.mark.parametrize("plane", ["same", "separate"])
async def test_confirmation8_inputs_through_real_guard(
    tmp_path, monkeypatch, variant, case, workspace, plane
):
    """Guards all three V2-R7 findings at aed7ce5a through real hardening/pytest."""
    proof = tmp_path / "hostile-imported"
    original_make_task = uv_state.make_task
    filename = "requirements.txt" if variant != "delimiter" else "deps.txt"

    def make_task(layout, env=None):
        interpreter = f"{layout.workspace}/.venv/bin/python"
        if variant == "wrapper":
            dependency = (
                f"{layout.workspace}/{filename}"
                if case == "hostile"
                else f"/tests/{filename}"
            )
            body = f"env bash -s <<'SH'\nuv pip install --python {interpreter} -r {dependency}\nSH\n"
        else:
            opening = (
                "sh -c 'python3 -' <<'PY'"
                if variant == "forwarded"
                else "python3 - <<$'PY'"
            )
            body = (
                opening + f"\nimport subprocess\nAPP = {str(layout.workspace)!r}\n"
                'subprocess.run(["uv", "pip", "install", "--python", '
                f'{interpreter!r}, "-r", APP + "/{filename}"], check=True)\nPY\n'
            )
        return original_make_task(
            layout,
            test_sh=f"uv venv {layout.workspace}/.venv\n"
            + body
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
        before=edit_workspace if case == "hostile" or variant == "wrapper" else None,
        workspace_name=workspace,
        agent_paths=("{workspace}",) if plane == "separate" else None,
    )
    assert not proof.exists()
    # Heredoc wrappers are conservative, including unrelated agent files.
    # Honest bytes load however the install was spelled.
    if case == "hostile":
        assert plugin_never_ran(rewards, error), (rewards, error)
        assert markers(layout, "installed") == []
    else:
        assert rewards == {"reward": float(case == "oracle")} and error is None, (
            rewards,
            error,
        )
