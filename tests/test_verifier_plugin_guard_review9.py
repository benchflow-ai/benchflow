"""Guard V2-R8-CONF-1/2 against the remaining heredoc aliases at 5fd31770."""

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
    "word,delimiter",
    [
        (r'$"PY"', "PY"),
        (r'P$"Y"', "PY"),
        (r'$"P"\Y', "PY"),
        ("$\"P\"$'Y'", "PY"),
        (r'$"P\\Y"', r"P\Y"),
    ],
)
@pytest.mark.parametrize(
    "executable", ["sys.executable", "current", "runtime.executable"]
)
@pytest.mark.parametrize("flag", ["-r", "-c", "--requirement", "--constraint"])
@pytest.mark.parametrize("operand", ['APP + "/deps.txt"', '"/tests/limits.txt"'])
def test_locale_delimiters_keep_bound_python_installers(
    tmp_path, word, delimiter, executable, flag, operand
):
    """Guards V2-R8-CONF-1 at 5fd31770 with Bash's real delimiter consumer."""
    result = subprocess.run(
        ["bash", "-c", f"cat <<{word}\nDUMMY\n{delimiter}\n"],
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert (result.stdout, result.stderr) == ("DUMMY\n", "")
    source = (
        f"python3 - <<{word}\nimport subprocess, sys\n"
        "import sys as runtime\nfrom sys import executable as current\n"
        f'subprocess.run([{executable}, "-m", "pip", "install", "{flag}", {operand}])\n'
        f"{delimiter}\nuv pip install -r /tests/last.txt\n"
    )
    assert named(tmp_path, source) == sorted(
        [
            "/tests/last.txt",
            "deps.txt" if operand.startswith("APP") else "/tests/limits.txt",
        ]
    )


@pytest.mark.parametrize(
    "executable",
    ["sys.executable", "current", "runtime.executable", "PYTHON", "choose_python()"],
)
@pytest.mark.parametrize("module", ["pip", "pip.__main__"])
@pytest.mark.parametrize(
    "argument",
    [
        '"-r", APP + "/deps.txt"',
        '"-c", "/tests/limits.txt"',
        '"--requirement=/tests/deps.txt"',
    ],
)
@pytest.mark.parametrize(
    "opening", ["python3 - <<MISSING", "python3 - <<", "python3 - <<$'\\x'"]
)
def test_unknown_boundaries_keep_possible_module_installers(
    tmp_path, executable, module, argument, opening
):
    """Guards V2-R8-CONF-1 at 5fd31770 without trusting missing import bindings."""
    source = f'{opening}\nsubprocess.run([{executable}, "-I", "-m", "{module}", "install", {argument}])\nPY\n'
    assert named(tmp_path, source) == [
        "limits.txt" if "limits" in argument else "deps.txt"
    ]


@pytest.mark.parametrize("operand", ["3", "'3'", '"3"', r"\3", "$'3'", '$"3"'])
@pytest.mark.parametrize("operator", ["<&", ">&"])
@pytest.mark.parametrize(
    "path", ["/tests/requirements.txt", "/verifier/deps.txt", "/app/limits.txt"]
)
def test_quoted_descriptor_duplication_keeps_protected_paths(
    tmp_path, operand, operator, path
):
    """Guards V2-R8-CONF-2 at 5fd31770 with quote removal before fd lookup."""
    source = f"bash -s 3<<'SH' {operator}{operand}\nuv pip install -r {path}\nSH\n"
    # An output duplication must explicitly target stdin.
    if operator == ">&":
        source = source.replace(f" {operator}", f" 0{operator}")
    assert named(tmp_path, source) == [path]


@pytest.mark.parametrize(
    "alias",
    ["/dev/fd/3", "/proc/self/fd/3", "/dev/stdout", "/dev/stderr", "/dev/stdin"],
)
@pytest.mark.parametrize("quote", ["{}", "'{}'", '"{}"', "$'{}'", '$"{}"'])
@pytest.mark.parametrize("consumer", ["bash -s <", "bash "])
def test_descriptor_paths_select_the_shell_script(tmp_path, alias, quote, consumer):
    """Guards V2-R8-CONF-2 at 5fd31770 for fd paths and their standard aliases."""
    fd = {"/dev/stdin": 0, "/dev/stdout": 1, "/dev/stderr": 2}.get(alias, 3)
    actual = subprocess.run(
        [
            "bash",
            "-c",
            f"9>&1 {fd}<<'SH' {consumer}{quote.format(alias)}\nprintf DUMMY >&9\nSH\n",
        ],
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert actual.stdout == "DUMMY"
    source = f"{fd}<<'SH' {consumer}{quote.format(alias)}\nuv pip install -r /tests/requirements.txt\nSH\n"
    assert named(tmp_path, source) == ["/tests/requirements.txt"]


@pytest.mark.parametrize(
    "executable", ["PYTHON", "runtime.executable", "choose_python()"]
)
@pytest.mark.parametrize("module", ["pip", "pip.__main__", "unrelated"])
@pytest.mark.parametrize(
    "imports", ["import subprocess", "from sys import *\nimport subprocess"]
)
def test_unbound_executables_never_grant_absolute_precision(
    tmp_path, executable, module, imports
):
    """Guards V2-R8-CONF-1 at 5fd31770 for complete ASTs with unknown bindings."""
    source = f'{imports}\nsubprocess.run([{executable}, "-m", "{module}", "install", "-r", "/tests/deps.txt"])\n'
    assert named(tmp_path, "#!/usr/bin/python3\n" + source) == (
        [] if module == "unrelated" else ["deps.txt"]
    )


@pytest.mark.parametrize(
    "header,shell",
    [
        ("bash -s 3<<'SH' 4<&'3' < /dev/fd/4", True),
        ("bash -s 3<<'SH' < /dev/fd/3 3<&-", True),
        ("bash -s 3<<'SH' 3<&- < /dev/fd/3", False),
        ("bash -s < /dev/fd/3 3<<'SH'", False),
        ("bash -s 3<<'SH' < /dev/fd/3 < /dev/null", False),
        ("bash -s 3<<'SH' <&'3-'", True),
        ("bash -s 3<<'SH' 4<&'3-' <&3", False),
        ("bash -s 3<<'SH' 4<&'3-' <&4", True),
        ("bash -c 'python3 -' 3<<'SH' < /dev/fd/3", False),
        ("bash -s 3<<'SH' < /dev/fd/$FD", False),
        ("bash -s 3<<'SH' <&\"$FD\"", False),
        ("bash -s 3<<'SH' <&$(printf 3)", False),
        ("bash -s 3<<'SH' <&`printf 3`", False),
    ],
)
def test_descriptor_aliases_obey_binding_order(header, shell):
    """Guards V2-R8-CONF-2 at 5fd31770; unknown/closed/forwarded fds stay opaque."""
    body = "uv pip install -r /tests/requirements.txt\n"
    assert (body, not shell) in _dependency_script_bodies(
        header + "\n" + body + "SH\n", ".sh"
    )


@pytest.mark.parametrize("variant", ["locale", "fd-path", "fd-quoted"])
@pytest.mark.parametrize("case", ["oracle", "nop", "hostile"])
@pytest.mark.parametrize("workspace", ["app", "testbed", "root"])
@pytest.mark.parametrize("plane", ["same", "separate"])
async def test_confirmation9_inputs_through_real_guard(
    tmp_path, monkeypatch, variant, case, workspace, plane
):
    """Guards V2-R8-CONF-1/2 at 5fd31770 through hardening, guard and real pytest."""
    proof = tmp_path / "hostile-imported"
    original_make_task = uv_state.make_task
    filename = "deps.txt" if variant == "locale" else "requirements.txt"

    def make_task(layout, env=None):
        interpreter = f"{layout.workspace}/.venv/bin/python"
        if variant == "locale":
            body = (
                f'{interpreter} - <<$"PY"\nimport subprocess, sys\nAPP = {str(layout.workspace)!r}\n'
                'subprocess.run([sys.executable, "-m", "pip", "install", "-r", '
                f'APP + "/{filename}"], check=True)\nPY\n'
            )
        else:
            dependency = (
                f"{layout.workspace}/{filename}"
                if case == "hostile"
                else f"/tests/{filename}"
            )
            redirect = "< /dev/fd/3" if variant == "fd-path" else "<&'3'"
            body = f"bash -s 3<<'SH' {redirect}\nuv pip install --python {interpreter} -r {dependency}\nSH\n"
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
        before=edit_workspace if case == "hostile" or variant != "locale" else None,
        workspace_name=workspace,
        agent_paths=("{workspace}",) if plane == "separate" else None,
    )
    assert not proof.exists()
    # Honest bytes load however the install was spelled; hostile ones never
    # do, whatever chose them.
    if case == "hostile":
        assert plugin_never_ran(rewards, error), (rewards, error)
        assert markers(layout, "installed") == []
    else:
        assert rewards == {"reward": float(case == "oracle")} and error is None, (
            rewards,
            error,
        )
