"""Guard V2-R9-CONF-1/2/3 and option grammar drift at 7626cb46."""

import shlex
import subprocess

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

TOOLS = [
    ["pip", "install"],
    ["pip3.12", "download"],
    ["python3", "-mpip", "install"],
    ["python3", "-Impip.__main__", "install"],
    ["python3", "-bbI", "-Wignore", "-X", "dev", "-m", "pip", "wheel"],
    ["uv", "pip", "install"],
    ["uv", "--directory", "/app", "pip", "install"],
    ["uv", "add"],
]


@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("flag", ["-r", "-c"])
@pytest.mark.parametrize("form", ["separated", "attached", "long", "equals"])
@pytest.mark.parametrize("language", ["shell", "python", "heredoc", "unknown"])
def test_dependency_option_grammar(tmp_path, tool, flag, form, language):
    """Guards CONF-1/2 at 7626cb46 across every caller of argv discovery."""
    path = "/tests/custom deps.txt"
    long = (
        ("--requirements" if tool[0] == "uv" else "--requirement")
        if flag == "-r"
        else ("--constraints" if tool[0] == "uv" else "--constraint")
    )
    option = {
        "separated": [flag, path],
        "attached": [flag + path],
        "long": [long, path],
        "equals": [long + "=" + path],
    }[form]
    argv = tool + option
    source = "import subprocess\nsubprocess.run(" + repr(argv) + ")\n"
    if language == "shell":
        source = shlex.join(argv)
    elif language == "python":
        source = "#!/usr/bin/python3\n" + source
    else:
        source = (
            ("python3 - <<'PY'\n" if language == "heredoc" else "python3 - <<MISSING\n")
            + source
            + "PY\n"
        )
    assert named(tmp_path, source) == [
        "custom deps.txt" if language == "unknown" else path
    ]


@pytest.mark.parametrize(
    "module", ["-mpip", "-mpip.__main__", "-Impip", "-bbImpip.__main__"]
)
@pytest.mark.parametrize("flag", ["-r", "-c", "--requirement=", "--constraint="])
@pytest.mark.parametrize("executable", ["sys.executable", "PYTHON", "choose_python()"])
def test_expression_operands_keep_basename(tmp_path, module, flag, executable):
    """Guards CONF-1/2 at 7626cb46 when either executable or operand is dynamic."""
    argument = (
        repr(flag) + ', APP + "/custom.txt"'
        if flag in ("-r", "-c")
        else repr(flag) + ' + APP + "/custom.txt"'
    )
    source = f'#!/usr/bin/python3\nimport subprocess, sys\nsubprocess.run([{executable}, {module!r}, "install", {argument}])\n'
    assert named(tmp_path, source) == ["custom.txt"]


@pytest.mark.parametrize(
    "prefix", ["-I", "-bbI", "-Wignore", "-Xdev", "--check-hash-based-pycs always"]
)
def test_python_command_option_uses_interpreter_grammar(tmp_path, prefix):
    """Guards CONF-1 at 7626cb46: -W/-X values never become module/command flags."""
    code = 'import subprocess; subprocess.run(["pip", "install", "-r/custom.txt"])'
    assert named(tmp_path, f"python3 {prefix} -c {shlex.quote(code)}") == [
        "/custom.txt"
    ]


@pytest.mark.parametrize("tool", [["pip", "install"], ["uv", "pip", "install"]])
@pytest.mark.parametrize(
    "selector",
    [
        ["--python", "/usr/bin/python3"],
        ["--python=/usr/bin/python3"],
        ["--cache-dir", "-r/decoy.txt"],
    ],
)
def test_non_dependency_option_values_are_consumed(tmp_path, tool, selector):
    """Guards CONF-2 at 7626cb46: a value cannot become a second option."""
    assert named(tmp_path, shlex.join(tool + selector + ["-r/actual.txt"])) == [
        "/actual.txt"
    ]


@pytest.mark.parametrize(
    "command",
    [
        "echo pip install -r/decoy.txt",
        "python3 -c 'print(1)' -r/decoy.txt",
        "pip install -- -r/decoy.txt",
        "uv pip install -- -r/decoy.txt",
        "python3 -W -mpip -c 'print(1)' -r/decoy.txt",
    ],
)
def test_option_roles_and_termination(tmp_path, command):
    """Guards CONF-1/2 at 7626cb46: tool names in data cannot grant option roles."""
    assert named(tmp_path, command) == []


@pytest.mark.parametrize(
    "command,expected",
    [
        ("pip install -qqr/tests/deps.txt", "/tests/deps.txt"),
        ("pip install --requirem=/tests/deps.txt", "/tests/deps.txt"),
        ("pip install --const /tests/deps.txt", "/tests/deps.txt"),
        ("pip install --require=/tests/decoy.txt", None),
        ("pip install -r=/tests/deps.txt", "deps.txt"),
        ("uv pip install -r=/tests/deps.txt", "/tests/deps.txt"),
        ("uv pip install -qc/tests/deps.txt", "/tests/deps.txt"),
        ("uv pip install -b/tests/deps.txt", "/tests/deps.txt"),
        ("uv pip install --build-constraint=/tests/deps.txt", "/tests/deps.txt"),
        ("uv pip install --override /tests/deps.txt", "/tests/deps.txt"),
        ("uv pip install --excludes=/tests/deps.txt", "/tests/deps.txt"),
        (
            "uv add --editable --no-binary --workspace -r/tests/deps.txt",
            "/tests/deps.txt",
        ),
        (
            "uv run --no-binary --script --with-requirements=/tests/deps.txt /tests/verify.py",
            "/tests/deps.txt",
        ),
        ("uv run pip install -r/tests/deps.txt", "/tests/deps.txt"),
        (
            "uv run --with-requirements=/tests/deps.txt pytest -r/decoy.txt",
            "/tests/deps.txt",
        ),
        ("uv pip sync /tests/deps.txt", "/tests/deps.txt"),
        ("uv pip compile /tests/deps.txt", "/tests/deps.txt"),
        ("uv --offline pip --offline install -r/tests/deps.txt", "/tests/deps.txt"),
        ("env -u UNUSED command -p pip install -r/tests/deps.txt", "/tests/deps.txt"),
    ],
)
@pytest.mark.parametrize("language", ["shell", "python", "unknown"])
def test_subcommand_option_roles(tmp_path, command, expected, language):
    """Guards CONF-1/2 at 7626cb46 across tool-specific arities and spellings."""
    source = command
    if language != "shell":
        source = (
            "import subprocess\nsubprocess.run(" + repr(shlex.split(command)) + ")\n"
        )
        source = (
            "#!/usr/bin/python3\n" if language == "python" else "python3 - <<MISSING\n"
        ) + source
    result = (
        []
        if expected is None
        else [expected.rsplit("/", 1)[-1] if language == "unknown" else expected]
    )
    assert named(tmp_path, source) == result


@pytest.mark.parametrize(
    "command",
    [
        "if pip install -r/tests/deps.txt; then true; fi",
        "while pip install -r/tests/deps.txt; do break; done",
        "true; then pip install -r/tests/deps.txt",
        "2>/dev/null pip install -r/tests/deps.txt",
        "pip install 2>/dev/null -r/tests/deps.txt",
        "</dev/null pip install -r/tests/deps.txt",
        "echo DUMMY # ordinary comment\npip install -r/tests/deps.txt",
        "pip \\\ninstall -r/tests/deps.txt",
    ],
)
def test_shell_execution_positions_keep_option_grammar(tmp_path, command):
    """Guards CONF-2 at 7626cb46 when shell structure surrounds installer argv."""
    assert named(tmp_path, command) == ["/tests/deps.txt"]


@pytest.mark.parametrize(
    "tool", ["pip install", "python3 $'-mpip' install", "uv pip install"]
)
@pytest.mark.parametrize(
    "option",
    [
        "$'-r/tests/deps.txt'",
        '-r$"/tests/deps.txt"',
        '$"-r" $"/tests/deps.txt"',
        "$'-r\\x2ftests/deps.txt'",
        "$'-c' $'/tests/deps.txt'",
    ],
)
def test_shell_quote_removal_precedes_option_grammar(tmp_path, tool, option):
    """Guards CONF-1/2 at 7626cb46: quoted argv has the same option semantics."""
    assert named(tmp_path, f"{tool} {option}") == ["/tests/deps.txt"]


def test_quoted_python_command_flag_keeps_python_semantics(tmp_path):
    """Guards CONF-1 at 7626cb46 using the same word removal before -c parsing."""
    code = 'import subprocess; subprocess.run(["pip", "install", "-r/tests/deps.txt"])'
    assert named(tmp_path, "python3 $'-Ic' " + shlex.quote(code)) == ["/tests/deps.txt"]


ALIASES = [
    "/dev/./fd/3",
    "/dev//fd///3",
    "//dev/fd/3",
    "/proc/self/./fd/3",
    "/proc//thread-self/fd/3",
    "/proc/thread-self/fd/3",
    "/dev/./stdin",
]


@pytest.mark.parametrize("alias", ALIASES)
@pytest.mark.parametrize("consumer", ["bash -s <", "bash "])
@pytest.mark.parametrize("quote", ["{}", "'{}'", '"{}"', "$'{}'", '$"{}"'])
def test_descriptor_path_equivalence(tmp_path, alias, consumer, quote):
    """Guards CONF-3 at 7626cb46 against Linux's actual descriptor resolution."""
    fd = 0 if alias.endswith("stdin") else 3
    header = f"9>&1 {fd}<<'SH' {consumer}{quote.format(alias)}"
    actual = subprocess.run(
        ["bash", "-c", header + "\nprintf DUMMY >&9\nSH\n"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert actual.stdout == "DUMMY"
    assert named(tmp_path, header + "\nuv pip install -r/tests/custom.txt\nSH\n") == [
        "/tests/custom.txt"
    ]


@pytest.mark.parametrize(
    "redirect,precise",
    [
        ("3<<'SH' 4<&3 < /proc/thread-self/./fd/4", True),
        ("3<<'SH' < /dev/./fd/3 3<&-", True),
        ("3<<'SH' 3<&- < /dev/./fd/3", False),
        ("< /dev/./fd/3 3<<'SH'", False),
        ("3<<'SH' < /dev/./fd/3 < /dev/null", False),
        ("3<<'SH' < /dev/fd/../fd/3", False),
        ("3<<'SH' < /app/fd/3", False),
    ],
)
def test_normalization_keeps_order_and_fixed_roots(tmp_path, redirect, precise):
    """Guards CONF-3 at 7626cb46; normalization cannot add descriptor authority."""
    assert named(
        tmp_path, f"bash -s {redirect}\nuv pip install -r /tests/custom.txt\nSH\n"
    ) == ["/tests/custom.txt" if precise else "custom.txt"]


@pytest.mark.parametrize("variant", ["module", "attached", "descriptor", "quoted"])
@pytest.mark.parametrize("case", ["oracle", "nop", "hostile"])
@pytest.mark.parametrize("workspace", ["app", "testbed", "root"])
@pytest.mark.parametrize("plane", ["same", "separate"])
async def test_confirmation10_inputs_through_real_guard(
    tmp_path, monkeypatch, variant, case, workspace, plane
):
    """Guards CONF-1/2/3 at 7626cb46 through hardening, real pytest and scoring."""
    proof = tmp_path / "hostile-imported"
    original = uv_state.make_task

    def make_task(layout, env=None):
        interpreter = f"{layout.workspace}/.venv/bin/python"
        if variant in {"descriptor", "quoted"}:
            dependency = (
                f"{layout.workspace}/deps.txt"
                if case == "hostile"
                else "/tests/deps.txt"
            )
            body = (
                f"bash -s 3<<'SH' < /dev/./fd/3\nuv pip install --python {interpreter} -r {dependency}\nSH\n"
                if variant == "descriptor"
                else f"uv pip install --python {interpreter} $'-r{dependency}'\n"
            )
        else:
            argument = (
                '"-r", APP + "/deps.txt"'
                if variant == "module"
                else '"-r" + APP + "/deps.txt"'
            )
            module = '"-mpip"' if variant == "module" else '"-m", "pip"'
            body = f'{interpreter} - <<$"PY"\nimport subprocess, sys\nAPP = {str(layout.workspace)!r}\nsubprocess.run([sys.executable, {module}, "install", {argument}])\nPY\n'
        return original(
            layout,
            test_sh=f"uv venv {layout.workspace}/.venv\n"
            + body
            + f"{interpreter} -m pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py\n",
            env=env,
        )

    monkeypatch.setattr(uv_state, "make_task", make_task)

    def edit(layout):
        (layout.workspace / "deps.txt").write_text("ctrf @ file:///dummy-wheel\n")

    def install(env, layout):
        site = layout.workspace / ".venv/lib/python3.12/site-packages"
        install_plugin(site, hostile_ctrf(proof) if case == "hostile" else CTRF_PLUGIN)
        (layout.workspace / ".venv/pyvenv.cfg").write_text("home = /usr/bin\n")
        return [site]

    layout, _, rewards, error = await score(
        tmp_path,
        monkeypatch,
        PASSING if case == "oracle" else FAILING,
        install,
        before=edit
        if case == "hostile" or variant in {"descriptor", "quoted"}
        else None,
        workspace_name=workspace,
        agent_paths=("{workspace}",) if plane == "separate" else None,
    )
    assert not proof.exists()
    # Descriptor and quoted command transports taint all workspace files.
    # Honest bytes load however the install was spelled.
    if case == "hostile":
        assert plugin_never_ran(rewards, error), (rewards, error)
        assert markers(layout, "installed") == []
    else:
        assert rewards == {"reward": float(case == "oracle")} and error is None, (
            rewards,
            error,
        )
