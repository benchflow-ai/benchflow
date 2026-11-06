"""Guard V2-R10-CONF-1/2 against the regressions at d9c1b733."""

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

TOOLS = [
    "pip install",
    "python3 -mpip install",
    "python3 -bbImpip.__main__ install",
    "uv pip install",
    "uv add",
    "uv run --with-requirements=/tests/other.txt pip install",
]

GROUPS = [
    "{{ {command}; }}",
    "{{\n{command}\n}}",
    "{{ {{ {command}; }}; }}",
    "if true; then {{ {command}; }}; fi",
    "! {{ {command}; }}",
    "{{ 2>/dev/null {command}; }}",
    "{{ echo DUMMY; {command}; }}",
    "( {{ {command}; }} )",
]


@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize(
    "option",
    ["-r /tests/deps.txt", "-c/tests/deps.txt", "--requirement=/tests/deps.txt"],
)
def test_brace_groups_preserve_installer_argv(tmp_path, tool, group, option):
    """Guards CONF-1 at d9c1b733: reserved braces cannot become executables."""
    expected = ["/tests/deps.txt"]
    if tool.startswith("uv run"):
        expected.append("/tests/other.txt")
    assert named(tmp_path, group.format(command=f"{tool} {option}")) == expected


@pytest.mark.parametrize("brace", ["'{'", '"{"', r"\{", "$'{'", '$"{"'])
def test_quoted_brace_is_an_executable_word(tmp_path, brace):
    """Guards CONF-1 at d9c1b733: quote removal cannot invent a reserved word."""
    assert named(tmp_path, f"{brace} pip install -r/tests/deps.txt") == []


@pytest.mark.parametrize("consumer", ["bash -s", "sh -s", "bash /dev/stdin"])
def test_brace_group_preserves_shell_heredoc_language(tmp_path, consumer):
    """Guards CONF-1 at d9c1b733 in the other shell argv consumer."""
    source = f"{{ {consumer} <<'SH'\n{{ uv pip install -r/tests/deps.txt; }}\nSH\n}}"
    assert named(tmp_path, source) == ["/tests/deps.txt"]


def formatted(template, form):
    if form == "percent":
        return repr(template.replace("{}", "%s")) + " % APP"
    if form == "mapping":
        return repr(template.replace("{}", "%(app)s")) + " % {'app': APP}"
    if form == "format":
        return repr(template) + ".format(APP)"
    if form == "format_map":
        return repr(template.replace("{}", "{app}")) + ".format_map({'app': APP})"
    if form == "fstring":
        return "f" + repr(template.replace("{}", "{APP}"))
    return repr(template.split("{}")[0]) + " + APP + " + repr(template.split("{}")[1])


@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize(
    "form", ["percent", "mapping", "format", "format_map", "fstring", "concat"]
)
@pytest.mark.parametrize(
    "option", ["-r", "-c", "--requirement=", "--constraint=", "-qr"]
)
@pytest.mark.parametrize("language", ["python", "heredoc", "command", "unknown"])
def test_formatted_options_keep_grammar_and_basename(
    tmp_path, tool, form, option, language
):
    """Guards CONF-2 at d9c1b733 through every argv discovery caller."""
    words = shlex.split(tool)
    argv = ", ".join(repr(word) for word in words)
    source = f"import subprocess\nAPP = '/app'\nsubprocess.run([{argv}, {formatted(option + '{}/deps.txt', form)}])\n"
    if language == "python":
        source = "#!/usr/bin/python3\n" + source
    elif language == "command":
        source = "python3 -Ic " + shlex.quote(source)
    else:
        source = (
            ("python3 - <<'PY'\n" if language == "heredoc" else "python3 - <<MISSING\n")
            + source
            + "PY\n"
        )
    expected = ["deps.txt"]
    if tool.startswith("uv run"):
        expected.insert(0, "other.txt" if language == "unknown" else "/tests/other.txt")
    assert named(tmp_path, source) == sorted(expected)


@pytest.mark.parametrize(
    "expression, expected",
    [
        ('"-r%10.5s/deps.txt" % APP', ["deps.txt"]),
        ('"-r%*s/deps.txt" % (2, APP)', ["deps.txt"]),
        ('"-r%(app)-10s/deps.txt" % {"app": APP}', ["deps.txt"]),
        ('"--requirement=%s/100%%/deps.txt" % APP', ["deps.txt"]),
        ('"-r{app!s:>{width}}/deps.txt".format(app=APP, width=2)', ["deps.txt"]),
        ('"-r{0[app]}/deps.txt".format(VALUES)', ["deps.txt"]),
        ('"-r{app.path}/deps.txt".format(app=APP)', ["deps.txt"]),
        ('"-r{}/{{literal}}/deps.txt".format(APP)', ["deps.txt"]),
        ('("-r" + "%s/deps.txt") % APP', ["deps.txt"]),
        ('("-r" + "{}/deps.txt").format(APP)', ["deps.txt"]),
        ('("-r%s" % APP) + "/deps.txt"', ["deps.txt"]),
        ('"-r{}".format(APP) + "/deps.txt"', ["deps.txt"]),
        ('"-r%s" % FILE', None),
        ('"-r{}".format(FILE)', None),
        ('"-r{file}".format_map(VALUES)', None),
    ],
)
def test_format_grammar_keeps_substitutions_opaque(tmp_path, expression, expected):
    """Guards CONF-2 at d9c1b733 without evaluating fields or substitutions."""
    source = f"#!/usr/bin/python3\nimport subprocess\nsubprocess.run(['pip', 'install', {expression}])\n"
    assert named(tmp_path, source) == expected


@pytest.mark.parametrize(
    "expression",
    [
        '"-r%s" % (APP + "/deps.txt")',
        '"-r%*.*s" % (100, 100, APP + "/deps.txt")',
        '"-r%(path)s" % {"path": APP + "/deps.txt"}',
        '"-r{}".format(APP + "/deps.txt")',
        '"-r{0}".format(APP + "/deps.txt")',
        '"-r{path}".format(path=APP + "/deps.txt")',
        '"-r{path}".format_map({"path": APP + "/deps.txt"})',
        '"%s%s" % ("-r", APP + "/deps.txt")',
        '"{}{}".format("-r", APP + "/deps.txt")',
        '"{flag}{path}".format(flag="-r", path=APP + "/deps.txt")',
        '"{flag}{path}".format_map({"flag": "-r", "path": APP + "/deps.txt"})',
    ],
)
@pytest.mark.parametrize(
    "tool", ["pip install", "python3 -mpip install", "uv pip install"]
)
@pytest.mark.parametrize("language", ["python", "heredoc", "command", "unknown"])
def test_substitution_fragments_reach_the_same_grammar(
    tmp_path, expression, tool, language
):
    """Guards CONF-2 at d9c1b733 when the substitution carries the basename."""
    argv = ", ".join(repr(word) for word in shlex.split(tool))
    source = f"import subprocess\nsubprocess.run([{argv}, {expression}])\n"
    if language == "python":
        source = "#!/usr/bin/python3\n" + source
    elif language == "command":
        source = "python3 -c " + shlex.quote(source)
    else:
        source = (
            ("python3 - <<'PY'\n" if language == "heredoc" else "python3 - <<MISSING\n")
            + source
            + "PY\n"
        )
    assert named(tmp_path, source) == ["deps.txt"]


@pytest.mark.parametrize(
    "variant",
    ["brace", "percent", "format", "format_map", "percent_argument", "format_argument"],
)
@pytest.mark.parametrize("case", ["oracle", "nop", "hostile"])
@pytest.mark.parametrize("workspace", ["app", "testbed", "root"])
@pytest.mark.parametrize("plane", ["same", "separate"])
async def test_confirmation11_inputs_through_real_guard(
    tmp_path, monkeypatch, variant, case, workspace, plane
):
    """Guards CONF-1/2 at d9c1b733 through hardening, real pytest and scoring."""
    proof = tmp_path / "hostile-imported"
    original = uv_state.make_task

    def make_task(layout, env=None):
        interpreter = f"{layout.workspace}/.venv/bin/python"
        directory = str(layout.workspace) if case == "hostile" else "/tests"
        if variant == "brace":
            body = f"{{ uv pip install --python {interpreter} -r {directory}/deps.txt; }}\n"
        else:
            argument = {
                "percent_argument": "'-r%s' % (APP + '/deps.txt')",
                "format_argument": "'-r{}'.format(APP + '/deps.txt')",
            }.get(variant, formatted("-r{}/deps.txt", variant))
            body = f"{interpreter} - <<'PY'\nimport subprocess, sys\nAPP = {directory!r}\nsubprocess.run([sys.executable, '-mpip', 'install', {argument}])\nPY\n"
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
        before=edit if case == "hostile" else None,
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
