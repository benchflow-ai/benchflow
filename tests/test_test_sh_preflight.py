"""EN-715: what test.sh needs, read from its own text before the agent's turn.

A refusal must never come from a guess: only the interpreter and the fatal
errexit prefix (plain top-level simple commands before anything that could
install a tool, change PATH or hand control elsewhere) can refuse a trial.
Every other command test.sh names is a warning that changes no score. The
real-Docker trials are in tests/test_test_sh_preflight_docker.py.
"""

from __future__ import annotations

import pytest

from benchflow.task.test_sh_preflight import (
    REFUSAL_HEAD,
    WARNING_TEMPLATE,
    analyse,
    last_not_found,
    wrong_zero_candidate,
)

# autocodebench lite200's test.sh (all 200 tasks share it).
AUTOCODEBENCH = """#!/bin/bash
set -Eeuo pipefail

TEST_DIR="${TEST_DIR:-/tests}"

uv init
uv add pytest==8.4.1
uv add /data/sandbox_local_package/dist/sandbox_local-1.0.0-py3-none-any.whl


if uv run pytest $TEST_DIR/test_outputs.py -rA; then
   echo "All tests passed! "
   echo 1 > /logs/verifier/reward.txt

else
    echo "Tests failed!"
    echo 0 > /logs/verifier/reward.txt
fi
"""
# Terminal-Bench's usual shape: install uv in test.sh itself, then use it.
INSTALLS_ITS_OWN = """#!/bin/bash
set -e
apt-get update
apt-get install -y curl
curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh
source $HOME/.local/bin/env
uvx -p 3.12 -w pytest pytest /tests/test_outputs.py
"""


def test_autocodebench_needs_uv_before_anything_could_install_it():
    needs = analyse(AUTOCODEBENCH)
    assert needs is not None
    assert needs.interpreter == ("path", "/bin/bash")
    assert needs.fatal == ("uv",)
    assert needs.named == ("uv",)
    assert "uv" in needs.prefix_end  # `uv add` can install


def test_a_script_that_installs_its_own_tools_is_never_refused_for_them():
    needs = analyse(INSTALLS_ITS_OWN)
    assert needs.fatal == ("apt-get",)
    assert "uvx" not in needs.fatal and "curl" not in needs.fatal
    # Still named, for the warning.
    assert {"apt-get", "curl", "sh", "uvx"} <= set(needs.named)


@pytest.mark.parametrize(
    "body",
    [
        "foo || true\n",
        "foo && echo ok\n",
        "if foo; then echo 1 > /logs/verifier/reward.txt; fi\n",
        "foo | tee /tmp/log\n",
        "! foo\n",
        "( foo )\n",
        "foo &\n",
        "while foo; do :; done\n",
        "command -v foo >/dev/null && foo\n",
    ],
)
def test_a_command_the_script_can_survive_is_named_not_fatal(body):
    needs = analyse("#!/bin/bash\nset -e\n" + body)
    assert needs.fatal == ()
    assert "foo" in needs.named


def test_without_errexit_nothing_is_fatal():
    needs = analyse("#!/bin/bash\nfoo\necho 0 > /logs/verifier/reward.txt\n")
    assert needs.fatal == ()
    assert needs.named == ("foo",)


@pytest.mark.parametrize(
    ("first", "interpreter"),
    [
        ("#!/bin/bash", ("path", "/bin/bash")),
        ("#!/bin/sh", ("path", "/bin/sh")),
        ("#!/usr/bin/env bash", ("env", "bash")),
        ("#!/usr/bin/env -S bash -e", ("env", "bash")),
        ("set -e", ("default", None)),
    ],
)
def test_the_interpreter_is_the_shebang_s_else_bash_or_sh(first, interpreter):
    needs = analyse(first + "\nfoo\n")
    assert needs.interpreter == interpreter


@pytest.mark.parametrize(
    "first", ["#!/usr/bin/env python3", "#!/usr/bin/python3", "#!/bin/zsh"]
)
def test_a_script_that_is_not_sh_or_bash_is_not_read(first):
    assert analyse(first + "\nprint(1)\n") is None


@pytest.mark.parametrize(
    "script",
    [
        "#!/bin/bash -e\nfoo\n",
        "#!/bin/bash\nset -o errexit\nfoo\n",
        "#!/bin/bash\nset -euo pipefail\nfoo --flag > /tmp/out 2>&1\n",
        "#!/bin/bash\nset -e\nX=1 foo\n",
        "#!/bin/bash\nset -e\ncommand foo\n",
        "#!/bin/bash\nset -e\nfoo; echo done\n",
    ],
)
def test_errexit_makes_a_plain_top_level_command_fatal(script):
    assert analyse(script).fatal == ("foo",)


@pytest.mark.parametrize(
    ("body", "why"),
    [
        ("set +e\n", None),
        ("PATH=/opt/x/bin:$PATH\n", "PATH"),
        ("export PATH=/opt/x/bin:$PATH\n", "export"),
        ("source ./env.sh\n", "source"),
        (". ./env.sh\n", "`.`"),
        ('eval "$SETUP"\n', "eval"),
        ("helper() { echo; }\n", "function"),
        ("cat <<EOF\nhello\nEOF\n", "heredoc"),
        ("$PYTHON -m pytest\n", "cannot name"),
        ("x=$(setup)\n", "substitution"),
        ("pip install pytest\n", "pip"),
        ("uv tool install ruff\n", "uv"),
        ("cd /app && make\n", "list"),
        ("if [ -x /opt/x ]; then :; fi\n", "compound"),
    ],
)
def test_the_fatal_prefix_ends_before_anything_that_could_change_what_runs(body, why):
    needs = analyse("#!/bin/bash\nset -e\n" + body + "foo\n")
    assert "foo" not in needs.fatal
    if why is not None:
        assert why in needs.prefix_end, needs.prefix_end


def test_a_function_body_is_not_top_level():
    needs = analyse("#!/bin/bash\nhelper() {\n  inside\n}\noutside\n")
    assert "inside" not in needs.named
    assert "outside" in needs.named


def test_builtins_keywords_assignments_and_relative_paths_are_never_names():
    needs = analyse(
        "#!/bin/bash\nset -e\ncd /tests\necho hi\nX=1\n[ -f x ]\n"
        "test -d y\ntrue\nfoo\n./run.sh\nbar\n"
    )
    assert needs.fatal == ("foo",)
    # A relative script is not resolved, and could change what runs.
    assert "cannot name" in needs.prefix_end
    assert needs.named == ("foo", "bar")


def test_an_absolute_path_is_named_as_written():
    assert analyse("#!/bin/bash\nset -e\n/opt/tool/bin/check\n").fatal == (
        "/opt/tool/bin/check",
    )


@pytest.mark.parametrize(
    ("stdout", "name"),
    [
        ("/tests/test.sh: line 6: uv: command not found\n", "uv"),
        ("/tests/test.sh: 6: uv: not found\n", "uv"),
        (
            "a: command not found\nok\n/tests/test.sh: line 9: b: command not found\n",
            "b",
        ),
        ("all good\n", None),
    ],
)
def test_the_last_command_the_shell_reported_not_found(stdout, name):
    assert last_not_found(stdout) == name


@pytest.mark.parametrize(
    ("reward", "stdout", "candidate"),
    [
        (0.0, "test.sh: line 3: foo: command not found\nTests failed\n", "foo"),
        (1.0, "test.sh: line 3: foo: command not found\n", None),
        (0.0, "test.sh: line 3: bar: command not found\n", None),
        (0.0, "Tests failed\n", None),
        (None, "test.sh: line 3: foo: command not found\n", None),
    ],
)
def test_a_wrong_zero_candidate_needs_a_0_and_the_warned_command_last(
    reward, stdout, candidate
):
    record = {"missing": ["foo"], "warnings": [WARNING_TEMPLATE.format("foo")]}
    assert wrong_zero_candidate(record, reward, stdout) == candidate


def test_the_refusal_and_the_warning_read_as_documented():
    assert REFUSAL_HEAD.startswith("sandbox startup refused: ")
    assert WARNING_TEMPLATE.format("uv") == (
        "test.sh names `uv`, not on the verifier PATH at preflight"
    )


# SWE-bench Multilingual's test.sh shape: a quoted-delimiter heredoc with a
# diff, a second heredoc, then tools looked up in conditions.
MULTILINGUAL = """#!/bin/bash
set -uo pipefail -x
git apply --verbose --reject - <<'EOF_114329324912'
diff --git a/x b/x
index 1..2 100644
--- a/x
+++ b/x
@@ -1 +1 @@
-old
+new
EOF_114329324912
cat > parser.py <<EOF
import sys
EOF
if command -v uv &> /dev/null; then
    uv run parser.py
elif command -v pip &> /dev/null; then
    pip install -q swebench || true
    python parser.py | tee -a "$LOG_FILE"
fi
"""


def test_a_quoted_heredoc_delimiter_hides_its_body():
    needs = analyse(MULTILINGUAL)
    assert needs.named == ("git", "cat", "uv", "pip", "python", "tee")
    assert not any(w in needs.named for w in ("index", "---", "+++", "@@", "-old"))


def test_functions_test_sh_defines_and_arithmetic_are_never_names():
    needs = analyse(
        "#!/bin/bash\n_collect() {\n  cp a b\n}\nfilter() { :; }\n"
        "ELAPSED=$((ELAPSED + 5))\n(( ELAPSED > 3 )) && echo late\n"
        "_collect\nfilter x\nfoo\n"
    )
    assert needs.named == ("foo",)


# --- a branch child carries its source's verdict (RUSAGE on EN-715) ------------


def _applicable_task(tmp_path):
    from types import SimpleNamespace

    test_sh = tmp_path / "tests" / "test.sh"
    test_sh.parent.mkdir(parents=True)
    test_sh.write_text("#!/bin/bash\nset -e\nmytool\n")
    verifier = SimpleNamespace(
        type="test-script", service="main", sandbox=None, sandbox_mode=None
    )
    return SimpleNamespace(
        config=SimpleNamespace(verifier=verifier),
        paths=SimpleNamespace(uses_native_verifier_dir=False, test_path=test_sh),
    )


def _child(task, inherited_record):
    from benchflow.rollout import Rollout

    child = Rollout.__new__(Rollout)
    child._from_branch_snapshot = True
    child._task = task
    child._test_sh_preflight_inherited = inherited_record
    return child


def test_a_branch_child_carries_the_source_verdict_marked():
    from benchflow.task.test_sh_preflight import inherited

    source = {
        "checked": True,
        "fatal": ["mytool"],
        "missing": ["othertool"],
        "warnings": [WARNING_TEMPLATE.format("othertool")],
        "wrong_zero_candidate": "othertool",
    }
    carried = inherited(source)
    assert carried["inherited"] is True
    assert (carried["fatal"], carried["missing"]) == (["mytool"], ["othertool"])
    # The source's own wrong-0 reading is its own.
    assert "wrong_zero_candidate" not in carried


def test_the_source_record_comes_from_its_result_json(tmp_path):
    import json

    from benchflow.task.test_sh_preflight import source_record

    (tmp_path / "result.json").write_text(
        json.dumps({"test_sh_preflight": {"checked": True, "fatal": ["mytool"]}})
    )
    assert source_record(tmp_path) == {"checked": True, "fatal": ["mytool"]}
    assert source_record(tmp_path / "missing") is None


@pytest.fixture
def never_checked(monkeypatch):
    from benchflow.task import test_sh_preflight

    async def judged(*args, **kwargs):
        raise AssertionError("a branch child judged a post-agent sandbox")

    monkeypatch.setattr(test_sh_preflight, "check", judged)


async def test_a_branch_child_never_judges_the_restored_sandbox(
    tmp_path, never_checked
):
    """The restored checkpoint is the source's sandbox after its agent ran: a
    tool the agent removed must stay its failure, never become a refusal."""
    from benchflow.rollout import Rollout

    child = _child(
        _applicable_task(tmp_path),
        {"checked": True, "fatal": ["mytool"], "missing": [], "warnings": []},
    )
    await Rollout._check_test_sh(child)
    assert child._test_sh_preflight == {
        "checked": True,
        "fatal": ["mytool"],
        "missing": [],
        "warnings": [],
        "inherited": True,
    }


async def test_a_carried_refusal_refuses_the_child(tmp_path, never_checked):
    from benchflow.rollout import Rollout
    from benchflow.task.test_sh_preflight import TestShPreflightRefused

    child = _child(
        _applicable_task(tmp_path),
        {
            "checked": True,
            "fatal": ["uv"],
            "missing": ["uv"],
            "refused": "uv",
            "user": "root",
            "path": "/usr/bin:/bin",
        },
    )
    with pytest.raises(TestShPreflightRefused) as raised:
        await Rollout._check_test_sh(child)
    assert str(raised.value).startswith(REFUSAL_HEAD + "`uv`")
    assert "decided before the source trial's turn" in str(raised.value)


async def test_no_verdict_to_carry_ends_the_child_never_as_the_task_s_fault(
    tmp_path, never_checked
):
    """A checkpoint from before the check, with no fresh-image verdict either:
    the child is not run, unscored, as BenchFlow's setup."""
    from benchflow._utils.scoring import is_unrecoverable_startup_error
    from benchflow.failures import FAULT_WORDS, cause_of
    from benchflow.rollout import Rollout
    from benchflow.task.test_sh_preflight import (
        UNAVAILABLE_HEAD,
        TestShPreflightUnavailable,
    )

    child = _child(_applicable_task(tmp_path), None)
    with pytest.raises(TestShPreflightUnavailable) as raised:
        await Rollout._check_test_sh(child)
    message = str(raised.value)
    assert message.startswith(UNAVAILABLE_HEAD)
    assert is_unrecoverable_startup_error(message.lower())
    cause = cause_of({"error": message, "error_category": "sandbox_setup"})
    assert cause is not None and cause.key == "test_sh_preflight_unavailable"
    assert FAULT_WORDS[cause.fault] != "task problem"


async def test_a_task_the_check_does_not_read_proceeds(never_checked):
    from types import SimpleNamespace

    from benchflow.rollout import Rollout

    child = _child(SimpleNamespace(config=None, paths=None), None)
    await Rollout._check_test_sh(child)
    assert child._test_sh_preflight is None


def test_a_trap_ends_the_fatal_prefix():
    """An EXIT trap can still write the reward when a command is missing
    (RUSAGE): what follows a trap is named, never fatal."""
    needs = analyse(
        "#!/bin/bash\ntrap 'echo 0 > /logs/verifier/reward.txt' EXIT\nset -e\nmytool\n"
    )
    assert needs.fatal == ()
    assert needs.named == ("mytool",)
    assert "trap" in needs.prefix_end


async def test_a_branch_child_of_a_test_sh_the_check_does_not_read_proceeds(
    tmp_path, never_checked
):
    """RUSAGE: a ``#!/usr/bin/env python3`` test.sh is run by the verifier but
    never read by the check, so its source records no verdict; the child has
    none to miss and proceeds."""
    from benchflow.rollout import Rollout
    from benchflow.task.test_sh_preflight import applies, needs_verdict

    task = _applicable_task(tmp_path)
    task.paths.test_path.write_text(
        "#!/usr/bin/env python3\nopen('/logs/verifier/reward.txt', 'w').write('1')\n"
    )
    assert applies(task) and not needs_verdict(task)
    child = _child(task, None)
    await Rollout._check_test_sh(child)
    assert child._test_sh_preflight is None


def test_a_child_runs_as_its_source_s_sandbox_user(tmp_path):
    import json

    from benchflow.branch_run import _UNKNOWN, _source_sandbox_user

    trial = tmp_path / "job" / "t__1"
    trial.mkdir(parents=True)
    assert _source_sandbox_user(trial) is _UNKNOWN
    (trial.parent / "evaluation.json").write_text(
        json.dumps({"config": {"sandbox_user": "agent"}})
    )
    assert _source_sandbox_user(trial) == "agent"
    # The trial's own config.json comes first: a root agent's records None.
    (trial / "config.json").write_text(json.dumps({"sandbox_user": None}))
    assert _source_sandbox_user(trial) is None


@pytest.mark.parametrize(
    "key", ["test_sh_command_missing", "test_sh_preflight_unavailable"]
)
def test_bench_explain_has_a_page_for_each_preflight_cause(key):
    """RUSAGE F6: the refusal and the CHANGELOG point at `bench explain`."""
    import os
    import subprocess
    import sys

    from benchflow.failures import cause_keys

    assert key in cause_keys()
    out = subprocess.run(
        [sys.executable, "-m", "benchflow.cli.main", "explain", key],
        env={
            **os.environ,
            "NO_COLOR": "1",
            "COLUMNS": "200",
            "HARBOR_TELEMETRY": "off",
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "no explain page" not in (out.stdout + out.stderr).lower()
