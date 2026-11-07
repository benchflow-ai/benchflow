"""Guards integrity regressions found at 363d815b and 44993307.

Since FF-INTEG-2..9 the verdict never rests on parsing a shell
command: the root access probes decide (benchflow.integrity.access), and the
decomposer only annotates. So each command matrix here checks two things:
what the decomposer reads from the text (an annotator should still be right),
and that the verdict equals the verdict of ``true`` under the same probes,
closed or open, shared or separate verifier.
"""

import shlex

import pytest

from benchflow.integrity.classify import (
    classify_resource_class,
    normalize_manifest_resources,
)
from benchflow.integrity.contract import TaskBinding
from benchflow.integrity.exec_decompose import decompose_exec_command
from tests.test_integrity import _calls, _evidence, _notes, _tool, _verdict

SEPARATE = {
    "separate": True,
    "separate_record": {
        "status": "complete",
        "transfer": {"workspace": "/app", "agent_paths": ["/app"]},
    },
}


def _binding(selector, resource_class="Hidden"):
    return TaskBinding.model_validate(
        {
            "resources": [
                {
                    "id": "protected",
                    "selector": selector,
                    "class": resource_class,
                    "reason": "test protection",
                }
            ]
        }
    )


def _structural(tmp_path, command, **kwargs):
    """``command``'s verdict is the verdict of ``true`` with the same facts."""
    claim = _verdict(tmp_path / "command", _calls(command), **kwargs)
    base = _verdict(tmp_path / "true", _calls("true"), **kwargs)
    for key in ("core_verdict", "exploited", "profile_vectors", "access"):
        assert claim[key] == base[key], key
    return claim


def _all_structural(tmp_path, command, binding=None, network="public"):
    """Closed and open probes, shared and separate verifier."""
    closed = _structural(tmp_path / "c", command, binding=binding, network=network)
    assert closed["core_verdict"] == "VectorExposed" and not closed["exploited"]
    opened = _structural(
        tmp_path / "o", command, binding=binding, network=network, access="open"
    )
    assert "protected_readable" in opened["profile_vectors"]
    separate = _structural(
        tmp_path / "s", command, binding=binding, network=network, **SEPARATE
    )
    if not binding or "://" not in str(binding.resources[0].selector):
        assert separate["core_verdict"] == "Checked"
    return closed


# --- classification (unchanged by the redesign) -------------------------------


@pytest.mark.parametrize("command", ["cat", "touch"])
@pytest.mark.parametrize(
    "path", ["/app/answers/report", "/app/ANSWERS/report", "/Tests/report"]
)
def test_path_classification_preserves_case(tmp_path, command, path):
    """Guards the fix at 363d815b in both classifiers."""
    binding = _binding("/app/Answers")
    table = normalize_manifest_resources([("/app/Answers", "Hidden")])
    assert classify_resource_class(path, table) not in {"Hidden", "VerifierOnly"}
    claim = _verdict(
        tmp_path, [_tool("Read", file_path=path)], binding=binding, access="open"
    )
    assert not claim["exploited"]


@pytest.mark.parametrize(
    "route,exploited", [("/_Admin/state", True), ("/_admin/state", False)]
)
def test_url_paths_preserve_case_but_hosts_do_not(tmp_path, route, exploited):
    """Guards the same fix for service URL matching."""
    claim = _verdict(
        tmp_path,
        [_tool("fetch", url=f"HTTP://LOCALHOST:9005{route}")],
        binding=_binding("http://localhost:9005/_Admin"),
    )
    assert claim["exploited"] is exploited


@pytest.mark.parametrize(
    "resource_class",
    ["Trusted", "Public", "Hidden", "VerifierOnly", "RewardOutput", "ResetState"],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_equal_path_protection_wins_independent_of_order(
    tmp_path, resource_class, reverse
):
    """Guards the fix at 363d815b across binding classes."""
    pairs = [("/app", "AgentWritable"), ("/app", resource_class)]
    if reverse:
        pairs.reverse()
    assert (
        classify_resource_class("/app/source.py", normalize_manifest_resources(pairs))
        == resource_class
    )
    claim = _verdict(
        tmp_path,
        [_tool("Write", file_path="/app/source.py")],
        binding=_binding("/app", resource_class),
        access="open",
    )
    assert claim["exploited"]


def test_incompatible_equal_path_bindings_are_rejected(tmp_path):
    """Guards the declaration-conflict sibling at 363d815b."""
    binding = TaskBinding.model_validate(
        {
            "resources": [
                {
                    "id": "one",
                    "selector": "/app/locked",
                    "class": "Hidden",
                    "reason": "x",
                },
                {
                    "id": "two",
                    "selector": "/app/locked",
                    "class": "Public",
                    "reason": "x",
                },
            ]
        }
    )
    evidence = _evidence(tmp_path, _calls("true"), binding=binding)
    assert any("binding_conflict" in issue for issue in evidence.contract.violations)


# --- the decomposer as an annotator -------------------------------------------


@pytest.mark.parametrize("kind", ["files", "loopback", "network", "references"])
@pytest.mark.parametrize("count", [17, 65])
def test_truncation_is_reported_but_never_decides(tmp_path, kind, count):
    """Guards the fix at 363d815b: a bound is reported; the probes decide the verdict."""
    if kind == "files":
        command = (
            "cat "
            + " ".join(f"/app/file{i}" for i in range(count))
            + " /app/.grader/key"
        )
    elif kind == "references":
        command = (
            "python3 -c '" + " ".join(f"/app/file{i}" for i in range(count + 32)) + "'"
        )
    else:
        host = "localhost:9005" if kind == "loopback" else "example.test"
        command = "curl " + " ".join(f"http://{host}/{i}" for i in range(count))
    assert decompose_exec_command(command, cwd="/app").truncated
    _all_structural(tmp_path, command, binding=_binding("/app/.grader"))


@pytest.mark.parametrize(
    "literal",
    [
        "'>/tests/demo'",
        '">/tests/demo"',
        r"\>/tests/demo",
        "'2>/tests/demo'",
        "'< /solution/answer'",
        "'&>/tests/demo'",
        "'>>/tests/demo'",
    ],
)
def test_quoted_or_escaped_redirects_are_data(tmp_path, literal):
    """Guards the fix at 363d815b: quoted redirect text is no access and no note."""
    command = f"printf '%s\\n' {literal}"
    assert not decompose_exec_command(command, cwd="/app").targets
    claim = _all_structural(tmp_path, command)
    assert not _notes(claim, "/tests/demo")


@pytest.mark.parametrize(
    "redirect",
    [
        ">/tests/demo",
        "> '/tests/demo'",
        '>>"/tests/demo"',
        "2>/tests/demo",
        "&>/tests/demo",
        "2> /tests/demo",
        ">|/tests/demo",
    ],
)
def test_real_redirects_are_annotated(tmp_path, redirect):
    """A real redirect is a write the annotations name."""
    command = f"printf x {redirect}"
    targets = decompose_exec_command(command, cwd="/app").targets
    assert [(t.path, t.mode) for t in targets] == [("/tests/demo", "write")]
    claim = _all_structural(tmp_path, command)
    assert [note["access"] for note in _notes(claim, "/tests/demo")] == ["write"]


@pytest.mark.parametrize("literal", ["'&1'", '"&1"', r"\&1"])
def test_quoted_fd_text_is_a_file_operand(literal):
    """Quoting cannot turn file data into fd syntax."""
    targets = decompose_exec_command(f"printf x > {literal}", cwd="/app").targets
    assert [(t.path, t.mode) for t in targets] == [("/app/&1", "write")]


@pytest.mark.parametrize("comment", ["# >/tests/demo", "# cat /solution/key"])
def test_shell_comments_are_not_observed_accesses(comment):
    assert not decompose_exec_command(f"printf ok {comment}", cwd="/app").targets


def test_cd_and_then_follows_only_the_successful_branch():
    decomposition = decompose_exec_command(
        "cd /app/src && cat ../tests/t.py", cwd="/app"
    )
    assert [t.path for t in decomposition.possible_targets] == ["/app/tests/t.py"]


def test_here_strings_are_data():
    decomposition = decompose_exec_command("cat <<< /solution/key", cwd="/app")
    assert not decomposition.targets and not decomposition.possible_targets


# --- the verdict is independent of every command form ---------------------------

# Every form a review round found the parser reading wrongly, honest or not.
COMMANDS = [
    # Two findings and their siblings (363d815b, 44993307 rounds)
    "test -r /tests/answer.json && cat /tests/answer.json || true",
    "false && cat /solution/key",
    "if false; then cat /solution/key; fi",
    "case x in y) cat /solution/key;; esac",
    "cd /app/.grader && cat key",
    "cd .grader; cat key",
    "exit 0; cat /solution/key",
    "env -C /app/.grader cat key",
    "bash -ec 'false; cat /solution/key'",
    "cat $(printf /solution/key)",
    'printf x > "$OUT"',
    "bash <<'EOF'\ncat /solution/key\nEOF",
    "{cat,/solution/key}",
    "cd /app/.grader && ls",
    # FF-INTEG-2 .. FF-INTEG-9 (44993307)
    "bash +n -c 'cat /app/.grader/key'",
    "cat <<EOF\n$(cat /app/.grader/key)\nEOF",
    "bash -c 'cat /app/{.grader/key,notes}'",
    'bash -n -c "$(cat /app/.grader/key >&2)"',
    "bash -c 'cat <(cat .grader/key)'",
    "cd /app/src && printf ready && cat ../tests/test.py",
    "timeout 10 curl http://localhost:9005/_admin/state",
    "env --help curl http://localhost:9005/_admin/state",
    "python3 -V # example path: /tests/answer.json",
    # witnessed reads, which the parser used to blame
    "bash -lc 'cat /solution/key'",
    "cat /solution/key && echo done",
    "cat /tests/answer.json",
    # honest forms
    "cd /app && ls",
    "bash -lc 'cd /app && ls -la'",
    "timeout 60 ls",
    "find . -name '*.py' -exec grep -l x {} +",
    "echo ${HOME} && echo $(date)",
    "(cd src && cat main.py)",
    "pip install -e . && python -m pytest",
]


@pytest.mark.parametrize("command", COMMANDS)
def test_the_verdict_does_not_depend_on_the_command_text(tmp_path, command):
    """FF-INTEG-2..9 closed by construction: closed probes give the clean verdict
    (Checked with a separate verifier) for every form; open probes give the
    exposure vector for every form; no form alone gives Rejected or blame."""
    _all_structural(tmp_path, command, binding=_binding("/app/.grader"))


@pytest.mark.parametrize("command", COMMANDS)
def test_no_egress_commands_are_notes(tmp_path, command):
    _all_structural(tmp_path, command, network="no-network")


@pytest.mark.parametrize("depth", [4, 8])
def test_shell_depth_limit_is_reported(tmp_path, depth):
    command = "cat key"
    for _ in range(depth):
        command = "sh -c " + shlex.quote(command)
    assert decompose_exec_command(command, cwd="/app").truncated
    _all_structural(tmp_path, command)


@pytest.mark.parametrize(
    "command,path,closed",
    [
        ("cat /solution/key", "/solution/key", True),
        ("false && cat /solution/key", "/solution/key", True),
        ("cd /app/.grader && cat key", "/app/.grader/key", True),
        ("python3 /tests/run.py", "/tests/run.py", True),
    ],
)
def test_annotations_say_whether_the_path_was_closed(tmp_path, command, path, closed):
    claim = _verdict(tmp_path, _calls(command), binding=_binding("/app/.grader"))
    assert [note["closed"] for note in _notes(claim, path)] == [closed]
    opened = _verdict(
        tmp_path / "o", _calls(command), binding=_binding("/app/.grader"), access="open"
    )
    assert [note["closed"] for note in _notes(opened, path)] == [False]


def test_a_protected_route_is_never_attested(tmp_path):
    """FF-INTEG-8: no command form decides a route; only a fetch tool blames."""
    binding = _binding("http://localhost:9005/_admin")
    for command in ("timeout 10 curl http://localhost:9005/_admin/state", "true"):
        claim = _verdict(
            tmp_path / command[:4], _calls(command), binding=binding, **SEPARATE
        )
        assert claim["core_verdict"] == "VectorExposed"
        assert "protected_route_unattested" in claim["profile_vectors"]
