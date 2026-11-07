"""The exec receipt (``_exec_receipt.py``): what lockdown decides on is what
the sandbox printed, byte for byte, or the read fails; never a rewritten value.
"""

from __future__ import annotations

import ast
import itertools
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox import remote_docker as remote
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox._exec_receipt import (
    RECEIPT_FAILED,
    encode_receipt,
    new_receipt_nonce,
    posix_cksum,
    read_receipt,
    receipt_command,
    receipt_exec,
    unwrap_receipt_command,
)

SH = shutil.which("sh") or "/bin/sh"


@pytest.fixture(autouse=True)
def no_ssh_config(monkeypatch):
    if hasattr(remote.RemoteDockerHost, "ssh_view"):
        monkeypatch.setattr(
            remote.RemoteDockerHost,
            "ssh_view",
            lambda self: remote._SshView((), (), True, False),
        )


def _run(script: str, env: dict[str, str] | None = None) -> ExecResult:
    done = subprocess.run(
        [SH, "-c", script], capture_output=True, timeout=30, env=env, check=False
    )
    return ExecResult(
        stdout=done.stdout.decode("utf-8", errors="replace"),
        stderr=done.stderr.decode("utf-8", errors="replace"),
        return_code=done.returncode,
    )


# --- the checksum --------------------------------------------------------


@pytest.mark.parametrize(
    "data,expected",
    [(b"", 4294967295), (b"123456789", 930766865), (b"a", 1220704766)],
)
def test_checksum_is_posix_cksum(data, expected):
    assert posix_cksum(data) == expected


@pytest.mark.parametrize("data", [b"", b"0a30", bytes(range(256)).hex().encode()])
def test_checksum_matches_the_sandbox_cksum(data):
    done = subprocess.run(
        [SH, "-c", "cksum"], input=data, capture_output=True, timeout=30, check=True
    )
    crc, length = done.stdout.split()
    assert (int(crc), int(length)) == (posix_cksum(data), len(data))


# --- the script, run by a real shell --------------------------------------

OUTPUTS = [
    ("printf ''", b"", 0),
    ("printf 'one line'", b"one line", 0),
    ("printf 'a\\nb\\n\\n'", b"a\nb\n\n", 0),
    ("printf 'x\\000y\\000'", b"x\0y\0", 0),
    ("printf '%s' 'tab\there'; exit 3", b"tab\there", 3),
    ("exit 127", b"", 127),
    ("printf 'out'; echo err >&2; false", b"out", 1),
    ("printf 'ok' # a trailing comment", b"ok", 0),
    ("set -e; printf 'gone'; false; printf 'never'", b"gone", 1),
    ("f() ( case a in a) printf case;; esac ); f", b"case", 0),
    ("printf '%s' '\"$HOME`false`'; exit 7", b'"$HOME`false`', 7),
]


@pytest.mark.parametrize("command,stdout,status", OUTPUTS)
def test_script_carries_output_and_status_exactly(command, stdout, status):
    nonce = new_receipt_nonce()
    raw = _run(receipt_command(command, nonce))

    assert raw.return_code == 0, raw.stderr
    read = read_receipt(raw, nonce)
    assert (read.stdout.encode(), read.return_code) == (stdout, status)
    assert raw.stdout.endswith(encode_receipt(stdout, status, nonce))


def test_command_stderr_stays_a_message():
    nonce = new_receipt_nonce()
    raw = _run(receipt_command("echo note >&2; printf data", nonce))
    merged = ExecResult(
        stdout=raw.stderr + raw.stdout, stderr="", return_code=raw.return_code
    )
    read = read_receipt(merged, nonce)
    assert read.stdout == "data"
    assert "note" in (read.stderr or "")


def test_encoder_ignores_tools_the_exec_path_puts_first(tmp_path):
    """The encoder runs as root; a folder the agent wrote on PATH is not used."""
    fake = tmp_path / "bin"
    fake.mkdir()
    ran = tmp_path / "ran"
    for tool in ("od", "tr", "cksum"):
        script = fake / tool
        script.write_text(f"#!/bin/sh\ntouch {ran}\nprintf '0a30'\n")
        script.chmod(0o755)
    nonce = new_receipt_nonce()
    path = f"{fake}:{os.environ.get('PATH', '/usr/bin:/bin')}"
    raw = _run(receipt_command("printf real", nonce), env={"PATH": path})

    assert read_receipt(raw, nonce).stdout == "real"
    assert not ran.exists()


def test_unwrap_returns_the_command_and_nonce():
    nonce = new_receipt_nonce()
    command = "printf 'x'\n# note\n"
    assert unwrap_receipt_command(receipt_command(command, nonce)) == (command, nonce)
    assert unwrap_receipt_command(command) is None


def test_receipt_keeps_shell_arguments_functions_and_variables():
    """Guards FIX-RUNTIME-SH at 1f02fbc4c: keep the executing shell's state."""
    nonce = new_receipt_nonce()
    script = (
        'set -- "one two" three; value=local; f() { printf "%s" "$value/$1/$2"; }; '
    )
    raw = _run(script + receipt_command('f "$@"', nonce))
    read = read_receipt(raw, nonce)
    assert (read.stdout, read.return_code) == ("local/one two/three", 0)


def test_payload_syntax_error_is_a_failed_receipt_not_encoder_execution():
    """Guards FIX-RUNTIME-SH at 1f02fbc4c: malformed payload stays isolated."""
    nonce = new_receipt_nonce()
    raw = _run(receipt_command("printf 'unterminated", nonce))
    read = read_receipt(raw, nonce)
    assert read.return_code != 0
    assert read.stdout == ""
    assert "syntax" in (read.stderr or "").lower()


# --- every redactable spelling inside a receipt ----------------------------

USERS = ["worker", "tokenuser42", "a", "0a", "3", "deadbeef", "deadbeefcafe"]
HOSTS = ["build.example.test", "docker", "abc", "30"]


def _spellings(user: str, host: str) -> list[str]:
    return [
        f"{user}@x",
        f"-l {user}",
        f"-l{user}",
        f"'{user}'",
        f'"{user}"',
        f"x@{host}",
        f"//{host}",
        f"host {host}",
        f"[{host}]",
        f"{host}:22",
        f"ssh://{user}@{host}",
        f"{user}:secret@{host}",
        f"login {user}@{host}: Permission denied",
        user,
        host,
    ]


def _all_redactions(user: str, host: str):
    for show, command_output, transport_errors in itertools.product(
        (False, True), (False, True), (False, True)
    ):
        docker_host = remote.RemoteDockerHost(
            f"ssh://{user}@{host}", user=user, show_address=show
        )
        yield (
            lambda text, d=docker_host, c=command_output, t=transport_errors: d.redact(
                text, command_output=c, transport_errors=t
            )
        )


@pytest.mark.parametrize("user", USERS)
@pytest.mark.parametrize("host", HOSTS)
def test_redaction_never_changes_a_receipt_into_another_answer(user, host):
    """Every spelling, every redaction mode: the exact answer, or a failed read."""
    nonce = new_receipt_nonce()
    for spelling in _spellings(user, host):
        payload = f"/opt/{spelling}/plugins\n".encode()
        for status in (0, 1):
            line = encode_receipt(payload, status, nonce)
            for redact in _all_redactions(user, host):
                read = read_receipt(
                    ExecResult(stdout=redact(line), stderr="", return_code=0), nonce
                )
                if read.return_code == RECEIPT_FAILED and read.stdout == "":
                    continue
                assert (read.stdout.encode(), read.return_code) == (payload, status)


def test_command_output_redaction_leaves_receipts_exact():
    """The mode exec output gets: no spelling stops a read."""
    nonce = new_receipt_nonce()
    for user, host in itertools.product(USERS, HOSTS):
        docker_host = remote.RemoteDockerHost(f"ssh://{user}@{host}", user=user)
        for spelling in _spellings(user, host):
            payload = f"/opt/{spelling}\n".encode()
            line = encode_receipt(payload, 0, nonce)
            seen = docker_host.redact(line, command_output=True, transport_errors=False)
            assert (
                read_receipt(
                    ExecResult(stdout=seen, stderr="", return_code=0), nonce
                ).stdout
                == payload.decode()
            )


@pytest.mark.parametrize(
    "mangle",
    [
        lambda line: line.replace("BF-RECEIPT-", "BF-RECEIPT-x"),
        lambda line: line[:-6] + "\n",
        lambda line: line.rstrip("\n") + "00\n",
        lambda line: line.replace(" 2f", " 2a", 1),
        lambda line: line + line,
        lambda line: "",
        lambda line: line.upper(),
    ],
    ids=["mark", "truncated", "extended", "changed", "repeated", "missing", "case"],
)
def test_any_change_fails_closed(mangle):
    nonce = new_receipt_nonce()
    line = encode_receipt(b"/opt/plugins\n", 0, nonce)
    read = read_receipt(
        ExecResult(stdout=mangle(line), stderr="", return_code=0), nonce
    )
    assert (read.stdout, read.return_code) == ("", RECEIPT_FAILED)


def test_another_nonce_or_a_failed_exec_is_a_failed_read():
    line = encode_receipt(b"x", 0, new_receipt_nonce())
    other = read_receipt(ExecResult(stdout=line, return_code=0), new_receipt_nonce())
    assert (other.stdout, other.return_code) == ("", RECEIPT_FAILED)
    assert (
        read_receipt(
            ExecResult(stdout="", stderr="boom", return_code=137), new_receipt_nonce()
        ).return_code
        == 137
    )
    assert read_receipt(None, new_receipt_nonce()).return_code == RECEIPT_FAILED


def test_non_utf8_output_is_a_failed_read_not_a_replacement():
    nonce = new_receipt_nonce()
    line = encode_receipt(b"/opt/\xff/plugins", 0, nonce)
    read = read_receipt(ExecResult(stdout=line, return_code=0), nonce)
    assert (read.stdout, read.return_code) == ("", RECEIPT_FAILED)


async def test_checked_read_keeps_the_commands_failure_status():
    async def execute(command, **kwargs):
        unwrapped = unwrap_receipt_command(command)
        assert unwrapped is not None
        return ExecResult(
            stdout=encode_receipt(b"partial", 4, unwrapped[1]), return_code=0
        )

    env = type("Env", (), {"exec": staticmethod(execute)})()
    with pytest.raises(RuntimeError, match="rc=4"):
        await lockdown._checked_read(env, "true", "reading")
    assert (await receipt_exec(env, "true")).return_code == 4


async def test_raw_output_without_a_receipt_refuses_provenance():
    async def execute(command, **kwargs):
        return ExecResult(
            stdout="1 0 0:1 / / rw - overlay overlay rw\n# end\n\0", return_code=0
        )

    env = type("Env", (), {"exec": staticmethod(execute)})()
    with pytest.raises(RuntimeError, match="receipt"):
        await lockdown._blocked_before_freeze(env, ("/app",))


# --- by construction: every stdout lockdown reads came from a receipt -------

_RECEIPT_PRODUCERS = {
    "_read_exec",
    "_checked_read",
    "_cloud_metadata_exec",
    "_cloud_metadata_agent_exec",
    "_prepare_bash_env_inspector",
    # The metadata helper's runs carry the same receipts
    # (test_metadata_helper.py::test_the_helpers_answers_are_read_from_a_receipt).
    "metadata_rules_run",
}


def _stdout_reads_outside_receipts(source: str) -> list[str]:
    tree = ast.parse(source)
    bad = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        produced: dict[str, str] = {}
        for node in ast.walk(func):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Await):
                call = node.value.value
                if not isinstance(call, ast.Call):
                    continue
                callee = ast.unparse(call.func).rsplit(".", 1)[-1]
                for target in node.targets:
                    names = target.elts if isinstance(target, ast.Tuple) else [target]
                    for name in names:
                        if isinstance(name, ast.Name):
                            produced[name.id] = callee
            elif (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "diagnostic_result"
                and len(node.value.args) == 1
                and isinstance(node.value.args[0], ast.Name)
            ):
                # A display copy has the same receipt origin. The separate
                # raw-output source contract forbids parsing that copy.
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        produced[target.id] = produced.get(node.value.args[0].id, "")
        for node in ast.walk(func):
            holder = None
            if isinstance(node, ast.Attribute) and node.attr == "stdout":
                holder = node.value
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "stdout"
            ):
                holder = node.args[0]
            if holder is None:
                continue
            name = holder.id if isinstance(holder, ast.Name) else ast.unparse(holder)
            if produced.get(name) not in _RECEIPT_PRODUCERS:
                bad.append(f"{func.name}:{node.lineno} reads {name}.stdout")
    return bad


def test_every_lockdown_stdout_read_comes_from_a_receipt():
    source = Path(lockdown.__file__).read_text()
    assert _stdout_reads_outside_receipts(source) == []


def test_the_scan_sees_a_raw_read():
    source = (
        "async def f(env):\n"
        "    result = await env.exec('x')\n"
        "    return result.stdout\n"
    )
    assert _stdout_reads_outside_receipts(source) == ["f:3 reads result.stdout"]


def test_guard_markers_are_read_from_a_receipt():
    from benchflow.task import verifier_core

    source = Path(verifier_core.__file__).read_text()
    tree = ast.parse(source)
    (func,) = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_plugin_guard_markers"
    ]
    calls = {ast.unparse(n.func) for n in ast.walk(func) if isinstance(n, ast.Call)}
    assert "receipt_exec" in calls
    assert "self._sandbox.exec" not in calls


async def test_exec_keywords_pass_through_unchanged():
    """``env=`` is the exec's environment, not the sandbox argument."""
    seen = []

    async def execute(command, **kwargs):
        seen.append(kwargs)
        unwrapped = unwrap_receipt_command(command)
        assert unwrapped is not None
        return ExecResult(stdout=encode_receipt(b"", 0, unwrapped[1]), return_code=0)

    env = type("Env", (), {"exec": staticmethod(execute)})()
    await receipt_exec(env, "true", env={"PATH": "/bin"}, user="root")
    await lockdown._checked_read(env, "true", "reading", env={"A": "1"})
    assert seen == [{"env": {"PATH": "/bin"}, "user": "root"}, {"env": {"A": "1"}}]
