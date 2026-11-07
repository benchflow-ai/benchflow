"""Exec output the harness decides on, carried as a checked receipt.

What a sandbox command prints is not what reaches the caller byte for byte.
Remote Docker masks the ssh user and the host in every exec's output
(``RemoteDockerHost.redact``), so a mount path, a plugin name or a PATH entry
that happens to contain them reaches the caller rewritten (CONFIRM-lockdown-7:
``/opt/<user>@plugins`` came back as ``/opt/***@plugins``, and the provenance
check then blocked a path that does not exist and trusted the real one).
Docker puts the command's stderr into the same stream, and every backend
decodes bytes it cannot read as replacement characters. A check that parses
that text judges something the sandbox never printed.

:func:`receipt_command` runs the command, unchanged, in its own subshell and
prints its stdout and exit status as one line: a random mark, the POSIX
``cksum`` of the encoded text, and the text itself in lowercase hex, an
alphabet no redaction, merged stream or decoding step leaves as it is while
still matching the sum. :func:`read_receipt` gives back exactly what the
command printed and its status, or, when the line is missing, repeated,
outside that alphabet or does not match its sum, a failed result with no
output (status 125, or the exec's own non-zero status), so a reader treats it
as the failed read it is, never as an answer.

The encoder runs ``od``, ``tr`` and ``cksum`` in subshells on the executing
service's protected pre-agent route, or the system route before capture.
Only these immutable copies are accessible to non-root verifier users;
provenance files stay private. The command itself keeps the exec's PATH,
user and environment.
"""

from __future__ import annotations

import re
import secrets
import shlex
from typing import Any

from benchflow._runtime_tree import KIT_BIN
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox._exec_output import exec_raw

# Status of a read whose receipt is missing or does not check.
RECEIPT_FAILED = 125

_MARK = "BF-RECEIPT-"
# The protected kit's bin first (root's alone, or absent: nobody else may
# create it, C8), then the system folders.
_SYSTEM_PATH = f"{KIT_BIN}:/usr/bin:/bin"
_LINE = re.compile(r"BF-RECEIPT-([0-9a-f]{24}) ([0-9]+) ([0-9]+) ([0-9a-f]*)")


class ReceiptFailure(ExecResult):
    """A failed receipt read, distinct from a checked payload's exit status.

    Created by ``read_receipt`` after the public exec/transport adapters have
    copied their results. This provenance never has to survive those copies;
    neither stderr text nor a payload's status can manufacture it. The raw
    fields and refusal status keep their existing meaning.
    """


def _crc_table() -> tuple[int, ...]:
    table = []
    for i in range(256):
        c = i << 24
        for _ in range(8):
            c = ((c << 1) ^ 0x04C11DB7) if c & 0x80000000 else (c << 1)
        table.append(c & 0xFFFFFFFF)
    return tuple(table)


_CRC_TABLE = _crc_table()


def posix_cksum(data: bytes) -> int:
    """The CRC ``cksum`` prints for *data* (POSIX.1, GNU and BusyBox alike)."""
    crc = 0
    for byte in data:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ _CRC_TABLE[(crc >> 24) ^ byte]
    length = len(data)
    while length:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ _CRC_TABLE[(crc >> 24) ^ (length & 0xFF)]
        length >>= 8
    return ~crc & 0xFFFFFFFF


def new_receipt_nonce() -> str:
    return secrets.token_hex(12)


def receipt_command(command: str, nonce: str, *, tools_path: str = _SYSTEM_PATH) -> str:
    """The ``sh`` script that runs *command* and prints its receipt line.

    The command runs in a subshell, on lines of its own, so its ``exit`` or
    trailing comment cannot reach the encoder; its exit
    status follows its output after one newline. The receipt line starts on
    a new line, after anything the command left on stderr.

    Bash before 5.2 miscounts bare case-arm closing parentheses inside nested
    subshells in a command substitution. Bash 3.2 also loses completed child
    statuses in that context. Run the payload outside command substitution;
    only the encoder captures its input. The same shell keeps the environment,
    positional arguments and subshell isolation. No child shell is selected.
    """
    if not re.fullmatch(r"[0-9a-f]{24}", nonce):
        raise ValueError("receipt nonce must be 24 lowercase hex digits")
    tools = f"PATH={shlex.quote(tools_path)}; export PATH"
    return (
        "{ (\n"
        f"{command}\n"
        "); command printf '\\n%s' \"$?\"; } "
        f"| ({tools};\n"
        "bf_receipt=$(od -An -v -tx1 | tr -d ' \\n') "
        f"|| exit {RECEIPT_FAILED}\n"
        f"case $bf_receipt in ''|*[!0-9a-f]*) exit {RECEIPT_FAILED};; esac\n"
        "bf_receipt_sum=$(command printf '%s' \"$bf_receipt\" "
        f"| ({tools}; cksum)) || exit {RECEIPT_FAILED}\n"
        f"command printf '\\n{_MARK}%s %s %s\\n' {nonce} "
        '"$bf_receipt_sum" "$bf_receipt"\n)\n'
    )


def unwrap_receipt_command(command: str) -> tuple[str, str] | None:
    """``(command, nonce)`` when *command* is :func:`receipt_command`'s script."""
    found = re.search(
        rf"command printf '\\n{_MARK}%s %s %s\\n' ([0-9a-f]{{24}}) "
        r"\"\$bf_receipt_sum\" \"\$bf_receipt\"\n\)\n\Z",
        command,
    )
    if found is None:
        return None
    route = re.search(r"\| \(PATH=(.+); export PATH;\nbf_receipt=", command)
    if route is None:
        return None
    paths = shlex.split(route[1])
    if len(paths) != 1:
        return None
    head, _, tail = receipt_command("", found[1], tools_path=paths[0]).partition("\n\n")
    head, tail = head + "\n", "\n" + tail
    if not (command.startswith(head) and command.endswith(tail)):
        return None
    if len(command) < len(head) + len(tail):
        return None
    return command[len(head) : len(command) - len(tail)], found[1]


def encode_receipt(stdout: bytes, return_code: int, nonce: str) -> str:
    """The receipt line the script prints for this output and status."""
    hexed = (stdout + b"\n" + str(return_code).encode()).hex()
    body = hexed.encode()
    return f"\n{_MARK}{nonce} {posix_cksum(body)} {len(body)} {hexed}\n"


def _exit_status(result: Any) -> int | None:
    for name in ("return_code", "exit_code"):
        value = getattr(result, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _failed(result: Any, reason: str) -> ExecResult:
    code = _exit_status(result)
    if not code:
        code = RECEIPT_FAILED
    stderr = getattr(result, "stderr", None)
    stderr = stderr if isinstance(stderr, str) else ""
    stdout = getattr(result, "stdout", None)
    stdout = stdout if isinstance(stdout, str) else ""
    # What the exec printed besides the receipt is diagnostics, never data.
    detail = "\n".join(
        part for part in (stderr.strip(), _without_receipt(stdout).strip()) if part
    )
    message = f"exec output receipt {reason}"
    return ReceiptFailure(
        stdout="",
        stderr=f"{detail}\n{message}" if detail else message,
        return_code=code,
    )


def _without_receipt(stdout: str) -> str:
    return "\n".join(line for line in stdout.splitlines() if not line.startswith(_MARK))


def read_receipt(result: Any, nonce: str) -> ExecResult:
    """What the command printed and its status, checked, or a failed read.

    The command's own stderr and anything else the exec printed (Docker's
    merged stderr) are kept as ``stderr``, for messages only.
    """
    exec_status = _exit_status(result)
    if exec_status != 0:
        # The script exits 0 once it printed the receipt: anything else is
        # the exec failing (a timeout, a lost connection), whatever it printed.
        return _failed(result, f"not printed (exec status {exec_status})")
    stdout = getattr(result, "stdout", None)
    if not isinstance(stdout, str):
        return _failed(result, "missing")
    marked = [line for line in stdout.split("\n") if f"{_MARK}{nonce}" in line]
    if len(marked) != 1:
        return _failed(result, "missing" if not marked else "repeated")
    line = _LINE.fullmatch(marked[0].removesuffix("\r"))
    if line is None or line[1] != nonce:
        return _failed(result, "malformed")
    crc, length, hexed = int(line[2]), int(line[3]), line[4]
    body = hexed.encode()
    if len(body) != length or len(body) % 2 or posix_cksum(body) != crc:
        return _failed(result, "does not match its checksum")
    payload = bytes.fromhex(hexed)
    output, newline, status = payload.rpartition(b"\n")
    if not newline or not re.fullmatch(rb"[0-9]{1,3}", status) or int(status) > 255:
        return _failed(result, "has no exit status")
    try:
        text = output.decode("utf-8")
    except UnicodeDecodeError:
        return _failed(result, "is not UTF-8 text")
    stderr = getattr(result, "stderr", None)
    rest = _without_receipt(stdout).strip("\n")
    extra = "\n".join(
        part for part in ((stderr if isinstance(stderr, str) else ""), rest) if part
    )
    return ExecResult(stdout=text, stderr=extra or None, return_code=int(status))


async def receipt_exec(sandbox: Any, command: str, /, **kwargs: Any) -> ExecResult:
    """``sandbox.exec(command, **kwargs)``, its stdout and status carried as
    a receipt (see the module docstring). Positional-only, so an exec's own
    ``env=`` keyword passes through."""
    nonce = new_receipt_nonce()
    # Inspector copies belong to the service executing this receipt, never to
    # another container that happens to expose the same directory name.
    owner = getattr(sandbox, "_env", sandbox)
    service = kwargs.get("service", getattr(sandbox, "_service", "main"))
    saved = getattr(owner, "_benchflow_path_inspectors", None)
    folder = saved.get(service) if isinstance(saved, dict) else None
    tools_path = folder if isinstance(folder, str) and folder else _SYSTEM_PATH
    result = await exec_raw(
        sandbox.exec, receipt_command(command, nonce, tools_path=tools_path), **kwargs
    )
    return read_receipt(result, nonce)
