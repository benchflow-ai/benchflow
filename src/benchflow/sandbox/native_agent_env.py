"""Bounded native agent environment transport (K1-R17-1 at 02648360).

The decoder is fixed, isolated, and runs only after account selection and the
native no_new_privs barrier. It never evaluates names or values as shell text.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from benchflow.sandbox._base import ExecResult

# -I -S excludes cwd, caller Python settings, user packages and site hooks.
# Use only the fixed system locations, never the agent's PATH or a task venv.
_RESTORE = """import errno, json, os, sys
mode, shell, command = sys.argv[1:]
environment = dict(os.environ)
if mode == 'stdin':
    # One line, a byte at a time: what follows it on stdin (a staged
    # command) stays for the shell this execs.
    data = bytearray()
    while True:
        byte = os.read(0, 1)
        if not byte:
            sys.exit(97)
        if byte == b'\\n':
            break
        data += byte
else:
    with os.fdopen(9, 'rb') as stream:
        data = stream.read()
environment.update(json.loads(data))
try:
    os.execve(shell, [shell, '-c', command], environment)
except OSError as error:
    if error.errno != errno.E2BIG:
        raise
    size = sum(len(os.fsencode(k)) + len(os.fsencode(v)) + 2
               for k, v in environment.items())
    print('Native agent environment is %d bytes; exec limit is %d bytes '
          '(including arguments and pointers, with a separate per-entry limit). '
          'Use a smaller agent environment or move large values to uploaded files.'
          % (size, os.sysconf('SC_ARG_MAX')), file=sys.stderr)
    sys.exit(97)
"""


def restore_command(command: str, *, stdin: bool = False) -> str:
    """Return a command whose length is independent of the environment.

    *stdin*: the environment is the first line of stdin (JSON, which has no
    raw newline), read without buffering, so the rest of stdin (a staged
    agent command, the transport contract) is left to the command; else it
    is descriptor 9."""
    # Only a real agent command can bypass its older export/shell bootstrap.
    # Providers enclose this decoder in native_agent_argv and select the user.
    from benchflow.sandbox.agent_exec import AgentExecCommand

    if not isinstance(command, AgentExecCommand):
        raise TypeError("Native environment restoration requires an agent command")
    arguments = shlex.join(
        [
            "-I",
            "-S",
            "-c",
            _RESTORE,
            "stdin" if stdin else "fd",
            command.agent_shell,
            command.agent_command,
        ]
    )
    return (
        f"if [ -x /usr/bin/python3 ]; then exec /usr/bin/python3 {arguments}; "
        f"elif [ -x /usr/local/bin/python3 ]; then exec /usr/local/bin/python3 {arguments}; "
        "else echo 'Native agent environment requires Python 3 at "
        "/usr/bin/python3 or /usr/local/bin/python3' >&2; exit 97; fi"
    )


@asynccontextmanager
async def uploaded_agent_env(
    command: str,
    env: dict[str, str],
    *,
    root_exec: Callable[[str], Awaitable[ExecResult]],
    upload: Callable[[Path, str], Awaitable[None]],
) -> AsyncIterator[tuple[str, str]]:
    """Upload data into a fresh root-only directory and pass an open descriptor.

    The root dispatch opens the file and removes its name before selecting the
    account. Only the selected account reads the inherited descriptor. No task
    value enters any process argv, including setup, cleanup or the decoder.
    """
    folder = "/.benchflow-agent-env-" + uuid4().hex
    path = folder + "/payload"
    made = await root_exec(f"umask 077; /bin/mkdir -- {folder}")
    if made.return_code != 0:
        raise RuntimeError("Could not create private native agent environment folder")
    cleanup = f"/bin/rm -f -- {path}; /bin/rmdir -- {folder}"
    try:
        with tempfile.TemporaryDirectory() as local:
            payload = Path(local) / "payload"
            payload.write_text(json.dumps(env, ensure_ascii=True), encoding="ascii")
            await upload(payload, path)
        # Open on the root side of su, consume on the selected account's side.
        opening = f"exec 9<{path} || exit 97; {cleanup} || exit 97; "
        yield restore_command(command), opening
    finally:
        # The dispatch removes the file before running agent code. This also
        # covers upload/dispatch errors and cancellation before it can start.
        await asyncio.shield(root_exec(f"{cleanup} 2>/dev/null; :"))
