"""The command that removes Daytona's session files, exactly as it is sent.

The scrub runs as root right before a snapshot and at stop. Its programs
(``find``, ``rm``) are found on the PATH it pins, so a scrub built without
root's tools, or one that leaves the image's PATH, would run what the agent
put first on that PATH. It goes by the toolbox's own exec: a session command
would keep its text in the folder the scrub empties. No key, no network: the
provider is a fake that answers the toolbox exec and fails a session at once.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
from types import SimpleNamespace

import pytest

pytest.importorskip("daytona")

from benchflow._runtime_tree import KIT_BIN, KIT_SH, RuntimeKit
from benchflow.sandbox._base import (
    SYSTEM_PATH,
    SYSTEM_SHELL,
    ExecRoute,
    system_shell_prefix,
)
from benchflow.sandbox.daytona import DaytonaSandbox, _load_daytona_sdk
from benchflow.sandbox.daytona_dind import VM_ROUTE

_load_daytona_sdk()

KIT = RuntimeKit(trusted=("/usr/bin", "/bin"), python=True, bash=True)
#: Root's system-shell route once the sandbox holds its protected kit: its
#: shell and PATH are the kit's, not the image's or the system's.
KIT_ROUTE = ExecRoute(
    pin=KIT.path,
    task_root=False,
    system_shell=True,
    explicit_root=True,
    shell=KIT_SH,
    bash=f"{KIT_BIN}/bash",
    tools=KIT.path,
    kit=True,
)


def scrub_command(shell: str, path: str, *, refuse_missing: bool) -> str:
    """The scrub as it must be sent: PATH pinned to *path* before *shell*
    starts and again inside it, and only the attach env folders and the
    entries of Daytona's sessions folder removed."""
    missing = (
        "{ echo 'no /root/.daytona/sessions on this box' >&2; exit 3; }"
        if refuse_missing
        else "exit 0"
    )
    body = (
        f"{system_shell_prefix(path)}; "
        'for d in /opt/runtime/run/attach-*; do [ ! -e "$d" ] || rm -rf -- "$d"; done; '
        f"[ -d /root/.daytona/sessions ] || {missing}; "
        "find /root/.daytona/sessions -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +"
    )
    return f"{system_shell_prefix(path)}; {shell} -c {shlex.quote(body)}"


class Provider:
    """A Daytona sandbox's ``process`` and ``fs``: the toolbox exec answers
    0; a session, an upload or a delete fails at once."""

    def __init__(self) -> None:
        self.process = self
        self.toolbox: list[tuple[str, object, object, object]] = []
        self.sessions: list[str] = []
        self.fs = SimpleNamespace(upload_file=self._refuse, delete_file=self._refuse)

    async def exec(self, command, cwd=None, env=None, timeout=None):
        self.toolbox.append((command, cwd, env, timeout))
        return SimpleNamespace(exit_code=0, result="")

    async def create_session(self, session_id):
        self.sessions.append(session_id)
        raise AssertionError("the scrub opened a session")

    async def execute_session_command(self, session_id, request, timeout=None):
        self.sessions.append(request.command)
        raise AssertionError("the scrub sent a session command")

    async def _refuse(self, *args, **kwargs):
        raise AssertionError("the scrub moved a file")


def _sandbox(execs: int) -> tuple[DaytonaSandbox, Provider, list]:
    provider = Provider()
    asked: list = []

    async def exec_route(user, *, system_shell=False, plumbing=False, service="main"):
        asked.append((user, system_shell, service))
        return KIT_ROUTE

    env = DaytonaSandbox.__new__(DaytonaSandbox)
    env.logger = logging.getLogger(__name__)
    env._sandbox = provider
    env.exec_route = exec_route  # type: ignore[method-assign]
    env._exec_counts().exec_calls = execs
    return env, provider, asked


@pytest.mark.parametrize(
    ("case", "execs", "shell", "path", "refuse_missing", "routes"),
    [
        # Before a snapshot, on the task box: root's kit route, required.
        ("snapshot", 3, KIT_SH, KIT.path, True, {("root", True, "main")}),
        ("snapshot", 0, KIT_SH, KIT.path, False, {("root", True, "main")}),
        # At stop (best effort), on the task box.
        ("stop", 3, KIT_SH, KIT.path, False, {("root", True, "main")}),
        # At a DinD stop, on the VM: the VM's system tools, not the kit of the
        # task box (no route asked).
        ("dind stop", 3, SYSTEM_SHELL, SYSTEM_PATH, False, set()),
    ],
    ids=["snapshot-sessions-sent", "snapshot-none-sent", "stop", "dind-stop"],
)
async def test_the_session_scrub_is_one_toolbox_exec_on_roots_pinned_tools(
    case, execs, shell, path, refuse_missing, routes
):
    env, provider, asked = _sandbox(execs)
    if case == "snapshot":
        run = env._scrub_sessions_before_snapshot()
    elif case == "stop":
        run = env._remove_exec_traces()
    else:
        run = env._remove_exec_traces(VM_ROUTE)
    await asyncio.wait_for(run, 10)
    assert provider.sessions == []
    assert provider.toolbox == [
        (scrub_command(shell, path, refuse_missing=refuse_missing), None, None, 60)
    ]
    # Root's system-shell route of the task box, asked for nothing else.
    assert set(asked) == routes
