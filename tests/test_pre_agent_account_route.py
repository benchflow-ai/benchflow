"""Root route and account discovery from facts taken while pristine
(TRUSTED-RUNTIME v2 §4): where the account table read before the agent
started names the account, exec_route's uid lookup runs nothing in the
sandbox; where it does not, the lookup is today's."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from benchflow.sandbox._base import BaseSandbox
from benchflow.sandbox._identities import Identities
from benchflow.sandbox.protocol import ExecResult

PASSWD = "root:x:0:0::/root:/bin/sh\nagent:x:1000:1000::/home/agent:/bin/sh\n"
GROUP = "root:x:0:\nagent:x:1000:\n"


class Box(BaseSandbox):
    supports_system_shell = True

    def __init__(self) -> None:  # no task config: only the lookup is exercised
        self.logger = logging.getLogger("test")
        self.commands: list[str] = []

    async def exec(self, command: str, **kwargs: Any) -> ExecResult:
        self.commands.append(command)
        return ExecResult(return_code=0, stdout="4242\n", stderr="")


# Only the lookup is exercised: the backend's other methods are never called.
Box.__abstractmethods__ = frozenset()


@pytest.fixture
def box() -> Box:
    return Box()


async def test_a_name_in_the_pre_agent_table_runs_nothing(box):
    box._benchflow_identities = Identities(PASSWD, GROUP)
    assert await box._named_user_uid("agent") == 1000
    assert await box._named_user_uid("root") == 0
    assert box.commands == []


async def test_a_service_table_answers_for_its_service(box):
    box._benchflow_service_identity_tables = {
        "db": Identities("postgres:x:999:999::/:/bin/sh\n", "postgres:x:999:\n")
    }
    assert await box._named_user_uid("postgres", service="db") == 999
    assert box.commands == []


async def test_a_name_the_table_lacks_is_looked_up_as_before(box):
    box._benchflow_identities = Identities(PASSWD, GROUP)
    assert await box._named_user_uid("late") == 4242
    assert box.commands == ["id -u late"]


async def test_without_a_table_the_lookup_is_today_s(box):
    assert await box._named_user_uid("agent") == 4242
    assert box.commands == ["id -u agent"]
