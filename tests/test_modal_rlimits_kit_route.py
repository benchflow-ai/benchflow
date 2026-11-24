"""BENCHFLOW_MODAL_RLIMITS on the protected runtime kit (C8).

The held prefix runs the image's /usr/bin/prlimit, timeout and setpriv as root
ahead of a command. Once the sandbox holds its kit, root runs only the kit's
programs and those in system folders the kit's preparation proved root's alone
(``RuntimeKit.trusted``, the route's ``tools``), so the prefix is allowed on a
kit route only where /usr/bin is one of those folders.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tenacity")

from benchflow._runtime_tree import KIT_BIN, RuntimeKit
from benchflow.sandbox._base import ExecRoute
from benchflow.sandbox.modal_impl import (
    DOCKER_RLIMITS,
    HELD_TOOLS_DIR,
    PRLIMIT,
    ModalRlimitsRefused,
    ModalSandbox,
)


def _held_sandbox() -> ModalSandbox:
    sandbox = object.__new__(ModalSandbox)
    sandbox._rlimits = dict(DOCKER_RLIMITS)
    return sandbox


def _kit_route(*trusted: str) -> ExecRoute:
    kit = RuntimeKit(trusted=trusted, python=True, bash=True)
    return ExecRoute(pin=kit.path, task_root=False, tools=kit.path, kit=True)


def test_a_kit_route_with_usr_bin_proven_is_held():
    argv = _held_sandbox().hold_argv(
        ["/bin/true"], _kit_route("/usr/local/bin", HELD_TOOLS_DIR, "/bin")
    )
    assert argv[0] == PRLIMIT and argv[-1] == "/bin/true"


@pytest.mark.parametrize(
    "trusted", [(), ("/usr/local/bin", "/bin"), (f"{KIT_BIN}-view/usr/bin",)]
)
def test_a_kit_route_without_usr_bin_proven_is_refused(trusted):
    with pytest.raises(ModalRlimitsRefused, match="not proven root's alone"):
        _held_sandbox().hold_argv(["/bin/true"], _kit_route(*trusted))


def test_a_route_off_the_kit_is_held_as_before():
    """Before the kit exists (start-up, before the agent's turn)."""
    route = ExecRoute(pin="/usr/bin:/bin", task_root=False)
    assert _held_sandbox().hold_argv(["/bin/true"], route)[0] == PRLIMIT
    assert _held_sandbox().hold_argv(["/bin/true"])[0] == PRLIMIT


def test_no_setting_changes_nothing_on_any_route():
    sandbox = object.__new__(ModalSandbox)
    sandbox._rlimits = None
    assert sandbox.hold_argv(["/bin/true"], _kit_route()) == ["/bin/true"]
