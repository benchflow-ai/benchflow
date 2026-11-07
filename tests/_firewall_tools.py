"""Explicit trusted-image tool resolver for legacy scan/launch fixtures.

Those tests install a minimal image's utilities on a temporary PATH. Their
failure injection tests scan mechanics, not tool custody; production custody
is exercised without this fixture in test_su_integration_audit.py.
"""

import pytest


@pytest.fixture
def audit_image_tools(monkeypatch):
    from benchflow.sandbox import firewall_power

    original_exec = firewall_power._exec

    async def exec_fixture(env, command, timeout_sec, **kwargs):
        # Legacy doubles predate the explicit backend capability. A double
        # that explicitly refuses it still exercises the production refusal.
        if not hasattr(env, "supports_system_shell"):
            env.supports_system_shell = True
        return await original_exec(env, command, timeout_sec, **kwargs)

    monkeypatch.setattr(firewall_power, "_exec", exec_fixture)
    monkeypatch.setattr(
        firewall_power,
        "_audit_prelude",
        lambda: (
            "fwp_inventory_path=${PATH-}; "
            'tool() { command -v "$1"; }; '
            'walk() { W=$1; [ -e "$W" ]; }; '
            'run() { _fwp_tool=$(tool "$1") || return 127; shift; '
            '"$_fwp_tool" "$@"; }; '
            'have() { tool "$1" >/dev/null 2>&1; }; '
        ),
    )
