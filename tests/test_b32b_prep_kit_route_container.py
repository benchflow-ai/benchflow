"""Root's preparation steps keep PT's kit route, in a real container (merge
wave batch 32b, review A M-1).

V3's ``_prep_on_path`` launched every verifier-preparation step through
``_on_path``, which re-pinned PATH to the image's system folders inside PT's
kit route: the conftest and ``.py`` clean-up, the chown, the freeze, the
kills, the reclaim and the restore took ``find``, ``stat`` and the rest from
``/usr/bin`` instead of the kit's bin (C8). With the kit, a root step keeps
the route's PATH, the kit's bin first; without one, V3's system route
still pins the system folders. Skipped without a Docker daemon.
"""

from __future__ import annotations

import pytest

from benchflow._runtime_tree import KIT_BIN
from benchflow.sandbox import lockdown
from benchflow.sandbox._base import SYSTEM_PATH
from tests.test_c8_protected_runtime_container import (  # noqa: F401
    _sandbox,
    _setup,
    box,
)


@pytest.mark.asyncio
async def test_a_preparation_step_takes_its_tools_from_the_kit(box):  # noqa: F811
    sandbox = await _setup(box)
    probe = 'printf "%s\\n" "$PATH"; command -v find; command -v stat'
    for env in (lockdown._prep_env(sandbox), lockdown._route_env(sandbox)):
        result = await env.exec(probe, user="root", timeout_sec=60)
        assert result.return_code == 0, result.stderr
        path, find, stat = result.stdout.split()
        assert path.split(":")[0] == KIT_BIN, path
        assert (find, stat) == (f"{KIT_BIN}/find", f"{KIT_BIN}/stat")


@pytest.mark.asyncio
async def test_a_root_probe_keeps_the_kit_route(box):  # noqa: F811
    sandbox = await _setup(box)
    result = await sandbox.exec(
        lockdown._on_path('printf "%s" "$PATH"'), user="root", timeout_sec=60
    )
    assert result.return_code == 0, result.stderr
    assert result.stdout.split(":")[0] == KIT_BIN


@pytest.mark.asyncio
async def test_without_a_kit_the_system_route_still_pins_the_system_folders(box):  # noqa: F811
    sandbox = _sandbox(box)  # no setup: no kit yet
    result = await sandbox.exec(
        lockdown._on_path('printf "%s" "$PATH"'), user="root", timeout_sec=60
    )
    assert result.return_code == 0, result.stderr
    assert result.stdout == SYSTEM_PATH
