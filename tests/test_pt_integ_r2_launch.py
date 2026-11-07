"""PT-integ round 2: agent launch details (review A).

- PT-INTEG-B-5 (Low, regression): under K1's barrier AgentCore started bash
  (PT's route wrapper) where integ started ``/bin/sh`` only.
- Review A on D913 call 7: the live ACP launch's decoder is the kit's
  ``base64`` once the kit's RECORD exists, never the image's.
"""

from __future__ import annotations

import base64
import re
import shlex
import subprocess
from pathlib import Path

import pytest

from benchflow._runtime_tree import KIT_BIN, KIT_RECORD, KIT_SH
from benchflow.sandbox import _base

ROUTES = {
    "task": _base.ExecRoute(pin=None, task_root=False, root=False),
    "plumbing": _base.SYSTEM_SHELL_ROUTE,
    "kit-plumbing": _base.ExecRoute(
        pin=f"{KIT_BIN}:{_base.SYSTEM_PATH}",
        task_root=False,
        system_shell=True,
        explicit_root=True,
        shell=KIT_SH,
        bash=f"{KIT_BIN}/bash",
        tools=KIT_BIN,
        kit=True,
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("route", sorted(ROUTES))
async def test_agentcore_agent_bootstrap_starts_no_bash(route):
    """PT-INTEG-B-5, fails on 403505e5: the routed payload inside K1's
    barrier started ``bash`` (the task route's BASH_ENV-hiding wrapper, or
    ``bash --noprofile --norc`` on a pinned route). Integ starts POSIX sh."""
    from benchflow.sandbox.agentcore import AgentCoreSandbox

    box = AgentCoreSandbox.__new__(AgentCoreSandbox)
    payloads: list[str] = []

    def invoke(payload, timeout):
        payloads.append(payload)
        return _base.ExecResult(stdout="", stderr="", return_code=0)

    box._invoke_command = invoke
    await box._dispatch_command(
        "true",
        timeout_sec=10,
        resolved_user="agent",
        route=ROUTES[route],
        agent_bootstrap=True,
    )
    (payload,) = payloads
    assert shlex.split(payload)[:4] == [
        "/usr/bin/setpriv",
        "--no-new-privs",
        "--",
        "/bin/sh",
    ]
    assert not re.search(r"(^|[/ '])bash\b", payload), payload


@pytest.mark.parametrize("record", [True, False], ids=["kit", "no-kit"])
def test_the_live_launch_decoder_is_the_kits_when_its_record_exists(tmp_path, record):
    """Fails on 403505e5 (``/usr/bin/base64`` always): with the kit's RECORD
    the launch decodes its env line with the kit's base64; without a kit,
    the system one by absolute path."""
    from benchflow.sandbox._root_launch import root_launch_script

    kit_bin = tmp_path / "kit-bin"
    kit_bin.mkdir()
    used = tmp_path / "kit-base64-used"
    (kit_bin / "base64").write_text(
        f'#!/bin/sh\n: > {shlex.quote(str(used))}\nexec /usr/bin/base64 "$@"\n'
    )
    (kit_bin / "base64").chmod(0o755)
    record_path = tmp_path / "RECORD"
    if record:
        record_path.write_text("kit\n")
    out = tmp_path / "out"
    script = root_launch_script(
        f'printf %s "$K1_VALUE" > {shlex.quote(str(out))}', env_stdin=True
    )
    script = script.replace(KIT_RECORD, str(record_path)).replace(KIT_BIN, str(kit_bin))
    line = base64.b64encode(b"export K1_VALUE='it works'\n").decode() + "\n"
    ran = subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-c", script],
        input=line,
        env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert ran.returncode == 0, ran.stderr
    assert Path(out).read_text() == "it works"
    assert used.exists() is record
