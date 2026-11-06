"""Explicit tools for the policy settings fixtures (FF-test-env at 7a2387ac)."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from benchflow.agents.registry import _BENCHFLOW_NODE_PREFIX


def run_policy_setup(command: str, home: Path) -> subprocess.CompletedProcess[str]:
    """Run the real setup with this interpreter and a closed tool PATH.

    Do not inherit a developer's Node, Python, HOME or startup variables. The
    policy assertions exercise Python fallback even on Node-equipped hosts.
    """
    with TemporaryDirectory(prefix="policy-tools-", dir=home) as directory:
        tools = Path(directory)
        for name in ("bash", "sh", "mkdir", "cat"):
            executable = shutil.which(name, path="/usr/bin:/bin")
            assert executable is not None, f"policy fixture needs {name}"
            (tools / name).symlink_to(executable)
        (tools / "python3").symlink_to(sys.executable)
        # The JS setup also probes a fixed private Node install before PATH.
        # Relocate that fixture path so a host install cannot select the JS arm.
        command = command.replace(
            _BENCHFLOW_NODE_PREFIX, shlex.quote(str(tools / "private-node"))
        )
        return subprocess.run(
            [str(tools / "bash"), "-c", command],
            env={
                "BENCHFLOW_AGENT_HOME": str(home),
                "HOME": str(home),
                "PATH": str(tools),
            },
            capture_output=True,
            text=True,
            timeout=15,
        )
