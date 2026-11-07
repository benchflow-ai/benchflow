"""A Python whose standard library the test user cannot write.

A venv keeps its base interpreter's standard library first on its path, and
the verifier's pytest plugin guard is installed into the first folder of that
path its installer can write (``lockdown._INSTALL_PYTEST_PLUGIN_GUARD_SCRIPT``;
as root in a sandbox, the standard library). A test that means a venv's own
site-packages must therefore build the venv on a Python whose standard
library it cannot write. The suite's own Python may not be one: CI installs
it with ``uv python install``, into the runner user's home, and a venv made
from it had the guard written into that shared standard library instead.
"""

from __future__ import annotations

import os
import subprocess
import sys

_CANDIDATES = (sys.executable, "/usr/bin/python3", "/usr/local/bin/python3")


def readonly_stdlib_python() -> str | None:
    """The first of this test's Python, /usr/bin/python3 and
    /usr/local/bin/python3 whose standard library this user cannot write,
    or None (as root every folder is writable)."""
    for candidate in dict.fromkeys(_CANDIDATES):
        if not os.path.isfile(candidate):
            continue
        try:
            probe = subprocess.run(
                [
                    candidate,
                    "-I",
                    "-S",
                    "-c",
                    "import sysconfig; print(sysconfig.get_path('stdlib'))",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        stdlib = probe.stdout.strip()
        if probe.returncode == 0 and stdlib and not os.access(stdlib, os.W_OK):
            return candidate
    return None


def python_version(python: str) -> str:
    """``pythonX.Y`` of *python*, the folder name its venv's lib uses."""
    return subprocess.run(
        [
            python,
            "-I",
            "-S",
            "-c",
            "import sys; print('python%d.%d' % sys.version_info[:2])",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.strip()
