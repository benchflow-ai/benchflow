"""Provision the stdlib runtime used for trusted workspace capture."""

from __future__ import annotations

import shlex

from benchflow.review.evidence import EvidenceError
from benchflow.sandbox.protocol import Sandbox

_PYTHON_PROBE = "import sys; raise SystemExit(sys.version_info < (3, 10))"
_SETUP_SCRIPT = f"""
set -eu
if python3 -I -c {shlex.quote(_PYTHON_PROBE)} >/dev/null 2>&1; then
    exit 0
fi
if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends python3
elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache python3
elif command -v dnf >/dev/null 2>&1; then
    dnf install -y python3
elif command -v microdnf >/dev/null 2>&1; then
    microdnf install -y python3
elif command -v yum >/dev/null 2>&1; then
    yum install -y python3
else
    echo 'Automatic review requires Python 3.10+ in the task image; no supported package manager is available.' >&2
    exit 1
fi
python3 -I -c {shlex.quote(_PYTHON_PROBE)}
"""


async def ensure_evidence_python(
    env: Sandbox, *, timeout_sec: int = 600, allow_install: bool = True
) -> None:
    """Prepare capture before the solver runs, including shell-only images.

    Existing compatible interpreters require no package installation. Failure
    is a setup error, before a solver can spend its budget producing evidence
    that the controller cannot preserve.
    """
    command = (
        shlex.join(["sh", "-c", _SETUP_SCRIPT])
        if allow_install
        else shlex.join(["python3", "-I", "-c", _PYTHON_PROBE])
    )
    result = await env.exec(
        command,
        user="root",
        timeout_sec=timeout_sec,
    )
    if result.return_code != 0:
        details = (result.stderr or result.stdout or "")[-2000:]
        raise EvidenceError(f"Could not prepare workspace capture Python: {details}")


_CAPTURE_TOOLS_PROBE = (
    f"python3 -I -c {shlex.quote(_PYTHON_PROBE)} >/dev/null 2>&1 "
    "|| command -v tar >/dev/null 2>&1"
)


async def ensure_capture_tools(env: Sandbox, *, timeout_sec: int = 60) -> None:
    """Require python3 (3.10+) or tar for capture, without installing either."""
    result = await env.exec(
        shlex.join(["sh", "-c", _CAPTURE_TOOLS_PROBE]),
        user="root",
        timeout_sec=timeout_sec,
    )
    if result.return_code != 0:
        raise EvidenceError(
            "Workspace capture needs python3 (3.10+) or tar in the task image; "
            "it has neither"
        )
