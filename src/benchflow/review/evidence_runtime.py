"""Provision the stdlib runtime used for trusted workspace capture."""

from __future__ import annotations

import shlex

from benchflow.review.evidence import (
    _PYTHON_PROBE,
    EvidenceError,
    root_python_command,
)
from benchflow.sandbox._exec_output import diagnostic_result, exec_raw
from benchflow.sandbox.protocol import Sandbox


async def _runtime_kit(env: object) -> object | None:
    lookup = getattr(env, "runtime_kit_for_route", None)
    return await lookup() if callable(lookup) else None


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
    kit = await _runtime_kit(env)
    if kit is not None:
        # Root's helpers run on the protected runtime's Python (C8): an
        # interpreter installed now would not be theirs.
        command = await root_python_command(env, ["-c", _PYTHON_PROBE])
    elif allow_install:
        command = shlex.join(["sh", "-c", _SETUP_SCRIPT])
    else:
        command = shlex.join(["python3", "-I", "-c", _PYTHON_PROBE])
    result = await exec_raw(
        env.exec,
        command,
        user="root",
        timeout_sec=timeout_sec,
    )
    if result.return_code != 0:
        details = (
            diagnostic_result(result).stderr.render()
            or diagnostic_result(result).stdout.render()
            or ""
        )[-2000:]
        raise EvidenceError(f"Could not prepare workspace capture Python: {details}")


_CAPTURE_TOOLS_PROBE = (
    f"python3 -I -c {shlex.quote(_PYTHON_PROBE)} >/dev/null 2>&1 "
    "|| command -v tar >/dev/null 2>&1"
)


async def ensure_capture_tools(env: Sandbox, *, timeout_sec: int = 60) -> None:
    """Require python3 (3.10+) or tar for capture, without installing either."""
    kit = await _runtime_kit(env)
    probe = (
        f"( {await root_python_command(env, ['-c', _PYTHON_PROBE])} ) "
        ">/dev/null 2>&1 || command -v tar >/dev/null 2>&1"
        if kit is not None
        else _CAPTURE_TOOLS_PROBE
    )
    result = await exec_raw(
        env.exec,
        shlex.join(["sh", "-c", probe]),
        user="root",
        timeout_sec=timeout_sec,
    )
    if result.return_code != 0:
        raise EvidenceError(
            "Workspace capture needs python3 (3.10+) or tar in the task image; "
            "it has neither"
        )
