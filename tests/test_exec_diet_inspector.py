"""The snapshot's copy and manifest in the bash-env inspectors' exec (PY8 U12
on top of V7).

V7 prepares protected copies of ``stat``, ``readlink`` and ``python3`` in a
fresh root-only folder before the agent starts, in one root exec whose
prefix is the build-config snapshot's copy. U12 makes that prefix the merged
copy and manifest script, so the snapshot needs no exec of its own for the
manifest. The inspectors still run after the prefix whatever it answered,
with their own environment and budget. On integ/py-0.8 363d815b the first
test fails by its exec count (the manifest is a second exec).

Needs Docker.
"""

from __future__ import annotations

import json

import pytest

from benchflow.sandbox import lockdown
from tests.exec_diet_fakes import (
    PYTHON,
    BoxSandbox,
    as_root,
    box_factory,  # noqa: F401  (fixture)
)

pytestmark = pytest.mark.deterministic


def _manifest(present: set[str]) -> dict[str, bool]:
    return {f: f in present for f in lockdown._BUILD_CONFIG_FILES}


async def test_the_manifest_rides_in_the_inspector_exec(box_factory):  # noqa: F811
    box = box_factory(PYTHON)
    as_root(box, "mkdir -p /app && echo 'x' > /app/setup.py && echo y > /app/Makefile")
    sandbox = BoxSandbox(box)

    await lockdown._snapshot_build_config(sandbox, "/app")

    assert [(e.caller, e.user) for e in sandbox.execs] == [
        ("_prepare_bash_env_inspector", "root")
    ]
    (only,) = sandbox.execs
    assert only.timeout_sec == lockdown.VERIFIER_SETUP_TIMEOUT_SEC
    written = as_root(box, f"cat {lockdown._SNAPSHOT_MANIFEST}")
    assert json.loads(written) == _manifest({"setup.py", "Makefile"})
    assert as_root(box, f"cat {lockdown._SNAPSHOT_DIR}/setup.py") == "x\n"
    # The inspectors are ready, root's alone, after the merged steps.
    inspector = sandbox._benchflow_bash_env_inspector
    assert isinstance(inspector, str)
    assert inspector.startswith("/etc/.runtime-bash-env-")
    assert as_root(box, f"stat -c '%u %a' {inspector}").split() == ["0", "711"]
    assert set(as_root(box, f"ls {inspector}").split()) >= {
        "before",
        "before-links",
        "python3",
        "readlink",
        "stat",
        "sh",
        "ls",
        "id",
        "find",
        "mkdir",
        "ln",
        "cat",
        "od",
        "tr",
        "cksum",
    }


async def test_the_inspectors_run_when_the_copy_fails(box_factory):  # noqa: F811
    """A snapshot folder the copy cannot make (a file in its place) fails the
    copy and the manifest steps; the inspectors run after them all the same,
    and the host still writes what it can (as on integ, where the copy alone
    was the prefix)."""
    box = box_factory(PYTHON)
    as_root(box, "mkdir -p /app && echo x > /app/setup.py")
    as_root(box, f"echo not-a-folder > {lockdown._SNAPSHOT_DIR}")
    sandbox = BoxSandbox(box)

    await lockdown._snapshot_build_config(sandbox, "/app")

    inspector = sandbox._benchflow_bash_env_inspector
    assert isinstance(inspector, str)
    assert inspector.startswith("/etc/.runtime-bash-env-")
    assert as_root(box, f"stat -c '%u %a' {inspector}").split() == ["0", "711"]
    # The first exec is the inspectors' with the merged prefix; the host's
    # own manifest echo follows, because the script's write failed.
    assert sandbox.execs[0].caller == "_prepare_bash_env_inspector"
    assert any(
        e.caller == "_snapshot_build_config" and e.command.startswith("echo ")
        for e in sandbox.execs[1:]
    )
