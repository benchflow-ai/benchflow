"""AgentCore's opt-in bfx route (rows A1, A2, A4, A5).

AgentCore runs every command through the runtime's first ``/bin/sh`` (the
image's), as root, and has no file API: so bfx, uploaded by root to its
0555 /.bfx before the agent (``delivery.deliver_upload``), holds against a
non-root agent only, whatever is delivered. A call is one command:
``/.bfx/bfx <cmd> --json '<arguments>'`` on root's system-shell route. Its
answer comes back through the platform's event stream: one over
:data:`~benchflow.sandbox.agentcore.BFX_ANSWER_MAX` bytes is bfx unavailable
here (BenchFlow's gap), never a partial parse. Until AgentCore passes a live
run, the route is on only with ``BENCHFLOW_BFX_PROVIDERS=on``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import shutil
import subprocess
import uuid
from types import SimpleNamespace

import pytest

from benchflow._bfx import delivery, pins
from benchflow._bfx.locate import SANDBOX_PATH, BfxUnavailable

pytestmark = pytest.mark.bfx_refusal


def agentcore_box(answer: str = "", code: int = 0, run=None):
    from benchflow.sandbox.agentcore import AgentCoreSandbox

    box = object.__new__(AgentCoreSandbox)
    box.logger = logging.getLogger("agentcore-test")
    box.bfx_delivery = "upload"
    seen = []

    async def exec_(command, user=None, timeout_sec=None, system_shell=False, **kw):
        seen.append(
            SimpleNamespace(
                command=command, user=user, system_shell=system_shell, kw=kw
            )
        )
        if run is not None:
            return await run(command)
        return SimpleNamespace(stdout=answer, stderr="", return_code=code)

    box.exec = exec_
    return box, seen


async def test_a1_bfx_call_runs_the_uploaded_bfx_as_root_on_the_system_shell():
    """A1. One command, as root on the system shell: the uploaded bfx by
    its path, the command and its JSON arguments; the answer parsed.
    Without a delivery nothing runs.

    Mutant: the user dropped (the session's default user runs it)."""
    body = {"ok": True, "result": {"entries": [{"path": "/etc/passwd"}]}}
    box, seen = agentcore_box(json.dumps(body) + "\n")
    args = {"paths": ["/etc/passwd"], "acl": False, "mounts": False}
    got = await box.bfx_call("scan", args, timeout_sec=9)
    assert got == body["result"]
    (call,) = seen
    assert call.user == "root" and call.system_shell is True
    words = shlex.split(call.command)
    assert words[:3] == [SANDBOX_PATH, "scan", "--json"]
    assert json.loads(words[3]) == args

    box.bfx_delivery = None
    with pytest.raises(BfxUnavailable):
        await box.bfx_call("scan", args, timeout_sec=9)
    assert len(seen) == 1


async def test_a2_an_answer_over_the_cap_is_unavailable_never_parsed(monkeypatch):
    """A2. The stream brought back more than the cap: bfx unavailable here
    (BenchFlow's gap), not an answer and not a protocol error (which would
    read as a check the agent defeated).

    Mutant: no cap."""
    from benchflow.sandbox import agentcore

    monkeypatch.setattr(agentcore, "BFX_ANSWER_MAX", 64)
    body = {"ok": True, "result": {"entries": [{"path": "/" + "x" * 200}]}}
    box, _ = agentcore_box(json.dumps(body) + "\n")
    with pytest.raises(BfxUnavailable, match="cap"):
        await box.bfx_call("scan", {"paths": ["/"]}, timeout_sec=9)
    assert agentcore.BFX_ANSWER_MAX == 64
    monkeypatch.undo()
    assert agentcore.BFX_ANSWER_MAX == 64 << 20


async def test_a4_agentcore_holds_against_a_non_root_agent_whatever_is_delivered():
    """A4. AgentCore starts bfx through the image's shell: its guarantee is
    non-root-agent for any delivery, and the upload records the arch the
    walker's record names.

    Mutant: the guarantee read from the delivery alone."""
    from benchflow.sandbox.agentcore import AgentCoreSandbox

    assert AgentCoreSandbox.bfx_launch == "image-shell"
    box = object.__new__(AgentCoreSandbox)
    for mode in ("upload", "volume", "bind"):
        box.bfx_delivery = mode
        assert box.bfx_guarantee == "non-root-agent", mode
    box.bfx_delivery = None
    assert box.bfx_guarantee is None


async def test_a4_the_upload_records_the_arch(monkeypatch):
    """A4. ``deliver_upload`` keeps the arch the delivered bfx reported."""

    async def deliver(upload, run_root):
        return "aarch64"

    monkeypatch.setattr(delivery, "deliver_by_upload", deliver)
    monkeypatch.setenv(delivery.PROVIDER_ENV, "on")
    box = SimpleNamespace(bfx_delivery=None, logger=logging.getLogger("t"))
    await delivery.deliver_upload(box, "agentcore", None, None)
    assert box.bfx_delivery == "upload" and box._bfx_arch == "aarch64"


def _real_bfx():
    from benchflow._bfx import locate

    try:
        return locate.verified_binary("x86_64")
    except BfxUnavailable as exc:
        pytest.skip(f"no pinned bfx in this checkout: {exc}")


@pytest.fixture
def shell_container():
    """A container whose root ``/bin/sh`` runs the route's command, as
    AgentCore's runtime does, with the pinned bfx at /.bfx/bfx (0555, root)."""
    if shutil.which("docker") is None:
        pytest.skip("no docker here")
    binary = _real_bfx()
    image = "python:3.12-alpine"
    if subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True
    ).returncode:
        pytest.skip(f"{image} is not on this machine")
    name = f"b2-a5-{uuid.uuid4().hex[:12]}"
    started = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            "--network",
            "none",
            "--mount",
            f"type=bind,src={binary},dst=/.bfx/bfx,readonly",
            image,
            "sleep",
            "600",
        ],
        capture_output=True,
        text=True,
    )
    if started.returncode:
        pytest.skip(f"docker run failed: {started.stderr[-300:]}")
    try:
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


async def test_a5_the_route_runs_the_pinned_bfx_through_a_real_shell(shell_container):
    """A5. The command A1 builds, run by a real image's root /bin/sh with the
    pinned bfx: bfx answers, its own report names the pin, and a planted link
    into the verifier's folder is found.

    Mutant: the arguments quoted for another shell (the answer is then a
    bfx refusal)."""

    async def run(command):
        process = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-u",
            "0",
            shell_container,
            "/bin/sh",
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await process.communicate()
        return SimpleNamespace(
            stdout=out.decode(), stderr=err.decode(), return_code=process.returncode
        )

    setup = await run(
        "mkdir -p /app && ln -s /logs/verifier/reward.txt /app/grade.log && echo ok"
    )
    assert setup.return_code == 0, setup.stderr
    box, _ = agentcore_box(run=run)
    version = await box.bfx_call("version", {}, timeout_sec=60)
    assert version.get("sha256") == pins.SHA256["x86_64"]
    walked = await box.bfx_call(
        "scan",
        {"walks": [{"root": "/app", "depth": 8}], "mounts": False, "acl": False},
        timeout_sec=60,
    )
    links = {
        e.get("path"): e.get("link")
        for e in walked.get("entries") or ()
        if e.get("type") == "link"
    }
    assert links == {"/app/grade.log": "/logs/verifier/reward.txt"}
