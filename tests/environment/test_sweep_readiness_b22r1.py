"""Guards FF-sw-evaluation-C3 against d97bf9f6 premature probe timeouts."""

import asyncio
import contextlib
import time

import pytest

from benchflow.environment.manifest import EnvironmentManifest, Readiness, ServiceSpec
from benchflow.environment.manifest_env import ManifestEnvironment
from tests.environment.test_sweep_readiness import ShellSandbox


@pytest.mark.parametrize("mode", ["entrypoint", "explicit", "derived"])
async def test_readiness_accepts_response_within_declared_deadline(mode):
    handlers = set()

    async def handle(reader, writer):
        handlers.add(asyncio.current_task())
        try:
            await reader.readuntil(b"\r\n\r\n")
            await asyncio.sleep(1.5)
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()
            handlers.discard(asyncio.current_task())

    for port in range(8900, 9000):
        try:
            server = await asyncio.start_server(handle, "127.0.0.1", port)
            break
        except OSError:
            continue
    else:
        pytest.fail("no free test port")
    services = (
        []
        if mode == "entrypoint"
        else [ServiceSpec(name="svc", command="svc", port=port)]
    )
    env = ManifestEnvironment(
        EnvironmentManifest(
            name="fixture",
            image="fixture",
            owns_lifecycle=mode == "entrypoint",
            services=services,
            readiness=Readiness(
                http=[] if mode == "derived" else [f"http://127.0.0.1:{port}/ready"],
                timeout_sec=4,
            ),
        ),
        sandbox=ShellSandbox(),
    )
    env._started = services
    try:
        async with server:
            started = time.monotonic()
            result = await env.readiness()
            assert result.ready, result.error
            assert time.monotonic() - started < 4
    finally:
        if handlers:
            await asyncio.gather(*handlers)


@pytest.mark.parametrize("tcp", [False, True])
async def test_probe_retries_share_the_remaining_deadline(monkeypatch, tcp):
    import shlex

    from benchflow.environment import manifest_env
    from benchflow.sandbox.protocol import ExecResult

    now = [0.0]
    calls = []
    monkeypatch.setattr(manifest_env.time, "monotonic", lambda: now[0])

    async def sleep(delay):
        now[0] += delay

    monkeypatch.setattr(manifest_env.asyncio, "sleep", sleep)

    class Sandbox:
        async def exec(self, command, *, timeout_sec):
            argv = shlex.split(command)
            bound = float(argv[1] if tcp else argv[argv.index("--max-time") + 1])
            calls.append(bound)
            assert timeout_sec == bound + 1
            now[0] += 1.5
            return ExecResult(return_code=1, stdout="", stderr="")

    env = ManifestEnvironment(
        EnvironmentManifest(
            name="fixture",
            image="fixture",
            readiness=Readiness(
                http=[] if tcp else ["http://localhost:8900/ready"],
                tcp=[8900] if tcp else [],
                timeout_sec=5,
            ),
        ),
        sandbox=Sandbox(),
    )
    assert not (await env.readiness()).ready
    assert calls == [5, 2.5]
    assert now[0] == 5
