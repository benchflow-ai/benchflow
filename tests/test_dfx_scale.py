"""Guards batch 49 DFR-p3-2/3/6/7/9 against dc509087a."""

import asyncio
import gzip
import json
import os
import shlex
import subprocess
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from benchflow.acp.session import ACPSession
from benchflow.acp.watchdog import IdleWatchdog
from benchflow.providers import litellm_request_guard as guard
from benchflow.providers.litellm_runtime import LiteLLMProcess
from benchflow.sandbox.daytona_layers import InstallLayer, layer_run_line


def test_large_layer_preserves_installer_bytes(tmp_path, monkeypatch):
    from benchflow.sandbox import daytona_layers

    monkeypatch.setattr(daytona_layers, "_LAYER_SCRIPT", str(tmp_path / "install.sh"))
    script = "# unicode café ' $ \\\n" * 20000 + "printf installed"
    line = layer_run_line(InstallLayer("agent", "test", script), "key")
    assert max(len(part.encode()) for part in line.splitlines()) < 65535
    assert line.index("chmod 700 /opt/benchflow/daytona-proxy") > line.index(
        "chmod -R a+rX /opt/benchflow"
    )
    # Run the Dockerfile's actual decoder, before it invokes the installer.
    commands = line.splitlines()
    for command in commands[:-1]:
        subprocess.run(["sh", "-c", command.removeprefix("RUN ")], check=True)
    command = commands[-1].removeprefix("RUN ").split(" > ", 1)[0]
    decoded = subprocess.check_output(["sh", "-c", command])
    assert decoded.split(b"\n", 1)[1] == (script + "\n").encode()
    output = tmp_path / "installed"
    subprocess.run(["bash"], input=decoded, stdout=output.open("wb"), check=True)
    assert output.read_bytes() == subprocess.check_output(
        ["bash"], input=script.encode()
    )


def test_later_agent_layer_keeps_proxy_private(tmp_path):
    root = tmp_path / "daytona-proxy"
    root.mkdir(mode=0o700)
    line = layer_run_line(InstallLayer("agent", "test", "true"), "key")
    command = "chmod -R" + line.split("chmod -R", 1)[1]
    command = command.replace("/opt/benchflow", shlex.quote(str(tmp_path)))
    subprocess.run(["sh", "-c", command], check=True)
    assert root.stat().st_mode & 0o777 == 0o700


@pytest.mark.skipif(
    os.environ.get("BENCHFLOW_DFX_DOCKER") != "1", reason="explicit VM Docker check"
)
def test_large_layer_builds_and_matches_fresh_install_in_docker(tmp_path):
    """DFR-p3-3: a real builder and two containers produce identical files."""
    payload = ("installed café ' $ \\\n" * 15000).encode()
    script = (
        "mkdir -p /opt/benchflow/daytona-proxy\n"
        "chmod 700 /opt/benchflow/daytona-proxy\nprintf '%s' "
        + shlex.quote(payload.decode())
        + " > /opt/benchflow/payload\n"
    )
    layer = InstallLayer("agent", "dfx", script)
    tag = "dfx-scale-layer-" + uuid.uuid4().hex[:12]
    base = "python:3.13-slim"
    recipe = f"FROM {base}\n" + layer_run_line(layer, "test") + "\n"
    (tmp_path / "Dockerfile").write_text(recipe)
    try:
        subprocess.run(
            ["docker", "build", "--label", "owner=dfx-scale", "-t", tag, str(tmp_path)],
            check=True,
            capture_output=True,
            timeout=180,
        )
        cached = subprocess.check_output(
            [
                "docker",
                "run",
                "--rm",
                "--name",
                tag + "-cached",
                tag,
                "cat",
                "/opt/benchflow/payload",
            ],
            timeout=30,
        )
        fresh = subprocess.check_output(
            ["docker", "run", "--rm", "-i", "--name", tag + "-fresh", base, "bash"],
            input=(script + "cat /opt/benchflow/payload").encode(),
            timeout=30,
        )
        assert cached == fresh == payload
        assert (
            subprocess.check_output(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--name",
                    tag + "-mode",
                    tag,
                    "stat",
                    "-c",
                    "%a",
                    "/opt/benchflow/daytona-proxy",
                ],
                timeout=30,
            ).strip()
            == b"700"
        )
    finally:
        subprocess.run(["docker", "image", "rm", tag], capture_output=True)


def _kept_task(tmp_path, *, verifier_user=None, dockerfile_user=None, prebuilt=True):
    """A task that declares kept services, with the verifier's and the
    image's accounts as given (the TB2 fixture otherwise). ``prebuilt=False``
    drops its ``docker_image``, so its own Dockerfile is built."""
    from tests.test_keep_services_tb2 import _copy

    task = _copy(tmp_path)
    toml = task / "task.toml"
    text = toml.read_text().replace(
        "[verifier]\n",
        "[verifier]\n"
        + (
            f"user = {json.dumps(verifier_user)}\n" if verifier_user is not None else ""
        ),
    )
    if not prebuilt:
        text = "\n".join(
            line for line in text.splitlines() if not line.startswith("docker_image")
        )
    if "[metadata]" in text:
        text = text.replace(
            "[metadata]\n",
            '[metadata]\n"x-keep-services" = [{ port = 8080, name = "index" }]\n',
        )
    else:
        text += '\n[metadata]\n"x-keep-services" = [{ port = 8080, name = "index" }]\n'
    toml.write_text(text)
    if dockerfile_user:
        dockerfile = task / "environment/Dockerfile"
        dockerfile.write_text(dockerfile.read_text() + f"\nUSER {dockerfile_user}\n")
    return task


def _prestart_rollout(tmp_path, monkeypatch, task, agent, sandbox_user):
    from benchflow.rollout import Rollout, RolloutConfig

    rollout = Rollout(
        RolloutConfig(
            task_path=task,
            agent=agent,
            environment="daytona",
            sandbox_user=sandbox_user,
            jobs_dir=tmp_path / "jobs",
        )
    )
    # Stop right after the preflight: anything later is out of scope.
    monkeypatch.setattr(
        "benchflow.rollout.prepare_terminal_review",
        lambda _r: (_ for _ in ()).throw(RuntimeError("past preflight")),
    )
    return rollout


async def test_model_agent_refused_before_provision(tmp_path, monkeypatch):
    """DFR-p3-2: the run the freezer check would refuse is refused first."""
    from benchflow.errors import UserError, user_message
    from benchflow.sandbox.protocol import KeptServiceFreezerUnavailable
    from tests.test_keep_services_tb2 import _copy

    rollout = _prestart_rollout(
        tmp_path, monkeypatch, _copy(tmp_path), "claude-agent-acp", "agent"
    )
    start = AsyncMock(side_effect=AssertionError("must not provision"))
    monkeypatch.setattr(rollout, "start", start)
    with pytest.raises(KeptServiceFreezerUnavailable, match=r"next:.*docker") as caught:
        await rollout._prestart_steps()
    start.assert_not_called()
    assert isinstance(caught.value, UserError)
    assert "non-root user 'agent'" in str(caught.value)
    assert "\nnext: run with --sandbox docker" in user_message(caught.value)


@pytest.mark.parametrize(
    ("agent", "sandbox_user"),
    [
        ("nop", "agent"),
        ("oracle", "agent"),
        ("oracle", None),
        ("claude-agent-acp", None),
        ("claude-agent-acp", "root"),
    ],
)
async def test_controls_and_root_agents_not_refused_early(
    tmp_path, monkeypatch, agent, sandbox_user
):
    """DFX-SCALE r2: oracle, nop and root agents scored on Daytona at
    dc509087a; the early check must leave them to setup."""
    from tests.test_keep_services_tb2 import _copy

    rollout = _prestart_rollout(
        tmp_path, monkeypatch, _copy(tmp_path), agent, sandbox_user
    )
    with pytest.raises(RuntimeError, match="past preflight"):
        await rollout._prestart_steps()


@pytest.mark.parametrize(
    ("verifier_user", "dockerfile_user", "prebuilt", "refused", "early"),
    [
        # Unset on a prebuilt image: the image's own USER decides (L1).
        (None, None, True, True, False),
        ("root", None, True, True, True),
        (None, "root", True, True, False),
        (None, "app", True, False, False),
        ("app", None, True, False, False),
        ("root", "app", True, True, True),
        # Built from its own Dockerfile: only an explicit final USER root.
        (None, None, False, True, False),
        (None, "root", False, True, True),
        (None, "app", False, False, False),
    ],
)
def test_early_refusal_agrees_with_freezer_check(
    tmp_path, verifier_user, dockerfile_user, prebuilt, refused, early
):
    """The early check refuses only what the check after setup refuses
    (agent UID 1000, verifier UID 0 when root), and only when the task
    alone makes the verifier root."""
    from benchflow.sandbox._identities import Identity
    from benchflow.sandbox.protocol import KeptServiceFreezerUnavailable
    from benchflow.task import Task, keep_services

    path = _kept_task(
        tmp_path,
        verifier_user=verifier_user,
        dockerfile_user=dockerfile_user,
        prebuilt=prebuilt,
    )
    root_verifier = verifier_user == "root" or (
        verifier_user is None and dockerfile_user in (None, "root")
    )
    task = Task(path)
    task._benchflow_service_identities = (
        Identity(1000, 1000),
        Identity(0 if root_verifier else 1001, 0 if root_verifier else 1001),
    )
    late = keep_services.provider_refusal(
        task,
        path,
        sandbox="daytona",
        keep=keep_services.trial_keep(task, "agent"),
        agent="claude-agent-acp",
    )
    assert (late is not None) is refused
    assert refused or not early

    def run_early():
        keep_services.preflight_daytona(
            Task(path),
            path,
            sandbox="daytona",
            agent="claude-agent-acp",
            sandbox_user="agent",
        )

    if early:
        with pytest.raises(KeptServiceFreezerUnavailable):
            run_early()
    else:
        run_early()


async def test_refusal_is_unscored_with_full_next_line(tmp_path, monkeypatch, caplog):
    """DFR-p3-2/9: the public trial path records an expected capability refusal."""
    from benchflow.rollout import Rollout, RolloutConfig
    from tests.test_keep_services_tb2 import _copy

    trial = Rollout(
        RolloutConfig(
            task_path=_copy(tmp_path),
            agent="claude-agent-acp",
            environment="daytona",
            jobs_dir=tmp_path / "jobs",
        )
    )
    start = AsyncMock(side_effect=AssertionError("must not provision"))
    monkeypatch.setattr(trial, "start", start)
    result = await trial.run()
    assert result.error_category == "kept_service_freezer"
    assert result.rewards is None
    start.assert_not_called()
    assert "\nnext: run with --sandbox docker" in caplog.text
    assert not (result.rollout_dir / "error-traceback.txt").exists()


def test_idle_inference_and_diagnostic():
    session = ACPSession("test")
    session.model_request_activity = lambda: (1, 619.0, 1.0)
    watchdog = IdleWatchdog.start(
        session, idle_timeout_sec=600, wall_timeout_sec=900, now=0
    )
    watchdog.observe(session, now=619)
    assert not watchdog.idle_expired(619)
    assert watchdog.wall_clock_expired(901)
    session.model_request_activity = lambda: (0, 0.0, 620.0)
    watchdog.observe(session, now=620)
    watchdog.observe(session, now=1221)
    error = watchdog.timeout_error(session, now=1221)
    assert "model requests in flight: 0, oldest 0.0s" in str(error)
    assert error.diagnostic.to_dict()["model_requests_in_flight"] == 0


@pytest.fixture
def capture(tmp_path, monkeypatch):
    log = tmp_path / "callback.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log))
    monkeypatch.delenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", raising=False)
    monkeypatch.setattr(httpx.Client, "send", httpx.Client.send)
    monkeypatch.setattr(httpx.AsyncClient, "send", httpx.AsyncClient.send)
    import aiohttp

    monkeypatch.setattr(
        aiohttp.ClientSession, "_request", aiohttp.ClientSession._request
    )
    guard.install_outbound_check()
    process = LiteLLMProcess.__new__(LiteLLMProcess)
    process._live_trajectory_path = tmp_path / "llm_trajectory.jsonl"
    process._request_events = {}
    return log, process


async def _request(app):
    messages = [
        {
            "type": "http.request",
            "body": b'{"model":"test","messages":[]}',
            "more_body": False,
        }
    ]

    async def receive():
        return messages.pop()

    await guard.SingleRouteGuard(app)(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"content-type", b"application/json")],
            "query_string": b"",
        },
        receive,
        AsyncMock(),
    )


@pytest.mark.parametrize("never", [False, True])
async def test_slow_fake_endpoint_obeys_wall_not_idle(capture, never):
    """DFR-p3-6: a real local HTTP endpoint delays beyond the idle deadline."""
    from benchflow.acp.runtime import execute_prompts
    from benchflow.acp.types import PromptResult
    from benchflow.diagnostics import AgentPromptTimeoutError

    log, process = capture
    tasks = set()

    async def handler(reader, writer):
        tasks.add(asyncio.current_task())
        try:
            await reader.readuntil(b"\r\n\r\n")
            if never:
                await asyncio.Event().wait()
            else:
                await asyncio.sleep(2.2)
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}"
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    async def app(scope, receive, send):
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(f"http://127.0.0.1:{port}", json={})
        await send({"type": "http.response.start", "status": response.status_code})
        await send({"type": "http.response.body", "body": response.content})

    async def prompt(_):
        await _request(app)
        return PromptResult(stop_reason="end_turn")

    def activity():
        if log.exists():
            process._capture_request_events(log.read_text())
        return process.model_request_activity()

    session = ACPSession("fake")
    session.model_request_activity = activity
    client = SimpleNamespace(
        prompt=prompt, cancel=AsyncMock(), expect_silence=lambda _: None
    )
    try:
        if never:
            # Past its idle budget but within three, so the wall ends it
            # (L6 caps how long a pending request counts).
            with pytest.raises(AgentPromptTimeoutError, match="wall-clock budget"):
                await execute_prompts(
                    client, session, ["hello"], timeout=4, idle_timeout=2
                )
        else:
            await execute_prompts(client, session, ["hello"], timeout=4, idle_timeout=1)
        activity()
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        assert rows[0]["event"] == "request_start"
        assert any(row["event"] == "upstream_start" for row in rows)
        assert rows[-1]["event"] == "request_end"
    finally:
        server.close()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await server.wait_closed()


async def test_invalid_provider_status_bytes_and_retry_time(capture):
    log, process = capture
    replies = [
        httpx.Response(502, content=b"not JSON"),
        httpx.Response(503, content=b"also not JSON"),
    ]

    async def app(scope, receive, send):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: replies.pop(0))
        ) as client:
            for _ in range(2):
                response = await client.post(
                    "https://provider.test/v1/chat/completions",
                    json={"prompt": "private"},
                )
                await asyncio.sleep(0.02)
        await send({"type": "http.response.start", "status": response.status_code})
        await send({"type": "http.response.body", "body": b"error"})

    await _request(app)
    process._capture_request_events(log.read_text())
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    ends = [row for row in rows if row["event"] == "upstream_end"]
    assert [row["status"] for row in ends] == [502, 503]
    assert [row["response_bytes"] for row in ends] == [8, 13]
    assert all(row["invalid_provider_response"] for row in ends)
    timing, error = process.request_summary()
    assert timing["provider_retry_wait"] >= 0.02
    assert "HTTP 503" in error and "upstream" in error
    assert "private" not in log.read_text()
    assert (log.parent / "provider_requests.jsonl").read_text() == log.read_text()


async def test_streaming_request_stays_active_until_last_byte(capture):
    """DFR-p3-6/7: headers alone must not end inference or its byte count."""
    log, process = capture

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"first"
            process._capture_request_events(log.read_text())
            assert process.model_request_activity()[0] == 1
            assert not any(
                row.get("event") == "upstream_end"
                for row in process._request_events.values()
            )
            yield b"last"

    async def app(scope, receive, send):
        transport = httpx.MockTransport(lambda _: httpx.Response(200, stream=Stream()))
        async with (
            httpx.AsyncClient(transport=transport) as client,
            client.stream("POST", "https://provider.test", json={}) as response,
        ):
            await send({"type": "http.response.start", "status": response.status_code})
            async for chunk in response.aiter_bytes():
                await send({"type": "http.response.body", "body": chunk})

    await _request(app)
    process._capture_request_events(log.read_text())
    assert process.model_request_activity()[0] == 0
    ends = [
        row
        for row in process._request_events.values()
        if row["event"] == "upstream_end"
    ]
    assert len(ends) == 1 and ends[0]["response_bytes"] == 9
    from benchflow.providers.litellm_logging import trajectory_from_litellm_callback_log

    trajectory = trajectory_from_litellm_callback_log(
        log.read_text(), session_id="test", agent_name="test"
    )
    assert not trajectory.exchanges  # Metadata is never a fabricated LLM exchange.


@pytest.mark.parametrize("compressed_response", [False, True])
@pytest.mark.parametrize(
    "body,invalid", [(b'{"error":"busy"}', False), (b"<html>bad gateway</html>", True)]
)
async def test_provider_error_uses_complete_decoded_json(
    capture, body, invalid, compressed_response
):
    log, process = capture
    body = b"\n" * 70000 + body
    wire_body = gzip.compress(body) if compressed_response else body

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield wire_body

    async def app(scope, receive, send):
        transport = httpx.MockTransport(
            lambda _: httpx.Response(
                502,
                headers={"content-encoding": "gzip"} if compressed_response else {},
                stream=Stream(),
            )
        )
        async with (
            httpx.AsyncClient(transport=transport) as client,
            client.stream("POST", "https://provider.test", json={}) as response,
        ):
            decoded = await response.aread()
            assert decoded == body
            await send({"type": "http.response.start", "status": 502})
            await send({"type": "http.response.body", "body": decoded})

    await _request(app)
    process._capture_request_events(log.read_text())
    ends = [
        row
        for row in process._request_events.values()
        if row["event"] == "upstream_end"
    ]
    assert ends[0]["invalid_provider_response"] is invalid
    assert ends[0]["response_bytes"] == len(wire_body)


@pytest.mark.parametrize(
    "error,category",
    [
        ("KeptServiceFreezerUnavailable: cannot freeze", "kept_service_freezer"),
        (
            "RuntimeError: BenchFlow could not prepare its protected runtime",
            "runtime_sealing",
        ),
        (
            "SandboxDaemonGuardError: sandbox startup refused: the guard did not report (exec status 124)",
            "guard_timeout",
        ),
        (
            "invalid provider response: upstream HTTP 502, non-JSON body, upstream 301s",
            "invalid_provider_response",
        ),
    ],
)
def test_categories_have_remedies_and_explain(error, category):
    from benchflow._utils.scoring import classify_error
    from benchflow.cli.explain import find
    from benchflow.failures import describe

    assert classify_error(error) == category
    assert describe(category).next_step
    assert find(category).body
