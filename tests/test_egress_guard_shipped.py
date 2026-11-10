"""The deny-mode egress proxy judges model-shaped bodies with the files it ships with.

The proxy runs in the sandbox as ``python -I -S -B proxy.py``, so BenchFlow is
not importable there: whatever it loads must be uploaded beside it. In-process
tests import BenchFlow's own copy of the request guard and cannot see a missing
upload; these tests run the proxy with only its uploaded files.
"""

# Re-export the real socket fixture for this module.
# ruff: noqa: F811
from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import _egress_denylist_proxy as proxy
from benchflow.sandbox import egress_denylist
from benchflow.sandbox.egress_denylist import EgressDenylist
from tests.test_egress_connect_authority import (  # noqa: F401
    _read_response,
    _tunnel,
    authority_stack,
)


async def _plain_deny_uploads(monkeypatch) -> dict[str, bytes]:
    uploads: list[dict[str, bytes]] = []
    monkeypatch.setattr(egress_denylist, "_run", AsyncMock())
    monkeypatch.setattr(egress_denylist, "_wait_healthy", AsyncMock())

    async def upload(env, files):
        uploads.append(files)

    monkeypatch.setattr(egress_denylist, "_upload", upload)
    await egress_denylist.start_egress_denylist(
        object(), "agent", EgressDenylist(("https://paper.test/protected",), ())
    )
    return uploads[-1]


@pytest.mark.asyncio
async def test_a_plain_denylist_uploads_the_request_guard(monkeypatch):
    """Guards the fix for the deny-mode proxy's missing request guard (the guard ships
    with every egress proxy)."""
    files = await _plain_deny_uploads(monkeypatch)
    policy = json.loads(files["policy.json"])
    assert "native_codex_model_only" not in policy
    assert "allowed_hosts" not in policy
    assert "request_guard.py" in files
    assert (
        files["request_guard.py"] == egress_denylist._REQUEST_GUARD_SCRIPT.read_bytes()
    )


@pytest.mark.asyncio
async def test_the_proxy_judges_a_model_keyed_body_with_only_its_uploaded_files(
    monkeypatch, tmp_path
):
    """Guards the fix for the deny-mode proxy's missing request guard (the proxy's
    isolated Python loads the
    guard it was shipped with, not BenchFlow's)."""
    files = await _plain_deny_uploads(monkeypatch)
    for name, data in files.items():
        if name.endswith(".py"):
            (tmp_path / name).write_bytes(data)
    code = (
        "import importlib.util,json;"
        f"spec=importlib.util.spec_from_file_location('proxy',{str(tmp_path / 'proxy.py')!r});"
        "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
        "policy=m.Policy([],[]);"
        'print(json.dumps([m.deny_body_rule(policy,b\'{"model":"x","features":[1,2]}\',\'/upload\'),'
        "m.deny_body_rule(policy,"
        'b\'{"model":"m","messages":[],"tools":[{"type":"web_search"}]}\',\'/relay\')]))'
    )
    out = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", code],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert out.returncode == 0, out.stderr
    data, hosted = json.loads(out.stdout)
    assert data is None
    assert hosted.startswith("deny-model-body: ")


def test_a_body_check_that_fails_is_a_logged_refusal(authority_stack, monkeypatch):
    """Guards the fix for the deny-mode proxy's missing request guard (a body the
    proxy cannot judge gets a 403
    and a blocked-log line, not a dropped connection)."""

    def unavailable():
        raise ImportError("request guard unavailable")

    monkeypatch.setattr(proxy, "_request_guard", unavailable)
    body = b'{"model":"m","messages":[{"role":"user","content":"hi"}]}'
    with _tunnel(authority_stack) as tls:
        tls.sendall(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: carrier.test\r\n"
            + b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        status, headers, _ = _read_response(tls)
    assert status == 403
    assert headers.get("x-benchflow-blocked") == "1"
    assert not authority_stack.requests
    events = [json.loads(line) for line in authority_stack.log.read_text().splitlines()]
    assert [event["rule"] for event in events] == ["body-check-failed: ImportError"]


def test_a_streamed_text_body_that_cannot_be_judged_is_a_logged_refusal(
    authority_stack, monkeypatch
):
    """Guards the same fix for a streamed body: a model-keyed JSON line past the first chunk of a
    text upload is judged while the body streams, and a guard that fails there
    dropped the connection with no reply and no log line."""

    def unavailable():
        raise ImportError("request guard unavailable")

    monkeypatch.setattr(proxy, "_request_guard", unavailable)
    body = (
        b"notes " * 40_000
        + b'\n{"model":"m","messages":[{"role":"user","content":"hi"}]}\n'
    )
    with _tunnel(authority_stack) as tls:
        tls.sendall(
            b"POST /upload HTTP/1.1\r\nHost: carrier.test\r\n"
            + b"Content-Type: text/plain\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        status, headers, _ = _read_response(tls)
    assert status == 403
    assert headers.get("x-benchflow-blocked") == "1"
    events = [json.loads(line) for line in authority_stack.log.read_text().splitlines()]
    assert [event["rule"] for event in events] == ["body-check-failed: ImportError"]


# ------------------------------------------------------------- real sandbox

_PROBE = dedent("""\
    import json, os, urllib.error, urllib.request
    assert os.getuid() != 0, 'probe must run as the agent'

    def post(body):
        request = urllib.request.Request(
            'http://example.com/upload', data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=30) as reply:
                return [reply.status, reply.headers.get('X-BenchFlow-Blocked')]
        except urllib.error.HTTPError as e:
            return [e.code, e.headers.get('X-BenchFlow-Blocked')]
        except Exception as e:  # the connection ended without an HTTP reply
            return [type(e).__name__ + ': ' + str(e), None]

    print(json.dumps({
        'data': post({'name': 'x', 'features': [1, 2]}),
        'model_keyed_data': post({'model': 'x', 'features': [1, 2]}),
        'graphql': post({'query': 'mutation($input: In!) { make(input: $input) { id } }',
                         'variables': {'input': {'title': 't'}}}),
        'chat_request': post({'model': 'x', 'messages': [{'role': 'user', 'content': 'hi'}]}),
        'hosted_search': post({'model': 'x', 'messages': [], 'tools': [{'type': 'web_search'}]}),
    }))
    """)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_denylist_sandbox_answers_every_model_keyed_body(tmp_path: Path):
    """Guards the fix for the deny-mode proxy's missing request guard (in a Docker
    sandbox, a JSON body with a
    model-request key was dropped with no reply and no log line).

    Needs Docker and the public network (example.com); run with ``-m integration``.
    """
    from benchflow.sandbox.egress_denylist import (
        denylist_agent_env,
        egress_denylist_for,
        start_egress_denylist,
        stop_egress_denylist,
    )
    from benchflow.sandbox.lockdown import enforce_agent_egress_firewall
    from benchflow.sandbox.setup import _create_sandbox_environment
    from benchflow.task import RolloutPaths, Task

    if not shutil.which("docker"):
        pytest.skip("Docker not installed")
    if subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode:
        pytest.skip("Docker daemon unavailable")

    task_path = tmp_path / "denylist-guard"
    (task_path / "environment").mkdir(parents=True)
    (task_path / "environment" / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends "
        "ca-certificates curl iptables && rm -rf /var/lib/apt/lists/*\n"
        "RUN useradd -m -s /bin/bash agent\nWORKDIR /app\n"
    )
    (task_path / "task.md").write_text(
        dedent("""\
        ---
        schema_version: "1.3"
        sandbox:
          network_mode: denylist
          blocked_hosts:
            - example.net
          cpus: 1
          memory_mb: 2048
          storage_mb: 4096
          workdir: /app
        ---
        Check the network policy.
        """)
    )
    task = Task(task_path)
    policy = egress_denylist_for(task.config.sandbox)
    assert policy is not None and not policy.allow_mode
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    sandbox = _create_sandbox_environment(
        "docker",
        task,
        task_path,
        f"denylist-guard-{uuid.uuid4().hex[:12]}",
        RolloutPaths(rollout_dir),
    )
    try:
        await sandbox.start(force_build=False)
        await start_egress_denylist(sandbox, "agent", policy)
        agent_env = denylist_agent_env({})
        await enforce_agent_egress_firewall(sandbox, "agent", agent_env)
        result = await sandbox.exec(
            "python3 -c " + shlex.quote(_PROBE),
            user="agent",
            env=agent_env,
            timeout_sec=180,
        )
        stderr_log = await sandbox.exec(
            "cat /opt/runtime-egress/stderr.log", user="root", timeout_sec=30
        )
    finally:
        try:
            await stop_egress_denylist(sandbox, rollout_dir)
        finally:
            await sandbox.stop(delete=True)

    assert result.return_code == 0, result.stdout + result.stderr
    replies = json.loads(result.stdout)
    proxy_stderr = stderr_log.stdout or ""
    print(json.dumps(replies, indent=2))
    print("proxy stderr.log:\n" + proxy_stderr)
    log = rollout_dir / "trajectory" / "egress_denylist.jsonl"
    events = (
        [json.loads(line) for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )
    print("egress_denylist.jsonl:", json.dumps(events))
    # Ordinary data and data sharing one model key reach the origin alike.
    assert isinstance(replies["data"][0], int), replies
    assert replies["model_keyed_data"] == replies["data"], replies
    assert replies["graphql"] == replies["data"], replies
    assert isinstance(replies["chat_request"][0], int), replies
    assert replies["chat_request"][1] is None, replies
    # A body the guard refuses is a logged 403.
    assert replies["hosted_search"] == [403, "1"], replies
    assert [e["rule"].split(":")[0] for e in events] == ["deny-model-body"], events
    assert "Traceback" not in proxy_stderr, proxy_stderr
