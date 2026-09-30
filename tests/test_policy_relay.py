"""The policy relay in front of a trainer's inference server.

PostTrain Arena exposed its TRL vLLM server through ad hoc tunnels and a
bridge with a side store at ``/v1/benchflow/logprobs/{id}``, because rollouts
could not reach a loopback server and LiteLLM's stream assembly dropped
logprobs. The relay is BenchFlow's answer: rollouts get a per-rollout
credential for an inference-only proxy, the server's key and address stay
in the trainer's process, and every call's tokens are digested from the
server's raw answer, streamed or not, for attestation.
"""

from __future__ import annotations

import json

import httpx
import pytest

from benchflow.training.policy import Policy
from benchflow.training.relay import PolicyRelay, is_loopback_url, with_host
from benchflow.trajectories.token_capture import build_token_capture, token_digest
from tests.fixtures.mock_token_logprobs_server import start_server

BODY = {
    "model": "mock-policy",
    "messages": [{"role": "user", "content": "hi"}],
    "logprobs": True,
    "return_token_ids": True,
}


@pytest.fixture
def server():
    srv = start_server(flavor="vllm")
    yield srv
    srv.shutdown()


@pytest.fixture
def sglang_server():
    srv = start_server(flavor="sglang")
    yield srv
    srv.shutdown()


def _expected_digest(response: dict) -> str:
    capture = build_token_capture(
        {
            "event": "success",
            "token_capture": {
                "enabled": True,
                "wire": "openai-chat",
                "provider": "vllm",
            },
            "request": {"body": BODY},
            "response": response,
        }
    )
    assert capture is not None and capture["unavailable"] == {}
    return capture["digest"]


async def test_non_streamed_call_passes_through_and_is_digested(server):
    versions = iter([7, 7])
    relay = PolicyRelay(
        f"{server.base_url}/v1",
        upstream_api_key="server-secret",
        model="vllm/mock-policy",
        version=lambda: next(versions),
    )
    async with relay:
        grant = relay.grant("task/g/0/1")
        async with httpx.AsyncClient() as client:
            answer = await client.post(
                f"{relay.host_url}/chat/completions",
                json=BODY,
                headers={"Authorization": f"Bearer {grant.token}"},
            )
        calls = await relay.revoke(grant)
    assert answer.status_code == 200
    data = answer.json()
    assert data["prompt_token_ids"] == [ord("h"), ord("i")]
    # The server saw the relay's key, never the rollout's credential.
    assert server.requests[-1]["authorization"] == "Bearer server-secret"
    assert server.requests[-1]["body"] == BODY
    [call] = calls
    assert call.status == "ok" and call.http_status == 200
    assert call.version == 7 and call.version_end == 7
    assert call.digest == _expected_digest(data)
    assert call.prompt_tokens == 2 and call.completion_tokens == 3


async def test_streamed_call_is_relayed_unbuffered_and_digests_like_the_body(server):
    relay = PolicyRelay(
        f"{server.base_url}/v1", model="vllm/mock-policy", version=lambda: 1
    )
    async with relay:
        grant = relay.grant("stream")
        async with httpx.AsyncClient() as client:
            direct = await client.post(
                f"{server.base_url}/v1/chat/completions", json={**BODY, "stream": True}
            )
            relayed = await client.post(
                f"{relay.host_url}/chat/completions",
                json={**BODY, "stream": True},
                headers={"Authorization": f"Bearer {grant.token}"},
            )
            plain = await client.post(
                f"{server.base_url}/v1/chat/completions", json=BODY
            )
        [call] = await relay.revoke(grant)
    assert relayed.status_code == 200
    assert relayed.headers["content-type"].startswith("text/event-stream")
    assert relayed.text == direct.text  # byte for byte
    assert call.stream is True and call.status == "ok"
    # Same tokens and logprobs, streamed or not: the same digest.
    assert call.digest == _expected_digest(plain.json())


async def test_sglang_stream_digest_uses_the_sglext_chunk(sglang_server):
    body = {
        "model": "mock-policy",
        "messages": [{"role": "user", "content": "hi"}],
        "logprobs": True,
        "stream": True,
        "return_input_ids_in_sglext": True,
        "return_output_ids_in_sglext": True,
    }
    relay = PolicyRelay(f"{sglang_server.base_url}/v1", model="sglang/mock-policy")
    async with relay:
        grant = relay.grant("sglang")
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{relay.host_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {grant.token}"},
            )
        [call] = await relay.revoke(grant)
    reply_ids = [1000 + ord(c) for c in "ok!"]
    assert call.digest == token_digest(
        [ord("h"), ord("i")],
        [{"index": 0, "token_ids": reply_ids, "logprobs": [-0.125, -0.25, -0.375]}],
    )


@pytest.mark.parametrize(
    ("method", "path", "auth", "status"),
    [
        ("POST", "/v1/chat/completions", None, 401),
        ("POST", "/v1/chat/completions", "Bearer bfr-000000000000-guess", 401),
        ("POST", "/v1/chat/completions", "Basic abc", 401),
        ("POST", "/update_named_param", "GRANT", 404),
        ("POST", "/v1/messages", "GRANT", 404),
        ("GET", "/v1/chat/completions", "GRANT", 404),
    ],
)
async def test_only_inference_routes_with_a_live_grant(
    server, method, path, auth, status
):
    relay = PolicyRelay(f"{server.base_url}/v1", upstream_api_key="server-secret")
    async with relay:
        grant = relay.grant("x")
        base = relay.host_url.removesuffix("/v1")
        headers = {}
        if auth is not None:
            headers["Authorization"] = (
                f"Bearer {grant.token}" if auth == "GRANT" else auth
            )
        async with httpx.AsyncClient() as client:
            answer = await client.request(
                method, base + path, json=BODY, headers=headers
            )
    assert answer.status_code == status
    assert "server-secret" not in answer.text
    assert server.requests == []


async def test_models_route_and_revocation(server):
    relay = PolicyRelay(f"{server.base_url}/v1")
    async with relay:
        grant = relay.grant("x")
        headers = {"Authorization": f"Bearer {grant.token}"}
        async with httpx.AsyncClient() as client:
            models = await client.get(f"{relay.host_url}/models", headers=headers)
            await relay.revoke(grant)
            after = await client.post(
                f"{relay.host_url}/chat/completions", json=BODY, headers=headers
            )
    assert models.status_code == 200 and models.json()["data"][0]["id"] == "mock-policy"
    assert after.status_code == 401


async def test_upstream_failures_are_recorded_and_never_name_the_server(server):
    server.fail_next.append(503)
    relay = PolicyRelay(f"{server.base_url}/v1")
    async with relay:
        grant = relay.grant("x")
        headers = {"Authorization": f"Bearer {grant.token}"}
        async with httpx.AsyncClient() as client:
            failed = await client.post(
                f"{relay.host_url}/chat/completions", json=BODY, headers=headers
            )
            ok = await client.post(
                f"{relay.host_url}/chat/completions", json=BODY, headers=headers
            )
        calls = await relay.revoke(grant)
    assert failed.status_code == 503 and ok.status_code == 200
    assert [(c.status, c.http_status) for c in calls] == [("error", 503), ("ok", 200)]
    assert calls[0].digest is None

    dead = PolicyRelay("http://127.0.0.1:9/v1")  # nothing listens on port 9
    async with dead:
        grant = dead.grant("x")
        async with httpx.AsyncClient() as client:
            answer = await client.post(
                f"{dead.host_url}/chat/completions",
                json=BODY,
                headers={"Authorization": f"Bearer {grant.token}"},
            )
        [call] = await dead.revoke(grant)
    assert answer.status_code == 502
    assert "127.0.0.1:9" not in answer.text
    assert call.status == "error" and call.http_status is None


async def test_docker_and_public_listeners(server, monkeypatch):
    import benchflow.providers.litellm_runtime as rt

    monkeypatch.setattr(rt, "_docker_host_address", lambda: "127.0.0.1")
    relay = PolicyRelay(
        f"{server.base_url}/v1",
        public_bind="127.0.0.1:0",
        public_url="https://relay.example.test",
    )
    async with relay:
        docker_url = await relay.url_for("docker")
        assert docker_url is not None and docker_url != relay.host_url
        assert await relay.url_for("daytona") == "https://relay.example.test/v1"
        grant = relay.grant("x")
        async with httpx.AsyncClient() as client:
            answer = await client.post(
                f"{docker_url}/chat/completions",
                json=BODY,
                headers={"Authorization": f"Bearer {grant.token}"},
            )
        assert answer.status_code == 200
    plain = PolicyRelay(f"{server.base_url}/v1")
    async with plain:
        assert await plain.url_for("daytona") is None


def test_relay_refuses_bad_configuration():
    with pytest.raises(ValueError, match="http"):
        PolicyRelay("127.0.0.1:8000/v1")
    with pytest.raises(ValueError, match="public_bind"):
        PolicyRelay("http://127.0.0.1:8000/v1", public_url="https://x")


def test_url_helpers():
    assert is_loopback_url("http://127.0.0.1:8000/v1")
    assert is_loopback_url("http://localhost:8000/v1")
    assert not is_loopback_url("http://10.0.0.2:8000/v1")
    assert (
        with_host("http://127.0.0.1:8000/v1", "172.17.0.1")
        == "http://172.17.0.1:8000/v1"
    )


async def test_policy_hands_rollouts_a_credential_not_the_key(server, monkeypatch):
    monkeypatch.setenv("POLICY_KEY_FOR_TEST", "server-secret")
    policy = Policy(
        "vllm/mock-policy",
        base_url=f"{server.base_url}/v1",
        api_key_env="POLICY_KEY_FOR_TEST",
        version=3,
        capture_routed_experts=True,
    )
    async with policy:
        grant = policy.relay.grant("x")
        env = await policy.agent_env(grant.token, "daytona")
        assert env == {
            "BENCHFLOW_PROVIDER_BASE_URL": policy.relay.host_url,
            "BENCHFLOW_PROVIDER_API_KEY": grant.token,
            "BENCHFLOW_CAPTURE_TOKEN_LOGPROBS": "1",
            "BENCHFLOW_CAPTURE_ROUTED_EXPERTS": "1",
        }
        assert "server-secret" not in json.dumps(env)
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{policy.relay.host_url}/chat/completions",
                json=BODY,
                headers={"Authorization": f"Bearer {grant.token}"},
            )
        [call] = await policy.relay.revoke(grant)
    assert call.version == 3
    assert server.requests[-1]["authorization"] == "Bearer server-secret"


def test_policy_validation(monkeypatch):
    with pytest.raises(ValueError, match="route"):
        Policy("my-policy", base_url="http://127.0.0.1:8000/v1")
    with pytest.raises(ValueError, match="not both"):
        Policy("vllm/p", base_url="http://x/v1", api_key="a", api_key_env="B")
    monkeypatch.delenv("NOT_SET_FOR_TEST", raising=False)
    policy = Policy("vllm/p", base_url="http://x/v1", api_key_env="NOT_SET_FOR_TEST")
    with pytest.raises(ValueError, match="not set"):
        policy._resolve_key()
    assert Policy("sglang/p", base_url="http://x/v1").captures_token_ids
    assert not Policy("openai/p", base_url="http://x/v1").captures_token_ids


class _RawUpstream:
    """A one-route upstream that answers with raw bytes after a delay."""

    def __init__(self, answer: bytes, delay: float = 0.0) -> None:
        self.answer, self.delay = answer, delay

    async def __aenter__(self) -> str:
        import asyncio

        async def handle(reader, writer):
            await reader.readuntil(b"\r\n\r\n")
            await asyncio.sleep(self.delay)
            writer.write(self.answer)
            await writer.drain()
            writer.close()

        self._server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}/v1"

    async def __aexit__(self, *exc) -> None:
        self._server.close()


async def test_an_answer_cut_off_mid_body_is_recorded_not_lost():
    truncated = (
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
        b'Content-Length: 1000\r\n\r\n{"id": "cut'
    )
    async with _RawUpstream(truncated) as url:
        relay = PolicyRelay(url, model="vllm/mock-policy", version=lambda: 3)
        async with relay:
            grant = relay.grant("task/g/0/1")
            async with httpx.AsyncClient() as client:
                answer = await client.post(
                    f"{relay.host_url}/chat/completions",
                    json=BODY,
                    headers={"Authorization": f"Bearer {grant.token}"},
                )
            [call] = await relay.revoke(grant)
    assert answer.status_code == 502
    assert call.status == "error" and call.error


async def test_revoking_waits_for_a_call_in_flight():
    import asyncio

    body = json.dumps({"id": "x", "choices": []}).encode()
    slow = (
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body
    )
    async with _RawUpstream(slow, delay=0.5) as url:
        relay = PolicyRelay(url, model="vllm/mock-policy", version=lambda: 3)
        async with relay:
            grant = relay.grant("task/g/0/1")
            async with httpx.AsyncClient() as client:
                request = asyncio.create_task(
                    client.post(
                        f"{relay.host_url}/chat/completions",
                        json=BODY,
                        headers={"Authorization": f"Bearer {grant.token}"},
                    )
                )
                while grant.in_flight == 0:
                    await asyncio.sleep(0.01)
                calls = await relay.revoke(grant)
                await request
    assert [c.status for c in calls] == ["ok"]
