"""Codex ACP on a ChatGPT login under no-web: the native model-only transport.

Change that admits ``codex-acp`` on a ChatGPT login
(``CODEX_AUTH_JSON``) on tasks without network: before it, BenchFlow refused
the route ("Native subscription no-web transport is supported only for direct
Claude ACP"). The egress proxy now admits only Responses requests to Codex's
ChatGPT endpoint, and only bodies the model gateway itself admits under web
restrictions. No public network is used: chatgpt.com resolves to a local TLS
server.
"""

from __future__ import annotations

import http.server
import json
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import AsyncMock

import pytest

from benchflow.agents.codex_config import apply_codex_launch_config
from benchflow.sandbox import _egress_denylist_proxy as proxy
from benchflow.sandbox import egress_denylist
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox._egress_denylist_proxy import (
    CODEX_MODEL_PATH,
    Policy,
    native_codex_body_rule,
)
from benchflow.sandbox.egress_denylist import EgressDenylist, certificate_material
from benchflow.sandbox.native_oauth import (
    native_oauth_egress_policy,
    validate_native_codex_transport,
)

LOGIN = {"CODEX_AUTH_JSON": "{}"}
URL = f"https://chatgpt.com{CODEX_MODEL_PATH}"
IMAGE = "data:image/png;base64,iVBORw0KGgo="


def codex_request(**changes):
    """A Responses request shaped like the ones Codex sends on a ChatGPT login."""
    body = {
        "model": "gpt-6.1-sol",
        "instructions": "You are Codex.",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Build the page."},
                    {"type": "input_image", "image_url": IMAGE},
                ],
            },
            {"type": "reasoning", "summary": [], "encrypted_content": "gAAAA"},
            {
                "type": "function_call",
                "name": "exec_command",
                "arguments": '{"cmd":"ls"}',
                "call_id": "call_1",
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "a b"},
            {
                "type": "custom_tool_call",
                "name": "apply_patch",
                "input": "*** Begin Patch",
                "call_id": "call_2",
            },
            {"type": "custom_tool_call_output", "call_id": "call_2", "output": "ok"},
        ],
        "tools": [
            {
                "type": "function",
                "name": "exec_command",
                "description": "Run a command.",
                "strict": False,
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "type": "custom",
                "name": "apply_patch",
                "description": "Edit files.",
                "format": {"type": "grammar", "syntax": "lark", "definition": "x"},
            },
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "reasoning": {"effort": "xhigh", "summary": "auto"},
        "store": False,
        "stream": True,
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": "019a-thread",
        "text": {"verbosity": "medium"},
    }
    body.update(changes)
    return body


def encoded(body) -> bytes:
    return json.dumps(body).encode()


# --------------------------------------------------------------- selection


def test_codex_login_under_no_web_gets_the_codex_model_only_policy():
    policy = native_oauth_egress_policy("codex-acp", "gpt-6.1-sol", LOGIN, no_web=True)
    assert policy is not None and policy.native_codex_model_only
    assert not policy.native_claude_model_only and not policy.allow_mode
    assert policy.inspect_hosts == ("chatgpt.com",)


def test_codex_api_key_and_open_runs_keep_their_routes():
    assert (
        native_oauth_egress_policy("codex-acp", "gpt-6.1-sol", LOGIN, no_web=False)
        is None
    )
    for api_route in (
        {"OPENAI_API_KEY": "sk-test"},
        {"BENCHFLOW_PROVIDER_API_KEY": "sk-test"},
        {"OPENAI_BASE_URL": "https://other.example/v1"},
    ):
        assert (
            native_oauth_egress_policy(
                "codex-acp", "gpt-6.1-sol", {**LOGIN, **api_route}, no_web=True
            )
            is None
        )


@pytest.mark.parametrize(
    "extra,match",
    [
        ({"OPENAI_BASE_URL": "https://api.openai.com/v1"}, "OPENAI_BASE_URL"),
        (
            {"CODEX_CONFIG": '{"chatgpt_base_url":"https://x.example"}'},
            "chatgpt_base_url",
        ),
        ({"CODEX_CONFIG": '{"model_provider":"proxy"}'}, "model_provider"),
        ({"CODEX_CONFIG": "not json"}, "JSON object"),
    ],
)
def test_codex_routing_elsewhere_is_refused_before_a_sandbox(extra, match):
    with pytest.raises(ValueError, match=match):
        native_oauth_egress_policy(
            "codex-acp", "gpt-6.1-sol", {**LOGIN, **extra}, no_web=True
        )


def test_codex_config_without_routing_keeps_the_policy():
    env = {**LOGIN, "CODEX_CONFIG": '{"web_search":"disabled"}'}
    assert native_oauth_egress_policy(
        "codex-acp", "gpt-6.1-sol", env, no_web=True
    ).native_codex_model_only


def test_other_native_logins_are_still_refused_under_no_web():
    with pytest.raises(ValueError, match="only"):
        native_oauth_egress_policy(
            "muse-code", "muse-spark", {"MUSE_AUTH_JSON": "{}"}, no_web=True
        )


# --------------------------------------------------------------- the gate


@pytest.mark.parametrize("path", [CODEX_MODEL_PATH, CODEX_MODEL_PATH + "/compact"])
def test_exact_codex_model_request(path):
    p = Policy([], [], native_codex_model_only=True)
    assert p.host_rule("chatgpt.com", 443) is None
    assert p.request_rule("chatgpt.com", 443, path, "POST", True) is None


@pytest.mark.parametrize(
    "host,port,path,method,secure",
    [
        ("chatgpt.com", 80, CODEX_MODEL_PATH, "POST", False),
        ("chatgpt.com", 444, CODEX_MODEL_PATH, "POST", True),
        ("chatgpt.com", 443, CODEX_MODEL_PATH, "GET", True),
        ("www.chatgpt.com", 443, CODEX_MODEL_PATH, "POST", True),
        ("ab.chatgpt.com", 443, "/v1/rgstr", "POST", True),
        ("api.openai.com", 443, "/v1/responses", "POST", True),
        ("api.anthropic.com", 443, "/v1/messages", "POST", True),
        ("auth.openai.com", 443, "/oauth/token", "POST", True),
        ("chatgpt.com", 443, "/backend-api/codex/models", "POST", True),
        ("chatgpt.com", 80, "/backend-api/codex/models", "GET", False),
        ("chatgpt.com", 443, "/backend-api/wham/accounts/check", "POST", True),
        ("chatgpt.com", 443, "/backend-api/wham/accounts/check/../x", "GET", True),
        ("chatgpt.com", 443, "/backend-api/wham/accounts", "GET", True),
        ("chatgpt.com", 443, "/backend-api/ps/plugins/installed", "GET", True),
        ("chatgpt.com", 443, CODEX_MODEL_PATH, "OPTIONS", True),
        ("chatgpt.com", 443, "/backend-api/codex/responses/compact", "GET", True),
        ("chatgpt.com", 443, "/backend-api/codex/responses/compact?x=1", "POST", True),
        (
            "chatgpt.com",
            443,
            "/backend-api/codex/analytics-events/events",
            "POST",
            True,
        ),
        ("chatgpt.com", 443, "/backend-api/wham/settings/user", "GET", True),
        ("chatgpt.com", 443, CODEX_MODEL_PATH + "?x=1", "POST", True),
        ("chatgpt.com", 443, CODEX_MODEL_PATH + "/", "POST", True),
        ("chatgpt.com", 443, "/backend-api/codex/%72esponses", "POST", True),
        ("chatgpt.com", 443, "/backend-api/codex/x/../responses", "POST", True),
        ("chatgpt.com", 443, "/backend-api/plugins/featured", "GET", True),
        ("chatgpt.com", 443, "/backend-api/wham/usage", "GET", True),
    ],
)
def test_codex_gate_refuses_every_other_request(host, port, path, method, secure):
    assert Policy([], [], native_codex_model_only=True).request_rule(
        host, port, path, method, secure
    )


@pytest.mark.parametrize(
    "path",
    [
        "/backend-api/codex/models",
        "/backend-api/codex/models?client_version=0.159.3",
        "/backend-api/wham/accounts/check",
    ],
)
def test_codex_session_metadata_gets_are_admitted(path):
    """Codex 0.159.3 opens no session without them (seen in a canary)."""
    assert (
        Policy([], [], native_codex_model_only=True).request_rule(
            "chatgpt.com", 443, path, "GET", True
        )
        is None
    )


def test_codex_gate_admits_no_other_origin_or_tunnel():
    p = Policy([], [], native_codex_model_only=True)
    assert p.host_rule("github.com", 443) == "native-model-origin"
    assert p.host_rule("chatgpt.com", 80) == "native-model-origin"
    assert not p.tunnels("chatgpt.com", 443)
    assert p.model_only("chatgpt.com")


def test_codex_policy_is_controller_only_and_exclusive():
    with pytest.raises(ValueError):
        Policy([], [], native_codex_model_only="true")
    with pytest.raises(ValueError, match="one model origin"):
        Policy([], [], native_claude_model_only=True, native_codex_model_only=True)
    with pytest.raises(ValueError, match="gateway"):
        Policy([], [], 12345, native_codex_model_only=True)
    with pytest.raises(ValueError, match="allowlist"):
        Policy([], [], allowed_hosts=["chatgpt.com"], native_codex_model_only=True)


def test_codex_policy_survives_the_policy_file(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {"blocked_urls": [], "blocked_hosts": [], "native_codex_model_only": True}
        )
    )
    assert Policy.load(str(path)).native_codex_model_only


def test_no_web_codex_config_turns_off_what_the_gate_refuses():
    """Codex 0.159+ zstd-compresses Responses bodies on a ChatGPT login (seen in a canary)."""
    env, _ = apply_codex_launch_config(
        "codex-acp",
        {**LOGIN, "BENCHFLOW_DISALLOW_WEB_TOOLS": "1"},
        model="gpt-6.1-sol",
        reasoning_effort="xhigh",
    )
    config = json.loads(env["CODEX_CONFIG"])
    assert config["web_search"] == "disabled"
    assert config["features"] == {
        "enable_request_compression": False,
        "image_generation": False,
        "plugins": False,
    }
    assert config["analytics"] == {"enabled": False}
    assert config["feedback"] == {"enabled": False}
    assert config["check_for_update_on_startup"] is False
    own, _ = apply_codex_launch_config(
        "codex-acp",
        {
            **LOGIN,
            "BENCHFLOW_DISALLOW_WEB_TOOLS": "1",
            "CODEX_CONFIG": '{"features":{"plugins":true}}',
        },
        model="gpt-6.1-sol",
        reasoning_effort="xhigh",
    )
    assert json.loads(own["CODEX_CONFIG"])["features"]["plugins"] is True
    open_run, _ = apply_codex_launch_config(
        "codex-acp", dict(LOGIN), model="gpt-6.1-sol", reasoning_effort="xhigh"
    )
    assert "features" not in json.loads(open_run.get("CODEX_CONFIG") or "{}")


# --------------------------------------------------------------- the body


def test_codex_shaped_request_is_admitted():
    assert native_codex_body_rule(encoded(codex_request()), []) is None


PASSTHROUGH = {
    "turn_id": "01a1a26b-3a89-7af3-8801-051c1fe1a002",
    "create_time": 1793726298.92,
    "content_item_kinds": ["user.text"],
    "content_item_metadata": [{"provenance": {"type": "user", "hook": False}}],
}


def test_codex_bookkeeping_items_are_admitted():
    """Codex 0.159+ labels messages and sends reasoning updates (seen in a canary)."""
    body = codex_request()
    body["input"] = [
        {
            "type": "additional_tools",
            "role": "developer",
            "id": "at_1",
            "tools": [{"type": "namespace", "name": "collaboration", "tools": []}],
        },
        {
            "type": "message",
            "role": "user",
            "id": "msg_1",
            "content": [{"type": "input_text", "text": "Build the page."}],
            "internal_chat_message_metadata_passthrough": PASSTHROUGH,
        },
        {"type": "configuration_update", "reasoning": {"effort": "xhigh"}},
        {
            "type": "custom_tool_call",
            "id": "ctc_1",
            "status": "completed",
            "call_id": "call_1",
            "name": "exec",
            "input": "ls",
            "internal_chat_message_metadata_passthrough": PASSTHROUGH,
        },
        {
            "type": "custom_tool_call_output",
            "call_id": "call_1",
            "output": "a b",
            "internal_chat_message_metadata_passthrough": PASSTHROUGH,
        },
    ]
    assert native_codex_body_rule(encoded(body), []) is None


@pytest.mark.parametrize(
    "item",
    [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "x"}],
            "internal_chat_message_metadata_passthrough": {
                "parts": [
                    {"type": "input_image", "image_url": "https://x.example/a.png"}
                ]
            },
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "x"}],
            "internal_chat_message_metadata_passthrough": {"source": {"type": "url"}},
        },
        {"type": "configuration_update", "reasoning": {"effort": "xhigh"}, "tools": []},
        {"type": "configuration_update", "reasoning": {"file_id": "file_1"}},
    ],
)
def test_bookkeeping_that_carries_more_than_plain_values_is_refused(item):
    body = codex_request()
    body["input"] = [item]
    assert native_codex_body_rule(encoded(body), []).startswith(
        "native-model-body-hosted: input is not self-contained (item 0: "
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"tools": [{"type": "web_search"}]},
        {"tools": [{"type": "web_search_preview"}]},
        {"tools": [{"type": "file_search", "vector_store_ids": ["vs_1"]}]},
        {"tools": [{"type": "code_interpreter", "container": {"type": "auto"}}]},
        {"tools": [{"type": "image_generation"}]},
        {
            "tools": [
                {"type": "mcp", "server_label": "x", "server_url": "https://x.example"}
            ]
        },
        {"previous_response_id": "resp_1"},
        {"conversation": "conv_1"},
        {"prompt": {"id": "pmpt_1"}},
        {"web_search_options": {}},
        {"x_capability": True},
        {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": "https://x.example/a.png"}
                    ],
                }
            ]
        },
        {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_file", "file_url": "https://x.example/a.pdf"}
                    ],
                }
            ]
        },
        {
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [
                        {"type": "input_image", "image_url": "https://x.example/b.png"}
                    ],
                }
            ]
        },
        {"model": "gpt-4o-search-preview"},
    ],
)
def test_requests_openai_would_run_or_fetch_for_are_refused(changes):
    assert native_codex_body_rule(encoded(codex_request(**changes)), []).startswith(
        "native-model-body-hosted: "
    )


@pytest.mark.parametrize(
    "body,headers,rule",
    [
        (
            encoded(codex_request()),
            [("Content-Encoding", "zstd")],
            "native-model-body-encoding",
        ),
        (b"not json", [], "native-model-body-json"),
        (b"[1]", [], "native-model-body-json"),
        (b'{"model":"a","model":"b"}', [], "native-model-body-json"),
        (b"\xff\xfe", [], "native-model-body-json"),
    ],
)
def test_unreadable_bodies_are_refused(body, headers, rule):
    assert native_codex_body_rule(body, headers) == rule


def test_a_refused_item_is_named_by_type_and_fields_never_values():
    body = codex_request()
    body["input"] = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "a"}],
        },
        {
            "type": "reasoning",
            "summary": [],
            "content": None,
            "encrypted_content": "SECRET",
        },
    ]
    rule = native_codex_body_rule(encoded(body), [])
    assert rule == (
        "native-model-body-hosted: input is not self-contained "
        "(item 1: reasoning; fields content,encrypted_content,summary,type; "
        "content=NoneType; summary=)"
    )
    assert "SECRET" not in rule


def test_the_sandbox_copy_of_the_guard_is_loaded_from_beside_the_proxy(tmp_path):
    """A sandbox runs proxy.py with request_guard.py beside it, on an isolated Python."""
    shutil.copy(proxy.__file__, tmp_path / "proxy.py")
    shutil.copy(egress_denylist._REQUEST_GUARD_SCRIPT, tmp_path / "request_guard.py")
    script = (
        "import importlib.util,json,sys;"
        f"spec=importlib.util.spec_from_file_location('proxy',{str(tmp_path / 'proxy.py')!r});"
        "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
        "ok=json.dumps({'model':'m','input':'hi'}).encode();"
        "bad=json.dumps({'model':'m','input':'hi','tools':[{'type':'web_search'}]}).encode();"
        "print(json.dumps([m.native_codex_body_rule(ok,[]),m.native_codex_body_rule(bad,[]),"
        "m._request_guard().__name__]))"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", script], capture_output=True, text=True, check=True
    )
    assert json.loads(out.stdout) == [
        None,
        "native-model-body-hosted: a tool is not client-executed",
        "benchflow_request_guard",
    ]


# ------------------------------------------------------- through the proxy


class _ChatGPT(http.server.BaseHTTPRequestHandler):
    received: ClassVar[list[tuple[str, str, bytes]]] = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        type(self).received.append(("POST", self.path, body))
        reply = b'data: {"type":"response.completed"}\n\n'
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def do_GET(self):
        type(self).received.append(("GET", self.path, b""))
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass


@pytest.fixture
def codex_stack(tmp_path: Path, monkeypatch):
    """The proxy under the Codex policy, chatgpt.com served by a local TLS server."""
    upstream_ca = certificate_material(("chatgpt.com",))
    proxy_ca = certificate_material(("chatgpt.com",))
    (tmp_path / "upstream-ca.crt").write_bytes(upstream_ca["ca.crt"])
    (tmp_path / "client-ca.crt").write_bytes(proxy_ca["ca.crt"])
    certs = tmp_path / "certs"
    certs.mkdir()
    for name, material in proxy_ca.items():
        if name.endswith(".pem"):
            (certs / name).write_bytes(material)
    pem = tmp_path / "upstream-chatgpt.com.pem"
    pem.write_bytes(upstream_ca["chatgpt.com.pem"])
    _ChatGPT.received = []
    upstream = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ChatGPT)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(pem))
    upstream.socket = ctx.wrap_socket(upstream.socket, server_side=True)
    port = upstream.server_address[1]

    def connect(host, upstream_port, **kwargs):
        assert (host, upstream_port) == ("chatgpt.com", 443)
        return socket.create_connection(("127.0.0.1", port), timeout=10)

    monkeypatch.setattr(proxy, "_connect_upstream", connect)
    monkeypatch.setattr(proxy, "_resolve", lambda host, p: ["93.184.216.34"])
    log = tmp_path / "blocked.jsonl"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
    server = proxy.serve(
        free,
        Policy([], [], native_codex_model_only=True),
        proxy.CertStore(str(certs)),
        proxy.Log(str(log)),
        upstream_ca=str(tmp_path / "upstream-ca.crt"),
    )
    for s in (upstream, server):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    proxy_url = f"http://127.0.0.1:{server.server_address[1]}"
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url}),
        urllib.request.HTTPSHandler(
            context=ssl.create_default_context(cafile=str(tmp_path / "client-ca.crt"))
        ),
    )
    yield SimpleNamespace(opener=opener, log=log)
    for s in (server, upstream):
        s.shutdown()
        s.server_close()


def _send(opener, url, body=None, method="POST"):
    request = urllib.request.Request(
        url, data=body, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with opener.open(request, timeout=10) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_codex_model_request_passes_the_proxy_unchanged(codex_stack):
    body = encoded(codex_request())
    status, reply = _send(codex_stack.opener, URL, body)
    assert status == 200 and b"response.completed" in reply
    assert _ChatGPT.received == [("POST", CODEX_MODEL_PATH, body)]


def test_hosted_search_never_reaches_chatgpt(codex_stack):
    body = encoded(codex_request(tools=[{"type": "web_search"}]))
    status, _ = _send(codex_stack.opener, URL, body)
    assert status == 403
    assert _ChatGPT.received == []
    entry = json.loads(codex_stack.log.read_text().splitlines()[-1])
    assert entry["rule"] == "native-model-body-hosted: a tool is not client-executed"
    assert entry["url"] == URL


def test_refusals_record_origin_and_path_never_the_query(codex_stack):
    _send(codex_stack.opener, URL + "?token=secret-value", encoded(codex_request()))
    entry = json.loads(codex_stack.log.read_text().splitlines()[-1])
    assert entry["url"] == URL and "secret-value" not in codex_stack.log.read_text()


def test_session_metadata_passes_bodiless_and_only_bodiless(codex_stack):
    models = "https://chatgpt.com/backend-api/codex/models?client_version=0.159.3"
    assert _send(codex_stack.opener, models, method="GET")[0] == 200
    assert _ChatGPT.received == [
        ("GET", "/backend-api/codex/models?client_version=0.159.3", b"")
    ]
    assert _send(codex_stack.opener, models, b"{}", method="GET")[0] == 403
    assert len(_ChatGPT.received) == 1


def test_other_chatgpt_paths_and_origins_are_refused(codex_stack):
    assert (
        _send(
            codex_stack.opener,
            "https://chatgpt.com/backend-api/ps/plugins/installed",
            method="GET",
        )[0]
        == 403
    )
    assert (
        _send(codex_stack.opener, "https://chatgpt.com/backend-api/wham/apps", b"{}")[0]
        == 403
    )
    with pytest.raises(urllib.error.URLError):
        codex_stack.opener.open("https://example.com/", timeout=10)
    assert _ChatGPT.received == []


# ------------------------------------------------------- upload and admission


@pytest.mark.asyncio
async def test_codex_policy_ships_the_guard_beside_the_proxy(monkeypatch):
    uploads: list[dict[str, bytes]] = []
    monkeypatch.setattr(egress_denylist, "_run", AsyncMock())
    monkeypatch.setattr(egress_denylist, "_wait_healthy", AsyncMock())

    async def upload(env, files):
        uploads.append(files)

    monkeypatch.setattr(egress_denylist, "_upload", upload)
    await egress_denylist.start_egress_denylist(
        object(), "agent", EgressDenylist((), (), native_codex_model_only=True)
    )
    files = uploads[-1]
    assert json.loads(files["policy.json"])["native_codex_model_only"] is True
    assert (
        files["request_guard.py"] == egress_denylist._REQUEST_GUARD_SCRIPT.read_bytes()
    )
    assert "certs/chatgpt.com.pem" in files
    await egress_denylist.start_egress_denylist(
        object(), "agent", EgressDenylist((), (), native_claude_model_only=True)
    )
    assert "request_guard.py" not in uploads[-1]
    assert "native_codex_model_only" not in json.loads(uploads[-1]["policy.json"])


def _env(uid="1000", found=None, code=0):
    found = {"acp": "2.0.1", "codex": "0.159.3"} if found is None else found

    async def run(cmd, **kwargs):
        if cmd.startswith("id -u"):
            return ExecResult(stdout=uid + "\n", stderr="", return_code=0)
        return ExecResult(stdout=json.dumps(found), stderr="", return_code=code)

    return SimpleNamespace(exec=run)


@pytest.mark.asyncio
async def test_codex_admission_records_the_pinned_adapter():
    receipt = await validate_native_codex_transport(_env(), "agent")
    assert receipt["origin"] == "https://chatgpt.com"
    assert receipt["paths"] == [CODEX_MODEL_PATH]
    assert receipt["versions"] == {"acp": "2.0.1", "codex": "0.159.3"}
    assert receipt["sandbox_uid"] == 1000


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "env,user",
    [
        (_env(uid="0"), "agent"),
        (_env(), None),
        (_env(found={"acp": "2.0.0", "codex": "0.159.3"}), "agent"),
        (_env(code=1), "agent"),
    ],
)
async def test_codex_admission_refuses_root_or_another_adapter(env, user):
    with pytest.raises(ValueError, match="Native Codex"):
        await validate_native_codex_transport(env, user)
