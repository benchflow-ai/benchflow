"""Guards the adaptation of PR #1209: deny-mode bodies and normal traffic."""

import gzip

# Re-export the real socket fixture for this module.
# ruff: noqa: F811
import json
import socket

import pytest

from benchflow.sandbox import _egress_denylist_proxy as proxy
from tests.test_egress_connect_authority import (  # noqa: F401
    _read_response,
    _tunnel,
    authority_stack,
)


@pytest.mark.parametrize(
    "body",
    [
        b"url=https%3A%2F%2Fpaper.test%2Fprotected",
        b'{"url":"https://paper.test/protected"}',
        b'{"model":"dummy","messages":[],"tools":[{"type":"web_search"}]}',
        b'{"model":"dummy","input":[{"type":"input_file","file_url":"https://paper.test/protected"}]}',
        b'{"model":"dummy","messages":[],"tools":[{"type":"web_search"}],"tools":[]}',
    ],
)
@pytest.mark.parametrize("chunked", [False, True])
def test_relay_body_never_completes_at_origin(authority_stack, body, chunked):
    with _tunnel(authority_stack) as tls:
        framing = (
            b"Transfer-Encoding: chunked"
            if chunked
            else f"Content-Length: {len(body)}".encode()
        )
        wire = (
            f"{len(body):x}\r\n".encode() + body + b"\r\n0\r\n\r\n" if chunked else body
        )
        tls.sendall(
            b"POST /relay HTTP/1.1\r\nHost: carrier.test\r\n"
            + framing
            + b"\r\n\r\n"
            + wire
        )
        assert _read_response(tls)[0] == 403
    assert not authority_stack.requests


@pytest.mark.parametrize(
    "body",
    [
        b'{"messages":[{"role":"user","content":"Read https://paper.test/protected"}],"model":"dummy"}',
        b'{"plugins":["ordinary app data"],"query":{"match_all":{}}}',
        b'[build]\nname="ordinary TOML"',
        b'{"url":"https://paper.test/allowed"}',
    ],
)
def test_ordinary_body_passes(authority_stack, body):
    with _tunnel(authority_stack) as tls:
        tls.sendall(
            b"POST /upload HTTP/1.1\r\nHost: carrier.test\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        status, _, response = _read_response(tls)
        assert status == 200
        assert json.loads(response)["body"] == body.decode()


@pytest.mark.parametrize("coding", ["gzip", "zstd"])
def test_coded_relay_is_refused(authority_stack, coding):
    body = gzip.compress(b'{"url":"https://paper.test/protected"}')
    with _tunnel(authority_stack) as tls:
        tls.sendall(
            b"POST /relay HTTP/1.1\r\nHost: carrier.test\r\n"
            + f"Content-Encoding: {coding}\r\nContent-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        assert _read_response(tls)[0] == 403
    assert not authority_stack.requests


def test_allowlist_does_not_gain_denylist_body_rules():
    policy = proxy.Policy([], [], allowed_hosts=["relay.test"])
    assert policy.url_rule("relay.test", "/?url=https://other.test/x") is None


@pytest.mark.parametrize(
    "body",
    [
        b'prefix\n{"model":"dummy","messages":[],"tools":[{"type":"web_search"}]}',
        b'{"model":"dummy","messages":[]} {"model":"dummy","messages":[],"tools":[{"type":"web_search"}]}',
        b'{"model":"dummy","input":[],"tools":[{"type":"code_interpreter","container":"existing"}]}',
    ],
)
def test_multiple_documents_and_text_lines_are_checked(body):
    policy = proxy.Policy(["https://paper.test/protected"], [])
    assert proxy.deny_body_rule(policy, body)


def test_split_text_url_never_completes(authority_stack):
    first, last = b"url=https://paper.test/pro", b"tected"
    with _tunnel(authority_stack) as tls:
        tls.sendall(
            b"POST /upload HTTP/1.1\r\nHost: carrier.test\r\n"
            + f"Content-Length: {len(first) + len(last)}\r\n\r\n".encode()
            + first
        )
        assert authority_stack.body_prefix_seen.wait(1)
        tls.sendall(last)
        assert _read_response(tls)[0] == 403
    assert not authority_stack.requests


@pytest.mark.parametrize(
    "data,headers",
    [
        (
            # mtime=0: the bytes are this test's id, which every xdist worker
            # must collect alike (gzip stamps the current second otherwise).
            gzip.compress(b'{"url":"https://paper.test/allowed"}', mtime=0),
            [("Content-Encoding", "gzip")],
        ),
        (b"\x00\xffbinary", [("Content-Type", "application/octet-stream")]),
    ],
)
def test_ordinary_coded_and_binary_bytes_are_preserved(data, headers):
    policy = proxy.Policy(["https://paper.test/protected"], [])
    with socket.socket() as sock:
        body = proxy._DenyBody(policy, sock, headers, data, len(data), False)
        assert body.check() is None
        sink = proxy._BodyBuffer(1024)
        body.send(sink)
        assert bytes(sink.data) == data


@pytest.mark.parametrize(
    "body",
    [
        b'{"model":"linear-regression","features":[1,2]}',
        b'{"query":"mutation($input: IssueInput!) { createIssue(input: $input) { id } }",'
        b'"variables":{"input":{"title":"t"}}}',
        b'{"channel":"C1","messages":[{"text":"hi"}]}',
    ],
)
def test_application_json_sharing_a_model_key_reaches_its_origin(authority_stack, body):
    """Data that shares one key name with a model request is not held to its schema."""
    with _tunnel(authority_stack) as tls:
        tls.sendall(
            b"POST /upload HTTP/1.1\r\nHost: carrier.test\r\n"
            + b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        status, _, response = _read_response(tls)
    assert status == 200
    assert json.loads(response)["body"] == body.decode()


def test_a_model_request_names_two_of_its_fields_or_a_model_path():
    policy = proxy.Policy(["https://paper.test/protected"], [])
    # Still model requests: hosted search, remote inputs and relay switches are refused.
    for body, path in (
        (b'{"messages":[],"tools":[{"type":"web_search"}]}', "/relay"),
        (b'{"contents":[],"tools":[{"google_search":{}}]}', "/relay"),
        (b'{"model":"m","messages":[],"plugins":[{"id":"web"}]}', "/relay"),
        (b'{"model":"m","tools":[{"type":"web_search_20250305"}]}', "/relay"),
        (
            b'{"contents":[{"parts":[{"text":"x"}]}],"x":1}',
            "/v1beta/models/m:generateContent",
        ),
    ):
        assert proxy.deny_body_rule(policy, body, path), body
    # Data is scanned as data: a blocked address in it is refused, nothing else.
    assert proxy.deny_body_rule(
        policy, b'{"model":"x","note":"https://paper.test/protected"}'
    )
    assert (
        proxy.deny_body_rule(policy, b'{"model":"x","features":[1,2]}', "/upload")
        is None
    )
    # A model request earlier in the body does not make later data a model request.
    assert (
        proxy.deny_body_rule(
            policy, b'[{"model":"m","messages":[]},{"input":{"title":"t"}}]', "/upload"
        )
        is None
    )
