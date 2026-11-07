"""PR #1209 public body cases, adapted to the existing request guard.

Refusal reason strings differ because this branch uses a positive model schema.
"""

# ruff: noqa: F811
import gzip
import json

import pytest

from benchflow.sandbox import _egress_denylist_proxy as body_mod
from tests.test_egress_connect_authority import (  # noqa: F401
    _get,
    _read_response,
    _tunnel,
    authority_stack,
)


def _same_rule(actual, expected):
    if expected.startswith("hosted-fetch:") or expected == "url:paper.test/protected":
        return actual.startswith(
            ("deny-model-body:", "uninspectable-body", "url:paper.test/protected")
        )
    return actual == expected


def _post(body: bytes, *, path="/v1/messages", headers=""):
    return (
        f"POST {path} HTTP/1.1\r\nHost: carrier.test\r\n{headers}"
        f"Content-Length: {len(body)}\r\n\r\n"
    ).encode() + body


_HOSTED_SEARCH = json.dumps(
    {"model": "m", "tools": [{"type": "web_search_20250305", "name": "web_search"}]}
).encode()
_ORDINARY = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]})


def test_model_request_for_a_hosted_fetch_is_refused_and_logged(authority_stack):
    """Guards the denylist (PR #1113) against hosted fetches sent past its URL rules."""
    with _tunnel(authority_stack) as tls:
        tls.sendall(_post(_HOSTED_SEARCH))
        code, _, body = _read_response(tls)
    assert code == 403
    assert b"Blocked by the task network policy" in body
    entry = json.loads(authority_stack.log.read_text().splitlines()[-1])
    assert _same_rule(entry["rule"], "hosted-fetch:tool:web_search_20250305")
    assert entry["url"] == "https://carrier.test/v1/messages"
    assert authority_stack.requests == []


def test_ordinary_chunked_json_reaches_origin_whole(authority_stack):
    """The checked JSON body is forwarded unchanged, framed by its length."""
    first, second = _ORDINARY[:10].encode(), _ORDINARY[10:].encode()
    with _tunnel(authority_stack) as tls:
        tls.sendall(
            b"POST /v1/messages HTTP/1.1\r\nHost: carrier.test\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
            + f"{len(first):x}\r\n".encode()
            + first
            + f"\r\n{len(second):x}\r\n".encode()
            + second
            + b"\r\n0\r\n\r\n"
        )
        code, _, body = _read_response(tls)
    assert code == 200
    assert json.loads(body)["body"] == _ORDINARY


def test_gzip_coded_model_requests_are_decoded_before_the_check(authority_stack):
    blocked, ordinary = gzip.compress(_HOSTED_SEARCH), gzip.compress(_ORDINARY.encode())
    with _tunnel(authority_stack) as tls:
        tls.sendall(_post(blocked, headers="Content-Encoding: gzip\r\n"))
        assert _read_response(tls)[0] == 403
    with _tunnel(authority_stack) as tls:
        tls.sendall(_post(ordinary, headers="Content-Encoding: gzip\r\n"))
        code, _, body = _read_response(tls)
    assert code == 200
    assert json.loads(body)["body"].encode("latin-1") == ordinary


@pytest.mark.parametrize(
    ("headers", "body", "rule"),
    [
        ("Content-Encoding: br\r\n", b"\x1b\x03\x00", "uninspectable-body"),
        ("Content-Encoding: gzip\r\n", b"not gzip", "uninspectable-body"),
        ("", b"[" * 100_000 + b"]" * 100_000, "uninspectable-body"),
    ],
    ids=["unknown-coding", "corrupt-gzip", "nesting-deeper-than-the-parser"],
)
def test_body_the_proxy_cannot_read_is_refused(authority_stack, headers, body, rule):
    with _tunnel(authority_stack) as tls:
        tls.sendall(_post(body, headers=headers))
        assert _read_response(tls)[0] == 403
    assert _same_rule(json.loads(authority_stack.log.read_text())["rule"], rule)
    assert authority_stack.requests == []


def test_oversized_json_is_refused_while_other_large_bodies_stream(
    authority_stack, monkeypatch
):
    monkeypatch.setattr(body_mod, "MODEL_BODY_LIMIT", 64)
    with _tunnel(authority_stack) as tls:
        tls.sendall(_post(b'{"input": "' + b"x" * 200 + b'"}'))
        assert _read_response(tls)[0] == 403
    assert json.loads(authority_stack.log.read_text())["rule"] == "oversized-body"
    upload = b"PK" + b"y" * 200
    with _tunnel(authority_stack) as tls:
        tls.sendall(_post(upload, path="/upload"))
        code, _, body = _read_response(tls)
    assert code == 200
    assert json.loads(body)["body"].encode("latin-1") == upload


def test_listed_page_named_inside_another_url_is_refused(authority_stack):
    """Guards the denylist (PR #1113) against reader and archive services."""
    with _tunnel(authority_stack) as tls:
        tls.sendall(_get(path="/https://paper.test/protected?via=reader"))
        assert _read_response(tls)[0] == 403
    assert (
        json.loads(authority_stack.log.read_text())["rule"]
        == "url:paper.test/protected"
    )
    assert authority_stack.requests == []


@pytest.mark.parametrize(
    ("headers", "body", "rule"),
    [
        (
            "",
            b'{"input": "\xff", "tools": [{"type": "web_search"}]}',
            "hosted-fetch:tool:web_search",
        ),
        (
            "",
            json.dumps(
                {"model": "m", "request": {"tools": [{"urlContext": {}}]}}
            ).encode("utf-16-be"),
            "hosted-fetch:tool:urlContext",
        ),
        ("Content-Encoding: zstd\r\n", b"\x28\xb5\x2f\xfd", "uninspectable-body"),
        # Gemini merges a repeated field into one working tool list.
        (
            "",
            b'{"contents":[],"tools":[{"urlContext":{}}],"tools":[]}',
            "hosted-fetch:tool:urlContext",
        ),
        # A batch file uploaded as JSON Lines: every line is a request.
        (
            "",
            b'{"key":"1","request":{}}\n'
            + json.dumps(
                {
                    "key": "2",
                    "request": {"contents": [], "tools": [{"google_search": {}}]},
                }
            ).encode(),
            "hosted-fetch:tool:google_search",
        ),
        # Azure OpenAI fetches a file_url written without the slashes.
        (
            "",
            json.dumps(
                {
                    "input": [
                        {
                            "content": [
                                {
                                    "type": "input_file",
                                    "file_url": "https:paper.test/protected",
                                }
                            ]
                        }
                    ]
                }
            ).encode(),
            "hosted-fetch:input:input_file",
        ),
    ],
    ids=[
        "invalid-utf8",
        "utf16-code-assist-envelope",
        "zstd",
        "repeated-name",
        "json-lines-batch",
        "address-without-slashes",
    ],
)
def test_model_requests_are_read_as_leniently_as_servers_read_them(
    authority_stack, headers, body, rule
):
    """Guards the denylist body check against encodings a provider would still parse."""
    with _tunnel(authority_stack) as tls:
        tls.sendall(_post(body, headers=headers))
        assert _read_response(tls)[0] == 403
    assert _same_rule(json.loads(authority_stack.log.read_text())["rule"], rule)
    assert authority_stack.requests == []


def test_coded_upload_that_is_not_json_is_forwarded(authority_stack):
    """The proxy decodes before judging, so a compressed file upload still goes through."""
    upload = gzip.compress(b"PK\x03\x04 not a model request")
    with _tunnel(authority_stack) as tls:
        tls.sendall(_post(upload, path="/upload", headers="Content-Encoding: gzip\r\n"))
        code, _, body = _read_response(tls)
    assert code == 200
    assert json.loads(body)["body"].encode("latin-1") == upload


def test_ordinary_json_lines_upload_is_forwarded(authority_stack):
    """Every line of a JSON Lines body is judged, and an ordinary dataset still goes through."""
    upload = b"".join(
        json.dumps(
            {"id": i, "text": f"see https://example.test/{i}", "tags": ["a"]}
        ).encode()
        + b"\n"
        for i in range(50)
    )
    with _tunnel(authority_stack) as tls:
        tls.sendall(_post(upload, path="/upload"))
        code, _, body = _read_response(tls)
    assert code == 200
    assert json.loads(body)["body"].encode("latin-1") == upload


def _post_typed(body: bytes, content_type: str, *, path="/submit", chunks=None):
    if chunks is None:
        return _post(body, path=path, headers=f"Content-Type: {content_type}\r\n")
    framed = b"".join(f"{len(c):x}\r\n".encode() + c + b"\r\n" for c in chunks)
    return (
        (
            f"POST {path} HTTP/1.1\r\nHost: carrier.test\r\nContent-Type: {content_type}\r\n"
            "Transfer-Encoding: chunked\r\n\r\n"
        ).encode()
        + framed
        + b"0\r\n\r\n"
    )


@pytest.mark.parametrize(
    ("request_bytes", "rule"),
    [
        # A reader service takes the address in a form field.
        (
            _post_typed(
                b"url=https%3A%2F%2Fpaper.test%2Fprotected&format=text",
                "application/x-www-form-urlencoded",
            ),
            "url:paper.test/protected",
        ),
        # ...or in a JSON body, which a model request field cannot hide.
        (
            _post(
                json.dumps(
                    {"model": "x", "url": "https://paper.test/protected"}
                ).encode()
            ),
            "url:paper.test/protected",
        ),
        (
            _post(json.dumps({"q": "paper.test/protected", "n": 3}).encode()),
            "url:paper.test/protected",
        ),
        # A streamed body split mid-address is checked across the split.
        (
            _post_typed(
                b"",
                "text/plain",
                chunks=[b"see https://paper.te", b"st/protected please"],
            ),
            "url:paper.test/protected",
        ),
        # A JSON Lines batch file in a text upload that does not open as JSON.
        (
            _post_typed(
                b"0\n"
                + json.dumps(
                    {"request": {"contents": [], "tools": [{"google_search": {}}]}}
                ).encode()
                + b"\n",
                "application/jsonl",
            ),
            "hosted-fetch:tool:google_search",
        ),
    ],
    ids=[
        "form-field",
        "json-url-field",
        "json-search-query",
        "split-chunks",
        "json-lines-upload",
    ],
)
def test_bodies_that_hand_a_service_a_listed_page_are_refused(
    authority_stack, request_bytes, rule
):
    """Guards the denylist (PR #1113) against reader services handed the address in a body."""
    with _tunnel(authority_stack) as tls:
        tls.sendall(request_bytes)
        assert _read_response(tls)[0] == 403
    assert _same_rule(
        json.loads(authority_stack.log.read_text().splitlines()[-1])["rule"], rule
    )
    assert authority_stack.requests == []


@pytest.mark.parametrize(
    "request_bytes",
    [
        # A model request may discuss the listed page; its text is conversation.
        _post(
            json.dumps(
                {
                    "model": "m",
                    "messages": [
                        {
                            "role": "user",
                            "content": "curl https://paper.test/protected gave 403",
                        }
                    ],
                }
            ).encode()
        ),
        # Binary uploads are not read as text.
        _post_typed(
            b"PK\x03\x04 https://paper.test/protected", "application/octet-stream"
        ),
        _post_typed(b"name=paper&value=test", "application/x-www-form-urlencoded"),
    ],
    ids=["conversation", "binary-upload", "ordinary-form"],
)
def test_bodies_that_only_mention_or_carry_bytes_pass(authority_stack, request_bytes):
    with _tunnel(authority_stack) as tls:
        tls.sendall(request_bytes)
        assert _read_response(tls)[0] == 200
    assert len(authority_stack.requests) == 1


@pytest.mark.parametrize(
    "upload",
    [b'[project]\nname = "x"\n', b"[2026-10-29 10:00:00] INFO started\n"],
    ids=["toml", "log"],
)
def test_text_that_opens_like_an_array_is_forwarded(authority_stack, upload):
    """No provider takes a top-level array, so TOML and log uploads are text, not requests."""
    with _tunnel(authority_stack) as tls:
        tls.sendall(_post_typed(upload, "text/plain", path="/upload"))
        code, _, body = _read_response(tls)
    assert code == 200
    assert json.loads(body)["body"].encode("latin-1") == upload
