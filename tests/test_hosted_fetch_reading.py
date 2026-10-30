"""How a request body is read before the hosted-fetch rules judge it.

Guards the denylist egress mode from PR #1113: a body is read the way the most
lenient server reads it (codings, encodings, every JSON value and JSON Lines
line, every value of a repeated name), and the read stays linear, because it
runs on every request the agent sends.
"""

from __future__ import annotations

import gzip
import json
import time

import pytest

from benchflow.sandbox._hosted_fetch import (
    model_request_rule,
    named_addresses,
    read_body,
)

PAPER = "https://arxiv.org/abs/2401.12345"
_HOSTED = {"model": "m", "tools": [{"type": "web_search"}]}


def hosted_fetch_rule(request) -> str | None:
    """The rule a request breaks, sent as a JSON body."""
    return model_request_rule(json.dumps(request).encode())


def _file_part(url: str) -> dict:
    return {
        "model": "m",
        "input": [
            {"role": "user", "content": [{"type": "input_file", "file_url": url}]}
        ],
    }


class TestModelRequestRule:
    """Bodies are read the way a lenient server reads them, and refused when they cannot be."""

    @pytest.mark.parametrize(
        "body",
        [
            json.dumps(_HOSTED).encode(),
            json.dumps(_HOSTED).encode() + b" trailing data",
            b'{"input": "\xff\xfe", "tools": [{"type": "web_search"}]}',
            json.dumps(_HOSTED).encode("utf-16"),
            json.dumps(_HOSTED).encode("utf-16-be"),
            json.dumps(_HOSTED).encode("utf-32-le"),
            b"\xef\xbb\xbf  " + json.dumps(_HOSTED).encode(),
        ],
        ids=[
            "plain",
            "trailing-data",
            "invalid-utf8",
            "utf16-bom",
            "utf16-be",
            "utf32-le",
            "utf8-bom",
        ],
    )
    def test_hosted_fetch_is_found_however_the_body_is_encoded(self, body):
        assert model_request_rule(body) == "hosted-fetch:tool:web_search"

    def test_every_gzip_member_is_read(self):
        text = json.dumps(_HOSTED).encode()
        half = len(text) // 2
        body = gzip.compress(text[:half]) + gzip.compress(text[half:])
        assert model_request_rule(body, ["gzip"]) == "hosted-fetch:tool:web_search"

    @pytest.mark.parametrize(
        ("body", "codings"),
        [
            (b"\x28\xb5\x2f\xfd", ["zstd"]),
            (b"not gzip", ["gzip"]),
            (gzip.compress(b"{}")[:-4], ["gzip"]),
            (b"[" * 100_000 + b"]" * 100_000, []),
            (b"{not json", []),
            (
                b"".join(gzip.compress(b" ") for _ in range(17)) + gzip.compress(b"{}"),
                ["gzip"],
            ),
        ],
        ids=[
            "zstd",
            "corrupt-gzip",
            "truncated-gzip",
            "too-deep",
            "not-json",
            "many-members",
        ],
    )
    def test_unreadable_bodies_are_refused(self, body, codings):
        assert model_request_rule(body, codings) == "uninspectable-body"

    def test_ordinary_and_empty_bodies_pass(self):
        assert model_request_rule(b"") is None
        assert model_request_rule(json.dumps({"model": "m"}).encode()) is None

    def test_a_blank_body_is_refused(self):
        assert model_request_rule(b" \r\n ") == "uninspectable-body"

    @pytest.mark.parametrize(
        "body",
        [
            # Gemini merges a repeated field; others keep the first or the last.
            b'{"contents":[],"tools":[{"urlContext":{}}],"tools":[]}',
            b'{"contents":[],"tools":[],"tools":[{"urlContext":{}}]}',
            b'{"model":"m","tools":[{"type":"web_search","type":"function","name":"f"}]}',
            b'{"model":"m","web_search_options":{},"web_search_options":null}',
            b'{"model":"m","tools":[],"Tools":[{"type":"web_search"}]}',
        ],
        ids=["first", "last", "in-a-tool", "switch", "folded"],
    )
    def test_each_value_of_a_repeated_name_is_judged(self, body):
        assert model_request_rule(body).startswith("hosted-fetch:")

    def test_a_repeated_part_type_is_judged_per_value(self):
        part = (
            b'{"type":"input_file","type":"input_text","text":"x","file_url":"'
            + PAPER.encode()
            + b'"}'
        )
        body = b'{"model":"m","input":[{"role":"user","content":[' + part + b"]}]}"
        assert model_request_rule(body) == "hosted-fetch:input:input_file"

    @pytest.mark.parametrize(
        "body",
        [
            json.dumps({"model": "m"}).encode() + json.dumps(_HOSTED).encode(),
            json.dumps({"model": "m"}).encode() + b"\n" + json.dumps(_HOSTED).encode(),
            # A batch file a JSON Lines reader takes past a line it cannot parse.
            b'{"key":"1","request":{}}\n{broken\n'
            + json.dumps({"key": "2", "request": _HOSTED}).encode(),
        ],
        ids=["concatenated", "json-lines", "past-a-broken-line"],
    )
    def test_every_value_in_a_body_is_judged(self, body):
        assert model_request_rule(body) == "hosted-fetch:tool:web_search"


class TestCost:
    """The check stays linear: it runs on every request the agent sends."""

    def _seconds(self, body: bytes) -> float:
        started = time.perf_counter()
        model_request_rule(body)
        return time.perf_counter() - started

    def test_nested_repeated_names_stay_linear(self):
        body = b'{"model":"m","x":0}'
        for _ in range(40):
            body = b'{"a":0,"a":0,"b":' + body + b"}"
        assert self._seconds(body) < 0.5

    def test_many_repeated_names_stay_linear(self):
        pairs = [f'"k{i}":0' for i in range(6000)] + ['"t":0'] * 6000
        assert self._seconds(("{" + ",".join(pairs) + "}").encode()) < 1.0

    def test_lines_that_never_close_are_read_once(self):
        body = b'{"model":"m"}\n' + b"[1,\n" * 20000
        assert self._seconds(body) < 2.0

    def test_a_large_data_url_is_not_read_through(self):
        image = "data:image/png;base64," + "A" * 10_000_000
        body = json.dumps(
            {"model": "m", "input": [{"type": "input_image", "image_url": image}]}
        ).encode()
        assert model_request_rule(body) is None
        assert self._seconds(body) < 1.0

    def test_deeply_nested_tool_groups_are_refused_not_raised(self):
        tools: dict = {"type": "function", "name": "f"}
        for _ in range(3000):
            tools = {"type": "namespace", "tools": [tools]}
        body = '{"model":"m","tools":[' + json.dumps(tools)[: 10**9] + "]}"
        assert model_request_rule(body.encode()) in (None, "uninspectable-body")

    def test_a_scheme_padded_with_tabs_is_read(self):
        url = "h" + "\t" * 300 + "ttps://arxiv.org/pdf/2401.12345"
        assert hosted_fetch_rule(_file_part(url)) == "hosted-fetch:input:input_file"


class TestNamedAddresses:
    def test_a_request_offers_only_its_address_fields(self):
        request = {
            "model": "m",
            "url": "https://a.test/x",
            "messages": [{"role": "user", "content": "https://b.test/y"}],
        }
        assert list(named_addresses(read_body(json.dumps(request).encode()))) == [
            "https://a.test/x"
        ]

    def test_other_json_offers_every_string(self):
        body = {"q": "c.test/z", "list": ["d.test/w", {"deep": "e.test/v"}], "n": 3}
        found = set(named_addresses(read_body(json.dumps(body).encode())))
        assert found == {"c.test/z", "d.test/w", "e.test/v"}
