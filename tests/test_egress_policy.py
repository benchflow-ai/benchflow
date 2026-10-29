"""Egress policy rules: listed pages named inside other URLs, and model requests for hosted fetches.

Guards the denylist egress mode from PR #1113 against two routes around it:
a reader, archive or redirect service handed a blocked address, and a model
request that has the provider fetch or search the web (the "universal offline
sandbox escape"), whose pages never cross the sandbox network.
"""

from __future__ import annotations

import gzip
import json

import pytest

from benchflow.sandbox._egress_policy import (
    Policy,
    hosted_fetch_rule,
    model_request_rule,
)

PAPER = "https://arxiv.org/abs/2401.12345"
_HOSTED = {"model": "m", "tools": [{"type": "web_search"}]}


class TestEmbeddedUrls:
    @pytest.mark.parametrize(
        ("host", "target"),
        [
            ("r.jina.ai", "/https://arxiv.org/abs/2401.12345"),
            ("web.archive.org", "/web/2026id_/https://arxiv.org/abs/2401.12345v2"),
            ("web.archive.org", "/web/2026*/arxiv.org/abs/2401.12345"),
            (
                "translate.google.com",
                "/translate?u=https://www.arxiv.org/abs/2401.12345",
            ),
            ("redirect.test", "/go?to=https%3A%2F%2Farxiv.org%2Fabs%2F2401.12345&x=1"),
            (
                "redirect.test",
                "/go?to=https%253A%252F%252Farxiv.org%252Fabs%252F2401.12345",
            ),
            ("reader.test", "/https://arxiv.org:443/abs/../abs/2401.12345"),
            ("search.test", "/search?q=arxiv.org/ABS/2401.12345"),
            ("r.jina.ai", "/https://arxiv.org./abs/2401.12345"),
            ("r.jina.ai", "/https://arxiv\u3002org/abs/2401.12345"),
            ("r.jina.ai", "/https://\uff41rxiv\uff0eorg/abs/2401.12345"),
            ("r.jina.ai", "/https://arxiv%EF%BC%8Eorg/abs/2401.12345"),
            ("r.jina.ai", "/https://ar%C2%ADxiv.org/abs/2401.12345"),
            ("r.jina.ai", "/https://ar%E2%80%8Bxiv.org/abs/2401.12345"),
            ("r.jina.ai", "/https://arxiv.o%E2%81%A0rg/abs/2401.12345"),
            ("r.jina.ai", "/https://ar%09xiv.org/abs/2401.12345"),
            ("r.jina.ai", "/https://arxiv.org/ab%0As/2401.12345"),
            ("r.jina.ai", "/https://arx%0Div.org/abs/2401.12345"),
        ],
    )
    def test_listed_page_inside_another_url_is_refused(self, host, target):
        rule = Policy([PAPER], []).url_rule(host, target)
        assert rule == "url:arxiv.org/abs/2401.12345"

    def test_listed_host_inside_another_url_covers_subdomains(self):
        policy = Policy([], ["alphaxiv.org"])
        assert (
            policy.url_rule("r.jina.ai", "/https://www.alphaxiv.org/abs/2401.12345")
            == "host:alphaxiv.org"
        )

    @pytest.mark.parametrize(
        ("host", "target"),
        [
            ("r.jina.ai", "/https://arxiv.org/abs/2401.12346"),
            ("r.jina.ai", "/https://arxiv.org/abs/1706.03762"),
            ("r.jina.ai", "/https://notarxiv.org/abs/2401.12345"),
            ("r.jina.ai", "/https://arxiv.org.mirror.test/abs/2401.12345"),
            ("files.test", "/numpy-1.26.4.tar.gz?v=2401.12345"),
            ("search.test", "/search?q=2401.12345"),
        ],
    )
    def test_other_addresses_inside_urls_stay_reachable(self, host, target):
        assert Policy([PAPER], ["alphaxiv.org"]).url_rule(host, target) is None


class TestHostedFetchRule:
    @pytest.mark.parametrize(
        ("request_body", "feature"),
        [
            # OpenAI Responses and chat completions
            ({"tools": [{"type": "web_search"}]}, "tool:web_search"),
            ({"tools": [{"type": "web_search_preview"}]}, "tool:web_search_preview"),
            (
                {"tools": [{"type": "mcp", "server_url": "https://mcp.test"}]},
                "tool:mcp",
            ),
            ({"web_search_options": {}}, "web_search_options"),
            (
                {
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_file", "file_url": f"{PAPER}.pdf"}
                            ],
                        }
                    ]
                },
                "input:input_file",
            ),
            (
                {
                    "input": [
                        {"type": "input_image", "image_url": "https://x.test/a.png"}
                    ]
                },
                "input:input_image",
            ),
            (
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "HTTPS://x.test/a.png"},
                                }
                            ],
                        }
                    ]
                },
                "input:image_url",
            ),
            (
                {
                    "messages": [
                        {"content": [{"type": "file", "file": {"file_id": PAPER}}]}
                    ]
                },
                "input:file",
            ),
            # Anthropic Messages
            (
                {"tools": [{"type": "web_search_20250305", "name": "web_search"}]},
                "tool:web_search_20250305",
            ),
            (
                {"tools": [{"type": "web_fetch_20250910", "name": "web_fetch"}]},
                "tool:web_fetch_20250910",
            ),
            (
                {"mcp_servers": [{"type": "url", "url": "https://mcp.test"}]},
                "mcp_servers",
            ),
            (
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "content": [
                                        {
                                            "type": "document",
                                            "source": {"type": "url", "url": PAPER},
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                },
                "input:document",
            ),
            # Gemini GenerateContent
            ({"tools": [{"url_context": {}}]}, "tool:url_context"),
            # Google's JSON parser takes one object where a list is expected.
            ({"tools": {"urlContext": {}}}, "tool:urlContext"),
            (
                {
                    "messages": [
                        {
                            "content": [
                                {"type": "input_file", "file_url": "\x01" + PAPER}
                            ]
                        }
                    ]
                },
                "input:input_file",
            ),
            ({"tools": [{"googleSearch": {}}]}, "tool:googleSearch"),
            (
                {"tools": [{"google_search_retrieval": {}}]},
                "tool:google_search_retrieval",
            ),
            (
                {"contents": [{"parts": [{"fileData": {"fileUri": f"{PAPER}.pdf"}}]}]},
                "input:file_data",
            ),
            # xAI, DashScope, Moonshot, Azure OpenAI
            ({"tools": [{"type": "x_search"}]}, "tool:x_search"),
            ({"search_parameters": {"mode": "on"}}, "search_parameters"),
            ({"enable_search": True}, "enable_search"),
            (
                {
                    "tools": [
                        {
                            "type": "builtin_function",
                            "function": {"name": "$web_search"},
                        }
                    ]
                },
                "tool:builtin_function",
            ),
            ({"data_sources": [{"type": "azure_search"}]}, "data_sources"),
            # LiteLLM merges extra_body into the provider request.
            ({"extra_body": {"tools": [{"type": "web_search"}]}}, "extra_body"),
            # A client-chosen type at the root does not hide the request input.
            (
                {
                    "type": "tool_use",
                    "input": [{"content": [{"type": "input_file", "file_url": PAPER}]}],
                },
                "input:input_file",
            ),
            # Any address in a file part: LiteLLM fetches file_id and file_data URLs.
            ({"input": [{"type": "input_file", "file_id": PAPER}]}, "input:input_file"),
            (
                {"input": [{"type": "input_file", "file_data": f"{PAPER}.pdf"}]},
                "input:input_file",
            ),
            (
                {"contents": [{"parts": [{"file_data": {"fileUri": PAPER}}]}]},
                "input:file_data",
            ),
            # DashScope's native API nests its switch under "parameters".
            ({"model": "qwen", "parameters": {"enable_search": True}}, "enable_search"),
            # Gemini function responses can carry parts the provider fetches.
            (
                {
                    "contents": [
                        {
                            "parts": [
                                {
                                    "functionResponse": {
                                        "parts": [{"fileData": {"fileUri": PAPER}}]
                                    }
                                }
                            ]
                        }
                    ]
                },
                "input:file_data",
            ),
            # Envelopes: Gemini Code Assist, Gemini and Anthropic batches.
            (
                {"project": "p", "request": {"tools": [{"urlContext": {}}]}},
                "tool:urlContext",
            ),
            (
                {
                    "batch": {
                        "input_config": {
                            "requests": {
                                "requests": [
                                    {"request": {"tools": [{"google_search": {}}]}}
                                ]
                            }
                        }
                    }
                },
                "tool:google_search",
            ),
            (
                {"requests": [{"params": {"tools": [{"type": "web_fetch_20250910"}]}}]},
                "tool:web_fetch_20250910",
            ),
            # OpenRouter's web and file-parser plugins.
            ({"plugins": [{"id": "web"}]}, "plugins"),
            # Vertex reads gs:// objects itself, as LiteLLM does for some providers.
            (
                {
                    "contents": [
                        {"parts": [{"fileData": {"fileUri": "gs://arxiv/x.pdf"}}]}
                    ]
                },
                "input:file_data",
            ),
            (
                {
                    "messages": [
                        {
                            "content": [
                                {"type": "image_url", "image_url": "s3://b/x.png"}
                            ]
                        }
                    ]
                },
                "input:image_url",
            ),
        ],
    )
    def test_provider_side_fetches_are_named(self, request_body, feature):
        assert hosted_fetch_rule(request_body) == f"hosted-fetch:{feature}"

    @pytest.mark.parametrize(
        "request_body",
        [
            {"messages": [{"role": "user", "content": f"Summarize {PAPER}"}]},
            {
                "tools": [
                    {"type": "function", "function": {"name": "shell"}},
                    {"type": "custom", "name": "apply_patch"},
                    {"type": "namespace", "namespace": {"name": "codex"}},
                    {"name": "WebFetch", "input_schema": {"type": "object"}},
                    {"functionDeclarations": [{"name": "read_file"}]},
                    {"codeExecution": {}},
                ]
            },
            # A client tool may take an argument named like a URL field.
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "fetch_image",
                            "parameters": {
                                "type": "object",
                                "properties": {"image_url": {"type": "string"}},
                            },
                        },
                    }
                ],
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "fetch_image",
                                "input": {"image_url": "https://x.test/a.png"},
                            }
                        ],
                    }
                ],
            },
            {
                "messages": [
                    {
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,iVBORw0KGgo="
                                },
                            },
                            {
                                "type": "image",
                                "source": {"type": "base64", "data": "iVBORw0KGgo="},
                            },
                        ]
                    }
                ]
            },
            {
                "contents": [
                    {
                        "parts": [
                            {
                                "fileData": {
                                    "fileUri": "https://generativelanguage.googleapis.com/v1beta/files/abc"
                                }
                            },
                            {"inlineData": {"data": "iVBORw0KGgo="}},
                        ]
                    }
                ]
            },
            {"enable_search": False, "web_search_options": None, "mcp_servers": []},
            {"tools": "auto"},
            {"tools": 3},
            # Arguments of a tool call already made are the model's own output.
            {
                "messages": [
                    {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "attach",
                                "input": {
                                    "type": "image_url",
                                    "image_url": {"url": "https://x.test/a.png"},
                                },
                            }
                        ]
                    }
                ]
            },
            {
                "contents": [
                    {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "fetch",
                                    "args": {
                                        "fileData": {"fileUri": "https://x.test/a"}
                                    },
                                }
                            }
                        ]
                    }
                ]
            },
            # Tool results are data, whatever keys they hold.
            {
                "contents": [
                    {
                        "parts": [
                            {
                                "functionResponse": {
                                    "response": {
                                        "plugins": ["a"],
                                        "enable_search": True,
                                    }
                                }
                            }
                        ]
                    }
                ]
            },
            # So is a structured-output schema.
            {
                "text": {
                    "format": {
                        "type": "json_schema",
                        "schema": {
                            "type": "object",
                            "properties": {"plugins": {"type": "array"}},
                        },
                    }
                }
            },
            # A tool schema may name a property after a request field.
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search",
                            "parameters": {
                                "type": "object",
                                "properties": {"enable_search": {"type": "boolean"}},
                            },
                        },
                    }
                ]
            },
            "not a request",
        ],
    )
    def test_ordinary_requests_pass(self, request_body):
        assert hosted_fetch_rule(request_body) is None


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
