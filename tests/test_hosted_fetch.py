"""Model requests that have the provider fetch or search the web for the agent.

Guards the denylist egress mode from PR #1113 against the "universal offline
sandbox escape": a model request that asks the provider to fetch or search the
web, whose pages never cross the sandbox network. Spellings, repeated names and
address forms come from what real providers act on: Azure OpenAI fetched a
``file_url`` written ``https:host`` or with a tab in its scheme, and Gemini
merged a repeated ``tools`` key into a working URL context tool.
"""

from __future__ import annotations

import json

import pytest

from benchflow.sandbox._hosted_fetch import model_request_rule

PAPER = "https://arxiv.org/abs/2401.12345"
_HOSTED = {"model": "m", "tools": [{"type": "web_search"}]}


def hosted_fetch_rule(request) -> str | None:
    """The rule a request breaks, sent as a JSON body."""
    return model_request_rule(json.dumps(request).encode())


class TestHostedFetchRule:
    @pytest.mark.parametrize(
        ("request_body", "feature"),
        [
            # OpenAI Responses and chat completions
            ({"model": "m", "tools": [{"type": "web_search"}]}, "tool:web_search"),
            (
                {"model": "m", "tools": [{"type": "web_search_preview"}]},
                "tool:web_search_preview",
            ),
            (
                {
                    "model": "m",
                    "tools": [{"type": "mcp", "server_url": "https://mcp.test"}],
                },
                "tool:mcp",
            ),
            ({"model": "m", "web_search_options": {}}, "web_search_options"),
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
                {
                    "model": "m",
                    "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                },
                "tool:web_search_20250305",
            ),
            (
                {
                    "model": "m",
                    "tools": [{"type": "web_fetch_20250910", "name": "web_fetch"}],
                },
                "tool:web_fetch_20250910",
            ),
            (
                {
                    "model": "m",
                    "mcp_servers": [{"type": "url", "url": "https://mcp.test"}],
                },
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
            ({"contents": [], "tools": [{"url_context": {}}]}, "tool:url_context"),
            # Google's JSON parser takes one object where a list is expected.
            ({"contents": [], "tools": {"urlContext": {}}}, "tool:urlContext"),
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
            ({"contents": [], "tools": [{"googleSearch": {}}]}, "tool:googleSearch"),
            (
                {"contents": [], "tools": [{"google_search_retrieval": {}}]},
                "tool:google_search_retrieval",
            ),
            (
                {"contents": [{"parts": [{"fileData": {"fileUri": f"{PAPER}.pdf"}}]}]},
                "input:file_data",
            ),
            # xAI, DashScope, Moonshot, Azure OpenAI
            ({"model": "m", "tools": [{"type": "x_search"}]}, "tool:x_search"),
            ({"model": "m", "search_parameters": {"mode": "on"}}, "search_parameters"),
            ({"model": "m", "enable_search": True}, "enable_search"),
            (
                {
                    "model": "m",
                    "tools": [
                        {
                            "type": "builtin_function",
                            "function": {"name": "$web_search"},
                        }
                    ],
                },
                "tool:builtin_function",
            ),
            (
                {"messages": [], "data_sources": [{"type": "azure_search"}]},
                "data_sources",
            ),
            # LiteLLM merges extra_body into the provider request.
            (
                {"model": "m", "extra_body": {"tools": [{"type": "web_search"}]}},
                "extra_body",
            ),
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
                {
                    "model": "m",
                    "project": "p",
                    "request": {"tools": [{"urlContext": {}}]},
                },
                "tool:urlContext",
            ),
            (
                {
                    "batch": {
                        "input_config": {
                            "requests": {
                                "requests": [
                                    {
                                        "request": {
                                            "contents": [],
                                            "tools": [{"google_search": {}}],
                                        }
                                    }
                                ]
                            }
                        }
                    }
                },
                "tool:google_search",
            ),
            (
                {
                    "requests": [
                        {
                            "params": {
                                "model": "m",
                                "tools": [{"type": "web_fetch_20250910"}],
                            }
                        }
                    ]
                },
                "tool:web_fetch_20250910",
            ),
            # OpenRouter's web and file-parser plugins.
            ({"model": "m", "plugins": [{"id": "web"}]}, "plugins"),
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


def _file_part(url: str) -> dict:
    return {
        "model": "m",
        "input": [
            {"role": "user", "content": [{"type": "input_file", "file_url": url}]}
        ],
    }


class TestSpellings:
    """Names are compared folded: some servers ignore case, lenient decoders drop bytes."""

    @pytest.mark.parametrize(
        ("request_body", "feature"),
        [
            ({"model": "m", "Tools": [{"type": "web_search"}]}, "tool:web_search"),
            ({"model": "m", "TOOLS": [{"type": "web_search"}]}, "tool:web_search"),
            # NFKC reads a long s as "s", which Go's JSON decoder also folds.
            ({"model": "m", "tool\u017f": [{"type": "web_search"}]}, "tool:web_search"),
            (
                {"model": "m", "tools\u0000": [{"type": "web_search"}]},
                "tool:web_search",
            ),
            ({"model": "m", "tools": [{"Type": "WEB_SEARCH"}]}, "tool:WEB_SEARCH"),
            ({"model": "m", "tools": [{"type": " web-search"}]}, "tool: web-search"),
            ({"model": "m", "tools": [{"type": "webSearch"}]}, "tool:webSearch"),
            ({"model": "m", "Web_Search_Options": {}}, "web_search_options"),
            ({"model": "m", "webSearchOptions": {}}, "web_search_options"),
            ({"contents": [], "tools": [{"URL_CONTEXT": {}}]}, "tool:URL_CONTEXT"),
            ({"contents": [], "tools": [{"GoogleSearch": {}}]}, "tool:GoogleSearch"),
            (
                {
                    "model": "m",
                    "input": [{"content": [{"Type": "input_file", "File_Url": PAPER}]}],
                },
                "input:input_file",
            ),
            (
                {"contents": [{"parts": [{"FileData": {"FileUri": PAPER}}]}]},
                "input:file_data",
            ),
            (
                {
                    "messages": [
                        {
                            "content": [
                                {
                                    "type": "document",
                                    "Source": {"Type": "URL", "url": PAPER},
                                }
                            ]
                        }
                    ]
                },
                "input:document",
            ),
        ],
    )
    def test_spelling_does_not_hide_a_feature(self, request_body, feature):
        assert hosted_fetch_rule(request_body) == f"hosted-fetch:{feature}"

    @pytest.mark.parametrize(
        "body",
        [
            b'{"model":"m","tools":[{"ty\xffpe":"web_search"}]}',
            b'{"model":"m","to\xc0ols":[{"type":"web_search"}]}',
            b'{"model":"m","tools":[{"type":"web\xff_search"}]}',
        ],
        ids=["byte-in-key", "overlong-in-key", "byte-in-type"],
    )
    def test_a_byte_a_decoder_drops_does_not_hide_a_feature(self, body):
        assert model_request_rule(body).startswith("hosted-fetch:tool:")


class TestAddresses:
    """A string is an address when a URL parser would fetch it."""

    @pytest.mark.parametrize(
        "url",
        [
            # Azure OpenAI fetched both of these.
            "https:arxiv.org/pdf/2401.12345",
            "htt\tps://arxiv.org/pdf/2401.12345",
            "https:/arxiv.org/pdf/2401.12345",
            "https:\\\\arxiv.org\\pdf\\2401.12345",
            "//arxiv.org/pdf/2401.12345",
            "HTTPS://arxiv.org/pdf/2401.12345",
            " \x01https://arxiv.org/pdf/2401.12345",
            "\u3000https://arxiv.org/pdf/2401.12345",
            "\ufeff\u200bhttps://arxiv.org/pdf/2401.12345",
            "\uff48\uff54\uff54\uff50\uff53://arxiv.org/pdf/2401.12345",
            "gs://bucket/x.pdf",
            "https://generativelanguage.googleapis.com@arxiv.org/pdf/2401.12345",
            "https://generativelanguage.googleapis.com.example.test/x",
        ],
    )
    def test_address_forms_are_found(self, url):
        assert hosted_fetch_rule(_file_part(url)) == "hosted-fetch:input:input_file"

    @pytest.mark.parametrize(
        "value",
        [
            "data:application/pdf;base64,JVBERi0x",
            "data:" + "A" * 1_000_000,
            "file-abc123",
            "report:final.pdf",
            "C:\\papers\\x.pdf",
            "//uQxAAAAAAAAAAAAAAAAAAAAAAAWGluZwAAAA8=",
            "https://generativelanguage.googleapis.com/v1beta/files/abc",
        ],
    )
    def test_other_values_are_not_addresses(self, value):
        part = {
            "type": "input_file",
            "file_id": value,
            "filename": value,
            "file_data": value,
        }
        assert hosted_fetch_rule({"model": "m", "input": [{"content": [part]}]}) is None


class TestProviderFeatures:
    @pytest.mark.parametrize(
        ("request_body", "feature"),
        [
            # Zhipu documents, DashScope audio, video frames and native parts.
            (
                {
                    "messages": [
                        {"content": [{"type": "file_url", "file_url": {"url": PAPER}}]}
                    ]
                },
                "input:file_url",
            ),
            (
                {
                    "messages": [
                        {
                            "content": [
                                {
                                    "type": "input_audio",
                                    "input_audio": {"data": PAPER, "format": "wav"},
                                }
                            ]
                        }
                    ]
                },
                "input:input_audio",
            ),
            (
                {
                    "messages": [
                        {"content": [{"type": "video", "video": [PAPER, PAPER]}]}
                    ]
                },
                "input:video",
            ),
            (
                {"messages": [{"content": [{"image": PAPER}, {"text": "describe"}]}]},
                "input:image",
            ),
            (
                {"messages": [{"content": [{"type": "image", "url": PAPER}]}]},
                "input:image",
            ),
            (
                {"model": "m", "tools": [{"type": "web_extractor"}]},
                "tool:web_extractor",
            ),
            ({"model": "m", "tools": [{"type": "browser"}]}, "tool:browser"),
            ({"model": "m", "tools": [{"type": "image_search"}]}, "tool:image_search"),
            (
                {
                    "contents": [],
                    "tools": [{"mcp_servers": [{"url": "https://mcp.test"}]}],
                },
                "tool:mcp_servers",
            ),
            # Hosted containers: OpenAI and Azure hosted shell with network access.
            (
                {
                    "model": "m",
                    "tools": [
                        {
                            "type": "shell",
                            "environment": {
                                "type": "container_auto",
                                "network_policy": {
                                    "type": "allowlist",
                                    "allowed_domains": ["arxiv.org"],
                                },
                            },
                        }
                    ],
                },
                "tool:shell",
            ),
            (
                {
                    "model": "m",
                    "tools": [
                        {
                            "type": "shell",
                            "environment": {
                                "type": "container_reference",
                                "container_id": "cntr_1",
                            },
                        }
                    ],
                },
                "tool:shell",
            ),
            (
                {
                    "model": "m",
                    "tools": [
                        {
                            "type": "image_generation",
                            "input_image_mask": {"image_url": PAPER},
                        }
                    ],
                },
                "tool:image_generation",
            ),
            # Moonshot's built-in tools are named with a leading "$".
            (
                {
                    "model": "m",
                    "tools": [
                        {"type": "function", "function": {"name": "$web_search"}}
                    ],
                },
                "tool:$web_search",
            ),
            (
                {"model": "m", "tools": json.dumps([{"type": "web_search"}])},
                "tool:web_search",
            ),
            (
                {
                    "model": "m",
                    "tools": [{"type": "namespace", "tools": [{"type": "web_search"}]}],
                },
                "tool:web_search",
            ),
            ({"model": "hunyuan", "enable_enhancement": True}, "enable_enhancement"),
            ({"model": "ernie", "web_search": {"enable": True}}, "web_search"),
            # Models that search on every request.
            (
                {"model": "openai/gpt-5:online", "messages": []},
                "model:openai/gpt-5:online",
            ),
            (
                {"model": "gpt-4o-search-preview", "messages": []},
                "model:gpt-4o-search-preview",
            ),
            ({"model": "sonar-pro", "messages": []}, "model:sonar-pro"),
        ],
    )
    def test_provider_features_are_named(self, request_body, feature):
        assert hosted_fetch_rule(request_body) == f"hosted-fetch:{feature}"

    @pytest.mark.parametrize(
        "request_body",
        [
            # Tools that run on the client, or search the caller's own tools and files.
            {
                "tools": [
                    {"type": "shell"},
                    {"type": "shell", "environment": {"type": "local"}},
                    {"type": "shell", "environment": {"type": "container_auto"}},
                    {"type": "local_shell"},
                    {"type": "apply_patch"},
                    {"type": "tool_search_tool_regex_20251119", "name": "tool_search"},
                    {"type": "file_search", "vector_store_ids": ["vs_1"]},
                    {"type": "programmatic_tool_calling"},
                    {
                        "type": "code_interpreter",
                        "container": {"type": "auto", "file_ids": ["file-1"]},
                    },
                    {"type": "computer_use_preview", "display_width": 1024},
                    {"type": "bash_20250124", "name": "bash"},
                    {
                        "name": "WebFetch",
                        "description": "https://docs.test",
                        "input_schema": {"$id": "https://json-schema.org/x"},
                    },
                    {"functionDeclarations": [{"name": "web_fetch"}]},
                    {"googleMaps": {}},
                ]
            },
            # Notion and Slack blocks share part type names and are not model input.
            {
                "children": [
                    {
                        "type": "file",
                        "file": {"type": "external", "external": {"url": PAPER}},
                    }
                ]
            },
            {
                "children": [
                    {
                        "type": "image",
                        "image": {"type": "external", "external": {"url": PAPER}},
                    }
                ]
            },
            {"blocks": [{"type": "image", "image_url": PAPER, "alt_text": "x"}]},
            # A text document may open with an address; it is text, not fetched.
            {
                "messages": [
                    {
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "text",
                                    "data": f"{PAPER} is a paper",
                                },
                            }
                        ]
                    }
                ]
            },
            # History items and annotations that name addresses.
            {
                "input": [
                    {
                        "type": "local_shell_call",
                        "action": {"type": "exec", "command": ["curl", PAPER]},
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "c",
                        "output": f"{PAPER}\n<html>",
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "x",
                                "annotations": [{"type": "url_citation", "url": PAPER}],
                            }
                        ],
                    },
                ]
            },
            {
                "text": {
                    "format": {
                        "type": "json_schema",
                        "schema": {"$defs": {"plugins": {"type": "array"}}},
                    }
                }
            },
            {
                "model": "gpt-5.6-terra",
                "messages": [{"role": "user", "content": PAPER}],
            },
            {"model": "claude-opus-5-5", "messages": []},
            # Outside a model request these names are someone else's data.
            {
                "name": "my-project",
                "plugins": ["eslint"],
                "data_sources": [{"type": "postgres"}],
                "tools": [{"type": "web_search"}],
                "web_search": True,
            },
        ],
    )
    def test_ordinary_requests_pass(self, request_body):
        assert hosted_fetch_rule(request_body) is None


class TestReviewedRoutes:
    """Routes an independent review found, each shaped as the provider documents it."""

    @pytest.mark.parametrize(
        ("request_body", "feature"),
        [
            # Gemini's Interactions API: a hosted agent that browses, and a
            # document part given by URI.
            (
                {
                    "agent": "deep-research-preview-04-2026",
                    "input": "x",
                    "background": True,
                },
                "agent:deep-research-preview-04-2026",
            ),
            (
                {"model": "g", "input": [{"type": "document", "uri": PAPER}]},
                "input:document",
            ),
            # Bedrock system tools, which LiteLLM passes through unchanged.
            (
                {"model": "m", "tools": [{"systemTool": {"name": "nova_grounding"}}]},
                "tool:systemTool",
            ),
            # OpenRouter falls back through a list of models.
            (
                {
                    "model": "openai/gpt-5",
                    "models": ["openai/gpt-5:online"],
                    "messages": [],
                },
                "model:openai/gpt-5:online",
            ),
        ],
    )
    def test_routes_are_named(self, request_body, feature):
        assert hosted_fetch_rule(request_body) == f"hosted-fetch:{feature}"

    def test_a_repeated_function_name_is_read_whole(self):
        body = b'{"model":"m","tools":[{"type":"function","function":{"name":"lookup","name":"$web_search"}}]}'
        assert model_request_rule(body) == "hosted-fetch:tool:$web_search"

    @pytest.mark.parametrize(
        "body",
        [
            # Search-index and chat-card JSON share generic part type names.
            b'{"index":{"_id":"1"}}\n{"type":"video","url":"https://example.org/v.mp4"}\n',
            json.dumps(
                {
                    "type": "AdaptiveCard",
                    "body": [{"type": "Image", "url": "https://example.org/a.png"}],
                }
            ).encode(),
            json.dumps(
                {
                    "type": "Note",
                    "attachment": [
                        {"type": "Document", "url": "https://example.org/a.pdf"}
                    ],
                }
            ).encode(),
            json.dumps({"model": "sonarqube-community", "messages": []}).encode(),
            json.dumps(
                {
                    "agent": "claude-code",
                    "event": "tool_use",
                    "model": "claude-opus-5-5",
                }
            ).encode(),
        ],
        ids=[
            "search-bulk",
            "adaptive-card",
            "activitypub",
            "sonarqube",
            "telemetry-agent",
        ],
    )
    def test_lookalike_json_passes(self, body):
        assert model_request_rule(body) is None
