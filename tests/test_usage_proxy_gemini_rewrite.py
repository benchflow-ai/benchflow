"""Unit tests for the Gemini upstream-path rewrite in the sandbox usage proxy.

litellm's Google AI Studio provider, when given a custom ``api_base`` (this
proxy), builds URLs as ``{api_base}/models/{model}:{action}`` — it drops the
required ``/v1beta`` version prefix, and for context caching it emits the bogus
``:cachedContents`` model-action instead of Google's real top-level
``/v1beta/cachedContents`` collection
(litellm ``vertex_llm_base._check_custom_proxy``). Without normalization every
Gemini call — especially the prompt-cache probe — 404s through the proxy, which
forced ``--usage-tracking off`` (and thus loss of token telemetry).

These tests pin the proxy's ``normalizeGeminiUpstreamPath`` behavior by running
the exact deployed JS source under node, so the rewrite cannot silently drift.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from benchflow.providers.sandbox_usage_proxy import _NODE_PROXY_SOURCE

# (litellm-emitted incoming path, expected normalized upstream path)
_GEMINI_REWRITE_CASES = [
    # prompt-cache probe: bogus per-model action -> Google's real top-level collection
    (
        "/models/gemini-3.5-flash:cachedContents?key=secret",
        "/v1beta/cachedContents?key=secret",
    ),
    # main completion call: missing /v1beta restored
    (
        "/models/gemini-3.5-flash:generateContent",
        "/v1beta/models/gemini-3.5-flash:generateContent",
    ),
    (
        "/models/gemini-3.5-flash:streamGenerateContent?alt=sse",
        "/v1beta/models/gemini-3.5-flash:streamGenerateContent?alt=sse",
    ),
    (
        "/models/gemini-3.5-flash:countTokens",
        "/v1beta/models/gemini-3.5-flash:countTokens",
    ),
    # already-correct paths are left untouched (idempotent)
    ("/v1beta/cachedContents", "/v1beta/cachedContents"),
    (
        "/v1beta/models/gemini-3.5-flash:generateContent",
        "/v1beta/models/gemini-3.5-flash:generateContent",
    ),
    ("/v1/models/x:generateContent", "/v1/models/x:generateContent"),
]


def _extract_js_function(name: str) -> str:
    match = re.search(rf"function {name}\([\s\S]*?\n\}}", _NODE_PROXY_SOURCE)
    assert match is not None, f"{name} not found in proxy source"
    return match.group(0)


def test_gemini_upstream_path_rewrite_normalizes_litellm_paths():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")

    cases_js = ",".join(
        f"[{_js_str(inp)},{_js_str(want)}]" for inp, want in _GEMINI_REWRITE_CASES
    )
    harness = _extract_js_function("normalizeGeminiUpstreamPath") + (
        f"\nconst cases=[{cases_js}];\n"
        "for(const [inp,want] of cases){"
        "const got=normalizeGeminiUpstreamPath(inp);"
        'if(got!==want){console.error("FAIL",JSON.stringify(inp),"->",'
        'JSON.stringify(got),"want",JSON.stringify(want));process.exit(1);}}'
        '\nconsole.log("ALL_OK");\n'
    )
    result = subprocess.run(
        [node, "-e", harness], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "ALL_OK" in result.stdout


def test_gemini_rewrite_is_scoped_to_the_google_host():
    # The rewrite must only ever apply to the Gemini upstream, so every other
    # provider (openai/anthropic/bedrock) forwards byte-for-byte unchanged.
    assert "isGeminiTarget" in _NODE_PROXY_SOURCE
    # host appears as a regex literal (escaped dots), so match on the bare label
    assert "generativelanguage" in _NODE_PROXY_SOURCE
    # upstreamPath only normalizes when the target is the Gemini host.
    assert re.search(
        r"isGeminiTarget\s*\?\s*normalizeGeminiUpstreamPath", _NODE_PROXY_SOURCE
    ), "gemini path normalization is not gated on the Gemini host"


def _js_str(value: str) -> str:
    import json

    return json.dumps(value)
