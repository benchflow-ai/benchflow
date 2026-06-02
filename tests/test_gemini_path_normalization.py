"""Host-side (Docker) Gemini path normalization + parity with the sandbox JS.

The Daytona sandbox proxy rewrote litellm's malformed
``/models/{model}:cachedContents`` path, but the same-host ``TrajectoryProxy``
(used for Docker usage tracking) forwarded it unchanged — so Docker Gemini runs
still 404'd and lost usage tracking. These tests pin the Python rewrite, its
host-scoping, the proxy wiring, and byte-for-byte parity with the deployed JS.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from benchflow.providers.sandbox_usage_proxy import _NODE_PROXY_SOURCE
from benchflow.trajectories.gemini_paths import (
    is_gemini_host,
    is_gemini_target,
    normalize_gemini_upstream_path,
)
from benchflow.trajectories.proxy import TrajectoryProxy

# (litellm-emitted incoming path, expected normalized upstream path)
_CASES = [
    ("/models/gemini-3.5-flash:cachedContents?key=s", "/v1beta/cachedContents?key=s"),
    (
        "/models/gemini-3.5-flash:generateContent",
        "/v1beta/models/gemini-3.5-flash:generateContent",
    ),
    (
        "/models/g:streamGenerateContent?alt=sse",
        "/v1beta/models/g:streamGenerateContent?alt=sse",
    ),
    ("/models/g:countTokens", "/v1beta/models/g:countTokens"),
    # already-correct paths are idempotent
    ("/v1beta/cachedContents", "/v1beta/cachedContents"),
    ("/v1beta/models/g:generateContent", "/v1beta/models/g:generateContent"),
    ("/v1/models/x:generateContent", "/v1/models/x:generateContent"),
]


@pytest.mark.parametrize("incoming,expected", _CASES)
def test_python_normalizer_behavior(incoming, expected):
    assert normalize_gemini_upstream_path(incoming) == expected


@pytest.mark.parametrize(
    "host,expected",
    [
        ("generativelanguage.googleapis.com", True),
        ("GenerativeLanguage.GoogleAPIs.com", True),
        ("asia.generativelanguage.googleapis.com", True),
        ("api.openai.com", False),
        ("bedrock-runtime.us-east-1.amazonaws.com", False),
        # suffix-spoofing must NOT match (no dot/start boundary before the label)
        ("evil-generativelanguage.googleapis.com", False),
        ("generativelanguage.googleapis.com.attacker.test", False),
        (None, False),
        ("", False),
    ],
)
def test_is_gemini_host(host, expected):
    assert is_gemini_host(host) is expected


def test_is_gemini_target_parses_base_url():
    assert is_gemini_target("https://generativelanguage.googleapis.com") is True
    assert is_gemini_target("https://generativelanguage.googleapis.com/v1beta") is True
    assert is_gemini_target("https://api.anthropic.com") is False
    assert is_gemini_target(None) is False


def test_trajectory_proxy_rewrites_only_for_gemini_target():
    gemini = TrajectoryProxy(target="https://generativelanguage.googleapis.com")
    assert gemini._is_gemini_target is True
    # the Docker gap: litellm's malformed cache path is now normalized host-side
    assert (
        gemini._upstream_url("/models/gemini-3.5-flash:cachedContents?key=k")
        == "https://generativelanguage.googleapis.com/v1beta/cachedContents?key=k"
    )
    # already-versioned gemini traffic is untouched
    assert (
        gemini._upstream_url("/v1beta/models/x:generateContent")
        == "https://generativelanguage.googleapis.com/v1beta/models/x:generateContent"
    )
    # every other upstream forwards byte-for-byte (no rewrite, no gating drift)
    anthropic = TrajectoryProxy(target="https://api.anthropic.com")
    assert anthropic._is_gemini_target is False
    assert (
        anthropic._upstream_url("/models/x:cachedContents")
        == "https://api.anthropic.com/models/x:cachedContents"
    )


def test_python_and_js_normalizers_agree():
    """Run the deployed JS normalizer and assert it matches Python on every case
    — the two cannot share code across runtimes, so this is the anti-drift pin."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    match = re.search(
        r"function normalizeGeminiUpstreamPath\([\s\S]*?\n\}", _NODE_PROXY_SOURCE
    )
    assert match is not None, "normalizeGeminiUpstreamPath not found in proxy source"
    inputs = [c[0] for c in _CASES] + [
        "/models/m:cachedContents",
        "/v1/cachedContents",
        "/models/a-b_c.1:generateContent?x=1&y=2",
    ]
    import json

    harness = match.group(0) + (
        f"\nconst inputs={json.dumps(inputs)};\n"
        "console.log(JSON.stringify(inputs.map(normalizeGeminiUpstreamPath)));\n"
    )
    result = subprocess.run(
        [node, "-e", harness], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    import json as _json

    js_out = _json.loads(result.stdout)
    py_out = [normalize_gemini_upstream_path(i) for i in inputs]
    assert js_out == py_out, f"JS/Python drift:\n  js={js_out}\n  py={py_out}"
