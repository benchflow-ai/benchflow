"""Gemini upstream-path normalization for the host-side TrajectoryProxy.

LiteLLM, when pointed at a custom ``api_base`` (BenchFlow's usage/trajectory
proxy), builds a malformed Gemini context-cache URL: it emits
``/models/{model}:cachedContents`` — which omits the required ``/v1beta``
version prefix and invents a per-model ``:cachedContents`` action instead of
Google's real top-level ``/v1beta/cachedContents`` collection — so Google 404s
and usage tracking (and the run) dies.

This module is the **Python source of truth** for the rewrite, applied by the
same-host ``TrajectoryProxy`` (Docker). The Daytona sandbox-local proxy applies
the identical logic in JavaScript; the two cannot share code across runtimes, so
``providers/assets/sandbox_usage_proxy.js:normalizeGeminiUpstreamPath`` must be
kept in sync — ``tests/test_gemini_path_normalization.py`` pins their parity.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

# Gemini AI Studio host (and any subdomain of it).
_GEMINI_HOST_RE = re.compile(
    r"(^|\.)generativelanguage\.googleapis\.com$", re.IGNORECASE
)
# litellm's bogus per-model cache action at the path root.
_CACHED_CONTENTS_RE = re.compile(r"^/models/[^/]+:cachedContents\b")


def is_gemini_host(host: str | None) -> bool:
    """Whether ``host`` is (a subdomain of) the Gemini generative-language API."""
    return bool(host and _GEMINI_HOST_RE.search(host))


def is_gemini_target(target: str | None) -> bool:
    """Whether an upstream base URL points at the Gemini API host."""
    if not target:
        return False
    return is_gemini_host(urlsplit(target).hostname)


def normalize_gemini_upstream_path(path_with_query: str) -> str:
    """Rewrite litellm's malformed Gemini cache path to Google's real resource.

    Mirror of ``normalizeGeminiUpstreamPath`` in ``sandbox_usage_proxy.js``.
    Already-versioned paths (``/v1beta/…`` or ``/v1/…``) pass through unchanged,
    so non-cache Gemini traffic is never altered.
    """
    q_idx = path_with_query.find("?")
    path = path_with_query if q_idx == -1 else path_with_query[:q_idx]
    query = "" if q_idx == -1 else path_with_query[q_idx:]
    # already version-prefixed -> leave as-is
    if path.startswith("/v1beta/") or path.startswith("/v1/"):
        return path_with_query
    # litellm's bogus per-model cache action -> Google's real top-level collection
    path = _CACHED_CONTENTS_RE.sub("/cachedContents", path)
    # all Google AI Studio resources live under /v1beta
    if not path.startswith("/v1beta/"):
        path = f"/v1beta{path}"
    return f"{path}{query}"
