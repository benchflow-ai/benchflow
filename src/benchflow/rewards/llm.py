"""Multi-provider LLM routing and verdict parsing for judge calls."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)


class JudgeEnvironmentError(RuntimeError):
    """No LLM provider SDK is installed for the judge.

    Raised when *every* provider import fails with ``ImportError``. This is an
    *environment* failure — the judge could not run at all — and must be kept
    distinct from a genuine judge verdict (including a real score of 0). Callers
    surface it as a verifier error rather than recording it as reward ``0.0``.

    Install the judge SDKs with the ``judge`` extra: ``uv sync --extra judge``.
    """


# ------------------------------------------------------------------
# Verdict parsing
# ------------------------------------------------------------------


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def _loads_verdict_json(text: str) -> dict[str, Any]:
    parsed = json.loads(text, parse_constant=_reject_json_constant)
    if not isinstance(parsed, dict):
        raise ValueError("Judge verdict must be a JSON object")
    return parsed


def parse_verdict(text: str) -> dict[str, Any]:
    """Extract a JSON verdict from an LLM response.

    Tries code-fenced JSON first, then bare ``{...}`` blocks.
    """
    # Code-fenced JSON
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return _loads_verdict_json(match.group(1).strip())
        except ValueError:
            pass

    # Balanced braces
    for i, ch in enumerate(text):
        if ch == "{":
            depth = 0
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                if depth == 0:
                    try:
                        return _loads_verdict_json(text[i : j + 1])
                    except ValueError:
                        break
            break

    raise ValueError(f"Could not parse verdict from: {text[:300]}")


def parse_binary_verdict(text: str) -> bool:
    """Coerce common LLM outputs to a boolean pass/fail."""
    lowered = text.strip().lower()
    return lowered in {"yes", "true", "1", "pass", "passed"}


# ------------------------------------------------------------------
# Provider routing
# ------------------------------------------------------------------


def _is_anthropic_model(model: str) -> bool:
    return model.startswith(("claude-", "anthropic/"))


def _is_openai_model(model: str) -> bool:
    return model.startswith(("gpt-", "o1", "o3", "o4", "openai/"))


def _is_gemini_model(model: str) -> bool:
    return model.startswith(("gemini", "google/"))


def _strip_provider_prefix(model: str) -> str:
    """Remove ``provider/`` prefix if present."""
    if "/" in model:
        parts = model.split("/", 1)
        provider = parts[0]
        if provider in {"anthropic", "openai", "google"}:
            return parts[1]
    return model


async def call_judge(
    model: str,
    prompt: str,
    *,
    max_tokens: int = 2048,
    retries: int = 3,
    env: Mapping[str, str] | None = None,
) -> str:
    """Call an LLM judge, routing by model name prefix.

    ``env`` carries the resolved ``[verifier.judge]`` credentials (API keys
    etc.). They are passed explicitly into the provider clients rather than
    being written into the process-global ``os.environ`` — concurrent judge
    runs (``asyncio.gather`` with concurrency > 1) would otherwise race on
    the shared environment and see each other's credentials. ``env`` keys
    are layered over (and take precedence on a per-call basis without
    mutating) the ambient process environment.

    Tries Anthropic, OpenAI, and Google SDKs in order based on the
    model string.

    When the model name carries a *known* provider prefix (``claude-``,
    ``gpt-``, ``gemini``, ...), only that provider can serve it: a real
    API failure (bad key, model not found, retries exhausted) is raised
    as-is rather than being masked by a different provider rejecting the
    same model name.

    When the prefix is *unknown* (``mistral-large``, ``deepseek-chat``, a
    custom name), the provider cannot be determined up front, so every
    provider is tried in turn. A real API failure from one provider then
    falls through to the next instead of aborting the whole call.

    In both cases a missing SDK (``ImportError``) always falls through to
    the next provider.
    """
    bare_model = _strip_provider_prefix(model)
    creds: Mapping[str, str] = env or {}
    providers: list[str] = []
    # ``matched_provider`` is True only when the model name confidently
    # identifies a single provider. For an unknown prefix the providers
    # list is a best-effort "try all", so an API failure must not abort.
    matched_provider = True

    if _is_anthropic_model(model):
        providers = ["anthropic", "openai", "google"]
    elif _is_openai_model(model):
        providers = ["openai", "anthropic", "google"]
    elif _is_gemini_model(model):
        providers = ["google", "anthropic", "openai"]
    else:
        # Unknown prefix — try every provider in turn.
        providers = ["anthropic", "openai", "google"]
        matched_provider = False

    last_error: Exception | None = None

    for provider in providers:
        for attempt in range(retries):
            try:
                if provider == "anthropic":
                    return await _call_anthropic(bare_model, prompt, max_tokens, creds)
                if provider == "openai":
                    return await _call_openai(bare_model, prompt, max_tokens, creds)
                if provider == "google":
                    return await _call_google(bare_model, prompt, creds)
            except ImportError:
                logger.debug("SDK for %s not installed, skipping", provider)
                break  # SDK missing — fall through to the next provider
            except Exception as e:
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                last_error = e
                logger.warning(
                    "%s call failed after %d attempts: %s",
                    provider,
                    retries,
                    e,
                )
                if matched_provider:
                    # A real API failure for this provider's own model.
                    # Other providers cannot serve this model name, so
                    # do not advance — raise the original error directly.
                    raise
                # Unknown-prefix model: this provider could not serve it,
                # but another one might — fall through to the next.
                break

    # Either every provider's SDK was missing (each ImportError ``break``s to
    # the next), or — for an unknown-prefix model — every provider was tried
    # and each failed at the API. If we have a recorded API failure, surface
    # it so the caller sees a genuine error rather than a misleading
    # missing-SDK message.
    if last_error is not None:
        raise last_error

    # No API was ever reached: every provider's SDK was missing. This is an
    # environment failure, not a verdict — ``JudgeEnvironmentError`` lets
    # callers tell it apart from a real score.
    raise JudgeEnvironmentError(
        f"No LLM provider SDK is installed for model {model}. "
        "Install the judge extra: `uv sync --extra judge` "
        "(provides anthropic, openai, google-genai)."
    )


# ------------------------------------------------------------------
# Provider implementations
# ------------------------------------------------------------------


async def _call_anthropic(
    model: str, prompt: str, max_tokens: int, env: Mapping[str, str] | None = None
) -> str:
    import anthropic  # ty: ignore[unresolved-import]

    creds = env or {}
    # Pass the resolved key explicitly so concurrent judge runs cannot race
    # on a shared os.environ. Fall back to the SDK's own env lookup when no
    # key was threaded through.
    api_key = creds.get("ANTHROPIC_API_KEY")
    client = (
        anthropic.AsyncAnthropic(api_key=api_key)
        if api_key
        else anthropic.AsyncAnthropic()
    )
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    # The response may have no content blocks, or lead with a non-text
    # block (e.g. a tool-use block).  Return the first text block's text,
    # or "" if none is present.
    for block in response.content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            return text
    return ""


async def _call_openai(
    model: str, prompt: str, max_tokens: int, env: Mapping[str, str] | None = None
) -> str:
    import openai

    creds = env or {}
    # Pass the resolved key explicitly so concurrent judge runs cannot race
    # on a shared os.environ.
    api_key = creds.get("OPENAI_API_KEY")
    base_url = creds.get("OPENAI_BASE_URL")
    kwargs: dict[str, str] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    client = openai.AsyncOpenAI(**kwargs)
    response = await client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content
    return content or ""


async def _call_google(
    model: str, prompt: str, env: Mapping[str, str] | None = None
) -> str:
    import os

    from google import genai  # ty: ignore[unresolved-import]

    creds = env or {}
    # Prefer the explicitly threaded credentials (concurrency-safe); fall
    # back to the ambient process env only when none were provided.
    api_key = (
        creds.get("GOOGLE_API_KEY")
        or creds.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY / GEMINI_API_KEY not set")
    client = genai.Client(api_key=api_key)
    response = await client.aio.models.generate_content(model=model, contents=prompt)
    # ``response.text`` is ``None`` when the response has no text part
    # (e.g. content blocked by a safety filter, or a non-text part leads
    # the response).  Return "" so the ``-> str`` contract holds.
    return response.text or ""
