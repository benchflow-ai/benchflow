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
    """Raised when no judge provider SDK is available."""


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

    ``env`` carries per-call credentials resolved from ``[verifier.env]``.
    The provider clients receive those credentials explicitly so concurrent
    judge runs do not race on process-global ``os.environ``.
    """
    bare_model = _strip_provider_prefix(model)
    creds: Mapping[str, str] = env or {}
    providers: list[str] = []
    matched_provider = True

    if _is_anthropic_model(model):
        providers = ["anthropic", "openai", "google"]
    elif _is_openai_model(model):
        providers = ["openai", "anthropic", "google"]
    elif _is_gemini_model(model):
        providers = ["google", "anthropic", "openai"]
    else:
        # Unknown prefix — try all
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
                break  # No point retrying if SDK is missing
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                logger.warning(
                    "%s call failed after %d attempts: %s",
                    provider,
                    retries,
                    e,
                )
                if matched_provider:
                    raise
                break  # Move to next provider

    if last_error is not None:
        raise last_error
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

    api_key = (env or {}).get("ANTHROPIC_API_KEY")
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
    kwargs: dict[str, str] = {}
    if api_key := creds.get("OPENAI_API_KEY"):
        kwargs["api_key"] = api_key
    if base_url := creds.get("OPENAI_BASE_URL"):
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
    return response.text or ""
