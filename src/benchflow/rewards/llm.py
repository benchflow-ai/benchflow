"""Multi-provider LLM routing and verdict parsing for judge calls."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Verdict parsing
# ------------------------------------------------------------------


def parse_verdict(text: str) -> dict[str, Any]:
    """Extract a JSON verdict from an LLM response.

    Tries code-fenced JSON first, then bare ``{...}`` blocks.
    """
    # Code-fenced JSON
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
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
                        return json.loads(text[i : j + 1])
                    except json.JSONDecodeError:
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
) -> str:
    """Call an LLM judge, routing by model name prefix.

    Tries Anthropic, OpenAI, and Google SDKs in order based on the
    model string.  Falls back through providers if the primary is
    unavailable.
    """
    bare_model = _strip_provider_prefix(model)
    providers: list[str] = []

    if _is_anthropic_model(model):
        providers = ["anthropic", "openai", "google"]
    elif _is_openai_model(model):
        providers = ["openai", "anthropic", "google"]
    elif _is_gemini_model(model):
        providers = ["google", "anthropic", "openai"]
    else:
        # Unknown prefix — try all
        providers = ["anthropic", "openai", "google"]

    last_error: Exception | None = None
    for provider in providers:
        for attempt in range(retries):
            try:
                if provider == "anthropic":
                    return await _call_anthropic(bare_model, prompt, max_tokens)
                if provider == "openai":
                    return await _call_openai(bare_model, prompt, max_tokens)
                if provider == "google":
                    return await _call_google(bare_model, prompt)
            except ImportError:
                logger.debug("SDK for %s not installed, skipping", provider)
                break  # No point retrying if SDK is missing
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    time.sleep(2**attempt)
                    continue
                logger.warning(
                    "%s call failed after %d attempts: %s",
                    provider,
                    retries,
                    e,
                )
                break  # Move to next provider

    msg = f"All LLM providers failed for model {model}"
    if last_error:
        msg += f": {last_error}"
    raise RuntimeError(msg)


# ------------------------------------------------------------------
# Provider implementations
# ------------------------------------------------------------------


async def _call_anthropic(model: str, prompt: str, max_tokens: int) -> str:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


async def _call_openai(model: str, prompt: str, max_tokens: int) -> str:
    import openai

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content
    return content or ""


async def _call_google(model: str, prompt: str) -> str:
    import os

    from google import genai

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get(
        "GEMINI_API_KEY"
    )
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY / GEMINI_API_KEY not set")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text
