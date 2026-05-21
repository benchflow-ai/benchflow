"""Multi-provider LLM routing and verdict parsing for judge calls."""

from __future__ import annotations

import asyncio
import json
import logging
import re
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
    model string.  Cross-provider fallback only applies when an SDK is
    *not installed* (``ImportError``): a model name only makes sense for
    the provider it belongs to, so a real API failure (bad key, model
    not found, retries exhausted) is raised as-is rather than being
    masked by a different provider rejecting the same model name.
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
                break  # SDK missing — fall through to the next provider
            except Exception as e:
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                # A real API failure for this provider's own model.
                # Other providers cannot serve this model name, so do
                # not advance — raise the original error directly.
                logger.warning(
                    "%s call failed after %d attempts: %s",
                    provider,
                    retries,
                    e,
                )
                raise

    # Every provider's SDK was missing (each ImportError ``break``s to the
    # next).  A real API failure raises directly above, so this is the only
    # state that reaches here.  This is an environment failure, not a verdict —
    # ``JudgeEnvironmentError`` lets callers tell it apart from a real score.
    raise JudgeEnvironmentError(
        f"No LLM provider SDK is installed for model {model}. "
        "Install the judge extra: `uv sync --extra judge` "
        "(provides anthropic, openai, google-genai)."
    )


# ------------------------------------------------------------------
# Provider implementations
# ------------------------------------------------------------------


async def _call_anthropic(model: str, prompt: str, max_tokens: int) -> str:
    import anthropic  # ty: ignore[unresolved-import]

    client = anthropic.AsyncAnthropic()
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


async def _call_openai(model: str, prompt: str, max_tokens: int) -> str:
    import openai

    client = openai.AsyncOpenAI()
    response = await client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content
    return content or ""


async def _call_google(model: str, prompt: str) -> str:
    import os

    from google import genai  # ty: ignore[unresolved-import]

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY / GEMINI_API_KEY not set")
    client = genai.Client(api_key=api_key)
    response = await client.aio.models.generate_content(model=model, contents=prompt)
    # ``response.text`` is ``None`` when the response has no text part
    # (e.g. content blocked by a safety filter, or a non-text part leads
    # the response).  Return "" so the ``-> str`` contract holds.
    return response.text or ""
