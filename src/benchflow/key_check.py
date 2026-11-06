"""Ask the model provider whether it takes the run's key, before any sandbox.

A run whose agent calls Anthropic's or OpenAI's API directly with a key
BenchFlow holds (``ANTHROPIC_API_KEY`` for a Claude model, ``OPENAI_API_KEY``
for a GPT or o-series model, no base URL of the user's) lists the provider's
models with the model's key once, before a job exists (Rust 0.8.2's R212 and
its did-you-mean, c8be3d86, e47d0db1). The other provider's key, which the
agent's env inherits from the host too, is never asked about:

- a 401 is a rejected key: the run stops, naming the variable, before
  any image is built or sandbox started (before, every trial built its image,
  started its sandbox and installed its agent, then failed on the first model
  call). A 401 whose error names missing scopes or permissions is not: OpenAI
  answers a restricted key without Models: Read that way;
- a model the list does not hold gets a note with the closest id; the run
  goes on (the provider may still serve it: aliases, new models);
- anything else (a 403, which a key that may call models but not list them
  gets, a 429, a 5xx, no network) checks nothing and says nothing.

Listing models costs no tokens. The quota is not checked (that costs a
request). ``BENCHFLOW_KEY_CHECK=0`` turns the check off; it is also skipped
with the other preflight checks (``BENCHFLOW_SKIP_PREFLIGHT=1``). The key
goes only into the request's header and is never printed.
"""

from __future__ import annotations

import difflib
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

KEY_CHECK_OPT_OUT_ENV = "BENCHFLOW_KEY_CHECK"
_TIMEOUT_SEC = 10.0

# (status, model ids, the error body's code, type and message) for a GET with
# these headers; raises on a transport failure.
HttpGet = Callable[[str, Mapping[str, str], float], tuple[int, list[str], str]]


@dataclass(frozen=True)
class KeyCheck:
    provider: str
    env_name: str
    state: str  # "accepted" | "rejected" | "unchecked"
    status: int | None = None
    models: list[str] = field(default_factory=list)

    def model_note(self, model: str | None) -> str | None:
        """A note when ``model`` is not in the provider's list, or None."""
        if self.state != "accepted" or not model or not self.models:
            return None
        bare = model.split("/", 1)[1] if "/" in model else model
        bare = re.sub(r"\[1m\]$", "", bare)
        # Aliases (sonnet, opus, default) carry no version: the agent maps them.
        if bare in self.models or not re.search(r"\d", bare):
            return None
        close = difflib.get_close_matches(bare, self.models, n=1, cutoff=0.6)
        hint = f"; did you mean {close[0]}?" if close else ""
        return (
            f"model {bare} is not in {self.provider}'s model list for the key in "
            f"{self.env_name}{hint} The run goes on: the provider may still serve it."
        )


def _error_text(body: Any) -> str:
    """The code, type and message of a provider's error body, lowercased.

    OpenAI: ``{"error": {"message", "type", "code"}}``; Anthropic:
    ``{"type": "error", "error": {"type", "message"}}``. Never printed.
    """
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return ""
    parts = [error.get(k) for k in ("code", "type", "message")]
    return " ".join(str(p) for p in parts if isinstance(p, str))[:500].lower()


def _http_get(
    url: str, headers: Mapping[str, str], timeout: float
) -> tuple[int, list[str], str]:
    import logging

    import httpx

    logging.getLogger("httpx").setLevel(logging.WARNING)
    response = httpx.get(url, headers=dict(headers), timeout=timeout)
    ids: list[str] = []
    error = ""
    try:
        body = response.json()
    except ValueError:
        body = None
    if response.status_code == 200:
        data = (body.get("data") if isinstance(body, dict) else None) or []
        ids = [str(m.get("id")) for m in data if isinstance(m, dict) and m.get("id")]
    else:
        error = _error_text(body)
    return response.status_code, ids, error


# A 401 that says the key lacks a scope, not that it is bad: OpenAI answers a
# restricted key without Models: Read (api.model.read) with "You have
# insufficient permissions for this operation. Missing scopes: ...", and that
# key may still call the model.
_SCOPE_REFUSAL = ("missing scope", "insufficient permission", "insufficient_permission")


_DIRECT_PROVIDERS: dict[str, tuple[str, str, str]] = {
    # key variable: (provider, the user's own base URL variable, models URL)
    "ANTHROPIC_API_KEY": (
        "Anthropic",
        "ANTHROPIC_BASE_URL",
        "https://api.anthropic.com/v1/models?limit=1000",
    ),
    "OPENAI_API_KEY": ("OpenAI", "OPENAI_BASE_URL", "https://api.openai.com/v1/models"),
}


def _direct_provider(
    agent_env: Mapping[str, str], model: str | None
) -> tuple[str, str, str, dict[str, str]] | None:
    """(provider, key variable, models URL, headers) for the model's direct key.

    The key is the one the model uses, never whichever is set: the agent's env
    inherits every provider key the host holds, so a codex run on a gpt model
    with a stale ``ANTHROPIC_API_KEY`` in the shell must not ask Anthropic.
    """
    if not model or agent_env.get("BENCHFLOW_PROVIDER_BASE_URL"):
        return None  # BenchFlow's own route (a gateway or the model proxy)
    from benchflow.agents.registry import infer_env_key_for_model

    try:
        env_name = infer_env_key_for_model(model)
    except Exception:
        return None
    if env_name is None or env_name not in _DIRECT_PROVIDERS:
        return None
    provider, base_url_env, url = _DIRECT_PROVIDERS[env_name]
    key = agent_env.get(env_name, "").strip()
    if not key or agent_env.get(base_url_env):
        return None
    if env_name == "ANTHROPIC_API_KEY":
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    else:
        headers = {"Authorization": f"Bearer {key}"}
    return provider, env_name, url, headers


def key_check_enabled(environ: Mapping[str, str] | None = None) -> bool:
    value = (environ if environ is not None else os.environ).get(KEY_CHECK_OPT_OUT_ENV)
    return (value or "").strip().lower() not in ("0", "false", "no", "off")


def check_key(
    agent_env: Mapping[str, str], model: str | None, *, get: HttpGet | None = None
) -> KeyCheck | None:
    """Ask the provider about the key ``model`` uses; None when not checkable."""
    target = _direct_provider(agent_env, model)
    if target is None:
        return None
    provider, env_name, url, headers = target
    try:
        status, models, error = (get or _http_get)(url, headers, _TIMEOUT_SEC)
    except Exception:
        return KeyCheck(provider, env_name, "unchecked")
    # Only a 401 says the key is bad, and not one that names a missing scope.
    # A 403 or such a 401 on the models list can be a key that may call
    # models but not list them (an OpenAI restricted or project key without
    # Models: Read), so it checks nothing.
    if status == 401 and not any(s in error for s in _SCOPE_REFUSAL):
        return KeyCheck(provider, env_name, "rejected", status)
    if status == 200:
        return KeyCheck(provider, env_name, "accepted", status, models)
    return KeyCheck(provider, env_name, "unchecked", status)
