"""A seat brain that routes its raw LLM call through BenchFlow's provider proxy.

Reads the proxy config the SDK injects — ``BENCHFLOW_PROVIDER_BASE_URL`` /
``BENCHFLOW_PROVIDER_API_KEY`` / ``BENCHFLOW_PROVIDER_MODEL`` (the same vars
``deepagents_acp_shim`` reads, falling back to provider-native vars) — and calls
the OpenAI-compatible ``/chat/completions`` endpoint. Inside a BenchFlow eval that
endpoint is the LiteLLM proxy, so every seat's raw-LLM usage is tracked per agent
(``llm_trajectory.jsonl``); each request is also tagged with its ``seat`` so the
proxy can attribute calls. The caller supplies ``render(obs) -> prompt`` and
``pick(text, legal) -> action`` so the policy stays game-agnostic.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import httpx

from benchflow.arena.protocol import Observation
from benchflow.arena.trajectory import SeatTrajectory

__all__ = ["provider_config", "ProxyChatPolicy"]


def provider_config(model_default: str | None = None) -> tuple[str, str, str]:
    """(base_url, api_key, model) — prefer the BenchFlow proxy, fall back to
    provider-native env. ``base_url`` is the proxy when running under an eval."""
    base = (
        os.environ.get("BENCHFLOW_PROVIDER_BASE_URL")
        or os.environ.get("DEEPSEEK_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.deepseek.com"
    ).rstrip("/")
    key = (
        os.environ.get("BENCHFLOW_PROVIDER_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    model = os.environ.get("BENCHFLOW_PROVIDER_MODEL") or model_default or "deepseek-v4-pro"
    return base, key, model


class ProxyChatPolicy:
    """A ``SeatPolicy`` whose decision is one chat-completion through the proxy."""

    def __init__(
        self,
        seat: str,
        http: httpx.AsyncClient,
        *,
        render: Callable[[Observation], str],
        pick: Callable[[str, list[dict[str, Any]]], dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
        recorder: SeatTrajectory | None = None,
    ) -> None:
        self.seat, self.http = seat, http
        self.render, self.pick = render, pick
        self.base, self.key, self.model = provider_config(model)
        self.temperature, self.max_tokens = temperature, max_tokens
        self.recorder = recorder

    async def act(self, obs: Observation) -> dict[str, Any]:
        messages = [{"role": "user", "content": self.render(obs)}]
        text: str = ""
        usage: dict[str, Any] | None = None
        try:
            resp = await self.http.post(
                f"{self.base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.key}",
                    "x-bf-seat": self.seat,  # per-seat attribution for the proxy
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "metadata": {"seat": self.seat},
                },
                timeout=90.0,
            )
            data = resp.json()
            text = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage")
        except Exception as exc:  # a flaky call falls back to a legal default
            text = ""
            usage = {"error": repr(exc)}
        action = self.pick(text, list(obs.legal_actions))
        if self.recorder is not None:
            self.recorder.record(
                self.seat, status=obs.status, observation=obs.public,
                legal_actions=obs.legal_actions, action=action, request_id=obs.request_id,
                llm={"model": self.model, "messages": messages, "response": text,
                     "usage": usage},
            )
        return action
