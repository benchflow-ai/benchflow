"""``bf.Policy``: the trainer's inference server, as rollouts reach it."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

from benchflow.training.relay import PolicyRelay, RelayError

# Routes whose servers BenchFlow's gateway asks for prompt and sampled token
# ids (vLLM's return_token_ids, SGLang's sglext flags).
TOKEN_ID_ROUTES = frozenset({"vllm", "sglang"})


@dataclass
class Policy:
    """A policy served by the trainer, for :func:`benchflow.rollout_group`.

    ``model`` is the BenchFlow route and served model name, e.g.
    ``"vllm/my-policy"`` or ``"sglang/my-policy"`` (the gateway asks those
    servers for token ids). ``base_url`` is the server's OpenAI-compatible
    ``/v1`` URL as this machine sees it; a loopback URL is fine, because only
    BenchFlow's policy relay, in this process, connects to it. The server's
    key comes from ``api_key`` or the environment variable ``api_key_env``
    and never enters a sandbox: rollouts get a per-rollout relay credential
    instead.

    ``version`` is the policy version rollouts record: set it after each
    weight update (``policy.version = step``) and every model call is tagged
    with the version current when it started, so asynchronous or off-policy
    trainers can weigh or drop stale samples.

    Sandboxes on the trainer's machine (Docker) reach the relay without
    further setup. Sandboxes elsewhere (Daytona) run the model gateway inside
    the sandbox, so the relay must listen where they can reach it:
    ``relay_bind="0.0.0.0:8443"`` and ``relay_public_url`` (the address they
    use, ideally ``https://`` behind a TLS terminator, or pass
    ``relay_ssl``). The relay accepts only inference requests with a live
    rollout credential, so the server behind it is never exposed.

    ``max_concurrency`` caps the rollouts running against this policy at
    once, across every group (the server's throughput, or a sandbox quota).
    ``capture_routed_experts`` asks SGLang servers for MoE routing
    (``return_routed_experts``; vLLM returns it whenever the server runs
    with ``--enable-return-routed-experts``).

    Use it as an async context manager, or call :meth:`start` and
    :meth:`close`; groups start it on first use.
    """

    model: str
    base_url: str
    api_key: str | None = field(default=None, repr=False)
    api_key_env: str | None = None
    version: Any = None
    relay_bind: str | None = None
    relay_public_url: str | None = None
    relay_ssl: Any | None = field(default=None, repr=False)
    max_concurrency: int | None = None
    capture_routed_experts: bool = False
    _relay: PolicyRelay | None = field(default=None, init=False, repr=False)
    _gate: asyncio.Semaphore | None = field(default=None, init=False, repr=False)
    _start_lock: asyncio.Lock | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if "/" not in self.model:
            raise ValueError(
                f"Policy.model must be a route and served model name such as "
                f"'vllm/my-policy', got {self.model!r}"
            )
        if self.max_concurrency is not None and self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if self.api_key is not None and self.api_key_env is not None:
            raise ValueError("give api_key or api_key_env, not both")

    @property
    def route(self) -> str:
        """The route provider, e.g. ``vllm``."""
        return self.model.split("/", 1)[0].lower()

    @property
    def captures_token_ids(self) -> bool:
        """Whether the gateway asks this route's server for token ids."""
        return self.route in TOKEN_ID_ROUTES

    def _resolve_key(self) -> str | None:
        if self.api_key is not None:
            return self.api_key
        if self.api_key_env is not None:
            value = os.environ.get(self.api_key_env, "")
            if not value:
                raise ValueError(
                    f"Policy.api_key_env={self.api_key_env!r} is not set in this "
                    "process's environment"
                )
            return value
        return None

    @property
    def relay(self) -> PolicyRelay:
        """The running relay (after :meth:`start`)."""
        if self._relay is None or not self._relay.started:
            raise RelayError("the policy is not started; use `async with policy:`")
        return self._relay

    @property
    def started(self) -> bool:
        return self._relay is not None and self._relay.started

    async def start(self) -> Policy:
        """Start the relay in front of the server (idempotent)."""
        if self._start_lock is None:
            self._start_lock = asyncio.Lock()
        async with self._start_lock:
            if self.started:
                return self
            relay = PolicyRelay(
                self.base_url,
                upstream_api_key=self._resolve_key(),
                model=self.model,
                version=lambda: self.version,
                public_bind=self.relay_bind,
                public_url=self.relay_public_url,
                ssl_context=self.relay_ssl,
            )
            await relay.start()
            self._relay = relay
            if self.max_concurrency is not None and self._gate is None:
                self._gate = asyncio.Semaphore(self.max_concurrency)
        return self

    async def close(self) -> None:
        """Stop the relay; rollouts still running lose their model."""
        relay, self._relay = self._relay, None
        if relay is not None:
            await relay.close()

    async def __aenter__(self) -> Policy:
        return await self.start()

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def agent_env(self, grant_token: str, environment: str) -> dict[str, str]:
        """What a rollout's gateway needs to reach this policy through the relay.

        ``BENCHFLOW_PROVIDER_BASE_URL`` is the relay on this machine (for a
        gateway on the host); ``BENCHFLOW_PROVIDER_BASE_URL_SANDBOX`` is the
        relay as a gateway inside a sandbox of ``environment`` reaches it.
        ``BENCHFLOW_PROVIDER_API_KEY`` is the rollout's relay credential.
        """
        env = {
            "BENCHFLOW_PROVIDER_BASE_URL": self.relay.host_url,
            "BENCHFLOW_PROVIDER_API_KEY": grant_token,
            "BENCHFLOW_CAPTURE_TOKEN_LOGPROBS": "1",
        }
        sandbox_url = await self.relay.url_for(environment)
        if sandbox_url is not None:
            env["BENCHFLOW_PROVIDER_BASE_URL_SANDBOX"] = sandbox_url
        if self.capture_routed_experts:
            env["BENCHFLOW_CAPTURE_ROUTED_EXPERTS"] = "1"
        return env


__all__ = ["Policy", "TOKEN_ID_ROUTES"]
