"""BenchFlow agent function for ``agentic_tool_call.generate``.

`BenchFlow <https://github.com/benchflow-ai/benchflow>`_ runs agent tasks in
sandboxes and grades them. With this agent function, each Miles rollout is one
BenchFlow episode: ``run`` posts the rollout's session URL to a BenchFlow
environment server, which starts the task's sandbox (Daytona or Docker),
plays the episode against the session URL with a ``run_bash``/``submit`` tool
loop, runs the task's verifier, and answers with the reward and a named
``exit_status``::

    agentic_tool_call.generate
      -> run(base_url, ...) --POST /run--> BenchFlow environment server
                                             |- task sandbox: commands only
                                             '- model calls --> session URL

The model calls leave the environment server for the session URL directly,
one non-streamed request per turn, and each assistant message is sent back
exactly as the session server returned it. The sandbox never calls the model,
so nothing in it needs a route to the session server.

The environment server runs in its own Python environment (BenchFlow and
Miles pin different versions of shared packages). Start it on the rollout host
before the trainer; see the README.

Env vars (read on the rollout worker):
  BENCHFLOW_ENV_URL          the environment server (default
                             http://127.0.0.1:12100)
  BENCHFLOW_ENV_TOKEN_FILE   file holding the server's bearer token, when it
                             was started with --token-file
  BENCHFLOW_EPISODE_TIMEOUT  backstop in seconds for one /run call (default
                             3600). The server's own --episode-timeout fires
                             first and attributes the overrun (0 once the policy
                             acted); a server that has not answered by the
                             backstop failed on its side, and the sample is
                             discarded (EnvironmentTimeout)

Failure semantics (the rule of radixark/miles#2802): discard only what the
policy cannot have caused, score everything else 0 with a named exit_status.
BenchFlow can apply the rule after the agent starts, because the environment
server knows whether the policy acted and which side failed:

  discarded (InfraAbort)     SandboxUnavailable     the sandbox never started
                             ModelEndpointFailed    the session server failed
                                                    (unreachable, 404, 5xx)
                             GenerationAborted      SGLang aborted the turn (503)
                             VerifierCrashCleanRun  the verifier crashed on a
                                                    sandbox the policy never used
                             Aborted                cancelled by ``abort``
                             ServerUnreachable      no BenchFlow server
                             EnvironmentTimeout     no answer by the backstop
  scored by the verifier     Submitted, NoToolCall, TurnLimitExceeded,
                             SequenceLengthLimitExceeded (a reply cut at
                             max_tokens, or the context full), RequestRejected
                             (the session server refused a request the
                             policy's output can break: 400, 409, 422, 500)
  scored 0                   TimeLimitExceeded (the server's cap, after the
                             policy acted), VerifierError, AgentError,
                             NoReward, IntegrityViolation (the reward-integrity
                             audit caught the policy exploiting the grader;
                             ``eval_report.flagged`` is true)

On a Miles without ``InfraAbort`` (before #2801) a discarded sample returns
``reward: None`` instead, and Miles drops its group with the missing-reward
filter.
"""

import asyncio
import logging
import os
import socket
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

from miles.rollout.agentic.session import openai_session_url

try:  # radixark/miles#2801
    from miles.rollout.agentic.agent_function import InfraAbort
except ImportError:  # pragma: no cover - exercised on Miles before #2801
    InfraAbort = None

logger = logging.getLogger(__name__)

# No other miles imports: this module must load on CPU-only hosts and in the
# offline tests, and most of miles pulls in torch.

_DEFAULT_URL = "http://127.0.0.1:12100"
_DEFAULT_EPISODE_TIMEOUT_S = 3600
_client: httpx.AsyncClient | None = None


def env_url() -> str:
    return os.getenv("BENCHFLOW_ENV_URL", _DEFAULT_URL).rstrip("/")


def _headers() -> dict[str, str]:
    path = os.getenv("BENCHFLOW_ENV_TOKEN_FILE", "").strip()
    if not path:
        return {}
    token = Path(path).expanduser().read_text().strip()
    if not token:
        raise RuntimeError(f"BENCHFLOW_ENV_TOKEN_FILE={path} is empty")
    return {"Authorization": f"Bearer {token}"}


def _get_client() -> httpx.AsyncClient:
    """One client per worker; keepalive probes keep long /run calls open."""
    global _client
    if _client is None:
        socket_options = [
            (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
            (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPIDLE", 4), 60),
            (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPINTVL", 5), 30),
            (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPCNT", 6), 5),
        ]
        _client = httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(socket_options=socket_options, retries=2),
            limits=httpx.Limits(max_connections=512, max_keepalive_connections=64),
            timeout=httpx.Timeout(None, connect=30.0),
        )
    return _client


def _discard(exit_status: str, detail: str | None) -> dict[str, Any]:
    """Discard the sample: InfraAbort where Miles has it, else a reward of None."""
    message = f"BenchFlow discarded the sample ({exit_status}): {detail or 'no detail'}"
    if InfraAbort is not None:
        raise InfraAbort(exit_status, message)
    logger.warning(message)
    return {
        "reward": None,
        "exit_status": exit_status,
        "eval_report": {"detail": detail},
        "agent_metrics": {},
    }


def _failed(exit_status: str, detail: str) -> dict[str, Any]:
    return {
        "reward": 0.0,
        "exit_status": exit_status,
        "eval_report": {"detail": detail},
        "agent_metrics": {},
    }


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    """Run one BenchFlow episode for the sample; return its verdict as metadata."""
    metadata = metadata or {}
    url = env_url()
    payload = {
        "session_url": openai_session_url(base_url),
        "prompt": prompt,
        "request_kwargs": request_kwargs or {},
        # instance_id names the task; max_seq_len lets the server end the episode
        # before the context outgrows what Miles will train on.
        "metadata": metadata,
    }
    timeout_s = float(os.getenv("BENCHFLOW_EPISODE_TIMEOUT", _DEFAULT_EPISODE_TIMEOUT_S))
    try:
        response = await asyncio.wait_for(
            _get_client().post(f"{url}/run", json=payload, headers=_headers()),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        # The server scores a stalling policy itself, within its own episode cap;
        # past the backstop the server is what failed (a queue that never drained,
        # a hung sandbox provider), which this episode's policy cannot cause.
        # Cancelling the request makes the server cancel the episode and release
        # its sandbox.
        return _discard("EnvironmentTimeout", f"no answer from {url} within {timeout_s:.0f}s")
    except httpx.TransportError as e:
        return _discard(
            "ServerUnreachable",
            f"BenchFlow environment server at {url} unreachable: {e!r}",
        )

    if response.status_code != 200:
        text = response.text[:500]
        if 400 <= response.status_code < 500:
            # A bad request (unknown task, bad token) fails every sample the same
            # way; the launcher's preflight is meant to catch it before training.
            raise RuntimeError(f"BenchFlow refused the episode: HTTP {response.status_code}: {text}")
        logger.error(f"BenchFlow /run failed: HTTP {response.status_code}: {text}; scoring 0")
        return _failed("AgentError", f"HTTP {response.status_code}: {text}")

    body = response.json()
    if body.get("dropped"):
        return _discard(str(body.get("exit_status") or "InfraFailure"), body.get("detail"))
    report = dict(body.get("eval_report") or {})
    report["flagged"] = bool(body.get("flagged"))
    return {
        "reward": float(body["reward"]),
        "exit_status": body.get("exit_status") or "",
        "eval_report": report,
        "agent_metrics": body.get("agent_metrics") or {},
    }


async def abort(args) -> None:
    """Teardown hook for oversampling abort (called after SGLang is aborted).

    Once a rollout step has enough samples, the episodes still in flight would
    keep running their tool loops and sandboxes. Tell the environment server to
    cancel them; it releases their sandboxes and answers their /run calls with
    a discard.
    """
    url = env_url()
    try:
        response = await _get_client().post(f"{url}/abort", headers=_headers(), timeout=60.0)
        response.raise_for_status()
        logger.info(f"BenchFlow abort: {response.json()}")
    except Exception as e:
        logger.warning(f"BenchFlow abort at {url} failed: {e!r}")


# --- reward and metrics (read by --custom-rm-path and benchflow_rollout.RolloutFn) ---

_MEAN_KEYS = (
    "turns",
    "tool_calls",
    "tool_errors",
    "total_time",
    "env_setup_time",
    "agent_run_time",
    "eval_time",
    "model_query_time_sum",
    "env_execution_time_sum",
    "queue_time",
    "prompt_tokens",
    "completion_tokens",
    "context_tokens",
)


async def reward_func(args, samples, **kwargs) -> float | None | list[float | None]:
    """The episode's reward, computed by BenchFlow during generate().

    A discarded episode on a Miles without ``InfraAbort`` carries ``None``: the
    missing-reward filter then drops its group before training sees it.
    """
    if isinstance(samples, list):
        return [sample.metadata.get("reward", 0.0) for sample in samples]
    return samples.metadata.get("reward", 0.0)


def benchflow_metrics(samples) -> dict[str, float]:
    """Per-step metrics over the samples that entered training."""
    episodes = [s for s in samples if s.metadata and "exit_status" in s.metadata]
    if not episodes:
        return {}
    n = len(episodes)
    metrics: dict[str, float] = {}
    for status, count in Counter(s.metadata["exit_status"] for s in episodes).items():
        metrics[f"benchflow/exit_status/{status}"] = count / n
    agent = [s.metadata.get("agent_metrics") or {} for s in episodes]
    replies = sum(m.get("turns", 0) for m in agent)
    if replies:
        metrics["benchflow/clipped_reply_ratio"] = sum(m.get("clipped_replies", 0) for m in agent) / replies
    reports = [s.metadata.get("eval_report") or {} for s in episodes]
    metrics["benchflow/context_exhausted_ratio"] = sum(r.get("ended") == "context_exhausted" for r in reports) / n
    metrics["benchflow/flagged"] = float(sum(bool(r.get("flagged")) for r in reports))
    for key in _MEAN_KEYS:
        values = [m[key] for m in agent if m.get(key) is not None]
        if values:
            metrics[f"benchflow/{key}_mean"] = sum(values) / len(values)
    return metrics
