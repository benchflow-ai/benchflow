"""Orchestrate ``benchflow continue`` — boot, replay, continue live, stitch.

Reuses benchflow's normal :class:`~benchflow.rollout.Rollout` machinery (so the
new run produces a standard HF-compatible folder) but injects a record-replay
proxy in front of OpenHands and disables benchflow's own LiteLLM gateway:

- ``usage_tracking="off"`` makes ``ensure_litellm_runtime`` a no-op, so the
  ``agent_env`` we pass through (``LLM_BASE_URL`` → the replay proxy) is left
  untouched (``providers/litellm_runtime.py``).
- The original first prompt (task instruction) starts OpenHands; the proxy then
  drives the exact replay and, past the cut-point, the live continuation.

The async :func:`continue_run` is the integration entrypoint (it needs Docker +
the openhands install); the helpers it composes (config assembly, live
forwarder request-building, trajectory stitching) are pure and unit-tested.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from benchflow.continue_run.replay_proxy import ReplayProxy, ReplayRouter
from benchflow.continue_run.run_folder import RunFolder, RunFolderError, load_run_folder
from benchflow.trajectories.types import LLMExchange, redact_trajectory_text

logger = logging.getLogger(__name__)

# agent_env values handed to OpenHands so it talks to the replay proxy instead
# of a real provider. The model name is irrelevant (the proxy serves by index),
# but litellm needs an ``openai/`` prefix to treat LLM_BASE_URL as an
# OpenAI-compatible endpoint.
_REPLAY_API_KEY = "sk-benchflow-replay"
_REPLAY_MODEL = "openai/replay"


@dataclass
class ContinueResult:
    """Outcome of a ``benchflow continue`` run."""

    rollout_dir: Path
    rewards: dict[str, Any] | None
    error: str | None
    n_recorded: int
    n_live: int
    divergences: int


def resolve_task_path(run: RunFolder, tasks_dir: str | Path | None) -> Path:
    """Find the task directory (instruction + verifier + image) to run against.

    The uploaded run folder does **not** ship the task files, so we need the
    original task source: ``--tasks-dir/<task_name>`` if given, else the
    ``task_path`` recorded in ``config.json`` if it still exists on disk.
    """
    if tasks_dir is not None:
        candidate = Path(tasks_dir).expanduser() / run.task_name
        if candidate.is_dir():
            return candidate
        raise RunFolderError(
            f"--tasks-dir given but {candidate} does not exist; expected the "
            f"task source for {run.task_name!r} there."
        )
    recorded = Path(run.task_path) if run.task_path else None
    if recorded is not None and recorded.is_dir():
        return recorded
    raise RunFolderError(
        f"cannot locate task source for {run.task_name!r}. Pass --tasks-dir "
        "pointing at the directory that contains the task (its instruction and "
        "verifier are needed to re-run and re-verify)."
    )


def build_agent_env(proxy_base_url: str) -> dict[str, str]:
    """Env that points OpenHands' LiteLLM client at the replay proxy."""
    return {
        "LLM_BASE_URL": proxy_base_url,
        "LLM_API_KEY": _REPLAY_API_KEY,
        "LLM_MODEL": _REPLAY_MODEL,
    }


def build_rollout_config(
    run: RunFolder,
    *,
    task_path: Path,
    live_model: str | None,
    agent_env: dict[str, str],
    timeout: int | None,
    output_dir: str | Path,
    rollout_name: str,
) -> Any:
    """Assemble the RolloutConfig that re-runs the task through the proxy.

    Imported lazily so ``benchflow.continue_run`` stays importable without
    pulling the full rollout stack (keeps unit tests light).
    """
    from benchflow.rollout import RolloutConfig

    return RolloutConfig.from_legacy(
        task_path=task_path,
        agent="openhands",
        # model=None on purpose: the agent targets the replay proxy purely via
        # agent_env LLM_*. A real model here would make resolve_agent_env run
        # provider resolution and *validate the model's API key* — which would
        # wrongly fail the key-free replay phase and could clobber LLM_BASE_URL.
        # The live model lives in the forwarder + provenance instead.
        model=None,
        reasoning_effort=run.reasoning_effort,
        # values are str (a subset of str | None); list invariance needs the cast.
        prompts=cast("list[str | None] | None", run.prompts or None),
        environment=run.environment,
        sandbox_user=run.sandbox_user,
        agent_env=agent_env,
        # The seam that stops benchflow starting its own gateway / rewriting
        # LLM_BASE_URL — our replay proxy stays in front of the agent.
        usage_tracking="off",
        timeout=timeout,
        agent_idle_timeout=run.agent_idle_timeout_sec,
        jobs_dir=str(output_dir),
        rollout_name=rollout_name,
        source_provenance={
            "kind": "benchflow-continue",
            "continued_from": str(run.path),
            "original_error_category": run.error_category,
            "original_model": run.model,
            "live_model": live_model,
            "n_recorded_exchanges": run.n_recorded_exchanges,
        },
    )


class LiteLLMLiveForwarder:
    """Live forwarder backed by the in-process LiteLLM SDK.

    Past the replay cut-point the agent's request is sent to the real model and
    the final (non-streamed) ChatCompletion is returned for the proxy to emit.
    Resolving the provider route once keeps each call cheap; litellm reads
    provider credentials (e.g. ``GEMINI_API_KEY``) from the environment.
    """

    # Request fields worth forwarding verbatim when present.
    _PASSTHROUGH = (
        "tools",
        "tool_choice",
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "response_format",
        "reasoning_effort",
        "parallel_tool_calls",
    )

    def __init__(self, model: str, *, env: dict[str, str] | None = None) -> None:
        from benchflow.providers.litellm_config import resolve_litellm_route

        merged_env = {**os.environ, **(env or {})}
        route = resolve_litellm_route(model, merged_env)
        self.model = model
        self.upstream_model = route.upstream_model

    def build_kwargs(self, request_body: dict[str, Any]) -> dict[str, Any]:
        """Translate an OpenAI request body into ``litellm.completion`` kwargs."""
        kwargs: dict[str, Any] = {
            "model": self.upstream_model,
            "messages": request_body.get("messages") or [],
            "stream": False,
        }
        for key in self._PASSTHROUGH:
            if request_body.get(key) is not None:
                kwargs[key] = request_body[key]
        return kwargs

    def __call__(self, request_body: dict[str, Any]) -> dict[str, Any]:
        import litellm

        response = litellm.completion(**self.build_kwargs(request_body))
        dump = getattr(response, "model_dump", None)
        return dump() if callable(dump) else dict(response)


def stitched_trajectory_lines(
    original_llm_trajectory: Path, live_exchanges: list[LLMExchange]
) -> list[str]:
    """Build the continuous llm_trajectory: recorded prefix + live suffix.

    The recorded prefix is taken verbatim from the source file (already redacted
    and byte-identical to what the agent replayed); the live suffix is the
    exchanges the proxy captured after the cut-point, redacted on the way out.
    """
    lines: list[str] = []
    if original_llm_trajectory.is_file():
        for raw in original_llm_trajectory.read_text().splitlines():
            if raw.strip():
                lines.append(raw)
    for exchange in live_exchanges:
        raw = json.dumps(exchange.model_dump(mode="json"), default=str)
        lines.append(redact_trajectory_text(raw))
    return lines


def write_stitched_trajectory(
    rollout_dir: Path, original_llm_trajectory: Path, live_exchanges: list[LLMExchange]
) -> Path:
    """Write the stitched continuous trajectory into the new rollout folder."""
    out = rollout_dir / "trajectory" / "llm_trajectory.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = stitched_trajectory_lines(original_llm_trajectory, live_exchanges)
    out.write_text("\n".join(lines) + ("\n" if lines else ""))
    return out


def _host_proxy_binding(environment: str) -> tuple[str, str]:
    """(bind_host, advertise_host) so a Docker agent can reach the host proxy.

    Reuses benchflow's existing bridge-gateway logic so the container can talk
    back to the host-run proxy; non-docker environments use loopback.
    """
    from benchflow.providers.litellm_runtime import (
        _agent_endpoint_for_environment,
        _host_bind_address,
    )

    bind = _host_bind_address(environment)
    endpoint = _agent_endpoint_for_environment(0, environment, bind)
    advertise = endpoint.agent_base_url.split("//", 1)[-1].rsplit(":", 1)[0]
    return bind, advertise


async def continue_run(
    folder: str | Path,
    *,
    tasks_dir: str | Path | None = None,
    model: str | None = None,
    timeout: int | None = None,
    output_dir: str | Path | None = None,
    require_timeout: bool = False,
    strict_divergence: bool = False,
    replay_only: bool = False,
) -> ContinueResult:
    """Resume ``folder`` to completion via record-replay + live continuation.

    ``model`` overrides the live-continuation model (tests pass
    ``gemini-3.1-flash-lite-preview``); defaults to the original run's model.
    ``replay_only`` skips the live leg (rebuild-and-stop) — useful for testing
    the replay without provider credentials.
    """
    run = load_run_folder(folder, require_timeout=require_timeout)
    task_path = resolve_task_path(run, tasks_dir)

    live_model = model or run.model
    live_forwarder = None
    if not replay_only:
        if live_model is None:
            raise RunFolderError(
                "no live-continuation model: the run recorded no model and "
                "--model was not given."
            )
        live_forwarder = LiteLLMLiveForwarder(live_model)

    router = ReplayRouter(
        run.exchanges,
        live_forwarder=live_forwarder,
        strict_divergence=strict_divergence,
    )

    bind_host, advertise_host = _host_proxy_binding(run.environment)
    proxy = ReplayProxy(router, host=bind_host, advertise_host=advertise_host).start()

    out_root = Path(output_dir) if output_dir else run.path.parent / "continued"
    rollout_name = f"{run.task_name}__continued"

    try:
        from benchflow.rollout import Rollout

        config = build_rollout_config(
            run,
            task_path=task_path,
            live_model=live_model,
            agent_env=build_agent_env(proxy.base_url),
            timeout=timeout if timeout is not None else run.timeout_sec,
            output_dir=out_root,
            rollout_name=rollout_name,
        )
        rollout = await Rollout.create(config)
        result = await rollout.run()
        rollout_dir = Path(rollout._rollout_dir or (out_root / rollout_name))
    finally:
        proxy.stop()

    write_stitched_trajectory(
        rollout_dir,
        run.path / "trajectory" / "llm_trajectory.jsonl",
        router.live_exchanges,
    )

    return ContinueResult(
        rollout_dir=rollout_dir,
        rewards=getattr(result, "rewards", None),
        error=getattr(result, "error", None),
        n_recorded=run.n_recorded_exchanges,
        n_live=len(router.live_exchanges),
        divergences=router.divergences,
    )
