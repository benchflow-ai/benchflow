"""Orchestrate ``benchflow continue`` — boot, replay, continue live, stitch.

Reuses benchflow's normal :class:`~benchflow.rollout.Rollout` machinery (so the
new run produces a standard HF-compatible folder) but injects a record-replay
proxy in front of OpenHands:

- Host proxy mode uses ``usage_tracking="off"`` so ``ensure_litellm_runtime`` is
  a no-op and the
  ``agent_env`` we pass through (``LLM_BASE_URL`` → the replay proxy) is left
  untouched (``providers/litellm_runtime.py``).
- Sandbox proxy mode starts the provider LiteLLM proxy inside the sandbox and
  keeps the replay proxy on sandbox loopback, so remote environments such as
  Daytona do not need host-loopback connectivity.
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
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from benchflow._utils.text import describe_exception
from benchflow.continue_run.replay_proxy import ReplayProxy, ReplayRouter
from benchflow.continue_run.run_folder import RunFolder, RunFolderError, load_run_folder
from benchflow.continue_run.sandbox_proxy import (
    SandboxReplayProxy,
    sandbox_replay_base_url,
)
from benchflow.contracts import AgentProtocolError, SandboxStartupFailure
from benchflow.sandbox.providers import SANDBOX_MODEL_PROXY_PROVIDERS
from benchflow.scenes import compile_scenes_to_steps
from benchflow.trajectories.llm_capture_manifest import (
    LLM_TRAJECTORY_SCHEMA_VERSION,
    AuthMode,
    CaptureFidelity,
    CaptureSource,
    CaptureStatus,
    LLMRoleCapture,
    LLMTrajectoryManifest,
    capture_artifact_allows_training,
    read_llm_trajectory_manifest,
    successful_exchanges_have_positive_usage,
    write_llm_trajectory_manifest,
)
from benchflow.trajectories.types import LLMExchange, redact_trajectory_obj

logger = logging.getLogger(__name__)

# agent_env values handed to OpenHands so it talks to the replay proxy instead
# of a real provider. The model name is irrelevant (the proxy serves by index),
# but litellm needs an ``openai/`` prefix to treat LLM_BASE_URL as an
# OpenAI-compatible endpoint.
_REPLAY_API_KEY = "sk-benchflow-replay"
_REPLAY_MODEL = "openai/replay"
# Providers whose replay and model proxies must run inside the sandbox. This is
# the same placement contract used by the normal LiteLLM runtime.
_SANDBOX_LOCAL_REPLAY_ENVIRONMENTS = SANDBOX_MODEL_PROXY_PROVIDERS
_PROXY_MODES = frozenset({"auto", "host", "sandbox"})


@dataclass
class ContinueResult:
    """Outcome of a ``benchflow continue`` run."""

    rollout_dir: Path
    rewards: dict[str, Any] | None
    error: str | None
    n_recorded: int
    n_live: int
    divergences: int


def select_proxy_mode(requested: str, environment: str) -> str:
    """Resolve the replay proxy placement for a continuation run."""
    if requested not in _PROXY_MODES:
        raise RunFolderError(
            f"invalid proxy mode {requested!r}; expected one of "
            f"{', '.join(sorted(_PROXY_MODES))}."
        )
    if requested != "auto":
        return requested
    return "sandbox" if environment in _SANDBOX_LOCAL_REPLAY_ENVIRONMENTS else "host"


def continued_rollout_name(run: RunFolder) -> str:
    """Derive a stable, collision-resistant rollout name from the source run."""
    source = run.path.name or run.task_name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", source).strip("-")
    if not cleaned:
        cleaned = run.task_name or "run"
    return f"{cleaned}__continued"


@dataclass(frozen=True)
class ContinuedUsageSummary:
    """Token usage recovered from the stitched LLM trajectory."""

    n_input_tokens: int
    n_output_tokens: int
    n_cache_read_tokens: int
    n_cache_creation_tokens: int
    total_tokens: int
    recorded_total_tokens: int
    live_total_tokens: int
    usage_source: str

    def as_agent_result_patch(self) -> dict[str, Any]:
        return {
            "n_input_tokens": self.n_input_tokens,
            "n_output_tokens": self.n_output_tokens,
            "n_cache_read_tokens": self.n_cache_read_tokens,
            "n_cache_creation_tokens": self.n_cache_creation_tokens,
            "total_tokens": self.total_tokens,
            "usage_source": self.usage_source,
            "usage_details": {
                "source": "stitched_llm_trajectory",
                "recorded_total_tokens": self.recorded_total_tokens,
                "live_total_tokens": self.live_total_tokens,
            },
        }


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

    The recorded request/response payloads are preserved, while their metadata
    is promoted to the current schema so a newly stitched artifact can never be
    mistaken for sidecar-optional legacy data. The live suffix is redacted on
    the way out.
    """
    lines: list[str] = []
    if original_llm_trajectory.is_file():
        for raw in original_llm_trajectory.read_text().splitlines():
            if raw.strip():
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    lines.append(raw)
                    continue
                if not isinstance(payload, dict):
                    lines.append(raw)
                    continue
                metadata = payload.setdefault("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                    payload["metadata"] = metadata
                metadata["schema_version"] = LLM_TRAJECTORY_SCHEMA_VERSION
                lines.append(json.dumps(redact_trajectory_obj(payload), default=str))
    for exchange in live_exchanges:
        payload = redact_trajectory_obj(exchange.model_dump(mode="json"))
        lines.append(json.dumps(payload, default=str))
    return lines


def write_stitched_trajectory(
    rollout_dir: Path,
    original_llm_trajectory: Path,
    live_exchanges: list[LLMExchange],
    *,
    live_model: str | None = None,
    live_capture_trusted: bool = True,
) -> Path:
    """Write the stitched continuous trajectory into the new rollout folder."""
    out = rollout_dir / "trajectory" / "llm_trajectory.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = stitched_trajectory_lines(original_llm_trajectory, [])
    for exchange in live_exchanges:
        payload = exchange.model_dump(mode="json")
        metadata = payload.setdefault("metadata", {})
        metadata.update(
            {
                "agent": "openhands",
                "role": "agent",
                "model": live_model,
                "auth_mode": AuthMode.API_KEY.value,
                "capture_source": CaptureSource.LITELLM_PROXY.value,
                "capture_fidelity": (
                    CaptureFidelity.PROVIDER_WIRE.value
                    if live_capture_trusted
                    else CaptureFidelity.AGENT_SESSION.value
                ),
                "schema_version": LLM_TRAJECTORY_SCHEMA_VERSION,
                "request_complete": True,
                "response_complete": True,
                "role_attribution_complete": True,
                "payload_redacted": True,
            }
        )
        if not live_capture_trusted:
            metadata["capture_custody"] = "agent_writable_sandbox"
        lines.append(json.dumps(redact_trajectory_obj(payload), default=str))
    rendered = "\n".join(lines) + ("\n" if lines else "")
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(rendered)
    os.replace(temporary, out)
    return out


def refresh_stitched_trajectory_manifest(
    rollout_dir: Path,
    source_rollout_dir: Path,
    *,
    original_model: str | None,
    live_model: str | None,
    n_recorded: int,
    n_live: int,
    live_attempt_count: int,
    live_errors: list[str],
    live_capture_trusted: bool = True,
) -> LLMTrajectoryManifest:
    """Replace rollout-finalization provenance with the final stitched contract."""

    current_raw = read_llm_trajectory_manifest(rollout_dir)
    source_raw = read_llm_trajectory_manifest(source_rollout_dir)
    try:
        current = (
            LLMTrajectoryManifest.model_validate(current_raw)
            if current_raw is not None
            else None
        )
    except ValueError:
        current = None
    try:
        source = (
            LLMTrajectoryManifest.model_validate(source_raw)
            if source_raw is not None
            else None
        )
    except ValueError:
        source = None

    trajectory_path = rollout_dir / "trajectory" / "llm_trajectory.jsonl"
    trajectory_lines = [
        line for line in trajectory_path.read_text().splitlines() if line.strip()
    ]
    exchange_count = len(trajectory_lines)
    malformed_count = 0
    for line in trajectory_lines:
        try:
            LLMExchange.model_validate_json(line)
        except ValueError:
            malformed_count += 1
    rows_valid = malformed_count == 0
    parsed_rows: list[dict[str, Any]] = []
    if rows_valid:
        parsed_rows = [json.loads(line) for line in trajectory_lines]
    expected_count = n_recorded + live_attempt_count
    count_matches = exchange_count == expected_count
    source_rows: list[dict[str, Any]] = []
    try:
        source_rows = [
            json.loads(line)
            for line in (source_rollout_dir / "trajectory" / "llm_trajectory.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        source_rows = []
    source_allows_training = bool(
        source_raw is not None
        and len(source_rows) == n_recorded
        and capture_artifact_allows_training(source_raw, exchanges=source_rows)
    )
    live_capture_complete = live_attempt_count == n_live and not live_errors
    usage_complete = bool(
        parsed_rows and successful_exchanges_have_positive_usage(parsed_rows)
    )
    complete = (
        source_allows_training
        and count_matches
        and live_capture_complete
        and (live_capture_trusted or n_live == 0)
        and rows_valid
        and usage_complete
    )

    source_capture_source = source.capture_source if source else CaptureSource.NONE
    source_fidelity = source.capture_fidelity if source else CaptureFidelity.NONE
    source_auth = source.auth_mode if source else AuthMode.UNKNOWN
    if n_live:
        capture_source = (
            CaptureSource.LITELLM_PROXY
            if source_capture_source is CaptureSource.LITELLM_PROXY
            else CaptureSource.MIXED
        )
        live_fidelity = (
            CaptureFidelity.PROVIDER_WIRE
            if live_capture_trusted
            else CaptureFidelity.AGENT_SESSION
        )
        capture_fidelity = (
            live_fidelity if source_fidelity is live_fidelity else CaptureFidelity.MIXED
        )
        auth_mode = (
            AuthMode.API_KEY if source_auth is AuthMode.API_KEY else AuthMode.MIXED
        )
    else:
        capture_source = source_capture_source
        capture_fidelity = source_fidelity
        auth_mode = source_auth

    request_complete = bool(
        source
        and source.request_complete
        and count_matches
        and live_capture_complete
        and rows_valid
    )
    response_complete = bool(
        source
        and source.response_complete
        and count_matches
        and live_capture_complete
        and rows_valid
    )
    errors = list(source.errors) if source and not complete else []
    missing_fields = list(source.missing_fields) if source and not complete else []
    errors.extend(live_errors)
    if n_live and not live_capture_trusted:
        errors.append("sandbox replay capture shared root custody with the agent")
    if source is None:
        errors.append("source LLM trajectory manifest is missing or malformed")
        missing_fields.append("source_capture_provenance")
    elif not source_allows_training:
        errors.append("source LLM trajectory is not complete provider-wire capture")
    if not count_matches:
        errors.append(
            "stitched LLM trajectory count mismatch: "
            f"expected {expected_count}, found {exchange_count}"
        )
        missing_fields.append("exchange_count")
    if not rows_valid:
        errors.append(
            f"stitched LLM trajectory contains {malformed_count} malformed row(s)"
        )
        missing_fields.append("valid_provider_exchange")
    if rows_valid and not usage_complete:
        errors.append("stitched LLM trajectory lacks positive provider token usage")
        missing_fields.append("token_usage")
    if not live_capture_complete:
        missing_fields.append("live_provider_exchange")

    models = {
        value
        for value, present in (
            (original_model, n_recorded > 0),
            (live_model, live_attempt_count > 0),
        )
        if present and value
    }
    stitched_model = next(iter(models)) if len(models) == 1 else None
    manifest = LLMTrajectoryManifest(
        status=CaptureStatus.COMPLETE if complete else CaptureStatus.PARTIAL,
        capture_source=capture_source,
        capture_fidelity=capture_fidelity,
        auth_mode=auth_mode,
        agent="openhands",
        model=stitched_model,
        session_id=current.session_id if current else rollout_dir.name,
        exchange_count=exchange_count,
        request_complete=request_complete,
        response_complete=response_complete,
        payload_redacted=source.payload_redacted if source else True,
        started_at=current.started_at if current else datetime.now(UTC),
        finished_at=datetime.now(UTC),
        missing_fields=sorted(set(missing_fields)),
        errors=errors,
        role_captures=[
            LLMRoleCapture(
                role="agent",
                agent="openhands",
                model=stitched_model,
                auth_mode=auth_mode,
                capture_source=capture_source,
                capture_fidelity=capture_fidelity,
                exchange_count=exchange_count,
                request_complete=request_complete,
                response_complete=response_complete,
            )
        ],
    )
    write_llm_trajectory_manifest(rollout_dir, manifest)
    return manifest


def _usage_int(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def summarize_llm_trajectory_usage(
    trajectory_path: Path, *, n_recorded: int
) -> ContinuedUsageSummary:
    """Recover aggregate token usage from provider responses in a trajectory."""
    input_tokens = 0
    output_tokens = 0
    cache_read = 0
    cache_creation = 0
    total_tokens = 0
    recorded_total = 0
    live_total = 0
    for idx, raw in enumerate(trajectory_path.read_text().splitlines()):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        response = payload.get("response")
        if not isinstance(response, dict):
            continue
        body = response.get("body")
        if not isinstance(body, dict):
            continue
        usage = body.get("usage")
        if not isinstance(usage, dict):
            continue
        prompt = _usage_int(usage, "prompt_tokens", "input_tokens")
        completion = _usage_int(usage, "completion_tokens", "output_tokens")
        total = _usage_int(usage, "total_tokens")
        if total <= 0:
            total = prompt + completion
        input_tokens += prompt
        output_tokens += completion
        total_tokens += total
        cache_read += _usage_int(usage, "cache_read_input_tokens")
        cache_creation += _usage_int(usage, "cache_creation_input_tokens")
        if idx < n_recorded:
            recorded_total += total
        else:
            live_total += total
    return ContinuedUsageSummary(
        n_input_tokens=input_tokens,
        n_output_tokens=output_tokens,
        n_cache_read_tokens=cache_read,
        n_cache_creation_tokens=cache_creation,
        total_tokens=total_tokens,
        recorded_total_tokens=recorded_total,
        live_total_tokens=live_total,
        usage_source="provider_response" if total_tokens > 0 else "unavailable",
    )


def update_continued_metadata(
    rollout_dir: Path,
    *,
    live_model: str | None,
    usage: ContinuedUsageSummary,
    environment: str,
) -> None:
    """Patch output metadata that Rollout could not know while model=None.

    ``benchflow continue`` intentionally runs Rollout with ``model=None`` so
    provider resolution does not clobber the replay proxy. After the run, the
    HF-compatible artifacts still need the actual live model and token usage.
    The stitched LLM trajectory is authoritative for provider usage.
    """
    config_path = rollout_dir / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        config["model"] = live_model
        config.setdefault("source", {})["usage_source"] = "stitched_llm_trajectory"
        config["usage_tracking"] = {
            "requested": "required",
            "status": (
                "captured_from_stitched_llm_trajectory"
                if usage.total_tokens > 0
                else "unavailable"
            ),
            "environment": environment,
            "endpoint_kind": "sandbox"
            if environment in _SANDBOX_LOCAL_REPLAY_ENVIRONMENTS
            else "host",
            "usage_source": usage.usage_source,
        }
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    result_path = rollout_dir / "result.json"
    if not result_path.is_file():
        return
    result = json.loads(result_path.read_text())
    result["model"] = live_model
    agent_result = result.setdefault("agent_result", {})
    if isinstance(agent_result, dict):
        agent_result.update(usage.as_agent_result_patch())
    result["final_metrics"] = {
        "total_prompt_tokens": usage.n_input_tokens,
        "total_completion_tokens": usage.n_output_tokens,
        "total_cached_tokens": usage.n_cache_read_tokens,
        "total_cost_usd": agent_result.get("cost_usd")
        if isinstance(agent_result, dict)
        else None,
    }
    usage_tracking = result.setdefault("usage_tracking", {})
    if isinstance(usage_tracking, dict):
        usage_tracking["requested"] = "required"
        usage_tracking["usage_source"] = usage.usage_source
        usage_tracking["environment"] = environment
        usage_tracking["endpoint_kind"] = (
            "sandbox" if environment in _SANDBOX_LOCAL_REPLAY_ENVIRONMENTS else "host"
        )
        usage_tracking["status"] = (
            "captured_from_stitched_llm_trajectory"
            if usage.total_tokens > 0
            else usage_tracking.get("status", "off")
        )
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    from benchflow.trajectories.results import refresh_rollout_results_jsonl

    refresh_rollout_results_jsonl(rollout_dir)


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


async def _safe_sandbox_continuation_teardown(
    *,
    rollout: Any,
    replay_proxy: SandboxReplayProxy | None,
    provider_runtime: Any | None,
    stop_provider_runtime: Callable[[Any], Awaitable[None]],
    before_cleanup: Callable[[list[str]], Awaitable[None]] | None = None,
    after_cleanup: Callable[[list[str]], Awaitable[None]] | None = None,
) -> list[str]:
    """Stop continuation sidecars but always let Rollout write final artifacts."""
    errors: list[str] = []

    async def _capture(label: str, awaitable: Awaitable[Any]) -> None:
        try:
            await awaitable
        except Exception as exc:
            message = f"{label}: {describe_exception(exc)}"
            errors.append(message)
            logger.warning("Continuation teardown step failed: %s", message)

    if replay_proxy is not None:
        await _capture("sandbox replay proxy stop", replay_proxy.stop())
    if provider_runtime is not None:
        await _capture("provider runtime stop", stop_provider_runtime(provider_runtime))

    if errors and getattr(rollout, "_error", None) is None:
        rollout._error = "Continuation teardown warning: " + "; ".join(errors)

    if before_cleanup is not None:
        await _capture("continuation artifact write", before_cleanup(errors))

    await _capture("rollout cleanup", rollout.cleanup())
    if after_cleanup is not None:
        await _capture("continuation artifact finalization", after_cleanup(errors))

    return errors


def _result_after_sandbox_teardown(rollout: Any) -> Any | None:
    """Return a Rollout result, forcing artifact emission after cleanup warnings."""
    result = rollout.result
    if result is not None:
        return result
    if getattr(rollout, "_rollout_dir", None) is None:
        return None
    if getattr(rollout, "_phase", None) not in ("verified", "cleaned"):
        rollout._phase = "cleaned"
    return rollout.result


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
    proxy_mode: str = "auto",
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

    resolved_proxy_mode = select_proxy_mode(proxy_mode, run.environment)
    if replay_only and resolved_proxy_mode == "sandbox":
        raise RunFolderError("replay-only mode is only supported with host proxy mode.")

    out_root = Path(output_dir) if output_dir else run.path.parent / "continued"
    rollout_name = continued_rollout_name(run)

    if resolved_proxy_mode == "sandbox":
        if live_model is None:
            raise RunFolderError(
                "no live-continuation model: the run recorded no model and "
                "--model was not given."
            )
        return await _continue_run_with_sandbox_proxy(
            run,
            task_path=task_path,
            live_model=live_model,
            timeout=timeout,
            output_dir=out_root,
            rollout_name=rollout_name,
            strict_divergence=strict_divergence,
        )

    router = ReplayRouter(
        run.exchanges,
        live_forwarder=live_forwarder,
        strict_divergence=strict_divergence,
    )

    bind_host, advertise_host = _host_proxy_binding(run.environment)
    proxy = ReplayProxy(router, host=bind_host, advertise_host=advertise_host).start()

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

    stitched_path = write_stitched_trajectory(
        rollout_dir,
        run.path / "trajectory" / "llm_trajectory.jsonl",
        router.live_exchanges,
        live_model=live_model,
    )
    refresh_stitched_trajectory_manifest(
        rollout_dir,
        run.path,
        original_model=run.model,
        live_model=live_model,
        n_recorded=run.n_recorded_exchanges,
        n_live=len(router.live_exchanges),
        live_attempt_count=router.live_attempt_count,
        live_errors=list(router.live_errors),
    )
    update_continued_metadata(
        rollout_dir,
        live_model=live_model,
        usage=summarize_llm_trajectory_usage(
            stitched_path,
            n_recorded=run.n_recorded_exchanges,
        ),
        environment=run.environment,
    )

    return ContinueResult(
        rollout_dir=rollout_dir,
        rewards=getattr(result, "rewards", None),
        error=getattr(result, "error", None),
        n_recorded=run.n_recorded_exchanges,
        n_live=len(router.live_exchanges),
        divergences=router.divergences,
    )


async def _continue_run_with_sandbox_proxy(
    run: RunFolder,
    *,
    task_path: Path,
    live_model: str,
    timeout: int | None,
    output_dir: Path,
    rollout_name: str,
    strict_divergence: bool,
) -> ContinueResult:
    """Run continuation with replay and provider proxies inside the sandbox."""
    from benchflow.providers.runtime import (
        ensure_litellm_runtime,
        stop_provider_runtime,
    )
    from benchflow.rollout import Rollout

    config = build_rollout_config(
        run,
        task_path=task_path,
        live_model=live_model,
        agent_env=build_agent_env(sandbox_replay_base_url()),
        timeout=timeout if timeout is not None else run.timeout_sec,
        output_dir=output_dir,
        rollout_name=rollout_name,
    )
    rollout = await Rollout.create(config)
    replay_proxy: SandboxReplayProxy | None = None
    provider_runtime: Any | None = None
    result: Any | None = None
    pending_acp_error: AgentProtocolError | None = None
    agent_timed_out = False
    rollout_dir: Path | None = None
    live_exchanges: list[LLMExchange] = []
    artifacts_written = False

    async def _write_artifacts(
        teardown_errors: list[str] | None = None,
        *,
        force: bool = False,
    ) -> None:
        nonlocal artifacts_written, live_exchanges, result, rollout_dir
        if artifacts_written and not force:
            return
        result = _result_after_sandbox_teardown(rollout)
        if result is None:
            return
        rollout_dir = Path(rollout._rollout_dir or (output_dir / rollout_name))
        live_exchanges = replay_proxy.live_exchanges if replay_proxy is not None else []
        stitched_path = write_stitched_trajectory(
            rollout_dir,
            run.path / "trajectory" / "llm_trajectory.jsonl",
            live_exchanges,
            live_model=live_model,
            live_capture_trusted=bool(
                config.sandbox_user and config.sandbox_user not in {"root", "0"}
            ),
        )
        refresh_stitched_trajectory_manifest(
            rollout_dir,
            run.path,
            original_model=run.model,
            live_model=live_model,
            n_recorded=run.n_recorded_exchanges,
            n_live=len(live_exchanges),
            live_attempt_count=(
                replay_proxy.live_attempt_count if replay_proxy is not None else 0
            ),
            live_errors=[
                *(replay_proxy.live_errors if replay_proxy is not None else []),
                *(teardown_errors or []),
            ],
            live_capture_trusted=bool(
                config.sandbox_user and config.sandbox_user not in {"root", "0"}
            ),
        )
        update_continued_metadata(
            rollout_dir,
            live_model=live_model,
            usage=summarize_llm_trajectory_usage(
                stitched_path,
                n_recorded=run.n_recorded_exchanges,
            ),
            environment=run.environment,
        )
        artifacts_written = True

    try:
        await rollout.setup()
        await rollout.start()
        await rollout.install_agent()

        provider_env = rollout._planes.resolve_agent_env("openhands", live_model, {})
        provider_env, provider_runtime = await ensure_litellm_runtime(
            agent="openhands",
            agent_env=provider_env,
            model=live_model,
            runtime=None,
            environment=run.environment,
            session_id=rollout_name,
            usage_tracking="required",
            sandbox=rollout.env,
            sandbox_user=config.sandbox_user,
        )
        replay_proxy = await SandboxReplayProxy.start(
            sandbox=rollout.env,
            recorded=run.exchanges,
            upstream_url=provider_env["LLM_BASE_URL"],
            upstream_api_key=provider_env["LLM_API_KEY"],
            upstream_model=provider_env["LLM_MODEL"],
            strict_divergence=strict_divergence,
        )

        try:
            if config.user is not None:
                await rollout._run_user_loop()
            else:
                await rollout._run_steps(
                    compile_scenes_to_steps(
                        config.effective_scenes,
                        default_prompt=(
                            rollout._resolved_prompts[0]
                            if rollout._resolved_prompts
                            else None
                        ),
                    )
                )
        except TimeoutError as exc:
            agent_timed_out = True
            detail = str(exc).strip()
            rollout._error = detail or f"Agent timed out after {rollout._timeout}s"
            rollout._diagnostics.capture_idle(exc)
            logger.error(rollout._error)

        if not config.skip_verify:
            await rollout.verify()
            if (
                agent_timed_out
                and rollout._rewards is None
                and rollout._verifier_error is None
            ):
                rollout._rewards = {"reward": 0.0}
                rollout._verifier_error = None

    except TimeoutError as exc:
        detail = str(exc).strip()
        rollout._error = detail or f"Agent timed out after {rollout._timeout}s"
        rollout._diagnostics.capture_idle(exc)
        logger.error(rollout._error)
    except ConnectionError as exc:
        rollout._error = str(exc)
        rollout._diagnostics.capture_transport(exc)
        await rollout._probe_sandbox_health()
        logger.error("Agent connection lost: %s", rollout._error)
    except SandboxStartupFailure as exc:
        rollout._error = f"Sandbox startup failed: {exc}"
        rollout._diagnostics.set(exc.diagnostic)
        logger.error(rollout._error)
    except AgentProtocolError as exc:
        pending_acp_error = exc
        rollout._error = str(exc)
        logger.error(str(exc))
    except Exception as exc:
        rollout._error = str(exc)
        logger.error("Run failed", exc_info=True)
    finally:
        await _safe_sandbox_continuation_teardown(
            rollout=rollout,
            replay_proxy=replay_proxy,
            provider_runtime=provider_runtime,
            stop_provider_runtime=stop_provider_runtime,
            before_cleanup=_write_artifacts,
            after_cleanup=lambda errors: _write_artifacts(errors, force=True),
        )

    if pending_acp_error is not None:
        rollout._error = rollout._classify_acp_error(pending_acp_error)
        logger.error(rollout._error)

    await _write_artifacts()
    if rollout_dir is None:
        rollout_dir = Path(rollout._rollout_dir or (output_dir / rollout_name))

    return ContinueResult(
        rollout_dir=rollout_dir,
        rewards=getattr(result, "rewards", None),
        error=getattr(result, "error", None),
        n_recorded=run.n_recorded_exchanges,
        n_live=len(live_exchanges),
        divergences=0,
    )
