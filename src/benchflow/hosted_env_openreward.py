"""OpenReward (ORS) hosted environment runner.

Runs environments hosted on OpenReward — the reference deployment of the
Open Reward Standard (https://openrewardstandard.io) — and lands their
evidence in BenchFlow's rollout artifact contract, mirroring the
PrimeIntellect Verifiers runner in :mod:`benchflow.hosted_env`.

Unlike Verifiers (where ``vf-eval`` owns the agent loop and we reconstruct
artifacts afterwards), the ORS client hands *us* the loop: a session yields
a prompt and tools, the model picks tool calls, and every
``session.call_tool`` returns a ``ToolOutput`` carrying ``blocks`` /
``reward`` / ``finished`` / ``metadata``. This module drives that episode
loop, captures each step as native ACP-shaped trajectory events plus reward
events, and emits the standard scored artifacts — ``result.json``,
``rewards.jsonl``, ``trajectory/acp_trajectory.jsonl``, ``config.json``,
``timing.json``, ``prompts.json``, and ``trainer/verifiers.jsonl`` — with
``source.type="hosted_env"`` / ``trajectory_source="hosted_env"`` lineage.
Raw per-episode evidence is preserved under ``hosted_env/`` for forensics.

The model side goes through BenchFlow's existing provider plumbing: an
explicit ``BENCHFLOW_PROVIDER_*`` contract (e.g. injected by the LiteLLM
runtime) wins, otherwise the provider registry in
:mod:`benchflow.agents.providers` resolves an OpenAI-compatible
chat-completions endpoint for the model. Episodes are driven over that
endpoint with the environment's ``list_tools(format="openai")`` schema.

Primary sources for the hosted API surface (verified 2026-06-10):

- https://openreward.ai — platform; built on the Open Reward Standard.
- https://docs.openreward.ai/quickstart.md — ``OpenReward()`` client,
  ``environments.get(name="GeneralReasoning/CTF")`` addressing,
  ``environment.list_tasks(split=...)`` / ``list_tools(format="openai")``,
  ``with environment.session(task=task) as session``, ``session.get_prompt()``
  (block list; ``prompt[0].text``), ``session.call_tool(name, args)`` with
  JSON-parsed arguments, and the ``finished`` / ``reward`` episode loop.
- https://pypi.org/project/openreward/ — ``pip install openreward``
  (Python 3.11+), ``orwd`` CLI, ``ToolOutput`` fields
  (``blocks``/``reward``/``finished``/``metadata``), auth via
  ``OPENREWARD_API_KEY`` (+ ``OPENREWARD_URL`` base-URL override).
- https://docs.openreward.ai/environments/using-environment-variants.md —
  variant selection via the ``variant`` parameter of ``environments.get``.
- https://docs.openreward.ai/environments/ways-to-access-tasks.md — task
  dicts are environment-defined JSON objects (e.g. ``{"task_id": ...}``).
- https://docs.openreward.ai/environments/evaluation.md — the page does not
  state a general aggregation rule; our convention (every attempted task in
  the denominator, errored/unscored episodes count as 0) is consistent with
  its documented pass@1 example, which counts ``r == 1`` results (skipping
  ``None``) over ``num_samples``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from benchflow._utils.result_metadata import (
    final_metrics_from_agent_result,
    trajectory_summary_from_events,
)
from benchflow.diagnostics import RolloutDiagnostics
from benchflow.hosted_env import (
    HostedEnvError,
    HostedEnvRef,
    _hosted_source_provenance,
    _write_hosted_rewards_jsonl,
    normalize_verifiers_model,
)
from benchflow.trajectories.types import (
    redact_acp_trajectory_jsonl,
    redact_trajectory_text,
)

logger = logging.getLogger(__name__)

OPENREWARD_PROVIDER = "openreward"
OPENREWARD_API_KEY_ENV = "OPENREWARD_API_KEY"
OPENREWARD_RUNNER = "openreward"
_OPENAI_COMPLETIONS = "openai-completions"
_CHAT_TIMEOUT_SEC = 600.0
_DEFAULT_SPLIT = "train"


@dataclass(frozen=True)
class ModelEndpoint:
    """Resolved OpenAI-compatible chat-completions endpoint for the model."""

    base_url: str
    api_key: str
    model_id: str
    provider: str


@dataclass
class ChatTurn:
    """One assistant turn from the model endpoint.

    ``message`` is the OpenAI chat-completions assistant message dict
    (``content`` and/or ``tool_calls``); ``usage`` is the response-level
    token usage dict when the provider reports one.
    """

    message: dict[str, Any]
    usage: dict[str, Any] | None = None


ChatCompletionFn = Callable[..., ChatTurn]
ClientFactory = Callable[[], Any]


@dataclass
class OpenRewardRunConfig:
    """Configuration for running an OpenReward-hosted environment."""

    source_env: HostedEnvRef
    model: str
    env_args: dict[str, Any] = field(default_factory=dict)
    agent: str = ""
    jobs_dir: Path = Path("jobs")
    num_examples: int = 1
    max_turns: int = 16
    max_tokens: int = 1024
    temperature: float = 0.0

    @property
    def split(self) -> str:
        """Task split, selectable via ``--source-env-arg split=...``."""
        return str(self.env_args.get("split") or _DEFAULT_SPLIT)


@dataclass
class EpisodeOutcome:
    """Evidence from one driven episode (one task)."""

    example_index: int
    task: Any
    prompt: str
    events: list[dict[str, Any]]
    reward: float | None
    finished: bool
    truncated: bool
    n_tool_calls: int
    usage: dict[str, int]
    error: str | None

    @property
    def task_id(self) -> str:
        if isinstance(self.task, dict):
            raw = self.task.get("task_id") or self.task.get("id")
            if raw is not None:
                return str(raw)
        return f"example_{self.example_index}"


@dataclass
class OpenRewardRunResult:
    """Result from an OpenReward-hosted run (CLI display contract matches
    :class:`benchflow.hosted_env.HostedEnvRunResult`)."""

    source_env: HostedEnvRef
    run_dir: Path
    model: str
    normalized_model: str
    reward: float | None
    total_tool_calls: int
    episodes: list[EpisodeOutcome]
    error: str | None


def resolve_model_endpoint(
    model: str,
    env: Mapping[str, str],
) -> ModelEndpoint:
    """Resolve the model to an OpenAI-compatible endpoint, fail-closed.

    Resolution order:

    1. An explicit ``BENCHFLOW_PROVIDER_BASE_URL`` contract in *env* (the
       LiteLLM-runtime / pre-resolved agent-env path) wins.
    2. Otherwise the model's registered provider prefix
       (:func:`benchflow.agents.providers.find_provider`) supplies the
       ``openai-completions`` endpoint and API-key env var. Bare OpenAI
       model ids are normalized via
       :func:`benchflow.hosted_env.normalize_verifiers_model`.

    Anything else — unknown model, provider without an openai-completions
    endpoint, non-API-key auth, missing key — raises :class:`HostedEnvError`.
    """
    from benchflow.agents.providers import (
        find_provider,
        resolve_base_url,
        strip_provider_prefix,
    )

    if not model:
        raise HostedEnvError("--model is required for openreward source-env runs")

    protocol = (env.get("BENCHFLOW_PROVIDER_PROTOCOL") or "").strip()
    explicit_base = (env.get("BENCHFLOW_PROVIDER_BASE_URL") or "").strip()
    if explicit_base:
        if protocol and protocol != _OPENAI_COMPLETIONS:
            raise HostedEnvError(
                "The openreward runner drives models over openai-completions; "
                f"BENCHFLOW_PROVIDER_PROTOCOL={protocol!r} is not supported."
            )
        model_id = (env.get("BENCHFLOW_PROVIDER_MODEL") or "").strip()
        return ModelEndpoint(
            base_url=explicit_base,
            api_key=(env.get("BENCHFLOW_PROVIDER_API_KEY") or "").strip(),
            model_id=model_id or strip_provider_prefix(model),
            provider=(env.get("BENCHFLOW_PROVIDER_NAME") or "").strip() or "explicit",
        )

    normalized = normalize_verifiers_model(model)
    found = find_provider(normalized)
    if found is None:
        raise HostedEnvError(
            f"No OpenAI-compatible provider is registered for model {model!r}. "
            "Use a registered provider prefix (e.g. openai/, deepseek/) or set "
            "BENCHFLOW_PROVIDER_BASE_URL / BENCHFLOW_PROVIDER_API_KEY explicitly."
        )
    name, cfg = found
    if cfg.api_protocol != _OPENAI_COMPLETIONS and not cfg.endpoints.get(
        _OPENAI_COMPLETIONS
    ):
        raise HostedEnvError(
            f"Provider {name!r} does not expose an openai-completions endpoint; "
            "the openreward runner cannot drive this model. Set "
            "BENCHFLOW_PROVIDER_BASE_URL to an OpenAI-compatible endpoint instead."
        )
    if cfg.auth_type != "api_key":
        raise HostedEnvError(
            f"Provider {name!r} uses {cfg.auth_type!r} auth, which the openreward "
            "runner does not support. Set BENCHFLOW_PROVIDER_BASE_URL / "
            "BENCHFLOW_PROVIDER_API_KEY explicitly (e.g. via the LiteLLM runtime)."
        )
    try:
        base_url = resolve_base_url(cfg, dict(env), protocol=_OPENAI_COMPLETIONS)
    except KeyError as exc:
        raise HostedEnvError(str(exc.args[0]) if exc.args else str(exc)) from exc
    if not base_url:
        raise HostedEnvError(
            f"Provider {name!r} has no fixed endpoint; set "
            "BENCHFLOW_PROVIDER_BASE_URL to the server's OpenAI-compatible URL."
        )
    api_key = (env.get(cfg.auth_env) or "").strip() if cfg.auth_env else ""
    if not api_key:
        raise HostedEnvError(
            f"{cfg.auth_env or 'an API key'} is required for provider {name!r} "
            f"(model {model!r})."
        )
    return ModelEndpoint(
        base_url=base_url,
        api_key=api_key,
        model_id=strip_provider_prefix(normalized),
        provider=name,
    )


def _default_client_factory(env: Mapping[str, str]) -> Any:
    """Build the official OpenReward client, fail-closed on auth/install.

    The client authenticates via the ``OPENREWARD_API_KEY`` process
    environment variable (https://pypi.org/project/openreward/); we check it
    up front so a missing key fails before any network call. Because the SDK
    reads ``os.environ`` internally, the validated key from *env* is mirrored
    into the process environment before construction — an injected mapping
    therefore works even when it does not alias ``os.environ``.
    """
    api_key = (env.get(OPENREWARD_API_KEY_ENV) or "").strip()
    if not api_key:
        raise HostedEnvError(
            f"{OPENREWARD_API_KEY_ENV} is required to run OpenReward-hosted "
            "environments (see https://docs.openreward.ai/quickstart)."
        )
    try:
        from openreward import OpenReward
    except ImportError as exc:
        raise HostedEnvError(
            "The 'openreward' package is required for openreward source-env "
            "runs: pip install openreward (Python 3.11+, "
            "https://pypi.org/project/openreward/)."
        ) from exc
    os.environ[OPENREWARD_API_KEY_ENV] = api_key
    return OpenReward()


def _default_chat_completion(
    *,
    endpoint: ModelEndpoint,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
) -> ChatTurn:
    """POST one OpenAI chat-completions request to the resolved endpoint."""
    import httpx

    payload: dict[str, Any] = {
        "model": endpoint.model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
    headers = (
        {"Authorization": f"Bearer {endpoint.api_key}"} if endpoint.api_key else {}
    )
    response = httpx.post(
        endpoint.base_url.rstrip("/") + "/chat/completions",
        json=payload,
        headers=headers,
        timeout=_CHAT_TIMEOUT_SEC,
    )
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise HostedEnvError(
            f"Model endpoint returned no choices: {json.dumps(data)[:200]}"
        )
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise HostedEnvError(
            f"Model endpoint returned no assistant message: "
            f"{json.dumps(choices[0])[:200]}"
        )
    usage = data.get("usage")
    return ChatTurn(message=message, usage=usage if isinstance(usage, dict) else None)


def run_openreward_env(
    config: OpenRewardRunConfig,
    *,
    env: Mapping[str, str] | None = None,
    client_factory: ClientFactory | None = None,
    chat_completion: ChatCompletionFn | None = None,
) -> OpenRewardRunResult:
    """Run an OpenReward-hosted environment and emit contract artifacts.

    *client_factory* / *chat_completion* are injection seams for tests (a
    faked ORS client and model transport); production runs use the official
    ``openreward`` client and an httpx chat-completions transport.
    """
    ref = config.source_env
    if ref.provider != OPENREWARD_PROVIDER:
        raise HostedEnvError(
            f"run_openreward_env executes openreward references; "
            f"{ref.provider!r} references run through benchflow.hosted_env."
        )
    if config.num_examples < 1:
        raise HostedEnvError("--source-env-num-examples must be >= 1")
    if config.max_turns < 1:
        raise HostedEnvError("--source-env-max-turns must be >= 1")

    run_env: Mapping[str, str] = dict(env) if env is not None else os.environ.copy()
    endpoint = resolve_model_endpoint(config.model, run_env)
    chat = chat_completion or _default_chat_completion
    client = client_factory() if client_factory else _default_client_factory(run_env)

    try:
        get_kwargs: dict[str, Any] = {"name": ref.env_id}
        if ref.version:
            get_kwargs["variant"] = ref.version
        environment = client.environments.get(**get_kwargs)
        tools = list(environment.list_tools(format="openai") or [])
        tasks = list(environment.list_tasks(split=config.split) or [])
    except HostedEnvError:
        raise
    except Exception as exc:
        raise HostedEnvError(
            f"Could not resolve OpenReward environment {ref.env_uid}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not tasks:
        raise HostedEnvError(
            f"OpenReward environment {ref.env_uid} returned no tasks for "
            f"split {config.split!r}."
        )

    run_dir = _create_run_dir(config)
    started_at = datetime.now(UTC)
    episodes = [
        _run_episode(
            environment,
            task,
            example_index=i,
            chat=chat,
            endpoint=endpoint,
            tools=tools,
            max_turns=config.max_turns,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        for i, task in enumerate(tasks[: config.num_examples])
    ]
    finished_at = datetime.now(UTC)

    # Every attempted task is in the denominator; errored/unscored episodes
    # count as 0. This is consistent with the pass@1 example documented at
    # docs.openreward.ai/environments/evaluation.md (no general aggregation
    # rule is stated there).
    reward = round(
        sum(ep.reward if ep.reward is not None else 0.0 for ep in episodes)
        / len(episodes),
        6,
    )
    errors = [f"{ep.task_id}: {ep.error}" for ep in episodes if ep.error]
    result = OpenRewardRunResult(
        source_env=ref,
        run_dir=run_dir,
        model=config.model,
        normalized_model=endpoint.model_id,
        reward=reward,
        total_tool_calls=sum(ep.n_tool_calls for ep in episodes),
        episodes=episodes,
        error="; ".join(errors) if errors else None,
    )
    _write_run_artifacts(
        result,
        config,
        endpoint,
        started_at=started_at,
        finished_at=finished_at,
    )
    return result


def _create_run_dir(config: OpenRewardRunConfig) -> Path:
    """Collision-safe run dir under ``jobs/hosted-env/`` (same scheme as
    the Verifiers runner)."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d__%H-%M-%S-%f")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", config.source_env.env_id)
    jobs_dir = config.jobs_dir.expanduser().resolve()
    run_id = f"{timestamp}__pid-{os.getpid()}__{uuid4().hex[:8]}"
    run_dir = jobs_dir / "hosted-env" / f"{safe_name}__{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _run_episode(
    environment: Any,
    task: Any,
    *,
    example_index: int,
    chat: ChatCompletionFn,
    endpoint: ModelEndpoint,
    tools: list[dict[str, Any]],
    max_turns: int,
    max_tokens: int,
    temperature: float,
) -> EpisodeOutcome:
    """Drive one session to completion; failures are episode-local.

    Any exception from the hosted client or the model transport fails this
    episode closed (``reward=None`` + ``error``) while preserving the events
    captured so far — the run-level aggregate then scores it as 0.
    """
    events: list[dict[str, Any]] = []
    usage = {"input": 0, "output": 0}
    try:
        with environment.session(task=task) as session:
            return _drive_episode(
                session,
                events=events,
                usage=usage,
                example_index=example_index,
                task=task,
                chat=chat,
                endpoint=endpoint,
                tools=tools,
                max_turns=max_turns,
                max_tokens=max_tokens,
                temperature=temperature,
            )
    except Exception as exc:  # fail-closed per episode
        logger.warning(
            "openreward episode %d failed: %s: %s",
            example_index,
            type(exc).__name__,
            exc,
        )
        return EpisodeOutcome(
            example_index=example_index,
            task=task,
            prompt=_first_user_text(events),
            events=events,
            reward=None,
            finished=False,
            truncated=False,
            n_tool_calls=sum(1 for e in events if e.get("type") == "tool_call"),
            usage=usage,
            error=f"{type(exc).__name__}: {exc}",
        )


def _drive_episode(
    session: Any,
    *,
    events: list[dict[str, Any]],
    usage: dict[str, int],
    example_index: int,
    task: Any,
    chat: ChatCompletionFn,
    endpoint: ModelEndpoint,
    tools: list[dict[str, Any]],
    max_turns: int,
    max_tokens: int,
    temperature: float,
) -> EpisodeOutcome:
    """The ORS episode loop (docs.openreward.ai/quickstart.md):

    prompt -> model turn -> ``session.call_tool`` per tool call -> feed each
    ``ToolOutput``'s blocks back as a tool message -> stop when a tool output
    sets ``finished`` (or the model stops calling tools / hits *max_turns*).
    """
    prompt_text = _prompt_to_text(session.get_prompt())
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt_text}]
    events.append(_event("user_message", example_index, text=prompt_text))

    finished = False
    truncated = False
    last_reward: float | None = None
    n_tool_calls = 0
    for _turn in range(max_turns):
        turn = chat(
            endpoint=endpoint,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if turn.usage:
            usage["input"] += int(turn.usage.get("prompt_tokens") or 0)
            usage["output"] += int(turn.usage.get("completion_tokens") or 0)

        content = turn.message.get("content")
        tool_calls = turn.message.get("tool_calls") or []
        assistant: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        messages.append(assistant)
        if isinstance(content, str) and content:
            events.append(_event("agent_message", example_index, text=content))

        if not tool_calls:
            break  # the model stopped without a finishing tool call

        # Execute the full tool-call batch before checking the finished
        # flag — the documented quickstart loop does the same.
        for tool_call in tool_calls:
            name, arguments, call_id = _parse_tool_call(tool_call)
            output = session.call_tool(name, arguments)
            output_text = _blocks_to_text(getattr(output, "blocks", None))
            n_tool_calls += 1
            messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": output_text}
            )
            events.append(
                _event(
                    "tool_call",
                    example_index,
                    tool_call_id=call_id,
                    kind="other",
                    title=name,
                    status="completed",
                    content=[{"type": "content", "text": output_text}],
                )
            )
            step_reward = getattr(output, "reward", None)
            if isinstance(step_reward, (int, float)) and not isinstance(
                step_reward, bool
            ):
                last_reward = float(step_reward)
                events.append(
                    _event(
                        "reward",
                        example_index,
                        value=float(step_reward),
                        source="openreward",
                    )
                )
            if bool(getattr(output, "finished", False)):
                finished = True
        if finished:
            break
    else:
        truncated = True

    return EpisodeOutcome(
        example_index=example_index,
        task=task,
        prompt=prompt_text,
        events=events,
        reward=last_reward,
        finished=finished,
        truncated=truncated,
        n_tool_calls=n_tool_calls,
        usage=usage,
        error=None,
    )


def _event(event_type: str, example_index: int, **fields: Any) -> dict[str, Any]:
    return {
        "type": event_type,
        "ts": datetime.now(UTC).isoformat(),
        "example_index": example_index,
        **fields,
    }


def _parse_tool_call(tool_call: Any) -> tuple[str, dict[str, Any], str | None]:
    """Extract (name, arguments, call_id) from an OpenAI-format tool call.

    Malformed tool calls raise — the episode fails closed rather than
    sending the environment a guessed payload.
    """
    if not isinstance(tool_call, dict):
        raise HostedEnvError(f"Malformed tool call from model: {tool_call!r}")
    function = tool_call.get("function") or {}
    name = function.get("name")
    if not name or not isinstance(name, str):
        raise HostedEnvError(f"Tool call without a function name: {tool_call!r}")
    raw_arguments = function.get("arguments")
    if raw_arguments is None or raw_arguments == "":
        arguments: dict[str, Any] = {}
    elif isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise HostedEnvError(
                f"Tool call {name!r} has non-JSON arguments: {raw_arguments[:200]!r}"
            ) from exc
    elif isinstance(raw_arguments, dict):
        arguments = raw_arguments
    else:
        raise HostedEnvError(
            f"Tool call {name!r} has unsupported arguments type: "
            f"{type(raw_arguments).__name__}"
        )
    if not isinstance(arguments, dict):
        raise HostedEnvError(
            f"Tool call {name!r} arguments must be a JSON object, got: {arguments!r}"
        )
    call_id = tool_call.get("id")
    return name, arguments, str(call_id) if call_id is not None else None


def _block_text(block: Any) -> str:
    """Text of one ORS content block (object with ``.text`` or plain dict)."""
    text = getattr(block, "text", None)
    if text is None and isinstance(block, dict):
        text = block.get("text")
    return text if isinstance(text, str) else ""


def _blocks_to_text(blocks: Any) -> str:
    if isinstance(blocks, (list, tuple)):
        return "".join(_block_text(b) for b in blocks)
    return _block_text(blocks) if blocks is not None else ""


def _prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    return _blocks_to_text(prompt)


def _first_user_text(events: list[dict[str, Any]]) -> str:
    for event in events:
        if event.get("type") == "user_message":
            return str(event.get("text") or "")
    return ""


def _write_run_artifacts(
    result: OpenRewardRunResult,
    config: OpenRewardRunConfig,
    endpoint: ModelEndpoint,
    *,
    started_at: datetime,
    finished_at: datetime,
) -> None:
    """Write the rollout-contract artifacts plus openreward-specific evidence.

    The contract files match :func:`benchflow.hosted_env._write_run_artifacts`
    so dashboards, release checks, and trainer exports treat OpenReward runs
    exactly like Verifiers runs and native rollouts.
    """
    ref = result.source_env
    run_dir = result.run_dir
    (run_dir / "trajectory").mkdir(parents=True, exist_ok=True)
    (run_dir / "agent").mkdir(parents=True, exist_ok=True)
    (run_dir / "verifier").mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    hosted_dir = run_dir / "hosted_env"
    hosted_dir.mkdir(parents=True, exist_ok=True)

    trajectory = [event for ep in result.episodes for event in ep.events]
    rewards = _build_rewards_dict(result)
    prompts = _collect_prompts(result, config)
    timing = {"total": round((finished_at - started_at).total_seconds(), 1)}
    source_provenance = _hosted_source_provenance(
        ref,
        runner=OPENREWARD_RUNNER,
        env_args=config.env_args,
    )

    # OpenReward-specific evidence (forensics, debugging). Redacted line by
    # line — episode tasks/errors may echo headers or env material.
    hosted_payload = {
        "source_env": ref.env_id,
        "source_env_variant": ref.version,
        "env_uid": ref.env_uid,
        "hub_url": ref.hub_url,
        "runner": OPENREWARD_RUNNER,
        "agent": config.agent or None,
        "model": result.model,
        "normalized_model": result.normalized_model,
        "model_provider": endpoint.provider,
        "model_base_url": endpoint.base_url,
        "env_args": config.env_args,
        "split": config.split,
        "num_episodes": len(result.episodes),
        "rewards": rewards,
        "total_tool_calls": result.total_tool_calls,
        "error": result.error,
    }
    (hosted_dir / "hosted_run.json").write_text(
        redact_trajectory_text(json.dumps(hosted_payload, indent=2, default=str))
    )
    episode_lines = [
        json.dumps(
            {
                "example_index": ep.example_index,
                "task_id": ep.task_id,
                "task": ep.task,
                "reward": ep.reward,
                "finished": ep.finished,
                "truncated": ep.truncated,
                "n_tool_calls": ep.n_tool_calls,
                "usage": ep.usage,
                "error": ep.error,
            },
            default=str,
        )
        for ep in result.episodes
    ]
    (hosted_dir / "episodes.jsonl").write_text(
        redact_trajectory_text("\n".join(episode_lines) + "\n")
    )

    # Rollout-contract artifacts.
    (run_dir / "trajectory" / "acp_trajectory.jsonl").write_text(
        redact_acp_trajectory_jsonl(trajectory) + ("\n" if trajectory else "")
    )

    n_input = sum(ep.usage.get("input", 0) for ep in result.episodes)
    n_output = sum(ep.usage.get("output", 0) for ep in result.episodes)
    has_usage = (n_input + n_output) > 0
    agent_result = {
        "n_tool_calls": result.total_tool_calls,
        "n_prompts": len(prompts),
        "n_input_tokens": n_input if has_usage else None,
        "n_output_tokens": n_output if has_usage else None,
        "n_cache_read_tokens": None,
        "n_cache_creation_tokens": None,
        "total_tokens": (n_input + n_output) if has_usage else None,
        "cost_usd": None,
        "usage_source": "provider_response" if has_usage else "unavailable",
        "price_source": None,
    }

    result_payload: dict[str, Any] = {
        "task_name": ref.env_uid,
        "rollout_name": run_dir.name,
        "rewards": rewards,
        "agent": config.agent or None,
        "agent_name": OPENREWARD_RUNNER,
        "model": result.normalized_model or result.model or None,
        "n_tool_calls": result.total_tool_calls,
        "n_prompts": len(prompts),
        "agent_result": agent_result,
        "final_metrics": final_metrics_from_agent_result(agent_result),
        "trajectory_summary": trajectory_summary_from_events(
            trajectory,
            partial_trajectory=False,
            trajectory_source="hosted_env" if trajectory else None,
        ),
        "error": result.error,
        "error_category": None,
        "verifier_error": None,
        "verifier_error_category": None,
        # Empty diagnostic slots — hosted runs execute remotely, so every
        # registered diagnostic serializes as ``None`` (see hosted_env.py).
        **RolloutDiagnostics().to_result_fields(),
        "partial_trajectory": False,
        "trajectory_source": "hosted_env" if trajectory else None,
        "started_at": str(started_at),
        "finished_at": str(finished_at),
        "timing": timing,
        "scenes": [],
        "source": source_provenance,
    }
    (run_dir / "result.json").write_text(json.dumps(result_payload, indent=2))
    (run_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    (run_dir / "prompts.json").write_text(
        redact_trajectory_text(json.dumps(prompts, indent=2))
    )

    config_payload: dict[str, Any] = {
        "task_path": None,
        "agent": config.agent or None,
        "model": result.normalized_model or result.model or None,
        "environment": "hosted_env",
        "skills_dir": None,
        "sandbox_user": None,
        "sandbox_locked_paths": None,
        "sandbox_setup_timeout": None,
        "context_root": None,
        "timeout_sec": None,
        "concurrency": 1,
        "agent_idle_timeout_sec": None,
        "started_at": str(started_at),
        "agent_env": {},
        "scenes": [],
        "source": source_provenance,
        "hosted_env": {
            "provider": ref.provider,
            "env_uid": ref.env_uid,
            "runner": OPENREWARD_RUNNER,
            "split": config.split,
            "num_examples": config.num_examples,
            "max_turns": config.max_turns,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "env_args": config.env_args,
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2))

    if rewards:
        _write_hosted_rewards_jsonl(run_dir, rewards, finished_at)
    _write_trainer_verifiers_jsonl(result)


def _build_rewards_dict(result: OpenRewardRunResult) -> dict[str, Any] | None:
    """Rewards dict in the native rollout shape — headline mean + rubric."""
    if result.reward is None:
        return None
    rubric = [
        {
            "name": f"example_{ep.example_index}",
            "score": float(ep.reward) if ep.reward is not None else 0.0,
        }
        for ep in result.episodes
    ]
    payload: dict[str, Any] = {"reward": float(result.reward)}
    if rubric:
        payload["rubric"] = rubric
    return payload


def _collect_prompts(
    result: OpenRewardRunResult,
    config: OpenRewardRunConfig,
) -> list[str]:
    prompts: list[str] = []
    seen: set[str] = set()
    for ep in result.episodes:
        if ep.prompt and ep.prompt not in seen:
            seen.add(ep.prompt)
            prompts.append(ep.prompt)
    if not prompts:
        prompts.append(
            f"<hosted_env:{config.source_env.env_uid} "
            f"num_examples={config.num_examples}>"
        )
    return prompts


def _write_trainer_verifiers_jsonl(result: OpenRewardRunResult) -> None:
    """Emit the trainer seam — ``trainer/verifiers.jsonl``, one record per
    episode — through the shared ORS-backed export pipeline."""
    from benchflow.trajectories.export import (
        ROLLOUT_ARTIFACT_RELPATH,
        acp_events_to_messages,
        export_trajectories_to_jsonl,
        reward_map_to_verify_result,
        trajectory_to_verifiers_record,
    )

    records: list[dict[str, Any]] = []
    for ep in result.episodes:
        messages = acp_events_to_messages(ep.events)
        verify_result = reward_map_to_verify_result(
            {"reward": ep.reward if ep.reward is not None else 0.0},
            error=ep.error,
        )
        records.append(
            trajectory_to_verifiers_record(
                task_id=f"{result.source_env.env_uid}#{ep.task_id}",
                messages=messages,
                verify_result=verify_result,
                model=result.normalized_model or result.model,
                environment=result.source_env.env_uid,
                example_id=ep.example_index,
                is_completed=ep.finished,
                is_truncated=ep.truncated,
            )
        )
    if records:
        export_trajectories_to_jsonl(records, result.run_dir / ROLLOUT_ARTIFACT_RELPATH)
