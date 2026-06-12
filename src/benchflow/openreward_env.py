"""OpenReward / ORS hosted-environment driver (v0.5 Track-2).

This is the OpenReward sibling of the PrimeIntellect ``vf-eval`` path in
:mod:`benchflow.hosted_env`. Instead of shelling out to ``vf-eval`` against
Prime's hub, it drives an OpenReward environment directly through the
``openreward`` Python client's session loop:

    c = openreward.OpenReward(api_key=...)
    env = c.environments.get("<owner>/<name>")
    task = env.get_task(split, index)
    with env.session(task=task) as session:
        session.get_prompt()
        session.list_tools()
        out = session.call_tool(name, input)   # -> ToolOutput
        # loop until out.finished; final reward is out.reward

The reward of the loop is ``ToolOutput.reward`` (NOT a ``RunResult`` /
``ToolResult``). ``client.rollout.create(...)`` is telemetry/reporting, not the
env-interaction handle; when a live OpenReward client is available we also
create a rollout recording so the run appears in OpenReward's runs UI.

The loop is driven by a :class:`Policy`. We ship a :class:`ScriptedPolicy`
(schema-driven, no LLM) so the path is fully exercisable offline with a fake
session; :class:`ModelPolicy` is left as a clearly-marked seam for a real LLM
agent (PR3+).

Artifacts: the final reward is lifted into a canonical ``VerifyResult`` via
:func:`benchflow.rewards.node.verify_result_from_reward_map` (the single
dict→VerifyResult conversion point — we do not duplicate ``adapters/ors.py``),
and the BenchFlow rollout artifact contract is reconstructed by reusing the
same writers the rest of the codebase uses: the legacy ``result.json`` /
``rewards.jsonl`` / ``config.json`` / ``timing.json`` / ``prompts.json`` /
``trajectory/acp_trajectory.jsonl`` shape from :mod:`benchflow.hosted_env`,
plus the canonical ``verifier/verify_result.json`` and
``trainer/verifiers.jsonl`` from :mod:`benchflow.rollout` /
:mod:`benchflow.trajectories.export`. Lineage is stamped
``trajectory_source="openreward"``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from benchflow.hosted_env import (
    HostedEnvError,
    HostedEnvRunConfig,
    HostedEnvRunResult,
    _hosted_source_provenance,
    _write_hosted_rewards_jsonl,
    build_hosted_config_payload,
    build_hosted_result_payload,
    normalize_verifiers_model,
)

if TYPE_CHECKING:
    from benchflow.rewards.protocol import VerifyResult

logger = logging.getLogger(__name__)

OPENREWARD_API_KEY_ENV_VARS = ("OPENREWARD_API_KEY", "ORS_API_KEY")

# Guard rail: a runaway env would otherwise loop forever if it never returns
# ``finished``. Keep this small — the scripted policy makes one structured call
# per tool, and real environments converge well under this.
_DEFAULT_MAX_STEPS = 50


class OpenRewardSession(Protocol):
    """The slice of ``openreward.environments...Session`` the driver uses.

    Declared structurally so the loop is testable with a fake session that
    records ``call_tool`` and returns scripted ``ToolOutput``-shaped objects —
    no network, no platform, no paid rollout.
    """

    def get_prompt(self) -> Any: ...
    def list_tools(self, format: Any = ...) -> Any: ...
    def call_tool(self, tool_name: str, input: Any = ...) -> Any: ...


class Policy(Protocol):
    """Chooses the next ``(tool_name, input)`` given the prompt and tools.

    Returning ``None`` ends the loop early (the policy has nothing more to do);
    otherwise the driver keeps calling tools until the environment reports
    ``finished`` or the step budget is exhausted.
    """

    def act(
        self,
        prompt_text: str,
        tools: list[Any],
        last_output: Any | None,
        step: int,
    ) -> tuple[str, dict[str, Any]] | None: ...


@dataclass(frozen=True)
class ToolAction:
    """A model-selected OpenReward tool call."""

    name: str
    input: dict[str, Any]
    call_id: str | None = None
    raw_response: Any | None = None


@dataclass(frozen=True)
class OpenRewardSessionContext:
    """Live OpenReward handles opened together for one hosted-env task."""

    session: Any
    client: Any | None = None
    task: Any | None = None
    environment: Any | None = None


class ScriptedPolicy:
    """Deterministic, schema-driven policy — no LLM in the loop.

    Drives the environment by calling the available tools in order, sending an
    empty input by default. When an answer/submit-style tool is present (a tool
    whose name suggests it terminates the episode) it is preferred and called
    with ``answer`` populated from ``answer`` (if supplied) so an offline run
    can exercise the full terminate-on-``finished`` path. This is intentionally
    minimal: it exists to drive the loop in tests and as a smoke policy, not to
    solve tasks. A real agent plugs in via :class:`ModelPolicy`.
    """

    # Substrings that flag a tool as episode-terminating, in priority order.
    _SUBMIT_HINTS = ("submit", "answer", "finish", "done", "respond")

    def __init__(self, answer: str | None = None) -> None:
        self._answer = answer

    def _tool_name(self, tool: Any) -> str:
        # ToolSpec is a dataclass with ``.name``; dicts use ``["name"]``.
        if isinstance(tool, dict):
            return str(tool.get("name", ""))
        return str(getattr(tool, "name", ""))

    def _build_input(self, tool: Any) -> dict[str, Any]:
        name = self._tool_name(tool).lower()
        if self._answer is not None and any(h in name for h in self._SUBMIT_HINTS):
            return {"answer": self._answer}
        return {}

    def act(
        self,
        prompt_text: str,
        tools: list[Any],
        last_output: Any | None,
        step: int,
    ) -> tuple[str, dict[str, Any]] | None:
        if not tools:
            return None
        # Prefer a submit/answer-style tool so the episode can terminate.
        for tool in tools:
            name = self._tool_name(tool)
            if any(h in name.lower() for h in self._SUBMIT_HINTS):
                return name, self._build_input(tool)
        # Otherwise walk tools in order, one per step, then stop.
        if step < len(tools):
            tool = tools[step]
            return self._tool_name(tool), self._build_input(tool)
        return None


class ModelPolicy:
    """LLM-driven OpenReward policy.

    The policy owns model-side tool selection only. The driver still owns
    environment execution: this class returns ``(tool_name, input)`` and the
    run loop calls ``session.call_tool(...)``.
    """

    def __init__(
        self,
        model: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        if not model:
            raise HostedEnvError("--model is required for OpenReward ModelPolicy")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tool_format = _tool_format_for_model(model)
        self._transcript: list[str] = []
        self._last_observation_step: int | None = None
        self.last_action: ToolAction | None = None
        self.last_model_response: Any | None = None

    def act(
        self,
        prompt_text: str,
        tools: list[Any],
        last_output: Any | None,
        step: int,
    ) -> tuple[str, dict[str, Any]] | None:
        if not tools:
            return None
        self._update_transcript(prompt_text, last_output, step)
        action = _call_model_for_tool_action(
            model=self.model,
            prompt="\n\n".join(self._transcript),
            tools=tools,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        self.last_action = action
        self.last_model_response = action.raw_response
        self._transcript.append(
            f"Action {step + 1}: call {action.name} with "
            f"{json.dumps(action.input, sort_keys=True)}"
        )
        return action.name, action.input

    def _update_transcript(
        self,
        prompt_text: str,
        last_output: Any | None,
        step: int,
    ) -> None:
        if not self._transcript:
            self._transcript.append(f"Task:\n{prompt_text}")
            self._transcript.append(
                "You are controlling an OpenReward environment. Select exactly "
                "one available tool call for the next step. Do not answer in prose."
            )
        if last_output is None or self._last_observation_step == step:
            return
        observation = _tool_output_to_text(last_output)
        reward = _read_attr(last_output, "reward")
        finished = _read_attr(last_output, "finished", False)
        self._transcript.append(
            "Observation from previous tool call:\n"
            f"{observation or '<no text output>'}\n"
            f"reward={reward!r}, finished={finished!r}"
        )
        self._last_observation_step = step


def _prompt_to_text(prompt: Any) -> str:
    """Render an openreward prompt (list of TextBlock/ImageBlock) to text.

    Best-effort: blocks expose ``.text`` (TextBlock) or are dict-shaped; images
    and unknown blocks are summarised, never raised on.
    """
    if isinstance(prompt, str):
        return prompt
    if not isinstance(prompt, list):
        return str(prompt)
    parts: list[str] = []
    for block in prompt:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
        elif getattr(block, "type", None) == "image" or (
            isinstance(block, dict) and block.get("type") == "image"
        ):
            parts.append("[image]")
    return "\n".join(parts)


def _tool_output_to_text(output: Any) -> str:
    """Render a ToolOutput's blocks to a single text string (best-effort)."""
    blocks = getattr(output, "blocks", None)
    if blocks is None and isinstance(output, dict):
        blocks = output.get("blocks")
    if not isinstance(blocks, list):
        return ""
    return _prompt_to_text(blocks)


def _read_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _strip_known_provider_prefix(model: str) -> str:
    if "/" not in model:
        return model
    provider, bare = model.split("/", 1)
    if provider in {"openai", "anthropic", "google", "gemini", "openrouter"}:
        return bare
    return model


def _is_openai_model(model: str) -> bool:
    return model.startswith(("openai/", "gpt-", "o1", "o3", "o4"))


def _tool_format_for_model(model: str) -> str:
    if _is_openai_model(model):
        return "openai"
    if model.startswith(("anthropic/", "claude-")):
        raise HostedEnvError(
            "OpenReward ModelPolicy currently supports OpenAI Responses models "
            f"only; got {model!r}."
        )
    if model.startswith(("google/", "gemini/", "gemini")):
        raise HostedEnvError(
            "OpenReward ModelPolicy currently supports OpenAI Responses models "
            f"only; got {model!r}."
        )
    raise HostedEnvError(
        "OpenReward ModelPolicy could not infer a supported provider for "
        f"model {model!r}; use an OpenAI model such as gpt-5.4-mini."
    )


def _openreward_tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name") is not None:
            return str(function["name"])
        return str(tool.get("name") or "")
    return str(getattr(tool, "name", ""))


def _openreward_tool_description(tool: Any) -> str:
    if isinstance(tool, dict):
        function = tool.get("function")
        if isinstance(function, dict) and function.get("description") is not None:
            return str(function["description"])
        return str(tool.get("description") or "")
    return str(getattr(tool, "description", "") or "")


def _openreward_tool_parameters(tool: Any) -> dict[str, Any]:
    schema: Any
    if isinstance(tool, dict):
        function = tool.get("function")
        if isinstance(function, dict):
            schema = function.get("parameters") or function.get("input_schema")
        else:
            schema = tool.get("parameters") or tool.get("input_schema")
    else:
        schema = getattr(tool, "parameters", None) or getattr(tool, "input_schema", None)
    if isinstance(schema, dict):
        return schema
    return {"type": "object", "properties": {}, "additionalProperties": True}


def _tools_to_openai_responses(tools: list[Any]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        name = _openreward_tool_name(tool)
        if not name:
            continue
        converted.append(
            {
                "type": "function",
                "name": name,
                "description": _openreward_tool_description(tool),
                "parameters": _openreward_tool_parameters(tool),
            }
        )
    if not converted:
        raise HostedEnvError("OpenReward environment did not expose any named tools")
    return converted


def _call_model_for_tool_action(
    *,
    model: str,
    prompt: str,
    tools: list[Any],
    max_tokens: int,
    temperature: float,
) -> ToolAction:
    if _is_openai_model(model):
        return _call_openai_responses_tool_action(
            model=model,
            prompt=prompt,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    raise HostedEnvError(f"Unsupported OpenReward ModelPolicy model: {model}")


def _call_openai_responses_tool_action(
    *,
    model: str,
    prompt: str,
    tools: list[Any],
    max_tokens: int,
    temperature: float,
) -> ToolAction:
    try:
        import openai
    except ImportError as exc:
        raise HostedEnvError(
            "OpenAI SDK is required for OpenReward ModelPolicy with "
            f"model {model!r}. Install the OpenAI dependency."
        ) from exc

    client = openai.OpenAI()
    response = client.responses.create(
        model=_strip_known_provider_prefix(model),
        instructions=(
            "You are an agent inside an OpenReward environment. Use exactly one "
            "tool call. Do not return a plain text answer."
        ),
        input=prompt,
        tools=_tools_to_openai_responses(tools),
        tool_choice="required",
        max_output_tokens=max_tokens,
        temperature=temperature,
        parallel_tool_calls=False,
    )
    action = _parse_openai_responses_tool_action(response)
    return ToolAction(
        name=action.name,
        input=action.input,
        call_id=action.call_id,
        raw_response=response,
    )


def _parse_openai_responses_tool_action(response: Any) -> ToolAction:
    output = _read_attr(response, "output", [])
    if not isinstance(output, list):
        raise HostedEnvError("OpenAI response did not contain an output list")
    for item in output:
        item_type = _read_attr(item, "type")
        if item_type != "function_call":
            continue
        name = _read_attr(item, "name")
        raw_arguments = _read_attr(item, "arguments", "{}")
        call_id = _read_attr(item, "call_id")
        if not isinstance(name, str) or not name:
            raise HostedEnvError("OpenAI function_call output is missing a tool name")
        try:
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as exc:
            raise HostedEnvError(
                f"OpenAI function_call arguments for {name!r} were not valid JSON"
            ) from exc
        if not isinstance(arguments, dict):
            raise HostedEnvError(
                f"OpenAI function_call arguments for {name!r} must be a JSON object"
            )
        return ToolAction(
            name=name,
            input=arguments,
            call_id=str(call_id) if call_id else None,
        )
    raise HostedEnvError("OpenAI model did not return a function_call tool action")


def _policy_tool_format(policy: Policy) -> str | None:
    value = getattr(policy, "tool_format", None)
    return str(value) if value else None


class _NoopOpenRewardRolloutRecorder:
    recording_info: dict[str, Any] | None = None

    def log_prompt(self, prompt_text: str) -> None:
        return

    def log_model_response(self, response: Any, *, step: int) -> bool:
        return False

    def log_tool_call(
        self,
        *,
        name: str,
        tool_input: dict[str, Any],
        call_id: str,
        step: int,
    ) -> None:
        return

    def log_tool_output(
        self,
        output: Any,
        *,
        call_id: str,
        step: int,
    ) -> None:
        return


class _OpenRewardRolloutRecorder:
    """Best-effort mirror of the BenchFlow loop into OpenReward runs."""

    def __init__(self, rollout: Any) -> None:
        self._rollout = rollout
        rollout_id = str(getattr(rollout, "event_id", "") or "")
        web_base_url = str(getattr(rollout, "web_base_url", "") or "")
        if not web_base_url:
            web_base_url = "https://openreward.ai"
        self.recording_info = {
            "rollout_id": rollout_id or None,
            "url": f"{web_base_url.rstrip('/')}/rollout/{rollout_id}"
            if rollout_id
            else None,
        }

    @classmethod
    def create(
        cls,
        opened: Any,
        *,
        config: HostedEnvRunConfig,
        run_dir: Path,
        split: str,
        index: int,
        normalized_model: str,
    ) -> _OpenRewardRolloutRecorder | _NoopOpenRewardRolloutRecorder:
        client = getattr(opened, "client", None)
        rollout_api = getattr(client, "rollout", None)
        create = getattr(rollout_api, "create", None)
        if not callable(create):
            return _NoopOpenRewardRolloutRecorder()

        run_info = _openreward_run_info(
            normalized_model=normalized_model,
            model=config.model,
        )
        metadata = {
            "benchflow_run_dir": str(run_dir),
            "benchflow_model": config.model,
            "benchflow_normalized_model": normalized_model,
            "benchflow_source_env": config.source_env.env_uid,
            "benchflow_task_index": index,
        }
        rollout = create(
            run_name=str(
                config.env_args.get("openreward_run_name")
                or config.env_args.get("run_name")
                or "benchflow-openreward"
            ),
            rollout_name=str(config.env_args.get("rollout_name") or run_dir.name),
            environment=config.source_env.env_id,
            variant=_openreward_variant(config.source_env),
            split=split,
            metadata=metadata,
            task_spec=_openreward_task_spec(getattr(opened, "task", None)),
            run_info=run_info,
            print_messages=True,
        )
        recorder = cls(rollout)
        if getattr(rollout, "_rollouts_disabled", False):
            logger.warning(
                "openreward rollout recording is disabled; run may not appear in UI"
            )
        return recorder

    def log_prompt(self, prompt_text: str) -> None:
        if not prompt_text:
            return
        self._safe_log({"type": "user_message", "content": prompt_text})

    def log_model_response(self, response: Any, *, step: int) -> bool:
        if response is None:
            return False
        try:
            self._rollout.log_openai_response(
                response,
                metadata={"benchflow_step": step, "benchflow_source": "model"},
            )
            return True
        except Exception:
            logger.warning("openreward rollout model log failed", exc_info=True)
            return False

    def log_tool_call(
        self,
        *,
        name: str,
        tool_input: dict[str, Any],
        call_id: str,
        step: int,
    ) -> None:
        self._safe_log(
            {
                "type": "tool_call",
                "name": name,
                "call_id": call_id,
                "content": json.dumps(tool_input, sort_keys=True),
            },
            metadata={"benchflow_step": step, "benchflow_source": "benchflow"},
        )

    def log_tool_output(
        self,
        output: Any,
        *,
        call_id: str,
        step: int,
    ) -> None:
        reward = _read_attr(output, "reward")
        finished = bool(_read_attr(output, "finished", False))
        payload = {
            "text": _tool_output_to_text(output),
            "reward": reward,
            "finished": finished,
        }
        self._safe_log(
            {
                "type": "tool_result",
                "call_id": call_id,
                "content": json.dumps(payload, sort_keys=True),
            },
            reward=float(reward)
            if isinstance(reward, (int, float)) and not isinstance(reward, bool)
            else None,
            is_finished=finished,
            metadata={"benchflow_step": step, "benchflow_source": "openreward_env"},
        )

    def _safe_log(self, message: Any, **kwargs: Any) -> None:
        try:
            self._rollout.log(message, **kwargs)
        except Exception:
            logger.warning("openreward rollout log failed", exc_info=True)


def _openreward_run_info(
    *,
    normalized_model: str,
    model: str,
) -> Any | None:
    try:
        from openreward.models import RunInfo
    except ImportError:
        return None
    return RunInfo(
        model_name=normalized_model or model,
        run_type="eval",
        framework="benchflow",
    )


def _openreward_variant(ref: Any) -> str | None:
    version = getattr(ref, "version", None)
    if version and version != "latest":
        return str(version)
    return None


def _openreward_task_spec(task: Any) -> dict[str, Any] | None:
    if task is None:
        return None
    if isinstance(task, dict):
        candidate = task.get("task_spec") or task.get("spec") or task
        return candidate if isinstance(candidate, dict) else None
    for attr in ("task_spec", "spec", "data"):
        candidate = getattr(task, attr, None)
        if isinstance(candidate, dict):
            return candidate
    model_dump = getattr(task, "model_dump", None)
    if callable(model_dump):
        candidate = model_dump()
        if isinstance(candidate, dict):
            return candidate
    return None


def _opened_session(opened: Any) -> Any:
    return getattr(opened, "session", opened)


def _policy_last_action(policy: Any) -> ToolAction | None:
    action = getattr(policy, "last_action", None)
    return action if isinstance(action, ToolAction) else None


def run_hosted_env_openreward(
    config: HostedEnvRunConfig,
    *,
    policy: Policy | None = None,
    split: str = "train",
    index: int = 0,
    session_factory: Any | None = None,
    max_steps: int = _DEFAULT_MAX_STEPS,
) -> HostedEnvRunResult:
    """Run a single OpenReward environment task and write BenchFlow artifacts.

    Drives the openreward session loop with *policy* (defaults to
    :class:`ScriptedPolicy`), terminating when a ``ToolOutput`` reports
    ``finished`` (or the step budget is hit). The final ``ToolOutput.reward``
    is lifted to a canonical ``VerifyResult`` and the full rollout artifact
    contract is written to ``run_dir``.

    ``session_factory`` is the offline/test seam: when supplied it is called as
    ``session_factory(config, split, index) -> context-manager yielding a
    session`` instead of opening a real ``OpenReward`` client. Production leaves
    it ``None`` and opens the client via :func:`_open_openreward_session`.

    Namespace/owner is REQUIRED for openreward (a bare name 400s on the
    platform) — raises :class:`HostedEnvError` before any session is opened.
    """
    ref = config.source_env
    if ref.provider != "openreward":
        raise HostedEnvError(
            f"run_hosted_env_openreward called with provider {ref.provider!r}; "
            "expected 'openreward'"
        )
    if ref.owner is None:
        raise HostedEnvError(
            "OpenReward requires an explicit owner/namespace "
            "(e.g. openreward:GeneralReasoning/KellyBench). A bare environment "
            "name is rejected by the platform."
        )

    split = str(config.env_args.get("split", split))
    try:
        index = int(config.env_args.get("index", index))
    except (TypeError, ValueError) as exc:
        raise HostedEnvError(
            f"OpenReward source-env arg 'index' must be an integer, got "
            f"{config.env_args.get('index')!r}"
        ) from exc

    scripted_answer = config.env_args.get("answer")
    if policy is None:
        if scripted_answer is not None or config.env_args.get("policy") == "scripted":
            policy = ScriptedPolicy(
                answer=str(scripted_answer) if scripted_answer is not None else None
            )
        else:
            policy = ModelPolicy(
                config.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )
    run_dir = _make_run_dir(config)
    normalized_model = normalize_verifiers_model(config.model) if config.model else ""

    started_at = datetime.now(UTC)
    trajectory: list[dict[str, Any]] = []
    prompts: list[str] = []
    final_reward: float | None = None
    n_tool_calls = 0
    error: str | None = None
    recording_info: dict[str, Any] | None = None

    factory = session_factory or _open_openreward_session
    try:
        with factory(config, split, index) as opened:
            session = _opened_session(opened)
            recorder = _OpenRewardRolloutRecorder.create(
                opened,
                config=config,
                run_dir=run_dir,
                split=split,
                index=index,
                normalized_model=normalized_model,
            )
            recording_info = recorder.recording_info
            prompt = session.get_prompt()
            prompt_text = _prompt_to_text(prompt)
            if prompt_text:
                prompts.append(prompt_text)
                recorder.log_prompt(prompt_text)
                trajectory.append(
                    {
                        "type": "user_message",
                        "ts": started_at.isoformat(),
                        "content": prompt_text,
                    }
                )
            tools = list(session.list_tools(format=_policy_tool_format(policy)))

            last_output: Any | None = None
            for step in range(max_steps):
                action = policy.act(prompt_text, tools, last_output, step)
                if action is None:
                    break
                tool_name, tool_input = action
                last_action = _policy_last_action(policy)
                call_id = (
                    last_action.call_id
                    if last_action is not None and last_action.call_id
                    else f"benchflow-step-{step}"
                )
                if not recorder.log_model_response(
                    getattr(policy, "last_model_response", None),
                    step=step,
                ):
                    recorder.log_tool_call(
                        name=tool_name,
                        tool_input=tool_input,
                        call_id=call_id,
                        step=step,
                    )
                output = session.call_tool(tool_name, tool_input)
                n_tool_calls += 1
                last_output = output
                recorder.log_tool_output(output, call_id=call_id, step=step)

                ts = datetime.now(UTC).isoformat()
                trajectory.append(
                    {
                        "type": "tool_call",
                        "ts": ts,
                        "title": tool_name,
                        "kind": tool_name,
                        "content": [{"type": "text", "text": json.dumps(tool_input)}],
                    }
                )
                out_text = _tool_output_to_text(output)
                if out_text:
                    trajectory.append(
                        {"type": "agent_message", "ts": ts, "content": out_text}
                    )

                reward = _read_attr(output, "reward")
                if isinstance(reward, (int, float)) and not isinstance(reward, bool):
                    final_reward = float(reward)
                if _read_attr(output, "finished", False):
                    break
            else:
                error = (
                    f"environment did not report finished within {max_steps} steps"
                )
    except HostedEnvError:
        raise
    except Exception as e:  # surface any driver/client failure as run error
        error = f"{type(e).__name__}: {e}"
        logger.warning("openreward run failed: %s", error)

    finished_at = datetime.now(UTC)

    result = HostedEnvRunResult(
        source_env=ref,
        run_dir=run_dir,
        command=["openreward", ref.env_id, f"split={split}", f"index={index}"],
        returncode=0 if error is None else 1,
        stdout="",
        stderr=error or "",
        model=config.model,
        normalized_model=normalized_model,
        reward=final_reward,
        total_tool_calls=n_tool_calls,
        verifiers_error=error,
    )

    _write_openreward_artifacts(
        result,
        config,
        trajectory=trajectory,
        prompts=prompts,
        started_at=started_at,
        finished_at=finished_at,
        split=split,
        index=index,
        error=error,
        recording_info=recording_info,
    )
    return result


def _make_run_dir(config: HostedEnvRunConfig) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d__%H-%M-%S-%f")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", config.source_env.env_id)
    jobs_dir = config.jobs_dir.expanduser().resolve()
    run_id = f"{timestamp}__pid-{os.getpid()}__{uuid4().hex[:8]}"
    run_dir = jobs_dir / "hosted-env" / f"{safe_name}__{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _open_openreward_session(
    config: HostedEnvRunConfig,
    split: str,
    index: int,
) -> Any:
    """Open a real OpenReward session (production path; not used in tests).

    Returns a context manager that yields a live session. The ``OpenReward``
    client is created here and closed in the wrapper's ``__exit__`` (try/finally
    via ``close()``), so the event-loop thread the sync client spins up is torn
    down even on error.
    """
    return _OpenRewardSessionCtx(config, split, index)


class _OpenRewardSessionCtx:
    """Context manager: open client → get env → get task → open session.

    Owns the ``OpenReward`` client lifecycle (``close()`` in ``__exit__``).
    """

    def __init__(self, config: HostedEnvRunConfig, split: str, index: int) -> None:
        self._config = config
        self._split = split
        self._index = index
        self._client: Any = None
        self._session_cm: Any = None
        self._environment: Any = None
        self._task: Any = None

    def __enter__(self) -> OpenRewardSessionContext:
        import openreward

        api_key = next(
            (os.environ[k] for k in OPENREWARD_API_KEY_ENV_VARS if os.environ.get(k)),
            None,
        )
        if not api_key:
            raise HostedEnvError(
                "OPENREWARD_API_KEY is required to run an openreward environment"
            )
        self._client = openreward.OpenReward(api_key=api_key)
        try:
            self._environment = self._client.environments.get(
                self._config.source_env.env_id
            )
            self._task = self._environment.get_task(self._split, self._index)
            self._session_cm = self._environment.session(task=self._task)
            return OpenRewardSessionContext(
                session=self._session_cm.__enter__(),
                client=self._client,
                task=self._task,
                environment=self._environment,
            )
        except Exception:
            self._close_client()
            raise

    def __exit__(self, *exc: Any) -> None:
        try:
            if self._session_cm is not None:
                self._session_cm.__exit__(*exc)
        finally:
            self._close_client()

    def _close_client(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # best-effort teardown
                    logger.debug("openreward client close() failed", exc_info=True)


def _build_verify_result(
    rewards: dict[str, Any] | None, *, error: str | None
) -> VerifyResult:
    """Lift the run outcome to a canonical ``VerifyResult``.

    Defers to the single dict→``VerifyResult`` conversion point
    (:func:`benchflow.rewards.node.verify_result_from_reward_map`) for the
    scored case and the genuine-failure case (a ``None`` map *with* an ``error``
    is a crash — that function stamps the error and yields ``reward=0.0``).

    The one case it handles directly is a **clean finish that produced no
    reward**: ``rewards`` is ``None`` but ``error`` is ``None`` too. Routing that
    through ``verify_result_from_reward_map`` would default ``error`` to the
    sentinel ``"no rewards"``, conflating "unscored" with "verifier crashed"
    (``result.json.error`` is ``None`` yet ``verify_result.json.error`` would be
    truthy). Instead we emit an explicit unscored result — ``reward=0.0`` with
    **no** error — so the two states stay distinguishable downstream.
    """
    from benchflow.rewards.node import verify_result_from_reward_map
    from benchflow.rewards.protocol import VerifyResult

    if rewards is None and error is None:
        return VerifyResult(
            reward=0.0,
            items={},
            events=[],
            error=None,
            space="output",
            granularity="terminal",
        )
    return verify_result_from_reward_map(rewards, error=error)


def _write_openreward_artifacts(
    result: HostedEnvRunResult,
    config: HostedEnvRunConfig,
    *,
    trajectory: list[dict[str, Any]],
    prompts: list[str],
    started_at: datetime,
    finished_at: datetime,
    split: str,
    index: int,
    error: str | None,
    recording_info: dict[str, Any] | None = None,
) -> None:
    """Write the BenchFlow rollout artifact contract for an openreward run.

    Mirrors :func:`benchflow.hosted_env._write_run_artifacts` (legacy
    ``result.json`` / ``rewards.jsonl`` / ``config.json`` / ``timing.json`` /
    ``prompts.json`` / ``trajectory/acp_trajectory.jsonl``) and adds the
    canonical Reward-plane artifacts (``verifier/verify_result.json``,
    ``trainer/verifiers.jsonl``) by reusing the shared writers. Lineage is
    stamped ``trajectory_source="openreward"``.
    """
    run_dir = result.run_dir
    for sub in ("trajectory", "agent", "verifier", "artifacts"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    rewards = {"reward": float(result.reward)} if result.reward is not None else None
    verify_result = _build_verify_result(rewards, error=error)

    # trajectory/acp_trajectory.jsonl
    (run_dir / "trajectory" / "acp_trajectory.jsonl").write_text(
        "\n".join(json.dumps(e, default=str) for e in trajectory)
        + ("\n" if trajectory else "")
    )

    if not prompts:
        prompts = [f"<openreward:{result.source_env.env_uid} split={split} index={index}>"]

    timing = {"total": round((finished_at - started_at).total_seconds(), 1)}
    source_provenance = _hosted_source_provenance(
        result.source_env,
        runner="openreward",
        env_args=config.env_args,
    )
    source_provenance["split"] = split
    source_provenance["index"] = index

    result_payload = build_hosted_result_payload(
        result,
        config,
        rewards=rewards,
        prompts=prompts,
        timing=timing,
        source_provenance=source_provenance,
        started_at=started_at,
        finished_at=finished_at,
        agent_name="openreward",
        trajectory_source="openreward",
        has_trajectory=bool(trajectory),
        # The driver already collapses any failure into ``result.error`` (see
        # the loop's ``verifiers_error`` plumbing); there is no separate
        # verifier-error channel, so this stays ``None``.
        error=result.error,
        verifier_error=None,
    )
    (run_dir / "result.json").write_text(json.dumps(result_payload, indent=2))
    (run_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    (run_dir / "prompts.json").write_text(json.dumps(prompts, indent=2))

    config_payload = build_hosted_config_payload(
        result,
        config,
        source_provenance=source_provenance,
        started_at=started_at,
        environment="openreward",
        runner="openreward",
        extra_hosted_env={"split": split, "index": index},
    )
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2))

    if rewards:
        _write_hosted_rewards_jsonl(run_dir, rewards, finished_at)
    if recording_info:
        (run_dir / "artifacts" / "openreward_rollout.json").write_text(
            json.dumps(recording_info, indent=2)
        )

    # Canonical Reward-plane artifacts — reuse the shared writers so the schema
    # never drifts from native rollouts.
    from benchflow.trajectories.export import write_verify_result_json

    write_verify_result_json(run_dir, verify_result)
    _write_verifiers_jsonl(
        run_dir,
        task_id=result.source_env.env_uid,
        prompts=prompts,
        trajectory=trajectory,
        verify_result=verify_result,
        model=result.normalized_model or result.model or None,
        is_completed=error is None,
    )


def _write_verifiers_jsonl(
    run_dir: Path,
    *,
    task_id: str,
    prompts: list[str],
    trajectory: list[dict[str, Any]],
    verify_result: VerifyResult,
    model: str | None,
    is_completed: bool,
) -> None:
    """Emit ``trainer/verifiers.jsonl`` via the shared trainer-export writer."""
    from benchflow.trajectories.export import write_rollout_verifiers_jsonl

    write_rollout_verifiers_jsonl(
        run_dir,
        task_id=task_id,
        prompts=prompts,
        trajectory=trajectory,
        verify_result=verify_result,
        model=model,
        environment="openreward",
        is_completed=is_completed,
    )
