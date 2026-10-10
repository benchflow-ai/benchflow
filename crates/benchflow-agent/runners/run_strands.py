"""Runs a user's Strands agent once on the task, for BenchFlow.

Written for BenchFlow. It keeps the factory contract of Harbor's `strands`
agent (Apache-2.0, harbor/agents/installed/strands_runner.py), so a factory
written for Harbor works unchanged: `factory(sandbox, model=None, **kwargs)`
returns a `strands.Agent`, where `sandbox` runs commands and file
operations in the task's working folder (Strands' own local environment)
and `model` is built from the run's model.

Options (`bench run --env NAME=VALUE`):

- `STRANDS_FACTORY` (required): `module:function`, or `file.py:function`,
  found in `STRANDS_FACTORY_PATH` (default: the task's working folder).
- `STRANDS_FACTORY_KWARGS`, `STRANDS_MODEL_KWARGS`: JSON objects passed to the
  factory and to the Strands model class.

Model: `$BENCHFLOW_MODEL` as `provider/model-id`: `anthropic/`, `openai/`
(`OPENAI_BASE_URL` is honoured), `google/` or `gemini/`, `bedrock/`, and any
other provider through LiteLLM. Each reads its provider's own key.
"""

from __future__ import annotations

import asyncio
import copy
import sys
from pathlib import Path
from typing import Any

import bf_events

FIX = (
    "add a Python file to the task with `def create_agent(sandbox, model=None, **kwargs)` returning a "
    "strands.Agent, then pass --env STRANDS_FACTORY=<file.py or module>:create_agent "
    "(and --env STRANDS_FACTORY_PATH=<folder in the sandbox> when it is not in the working folder)"
)

# provider prefix -> (module, class, how the model id is passed, defaults)
PROVIDERS: dict[str, tuple[str, str, dict[str, Any]]] = {
    "anthropic": ("strands.models.anthropic", "AnthropicModel", {"max_tokens": 8192}),
    "openai": ("strands.models.openai", "OpenAIModel", {}),
    "google": ("strands.models.gemini", "GeminiModel", {}),
    "gemini": ("strands.models.gemini", "GeminiModel", {}),
    "bedrock": ("strands.models.bedrock", "BedrockModel", {}),
    "amazon-bedrock": ("strands.models.bedrock", "BedrockModel", {}),
}


def build_model(name: str | None) -> Any:
    if not name:
        return None
    import importlib

    provider, _, model_id = name.partition("/")
    extra = bf_events.json_option("STRANDS_MODEL_KWARGS")
    if provider in PROVIDERS and model_id:
        module, cls, defaults = PROVIDERS[provider]
        kwargs: dict[str, Any] = copy.deepcopy(defaults)
        if provider == "openai" and (base_url := bf_events.option("OPENAI_BASE_URL")):
            kwargs["client_args"] = {"base_url": base_url}
        kwargs.update(extra)
        kwargs["model_id"] = model_id
    else:
        module, cls = "strands.models.litellm", "LiteLLMModel"
        kwargs = {**extra, "model_id": name}
    model_class = getattr(importlib.import_module(module), cls)
    return model_class(**kwargs)


def main(events: bf_events.Events) -> str | None:
    from strands import Agent
    from strands.hooks import AfterModelCallEvent, MessageAddedEvent
    from strands.sandbox.not_a_sandbox_local_environment import NotASandboxLocalEnvironment

    instruction = bf_events.instruction()
    reference = bf_events.option("STRANDS_FACTORY")
    if reference is None:
        raise bf_events.SetupError("strands runs your agent factory, and STRANDS_FACTORY is not set", FIX)
    folder = Path(bf_events.option("STRANDS_FACTORY_PATH") or bf_events.workdir())
    factory = bf_events.load_callable(reference, [folder], "the Strands factory", FIX)

    kwargs = bf_events.json_option("STRANDS_FACTORY_KWARGS")
    if "model" not in kwargs:
        model = build_model(bf_events.model())
        if model is not None:
            kwargs["model"] = model
    agent = factory(NotASandboxLocalEnvironment(), **kwargs)
    if not isinstance(agent, Agent):
        raise bf_events.SetupError(
            f"the Strands factory `{reference}` returned {type(agent).__name__}, not a strands.Agent", FIX
        )
    observed = getattr(agent.model, "config", None) or {}
    model_name = bf_events.model() or (observed.get("model_id") if isinstance(observed, dict) else None)
    events.start("strands", "strands-agents", bf_events.package_version("strands-agents"), model_name)

    answer: list[str] = []
    first_user = [True]

    def report_usage(_event: Any = None) -> None:
        usage = agent.event_loop_metrics.accumulated_usage
        events.set_usage(
            input_tokens=usage.get("inputTokens"),
            output_tokens=usage.get("outputTokens"),
            cache_read_tokens=usage.get("cacheReadInputTokens"),
            cache_write_tokens=usage.get("cacheWriteInputTokens"),
        )

    def on_message(event: Any) -> None:
        message = event.message
        role = message.get("role")
        for block in message.get("content") or []:
            if role == "assistant":
                if "reasoningContent" in block:
                    reasoning = block["reasoningContent"].get("reasoningText") or {}
                    events.thought(reasoning.get("text"))
                if block.get("text"):
                    answer.append(block["text"])
                    events.message(block["text"])
                if "toolUse" in block:
                    use = block["toolUse"]
                    events.tool_call(use.get("toolUseId"), use.get("name", "tool"), use.get("input"))
            elif role == "user":
                if "toolResult" in block:
                    result = block["toolResult"]
                    output = "\n".join(
                        c.get("text") or bf_events.text(c.get("json")) for c in result.get("content") or []
                    )
                    events.tool_result(result.get("toolUseId"), output, failed=result.get("status") == "error")
                elif block.get("text") and not first_user[0]:
                    events.thought(f"[user] {block['text']}")
        if role == "user":
            first_user[0] = False

    agent.add_hook(on_message, MessageAddedEvent)
    agent.add_hook(report_usage, AfterModelCallEvent)
    try:
        result = asyncio.run(agent.invoke_async(instruction))
    finally:
        report_usage()
        try:
            agent.cleanup()
        except Exception as exc:  # noqa: BLE001 - cleanup must not hide the run's result
            print(f"strands: cleanup failed: {exc}", file=sys.stderr)
    final = str(result).strip()
    return final or (answer[-1] if answer else None)


if __name__ == "__main__":
    bf_events.run(main)
