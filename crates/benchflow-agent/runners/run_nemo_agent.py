"""Runs an NVIDIA NeMo Agent Toolkit (NAT) workflow once on the task, for BenchFlow.

Written for BenchFlow; the default workflow follows Harbor's `nemo-agent`
(Apache-2.0, harbor/agents/installed/nemo_agent.py and
nemo_agent_run_wrapper.py): NAT's `chat_completion` workflow, one model call
at temperature 0 with no tools, run through NAT's Python API rather than
`nat run` (whose console formatting loses text when piped). The config is
built in memory, so the key never lands in a file.

Model: `$BENCHFLOW_MODEL` as `provider/model`: `nvidia/` (NAT's `nim` type,
`NVIDIA_API_KEY`, `NVIDIA_BASE_URL`), `openai/` (`OPENAI_API_KEY`,
`OPENAI_BASE_URL`) or `litellm/` (`LITELLM_API_KEY`, `LITELLM_BASE_URL`).

Options (`bench run --env NAME=VALUE`): `NEMO_CONFIG`, a NAT YAML config in
the sandbox to run instead (its own `llms` and `workflow`; tool calls it
makes are reported too), and `NEMO_SYSTEM_PROMPT`.

The default workflow only answers: it cannot run commands or edit files, so
it suits tasks graded on its answer. `NEMO_ANSWER_FILE` names a file the
runner writes the answer to, for verifiers that read one.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import bf_events

# provider prefix -> (NAT llm type, key variable, base URL variable, extra settings)
TYPES: dict[str, tuple[str, str, str, dict[str, Any]]] = {
    "nvidia": ("nim", "NVIDIA_API_KEY", "NVIDIA_BASE_URL", {"max_tokens": 2048}),
    "nim": ("nim", "NVIDIA_API_KEY", "NVIDIA_BASE_URL", {"max_tokens": 2048}),
    "openai": ("openai", "OPENAI_API_KEY", "OPENAI_BASE_URL", {}),
    "litellm": ("litellm", "LITELLM_API_KEY", "LITELLM_BASE_URL", {}),
}
SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Read the user's request carefully and follow the output "
    "format instructions exactly. Be concise and precise."
)


def default_config(model: str) -> dict[str, Any]:
    provider, sep, model_name = model.partition("/")
    if not sep or provider not in TYPES:
        raise bf_events.SetupError(
            f"nemo-agent cannot reach `{model}`: name the model as nvidia/<model>, openai/<model> or litellm/<model>",
            "pass -m nvidia/<model> (such as nvidia/nvidia/nemotron-3-super-120b-a12b) or openai/<model>, or give your own NAT config with --env NEMO_CONFIG=<file in the sandbox>",
        )
    llm_type, key_var, url_var, extra = TYPES[provider]
    llm: dict[str, Any] = {"_type": llm_type, "model_name": model_name, "temperature": 0.0, **extra}
    if key := os.environ.get(key_var):
        llm["api_key"] = key
    if url := os.environ.get(url_var):
        llm["base_url"] = url
    return {
        "llms": {"benchflow_llm": llm},
        "workflow": {
            "_type": "chat_completion",
            "llm_name": "benchflow_llm",
            "system_prompt": bf_events.option("NEMO_SYSTEM_PROMPT", SYSTEM_PROMPT),
        },
    }


class Failures(logging.Handler):
    """Keeps the exceptions NAT logs: its chat_completion workflow logs a
    provider failure and answers with an apology instead of raising."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.exceptions: list[BaseException] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.exc_info and record.exc_info[1] is not None:
            self.exceptions.append(record.exc_info[1])


def count_usage(events: bf_events.Events) -> None:
    """Report every LangChain model call's tokens (NAT's LangChain-backed
    workflows report none of their own unless profiling is on)."""
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.tracers.context import register_configure_hook

    class Usage(BaseCallbackHandler):
        def on_llm_end(self, response: Any, **kwargs: Any) -> None:
            for generations in response.generations:
                for generation in generations:
                    usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                    if usage:
                        details = usage.get("input_token_details") or {}
                        events.add_usage(
                            input_tokens=usage.get("input_tokens"),
                            output_tokens=usage.get("output_tokens"),
                            cache_read_tokens=details.get("cache_read"),
                        )
                        return

    handler: ContextVar[Any] = ContextVar("benchflow_usage", default=None)
    register_configure_hook(handler, inheritable=True)
    handler.set(Usage())


async def run_workflow(events: bf_events.Events) -> str | None:
    from nat.builder.context import Context
    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.data_models.config import Config
    from nat.runtime.loader import PluginTypes, discover_and_register_plugins
    from nat.runtime.session import SessionManager

    instruction = bf_events.instruction()
    model = bf_events.model() or "nvidia/nvidia/nemotron-3-super-120b-a12b"
    config_path = bf_events.option("NEMO_CONFIG")
    if config_path:
        from nat.utils.io.yaml_tools import yaml_load

        if not Path(config_path).is_file():
            raise bf_events.SetupError(
                f"NEMO_CONFIG names {config_path}, which is not in the sandbox",
                "point NEMO_CONFIG at a NAT YAML config the task ships",
            )
        raw = yaml_load(config_path)
    else:
        raw = default_config(model)

    failures = Failures()
    logging.getLogger("nat").addHandler(failures)
    count_usage(events)
    discover_and_register_plugins(PluginTypes.COMPONENT)
    config = Config(**raw)
    events.start("nemo-agent", "nvidia-nat", bf_events.package_version("nvidia-nat"), None if config_path else model)

    tools: dict[str, str] = {}

    def on_step(step: Any) -> None:
        payload = step.payload
        kind = str(getattr(payload.event_type, "value", payload.event_type))
        data = payload.data
        if kind == "LLM_END" and payload.usage_info is not None:
            usage = payload.usage_info.token_usage
            events.add_usage(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                cache_read_tokens=usage.cached_tokens,
                reasoning_tokens=usage.reasoning_tokens,
            )
        elif kind == "TOOL_START":
            tools[payload.UUID] = events.tool_call(payload.UUID, payload.name or "tool", getattr(data, "input", None))
        elif kind == "TOOL_END":
            call_id = tools.pop(payload.UUID, None) or events.tool_call(payload.UUID, payload.name or "tool")
            events.tool_result(call_id, getattr(data, "output", None))

    async with WorkflowBuilder.from_config(config) as builder:
        sessions = await SessionManager.create(config=config, shared_builder=builder)
        try:
            async with sessions.session(user_id="benchflow") as session:
                async with session.run(instruction) as runner:
                    Context.get().intermediate_step_manager.subscribe(on_next=on_step)
                    result = await runner.result()
        finally:
            await sessions.shutdown()
    answer = str(result).strip()
    if failures.exceptions and (not answer or answer.startswith("I apologize, but I encountered an error")):
        raise failures.exceptions[0]
    if answer_file := bf_events.option("NEMO_ANSWER_FILE"):
        Path(answer_file).write_text(answer + "\n")
    return answer


def main(events: bf_events.Events) -> str | None:
    answer = asyncio.run(run_workflow(events))
    if not answer:
        raise RuntimeError("the NAT workflow returned an empty answer")
    return answer


if __name__ == "__main__":
    bf_events.run(main)
