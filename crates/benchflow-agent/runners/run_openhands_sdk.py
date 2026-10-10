"""Runs OpenHands' software agent SDK once on the task, for BenchFlow.

Written for BenchFlow; the agent it builds follows Harbor's `openhands-sdk`
runner (Apache-2.0, harbor/agents/installed/openhands_sdk_runner.py): the
terminal, file editor and task tracker tools, skills from SKILL.md folders,
and LiteLLM model ids. Events are reported live through the conversation's
callback.

Model: `$BENCHFLOW_MODEL`, a LiteLLM id (`anthropic/claude-sonnet-4-6`,
`openai/gpt-5`). The key comes from the provider's own variable
(`ANTHROPIC_API_KEY`, ...), or `LLM_API_KEY` when set. Options (`bench run
--env NAME=VALUE`): `LLM_BASE_URL` (an OpenAI-compatible endpoint),
`OPENHANDS_MAX_ITERATIONS`, `OPENHANDS_TEMPERATURE`, `OPENHANDS_SKILL_PATHS`
(folders of skills, `:`-separated).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import bf_events

DEFAULT_SKILL_PATHS = ["~/.openhands-sdk/skills", "~/.agents/skills", "~/.claude/skills"]


def contents_text(contents: Any) -> str:
    if isinstance(contents, str):
        return contents
    return "\n".join(getattr(c, "text", "") for c in contents or [] if getattr(c, "text", None))


def discover_skills(Skill: Any) -> list[Any]:
    paths = (bf_events.option("OPENHANDS_SKILL_PATHS") or ":".join(DEFAULT_SKILL_PATHS)).split(":")
    skills: list[Any] = []
    seen: set[str] = set()
    for base in paths:
        root = Path(base).expanduser()
        if not root.is_dir():
            continue
        for folder in sorted(root.iterdir()):
            skill_file = folder / "SKILL.md"
            if folder.name in seen or not skill_file.is_file():
                continue
            seen.add(folder.name)
            skills.append(Skill(name=folder.name, content=skill_file.read_text(), source=str(skill_file), trigger=None))
    return skills


def main(events: bf_events.Events) -> str | None:
    from openhands.sdk import LLM, Agent, AgentContext, Conversation, Tool
    from openhands.sdk.context import Skill
    from openhands.sdk.event import (
        ActionEvent,
        AgentErrorEvent,
        MessageEvent,
        ObservationEvent,
    )
    from openhands.sdk.event.conversation_error import ConversationErrorEvent
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.task_tracker import TaskTrackerTool
    from openhands.tools.terminal import TerminalTool

    instruction = bf_events.instruction()
    model = bf_events.model() or os.environ.get("LLM_MODEL")
    if not model:
        raise bf_events.SetupError(
            "openhands-sdk needs a model, and the run named none",
            "pass -m <provider>/<model>, such as -m anthropic/claude-sonnet-4-6",
        )
    llm_kwargs: dict[str, Any] = {"model": model, "usage_id": "agent"}
    if api_key := os.environ.get("LLM_API_KEY"):
        llm_kwargs["api_key"] = api_key
    base_url = bf_events.option("LLM_BASE_URL")
    if base_url is None and model.startswith("openai/"):
        base_url = bf_events.option("OPENAI_BASE_URL")
    if base_url:
        llm_kwargs["base_url"] = base_url
    if effort := bf_events.effort():
        llm_kwargs["reasoning_effort"] = effort
    if temperature := bf_events.option("OPENHANDS_TEMPERATURE"):
        llm_kwargs["temperature"] = float(temperature)
    llm = LLM(**llm_kwargs)

    skills = discover_skills(Skill)
    agent = Agent(
        llm=llm,
        tools=[Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name), Tool(name=TaskTrackerTool.name)],
        agent_context=AgentContext(skills=skills),
    )
    events.start("openhands-sdk", "openhands-sdk", bf_events.package_version("openhands-sdk"), model)
    if skills:
        print(f"openhands-sdk: loaded {len(skills)} skill(s): {', '.join(s.name for s in skills)}", file=sys.stderr)

    answer: list[str] = []
    errors: list[Any] = []

    def report_usage() -> None:
        metrics = llm.metrics
        usage = metrics.accumulated_token_usage
        if usage is None:
            return
        events.set_usage(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            cost_usd=metrics.accumulated_cost,
        )

    def on_event(event: Any) -> None:
        if isinstance(event, ActionEvent):
            report_usage()
            events.thought(event.reasoning_content)
            events.thought(contents_text(event.thought))
            call = event.tool_call
            events.tool_call(event.tool_call_id, event.tool_name, getattr(call, "arguments", None))
        elif isinstance(event, ObservationEvent):
            observation = event.observation
            events.tool_result(event.tool_call_id, getattr(observation, "text", None) or str(observation), getattr(observation, "is_error", False))
        elif isinstance(event, AgentErrorEvent):
            events.tool_result(event.tool_call_id, event.error, failed=True)
        elif isinstance(event, MessageEvent) and event.source == "agent":
            report_usage()
            message = event.llm_message
            events.thought(getattr(message, "reasoning_content", None))
            text = contents_text(message.content)
            if text:
                answer.append(text)
                events.message(text)
        elif isinstance(event, ConversationErrorEvent):
            errors.append(event)
        elif type(event).__name__ == "UserRejectObservation":
            events.tool_result(event.tool_call_id, getattr(event, "rejection_reason", "rejected"), failed=True)

    conversation_kwargs: dict[str, Any] = {"agent": agent, "workspace": os.getcwd(), "callbacks": [on_event], "visualizer": None}
    if max_iterations := bf_events.option("OPENHANDS_MAX_ITERATIONS"):
        conversation_kwargs["max_iteration_per_run"] = int(max_iterations)
    try:
        conversation = Conversation(**conversation_kwargs)
    except TypeError:
        conversation_kwargs.pop("visualizer")
        conversation = Conversation(**conversation_kwargs)
    try:
        conversation.send_message(instruction)
        conversation.run()
    finally:
        report_usage()
    for error in errors:
        # The run loop's own stops (the iteration cap) end the run without
        # raising; the agent's work so far is still graded.
        events.error(f"{error.code}: {error.detail}", kind="agent", fatal=False)
    return answer[-1] if answer else None


if __name__ == "__main__":
    bf_events.run(main)
