"""Runs Google's Antigravity SDK agent once on the task, for BenchFlow.

Written for BenchFlow; the agent setup follows Harbor's `antigravity-sdk`
runner (Apache-2.0, harbor/agents/installed/antigravity_sdk_runner.py): a
local agent on a Gemini API model, every tool allowed (`policy.allow_all`),
the task's working folder as its workspace, skills from SKILL.md folders,
and the reasoning effort as Gemini's thinking level. The response's chunks
(text, thoughts, tool calls, tool results) are reported as they stream.

The SDK runs its own harness binary (`localharness`, shipped in the wheel)
and calls the Gemini API with a key from a Google account:
`GEMINI_API_KEY` (or `GOOGLE_API_KEY`), created at
https://aistudio.google.com/apikey. `GOOGLE_GEMINI_BASE_URL` points it at
another Gemini-compatible endpoint.

Model: `$BENCHFLOW_MODEL`, a Gemini model (`google/gemini-3.5-flash`; the
provider prefix is dropped). Option: `ANTIGRAVITY_SKILL_PATHS`
(`:`-separated folders of skills).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import bf_events

DEFAULT_SKILL_PATHS = ["~/.gemini/skills", "~/.agents/skills"]
THINKING = {"minimal": "MINIMAL", "low": "LOW", "medium": "MEDIUM", "high": "HIGH", "xhigh": "EXTRA_HIGH", "extra_high": "EXTRA_HIGH", "max": "EXTRA_HIGH"}


async def run_agent(events: bf_events.Events) -> str | None:
    from google.antigravity import Agent, LocalAgentConfig
    from google.antigravity.hooks import policy
    from google.antigravity.models import GeminiAPIEndpoint, GeminiModelOptions, ModelTarget, ThinkingLevel
    from google.antigravity.types import Text, Thought, ToolCall, ToolResult

    instruction = bf_events.instruction()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise bf_events.SetupError(
            "antigravity-sdk needs a Gemini API key, and neither GEMINI_API_KEY nor GOOGLE_API_KEY is set",
            "sign in with a Google account at https://aistudio.google.com/apikey, create a key, and export GEMINI_API_KEY=<key>",
        )
    model = (bf_events.model() or "google/gemini-3.5-flash").split("/")[-1]
    level_name = THINKING.get((bf_events.effort() or "").lower())
    thinking = getattr(ThinkingLevel, level_name) if level_name else None
    endpoint: dict[str, Any] = {"api_key": api_key, "options": GeminiModelOptions(thinking_level=thinking) if thinking else GeminiModelOptions()}
    if base_url := bf_events.option("GOOGLE_GEMINI_BASE_URL"):
        endpoint["base_url"] = base_url
    skills = [str(Path(p).expanduser()) for p in (bf_events.option("ANTIGRAVITY_SKILL_PATHS") or ":".join(DEFAULT_SKILL_PATHS)).split(":")]
    state = Path.home() / ".benchflow" / "antigravity"
    (state / "session").mkdir(parents=True, exist_ok=True)
    (state / "app_data").mkdir(parents=True, exist_ok=True)
    workspace = Path.cwd().resolve()
    config = LocalAgentConfig(
        model=ModelTarget(name=model, endpoint=GeminiAPIEndpoint(**endpoint)),
        api_key=api_key,
        workspaces=[str(workspace)],
        save_dir=str(state / "session"),
        app_data_dir=str(state / "app_data"),
        policies=[policy.allow_all()],
        skills_paths=[p for p in skills if Path(p).is_dir()] or None,
    )
    config = config.eval(thinking_level=None) if thinking else config.eval()
    events.start("antigravity-sdk", "google-antigravity", bf_events.package_version("google-antigravity"), model)

    text: list[str] = []
    calls: dict[str, str] = {}
    async with Agent(config) as agent:
        response = await agent.chat(instruction)
        try:
            async for chunk in response.chunks:
                if isinstance(chunk, Thought):
                    events.thought(chunk.text)
                elif isinstance(chunk, Text):
                    text.append(chunk.text)
                    events.message(chunk.text)
                elif isinstance(chunk, ToolCall):
                    name = getattr(chunk.name, "value", chunk.name)
                    key = chunk.id or chunk.step_id or ""
                    calls[key] = events.tool_call(chunk.id, str(name), chunk.args)
                elif isinstance(chunk, ToolResult):
                    key = chunk.id or chunk.step_id or ""
                    call_id = calls.pop(key, None) or events.tool_call(chunk.id, str(getattr(chunk.name, "value", chunk.name)))
                    failed = chunk.error is not None
                    events.tool_result(call_id, chunk.error if failed else chunk.result, failed=failed)
        finally:
            try:
                usage = agent.conversation.total_usage
            except Exception as exc:  # noqa: BLE001 - usage is best effort
                print(f"antigravity-sdk: no usage: {exc}", file=sys.stderr)
                usage = None
            if usage is not None:
                candidates = getattr(usage, "candidates_token_count", None) or 0
                thoughts = getattr(usage, "thoughts_token_count", None) or 0
                events.set_usage(
                    input_tokens=getattr(usage, "prompt_token_count", None),
                    output_tokens=(candidates + thoughts) or None,
                    cache_read_tokens=getattr(usage, "cached_content_token_count", None),
                    reasoning_tokens=thoughts or None,
                )
    return "".join(text).strip() or None


def main(events: bf_events.Events) -> str | None:
    return asyncio.run(run_agent(events))


if __name__ == "__main__":
    bf_events.run(main)
