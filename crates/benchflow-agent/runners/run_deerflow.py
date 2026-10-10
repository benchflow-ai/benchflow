"""Runs ByteDance's DeerFlow once on the task, for BenchFlow.

Written for BenchFlow; the DeerFlow setup follows Harbor's `deerflow` agent
(Apache-2.0, harbor/agents/installed/deerflow.py and deerflow_runner.py): a
generated `config.yaml` with one model from the run's model, DeerFlow's
standard tools (web search and fetch with keyless fallbacks, file read and
write, bash), its local sandbox with host bash and an identity mount of the
task's working folder, and `DeerFlowClient.stream` driven with feature
toggles the stock CLI cannot set.

Model: `$BENCHFLOW_MODEL` as `provider/model`: `openai/`, `anthropic/`,
`deepseek/`, `google/` or `gemini/` use their LangChain integration,
`openrouter/` goes through OpenRouter, anything else through an
OpenAI-compatible client (`OPENAI_BASE_URL`, `OPENAI_API_KEY`).

Options (`bench run --env NAME=VALUE`): `DEERFLOW_CONFIG` (a DeerFlow YAML
config in the sandbox to start from; BenchFlow still sets the model, the
sandbox and the mount), `DEERFLOW_SUBAGENT`, `DEERFLOW_THINKING` (default
on), `DEERFLOW_PLAN_MODE`, `DEERFLOW_SUMMARIZE`, `DEERFLOW_NO_WEB`,
`DEERFLOW_RECURSION_LIMIT` (default 1000).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import bf_events

MODEL_NAME = "benchflow-model"
DIRECT = {
    "openai": ("langchain_openai:ChatOpenAI", "OPENAI_API_KEY"),
    "anthropic": ("langchain_anthropic:ChatAnthropic", "ANTHROPIC_API_KEY"),
    "deepseek": ("langchain_deepseek:ChatDeepSeek", "DEEPSEEK_API_KEY"),
    "gemini": ("langchain_google_genai:ChatGoogleGenerativeAI", "GEMINI_API_KEY"),
    "google": ("langchain_google_genai:ChatGoogleGenerativeAI", "GEMINI_API_KEY"),
}
WEB_SEARCH = (
    ("TAVILY_API_KEY", "deerflow.community.tavily.tools:web_search_tool"),
    ("SERPER_API_KEY", "deerflow.community.serper.tools:web_search_tool"),
    ("EXA_API_KEY", "deerflow.community.exa.tools:web_search_tool"),
    ("FIRECRAWL_API_KEY", "deerflow.community.firecrawl.tools:web_search_tool"),
)
WEB_FETCH = (
    ("FIRECRAWL_API_KEY", "deerflow.community.firecrawl.tools:web_fetch_tool"),
    ("EXA_API_KEY", "deerflow.community.exa.tools:web_fetch_tool"),
)
REASONS = {"quota": "usage_limit", "auth": "login", "transient": "provider", "busy": "provider"}


def model_entry(model: str) -> dict[str, Any]:
    provider, sep, rest = model.partition("/")
    if sep and provider == "openrouter":
        use, model_id, key = "langchain_openai:ChatOpenAI", rest, "OPENROUTER_API_KEY"
        base_url: str | None = bf_events.option("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    elif sep and provider in DIRECT:
        (use, key), model_id, base_url = DIRECT[provider], rest, None
        if provider == "openai":
            base_url = bf_events.option("OPENAI_BASE_URL")
    else:
        use, model_id, key, base_url = "langchain_openai:ChatOpenAI", model, "OPENAI_API_KEY", bf_events.option("OPENAI_BASE_URL")
    entry: dict[str, Any] = {
        "name": MODEL_NAME,
        "use": use,
        "model": model_id,
        "api_key": f"${key}",
        "request_timeout": 600.0,
        "max_retries": 2,
    }
    if base_url:
        entry["base_url"] = base_url
    return entry


def pick(backends: tuple[tuple[str, str], ...], fallback: str) -> str:
    return next((use for key, use in backends if os.environ.get(key)), fallback)


def build_config(model: str, workdir: str) -> dict[str, Any]:
    import yaml

    user_config = bf_events.option("DEERFLOW_CONFIG")
    if user_config:
        path = Path(user_config)
        if not path.is_file():
            raise bf_events.SetupError(
                f"DEERFLOW_CONFIG names {path}, which is not in the sandbox",
                "point DEERFLOW_CONFIG at a DeerFlow config.yaml the task ships",
            )
        config = yaml.safe_load(path.read_text()) or {}
        if not isinstance(config, dict):
            raise bf_events.SetupError(f"{path} is not a YAML mapping", "fix the DeerFlow config")
    else:
        config = {
            "config_version": 15,
            "log_level": "info",
            "tool_groups": [{"name": g} for g in ("web", "file:read", "file:write", "bash")],
            "tools": [
                {"name": "web_search", "group": "web", "use": pick(WEB_SEARCH, "deerflow.community.ddg_search.tools:web_search_tool")},
                {"name": "web_fetch", "group": "web", "use": pick(WEB_FETCH, "deerflow.community.jina_ai.tools:web_fetch_tool")},
                {"name": "ls", "group": "file:read", "use": "deerflow.sandbox.tools:ls_tool"},
                {"name": "read_file", "group": "file:read", "use": "deerflow.sandbox.tools:read_file_tool"},
                {"name": "glob", "group": "file:read", "use": "deerflow.sandbox.tools:glob_tool"},
                {"name": "grep", "group": "file:read", "use": "deerflow.sandbox.tools:grep_tool"},
                {"name": "write_file", "group": "file:write", "use": "deerflow.sandbox.tools:write_file_tool"},
                {"name": "str_replace", "group": "file:write", "use": "deerflow.sandbox.tools:str_replace_tool"},
                {"name": "bash", "group": "bash", "use": "deerflow.sandbox.tools:bash_tool"},
            ],
        }
    if bf_events.flag("DEERFLOW_NO_WEB", False):
        config["tool_groups"] = [g for g in config.get("tool_groups", []) if g.get("name") != "web"]
        config["tools"] = [t for t in config.get("tools", []) if t.get("group") != "web"]
    models = [m for m in config.get("models") or [] if isinstance(m, dict) and m.get("name") != MODEL_NAME]
    config["models"] = [*models, model_entry(model)]
    sandbox = dict(config.get("sandbox") or {})
    sandbox["use"] = "deerflow.sandbox.local:LocalSandboxProvider"
    sandbox["allow_host_bash"] = True
    mounts = [m for m in sandbox.get("mounts") or [] if isinstance(m, dict) and m.get("container_path") != workdir]
    mounts.append({"host_path": workdir, "container_path": workdir, "read_only": False})
    sandbox["mounts"] = mounts
    config["sandbox"] = sandbox
    if bf_events.flag("DEERFLOW_SUMMARIZE", False):
        config["summarization"] = {"enabled": True, "trigger": [{"type": "fraction", "value": 0.8}]}
    return config


def steer(instruction: str, workdir: str) -> str:
    # DeerFlow's assistant otherwise writes deliverables to its virtual
    # /mnt/user-data, which the task's verifier never looks at.
    return (
        "You are working on a real Linux machine through the bash and file tools; use real absolute paths. "
        f"Your working directory is {workdir}: create and change the task's files there, not under /mnt/user-data, "
        f"because the task is checked on the files under {workdir}.\n\nTask:\n{instruction}"
    )


def main(events: bf_events.Events) -> str | None:
    import yaml

    instruction = bf_events.instruction()
    model = bf_events.model()
    if not model:
        raise bf_events.SetupError("deerflow needs a model, and the run named none", "pass -m <provider>/<model>, such as -m openai/gpt-5")
    workdir = str(Path.cwd().resolve())
    project = Path.home() / ".benchflow" / "deerflow"
    project.mkdir(parents=True, exist_ok=True)
    (project / "config.yaml").write_text(yaml.safe_dump(build_config(model, workdir), sort_keys=False))
    os.environ["DEER_FLOW_PROJECT_ROOT"] = str(project)
    os.environ.setdefault("DEER_FLOW_HOME", str(project / ".deer-flow"))
    os.environ.setdefault("DEER_FLOW_CONFIG_PATH", str(project / "config.yaml"))

    from deerflow.client import DeerFlowClient
    from langgraph.errors import GraphRecursionError

    client = DeerFlowClient(
        model_name=MODEL_NAME,
        subagent_enabled=bf_events.flag("DEERFLOW_SUBAGENT", False),
        thinking_enabled=bf_events.flag("DEERFLOW_THINKING", True),
        plan_mode=bf_events.flag("DEERFLOW_PLAN_MODE", False),
    )
    events.start("deerflow", "deerflow-harness", bf_events.package_version("deerflow-harness"), model)

    calls: set[str] = set()
    results: set[str] = set()
    answers: dict[str, str] = {}
    order: list[str] = []
    llm_error: dict[str, Any] | None = None

    def tool_calls(calls_list: Any) -> None:
        for call in calls_list or []:
            call_id = call.get("id")
            if call_id and call.get("name") and call_id not in calls:
                calls.add(call_id)
                events.tool_call(call_id, call["name"], call.get("args"))

    def tool_result(data: dict[str, Any]) -> None:
        call_id = data.get("tool_call_id")
        if call_id and call_id not in results:
            results.add(call_id)
            if call_id not in calls:
                calls.add(call_id)
                events.tool_call(call_id, data.get("name") or "tool", None)
            events.tool_result(call_id, data.get("content"))

    recursion_limit = int(bf_events.option("DEERFLOW_RECURSION_LIMIT", "1000"))
    try:
        for event in client.stream(steer(instruction, workdir), recursion_limit=recursion_limit):
            kind, data = str(event.type), event.data
            if not isinstance(data, dict):
                continue
            if kind == "messages-tuple":
                if data.get("type") == "ai":
                    extra = data.get("additional_kwargs") or {}
                    if extra.get("deerflow_error_fallback") and llm_error is None:
                        llm_error = extra
                        continue
                    events.thought(extra.get("reasoning_content") if isinstance(extra.get("reasoning_content"), str) else None)
                    text = data.get("content")
                    if isinstance(text, str) and text:
                        message_id = data.get("id") or "-"
                        if message_id not in answers:
                            order.append(message_id)
                        answers[message_id] = answers.get(message_id, "") + text
                        events.message(text)
                    if usage := data.get("usage_metadata"):
                        events.add_usage(input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"))
                elif data.get("type") == "tool":
                    tool_result(data)
            elif kind == "values":
                for message in data.get("messages") or []:
                    if message.get("type") == "ai":
                        tool_calls(message.get("tool_calls"))
                    elif message.get("type") == "tool":
                        tool_result(message)
            elif kind == "end" and isinstance(data.get("usage"), dict):
                usage = data["usage"]
                events.set_usage(input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"))
    except GraphRecursionError as exc:
        events.error(f"DeerFlow stopped at its recursion limit ({recursion_limit} steps): {exc}", kind="agent", fatal=False)
    if llm_error is not None:
        reason = str(llm_error.get("error_reason") or "generic")
        detail = llm_error.get("error_detail") or "the model provider failed after retries"
        print(f"deerflow: the provider failed ({reason}): {detail}", file=sys.stderr)
        events.error(f"{llm_error.get('error_type') or 'LLMError'}: {detail}", kind=REASONS.get(reason, "agent"))
        sys.exit(1)
    return answers[order[-1]] if order else None


if __name__ == "__main__":
    bf_events.run(main)
