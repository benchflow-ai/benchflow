"""Runs a user's LangGraph graph once on the task, for BenchFlow.

Written for BenchFlow. It reads a project the way Harbor's `langgraph` agent
does (Apache-2.0, harbor/agents/installed/langgraph_runner.py): a
`langgraph.json` whose `graphs` map ids to `./file.py:attribute`, where the
attribute is a compiled graph or a factory taking the run config. The graph
gets `{"messages": [{"role": "user", "content": <instruction>}]}` and the
model in `config["configurable"]["model"]` as `provider:model`; its updates
are streamed as events.

Options (`bench run --env NAME=VALUE`):

- `LANGGRAPH_PROJECT`: the folder in the sandbox holding `langgraph.json`
  (default: the task's working folder); `LANGGRAPH_CONFIG` names another file.
- `LANGGRAPH_GRAPH`: the graph id, required when the project has several.
- `LANGGRAPH_CONFIGURABLE`: a JSON object merged into `configurable`.
- `LANGGRAPH_INSTALL_DEPS=0`: skip installing the project's `dependencies`
  (other than `.`) at the start of the run.

Python graphs only; graphs in JavaScript or TypeScript are not supported.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import bf_events

FIX = (
    "add a LangGraph project to the task (a langgraph.json whose `graphs` map an id to `./file.py:graph`) "
    "and pass --env LANGGRAPH_PROJECT=<its folder in the sandbox> when it is not the working folder"
)
NODE_SUFFIXES = (".js", ".mjs", ".cjs", ".ts", ".mts", ".cts", ".jsx", ".tsx")


def load_project() -> tuple[Path, str, str, dict[str, Any]]:
    project = Path(bf_events.option("LANGGRAPH_PROJECT") or bf_events.workdir()).resolve()
    config_name = bf_events.option("LANGGRAPH_CONFIG", "langgraph.json")
    config_path = project / config_name
    if not config_path.is_file():
        raise bf_events.SetupError(f"langgraph runs your graph, and {config_path} does not exist", FIX)
    try:
        config = json.loads(config_path.read_text())
    except ValueError as exc:
        raise bf_events.SetupError(f"{config_path} is not JSON: {exc}", "fix the file") from exc
    graphs = config.get("graphs") if isinstance(config, dict) else None
    if not isinstance(graphs, dict) or not graphs:
        raise bf_events.SetupError(f"{config_path} defines no graphs", FIX)
    graph_id = bf_events.option("LANGGRAPH_GRAPH")
    if graph_id is None:
        if len(graphs) != 1:
            raise bf_events.SetupError(
                f"{config_path} defines several graphs ({', '.join(sorted(graphs))})",
                "pick one with --env LANGGRAPH_GRAPH=<id>",
            )
        graph_id = next(iter(graphs))
    if graph_id not in graphs:
        raise bf_events.SetupError(
            f"{config_path} has no graph `{graph_id}`; it has {', '.join(sorted(graphs))}",
            "pick one with --env LANGGRAPH_GRAPH=<id>",
        )
    spec = graphs[graph_id]
    path = spec.get("path") if isinstance(spec, dict) else spec
    if not isinstance(path, str) or path.count(":") != 1:
        raise bf_events.SetupError(f"graph `{graph_id}` is not `./file.py:attribute`", FIX)
    if path.split(":")[0].endswith(NODE_SUFFIXES):
        raise bf_events.SetupError(
            f"graph `{graph_id}` is JavaScript ({path}); BenchFlow's langgraph agent runs Python graphs only",
            "point LANGGRAPH_GRAPH at a Python graph, or run the JavaScript graph with another agent",
        )
    return project, graph_id, path, config


def install_dependencies(project: Path, config: dict[str, Any]) -> None:
    """Install the project's `dependencies` (all but `.`) into a folder of
    their own, put first on the import path."""
    deps = [d for d in config.get("dependencies") or [] if isinstance(d, str) and d.strip() not in (".", "./")]
    if not deps or not bf_events.flag("LANGGRAPH_INSTALL_DEPS", True):
        return
    uv = shutil.which("uv") or ("/opt/benchflow/uv/uv" if os.access("/opt/benchflow/uv/uv", os.X_OK) else None)
    if uv is None:
        print("langgraph: no uv to install the project's dependencies; continuing without them", file=sys.stderr)
        return
    target = Path.home() / ".benchflow" / "langgraph-deps"
    args: list[str] = []
    for dep in deps:
        local = project / dep
        if local.is_file():
            args += ["-r", str(local)]
        elif local.exists():
            args.append(str(local))
        else:
            args.append(dep)
    print(f"langgraph: installing the project's dependencies: {' '.join(args)}", file=sys.stderr)
    env = {**os.environ, "UV_CACHE_DIR": str(Path.home() / ".cache" / "uv"), "UV_NO_PROGRESS": "1"}
    subprocess.run(
        [uv, "pip", "install", "--python", sys.executable, "--target", str(target), *args],
        check=True,
        stdout=sys.stderr,
        env=env,
    )
    sys.path.insert(0, str(target))


def load_graph(project: Path, path: str) -> Any:
    module_path, attribute = path.split(":")
    file = (project / module_path).resolve()
    if not file.is_file():
        raise bf_events.SetupError(f"the graph's module {file} does not exist", FIX)
    sys.path.insert(0, str(project))
    spec = importlib.util.spec_from_file_location(f"bf_langgraph_{file.stem}_{uuid.uuid4().hex[:8]}", file)
    if spec is None or spec.loader is None:
        raise bf_events.SetupError(f"cannot import {file}", FIX)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    value: Any = module
    for part in attribute.split("."):
        if not hasattr(value, part):
            raise bf_events.SetupError(f"{file} has no `{attribute}`", FIX)
        value = getattr(value, part)
    return value


@contextlib.asynccontextmanager
async def resolved(loaded: Any, config: dict[str, Any]):
    """A compiled graph as is, or the graph a factory builds from the config."""
    if hasattr(loaded, "astream") or hasattr(loaded, "ainvoke"):
        yield loaded
        return
    if not callable(loaded):
        raise bf_events.SetupError("the graph target is neither a compiled graph nor a factory", FIX)
    try:
        takes_config = len(inspect.signature(loaded).parameters) >= 1
    except (TypeError, ValueError):
        takes_config = True
    produced = loaded(config) if takes_config else loaded()
    if hasattr(produced, "__aenter__"):
        async with produced as graph:
            yield graph
    elif hasattr(produced, "__enter__"):
        with produced as graph:
            yield graph
    elif inspect.isawaitable(produced):
        yield await produced
    else:
        yield produced


class Reporter:
    """Turns messages from graph updates into events, once each."""

    def __init__(self, events: bf_events.Events) -> None:
        self.events = events
        self.seen: set[str] = set()
        self.answer: str | None = None
        self.usage: dict[str, int] = {}

    def message(self, message: Any) -> None:
        kind = getattr(message, "type", None)
        if kind is None and isinstance(message, dict):
            kind = message.get("type") or message.get("role")
        key = getattr(message, "id", None) or (message.get("id") if isinstance(message, dict) else None)
        if key:
            if key in self.seen:
                return
            self.seen.add(key)
        if kind in ("ai", "assistant", "AIMessageChunk"):
            self.ai(message)
        elif kind == "tool":
            status = getattr(message, "status", "success")
            self.events.tool_result(
                getattr(message, "tool_call_id", None) or "",
                content_text(getattr(message, "content", "")),
                failed=status == "error",
            )

    def ai(self, message: Any) -> None:
        content = getattr(message, "content", "")
        blocks = content if isinstance(content, list) else [content]
        texts: list[str] = []
        for block in blocks:
            if isinstance(block, str):
                texts.append(block)
            elif isinstance(block, dict):
                if block.get("type") in ("thinking", "reasoning"):
                    self.events.thought(block.get("thinking") or block.get("reasoning") or block.get("text"))
                elif block.get("type") == "text":
                    texts.append(block.get("text", ""))
        reasoning = (getattr(message, "additional_kwargs", None) or {}).get("reasoning_content")
        self.events.thought(reasoning if isinstance(reasoning, str) else None)
        text = "".join(texts).strip()
        if text:
            self.answer = text
            self.events.message(text)
        for call in getattr(message, "tool_calls", None) or []:
            self.events.tool_call(call.get("id"), call.get("name", "tool"), call.get("args"))
        usage = getattr(message, "usage_metadata", None)
        if isinstance(usage, dict):
            details = usage.get("input_token_details") or {}
            output_details = usage.get("output_token_details") or {}
            self.events.add_usage(
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                cache_read_tokens=details.get("cache_read"),
                cache_write_tokens=details.get("cache_creation"),
                reasoning_tokens=output_details.get("reasoning"),
            )

    def update(self, update: Any) -> None:
        if not isinstance(update, dict):
            return
        for value in update.values():
            if not isinstance(value, dict):
                continue
            messages = value.get("messages")
            if messages is None:
                continue
            for message in messages if isinstance(messages, list) else [messages]:
                self.message(message)


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b if isinstance(b, str) else str(b.get("text", "")) for b in content if isinstance(b, (str, dict))
        )
    return bf_events.text(content)


async def run_graph(events: bf_events.Events) -> str | None:
    project, graph_id, path, config = load_project()
    install_dependencies(project, config)
    loaded = load_graph(project, path)
    instruction = bf_events.instruction()

    configurable = bf_events.json_option("LANGGRAPH_CONFIGURABLE")
    configurable.setdefault("thread_id", str(uuid.uuid4()))
    model = bf_events.model()
    if model:
        provider, sep, rest = model.partition("/")
        configurable.setdefault("model", f"{provider}:{rest}" if sep and ":" not in model else model)
        os.environ.setdefault("HARBOR_MODEL", configurable["model"])
    run_config = {
        "configurable": configurable,
        "recursion_limit": int(bf_events.option("LANGGRAPH_RECURSION_LIMIT", "1000")),
        "metadata": {"ls_runner": "benchflow", "langgraph_graph": graph_id},
    }
    events.start("langgraph", "langgraph", bf_events.package_version("langgraph"), configurable.get("model"))
    reporter = Reporter(events)
    final: Any = None
    async with resolved(loaded, run_config) as graph:
        stream = graph.astream(
            {"messages": [{"role": "user", "content": instruction}]},
            config=run_config,
            stream_mode=["updates", "values"],
        )
        async for item in stream:
            mode, chunk = item if isinstance(item, tuple) and len(item) == 2 else ("updates", item)
            if mode == "updates":
                reporter.update(chunk)
            elif mode == "values":
                final = chunk
    if reporter.answer is None and isinstance(final, dict):
        for message in reversed(final.get("messages") or []):
            if getattr(message, "type", None) == "ai":
                reporter.answer = content_text(getattr(message, "content", "")) or None
                break
    if reporter.answer is None and final is not None:
        reporter.answer = bf_events.text(final, 4000)
    return reporter.answer


def main(events: bf_events.Events) -> str | None:
    return asyncio.run(run_graph(events))


if __name__ == "__main__":
    bf_events.run(main)
