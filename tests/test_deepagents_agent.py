"""Unit tests for the deepagents agent integration.

Covers the registry wiring (the agent is registered and its config follows the
ACP-shim conventions the other in-process Python shims use) and the shim's pure
helpers — provider/model resolution and message normalization — none of which
require a live model. The cross-agent schema invariants in
``test_registry_invariants.py`` already parametrize over every AGENTS entry, so
this file asserts only the deepagents-specific contract.

The shim keeps all third-party imports (deepagents, langchain_openai) lazy
inside functions, so importing the module here — and unit-testing its pure
helpers — works without those packages installed in the dev venv.
"""

import ast
from pathlib import Path

from benchflow.agents import deepagents_acp_shim as shim
from benchflow.agents.registry import (
    _BENCHFLOW_BIN_PREFIX,
    AGENT_ALIASES,
    AGENTS,
    resolve_agent,
)

_SHIM_PATH = Path(shim.__file__)


# ── Registry wiring ──────────────────────────────────────────────────────────


def test_deepagents_is_registered():
    """The agent exists under its canonical key and resolves via alias."""
    assert "deepagents" in AGENTS
    cfg = AGENTS["deepagents"]
    assert cfg.name == "deepagents"
    assert cfg.protocol == "acp"
    assert AGENT_ALIASES["deepagents"] == "deepagents"
    assert resolve_agent("deepagents").name == "deepagents"


def test_deepagents_install_deploys_shim_and_isolated_venv():
    """Install builds an isolated venv with deepagents + langchain-openai and
    deploys the shim into BenchFlow's shared bin prefix."""
    cfg = AGENTS["deepagents"]
    install = cfg.install_cmd
    # Isolated venv on a uv-pinned interpreter (deepagents needs Python >=3.11,
    # but task base images ship Python as old as 3.6/3.8, so a system-python venv
    # would make pip report "No matching distribution found for deepagents").
    assert "uv venv --python 3.12 /opt/benchflow/deepagents-venv" in install
    # Packages are installed into that pinned venv via uv pip.
    assert (
        "uv pip install -q --python /opt/benchflow/deepagents-venv/bin/python"
        in install
    )
    # Both required pip packages: deepagents pulls langchain*, but langchain-openai
    # is the explicit dep for the OpenAI-compatible deepseek-v4-pro chat model.
    assert "deepagents" in install
    assert "langchain-openai" in install
    # Shim deployed into the shared bin prefix via _install_python_script's
    # base64 transport (so the shim source can't collide with shell tokens).
    assert f"{_BENCHFLOW_BIN_PREFIX}/deepagents-acp-shim" in install
    assert "base64 -d" in install
    # Install verifies deepagents imports through the pinned venv, so a failed
    # `uv pip install` surfaces as rc!=0 instead of being masked by the
    # non-fatal chmod / shim-deploy's trailing `chmod +x`.
    assert "/opt/benchflow/deepagents-venv/bin/python -c 'import deepagents'" in install


def test_deepagents_launch_runs_shim_with_venv_python():
    """Launch invokes the shim through the isolated venv interpreter."""
    launch = AGENTS["deepagents"].launch_cmd
    assert launch.startswith("/opt/benchflow/deepagents-venv/bin/python ")
    assert launch.endswith(f"{_BENCHFLOW_BIN_PREFIX}/deepagents-acp-shim")


def test_deepagents_infers_provider_from_model_at_runtime():
    """No static requires_env / env_mapping — the shim resolves provider creds
    from BENCHFLOW_PROVIDER_* (with DEEPSEEK_* fallback) at prompt time."""
    cfg = AGENTS["deepagents"]
    assert cfg.requires_env == []
    assert cfg.env_mapping == {}
    assert cfg.api_protocol == ""


def test_shim_source_parses():
    """The shim source is valid Python (it is uploaded + executed verbatim)."""
    ast.parse(_SHIM_PATH.read_text())


def test_shim_has_no_benchflow_import():
    """The shim runs in the sandbox with NO benchflow installed — guard against
    an accidental ``import benchflow`` that would crash on launch."""
    tree = ast.parse(_SHIM_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("benchflow"), (
                    f"shim must not import benchflow ({alias.name})"
                )
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("benchflow"), (
                f"shim must not import benchflow ({node.module})"
            )


# ── Provider / model resolution (pure helpers) ───────────────────────────────


def test_resolve_base_url_prefers_sdk_then_provider_then_default():
    """SDK BENCHFLOW_PROVIDER_BASE_URL wins, then DEEPSEEK_BASE_URL, then default."""
    assert (
        shim.resolve_base_url(
            {
                "BENCHFLOW_PROVIDER_BASE_URL": "https://sdk.example/v1",
                "DEEPSEEK_BASE_URL": "https://native.example/v1",
            }
        )
        == "https://sdk.example/v1"
    )
    assert (
        shim.resolve_base_url({"DEEPSEEK_BASE_URL": "https://native.example/v1"})
        == "https://native.example/v1"
    )
    assert shim.resolve_base_url({}) == shim._DEFAULT_DEEPSEEK_BASE_URL
    # Blank/whitespace values are treated as unset.
    assert shim.resolve_base_url({"BENCHFLOW_PROVIDER_BASE_URL": "  "}) == (
        shim._DEFAULT_DEEPSEEK_BASE_URL
    )


def test_resolve_api_key_prefers_sdk_then_provider():
    assert (
        shim.resolve_api_key(
            {"BENCHFLOW_PROVIDER_API_KEY": "sdk-key", "DEEPSEEK_API_KEY": "native-key"}
        )
        == "sdk-key"
    )
    assert shim.resolve_api_key({"DEEPSEEK_API_KEY": "native-key"}) == "native-key"
    assert shim.resolve_api_key({}) == ""


def test_resolve_model_id_priority_and_prefix_strip():
    """Explicit (set_model) > BENCHFLOW_PROVIDER_MODEL > default; prefix stripped."""
    # Explicit ACP-supplied model wins, prefix stripped.
    assert (
        shim.resolve_model_id(
            "deepseek/deepseek-v4-pro", {"BENCHFLOW_PROVIDER_MODEL": "other"}
        )
        == "deepseek-v4-pro"
    )
    # Falls back to the SDK-stripped model env var.
    assert (
        shim.resolve_model_id("", {"BENCHFLOW_PROVIDER_MODEL": "deepseek-v4-pro"})
        == "deepseek-v4-pro"
    )
    # Falls back to the built-in default when nothing is provided.
    assert shim.resolve_model_id("", {}) == shim._DEFAULT_MODEL


def test_float_and_int_env_helpers_are_lenient():
    assert shim._float_env({"X": "0.7"}, "X") == 0.7
    assert shim._float_env({"X": ""}, "X") is None
    assert shim._float_env({"X": "nope"}, "X") is None
    assert shim._int_env({"X": "2048"}, "X") == 2048
    assert shim._int_env({"X": "2048.0"}, "X") == 2048
    assert shim._int_env({}, "X") is None


# ── Tool-kind + message normalization ────────────────────────────────────────


def test_tool_kind_maps_known_tools():
    assert shim._tool_kind("execute") == "execute"
    assert shim._tool_kind("write_file") == "edit"
    assert shim._tool_kind("read_file") == "read"
    assert shim._tool_kind("grep_search") == "search"
    assert shim._tool_kind("write_todos") == "think"
    assert shim._tool_kind("mystery") == "other"


def test_message_text_flattens_str_and_block_content():
    assert shim._message_text({"content": "hello"}) == "hello"
    assert (
        shim._message_text(
            {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
        )
        == "ab"
    )
    assert shim._message_text({"content": []}) == ""


def test_message_role_and_tool_calls_from_dicts():
    ai = {
        "type": "ai",
        "content": "thinking",
        "tool_calls": [{"id": "tc1", "name": "execute", "args": {"command": "ls"}}],
    }
    assert shim._message_role(ai) == "ai"
    calls = shim._message_tool_calls(ai)
    assert calls == [{"id": "tc1", "name": "execute", "args": {"command": "ls"}}]
    # A message with no tool calls yields an empty list.
    assert shim._message_tool_calls({"type": "tool", "content": "ok"}) == []


# ── Streaming trajectory + dedup (run_deep_agent) ────────────────────────────


class _FakeStreamingAgent:
    """Stand-in LangGraph agent yielding pre-baked ``updates``-mode chunks."""

    def __init__(self, chunks):
        self._chunks = chunks

    def stream(self, _input, config=None, stream_mode=None):
        # The shim must request incremental updates, not values/debug.
        assert stream_mode == "updates"
        yield from self._chunks


def test_run_deep_agent_streams_and_dedups_tool_calls(monkeypatch):
    """run_deep_agent emits ACP updates per streamed step and announces each
    tool_call id exactly once even if a later chunk re-surfaces it (LangGraph
    can re-emit accumulated state), so the idle watchdog is fed without
    double-counting."""
    ai_with_text = {
        "type": "ai",
        "content": "writing the file",
        "tool_calls": [
            {"id": "call_1", "name": "write_file", "args": {"path": "a.py"}}
        ],
    }
    tool_result = {"type": "tool", "tool_call_id": "call_1", "content": "ok"}
    # A later chunk re-surfaces call_1 (empty text so only the tool_call dedup is
    # under test, not text emission), plus a genuinely new tool_call call_2.
    ai_dup_plus_new = {
        "type": "ai",
        "content": "",
        "tool_calls": [
            {"id": "call_1", "name": "write_file", "args": {"path": "a.py"}},
            {"id": "call_2", "name": "execute", "args": {"command": "pytest"}},
        ],
    }
    chunks = [
        {"agent": {"messages": [ai_with_text]}},
        {"tools": {"messages": [tool_result]}},
        {"agent": {"messages": [ai_dup_plus_new]}},
    ]

    monkeypatch.setattr(
        shim, "build_deep_agent", lambda *a, **k: _FakeStreamingAgent(chunks)
    )
    sent: list[dict] = []
    monkeypatch.setattr(shim, "send", lambda payload: sent.append(payload))

    metrics = shim.run_deep_agent("deepseek-v4-pro", "do it", "/tmp", "sess-1", env={})

    # call_1 announced once despite appearing in two chunks; call_2 added once.
    assert metrics["tool_call_count"] == 2
    # Only the first AI message carried text; the re-surfaced one was empty.
    assert metrics["text_count"] == 1
    assert metrics["message_count"] == 3

    kinds = [p["params"]["update"]["sessionUpdate"] for p in sent]
    tool_call_ids = [
        p["params"]["update"]["toolCallId"]
        for p in sent
        if p["params"]["update"]["sessionUpdate"] == "tool_call"
    ]
    # Each tool_call id emitted exactly once, in first-seen order.
    assert tool_call_ids == ["call_1", "call_2"]
    # The tool result emitted as a completion update keyed to its originating call.
    assert kinds.count("tool_call_update") == 1
    assert kinds.count("agent_message_chunk") == 1
