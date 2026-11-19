"""A LiteLLM patch whose target moved stops the proxy's start, naming it.

The proxy applies BenchFlow's LiteLLM patches from sitecustomize: the effort
capabilities, token capture, Gemini passthrough logging, MiMo's retry and the
messages bridge. Each wrapped a LiteLLM function and, when that function was
gone, quietly did nothing, so a LiteLLM upgrade could drop what a patch adds
(the effort on the wire, the token ids, Gemini's usage, the retry) without a
word. Each now checks its target exists with the parameters it wraps and
fails loud naming itself; sitecustomize turns that into the proxy's exit
(the lead's 2026-11-13 list).
"""

from __future__ import annotations

import runpy
import subprocess
import sys
import types
from pathlib import Path

import pytest

PROVIDERS = Path(__file__).resolve().parents[1] / "src" / "benchflow" / "providers"


def _fake_module(monkeypatch, name: str, **attrs) -> None:
    """Install ``name`` (and its parents) in sys.modules with ``attrs``."""
    parts = name.split(".")
    for depth in range(1, len(parts)):
        parent = ".".join(parts[:depth])
        monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


def test_token_capture_names_itself_when_its_target_changed(monkeypatch):
    class CustomStreamWrapper:
        def chunk_creator(self, data):  # LiteLLM renamed the chunk argument
            return data

    _fake_module(
        monkeypatch,
        "litellm.litellm_core_utils.streaming_handler",
        CustomStreamWrapper=CustomStreamWrapper,
    )
    with pytest.raises(RuntimeError, match=r"'token_capture'.*chunk"):
        runpy.run_path(str(PROVIDERS / "litellm_token_capture_patch.py"))


def test_token_capture_names_itself_when_its_module_moved(monkeypatch):
    _fake_module(monkeypatch, "litellm.litellm_core_utils")
    # Gone even when another test has imported the real one.
    monkeypatch.setitem(
        sys.modules, "litellm.litellm_core_utils.streaming_handler", None
    )
    with pytest.raises(RuntimeError, match=r"'token_capture'.*moved"):
        runpy.run_path(str(PROVIDERS / "litellm_token_capture_patch.py"))


def test_provider_retry_names_itself_when_its_target_changed(monkeypatch):
    class OpenAIChatCompletion:
        async def make_openai_chat_completion_request(self, client, data):
            return data

    _fake_module(
        monkeypatch,
        "litellm.llms.openai.openai",
        OpenAIChatCompletion=OpenAIChatCompletion,
    )
    with pytest.raises(RuntimeError, match=r"'provider_retry'.*openai_aclient"):
        runpy.run_path(str(PROVIDERS / "litellm_provider_retry_patch.py"))


def test_gemini_passthrough_names_itself_when_its_target_changed(monkeypatch):
    class PassThroughStreamingHandler:
        @staticmethod
        def _build_passthrough_logging_result(*, litellm_logging_obj, raw_bytes):
            return None, {}

    class EndpointType:
        VERTEX_AI = "vertex-ai"

    _fake_module(
        monkeypatch,
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers"
        ".gemini_passthrough_logging_handler",
        GeminiPassthroughLoggingHandler=object,
    )
    monkeypatch.setitem(
        sys.modules,
        "litellm.proxy.pass_through_endpoints.streaming_handler",
        types.SimpleNamespace(PassThroughStreamingHandler=PassThroughStreamingHandler),
    )
    _fake_module(
        monkeypatch,
        "litellm.types.passthrough_endpoints.pass_through_endpoints",
        EndpointType=EndpointType,
    )
    with pytest.raises(RuntimeError, match=r"'gemini_passthrough'.*url_route"):
        runpy.run_path(str(PROVIDERS / "litellm_gemini_passthrough_patch.py"))


def test_the_installed_litellm_still_has_every_target():
    """Against the pinned LiteLLM every patch applies (none raises). In a
    fresh interpreter, as the proxy loads them, so this process's LiteLLM is
    left unpatched."""
    script = "\n".join(
        [
            "import runpy",
            *(
                f"runpy.run_path({str(PROVIDERS / name)!r})"
                for name in (
                    "litellm_token_capture_patch.py",
                    "litellm_provider_retry_patch.py",
                    "litellm_gemini_passthrough_patch.py",
                )
            ),
            "from benchflow.providers.litellm_bedrock_patch import "
            "register_claude_effort_capabilities",
            "register_claude_effort_capabilities(['anthropic/claude-opus-5-5'])",
            "from benchflow.providers.litellm_messages_bridge_patch import "
            "install_messages_bridge_patch",
            "install_messages_bridge_patch()",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_effort_capability_names_itself_when_its_lookup_moved(monkeypatch):
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    from benchflow.providers.litellm_bedrock_patch import (
        register_claude_effort_capabilities,
    )

    monkeypatch.delattr(AnthropicModelInfo, "_get_model_capability")
    with pytest.raises(RuntimeError, match="'effort_capability'"):
        register_claude_effort_capabilities(["anthropic/claude-opus-5-5"])


def test_messages_bridge_names_itself_when_its_target_moved(monkeypatch):
    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )

    from benchflow.providers import litellm_messages_bridge_patch as bridge

    monkeypatch.setattr(bridge, "_litellm_checked", lambda: True)
    monkeypatch.delattr(
        LiteLLMAnthropicMessagesAdapter, "translate_anthropic_messages_to_openai"
    )
    with pytest.raises(RuntimeError, match="'messages_bridge'"):
        bridge.install_messages_bridge_patch()


def test_a_failed_patch_stops_the_proxy_start_and_names_it(tmp_path):
    """sitecustomize turns a patch's failure into the proxy's exit, with the
    patch's name in what it says."""
    from benchflow.providers.litellm_runtime import (
        _PATCH_MODULE,
        _sitecustomize_source,
    )

    (tmp_path / "sitecustomize.py").write_text(
        _sitecustomize_source({"model_list": []})
    )
    (tmp_path / f"{_PATCH_MODULE}.py").write_text(
        "raise RuntimeError(\"BenchFlow LiteLLM patch 'effort_capability': "
        'AnthropicModelInfo._get_model_capability moved")\n'
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", "import sitecustomize"],
        cwd=tmp_path,
        env={"PYTHONPATH": str(tmp_path), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "BenchFlow LiteLLM patch failed" in result.stderr
    assert "effort_capability" in result.stderr
