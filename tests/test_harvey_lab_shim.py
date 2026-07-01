from pathlib import Path

from benchflow.agents.harvey_lab_acp_shim import (
    DirectSandbox,
    _mirror_workspace_outputs,
)


def test_mirror_workspace_outputs_collects_root_deliverable(tmp_path: Path):
    """Guards ENG-79: workspace-root deliverables are collected for verifier."""
    workspace = tmp_path / "workspace"
    output = tmp_path / "output"
    workspace.mkdir()
    output.mkdir()
    (workspace / "buy-side-cim-analysis-memo.docx").write_bytes(b"docx")
    (workspace / "rubric.json").write_text("{}")
    (workspace / ".scratch").write_text("ignore")
    (workspace / "documents").mkdir()
    (workspace / "documents" / "source.docx").write_bytes(b"source")
    (workspace / "skills").mkdir()
    (workspace / "skills" / "helper.py").write_text("print('ignore')\n")

    mirrored = _mirror_workspace_outputs(workspace, output)

    assert mirrored == 1
    assert (output / "buy-side-cim-analysis-memo.docx").read_bytes() == b"docx"
    assert not (output / "rubric.json").exists()
    assert not (output / "source.docx").exists()
    assert not (output / "helper.py").exists()


def test_direct_sandbox_rewrites_harvey_absolute_paths(tmp_path: Path):
    """Guards ENG-79: Harvey read-tool parser paths map into BenchFlow /app."""
    documents = tmp_path / "documents"
    output = tmp_path / "output"
    workspace = tmp_path / "app"
    sandbox = DirectSandbox(documents, output, workspace)

    command = sandbox._rewrite_command_paths(
        "parse-doc docx /workspace/documents/source.docx && "
        "ls /workspace/output && cat /workspace/answer.md"
    )

    assert str(documents / "source.docx") in command
    assert f"ls {output}" in command
    assert str(workspace / "answer.md") in command


def test_create_adapter_defaults_unknown_models_to_openai(monkeypatch):
    # PR #871: _create_adapter must default any non-claude/non-gemini id (deepseek,
    # qwen, the benchflow-* proxy alias, ...) to the OpenAI-compatible adapter
    # instead of raising — a raise ended the turn with zero LLM calls
    # (suspected_api_error, reward 0). claude/gemini still route to their adapters.
    import sys
    import types

    import benchflow.agents.harvey_lab_acp_shim as shim

    def _fake_mod(mod_name, cls_name):
        module = types.ModuleType(mod_name)

        class _Adapter:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        _Adapter.__name__ = cls_name
        setattr(module, cls_name, _Adapter)
        return module, _Adapter

    openai_mod, OpenAIAdapter = _fake_mod("harness.adapters.openai", "OpenAIAdapter")
    anthropic_mod, AnthropicAdapter = _fake_mod(
        "harness.adapters.anthropic", "AnthropicAdapter"
    )
    google_mod, GoogleAdapter = _fake_mod("harness.adapters.google", "GoogleAdapter")

    for name, module in [
        ("harness", types.ModuleType("harness")),
        ("harness.adapters", types.ModuleType("harness.adapters")),
        ("harness.adapters.openai", openai_mod),
        ("harness.adapters.anthropic", anthropic_mod),
        ("harness.adapters.google", google_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, module)

    assert isinstance(shim._create_adapter("deepseek-v4-flash"), OpenAIAdapter)
    assert isinstance(shim._create_adapter("qwen-3-max"), OpenAIAdapter)
    assert isinstance(
        shim._create_adapter("benchflow-openai-gpt-5.4-mini"), OpenAIAdapter
    )
    assert isinstance(shim._create_adapter("claude-opus-4-5"), AnthropicAdapter)
    assert isinstance(shim._create_adapter("gemini-3-flash"), GoogleAdapter)
    # a provider-qualified id is stripped to the bare model before dispatch
    assert isinstance(shim._create_adapter("deepseek/deepseek-v4-flash"), OpenAIAdapter)
