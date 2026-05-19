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
