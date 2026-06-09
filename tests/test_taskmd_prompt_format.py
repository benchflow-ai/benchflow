"""task.md format: instruction body follows frontmatter directly (no ## prompt heading)."""
from benchflow.task.document import render_task_md, TaskDocument

FM = {"schema_version": "1.0"}

def test_render_omits_prompt_heading():
    out = render_task_md(FM, "Do the thing.")
    assert "## prompt" not in out
    assert out.startswith("---\n")
    assert out.rstrip().endswith("Do the thing.")

def test_roundtrip_preamble_is_instruction():
    out = render_task_md(FM, "Do the thing.\nLine two.")
    assert TaskDocument.from_text(out).instruction == "Do the thing.\nLine two."

def test_roundtrip_instruction_with_literal_reserved_heading():
    instr = "Write a file.\n## prompt\nnot a real section."
    out = render_task_md(FM, instr)
    assert "\\## prompt" in out  # escaped on render
    assert TaskDocument.from_text(out).instruction == instr  # unescaped on parse

def test_legacy_prompt_heading_still_parses():
    text = "---\nschema_version: \"1.0\"\n---\n\n## prompt\n\nLegacy body."
    assert TaskDocument.from_text(text).instruction == "Legacy body."

def test_roundtrip_with_role_section_keeps_preamble_instruction():
    text = render_task_md(FM, "Main instruction.") .rstrip() + "\n\n## role:solver\nsolver guidance.\n"
    doc = TaskDocument.from_text(text)
    assert doc.instruction == "Main instruction."
    assert doc.role_prompts.get("solver") == "solver guidance."
