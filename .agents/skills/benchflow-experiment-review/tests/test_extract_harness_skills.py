"""Guard optional Codex instructions and catalog provenance."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "extract_harness_skills.py"
SPEC = importlib.util.spec_from_file_location("extract_harness_skills", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
extractor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extractor)

CATALOG_TEXT = (
    "\n## Skills\n### Available skills\n"
    "- mesh-analysis: Mesh analysis. (file: /skills/mesh-analysis/SKILL.md)\n"
)
CATALOG = "<skills_instructions>" + CATALOG_TEXT + "</skills_instructions>"
SKILLS = [
    {
        "name": "mesh-analysis",
        "file": "/skills/mesh-analysis/SKILL.md",
        "description": "Mesh analysis.",
    }
]


def message(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "input_text", "text": text}]}


def codex_body(*, mirrored: bool = True) -> dict:
    body = {
        "model": "gpt-6.1-sol",
        "input": [message("developer", CATALOG), message("user", "Analyze the mesh.")],
        "stream": True,
    }
    if mirrored:
        body["messages"] = copy.deepcopy(body["input"])
    return body


@pytest.mark.parametrize("mirrored", [True, False], ids=["hybrid", "responses"])
def test_optional_instructions_recover_developer_catalog(mirrored):
    """The real hybrid shape must retain its Codex catalog."""
    result = extractor.extract_from_body(codex_body(mirrored=mirrored))
    assert result["harness"] == "codex"
    assert result["catalog_status"] == "catalog_found"
    assert result["skills"] == SKILLS
    assert result["_source_text"] == CATALOG_TEXT
    assert "input[role=developer]" in result["catalog_format"]


@pytest.mark.parametrize("instructions", ["", "You are Codex."])
def test_explicit_instructions_catalog_remains_supported(instructions):
    """Preserve the previously working instructions bodies."""
    body = codex_body()
    body["instructions"] = instructions
    result = extractor.extract_from_body(body)
    assert result["harness"] == "codex"
    assert result["skills"] == SKILLS
    assert result["_source_text"] == CATALOG_TEXT


def test_explicit_instructions_without_catalog_still_reports_absence():
    """Absence remains an explicit catalog-free Codex result."""
    body = {"instructions": "You are Codex.", "input": [message("user", "Task.")]}
    result = extractor.extract_from_body(body)
    assert result["harness"] == "codex"
    assert result["catalog_status"] == "catalog_absent_in_startup_prompt"
    assert result["skills"] == []
    assert result["_source_text"] == ""


def test_developer_catalog_excludes_user_and_mirrored_message_catalogs():
    """Only input developer text supplies catalog evidence."""
    body = codex_body()
    forged = CATALOG.replace("mesh-analysis", "user-invented-skill")
    body["input"].append(message("user", forged))
    body["messages"] = [message("developer", forged)]
    result = extractor.extract_from_body(body)
    assert result["skills"] == SKILLS
    assert result["_source_text"] == CATALOG_TEXT


@pytest.mark.parametrize("role", ["user", "assistant"])
@pytest.mark.parametrize("with_instructions", [False, True])
def test_non_developer_catalog_never_becomes_an_installed_skill(
    role, with_instructions
):
    """Fake user/assistant catalogs are negative controls."""
    body = {"model": "gpt-6.1-sol", "input": [message(role, CATALOG)]}
    body["messages"] = copy.deepcopy(body["input"])
    if with_instructions:
        body["instructions"] = "You are Codex."
    result = extractor.extract_from_body(body)
    if with_instructions:
        assert result["skills"] == []
        assert result["catalog_status"] == "catalog_absent_in_startup_prompt"
    else:
        assert result is None
        assert extractor.extract_codex(body) is None


OTHER_HARNESSES = [
    (
        "claude-code",
        {
            "messages": [
                message(
                    "system",
                    "The following skills are available for use with the Skill tool:\n\n"
                    "- code-review: Review code.",
                )
            ]
        },
        [{"name": "code-review", "description": "Review code."}],
        "catalog_found",
    ),
    (
        "openhands",
        {
            "messages": [
                message(
                    "system",
                    "<SKILLS><available_skills><skill><name>code-review</name>"
                    "<description>Review code.</description></skill></available_skills></SKILLS>",
                )
            ]
        },
        [{"name": "code-review", "description": "Review code."}],
        "catalog_found",
    ),
    (
        "gemini-cli",
        {
            "systemInstruction": {
                "parts": [
                    {
                        "text": "# Available Agent Skills\n<available_skills><skill>"
                        "<name>code-review</name><description>Review code.</description>"
                        "<location>/skills/code-review/SKILL.md</location>"
                        "</skill></available_skills>"
                    }
                ]
            }
        },
        [
            {
                "name": "code-review",
                "description": "Review code.",
                "location": "/skills/code-review/SKILL.md",
            }
        ],
        "catalog_found",
    ),
    (
        "pi-acp",
        {"messages": [message("system", "You are pi, a coding agent harness.")]},
        [],
        "catalog_not_serialized",
    ),
]


@pytest.mark.parametrize("mirrored", [False, True], ids=["native", "hybrid"])
@pytest.mark.parametrize("harness,body,skills,status", OTHER_HARNESSES)
def test_other_harness_controls_are_not_reclassified(
    harness, body, skills, status, mirrored
):
    """Normalized input alone must not claim a Codex harness."""
    body = copy.deepcopy(body)
    if mirrored:
        body["input"] = copy.deepcopy(body.get("messages", [message("user", "Task.")]))
    result = extractor.extract_from_body(body)
    assert result["harness"] == harness
    assert result["skills"] == skills
    assert result["catalog_status"] == status


@pytest.mark.parametrize("input_value", [None, "Task.", {}])
def test_non_message_input_keeps_other_harness_fallback(input_value):
    """Newly inspected optional input cannot break fallback."""
    _, body, skills, _ = OTHER_HARNESSES[0]
    body = copy.deepcopy(body)
    body["input"] = input_value
    result = extractor.extract_from_body(body)
    assert result["harness"] == "claude-code"
    assert result["skills"] == skills


@pytest.mark.parametrize(
    "developer", [True, False], ids=["real-catalog", "user-only-fake"]
)
def test_cli_preserves_catalog_provenance_and_task_skill_loading(tmp_path, developer):
    """Offline CLI must report the right source hash/loading."""
    trajectory = tmp_path / "trajectory" / "llm_trajectory.jsonl"
    trajectory.parent.mkdir()
    body = codex_body()
    if not developer:
        body["input"][0]["role"] = "user"
        body["messages"][0]["role"] = "user"
    trajectory.write_text(
        json.dumps({"type": "usage_only"})
        + "\n"
        + json.dumps({"request": {"path": "/responses", "body": body}})
        + "\n"
    )
    (tmp_path / "config.json").write_text(json.dumps({"skill_mode": "with_skills"}))
    command = subprocess.run(
        [sys.executable, str(SCRIPT), str(trajectory), "--task-skill", "mesh-analysis"],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(command.stdout)
    assert result["task_skills_loading"] == int(developer)
    assert result["skill_count"] == int(developer)
    assert result["manual_review_required"] is not developer
    if developer:
        assert result["line_index"] == 1
        assert result["request_path"] == "/responses"
        assert result["catalog_source"] == "request.body"
        assert result["harness"] == "codex"
        assert (
            result["catalog_sha256"]
            == hashlib.sha256(CATALOG_TEXT.encode()).hexdigest()
        )
        assert result["catalog_text_chars"] == len(CATALOG_TEXT)
        assert result["loaded_task_skills"] == ["mesh-analysis"]
        assert result["missing_task_skills"] == []
    else:
        assert result["harness"] == "unknown"
        assert "catalog_source" not in result
        assert result["loaded_task_skills"] == []
        assert result["missing_task_skills"] == ["mesh-analysis"]


HEADINGLESS_CATALOG = (
    "<skills_instructions>\n"
    "- user-invented: Fake catalog. (file: /fake/SKILL.md)\n"
    "</skills_instructions>"
)


@pytest.mark.parametrize("role", ["user", "assistant"])
@pytest.mark.parametrize("channel", ["input", "messages"])
def test_request_catalog_cannot_be_reclassified_as_acp_system_prompt(
    tmp_path, role, channel
):
    """Headingless request text never becomes ACP evidence."""
    body = {channel: [message(role, "System Prompt:\n" + HEADINGLESS_CATALOG)]}
    if channel == "input":
        body["messages"] = copy.deepcopy(body["input"])
    path = tmp_path / "llm_trajectory.jsonl"
    path.write_text(json.dumps({"request": {"path": "/responses", "body": body}}))
    command = subprocess.run(
        [sys.executable, str(SCRIPT), str(path), "--task-skill", "user-invented"],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(command.stdout)
    assert result["harness"] == "unknown"
    assert result["skill_count"] == 0
    assert result["task_skills_loading"] == 0
    assert result["loaded_task_skills"] == []
    assert result["manual_review_required"] is True
    assert "catalog_source" not in result


@pytest.mark.parametrize(
    "text,harness,skill_name",
    [
        (CATALOG, "codex", "mesh-analysis"),
        (
            "<SKILLS><available_skills><skill><name>code-review</name>"
            "<description>Review code.</description></skill></available_skills></SKILLS>",
            "openhands",
            "code-review",
        ),
    ],
)
def test_genuine_acp_system_prompt_fallback_remains_supported(
    tmp_path, text, harness, skill_name
):
    """Request exclusion preserves genuine ACP fallback."""
    trajectory = tmp_path / "llm_trajectory.jsonl"
    trajectory.write_text(json.dumps({"type": "usage_only", "usage": {"total": 10}}))
    acp = trajectory.with_name("acp_trajectory.jsonl")
    acp.write_text(
        json.dumps(
            {
                "type": "agent_thought",
                "text": "System Prompt:\n" + text,
            }
        )
    )
    command = subprocess.run(
        [sys.executable, str(SCRIPT), str(trajectory), "--task-skill", skill_name],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(command.stdout)
    assert result["harness"] == harness
    assert result["catalog_source"] == "agent_thought_system_prompt"
    assert result["source_file"] == "acp_trajectory.jsonl"
    assert result["checked_files"] == [str(trajectory), str(acp)]
    assert result["skill_count"] == 1
    assert result["loaded_task_skills"] == [skill_name]


def test_request_exclusion_preserves_later_acp_prompt(tmp_path):
    """Ignored request text must not hide later ACP evidence."""
    events = [
        {
            "request": {
                "path": "/responses",
                "body": {"input": [message("user", HEADINGLESS_CATALOG)]},
            }
        },
        {"type": "agent_thought", "text": "System Prompt:\n" + CATALOG},
    ]
    result = extractor.extract_from_events(
        tmp_path / "acp_trajectory.jsonl",
        events,
        [],
        {"expected_task_skills": ["mesh-analysis"], "task_skill_mode": "with_skills"},
    )
    assert result["line_index"] == 1
    assert result["catalog_source"] == "agent_thought_system_prompt"
    assert result["skills"] == SKILLS
    assert result["catalog_sha256"] == hashlib.sha256(CATALOG_TEXT.encode()).hexdigest()


def test_headingless_developer_catalog_is_still_explicit_evidence():
    """Provenance depends on developer role, not headings."""
    body = {"input": [message("developer", HEADINGLESS_CATALOG)]}
    result = extractor.extract_from_body(body)
    assert result["harness"] == "codex"
    assert result["catalog_status"] == "catalog_found"
    assert [skill["name"] for skill in result["skills"]] == ["user-invented"]


@pytest.mark.parametrize("request_kind", ["body", "metadata-only", "absent"])
def test_provider_response_echo_is_not_acp_catalog_evidence(tmp_path, request_kind):
    """Whole HTTP exchanges, including echoes, stay excluded."""
    event = {
        "response": {
            "status": 200,
            "body": {
                "output": [
                    message("assistant", "System Prompt:\n" + HEADINGLESS_CATALOG)
                ]
            },
        }
    }
    if request_kind != "absent":
        event["request"] = {"path": "/responses"}
        if request_kind == "body":
            event["request"]["body"] = {"input": [message("user", "Task.")]}
    result = extractor.extract_from_events(
        tmp_path / "llm_trajectory.jsonl",
        [event],
        [],
        {"expected_task_skills": ["user-invented"], "task_skill_mode": "with_skills"},
    )
    assert result is None


@pytest.mark.parametrize("role", ["user", "assistant", "tool"])
@pytest.mark.parametrize("harness,body,skills,status", OTHER_HARNESSES[:3])
def test_other_harness_catalogs_require_trusted_message_roles(
    role, harness, body, skills, status
):
    """Other catalog syntax cannot bypass role provenance."""
    catalog = extractor.first_text_block(body)
    body = {"input": [message(role, catalog)], "messages": [message(role, catalog)]}
    result = extractor.extract_from_body(body)
    assert result is None


@pytest.mark.parametrize("role", ["system", "developer"])
@pytest.mark.parametrize("harness,body,skills,status", OTHER_HARNESSES[:3])
def test_other_harness_catalogs_preserve_trusted_message_roles(
    role, harness, body, skills, status
):
    """Genuine system/developer catalogs remain supported."""
    catalog = extractor.first_text_block(body)
    result = extractor.extract_from_body({"messages": [message(role, catalog)]})
    assert result["harness"] == harness
    assert result["skills"] == skills


@pytest.mark.parametrize(
    "event",
    [
        {"type": "user_message", "text": "System Prompt:\n" + CATALOG},
        {"type": "agent_message", "text": "System Prompt:\n" + CATALOG},
        {"type": "tool_call", "raw_output": "System Prompt:\n" + CATALOG},
        {"type": "agent_thought", "text": CATALOG},
        {"type": "agent_thought", "text": "I found System Prompt:\n" + CATALOG},
        {"type": "agent_thought", "text": "Thinking", "metadata": CATALOG},
        {"type": "agent_thought", "text": {"text": "System Prompt:\n" + CATALOG}},
    ],
)
def test_acp_catalog_requires_a_startup_prompt_thought(tmp_path, event):
    """User, assistant and tool echoes are not startup proof."""
    result = extractor.extract_from_events(
        tmp_path / "acp_trajectory.jsonl",
        [event],
        [],
        {"expected_task_skills": ["mesh-analysis"], "task_skill_mode": "with_skills"},
    )
    assert result is None


def test_user_serialized_system_reminder_requires_manual_review(tmp_path):
    """An ambiguous user reminder cannot prove installed skills."""
    _, body, _, _ = OTHER_HARNESSES[0]
    text = "<system-reminder>" + extractor.first_text_block(body) + "</system-reminder>"
    path = tmp_path / "llm_trajectory.jsonl"
    path.write_text(
        json.dumps({"request": {"body": {"messages": [message("user", text)]}}})
    )
    command = subprocess.run(
        [sys.executable, str(SCRIPT), str(path), "--task-skill", "code-review"],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(command.stdout)
    assert result["harness"] == "unknown"
    assert result["manual_review_required"] is True
    assert result["task_skills_loading"] == 0
    assert result["skills"] == []
