#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def textify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(part for part in (textify(item) for item in value) if part)
    if isinstance(value, dict):
        parts: list[str] = []
        for key in ("text", "content", "parts"):
            if key in value and value[key] is not None:
                parts.append(textify(value[key]))
        return "\n".join(part for part in parts if part)
    return str(value)


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, line in enumerate(path.read_text(errors="replace").splitlines()):
        if limit is not None and idx >= limit:
            break
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def request_bodies(events: list[dict[str, Any]]) -> list[tuple[int, str, dict[str, Any]]]:
    bodies: list[tuple[int, str, dict[str, Any]]] = []
    for idx, event in enumerate(events):
        request = event.get("request")
        if not isinstance(request, dict):
            continue
        body = request.get("body")
        if isinstance(body, dict):
            bodies.append((idx, str(request.get("path", "")), body))
    return bodies


def first_text_block(body: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("instructions", "system", "systemInstruction", "system_instruction"):
        if key in body:
            parts.append(textify(body[key]))
    for key in ("messages", "input", "contents"):
        seq = body.get(key)
        if isinstance(seq, list):
            parts.extend(textify(item) for item in seq if isinstance(item, dict))
    return "\n\n".join(part for part in parts if part)


def extract_codex(body: dict[str, Any]) -> dict[str, Any] | None:
    if "instructions" not in body or "input" not in body:
        return None
    developer_parts: list[str] = []
    for message in body.get("input", []):
        if isinstance(message, dict) and message.get("role") == "developer":
            developer_parts.append(textify(message.get("content")))
    developer_text = "\n\n".join(developer_parts)
    block_match = re.search(
        r"<skills_instructions>(.*?)</skills_instructions>",
        developer_text,
        re.S,
    )
    if not block_match:
        return {
            "harness": "codex",
            "catalog_format": "OpenAI Responses: body.instructions + input[role=developer]; no <skills_instructions> block found",
            "skills": [],
        }
    block = block_match.group(1)
    list_match = re.search(
        r"### Available skills\n(.*?)(?:\n### |\Z)",
        block,
        re.S,
    )
    list_text = list_match.group(1) if list_match else block
    skills = [
        {"name": name.strip(), "file": file.strip(), "description": desc.strip()}
        for name, desc, file in re.findall(
            r"^- ([^:\n]+): (.*?)(?: \(file: ([^)]+)\))$",
            list_text,
            re.M,
        )
    ]
    return {
        "harness": "codex",
        "catalog_format": "<skills_instructions> Markdown; entries are '- name: description (file: path)' in input[role=developer]",
        "skills": skills,
    }


def extract_openhands(body: dict[str, Any]) -> dict[str, Any] | None:
    text = first_text_block(body)
    if "<SKILLS>" not in text and "invoke_skill" not in text:
        return None
    block_match = re.search(r"<available_skills>(.*?)</available_skills>", text, re.S)
    skills = []
    if block_match:
        for item in re.findall(r"<skill>(.*?)</skill>", block_match.group(1), re.S):
            name = re.search(r"<name>(.*?)</name>", item, re.S)
            desc = re.search(r"<description>(.*?)</description>", item, re.S)
            skills.append(
                {
                    "name": textify(name.group(1)).strip() if name else "",
                    "description": textify(desc.group(1)).strip() if desc else "",
                }
            )
    return {
        "harness": "openhands",
        "catalog_format": "<SKILLS><available_skills><skill><name>...</name><description>...</description></skill>...",
        "skills": [skill for skill in skills if skill["name"]],
    }


def extract_gemini(body: dict[str, Any]) -> dict[str, Any] | None:
    text = first_text_block(body)
    if "# Available Agent Skills" not in text and "activate_skill" not in text:
        return None
    block_match = re.search(r"<available_skills>(.*?)</available_skills>", text, re.S)
    skills = []
    if block_match:
        for item in re.findall(r"<skill>(.*?)</skill>", block_match.group(1), re.S):
            name = re.search(r"<name>(.*?)</name>", item, re.S)
            desc = re.search(r"<description>(.*?)</description>", item, re.S)
            loc = re.search(r"<location>(.*?)</location>", item, re.S)
            skills.append(
                {
                    "name": textify(name.group(1)).strip() if name else "",
                    "description": textify(desc.group(1)).strip() if desc else "",
                    "location": textify(loc.group(1)).strip() if loc else "",
                }
            )
    return {
        "harness": "gemini-cli",
        "catalog_format": "# Available Agent Skills + XML-ish <available_skills>; entries include name/description/location",
        "skills": [skill for skill in skills if skill["name"]],
    }


def extract_claude_code(body: dict[str, Any]) -> dict[str, Any] | None:
    text = first_text_block(body)
    marker = "The following skills are available for use with the Skill tool:"
    if marker not in text and "Skill" not in text:
        return None
    skills: list[dict[str, str]] = []
    marker_match = re.search(
        r"The following skills are available for use with the Skill tool:\n\n(.*?)(?:\n\n#|\n\n[A-Z][^\n]*:|\Z)",
        text,
        re.S,
    )
    if marker_match:
        for name, desc in re.findall(r"^- ([^:\n]+): (.*?)(?=\n- |\Z)", marker_match.group(1), re.M | re.S):
            skills.append({"name": name.strip(), "description": " ".join(desc.split())})
    return {
        "harness": "claude-code",
        "catalog_format": "Claude Code system reminder; entries are Markdown bullets '- name: description' after 'The following skills are available for use with the Skill tool:'",
        "skills": skills,
    }


def extract_pi(body: dict[str, Any]) -> dict[str, Any] | None:
    text = first_text_block(body)
    if "pi, a coding agent harness" not in text and "pi-coding-agent" not in text:
        return None
    return {
        "harness": "pi-acp",
        "catalog_format": "No startup skill catalog serialized in current PR4 trajectories; system prompt only references pi docs/skills.md",
        "skills": [],
    }


EXTRACTORS = [extract_codex, extract_openhands, extract_gemini, extract_claude_code, extract_pi]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path, help="Path to llm_trajectory.jsonl")
    parser.add_argument("--max-lines", type=int, default=64)
    args = parser.parse_args()

    events = read_jsonl(args.trajectory, limit=args.max_lines)
    for line_index, request_path, body in request_bodies(events):
        for extractor in EXTRACTORS:
            extracted = extractor(body)
            if extracted is not None:
                extracted.update(
                    {
                        "trajectory": str(args.trajectory),
                        "line_index": line_index,
                        "request_path": request_path,
                        "skill_count": len(extracted["skills"]),
                    }
                )
                print(json.dumps(extracted, indent=2, ensure_ascii=False))
                return
    print(
        json.dumps(
            {
                "trajectory": str(args.trajectory),
                "harness": "unknown",
                "catalog_format": "No supported startup skill catalog found in first request bodies",
                "skills": [],
                "skill_count": 0,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
