#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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


def iter_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        strings: list[str] = []
        for item in value:
            strings.extend(iter_strings(item))
        return strings
    if isinstance(value, dict):
        strings = []
        for item in value.values():
            strings.extend(iter_strings(item))
        return strings
    return []


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


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


def result_with_source(
    result: dict[str, Any],
    source_text: str,
    catalog_status: str,
) -> dict[str, Any]:
    result["_source_text"] = source_text
    result["catalog_status"] = catalog_status
    return result


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
        return result_with_source(
            {
                "harness": "codex",
                "catalog_format": "OpenAI Responses: body.instructions + input[role=developer]; no <skills_instructions> block found",
                "skills": [],
            },
            developer_text,
            "catalog_absent_in_startup_prompt",
        )
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
    return result_with_source(
        {
            "harness": "codex",
            "catalog_format": "<skills_instructions> Markdown; entries are '- name: description (file: path)' in input[role=developer]",
            "skills": skills,
        },
        block,
        "catalog_found",
    )


def extract_openhands(body: dict[str, Any]) -> dict[str, Any] | None:
    text = first_text_block(body)
    if "<SKILLS>" not in text and "invoke_skill" not in text:
        return None
    block_match = re.search(r"<available_skills>(.*?)</available_skills>", text, re.S)
    skills = []
    source_text = text
    if block_match:
        source_text = block_match.group(1)
        for item in re.findall(r"<skill>(.*?)</skill>", block_match.group(1), re.S):
            name = re.search(r"<name>(.*?)</name>", item, re.S)
            desc = re.search(r"<description>(.*?)</description>", item, re.S)
            skills.append(
                {
                    "name": textify(name.group(1)).strip() if name else "",
                    "description": textify(desc.group(1)).strip() if desc else "",
                }
            )
    return result_with_source(
        {
            "harness": "openhands",
            "catalog_format": "<SKILLS><available_skills><skill><name>...</name><description>...</description></skill>...",
            "skills": [skill for skill in skills if skill["name"]],
        },
        source_text,
        "catalog_found" if skills else "catalog_absent_in_startup_prompt",
    )


def extract_gemini(body: dict[str, Any]) -> dict[str, Any] | None:
    text = first_text_block(body)
    if "# Available Agent Skills" not in text and "activate_skill" not in text:
        return None
    block_match = re.search(r"<available_skills>(.*?)</available_skills>", text, re.S)
    skills = []
    source_text = text
    if block_match:
        source_text = block_match.group(1)
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
    return result_with_source(
        {
            "harness": "gemini-cli",
            "catalog_format": "# Available Agent Skills + XML-ish <available_skills>; entries include name/description/location",
            "skills": [skill for skill in skills if skill["name"]],
        },
        source_text,
        "catalog_found" if skills else "catalog_absent_in_startup_prompt",
    )


def extract_claude_code(body: dict[str, Any]) -> dict[str, Any] | None:
    text = first_text_block(body)
    marker = "The following skills are available for use with the Skill tool:"
    if marker not in text and "Skill" not in text:
        return None
    skills: list[dict[str, str]] = []
    source_text = text
    marker_match = re.search(
        r"The following skills are available for use with the Skill tool:\n\n(.*?)(?:\n\n#|\n\n[A-Z][^\n]*:|\Z)",
        text,
        re.S,
    )
    if marker_match:
        source_text = marker_match.group(1)
        for name, desc in re.findall(
            r"^- ([^:\n]+): (.*?)(?=\n- |\Z)",
            marker_match.group(1),
            re.M | re.S,
        ):
            skills.append({"name": name.strip(), "description": " ".join(desc.split())})
    return result_with_source(
        {
            "harness": "claude-code",
            "catalog_format": "Claude Code system reminder; entries are Markdown bullets '- name: description' after 'The following skills are available for use with the Skill tool:'",
            "skills": skills,
        },
        source_text,
        "catalog_found" if skills else "catalog_absent_in_startup_prompt",
    )


def extract_pi(body: dict[str, Any]) -> dict[str, Any] | None:
    text = first_text_block(body)
    if "pi, a coding agent harness" not in text and "pi-coding-agent" not in text:
        return None
    return result_with_source(
        {
            "harness": "pi-acp",
            "catalog_format": "No startup skill catalog serialized in current PR4 trajectories; system prompt only references pi docs/skills.md",
            "skills": [],
        },
        text,
        "catalog_not_serialized",
    )


EXTRACTORS = [extract_codex, extract_openhands, extract_gemini, extract_claude_code, extract_pi]


def extract_from_body(body: dict[str, Any]) -> dict[str, Any] | None:
    for extractor in EXTRACTORS:
        extracted = extractor(body)
        if extracted is not None:
            return extracted
    return None


def bodies_from_text(text: str) -> list[dict[str, Any]]:
    bodies = [{"messages": [{"role": "system", "content": text}]}]
    if "<skills_instructions>" in text:
        bodies.insert(
            0,
            {
                "instructions": "",
                "input": [{"role": "developer", "content": [{"type": "input_text", "text": text}]}],
            },
        )
    if "# Available Agent Skills" in text:
        bodies.insert(0, {"systemInstruction": {"parts": [{"text": text}]}})
    return bodies


def system_prompt_candidates(events: list[dict[str, Any]]) -> list[tuple[int, str]]:
    candidates: list[tuple[int, str]] = []
    anchors = (
        "System Prompt:",
        "<skills_instructions>",
        "<SKILLS>",
        "# Available Agent Skills",
        "The following skills are available for use with the Skill tool:",
    )
    for idx, event in enumerate(events):
        for text in iter_strings(event):
            if not any(anchor in text for anchor in anchors):
                continue
            if "System Prompt:" in text:
                text = text.split("System Prompt:", 1)[1].strip()
            candidates.append((idx, text))
    return candidates


def finalize_result(
    extracted: dict[str, Any],
    path: Path,
    line_index: int,
    request_path: str,
    catalog_source: str,
    checked_files: list[str],
) -> dict[str, Any]:
    source_text = extracted.pop("_source_text", "")
    skill_count = len(extracted.get("skills", []))
    extracted.update(
        {
            "trajectory": str(path),
            "source_file": path.name,
            "line_index": line_index,
            "request_path": request_path,
            "catalog_source": catalog_source,
            "catalog_sha256": sha256_text(source_text) if source_text else None,
            "catalog_text_chars": len(source_text),
            "skill_count": skill_count,
            "manual_review_required": skill_count == 0,
            "checked_files": checked_files,
        }
    )
    return extracted


def extract_from_events(
    path: Path,
    events: list[dict[str, Any]],
    checked_files: list[str],
) -> dict[str, Any] | None:
    for line_index, request_path, body in request_bodies(events):
        extracted = extract_from_body(body)
        if extracted is not None:
            return finalize_result(
                extracted,
                path,
                line_index,
                request_path,
                "request.body",
                checked_files,
            )

    for line_index, text in system_prompt_candidates(events):
        for body in bodies_from_text(text):
            extracted = extract_from_body(body)
            if extracted is not None:
                return finalize_result(
                    extracted,
                    path,
                    line_index,
                    "",
                    "agent_thought_system_prompt",
                    checked_files,
                )
    return None


def unknown_result(trajectory: Path, checked_files: list[str]) -> dict[str, Any]:
    return {
        "trajectory": str(trajectory),
        "harness": "unknown",
        "catalog_format": "No supported startup skill catalog found in checked request bodies or ACP system prompts",
        "catalog_status": "unknown",
        "skills": [],
        "skill_count": 0,
        "manual_review_required": True,
        "checked_files": checked_files,
    }


def sibling_acp_path(path: Path) -> Path | None:
    sibling = path.with_name("acp_trajectory.jsonl")
    if sibling.exists() and sibling != path:
        return sibling
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path, help="Path to llm_trajectory.jsonl")
    parser.add_argument(
        "--max-lines",
        type=int,
        default=512,
        help="Maximum JSONL lines to scan per trajectory; use 0 to scan all lines.",
    )
    args = parser.parse_args()

    limit = None if args.max_lines == 0 else args.max_lines
    checked_files = [str(args.trajectory)]
    events = read_jsonl(args.trajectory, limit=limit)
    extracted = extract_from_events(args.trajectory, events, checked_files)

    if extracted is None:
        fallback = sibling_acp_path(args.trajectory)
        if fallback is not None:
            checked_files.append(str(fallback))
            acp_events = read_jsonl(fallback, limit=limit)
            extracted = extract_from_events(fallback, acp_events, checked_files)

    print(
        json.dumps(
            extracted if extracted is not None else unknown_result(args.trajectory, checked_files),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
