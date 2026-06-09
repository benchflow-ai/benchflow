#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

TASK_SKILL_KEYS = {
    "expected_task_skill_names",
    "expected_task_skills",
    "required_task_skills",
    "task_bundled_skills",
    "task_skill_names",
    "task_skills",
}

TASK_PATH_KEYS = {"task_dir", "task_path", "tasks_dir"}


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


def normalize_skill_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def unique_sorted(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = value.strip()
        key = normalize_skill_name(clean)
        if clean and key not in seen:
            seen.add(key)
            result.append(clean)
    return sorted(result, key=normalize_skill_name)


def values_to_skill_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,;\n]", value) if part.strip()]
    if isinstance(value, list):
        names: list[str] = []
        for item in value:
            names.extend(values_to_skill_names(item))
        return names
    if isinstance(value, dict):
        for key in ("name", "skill", "id"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return [candidate]
    return []


def walk_dicts(value: Any) -> list[dict[str, Any]]:
    dicts: list[dict[str, Any]] = []
    if isinstance(value, dict):
        dicts.append(value)
        for item in value.values():
            dicts.extend(walk_dicts(item))
    elif isinstance(value, list):
        for item in value:
            dicts.extend(walk_dicts(item))
    return dicts


def infer_skill_mode(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"with_skill", "with_skills", "with_task_skills", "task_skills", "skills"}:
        return "with_skills"
    if text in {"without_skill", "without_skills", "no_skill", "no_skills", "baseline"}:
        return "without_skills"
    return "unknown"


def discover_task_skills_dir(task_path: Path) -> list[str]:
    skills_root = task_path / "environment" / "skills"
    if not skills_root.is_dir():
        return []
    return [
        child.name
        for child in sorted(skills_root.iterdir())
        if child.is_dir() and (child / "SKILL.md").exists()
    ]


def load_json_file(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None


def metadata_search_roots(trajectory: Path) -> list[Path]:
    roots: list[Path] = []
    for candidate in [
        trajectory.parent,
        trajectory.parent.parent,
        trajectory.parent.parent.parent,
        trajectory.parent.parent.parent.parent,
    ]:
        if candidate not in roots:
            roots.append(candidate)
    return [root for root in roots if root.exists()]


def metadata_files(trajectory: Path) -> list[Path]:
    names = (
        "run_config.json",
        "config.json",
        "metadata.json",
        "result.json",
        "results.json",
        "rollout_config.json",
        "task_config.json",
    )
    files: list[Path] = []
    for root in metadata_search_roots(trajectory):
        for name in names:
            path = root / name
            if path.is_file() and path not in files:
                files.append(path)
    return files


def infer_task_skill_context(
    trajectory: Path,
    cli_task_skills: list[str],
    cli_task_path: Path | None,
) -> dict[str, Any]:
    expected_names: list[str] = list(cli_task_skills)
    mode = "unknown"
    evidence: list[str] = []
    metadata_paths: list[str] = []
    task_paths: list[Path] = []

    if cli_task_skills:
        evidence.append("cli:--task-skill")
    if cli_task_path is not None:
        task_paths.append(cli_task_path)
        evidence.append("cli:--task-path")

    for path in metadata_files(trajectory):
        metadata = load_json_file(path)
        if metadata is None:
            continue
        metadata_paths.append(str(path))
        for item in walk_dicts(metadata):
            for key, value in item.items():
                if key in TASK_SKILL_KEYS:
                    names = values_to_skill_names(value)
                    if names:
                        expected_names.extend(names)
                        evidence.append(f"{path.name}:{key}")
                elif key == "skill_mode":
                    inferred = infer_skill_mode(value)
                    if inferred != "unknown":
                        mode = inferred
                elif key == "include_task_skills":
                    if value is True:
                        mode = "with_skills"
                    elif value is False and mode == "unknown":
                        mode = "without_skills"
                elif key in TASK_PATH_KEYS and isinstance(value, str):
                    candidate = Path(value).expanduser()
                    if candidate.exists():
                        task_paths.append(candidate)

    for task_path in task_paths:
        names = discover_task_skills_dir(task_path)
        if names:
            expected_names.extend(names)
            evidence.append(f"{task_path}:environment/skills")

    return {
        "task_skill_mode": mode,
        "expected_task_skills": unique_sorted(expected_names),
        "task_skill_evidence": unique_sorted(evidence),
        "metadata_files": unique_sorted(metadata_paths),
    }


def catalog_skill_names(skills: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for skill in skills:
        name = skill.get("name")
        if isinstance(name, str):
            names.append(name)
    return unique_sorted(names)


def task_skill_loading_fields(
    extracted: dict[str, Any],
    task_skill_context: dict[str, Any],
) -> dict[str, Any]:
    expected = task_skill_context.get("expected_task_skills", [])
    expected_norm = {normalize_skill_name(name) for name in expected}
    catalog_names = catalog_skill_names(extracted.get("skills", []))
    catalog_norm_to_name = {normalize_skill_name(name): name for name in catalog_names}

    loaded = [catalog_norm_to_name[name] for name in sorted(expected_norm) if name in catalog_norm_to_name]
    missing = [name for name in expected if normalize_skill_name(name) not in catalog_norm_to_name]
    mode = task_skill_context.get("task_skill_mode", "unknown")

    if expected and mode == "without_skills":
        loading = 1 if not missing else 0
        if loading:
            status = "unexpected_complete_task_skills_loaded_without_skills"
        elif loaded:
            status = "partial_unexpected_task_skills_loaded_without_skills"
        else:
            status = "not_loaded_without_skills"
    elif expected:
        loading = 1 if not missing else 0
        status = "complete" if loading else "missing_expected_task_skills"
    elif mode == "without_skills":
        loading = 0
        status = "not_expected_without_skills"
    elif mode == "with_skills":
        loading = 0
        status = "expected_with_skills_but_no_task_skill_manifest"
    else:
        loading = 0
        status = "unknown_expected_task_skills"

    return {
        "task_skills_loading": loading,
        "task_skills_loading_status": status,
        "task_skill_mode": mode,
        "expected_task_skills": expected,
        "loaded_task_skills": unique_sorted(loaded),
        "missing_task_skills": unique_sorted(missing),
        "task_skill_evidence": task_skill_context.get("task_skill_evidence", []),
        "metadata_files": task_skill_context.get("metadata_files", []),
    }


def skill_name_variants(name: str) -> set[str]:
    normalized = normalize_skill_name(name)
    return {
        name.strip().lower(),
        normalized,
        normalized.replace("-", "_"),
        normalized.replace("-", " "),
    }


def _contains_skill_variant(text: str, variants: set[str]) -> bool:
    lowered = text.lower()
    return any(variant and variant in lowered for variant in variants)


def _skill_path_evidence(event: dict[str, Any], variants: set[str]) -> list[str]:
    evidence: list[str] = []
    path_markers = (".agents/skills", ".codex/skills", "/skills/", "skill.md")
    for text in iter_strings(event):
        lowered = text.lower()
        if not any(marker in lowered for marker in path_markers):
            continue
        if _contains_skill_variant(lowered, variants):
            evidence.append(text)
    return evidence


def _skill_invocation_evidence(event: dict[str, Any], variants: set[str]) -> list[str]:
    evidence: list[str] = []
    invocation_markers = {"activate_skill", "invoke_skill", "skill"}
    invocation_keys = {"action", "function", "kind", "name", "tool", "tool_name", "type"}
    argument_keys = {"args", "arguments", "input", "parameters"}

    for item in walk_dicts(event):
        marker_values = [
            value
            for key, value in item.items()
            if key in invocation_keys and isinstance(value, str)
        ]
        if not any(value.lower() in invocation_markers for value in marker_values):
            continue

        argument_texts = [
            textify(value)
            for key, value in item.items()
            if key in argument_keys
        ]
        if not argument_texts:
            argument_texts = [textify(item)]
        if any(_contains_skill_variant(text, variants) for text in argument_texts):
            evidence.append(textify(item))
    return evidence


def _no_skill_event_evidence(
    event: dict[str, Any],
    variants: set[str],
) -> tuple[str, list[str]] | None:
    path_hits = _skill_path_evidence(event, variants)
    if path_hits:
        return "task skill path", path_hits

    invocation_hits = _skill_invocation_evidence(event, variants)
    if invocation_hits:
        return "task skill invocation", invocation_hits

    return None


def no_skill_leakage_fields(
    events: list[dict[str, Any]],
    task_skill_context: dict[str, Any],
) -> dict[str, Any]:
    mode = task_skill_context.get("task_skill_mode", "unknown")
    expected = task_skill_context.get("expected_task_skills", [])
    if mode != "without_skills" or not expected:
        return {
            "no_skill_leakage_detected": False,
            "no_skill_leakage_evidence": [],
        }

    evidence: list[str] = []
    variants_by_skill = {
        skill: skill_name_variants(skill)
        for skill in expected
        if isinstance(skill, str) and skill.strip()
    }

    for idx, event in enumerate(events):
        for skill, variants in variants_by_skill.items():
            event_evidence = _no_skill_event_evidence(event, variants)
            if event_evidence is None:
                continue
            evidence_type, hits = event_evidence
            evidence.append(
                f"line {idx}: expected task skill {skill!r} appeared in "
                f"{evidence_type}: {hits[0][:200]}"
            )
            break

    return {
        "no_skill_leakage_detected": bool(evidence),
        "no_skill_leakage_evidence": evidence[:20],
    }


def apply_no_skill_leakage_audit(
    result: dict[str, Any],
    events: list[dict[str, Any]],
    task_skill_context: dict[str, Any],
) -> dict[str, Any]:
    leakage = no_skill_leakage_fields(events, task_skill_context)
    result.update(leakage)
    if leakage["no_skill_leakage_detected"]:
        result["manual_review_required"] = True
    return result


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
    task_skill_context: dict[str, Any],
) -> dict[str, Any]:
    source_text = extracted.pop("_source_text", "")
    skill_count = len(extracted.get("skills", []))
    task_loading = task_skill_loading_fields(extracted, task_skill_context)
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
            "checked_files": checked_files,
        }
    )
    extracted.update(task_loading)
    extracted["manual_review_required"] = (
        skill_count == 0
        or task_loading["task_skills_loading_status"]
        in {
            "expected_with_skills_but_no_task_skill_manifest",
            "missing_expected_task_skills",
            "partial_unexpected_task_skills_loaded_without_skills",
            "unexpected_complete_task_skills_loaded_without_skills",
        }
    )
    return extracted


def extract_from_events(
    path: Path,
    events: list[dict[str, Any]],
    checked_files: list[str],
    task_skill_context: dict[str, Any],
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
                task_skill_context,
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
                    task_skill_context,
                )
    return None


def unknown_result(
    trajectory: Path,
    checked_files: list[str],
    task_skill_context: dict[str, Any],
) -> dict[str, Any]:
    task_loading = task_skill_loading_fields({"skills": []}, task_skill_context)
    return {
        "trajectory": str(trajectory),
        "harness": "unknown",
        "catalog_format": "No supported startup skill catalog found in checked request bodies or ACP system prompts",
        "catalog_status": "unknown",
        "skills": [],
        "skill_count": 0,
        "manual_review_required": True,
        "checked_files": checked_files,
        **task_loading,
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
    parser.add_argument(
        "--task-skill",
        action="append",
        default=[],
        help="Expected task-specific skill name. Repeat for multiple skills.",
    )
    parser.add_argument(
        "--task-path",
        type=Path,
        default=None,
        help="Optional task directory; environment/skills/*/SKILL.md names are treated as expected task skills.",
    )
    args = parser.parse_args()

    limit = None if args.max_lines == 0 else args.max_lines
    checked_files = [str(args.trajectory)]
    task_skill_context = infer_task_skill_context(
        args.trajectory,
        args.task_skill,
        args.task_path,
    )
    events = read_jsonl(args.trajectory, limit=limit)
    audit_events = list(events)
    fallback = sibling_acp_path(args.trajectory)
    fallback_events: list[dict[str, Any]] = []
    if fallback is not None:
        checked_files.append(str(fallback))
        fallback_events = read_jsonl(fallback, limit=limit)
        audit_events.extend(fallback_events)

    extracted = extract_from_events(args.trajectory, events, checked_files, task_skill_context)

    if extracted is None and fallback is not None:
        extracted = extract_from_events(
            fallback,
            fallback_events,
            checked_files,
            task_skill_context,
        )

    result = (
        extracted
        if extracted is not None
        else unknown_result(args.trajectory, checked_files, task_skill_context)
    )
    result = apply_no_skill_leakage_audit(result, audit_events, task_skill_context)

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
