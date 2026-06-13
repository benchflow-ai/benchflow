"""Markdown/frontmatter parse and render core for ``task.md`` documents.

This layer owns the :class:`TaskDocument` model, the render entry points, and
the parse helpers that turn frontmatter plus a markdown body into a config,
roles, scenes, turns, and prompt sections. It delegates frontmatter
normalization to :mod:`benchflow.task._document_normalize`.
"""

from __future__ import annotations

import re
import tomllib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from benchflow._types import Role, Scene, Turn
from benchflow.task._document_normalize import (
    TaskDocumentParseError,
    _mapping,
    normalize_task_document_frontmatter,
)
from benchflow.task.config import TaskConfig
from benchflow.task.imports import import_task_config_toml

TASK_DOCUMENT_FILENAME = "task.md"

_DOCUMENT_ONLY_FRONTMATTER_KEYS = {"agents", "benchflow", "scenes", "user"}
_SECTION_RE = re.compile(
    r"^##\s+(prompt|role:[A-Za-z0-9_.-]+|scene:[A-Za-z0-9_.-]+|user-persona)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class TaskDocument:
    """Parsed ``task.md`` document.

    Frontmatter carries the existing task configuration plus document-only
    blocks for roles, scenes, and simulated users. The markdown body carries
    prompts. ``instruction`` is the prompt text that should be exposed through
    the existing ``/instruction.md`` runtime contract.
    """

    frontmatter: dict[str, Any]
    body: str
    instruction: str
    config: TaskConfig
    roles: dict[str, Role]
    scenes: list[Scene]
    role_prompts: dict[str, str]
    scene_prompts: dict[str, str]
    user: dict[str, Any]
    user_persona: str | None
    benchflow: dict[str, Any]
    path: Path | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> TaskDocument:
        doc_path = Path(path)
        return cls.from_text(doc_path.read_text(), path=doc_path)

    @classmethod
    def from_text(cls, text: str, *, path: str | Path | None = None) -> TaskDocument:
        frontmatter, body = _split_frontmatter(text)
        doc_path = Path(path) if path is not None else None
        frontmatter = normalize_task_document_frontmatter(
            frontmatter,
            task_dir=doc_path.parent if doc_path is not None else None,
        )
        prompt_sections = _extract_prompt_sections(body)
        # Sidecar prompt files (prompts/role.<name>.md, prompts/scene.<name>.md,
        # prompts/user-persona.md) are the native authoring surface and take
        # precedence over reserved ## headings, which remain a compat-import path.
        role_prompts = dict(prompt_sections.role_prompts)
        scene_prompts = dict(prompt_sections.scene_prompts)
        user_persona = prompt_sections.user_persona
        if doc_path is not None:
            sidecars = _load_prompt_sidecars(doc_path.parent)
            role_prompts.update(sidecars.role_prompts)
            scene_prompts.update(sidecars.scene_prompts)
            if sidecars.user_persona is not None:
                user_persona = sidecars.user_persona
        config = _config_from_frontmatter(frontmatter)
        roles = _parse_roles(frontmatter)
        scenes = _parse_scenes(
            frontmatter,
            roles=roles,
            instruction=prompt_sections.instruction,
            role_prompts=role_prompts,
            scene_prompts=scene_prompts,
        )
        user = _mapping(frontmatter.get("user"), "user", default={})
        benchflow = _mapping(frontmatter.get("benchflow"), "benchflow", default={})
        return cls(
            frontmatter=frontmatter,
            body=body,
            instruction=prompt_sections.instruction,
            config=config,
            roles=roles,
            scenes=scenes,
            role_prompts=role_prompts,
            scene_prompts=scene_prompts,
            user=user,
            user_persona=user_persona,
            benchflow=benchflow,
            path=doc_path,
        )


@dataclass(frozen=True)
class _PromptSections:
    instruction: str
    role_prompts: dict[str, str]
    scene_prompts: dict[str, str]
    user_persona: str | None


def render_task_md(frontmatter: dict[str, Any] | str, instruction: str) -> str:
    """Render canonical ``task.md`` text from frontmatter plus a prompt body.

    ``frontmatter`` may be a parsed config mapping or raw ``task.toml`` text.
    A legacy ``solution`` block is emitted as the native ``oracle`` block when no
    ``oracle`` block is present; declaring both is rejected. Reserved section
    headings embedded in ``instruction`` are escaped so they round-trip as prompt
    text instead of fracturing the document into extra sections.
    """

    data = (
        tomllib.loads(frontmatter)
        if isinstance(frontmatter, str)
        else deepcopy(frontmatter)
    )
    if "solution" in data:
        if "oracle" in data:
            raise ValueError(
                "task config declares both 'oracle' and 'solution'; keep only 'oracle'"
            )
        data = {
            ("oracle" if key == "solution" else key): value
            for key, value in data.items()
        }
    rendered_frontmatter = yaml.safe_dump(data, sort_keys=False)
    body = _escape_reserved_section_headings(instruction.strip())
    return f"---\n{rendered_frontmatter}---\n\n## prompt\n\n{body}\n"


def render_task_md_from_legacy(task_dir: str | Path) -> str:
    """Render a legacy ``task.toml`` + ``instruction.md`` task as ``task.md``.

    The generated frontmatter stays minimal: only keys the source ``task.toml``
    actually declared are emitted, under the canonical ``schema_version``
    spelling, instead of materializing the full runtime config with defaults
    the author never wrote.
    """

    root = Path(task_dir)
    config_path = root / "task.toml"
    instruction_path = root / "instruction.md"
    imported = import_task_config_toml(config_path.read_text(), source="legacy")
    declared = dict(imported.declared)
    declared.pop("version", None)
    declared.pop("schema_version", None)
    frontmatter_data: dict[str, Any] = {
        "schema_version": imported.config.schema_version,
        **declared,
    }
    if imported.report.extra:
        frontmatter_data["benchflow"] = {
            "compat": {
                "source": imported.report.source,
                "extra_paths": list(imported.report.extra_paths),
                "extra": imported.report.extra,
            }
        }
    return render_task_md(frontmatter_data, instruction_path.read_text())


def render_normalized_task_md(text: str, *, path: str | Path | None = None) -> str:
    """Render a human-authored ``task.md`` as a canonical normalized document."""

    frontmatter, body = _split_frontmatter(text)
    doc_path = Path(path) if path is not None else None
    normalized = normalize_task_document_frontmatter(
        frontmatter,
        task_dir=doc_path.parent if doc_path is not None else None,
    )
    _config_from_frontmatter(normalized)
    _parse_roles(normalized)
    rendered_frontmatter = yaml.safe_dump(normalized, sort_keys=False)
    rendered_body = body.strip()
    suffix = f"\n\n{rendered_body}\n" if rendered_body else "\n"
    return f"---\n{rendered_frontmatter}---{suffix}"


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    normalized = text.replace("\r\n", "\n")
    lines = normalized.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise TaskDocumentParseError("task.md must start with YAML frontmatter")

    closing_index: int | None = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing_index = index
            break
    if closing_index is None:
        raise TaskDocumentParseError("task.md frontmatter is missing closing ---")

    frontmatter_text = "".join(lines[1:closing_index])
    body = "".join(lines[closing_index + 1 :]).lstrip("\n")
    try:
        loaded = yaml.safe_load(frontmatter_text) if frontmatter_text.strip() else {}
    except yaml.YAMLError as exc:
        raise TaskDocumentParseError(
            f"task.md frontmatter is not valid YAML: {exc}"
        ) from exc
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise TaskDocumentParseError("task.md frontmatter must be a mapping")
    return loaded, body


def _extract_prompt_sections(body: str) -> _PromptSections:
    matches = list(_SECTION_RE.finditer(body))
    if not matches:
        return _PromptSections(
            instruction=body.strip(),
            role_prompts={},
            scene_prompts={},
            user_persona=None,
        )

    preamble = body[: matches[0].start()].strip()
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        key = _normalize_section_key(match.group(1).strip())
        if key in sections:
            raise TaskDocumentParseError(f"task.md has duplicate section ## {key}")
        next_start = (
            matches[index + 1].start() if index + 1 < len(matches) else len(body)
        )
        sections[key] = body[match.end() : next_start].strip()

    instruction = _unescape_reserved_section_headings(
        sections.get("prompt", preamble).strip()
    )
    role_prompts = {
        key.removeprefix("role:"): _unescape_reserved_section_headings(value)
        for key, value in sections.items()
        if key.startswith("role:")
    }
    scene_prompts = {
        key.removeprefix("scene:"): _unescape_reserved_section_headings(value)
        for key, value in sections.items()
        if key.startswith("scene:")
    }
    user_persona = (
        _unescape_reserved_section_headings(sections["user-persona"])
        if "user-persona" in sections
        else None
    )
    return _PromptSections(
        instruction=instruction,
        role_prompts=role_prompts,
        scene_prompts=scene_prompts,
        user_persona=user_persona,
    )


def _load_prompt_sidecars(task_dir: Path) -> _PromptSections:
    """Load native sidecar prompt files from ``<task_dir>/prompts/``.

    Roles, scenes, and the simulated-user persona are authored as their own
    free-form markdown files — ``role.<name>.md``, ``scene.<name>.md``, and
    ``user-persona.md`` — so the ``task.md`` body stays one clean prompt with no
    reserved-heading ceremony. Each file's whole body is the prompt text.
    """
    prompts_dir = task_dir / "prompts"
    role_prompts: dict[str, str] = {}
    scene_prompts: dict[str, str] = {}
    user_persona: str | None = None
    if not prompts_dir.is_dir():
        return _PromptSections("", role_prompts, scene_prompts, user_persona)
    for prompt_file in sorted(prompts_dir.glob("*.md")):
        stem = prompt_file.name[: -len(".md")]
        text = prompt_file.read_text().strip()
        if stem == "user-persona":
            user_persona = text
        elif stem.startswith("role."):
            role_prompts[stem[len("role.") :]] = text
        elif stem.startswith("scene."):
            scene_prompts[stem[len("scene.") :]] = text
    return _PromptSections(
        instruction="",
        role_prompts=role_prompts,
        scene_prompts=scene_prompts,
        user_persona=user_persona,
    )


def _normalize_section_key(raw_key: str) -> str:
    prefix, separator, suffix = raw_key.partition(":")
    if separator:
        return f"{prefix.lower()}:{suffix}"
    return prefix.lower()


def _escape_reserved_section_headings(text: str) -> str:
    """Escape native task.md section headings embedded in legacy prompts."""

    return _SECTION_RE.sub(lambda match: "\\" + match.group(0), text)


def _unescape_reserved_section_headings(text: str) -> str:
    """Restore escaped native section headings in parsed section content."""

    return re.sub(
        r"^\\(##\s+(?:prompt|role:[A-Za-z0-9_.-]+|scene:[A-Za-z0-9_.-]+|user-persona)\s*)$",
        r"\1",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )


def _config_from_frontmatter(frontmatter: dict[str, Any]) -> TaskConfig:
    config_data = {
        key: value
        for key, value in frontmatter.items()
        if key not in _DOCUMENT_ONLY_FRONTMATTER_KEYS
    }
    return TaskConfig.model_validate(config_data)


def _parse_roles(frontmatter: dict[str, Any]) -> dict[str, Role]:
    agents = _mapping(frontmatter.get("agents"), "agents", default={})
    raw_roles = _mapping(agents.get("roles"), "agents.roles", default={})
    roles: dict[str, Role] = {}
    for name, raw_role in raw_roles.items():
        if not isinstance(name, str):
            raise TaskDocumentParseError("agents.roles keys must be role names")
        role_data = _mapping(raw_role, f"agents.roles.{name}", default={})
        agent = role_data.get("agent")
        if not isinstance(agent, str) or not agent:
            raise TaskDocumentParseError(f"agents.roles.{name}.agent is required")
        roles[name] = Role(
            name=name,
            agent=agent,
            model=_optional_str(role_data.get("model")),
            reasoning_effort=_optional_str(role_data.get("reasoning_effort")),
            env=_string_dict(role_data.get("env")),
            timeout_sec=_optional_int(role_data.get("timeout_sec")),
            idle_timeout_sec=_optional_int(role_data.get("idle_timeout_sec")),
            skills_dir=_optional_str(role_data.get("skills_dir")),
            capabilities=_string_list(role_data.get("capabilities")),
        )
    return roles


def _parse_scenes(
    frontmatter: dict[str, Any],
    *,
    roles: dict[str, Role],
    instruction: str,
    role_prompts: dict[str, str],
    scene_prompts: dict[str, str],
) -> list[Scene]:
    raw_scenes = frontmatter.get("scenes")
    if raw_scenes is None:
        return []
    if not isinstance(raw_scenes, list):
        raise TaskDocumentParseError("scenes must be a list")

    scenes: list[Scene] = []
    for index, raw_scene in enumerate(raw_scenes):
        scene_data = _mapping(raw_scene, f"scenes[{index}]", default={})
        name = _optional_str(scene_data.get("name")) or f"scene-{index}"
        scene_role_names = _scene_role_names(scene_data, roles)
        scene_roles = [
            _lookup_role(roles, role_name, f"scenes[{index}].roles")
            for role_name in scene_role_names
        ]
        turns = _parse_turns(
            scene_data.get("turns"),
            scene_name=name,
            roles=roles,
            scene_role_names=scene_role_names,
            role_prompts=role_prompts,
            scene_prompts=scene_prompts,
        )
        if not turns and scene_roles:
            turns = [
                Turn(
                    role=role.name,
                    prompt=scene_prompts.get(name)
                    or role_prompts.get(role.name)
                    or instruction,
                )
                for role in scene_roles
            ]
        scenes.append(
            Scene(
                name=name,
                roles=scene_roles,
                turns=turns,
                skills_dir=_optional_str(scene_data.get("skills_dir")),
            )
        )
    return scenes


def _scene_role_names(scene_data: dict[str, Any], roles: dict[str, Role]) -> list[str]:
    raw_names = scene_data.get("roles")
    if raw_names is None:
        raw_turns = scene_data.get("turns")
        if isinstance(raw_turns, list):
            names: list[str] = []
            for raw_turn in raw_turns:
                role_name = (
                    raw_turn
                    if isinstance(raw_turn, str)
                    else _mapping(raw_turn, "turn", default={}).get("role")
                )
                if isinstance(role_name, str) and role_name not in names:
                    names.append(role_name)
            if names:
                return names
        return list(roles)
    if not isinstance(raw_names, list) or not all(
        isinstance(name, str) for name in raw_names
    ):
        raise TaskDocumentParseError("scene roles must be a list of role names")
    return [name for name in raw_names if isinstance(name, str)]


def _parse_turns(
    raw_turns: Any,
    *,
    scene_name: str,
    roles: dict[str, Role],
    scene_role_names: list[str],
    role_prompts: dict[str, str],
    scene_prompts: dict[str, str],
) -> list[Turn]:
    if raw_turns is None:
        return []
    if not isinstance(raw_turns, list):
        raise TaskDocumentParseError("scene turns must be a list")

    turns: list[Turn] = []
    for index, raw_turn in enumerate(raw_turns):
        if isinstance(raw_turn, str):
            role_name = raw_turn
            prompt = None
        else:
            turn_data = _mapping(raw_turn, f"turns[{index}]", default={})
            role_name = turn_data.get("role")
            if not isinstance(role_name, str):
                raise TaskDocumentParseError(f"turns[{index}].role is required")
            prompt = _optional_str(turn_data.get("prompt"))
        _lookup_role(roles, role_name, f"turns[{index}].role")
        if role_name not in scene_role_names:
            raise TaskDocumentParseError(
                f"turns[{index}] references role {role_name!r} outside the scene"
            )
        turns.append(
            Turn(
                role=role_name,
                prompt=prompt
                or scene_prompts.get(scene_name)
                or role_prompts.get(role_name),
            )
        )
    return turns


def _lookup_role(roles: dict[str, Role], name: str, source: str) -> Role:
    role = roles.get(name)
    if role is None:
        raise TaskDocumentParseError(f"{source} references unknown role {name!r}")
    return role


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TaskDocumentParseError(
            f"Expected string value, got {type(value).__name__}"
        )
    return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TaskDocumentParseError("Expected integer value, got bool")
    if not isinstance(value, int):
        raise TaskDocumentParseError(
            f"Expected integer value, got {type(value).__name__}"
        )
    return value


def _string_dict(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TaskDocumentParseError("Expected mapping of strings")
    return {str(key): str(item) for key, item in value.items()}


def _string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TaskDocumentParseError("Expected list of strings")
    return [item for item in value if isinstance(item, str)]
