"""Unified ``task.md`` authoring document support.

The runtime still consumes the stable ``TaskConfig`` and instruction string.
This module owns the document-shaped authoring layer so ``task/config.py`` does
not become the home for prompt, role, scene, and simulated-user syntax.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from benchflow._types import Role, Scene, Turn
from benchflow.task.config import TaskConfig
from benchflow.task.prompt_composition import (
    PromptCompositionSettings,
    compose_task_prompt,
    prompt_composition_settings,
)

TASK_DOCUMENT_FILENAME = "task.md"

_DOCUMENT_ONLY_FRONTMATTER_KEYS = {"agents", "benchflow", "scenes", "user"}
_SECTION_RE = re.compile(
    r"^##\s+(prompt|role:[A-Za-z0-9_.-]+|scene:[A-Za-z0-9_.-]+|user-persona)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


class TaskDocumentParseError(ValueError):
    """Raised when a ``task.md`` document cannot be parsed."""


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
        prompt_sections = _extract_prompt_sections(body)
        config = _config_from_frontmatter(frontmatter)
        roles = _parse_roles(frontmatter)
        user = _mapping(frontmatter.get("user"), "user", default={})
        benchflow = _mapping(frontmatter.get("benchflow"), "benchflow", default={})
        prompt_settings = _prompt_composition_settings(benchflow)
        scenes = _parse_scenes(
            frontmatter,
            roles=roles,
            instruction=prompt_sections.instruction,
            role_prompts=prompt_sections.role_prompts,
            scene_prompts=prompt_sections.scene_prompts,
            prompt_settings=prompt_settings,
        )
        return cls(
            frontmatter=frontmatter,
            body=body,
            instruction=prompt_sections.instruction,
            config=config,
            roles=roles,
            scenes=scenes,
            role_prompts=prompt_sections.role_prompts,
            scene_prompts=prompt_sections.scene_prompts,
            user=user,
            user_persona=prompt_sections.user_persona,
            benchflow=benchflow,
            path=Path(path) if path is not None else None,
        )


@dataclass(frozen=True)
class _PromptSections:
    instruction: str
    role_prompts: dict[str, str]
    scene_prompts: dict[str, str]
    user_persona: str | None


def render_task_md_from_legacy(task_dir: str | Path) -> str:
    """Render a legacy ``task.toml`` + ``instruction.md`` task as ``task.md``."""

    root = Path(task_dir)
    config_path = root / "task.toml"
    instruction_path = root / "instruction.md"
    raw_config = tomllib.loads(config_path.read_text())
    if "solution" in raw_config and "oracle" not in raw_config:
        raw_config["oracle"] = raw_config.pop("solution")
    frontmatter = yaml.safe_dump(raw_config, sort_keys=False)
    instruction = instruction_path.read_text().strip()
    return f"---\n{frontmatter}---\n\n## prompt\n\n{instruction}\n"


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
    loaded = yaml.safe_load(frontmatter_text) if frontmatter_text.strip() else {}
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

    instruction = sections.get("prompt", preamble).strip()
    role_prompts = {
        key.removeprefix("role:"): value
        for key, value in sections.items()
        if key.startswith("role:")
    }
    scene_prompts = {
        key.removeprefix("scene:"): value
        for key, value in sections.items()
        if key.startswith("scene:")
    }
    user_persona = sections.get("user-persona")
    return _PromptSections(
        instruction=instruction,
        role_prompts=role_prompts,
        scene_prompts=scene_prompts,
        user_persona=user_persona,
    )


def _normalize_section_key(raw_key: str) -> str:
    prefix, separator, suffix = raw_key.partition(":")
    if separator:
        return f"{prefix.lower()}:{suffix}"
    return prefix.lower()


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
    prompt_settings: PromptCompositionSettings,
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
            base_prompt=instruction,
            roles=roles,
            scene_role_names=scene_role_names,
            role_prompts=role_prompts,
            scene_prompts=scene_prompts,
            prompt_settings=prompt_settings,
        )
        if not turns and scene_roles:
            turns = [
                Turn(
                    role=role.name,
                    prompt=_compose_scene_turn_prompt(
                        base_prompt=instruction,
                        role_name=role.name,
                        scene_name=name,
                        role_prompts=role_prompts,
                        scene_prompts=scene_prompts,
                        turn_prompt=None,
                        explicit_turn=False,
                        prompt_settings=prompt_settings,
                    ),
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


def _prompt_composition_settings(benchflow: dict[str, Any]) -> PromptCompositionSettings:
    try:
        return prompt_composition_settings(benchflow)
    except ValueError as exc:
        raise TaskDocumentParseError(str(exc)) from exc


def _compose_scene_turn_prompt(
    *,
    base_prompt: str,
    role_name: str,
    scene_name: str,
    role_prompts: dict[str, str],
    scene_prompts: dict[str, str],
    turn_prompt: str | None,
    explicit_turn: bool,
    prompt_settings: PromptCompositionSettings,
) -> str:
    return compose_task_prompt(
        base_prompt,
        role_prompts.get(role_name),
        scene_prompts.get(scene_name),
        turn_prompt,
        composition=prompt_settings.composition,
        order=prompt_settings.order,
        explicit_turn=explicit_turn,
    )


def _parse_turns(
    raw_turns: Any,
    *,
    scene_name: str,
    base_prompt: str,
    roles: dict[str, Role],
    scene_role_names: list[str],
    role_prompts: dict[str, str],
    scene_prompts: dict[str, str],
    prompt_settings: PromptCompositionSettings,
) -> list[Turn]:
    if raw_turns is None:
        return []
    if not isinstance(raw_turns, list):
        raise TaskDocumentParseError("scene turns must be a list")

    turns: list[Turn] = []
    for index, raw_turn in enumerate(raw_turns):
        if isinstance(raw_turn, str):
            role_name = raw_turn
            turn_prompt = None
            explicit_turn = False
        else:
            turn_data = _mapping(raw_turn, f"turns[{index}]", default={})
            role_name = turn_data.get("role")
            if not isinstance(role_name, str):
                raise TaskDocumentParseError(f"turns[{index}].role is required")
            explicit_turn = "prompt" in turn_data
            turn_prompt = (
                _optional_str(turn_data.get("prompt")) if explicit_turn else None
            )
        _lookup_role(roles, role_name, f"turns[{index}].role")
        if role_name not in scene_role_names:
            raise TaskDocumentParseError(
                f"turns[{index}] references role {role_name!r} outside the scene"
            )
        turns.append(
            Turn(
                role=role_name,
                prompt=_compose_scene_turn_prompt(
                    base_prompt=base_prompt,
                    role_name=role_name,
                    scene_name=scene_name,
                    role_prompts=role_prompts,
                    scene_prompts=scene_prompts,
                    turn_prompt=turn_prompt,
                    explicit_turn=explicit_turn,
                    prompt_settings=prompt_settings,
                ),
            )
        )
    return turns


def _lookup_role(roles: dict[str, Role], name: str, source: str) -> Role:
    role = roles.get(name)
    if role is None:
        raise TaskDocumentParseError(f"{source} references unknown role {name!r}")
    return role


def _mapping(
    value: Any, source: str, *, default: dict[str, Any] | None = None
) -> dict[str, Any]:
    if value is None:
        return {} if default is None else dict(default)
    if not isinstance(value, dict):
        raise TaskDocumentParseError(f"{source} must be a mapping")
    return value


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


__all__ = [
    "TASK_DOCUMENT_FILENAME",
    "TaskDocument",
    "TaskDocumentParseError",
    "render_task_md_from_legacy",
]
