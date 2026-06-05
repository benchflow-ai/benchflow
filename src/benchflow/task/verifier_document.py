"""Unified ``verifier/verifier.md`` authoring document support.

The runtime still consumes script verifiers and reward files directly. This
module owns the document-shaped verifier package layer so reward strategies,
rubrics, judge roles, and output contracts can be parsed without overloading
``task/config.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

VERIFIER_DOCUMENT_FILENAME = "verifier.md"

_ROLE_SECTION_RE = re.compile(
    r"^##\s+role:([A-Za-z0-9_.-]+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


class VerifierDocumentParseError(ValueError):
    """Raised when a ``verifier/verifier.md`` document cannot be parsed."""


@dataclass(frozen=True)
class VerifierRubricFiles:
    """Human and structured rubric file references declared in verifier metadata."""

    human: str | None = None
    structured: str | None = None


@dataclass(frozen=True)
class VerifierOutputs:
    """Declared verifier reward artifact paths and aggregate policy."""

    reward_text: str | None = None
    reward_json: str | None = None
    reward_details: str | None = None
    aggregate_policy: dict[str, Any] | None = None


@dataclass(frozen=True)
class VerifierDocument:
    """Parsed ``verifier/verifier.md`` document."""

    frontmatter: dict[str, Any]
    body: str
    document_version: str | None
    name: str | None
    default_strategy: str | None
    strategies: dict[str, dict[str, Any]]
    rubric: dict[str, Any]
    rubric_files: VerifierRubricFiles
    outputs: VerifierOutputs
    role_prompts: dict[str, str]
    path: Path | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> VerifierDocument:
        doc_path = Path(path)
        return cls.from_text(doc_path.read_text(), path=doc_path)

    @classmethod
    def from_text(cls, text: str, *, path: str | Path | None = None) -> VerifierDocument:
        frontmatter, body = _split_frontmatter(text)
        verifier = _mapping(frontmatter.get("verifier"), "verifier", default={})
        strategies = _parse_strategies(verifier.get("strategies"))
        rubric = _mapping(verifier.get("rubric"), "verifier.rubric", default={})
        rubric_files = _parse_rubric_files(rubric.get("files"), strategies)
        outputs = _parse_outputs(verifier.get("outputs"))
        role_prompts = _extract_role_prompts(body)
        return cls(
            frontmatter=frontmatter,
            body=body,
            document_version=_optional_str(frontmatter.get("document_version")),
            name=_optional_str(verifier.get("name")),
            default_strategy=_optional_str(verifier.get("default_strategy")),
            strategies=strategies,
            rubric=rubric,
            rubric_files=rubric_files,
            outputs=outputs,
            role_prompts=role_prompts,
            path=Path(path) if path is not None else None,
        )


def resolve_verifier_spec_path(task_dir: Path, spec: str) -> Path:
    """Resolve a ``benchflow.verifier.spec`` path relative to the task directory."""

    spec_path = Path(spec)
    if spec_path.is_absolute():
        return spec_path
    return (task_dir / spec_path).resolve()


def verifier_document_issues(
    task_dir: Path,
    *,
    benchflow_verifier: dict[str, Any] | None = None,
) -> list[str]:
    """Validate verifier document references for a task directory."""

    issues: list[str] = []
    if benchflow_verifier is None:
        return issues

    spec = benchflow_verifier.get("spec")
    if not isinstance(spec, str) or not spec.strip():
        return issues

    spec_path = resolve_verifier_spec_path(task_dir, spec.strip())
    if not spec_path.exists():
        issues.append(f"benchflow.verifier.spec references missing file: {spec}")
        return issues

    try:
        document = VerifierDocument.from_path(spec_path)
    except VerifierDocumentParseError as exc:
        issues.append(f"{spec} parse error: {exc}")
        return issues
    except OSError as exc:
        issues.append(f"{spec} read error: {exc}")
        return issues

    default_strategy = document.default_strategy
    if default_strategy and default_strategy not in document.strategies:
        issues.append(
            f"{spec} default_strategy {default_strategy!r} is not declared in "
            "verifier.strategies"
        )

    verifier_dir = spec_path.parent
    for label, path in _declared_rubric_file_paths(
        task_dir=task_dir,
        verifier_dir=verifier_dir,
        document=document,
        benchflow_verifier=benchflow_verifier,
    ):
        if not path.exists():
            issues.append(
                f"benchflow.verifier declares missing {label} rubric file: {path}"
            )

    return issues


def _declared_rubric_file_paths(
    *,
    task_dir: Path,
    verifier_dir: Path,
    document: VerifierDocument,
    benchflow_verifier: dict[str, Any],
) -> list[tuple[str, Path]]:
    refs: list[tuple[str, Path]] = []
    seen: set[Path] = set()

    def add(label: str, value: Any, *, base_dir: Path) -> None:
        if not isinstance(value, str) or not value.strip():
            return
        path = (base_dir / value.strip()).resolve()
        if path in seen:
            return
        seen.add(path)
        refs.append((label, path))

    add("human", document.rubric_files.human, base_dir=verifier_dir)
    add("structured", document.rubric_files.structured, base_dir=verifier_dir)
    add("human", benchflow_verifier.get("rubric"), base_dir=task_dir)
    add("structured", benchflow_verifier.get("structured_rubric"), base_dir=task_dir)

    rewardkit = document.strategies.get("rewardkit")
    if isinstance(rewardkit, dict):
        add("criteria", rewardkit.get("criteria"), base_dir=verifier_dir)

    return refs


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    normalized = text.replace("\r\n", "\n")
    lines = normalized.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise VerifierDocumentParseError(
            "verifier.md must start with YAML frontmatter"
        )

    closing_index: int | None = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing_index = index
            break
    if closing_index is None:
        raise VerifierDocumentParseError(
            "verifier.md frontmatter is missing closing ---"
        )

    frontmatter_text = "".join(lines[1:closing_index])
    body = "".join(lines[closing_index + 1 :]).lstrip("\n")
    loaded = yaml.safe_load(frontmatter_text) if frontmatter_text.strip() else {}
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise VerifierDocumentParseError("verifier.md frontmatter must be a mapping")
    return loaded, body


def _extract_role_prompts(body: str) -> dict[str, str]:
    matches = list(_ROLE_SECTION_RE.finditer(body))
    if not matches:
        return {}

    role_prompts: dict[str, str] = {}
    for index, match in enumerate(matches):
        role_name = match.group(1)
        if role_name in role_prompts:
            raise VerifierDocumentParseError(
                f"verifier.md has duplicate section ## role:{role_name}"
            )
        next_start = (
            matches[index + 1].start() if index + 1 < len(matches) else len(body)
        )
        role_prompts[role_name] = body[match.end() : next_start].strip()
    return role_prompts


def _parse_strategies(value: Any) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise VerifierDocumentParseError("verifier.strategies must be a mapping")
    strategies: dict[str, dict[str, Any]] = {}
    for name, raw_strategy in value.items():
        if not isinstance(name, str):
            raise VerifierDocumentParseError(
                "verifier.strategies keys must be strategy names"
            )
        strategies[name] = _mapping(
            raw_strategy,
            f"verifier.strategies.{name}",
            default={},
        )
    return strategies


def _parse_rubric_files(
    raw_files: Any,
    strategies: dict[str, dict[str, Any]],
) -> VerifierRubricFiles:
    human: str | None = None
    structured: str | None = None
    if raw_files is not None:
        files = _mapping(raw_files, "verifier.rubric.files", default={})
        human = _optional_str(files.get("human"))
        structured = _optional_str(files.get("structured"))

    rewardkit = strategies.get("rewardkit")
    if structured is None and isinstance(rewardkit, dict):
        structured = _optional_str(rewardkit.get("criteria"))

    return VerifierRubricFiles(human=human, structured=structured)


def _parse_outputs(value: Any) -> VerifierOutputs:
    if value is None:
        return VerifierOutputs()
    outputs = _mapping(value, "verifier.outputs", default={})
    aggregate_policy = outputs.get("aggregate_policy")
    if aggregate_policy is not None:
        aggregate_policy = _mapping(
            aggregate_policy,
            "verifier.outputs.aggregate_policy",
            default={},
        )
    return VerifierOutputs(
        reward_text=_optional_str(outputs.get("reward_text")),
        reward_json=_optional_str(outputs.get("reward_json")),
        reward_details=_optional_str(outputs.get("details_json")),
        aggregate_policy=aggregate_policy,
    )


def _mapping(
    value: Any, source: str, *, default: dict[str, Any] | None = None
) -> dict[str, Any]:
    if value is None:
        return {} if default is None else dict(default)
    if not isinstance(value, dict):
        raise VerifierDocumentParseError(f"{source} must be a mapping")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise VerifierDocumentParseError(
            f"Expected string value, got {type(value).__name__}"
        )
    return value


__all__ = [
    "VERIFIER_DOCUMENT_FILENAME",
    "VerifierDocument",
    "VerifierDocumentParseError",
    "VerifierOutputs",
    "VerifierRubricFiles",
    "resolve_verifier_spec_path",
    "verifier_document_issues",
]
