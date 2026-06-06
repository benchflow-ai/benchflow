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


def load_verifier_document(
    task_dir: Path | str,
    benchflow: dict[str, Any],
) -> VerifierDocument | None:
    """Load the verifier document referenced by ``benchflow.verifier.spec``."""
    if not isinstance(benchflow, dict):
        return None
    benchflow_verifier = benchflow.get("verifier")
    if not isinstance(benchflow_verifier, dict):
        return None
    spec_issues = verifier_document_issues(
        Path(task_dir).resolve(),
        benchflow_verifier=benchflow_verifier,
    )
    if spec_issues:
        raise ValueError("; ".join(spec_issues))
    spec = benchflow_verifier.get("spec")
    if not isinstance(spec, str) or not spec.strip():
        return None
    spec_path = resolve_verifier_spec_path(Path(task_dir).resolve(), spec.strip())
    try:
        return VerifierDocument.from_path(spec_path)
    except VerifierDocumentParseError as exc:
        raise ValueError(f"{spec} parse error: {exc}") from exc


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

    for strategy_name, strategy in document.strategies.items():
        try:
            strategy_type = verifier_strategy_type(strategy)
        except VerifierDocumentParseError as exc:
            issues.append(f"{spec} strategy {strategy_name!r} error: {exc}")
            continue
        if strategy_type == "agent-judge":
            _append_agent_judge_role_issues(
                issues,
                spec=spec,
                strategy_name=strategy_name,
                strategy=strategy,
                document=document,
                verifier_dir=verifier_dir,
            )
            continue
        if strategy_type != "reward-kit":
            continue
        raw_root = strategy.get("root")
        if not isinstance(raw_root, str) or not raw_root.strip():
            issues.append(
                f"{spec} reward-kit strategy {strategy_name!r} is missing root"
            )
            continue
        root_path = (verifier_dir / raw_root.strip()).resolve()
        if not root_path.is_dir():
            issues.append(
                f"{spec} reward-kit strategy {strategy_name!r} references "
                f"missing root directory: {raw_root.strip()}"
            )

    return issues


def _is_agent_judge_role_file_path(role: str) -> bool:
    return "/" in role or role.endswith(".md")


def _append_agent_judge_role_issues(
    issues: list[str],
    *,
    spec: str,
    strategy_name: str,
    strategy: dict[str, Any],
    document: VerifierDocument,
    verifier_dir: Path,
) -> None:
    raw_role = strategy.get("role")
    if not isinstance(raw_role, str) or not raw_role.strip():
        issues.append(
            f"{spec} agent-judge strategy {strategy_name!r} is missing role"
        )
        return

    role = raw_role.strip()
    if _is_agent_judge_role_file_path(role):
        role_path = (verifier_dir / role).resolve()
        if not role_path.is_file():
            issues.append(
                f"{spec} agent-judge strategy {strategy_name!r} references "
                f"missing role file: {role}"
            )
        return

    if role not in document.role_prompts:
        issues.append(
            f"{spec} agent-judge strategy {strategy_name!r} references "
            f"unknown role prompt ## role:{role}"
        )


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


_SCRIPT_STRATEGY_TYPES = frozenset({"script", "deterministic"})


def resolve_default_strategy(
    document: VerifierDocument,
) -> tuple[str, dict[str, Any]]:
    """Return the selected default strategy name and config."""

    strategy_name = document.default_strategy
    if not strategy_name:
        raise ValueError("verifier document has no default_strategy")
    strategy = document.strategies.get(strategy_name)
    if strategy is None:
        raise ValueError(
            f"verifier default_strategy {strategy_name!r} is not declared in "
            "verifier.strategies"
        )
    return strategy_name, strategy


def verifier_strategy_type(strategy: dict[str, Any]) -> str | None:
    """Return the declared strategy ``type`` when present."""

    raw_type = strategy.get("type")
    if raw_type is None:
        return None
    if not isinstance(raw_type, str):
        raise VerifierDocumentParseError(
            f"verifier strategy type must be a string, got {type(raw_type).__name__}"
        )
    return raw_type


def is_executable_script_strategy(strategy: dict[str, Any]) -> bool:
    """Return whether the runtime can execute the strategy via ``test.sh``."""

    return verifier_strategy_type(strategy) in _SCRIPT_STRATEGY_TYPES


def resolve_reward_kit_criteria_path(
    strategy: dict[str, Any],
    verifier_dir: Path,
) -> Path | None:
    """Return the reward-kit criteria path when declared and present."""

    raw_criteria = strategy.get("criteria")
    if not isinstance(raw_criteria, str) or not raw_criteria.strip():
        return None
    criteria_path = (verifier_dir / raw_criteria.strip()).resolve()
    return criteria_path if criteria_path.is_file() else None


def is_executable_reward_kit_strategy(
    strategy: dict[str, Any],
    verifier_dir: Path,
) -> bool:
    """Return whether a reward-kit strategy can run via the script verifier."""

    if verifier_strategy_type(strategy) != "reward-kit":
        return False
    if resolve_reward_kit_criteria_path(strategy, verifier_dir) is not None:
        return True
    raw_root = strategy.get("root")
    if isinstance(raw_root, str) and raw_root.strip():
        root_path = (verifier_dir / raw_root.strip()).resolve()
        return root_path.is_dir() and (root_path / "test.sh").is_file()
    return False


def resolve_agent_judge_role_prompt(
    strategy: dict[str, Any],
    document: VerifierDocument,
    verifier_dir: Path,
) -> str | None:
    """Resolve the verifier-scoped judge role prompt for an agent-judge strategy."""

    raw_role = strategy.get("role")
    if not isinstance(raw_role, str) or not raw_role.strip():
        return None
    role = raw_role.strip()
    if _is_agent_judge_role_file_path(role):
        role_path = (verifier_dir / role).resolve()
        if not role_path.is_file():
            return None
        return role_path.read_text(encoding="utf-8").strip() or None
    prompt = document.role_prompts.get(role)
    return prompt.strip() if isinstance(prompt, str) and prompt.strip() else None


def resolve_structured_rubric_path(
    document: VerifierDocument,
    verifier_dir: Path,
) -> Path | None:
    """Return the structured rubric path declared by a verifier document."""

    structured = document.rubric_files.structured
    if isinstance(structured, str) and structured.strip():
        path = (verifier_dir / structured.strip()).resolve()
        if path.is_file():
            return path
    rewardkit = document.strategies.get("rewardkit")
    if isinstance(rewardkit, dict):
        criteria = resolve_reward_kit_criteria_path(rewardkit, verifier_dir)
        if criteria is not None:
            return criteria
    return None


def is_executable_agent_judge_strategy(
    strategy: dict[str, Any],
    document: VerifierDocument,
    verifier_dir: Path,
) -> bool:
    """Return whether an agent-judge strategy has role + rubric inputs to run."""

    if verifier_strategy_type(strategy) != "agent-judge":
        return False
    if resolve_agent_judge_role_prompt(strategy, document, verifier_dir) is None:
        return False
    return resolve_structured_rubric_path(document, verifier_dir) is not None


__all__ = [
    "VERIFIER_DOCUMENT_FILENAME",
    "VerifierDocument",
    "VerifierDocumentParseError",
    "VerifierOutputs",
    "VerifierRubricFiles",
    "is_executable_agent_judge_strategy",
    "is_executable_reward_kit_strategy",
    "is_executable_script_strategy",
    "load_verifier_document",
    "resolve_agent_judge_role_prompt",
    "resolve_reward_kit_criteria_path",
    "resolve_structured_rubric_path",
    "resolve_default_strategy",
    "resolve_verifier_spec_path",
    "verifier_document_issues",
    "verifier_strategy_type",
]
