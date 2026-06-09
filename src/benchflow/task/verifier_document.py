"""Verifier package document support.

``verifier/verifier.md`` is the evaluation-side peer of ``task.md``. It
describes verifier strategies, rubric composition, verifier-scoped judge roles,
and reward artifact contracts without replacing the existing ``test.sh`` script
runner.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import yaml

VERIFIER_DOCUMENT_FILENAME = "verifier.md"

VerifierStrategyType = Literal[
    "script",
    "reward-kit",
    "llm-judge",
    "agent-judge",
    "ors-episode",
]

_ROLE_SECTION_RE = re.compile(
    r"^##\s+role:([A-Za-z0-9_.-]+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_KNOWN_STRATEGY_TYPES = {
    "script",
    "reward-kit",
    "llm-judge",
    "agent-judge",
    "ors-episode",
}


class VerifierDocumentParseError(ValueError):
    """Raised when ``verifier/verifier.md`` cannot be parsed."""


@dataclass(frozen=True)
class VerifierStrategy:
    """One named verifier strategy from ``verifier.verifier.strategies``."""

    name: str
    type: VerifierStrategyType
    config: dict[str, Any]

    @property
    def command(self) -> str | None:
        value = self.config.get("command")
        return value if isinstance(value, str) else None

    @property
    def rubric_path(self) -> str | None:
        value = self.config.get("rubric")
        return value if isinstance(value, str) else None

    @property
    def root_path(self) -> str | None:
        value = self.config.get("root")
        return value if isinstance(value, str) else None

    @property
    def criteria_path(self) -> str | None:
        value = self.config.get("criteria")
        return value if isinstance(value, str) else None

    @property
    def entrypoint(self) -> str | None:
        value = self.config.get("entrypoint")
        return value if isinstance(value, str) else None

    @property
    def role(self) -> str | None:
        value = self.config.get("role")
        return value if isinstance(value, str) else None

    @property
    def model(self) -> str | None:
        value = self.config.get("model")
        return value if isinstance(value, str) else None

    @property
    def input_dir(self) -> str | None:
        value = self.config.get("input_dir")
        return value if isinstance(value, str) else None

    @property
    def context(self) -> str | None:
        value = self.config.get("context")
        return value if isinstance(value, str) else None

    @property
    def context_file(self) -> str | None:
        value = self.config.get("context_file")
        return value if isinstance(value, str) else None

    @property
    def inputs(self) -> tuple[str, ...]:
        value = self.config.get("inputs")
        if not isinstance(value, list):
            return ()
        return tuple(str(item) for item in value)


@dataclass(frozen=True)
class VerifierOutputContract:
    """Declared verifier output files."""

    reward_text: str = "/logs/verifier/reward.txt"
    reward_json: str | None = "/logs/verifier/reward.json"
    details_json: str | None = None
    aggregate_policy: dict[str, Any] = field(default_factory=dict)
    declared_reward_text: bool = False
    declared_reward_json: bool = False
    declared_details_json: bool = False


@dataclass(frozen=True)
class VerifierDocument:
    """Parsed verifier package document."""

    frontmatter: dict[str, Any]
    body: str
    document_version: str | None
    name: str
    default_strategy: str
    strategies: dict[str, VerifierStrategy]
    rubric: dict[str, Any]
    outputs: VerifierOutputContract
    role_prompts: dict[str, str]
    path: Path | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> VerifierDocument:
        doc_path = Path(path)
        return cls.from_text(doc_path.read_text(), path=doc_path)

    @classmethod
    def from_verifier_dir(cls, verifier_dir: str | Path) -> VerifierDocument:
        return cls.from_path(Path(verifier_dir) / VERIFIER_DOCUMENT_FILENAME)

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        path: str | Path | None = None,
    ) -> VerifierDocument:
        frontmatter, body = _split_frontmatter(text)
        verifier = _mapping(frontmatter.get("verifier"), "verifier")
        role_prompts = _extract_role_prompts(body)
        strategies = _parse_strategies(
            _mapping(verifier.get("strategies"), "verifier.strategies"),
            role_prompts=role_prompts,
        )
        if not strategies:
            raise VerifierDocumentParseError(
                "verifier.strategies must contain at least one strategy"
            )

        default_strategy = _optional_str(verifier.get("default_strategy"))
        if default_strategy is None:
            default_strategy = next(iter(strategies))
        if default_strategy not in strategies:
            raise VerifierDocumentParseError(
                "verifier.default_strategy must name one of verifier.strategies"
            )

        name = _optional_str(verifier.get("name"))
        if name is None:
            name = Path(path).parent.name if path is not None else "verifier"

        return cls(
            frontmatter=frontmatter,
            body=body,
            document_version=_optional_str(frontmatter.get("document_version")),
            name=name,
            default_strategy=default_strategy,
            strategies=strategies,
            rubric=_mapping(verifier.get("rubric"), "verifier.rubric", default={}),
            outputs=_parse_outputs(
                _mapping(verifier.get("outputs"), "verifier.outputs", default={})
            ),
            role_prompts=role_prompts,
            path=Path(path) if path is not None else None,
        )

    @property
    def selected_strategy(self) -> VerifierStrategy:
        return self.strategies[self.default_strategy]


def load_verifier_document(verifier_dir: str | Path) -> VerifierDocument | None:
    """Load ``verifier/verifier.md`` when present."""

    path = Path(verifier_dir) / VERIFIER_DOCUMENT_FILENAME
    if not path.exists():
        return None
    return VerifierDocument.from_path(path)


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
        raise VerifierDocumentParseError(
            "verifier.md frontmatter must be a mapping"
        )
    return loaded, body


def _extract_role_prompts(body: str) -> dict[str, str]:
    matches = list(_ROLE_SECTION_RE.finditer(body))
    role_prompts: dict[str, str] = {}
    for index, match in enumerate(matches):
        name = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        role_prompts[name] = body[start:end].strip()
    return role_prompts


def _parse_strategies(
    raw_strategies: dict[str, Any],
    *,
    role_prompts: dict[str, str],
) -> dict[str, VerifierStrategy]:
    strategies: dict[str, VerifierStrategy] = {}
    for name, raw_strategy in raw_strategies.items():
        if not isinstance(name, str) or not name:
            raise VerifierDocumentParseError(
                "verifier.strategies keys must be non-empty strings"
            )
        config = _mapping(raw_strategy, f"verifier.strategies.{name}")
        raw_type = _required_str(config.get("type"), f"verifier.strategies.{name}.type")
        if raw_type not in _KNOWN_STRATEGY_TYPES:
            raise VerifierDocumentParseError(
                f"verifier.strategies.{name}.type is unsupported: {raw_type}"
            )
        _validate_strategy(name, raw_type, config, role_prompts=role_prompts)
        strategies[name] = VerifierStrategy(
            name=name,
            type=cast(VerifierStrategyType, raw_type),
            config=dict(config),
        )
    return strategies


def _validate_strategy(
    name: str,
    strategy_type: str,
    config: dict[str, Any],
    *,
    role_prompts: dict[str, str],
) -> None:
    prefix = f"verifier.strategies.{name}"
    if strategy_type == "script":
        _required_str(config.get("command"), f"{prefix}.command")
    elif strategy_type == "reward-kit":
        _safe_relative_path(_required_str(config.get("root"), f"{prefix}.root"), f"{prefix}.root")
        if "criteria" in config:
            _safe_relative_path(
                _required_str(config.get("criteria"), f"{prefix}.criteria"),
                f"{prefix}.criteria",
            )
        if "entrypoint" in config:
            _safe_relative_path(
                _required_str(config.get("entrypoint"), f"{prefix}.entrypoint"),
                f"{prefix}.entrypoint",
            )
    elif strategy_type == "llm-judge":
        _required_str(config.get("rubric"), f"{prefix}.rubric")
        if "model" in config:
            _required_str(config.get("model"), f"{prefix}.model")
        if "input_dir" in config:
            _required_str(config.get("input_dir"), f"{prefix}.input_dir")
        if "context" in config:
            _required_str(config.get("context"), f"{prefix}.context")
        if "context_file" in config:
            _safe_relative_path(
                _required_str(config.get("context_file"), f"{prefix}.context_file"),
                f"{prefix}.context_file",
            )
        if "context" in config and "context_file" in config:
            raise VerifierDocumentParseError(
                f"{prefix} must declare either context or context_file, not both"
            )
    elif strategy_type == "agent-judge":
        role = _required_str(config.get("role"), f"{prefix}.role")
        if role not in role_prompts:
            raise VerifierDocumentParseError(
                f"{prefix}.role references missing ## role:{role} section"
            )
        if "model" in config:
            _required_str(config.get("model"), f"{prefix}.model")
        isolation = _required_str(config.get("isolation"), f"{prefix}.isolation")
        if isolation != "verifier-only":
            raise VerifierDocumentParseError(
                f"{prefix}.isolation must be 'verifier-only'"
            )
        inputs = config.get("inputs")
        if not isinstance(inputs, list) or not inputs or not all(
            isinstance(item, str) and item for item in inputs
        ):
            raise VerifierDocumentParseError(
                f"{prefix}.inputs must be a non-empty list of strings"
            )
    elif strategy_type == "ors-episode":
        inputs = config.get("inputs")
        if not isinstance(inputs, list) or not inputs or not all(
            isinstance(item, str) and item for item in inputs
        ):
            raise VerifierDocumentParseError(
                f"{prefix}.inputs must be a non-empty list of strings"
            )
        if "format" in config:
            raw_format = _required_str(config.get("format"), f"{prefix}.format")
            if raw_format not in {"json", "jsonl", "auto"}:
                raise VerifierDocumentParseError(
                    f"{prefix}.format must be json, jsonl, or auto"
                )


def _parse_outputs(raw_outputs: dict[str, Any]) -> VerifierOutputContract:
    aggregate_policy = raw_outputs.get("aggregate_policy", {})
    if aggregate_policy is None:
        aggregate_policy = {}
    if not isinstance(aggregate_policy, dict):
        raise VerifierDocumentParseError(
            "verifier.outputs.aggregate_policy must be a mapping"
        )
    return VerifierOutputContract(
        reward_text=_optional_str(raw_outputs.get("reward_text"))
        or "/logs/verifier/reward.txt",
        reward_json=_optional_str(raw_outputs.get("reward_json"))
        or "/logs/verifier/reward.json",
        details_json=_optional_str(raw_outputs.get("details_json")),
        aggregate_policy=dict(aggregate_policy),
        declared_reward_text="reward_text" in raw_outputs,
        declared_reward_json="reward_json" in raw_outputs,
        declared_details_json="details_json" in raw_outputs,
    )


def _mapping(
    value: Any,
    path: str,
    *,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if value is None:
        if default is not None:
            return default
        raise VerifierDocumentParseError(f"{path} is required")
    if not isinstance(value, dict):
        raise VerifierDocumentParseError(f"{path} must be a mapping")
    return dict(value)


def _required_str(value: Any, path: str) -> str:
    parsed = _optional_str(value)
    if parsed is None:
        raise VerifierDocumentParseError(f"{path} must be a non-empty string")
    return parsed


def _safe_relative_path(value: str, path: str) -> str:
    parsed = Path(value)
    if parsed.is_absolute() or ".." in parsed.parts:
        raise VerifierDocumentParseError(f"{path} must be a safe relative path")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value:
        return value
    raise VerifierDocumentParseError("expected a non-empty string")
