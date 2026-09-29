"""Compatibility imports for foreign task configuration files."""

from __future__ import annotations

import copy
import difflib
import logging
import os
import re
import tomllib
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, get_args

from pydantic import AliasChoices, BaseModel, ValidationError

from benchflow.task.config import TaskConfig, convert_legacy_environment_keys

logger = logging.getLogger(__name__)

# task.toml tables whose keys decide isolation, network, timeouts, resources or
# grading. An unknown key in one of these is refused (fail closed) unless it is
# a known-ignorable newer-Harbor key, so a typo such as ``allow_internett`` can
# never silently fall back to a default (a public network, the default timeout).
# Keyed by the top-level table name; the value describes what it controls.
SENSITIVE_SECTIONS: dict[str, str] = {
    "sandbox": "network access, isolation and resource limits",
    "agent": "the agent's network access and timeout",
    "verifier": "grading, the verifier timeout and its network access",
    "steps": "per-step isolation and grading",
}

# Unknown keys that are safe to ignore even inside a sensitive table: real keys
# from newer Harbor schemas whose semantics change nothing BenchFlow enforces.
# Dotted paths with list indices removed. Extend this (with a reason) rather
# than widening leniency, so the fail-closed default keeps catching typos.
#
# Empty today: BenchFlow models the whole current Harbor sensitive surface, so
# no real key in these tables needs excusing. It is the maintained escape hatch
# for a future Harbor field BenchFlow chooses to ignore rather than enforce.
KNOWN_IGNORABLE_KEYS: dict[str, str] = {}

# Escape hatch: set to a truthy value to restore the old drop-everything
# leniency (no typo or sensitive-table refusals). Documented in
# docs/running-benchmarks.md; for corpora BenchFlow does not control.
LENIENT_OPT_OUT_ENV = "BENCHFLOW_TASK_TOML_ALLOW_UNKNOWN_KEYS"

# Foreign task.toml keys BenchFlow does not model and whose semantics it
# cannot honour: a lenient load keeps them out of the config, and the runtime
# capability check refuses the task instead of running it without them.
# Keys are dotted paths with list indices removed.
UNHONOURED_FOREIGN_KEYS: dict[str, str] = {
    "verifier.collect": (
        "Harbor verifier collect hooks (commands run in a service before "
        "verification) are not executed"
    ),
    "steps.verifier.collect": (
        "Harbor verifier collect hooks (commands run in a service before "
        "verification) are not executed"
    ),
}


_WARNED: set[tuple[str, tuple[str, ...]]] = set()


def unhonoured_foreign_key(path: str) -> tuple[str, str] | None:
    """``(key, reason)`` when ``path`` is under an unhonoured foreign key."""
    plain = re.sub(r"\[\d+\]", "", path)
    for key, reason in UNHONOURED_FOREIGN_KEYS.items():
        if plain == key or plain.startswith(key + "."):
            return key, reason
    return None


def _lenient_opt_out() -> bool:
    return os.environ.get(LENIENT_OPT_OUT_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _field_keys(model: type[BaseModel]) -> dict[str, Any]:
    """Every task.toml key ``model`` accepts (field names and their aliases)."""
    keys: dict[str, Any] = {}
    for name, field_info in model.model_fields.items():
        keys[name] = field_info
        alias = field_info.validation_alias
        if isinstance(alias, AliasChoices):
            for choice in alias.choices:
                if isinstance(choice, str):
                    keys[choice] = field_info
        elif isinstance(alias, str):
            keys[alias] = field_info
    return keys


def _child_model(annotation: Any) -> type[BaseModel] | None:
    """The first task-config model reachable from a field annotation.

    Unwraps ``X | None``, ``list[X]`` and ``list[str | X]`` so ``verifier`` maps
    to ``VerifierConfig`` and ``artifacts`` to ``ArtifactConfig``; a plain
    scalar or a free ``dict`` (e.g. ``metadata``) yields ``None``.
    """
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in get_args(annotation):
        found = _child_model(arg)
        if found is not None:
            return found
    return None


@dataclass(frozen=True)
class UnknownKeyRefusal:
    """An unknown ``task.toml`` key the run and ``bench tasks check`` refuse."""

    path: str
    message: str


def _closest_known(name: str, choices: list[str]) -> str | None:
    match = difflib.get_close_matches(name, choices, n=1, cutoff=0.8)
    return match[0] if match and match[0] != name else None


def classify_unknown_key(path: str) -> UnknownKeyRefusal | None:
    """Refuse a probable typo, or an unknown key in a decision-bearing table.

    Returns ``None`` for a key that stays leniently ignored: a genuinely
    unknown key outside the sensitive tables, or one on
    :data:`KNOWN_IGNORABLE_KEYS`. Refuses (a) any unknown key close in spelling
    to a sibling known key, with a did-you-mean, and (b) any other unknown key
    inside a :data:`SENSITIVE_SECTIONS` table.
    """
    if unhonoured_foreign_key(path) is not None:
        # Handled by the runtime capability check with its own semantic message
        # (e.g. Harbor verifier.collect hooks): loads, then refused before launch.
        return None
    plain = re.sub(r"\[\d+\]", "", path)
    parts = plain.split(".")
    section = parts[0]

    # Walk the known model tree to the first component the schema does not know;
    # that component is the unknown key and the model reached is its table.
    model: type[BaseModel] = TaskConfig
    unknown: str | None = None
    for part in parts:
        keys = _field_keys(model)
        if part in keys:
            child = _child_model(keys[part].annotation)
            if child is None:
                # Under a free-form container (dict/scalar, e.g. metadata); such
                # keys never reach extra="forbid" validation, so stay lenient.
                return None
            model = child
            continue
        unknown = part
        break

    if unknown is None:
        return None

    suggestion = _closest_known(unknown, list(_field_keys(model)))
    if suggestion is not None:
        where = f"[{section}]" if section != unknown else "the task.toml top level"
        return UnknownKeyRefusal(
            path,
            f"unknown task.toml key {path!r} in {where} — did you mean "
            f"{suggestion!r}? Fix the typo, or set "
            f"{LENIENT_OPT_OUT_ENV}=1 to load unknown keys anyway.",
        )
    if section in SENSITIVE_SECTIONS and plain not in KNOWN_IGNORABLE_KEYS:
        return UnknownKeyRefusal(
            path,
            f"unknown task.toml key {path!r} in the [{section}] table, which "
            f"controls {SENSITIVE_SECTIONS[section]}; BenchFlow refuses it rather "
            "than run with a default. If it is a newer Harbor key that changes "
            f"nothing BenchFlow enforces, add it to KNOWN_IGNORABLE_KEYS, or set "
            f"{LENIENT_OPT_OUT_ENV}=1 to load unknown keys anyway.",
        )
    return None


def refused_unknown_keys(paths: tuple[str, ...] | list[str]) -> list[UnknownKeyRefusal]:
    """Refusals among ``paths`` (empty under the documented opt-out)."""
    if _lenient_opt_out():
        return []
    refusals = []
    for path in paths:
        refusal = classify_unknown_key(path)
        if refusal is not None:
            refusals.append(refusal)
    return refusals


@dataclass(frozen=True)
class TaskConfigImportReport:
    """Report for a foreign ``task.toml`` import.

    Native ``TaskConfig`` remains strict. Foreign adapters use this report to
    preserve unknown upstream keys in an explicit compatibility envelope rather
    than accepting them as first-class BenchFlow schema.
    """

    source: str
    status: Literal["strict", "preserved-extra"]
    extra: dict[str, Any] = field(default_factory=dict)
    extra_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["extra_paths"] = list(self.extra_paths)
        return data


@dataclass(frozen=True)
class ImportedTaskConfig:
    """A validated task config plus any preserved foreign extensions.

    ``declared`` is the parsed source mapping restricted to natively supported
    keys (compat extras removed). It records what the author actually wrote,
    so emitters can stay minimal instead of materializing model defaults.
    """

    config: TaskConfig
    report: TaskConfigImportReport
    declared: dict[str, Any]


def import_task_config_toml(
    toml_data: str,
    *,
    source: str,
) -> ImportedTaskConfig:
    """Validate foreign TOML while preserving unknown extension keys.

    This is deliberately separate from :meth:`TaskConfig.model_validate_toml`.
    The native parser keeps rejecting unknown keys; compatibility importers can
    opt into this two-pass parse when their job is to ingest foreign tasks.
    """

    # Foreign task.toml uses Harbor's 'environment' spelling for the sandbox
    # spec; translate it before validation (and before recording `declared`)
    # so importers and re-emitters see only the native 'sandbox' key.
    raw = convert_legacy_environment_keys(tomllib.loads(toml_data))
    try:
        config = TaskConfig.model_validate(copy.deepcopy(raw))
    except ValidationError as exc:
        extra_errors = [
            error for error in exc.errors() if error.get("type") == "extra_forbidden"
        ]
        if not extra_errors:
            raise

        sanitized = copy.deepcopy(raw)
        extra: dict[str, Any] = {}
        for error in extra_errors:
            path = tuple(error["loc"])
            value = _pop_path(sanitized, path)
            _set_path(extra, path, value)

        declared = copy.deepcopy(sanitized)
        try:
            config = TaskConfig.model_validate(sanitized)
        except ValidationError as sanitized_exc:
            raise exc from sanitized_exc

        paths = tuple(sorted(_format_path(path) for path in _leaf_paths(extra)))
        return ImportedTaskConfig(
            config=config,
            report=TaskConfigImportReport(
                source=source,
                status="preserved-extra",
                extra=extra,
                extra_paths=paths,
            ),
            declared=declared,
        )

    return ImportedTaskConfig(
        config=config,
        report=TaskConfigImportReport(source=source, status="strict"),
        declared=raw,
    )


class TaskConfigKeyError(ValueError):
    """A task.toml carries an unknown key the run refuses (typo/decision table)."""


def load_task_config_toml(toml_data: str, *, source: str) -> TaskConfig:
    """Load a task.toml for running it: safe unknown keys are ignored, typos refused.

    The run-path counterpart of :meth:`TaskConfig.model_validate_toml`, which
    stays strict. Genuinely unknown keys outside the decision-bearing tables are
    ignored with a warning and recorded on ``config.ignored_keys``; wrong types
    and invalid values still raise. A probable typo of a known key, or an unknown
    key inside a table that decides isolation, network, timeouts, resources or
    grading (:data:`SENSITIVE_SECTIONS`), is refused with a
    :class:`TaskConfigKeyError` so it cannot silently fall back to a default —
    unless it is a known-ignorable newer-Harbor key or ``$``
    :data:`LENIENT_OPT_OUT_ENV` is set. Keys in :data:`UNHONOURED_FOREIGN_KEYS`
    are recorded and refused later by the runtime capability check.
    """

    imported = import_task_config_toml(toml_data, source=source)
    config = imported.config
    refusals = refused_unknown_keys(imported.report.extra_paths)
    if refusals:
        detail = "\n".join(f"- {refusal.message}" for refusal in refusals)
        raise TaskConfigKeyError(f"{source}: {detail}")
    config._ignored_keys = imported.report.extra_paths
    plain = sorted(
        p for p in imported.report.extra_paths if unhonoured_foreign_key(p) is None
    )
    # A run loads the same task several times; say it once per file and keys.
    key = (source, tuple(plain))
    if plain and key not in _WARNED:
        _WARNED.add(key)
        logger.warning(
            "%s: ignored %d task.toml key(s) BenchFlow does not know: %s",
            source,
            len(plain),
            ", ".join(plain),
        )
    return config


def merge_compat_extra(
    base: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Return ``base`` with preserved foreign keys restored.

    ``extra`` comes from a compatibility envelope and must not overwrite a
    supported native key. A collision means the native schema learned that key
    after import, so the native value is authoritative.
    """

    merged = copy.deepcopy(base)
    _merge_missing(merged, extra)
    return merged


def _pop_path(data: dict[str, Any], path: tuple[str | int, ...]) -> Any:
    current: Any = data
    for part in path[:-1]:
        if isinstance(current, dict):
            current = current.get(part)
            continue
        if isinstance(current, list) and isinstance(part, int):
            if part >= len(current):
                return None
            current = current[part]
            continue
        return None
    final_part = path[-1]
    if isinstance(current, dict) and isinstance(final_part, str):
        return current.pop(final_part, None)
    if isinstance(current, list) and isinstance(final_part, int):
        if final_part >= len(current):
            return None
        value = current[final_part]
        current[final_part] = None
        return value
    return None


def _set_path(data: dict[str, Any], path: tuple[str | int, ...], value: Any) -> None:
    current: Any = data
    for index, part in enumerate(path[:-1]):
        next_part = path[index + 1]
        child = _get_child(current, part)
        if not _container_matches(child, next_part):
            child = [] if isinstance(next_part, int) else {}
            _assign_child(current, part, child)
        current = child
    _assign_child(current, path[-1], value)


def _get_child(container: Any, part: str | int) -> Any:
    if isinstance(container, dict):
        return container.get(part)
    if isinstance(container, list) and isinstance(part, int) and part < len(container):
        return container[part]
    return None


def _assign_child(container: Any, part: str | int, value: Any) -> None:
    if isinstance(container, dict):
        container[part] = value
        return
    if isinstance(container, list) and isinstance(part, int):
        while len(container) <= part:
            container.append(None)
        container[part] = value
        return
    raise TypeError(f"cannot assign compatibility path segment {part!r}")


def _container_matches(value: Any, next_part: str | int) -> bool:
    return (
        isinstance(value, list)
        if isinstance(next_part, int)
        else isinstance(value, dict)
    )


def _leaf_paths(
    data: dict[str, Any] | list[Any],
    prefix: tuple[str | int, ...] = (),
) -> list[tuple[str | int, ...]]:
    paths: list[tuple[str | int, ...]] = []
    items = enumerate(data) if isinstance(data, list) else data.items()
    for key, value in items:
        path = (*prefix, key)
        if isinstance(value, dict | list):
            child_paths = _leaf_paths(value, path)
            paths.extend(child_paths or [path])
        elif value is not None:
            paths.append(path)
    return paths


def _format_path(path: tuple[str | int, ...]) -> str:
    rendered = ""
    for part in path:
        if isinstance(part, int):
            rendered += f"[{part}]"
        elif rendered:
            rendered += f".{part}"
        else:
            rendered = part
    return rendered


def _merge_missing(
    target: dict[str, Any] | list[Any], extra: dict[str, Any] | list[Any]
) -> None:
    if isinstance(target, list) and isinstance(extra, list):
        for index, value in enumerate(extra):
            if value is None:
                continue
            while len(target) <= index:
                target.append(None)
            if target[index] is None:
                target[index] = copy.deepcopy(value)
            elif isinstance(target[index], dict | list) and isinstance(
                value, dict | list
            ):
                _merge_missing(target[index], value)
        return

    if not isinstance(target, dict) or not isinstance(extra, dict):
        return

    for key, value in extra.items():
        if key not in target:
            target[key] = copy.deepcopy(value)
        elif isinstance(target[key], dict | list) and isinstance(value, dict | list):
            _merge_missing(target[key], value)
