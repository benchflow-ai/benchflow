"""One check of the keys in BenchFlow's config files, with errors that name
the file and the line.

Every file a user writes by hand goes through it:

* a run config (``bench run --config run.yaml``): :func:`check_keys` against
  the keys the loader reads, so ``concurrenzy: 2`` is refused, never run
  with the default;
* a task's ``task.md`` frontmatter: :func:`validation_error_text` turns
  ``TaskConfig``'s validation errors into ``task.md:10: agent.timeout:
  unknown key (did you mean timeout_sec?)`` instead of pydantic's dump;
* ``task.toml`` (``task/imports.py``): the same :func:`closest_key` and the
  model walk (:func:`model_keys`, :func:`child_model`).

Lines come from the YAML text itself (:func:`key_lines`), numbered as in the
file, so a frontmatter key's line counts the opening ``---``.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, get_args

from benchflow.errors import UserError


def parse_run_config(text: str, path: Path) -> Any:
    """Read JSON as JSON, preserving numeric spellings such as ``5e-05``.

    YAML remains the format for other run-config paths. Key locations are
    still collected from the original text by ``key_lines``.
    """
    import yaml

    return json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)


__all__ = [
    "ConfigKeyError",
    "check_keys",
    "child_model",
    "closest_key",
    "key_lines",
    "located",
    "model_keys",
    "validation_error_text",
]


class ConfigKeyError(ValueError, UserError):
    """A run config names a key BenchFlow does not read.

    The message names the file, the line and the key; ``hint`` is the next
    step. A task file's problems are the task's ``TaskDocumentParseError``.
    """

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


def closest_key(
    name: str, choices: Iterable[str], *, cutoff: float = 0.6
) -> str | None:
    """The known key ``name`` most likely misspells, or None."""
    match = difflib.get_close_matches(
        str(name), sorted(set(choices)), n=1, cutoff=cutoff
    )
    return match[0] if match and match[0] != name else None


def model_keys(model: Any) -> dict[str, Any]:
    """Every key a pydantic ``model`` accepts: field names and their aliases."""
    from pydantic import AliasChoices

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


def child_model(annotation: Any) -> Any | None:
    """The first pydantic model reachable from a field annotation.

    Unwraps ``X | None``, ``list[X]`` and ``list[str | X]``; a scalar or a
    free ``dict`` yields None.
    """
    from pydantic import BaseModel

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in get_args(annotation):
        found = child_model(arg)
        if found is not None:
            return found
    return None


def key_lines(text: str, *, first_line: int = 1) -> dict[tuple[str, ...], int]:
    """The line of every key in a YAML document, by its path of keys.

    ``first_line`` is the file line the text starts on (2 for a frontmatter
    after its ``---``). A list item's path holds its index. Text that does not
    parse gives no lines: the caller reports the YAML error instead.
    """
    import yaml

    try:
        root = yaml.compose(text)
    except yaml.YAMLError:
        return {}
    lines: dict[tuple[str, ...], int] = {}

    def walk(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, yaml.MappingNode):
            for key, value in node.value:
                name = str(getattr(key, "value", key))
                lines[(*path, name)] = key.start_mark.line + first_line
                walk(value, (*path, name))
        elif isinstance(node, yaml.SequenceNode):
            for index, item in enumerate(node.value):
                lines.setdefault((*path, str(index)), item.start_mark.line + first_line)
                walk(item, (*path, str(index)))

    if root is not None:
        walk(root, ())
    return lines


def located(file: str | Path | None, line: int | None) -> str:
    """``file:line`` (the file as the user names it from here: relative to
    the current folder when it is under it), ``file``, or nothing."""
    if file is None:
        return ""
    shown = Path(file)
    if shown.is_absolute() and shown.is_relative_to(Path.cwd()):
        shown = shown.relative_to(Path.cwd())
    return f"{shown}:{line}" if line is not None else str(shown)


def check_keys(
    data: Mapping[str, Any],
    allowed: Iterable[str],
    *,
    file: str | Path,
    lines: Mapping[tuple[str, ...], int] | None = None,
    prefix: tuple[str, ...] = (),
    renamed: Mapping[str, str] | None = None,
    what: str = "key",
) -> list[str]:
    """One line per key of ``data`` not in ``allowed``, with a did-you-mean.

    ``renamed`` maps a key people write to the key the file takes (the
    flag ``--sandbox`` is the YAML key ``environment``). Returns the lines;
    the caller raises them as one :class:`ConfigKeyError`.
    """
    known = set(allowed)
    problems: list[str] = []
    for key in data:
        if key in known:
            continue
        name = str(key)
        dotted = ".".join((*prefix, name))
        where = located(file, (lines or {}).get((*prefix, name)))
        if renamed and name in renamed:
            problems.append(f"{where}: {dotted}: this file calls it {renamed[name]!r}")
            continue
        guess = closest_key(name, known)
        tail = f" (did you mean {guess!r}?)" if guess else ""
        problems.append(f"{where}: {dotted}: unknown {what}{tail}")
    return problems


def _model_at(model: Any, loc: tuple[Any, ...]) -> Any | None:
    """The model a validation error's ``loc`` (without its last part) points into."""
    for part in loc:
        if isinstance(part, int):
            continue
        field = model_keys(model).get(str(part))
        if field is None:
            return None
        model = child_model(field.annotation)
        if model is None:
            return None
    return model


def validation_error_text(
    exc: Any,
    *,
    model: Any,
    file: str | Path | None,
    lines: Mapping[tuple[str, ...], int] | None = None,
) -> str:
    """A pydantic ``ValidationError`` as one line per problem.

    ``task.md:10: agent.timeout: unknown key (did you mean timeout_sec?)``,
    ``task.md:9: agent.timeout_sec: Input should be a valid number, got
    'five-minutes'``: the file and line, the key, what is wrong, and no
    pydantic link.
    """
    out: list[str] = []
    for error in exc.errors(include_url=False):
        loc = tuple(error.get("loc") or ())
        dotted = ".".join(str(p) for p in loc) or "(top level)"
        line = (lines or {}).get(tuple(str(p) for p in loc))
        where = located(file, line)
        kind = error.get("type")
        if kind == "extra_forbidden" and loc:
            parent = _model_at(model, loc[:-1])
            guess = closest_key(str(loc[-1]), model_keys(parent)) if parent else None
            message = "unknown key" + (f" (did you mean {guess}?)" if guess else "")
        else:
            message = str(error.get("msg") or "invalid value")
            message = message.removeprefix("Value error, ")
            value = error.get("input")
            if isinstance(value, str | int | float | bool) and kind != "missing":
                message += f", got {value!r}"
        out.append(f"{where}: {dotted}: {message}" if where else f"{dotted}: {message}")
    return "\n".join(out)
