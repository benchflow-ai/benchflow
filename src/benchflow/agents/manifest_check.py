"""A strict check of one agent ``manifest.toml`` (``bench agents validate``).

``load_agent_manifest`` reads what a manifest declares and drops what it
doesn't know; this module checks the file against the contract a release
speaks, without Docker or the network (a file study of the CLI: a misspelled
``modle_route`` was dropped and the run asked for a model; ``model_route = 7``
made the agent vanish from discovery; a missing file was "Unknown agent"):

- a ``contract_version`` this release knows (:data:`KNOWN_CONTRACT_VERSIONS`);
- every key one the contract has, an unknown one named with its line and the
  closest known key;
- every value of its key's type (``model_route`` one of ``proxy``,
  ``agent``; ``install_timeout`` a whole number of seconds; ...);
- the required keys present.

Each problem carries the file, its line (the key's, when the file parses) and
the key. :func:`manifest_identity` names a manifest by its bytes, so the same
manifest moved to another folder is the same agent.
"""

from __future__ import annotations

import difflib
import hashlib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The contract versions this release reads (``manifest.SUPPORTED_CONTRACT_MAJOR``
#: 1, minors 0 and 1; a patch level is accepted).
KNOWN_CONTRACT_VERSIONS = ("1.0", "1.1")

_STR, _BOOL, _INT = "a string", "true or false", "a whole number"
_STR_LIST, _STR_MAP = "an array of strings", "a table of strings"


def _is_str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _is_str_map(value: Any) -> bool:
    return isinstance(value, dict) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    )


def _model_routes() -> tuple[str, ...]:
    from benchflow.agents.registry import MODEL_ROUTES

    return MODEL_ROUTES


#: key -> (what it must be, check). ``contract_version`` is checked apart.
SCHEMA: dict[str, tuple[str, Any]] = {
    "name": (_STR, lambda v: isinstance(v, str) and bool(v.strip())),
    "description": (_STR, lambda v: isinstance(v, str)),
    "install_cmd": (_STR, lambda v: isinstance(v, str)),
    "launch_cmd": (_STR, lambda v: isinstance(v, str) and bool(v.strip())),
    "protocol": ("'acp'", lambda v: v == "acp"),
    "api_protocol": (_STR, lambda v: isinstance(v, str)),
    "acp_model_format": (_STR, lambda v: isinstance(v, str)),
    "supports_acp_set_model": (_BOOL, lambda v: isinstance(v, bool)),
    "requires_env": (_STR_LIST, _is_str_list),
    "install_timeout": (
        f"{_INT} of seconds, at least 1",
        lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 1,
    ),
    "env_mapping": (_STR_MAP, _is_str_map),
    "default_model": (_STR, lambda v: isinstance(v, str)),
    "skill_paths": (_STR_LIST, _is_str_list),
    "home_dirs": (_STR_LIST, _is_str_list),
    "model_route": (
        "one of " + ", ".join(repr(r) for r in ("proxy", "agent")),
        lambda v: isinstance(v, str) and v in _model_routes(),
    ),
    "aliases": (_STR_LIST, _is_str_list),
}
REQUIRED = ("contract_version", "name", "install_cmd", "launch_cmd")
KNOWN_KEYS = ("contract_version", *SCHEMA)


@dataclass(frozen=True)
class Problem:
    """One thing wrong with a manifest: where, which key, what."""

    path: str
    line: int | None
    key: str | None
    message: str

    def text(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        return (
            f"{where}: {self.key}: {self.message}"
            if self.key
            else f"{where}: {self.message}"
        )


@dataclass(frozen=True)
class ManifestIdentity:
    """A manifest by its bytes: its declared name, its SHA-256 and where it
    is relative to the agents folder that holds it."""

    name: str | None
    sha256: str
    relative_path: str

    def text(self) -> str:
        return f"{self.name or '?'} sha256:{self.sha256[:16]} ({self.relative_path})"


def manifest_identity(path: Path, root: Path | None = None) -> ManifestIdentity:
    data = path.read_bytes()
    try:
        name = tomllib.loads(data.decode()).get("name")
    except (UnicodeDecodeError, tomllib.TOMLDecodeError):
        name = None
    relative = path.name
    if root is not None:
        try:
            relative = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            relative = path.as_posix()
    return ManifestIdentity(
        name=name if isinstance(name, str) else None,
        sha256=hashlib.sha256(data).hexdigest(),
        relative_path=relative,
    )


_KEY_LINE = re.compile(
    r"""^\s*(?P<key>[A-Za-z0-9_-]+|"[^"]*"|'[^']*')\s*=\s*(?P<rest>.*)$"""
)
_TABLE_LINE = re.compile(r"^\s*\[+\s*(?P<key>[^\]]+?)\s*\]+")


def key_lines(text: str) -> dict[str, int]:
    """The line of each top-level key (a ``[table]``'s header line), skipping
    the inside of multi-line strings (an inline script's ``x = 1`` lines)."""
    lines: dict[str, int] = {}
    closing: str | None = None
    in_table = False
    for number, line in enumerate(text.splitlines(), start=1):
        if closing is not None:
            if closing in line:
                closing = None
            continue
        table = _TABLE_LINE.match(line)
        if table and "=" not in line.split("]", 1)[0]:
            in_table = True
            lines.setdefault(table["key"].split(".")[0].strip("\"'"), number)
            continue
        found = _KEY_LINE.match(line)
        if not found:
            continue
        if not in_table:
            lines.setdefault(found["key"].strip("\"'"), number)
        rest = found["rest"]
        for quote in ("'''", '"""'):
            if rest.startswith(quote) and rest.count(quote) == 1:
                closing = quote
                break
    return lines


def _toml_line(error: tomllib.TOMLDecodeError) -> int | None:
    found = re.search(r"line (\d+)", str(error))
    return int(found.group(1)) if found else None


def check_manifest(path: Path) -> list[Problem]:
    """Everything wrong with the manifest at ``path``; [] when it is valid.
    A missing or unreadable file is one problem with no line."""
    shown = path.as_posix()
    try:
        text = path.read_text()
    except FileNotFoundError:
        return [Problem(shown, None, None, "no such file")]
    except (OSError, UnicodeDecodeError) as exc:
        return [Problem(shown, None, None, f"cannot read it ({exc})")]
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        return [Problem(shown, _toml_line(exc), None, f"not valid TOML: {exc}")]
    lines = key_lines(text)
    problems: list[Problem] = []
    for key in data:
        if key in KNOWN_KEYS:
            continue
        close = difflib.get_close_matches(key, KNOWN_KEYS, n=1, cutoff=0.6)
        hint = f" (did you mean {close[0]!r}?)" if close else ""
        problems.append(Problem(shown, lines.get(key), key, f"unknown key{hint}"))
    for key in REQUIRED:
        if key not in data:
            problems.append(Problem(shown, None, key, "missing (required)"))
    version = data.get("contract_version")
    if "contract_version" in data:
        if not isinstance(version, str) or not re.fullmatch(
            r"\d+\.\d+(\.\d+)?", version
        ):
            problems.append(
                Problem(
                    shown,
                    lines.get("contract_version"),
                    "contract_version",
                    f'expected a version string such as "1.1"; got {version!r}',
                )
            )
        elif ".".join(version.split(".")[:2]) not in KNOWN_CONTRACT_VERSIONS:
            problems.append(
                Problem(
                    shown,
                    lines.get("contract_version"),
                    "contract_version",
                    f"{version!r} is not a contract version this BenchFlow reads "
                    f"({', '.join(KNOWN_CONTRACT_VERSIONS)})",
                )
            )
        elif version.startswith("1.0") and "model_route" in data:
            problems.append(
                Problem(
                    shown,
                    lines.get("model_route"),
                    "model_route",
                    "needs contract_version 1.1 or later",
                )
            )
    for key, (wanted, ok) in SCHEMA.items():
        if key in data and not ok(data[key]):
            problems.append(
                Problem(
                    shown,
                    lines.get(key),
                    key,
                    f"expected {wanted}; got {data[key]!r}",
                )
            )
    return sorted(problems, key=lambda p: (p.line or 10**9, p.key or ""))
