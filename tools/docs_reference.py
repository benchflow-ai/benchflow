"""Write the reference pages from the code.

- ``docs/reference/commands.md``: every ``bench`` command, with its arguments
  and options, from the live Typer app. The options ``--help`` leaves out have
  a table of their own, the commands ``bench --help`` hides are listed at the
  end, and a command or option the 0.9.2 release did not have
  (``tools/cli_baseline_0.9.2.json``) is labelled ``(0.10)`` when 0.10.0rc3
  has it (``tools/cli_baseline_0.10.0rc3.json``) and ``(unreleased)`` when no
  release has it yet.
- ``docs/reference/python.md``: every name in ``benchflow.__all__``, with its
  signature (or fields) and the first paragraph of its docstring. A name,
  field or argument the 0.9.2 release did not have
  (``tools/reference_baseline_0.9.2.json``) is labelled ``(0.10)`` or
  ``(unreleased)`` the same way (``tools/reference_baseline_0.10.0rc3.json``),
  here and on the next two pages.
- ``docs/reference/cli-python.md``: the tables of ``benchflow.cli_parity``,
  each flag with the Python that does the same.
- ``docs/reference/files.md``: the ``task.md`` header (``TaskConfig``'s
  fields, and the keys the task.md parser reads itself), the run config's
  keys, and every JSON schema in ``docs/reference/schemas/`` with its
  top-level fields.
- ``docs/versions/changelog.md``: ``CHANGELOG.md``, its links to old docs
  pages moved to their new pages with ``docs/redirects.json``.

The pages are rendered in a fresh interpreter with the fixed environment
``tools/cli_help_golden.py`` uses, so they come out the same on every machine.
``tests/test_docs_reference.py`` fails when a committed page differs. To change
a page, change the help text, docstring, field, schema or changelog it comes
from, then regenerate on a build VM with the test environment and read
``git diff``::

    uv run python tools/docs_reference.py --write

``--check`` exits 1 and prints a diff when a page is stale.

``--dump-inventory PATH`` writes every command (hidden ones included) and the
names of its options, as JSON, for the CLI the running interpreter imports.
Run with a release's own environment (a venv with that release installed from
PyPI with its constraints file), it makes the baseline the labels come from::

    /path/to/0.9.2/venv/bin/python tools/docs_reference.py \\
        --dump-inventory tools/cli_baseline_0.9.2.json

``--dump-names PATH --schemas DIR`` does the same for the Python and files
pages: the public names with their fields or arguments, the task.md, run
config and job config keys, and the fields of the schema files in ``DIR``
(that release's ``docs/reference/schemas``)::

    /path/to/0.9.2/venv/bin/python tools/docs_reference.py \\
        --dump-names tools/reference_baseline_0.9.2.json \\
        --schemas /path/to/v0.9.2-checkout/docs/reference/schemas
"""

from __future__ import annotations

import argparse
import ast
import difflib
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import click

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.cli_help_golden import _child_env  # noqa: E402

PAGES = {
    "commands": REPO / "docs" / "reference" / "commands.md",
    "python": REPO / "docs" / "reference" / "python.md",
    "cli-python": REPO / "docs" / "reference" / "cli-python.md",
    "files": REPO / "docs" / "reference" / "files.md",
    "changelog": REPO / "docs" / "versions" / "changelog.md",
}
SCHEMAS = REPO / "docs" / "reference" / "schemas"
CHANGELOG = REPO / "CHANGELOG.md"
# Every old docs page and the page it moved to; the changelog's links to old
# pages are rewritten with it.
REDIRECTS = REPO / "docs" / "redirects.json"
# The commands and options of the last release, and the label of what is new
# since. Remake the file with --dump-inventory from that release's tree.
BASELINE = REPO / "tools" / "cli_baseline_0.9.2.json"
BASELINE_RELEASE = "0.9.2"
NEW_LABEL = "0.10"
# The prerelease the NEW_LABEL items shipped in: what it lacks too is in no
# release yet, and is labelled NEXT_LABEL instead.
PRE_BASELINE = REPO / "tools" / "cli_baseline_0.10.0rc3.json"
PRE_RELEASE = "0.10.0rc3"
NEXT_LABEL = "unreleased"
# The same release's public Python names and their fields, its task.md and
# run config keys, and its schema files' fields: what the Python and files
# pages list beyond them is labelled too. Remake it with --dump-names.
NAMES_BASELINE = REPO / "tools" / "reference_baseline_0.9.2.json"
PRE_NAMES_BASELINE = REPO / "tools" / "reference_baseline_0.10.0rc3.json"
_SCHEMAS_FROM = "_BENCHFLOW_DOCS_REFERENCE_SCHEMAS"
REGENERATE = "uv run python tools/docs_reference.py --write"
_CHILD = "_BENCHFLOW_DOCS_REFERENCE_CHILD"
# A note for whoever edits the Markdown: a comment, so the site and GitHub's view don't show it to readers.
_GENERATED = (
    "<!-- This page is generated from the code by tools/docs_reference.py. "
    "To change it, change the {source}, then run `{regenerate}`. -->"
)


def _text(text: str) -> str:
    """Prose for a page the site renders as MDX: ``<``, ``>``, ``{`` and ``}``
    are escaped outside code spans; table pipes are escaped everywhere."""
    out = []
    for i, part in enumerate(re.split(r"(`[^`]*`)", text)):
        if i % 2:
            out.append(part.replace("|", "\\|"))
            continue
        out.append(
            part.replace("|", "\\|")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("{", "&#123;")
            .replace("}", "&#125;")
        )
    return "".join(out)


def _one_line(text: str | None) -> str:
    return _text(" ".join((text or "").split()))


def _paragraphs(text: str | None) -> list[str]:
    """Click help text as paragraphs: drop the no-rewrap markers, join lines. An
    indented run of ``bench`` commands (a docstring's examples) stays one command
    a line, in a code block, after the line that introduces it if there is one."""
    text = (text or "").replace("\b\n", "\n").replace("\b", "")
    out = []
    for para in re.split(r"\n\s*\n", text):
        if not para.strip():
            continue
        lines = para.split("\n")
        lead = [] if lines[0][:1].isspace() else lines[:1]
        rest = lines[len(lead) :]
        commands = textwrap.dedent("\n".join(rest)).split("\n") if rest else []
        if (
            commands
            and all(line[:1].isspace() for line in rest)
            and all(line.startswith("bench ") for line in commands)
        ):
            if lead:
                out.append(_text(" ".join(" ".join(lead).split())))
            out.append("```bash\n" + "\n".join(commands) + "\n```")
        else:
            out.append(_text(" ".join(para.split())))
    return out


def _code(value: str) -> str:
    # A pipe splits a table row even inside a code span unless escaped.
    value = value.replace("`", "'").replace("|", "\\|")
    return f"`{value}`" if value else ""


def _slug(heading: str) -> str:
    """A heading's anchor, as GitHub and the docs' link check make it."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    text = re.sub(r"<[^>]+>", "", text).strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


# --- commands ---------------------------------------------------------------


def _command_paths(
    command: click.Command, path: tuple[str, ...] = (), hidden: bool = False
) -> Iterator[tuple[tuple[str, ...], click.Command, bool]]:
    """Every command under ``command``, hidden ones included, depth first, in
    listing order, with whether it (or a group above it) is hidden."""
    import click

    yield path, command, hidden
    if isinstance(command, click.Group):
        ctx = click.Context(command, info_name=path[-1] if path else "bench")
        for name in command.list_commands(ctx):
            sub = command.get_command(ctx, name)
            if sub is None:
                continue
            yield from _command_paths(
                sub, (*path, name), hidden or bool(getattr(sub, "hidden", False))
            )


def _command_name(path: tuple[str, ...]) -> str:
    return " ".join(("bench", *path))


def _root_command() -> click.Command:
    import typer.main

    from benchflow.cli.main import app

    return typer.main.get_command(app)


def inventory() -> dict[str, list[str]]:
    """Every command, hidden ones included, with the names of its options
    (every spelling of each, ``--help`` aside)."""
    import click

    out: dict[str, list[str]] = {}
    for path, command, _hidden in _command_paths(_root_command()):
        names: set[str] = set()
        for param in command.params:
            if isinstance(param, click.Option):
                names.update(param.opts)
                names.update(param.secondary_opts)
        out[_command_name(path)] = sorted(names)
    return dict(sorted(out.items()))


# Options the baseline release has but that do more since; said in the row, since
# the option itself is not new. Found by comparing every option's help in 0.9.1
# and 0.10.0rc1 (the others that changed only say the same thing in other words;
# 0.9.2's commands and options are 0.9.1's), then 0.9.2's with 0.10.0rc2's.
_CHANGED_SINCE = {
    ("bench run", "--dry-run"): "In 0.9.2 it prints the plan only, with no run card.",
    (
        "bench eval run",
        "--dry-run",
    ): "In 0.9.2 it prints the plan only, with no run card.",
    (
        "bench eval create",
        "--dry-run",
    ): "In 0.9.2 it prints the plan only, with no run card.",
    ("bench train convert", "--format"): (
        "0.9.2 writes prime-sft, trl-sft and branch-tree only."
    ),
    ("bench train validate", "--format"): (
        "0.9.2 and 0.10.0rc2 check prime-sft, trl-sft and branch-tree rows only."
    ),
}


def _tier(in_release: bool, in_prerelease: bool) -> str:
    """No label for what the baseline release has, ``(0.10)`` for what the
    prerelease added, ``(unreleased)`` for what no release has yet."""
    if in_release:
        return ""
    return f" ({NEW_LABEL})" if in_prerelease else f" ({NEXT_LABEL})"


# Defaults the stable release has that the 0.10 line changed (0.10.0rc2 has the new ones): (public name,
# field or argument) -> the stable release's default.
_RELEASE_DEFAULTS = {
    ("JudgeConfig", "model"): "claude-sonnet-4-6",
    ("LLMJudgeRewardFunc", "model"): "claude-sonnet-4-6",
}


def _release_default_lines(name: str, params) -> list[str]:
    return [
        f"In {BASELINE_RELEASE}, `{p}` defaults to `{_RELEASE_DEFAULTS[name, p]}`."
        for p in params
        if (name, p) in _RELEASE_DEFAULTS
    ]


def _new_arguments(since: _Since, name: str, params) -> list[str]:
    """The lines naming a function's arguments the baseline release lacks."""
    out: list[str] = []
    for label, lead in (
        (f" ({NEW_LABEL})", f"New arguments in {NEW_LABEL}: "),
        (f" ({NEXT_LABEL})", "Arguments in no release yet: "),
    ):
        new = [p for p in params if since.python(name, p) == label]
        if new:
            out += ["", lead + ", ".join(f"`{p}`" for p in new) + "."]
    return out


def _labels() -> _Labels:
    return _Labels(
        json.loads(BASELINE.read_text(encoding="utf-8")),
        json.loads(PRE_BASELINE.read_text(encoding="utf-8")),
    )


class _Labels:
    """``(0.10)`` for a command or an option the baseline release lacks and
    the prerelease has; ``(unreleased)`` when the prerelease lacks it too."""

    def __init__(
        self, baseline: dict[str, list[str]], prerelease: dict[str, list[str]]
    ) -> None:
        self.baseline = {name: frozenset(opts) for name, opts in baseline.items()}
        self.prerelease = {name: frozenset(opts) for name, opts in prerelease.items()}

    def changed(self, path: tuple[str, ...], param) -> str:
        """What the baseline release did instead, for an option that does more since."""
        for name in param.opts:
            note = _CHANGED_SINCE.get((_command_name(path), name))
            if note:
                return f" {note}"
        return ""

    def command(self, path: tuple[str, ...]) -> str:
        name = _command_name(path)
        return _tier(name in self.baseline, name in self.prerelease)

    def option(self, path: tuple[str, ...], param) -> str:
        names = {*param.opts, *(getattr(param, "secondary_opts", None) or [])}
        return self.flag(_command_name(path), names)

    def flag(self, command: str, names: set[str]) -> str:
        known = self.baseline.get(command)
        pre = self.prerelease.get(command)
        return _tier(
            known is not None and bool(names & known),
            pre is not None and bool(names & pre),
        )


def names_inventory(schemas: Path) -> dict[str, Any]:
    """The public names and keys the Python and files pages list, for the
    package the running interpreter imports, and the fields of the schema
    files in ``schemas``."""
    import dataclasses

    import benchflow as bf
    from benchflow import job_config
    from benchflow._types import Role, Scene, Turn
    from benchflow.evaluation import LEGACY_RUN_CONFIG_KEYS, RUN_CONFIG_KEYS
    from benchflow.task._document_parse import _DOCUMENT_ONLY_FRONTMATTER_KEYS
    from benchflow.task.config import TaskConfig

    python: dict[str, list[str]] = {}
    for name in bf.__all__:
        obj = getattr(bf, name)
        rows = _fields(obj) if inspect.isclass(obj) else None
        if rows is not None:
            python[name] = sorted(n for n, _, _ in rows)
        elif callable(obj):
            try:
                python[name] = sorted(inspect.signature(obj).parameters)
            except (TypeError, ValueError):
                python[name] = []
        else:
            python[name] = []

    def keys(model: type) -> list[str]:
        return [_field_key(n, f) for n, f in model.model_fields.items()]  # ty: ignore[unresolved-attribute]

    sections, _ = _model_sections(TaskConfig)
    task_md = {"": sorted({*keys(TaskConfig), *_DOCUMENT_ONLY_FRONTMATTER_KEYS})}
    task_md.update((path, sorted(keys(model))) for path, model in sections)
    for path, cls in (
        ("agents.roles.*", Role),
        ("scenes[]", Scene),
        ("scenes[].turns[]", Turn),
    ):
        task_md[path] = sorted(f.name for f in dataclasses.fields(cls))
    package = Path(bf.__file__).parent
    for block in ("user", "benchflow"):
        task_md[block] = _document_keys_read(block, package)
    methods = {
        name: sorted(m for m, _, _ in _methods(getattr(bf, name)))
        for name in bf.__all__
        if inspect.isclass(getattr(bf, name))
    }
    return {
        "python": dict(sorted(python.items())),
        "methods": dict(sorted(methods.items())),
        "task_md": dict(sorted(task_md.items())),
        "run_config": sorted(RUN_CONFIG_KEYS),
        "job_config": sorted(LEGACY_RUN_CONFIG_KEYS | job_config.JOB_KEYS),
        "job_agent": sorted(job_config._AGENT_HONOURED),
        "job_dataset": sorted(job_config._DATASET_HONOURED),
        "schemas": {
            path.name: sorted(
                json.loads(path.read_text(encoding="utf-8")).get("properties") or {}
            )
            for path in sorted(schemas.glob("*.json"))
        },
    }


class _Since:
    """``(0.10)`` for a public name, field, key or schema field the baseline
    release lacks and the prerelease has, ``(unreleased)`` when the prerelease
    lacks it too. A member of something new isn't labelled again, unless the
    prerelease had its parent without it."""

    def __init__(self) -> None:
        self.base = json.loads(NAMES_BASELINE.read_text(encoding="utf-8"))
        self.pre = json.loads(PRE_NAMES_BASELINE.read_text(encoding="utf-8"))

    @staticmethod
    def _label(known: list[str] | None, pre: list[str] | None, item: str) -> str:
        if known is None:
            # The parent is new and labelled itself; only an addition after
            # the prerelease that had the parent is labelled again.
            return "" if pre is None or item in pre else f" ({NEXT_LABEL})"
        return _tier(item in known, pre is not None and item in pre)

    def python(self, name: str, member: str | None = None) -> str:
        known, pre = self.base["python"], self.pre["python"]
        if member is None:
            return _tier(name in known, name in pre)
        return self._label(known.get(name), pre.get(name), member)

    def method(self, name: str, member: str) -> str:
        return self._label(
            self.base["methods"].get(name), self.pre["methods"].get(name), member
        )

    def key(self, part: str, item: str, section: str | None = None) -> str:
        known, pre = self.base[part], self.pre[part]
        if section is not None:
            return self._label(known.get(section), pre.get(section), item)
        return self._label(known, pre, item)

    def schema(self, file: str, field: str | None = None) -> str:
        known, pre = self.base["schemas"], self.pre["schemas"]
        if field is None:
            return _tier(file in known, file in pre)
        return self._label(known.get(file), pre.get(file), field)


def _type_name(param) -> str:
    import click

    if getattr(param, "is_flag", False) and not getattr(param, "count", False):
        return "flag"
    kind = param.type
    if isinstance(kind, click.Choice):
        return " \\| ".join(_code(str(c)) for c in kind.choices)
    name = getattr(kind, "name", "") or type(kind).__name__
    name = {
        "text": "text",
        "integer": "integer",
        "float": "number",
        "boolean": "boolean",
    }.get(name, name)
    return _code(name + (" (repeatable)" if getattr(param, "multiple", False) else ""))


def _default(param) -> str:
    if getattr(param, "show_default", None) is False:
        return ""
    value = param.default
    if callable(value) or value is None or value == () or value == []:
        return ""
    if getattr(param, "is_flag", False) and value is False:
        return ""
    if hasattr(value, "value") and not isinstance(value, (str, int, float)):
        value = value.value
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(v) for v in value)
    return _code(str(value))


def _option_names(param) -> str:
    names = sorted(param.opts, key=lambda o: (not o.startswith("--"), o))
    names += list(getattr(param, "secondary_opts", []) or [])
    return ", ".join(_code(n) for n in names)


def _option_row(param, label: str, changed: str = "") -> str:
    text = _one_line(param.help) + (" Required." if param.required else "")
    if changed and not text.endswith((".", "?", "!")):
        text += "."
    return (
        f"| {_option_names(param)}{label} | {_type_name(param)} | {_default(param)} "
        f"| {text}{changed} |"
    )


def _usage(command, path: tuple[str, ...]) -> str:
    import click

    ctx = click.Context(command, info_name=_command_name(path))
    pieces = command.collect_usage_pieces(ctx)
    return " ".join(["bench", *path, *pieces])


def _short_help(command) -> str:
    if not hasattr(command, "get_short_help_str"):
        return ""
    return _one_line(command.get_short_help_str(limit=200))


def _commands_page() -> str:
    import click

    labels = _labels()
    entries = list(_command_paths(_root_command()))
    shown = [(path, command) for path, command, hidden in entries if not hidden]
    hidden_commands = [(path, command) for path, command, hidden in entries if hidden]
    heading = {path: _command_name(path) + labels.command(path) for path, _ in shown}
    # Commands no release has go after the released ones, under a heading of their own.
    unreleased = {
        path for path, _ in shown if labels.command(path) == f" ({NEXT_LABEL})"
    }
    shown = [e for e in shown if e[0] not in unreleased] + [
        e for e in shown if e[0] in unreleased
    ]
    unreleased_section = "Commands in no release yet"
    hidden_section = "Commands hidden from `bench --help`"
    out = [
        "# Commands",
        "",
        "Each `bench` command and flag has its type, default and help here; `benchflow` is "
        "the same command as `bench`. "
        f"A command or flag new since {BASELINE_RELEASE} has `({NEW_LABEL})` after its "
        f"name when {PRE_RELEASE} has it, or `({NEXT_LABEL})` when no release has it yet; the options `--help` leaves out have a table of their own under their "
        f"command, and [the commands `bench --help` hides](#{_slug(hidden_section)}) come last. "
        f"Which options `--help` leaves out follows {PRE_RELEASE}: 0.9.2's `bench run --help` also leaves out "
        "`--trials`, `--max-cost-usd`, `--max-tokens` and `--integrity`, which its `--help-all` lists. "
        "To see the commands at work, start with the [Quick start](../start/quick-start.md) "
        "and [Run a benchmark](../evaluate/run-a-benchmark.md); a run config's keys are in "
        "[Files BenchFlow reads and writes](./files.md), and [CLI and Python](./cli-python.md) "
        "gives each flag's Python argument.",
        "",
        _GENERATED.format(source="command's help text", regenerate=REGENERATE),
        "",
        "| Command | What it does |",
        "|---|---|",
    ]
    for path, command in shown:
        if not path or path in unreleased:
            continue
        out.append(
            f"| [`{_command_name(path)}`](#{_slug(heading[path])}){labels.command(path)} "
            f"| {_short_help(command)} |"
        )
    if unreleased:
        out += [
            "",
            f"The [commands in no release yet](#{_slug(unreleased_section)}) are listed after these.",
        ]
    first_unreleased = next((path for path, _ in shown if path in unreleased), None)
    for path, command in shown:
        if path == first_unreleased:
            out += [
                "",
                f"## {unreleased_section}",
                "",
                f"These commands are on the main line after {PRE_RELEASE}: neither {BASELINE_RELEASE} "
                f"nor {PRE_RELEASE} has them, and they can change before a release.",
            ]
        out += ["", f"## {heading[path]}", ""]
        if getattr(command, "deprecated", False):
            out += ["Deprecated: still runs, and prints which command replaces it.", ""]
        for para in _paragraphs(command.help):
            out += [para, ""]
        out += ["```text", _usage(command, path), "```"]
        if isinstance(command, click.Group):
            ctx = click.Context(command, info_name=_command_name(path))
            rows = []
            for name in command.list_commands(ctx):
                sub = command.get_command(ctx, name)
                if sub is None or getattr(sub, "hidden", False):
                    continue
                sub_path = (*path, name)
                rows.append(
                    f"| [`{name}`](#{_slug(heading[sub_path])}){labels.command(sub_path)} "
                    f"| {_short_help(sub)} |"
                )
            if rows:
                out += ["", "| Command | What it does |", "|---|---|", *rows]
        arguments = [p for p in command.params if isinstance(p, click.Argument)]
        if arguments:
            out += ["", "| Argument | Type | Required | Help |", "|---|---|---|---|"]
            for p in arguments:
                metavar = p.human_readable_name.upper()
                out.append(
                    f"| {_code(metavar)} | {_type_name(p)} | {'yes' if p.required else 'no'} "
                    f"| {_one_line(getattr(p, 'help', '') or '')} |"
                )
        options = [
            p
            for p in command.params
            if isinstance(p, click.Option) and "--help" not in p.opts
        ]
        panels: dict[str, list[str]] = {}
        for p in options:
            if p.hidden:
                continue
            panel = getattr(p, "rich_help_panel", None) or "Options"
            panels.setdefault(panel, []).append(
                _option_row(p, labels.option(path, p), labels.changed(path, p))
            )
        for panel, rows in panels.items():
            out += [
                "",
                f"**{_text(panel)}**",
                "",
                "| Option | Type | Default | Help |",
                "|---|---|---|---|",
                *rows,
            ]
        more = [
            _option_row(p, labels.option(path, p), labels.changed(path, p))
            for p in options
            if p.hidden
        ]
        if more:
            # bench run's --help-all lists them; elsewhere no help does.
            shows = any("--help-all" in p.opts for p in options)
            title = (
                "More options (shown by `--help-all`)"
                if shows
                else "More options (hidden from `--help`)"
            )
            out += [
                "",
                f"**{title}**",
                "",
                "| Option | Type | Default | Help |",
                "|---|---|---|---|",
                *more,
            ]

    def function(command) -> Any:
        # Typer wraps each command's function; an alias wraps the same one.
        callback = getattr(command, "callback", None)
        return inspect.unwrap(callback) if callback is not None else None

    callbacks: dict[Any, str] = {}
    for path, command in shown:
        if function(command) is not None:
            callbacks.setdefault(function(command), _command_name(path))
    out += [
        "",
        f"## {hidden_section}",
        "",
        "`bench --help` doesn't list these. They are internal, or older spellings kept "
        "so old scripts still run, and may change or go away in any release; each "
        "prints its own `--help`.",
        "",
        "| Command | What it does |",
        "|---|---|",
    ]
    for path, command in hidden_commands:
        same = callbacks.get(function(command))
        help_text = _short_help(command)
        if same:
            help_text = f"The same as `{same}`. {help_text}".strip()
        out.append(f"| `{_command_name(path)}`{labels.command(path)} | {help_text} |")
    return "\n".join(out) + "\n"


# --- python -----------------------------------------------------------------


def _summary(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    first = re.split(r"\n\s*\n", doc.strip(), maxsplit=1)[0] if doc.strip() else ""
    # Sphinx roles (:meth:`run`) read as plain code here.
    first = re.sub(
        r":(?:meth|func|class|attr|mod|data|exc|obj):`~?([^`]+)`", r"`\1`", first
    )
    return _text(" ".join(first.split()))


def _clean(text: str) -> str:
    # Python 3.13 names pathlib's classes from pathlib._local; the page is
    # the same on both CI Pythons.
    return re.sub(r" at 0x[0-9a-fA-F]+", "", text).replace(
        "pathlib._local.", "pathlib."
    )


def _fields(obj) -> list[tuple[str, str, str]] | None:
    import dataclasses

    if dataclasses.is_dataclass(obj):
        rows = []
        for f in dataclasses.fields(obj):
            if f.name.startswith("_"):
                continue
            if f.default is not dataclasses.MISSING:
                default = repr(f.default)
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                default = "(computed)"
            else:
                default = "required"
            rows.append((f.name, str(f.type), default))
        return rows
    model_fields = getattr(obj, "model_fields", None)
    if isinstance(model_fields, dict):
        rows = []
        for name, f in model_fields.items():
            if name.startswith("_"):
                continue
            annotation = getattr(f, "annotation", None)
            kind = getattr(annotation, "__name__", None) or str(annotation)
            if f.is_required():
                default = "required"
            elif f.default_factory is not None:
                default = "(computed)"
            else:
                default = repr(f.default)
            rows.append((name, kind, default))
        return rows
    return None


def _methods(cls: type) -> list[tuple[str, str | None, str]]:
    """A class's public methods and properties that BenchFlow defines: each
    name, its signature (None for a property) and its summary."""
    out: list[tuple[str, str | None, str]] = []
    for name in sorted(dir(cls)):
        if name.startswith("_"):
            continue
        owner = next((k for k in cls.__mro__ if name in vars(k)), None)
        if owner is None or not owner.__module__.startswith("benchflow"):
            continue
        raw = vars(owner)[name]
        if isinstance(raw, property):
            out.append((name, None, _summary(raw)))
            continue
        func = raw.__func__ if isinstance(raw, (classmethod, staticmethod)) else raw
        if not inspect.isfunction(func):
            continue
        try:
            signature = inspect.signature(getattr(cls, name))
        except (TypeError, ValueError):
            out.append((name, "(...)", _summary(func)))
            continue
        params = list(signature.parameters.values())
        if params and params[0].name == "self":
            signature = signature.replace(parameters=params[1:])
        summary = _summary(func)
        if inspect.iscoroutinefunction(func):
            summary = f"Async: `await` it. {summary}".strip()
        out.append((name, str(signature), summary))
    return out


def _python_page() -> str:
    import benchflow as bf

    since = _Since()
    out = [
        "# Python API",
        "",
        "Each public name in the `benchflow` package (`benchflow.__all__`) has its signature "
        "or fields, the methods BenchFlow defines on it, and the first paragraph of "
        "its docstring. A name, field, argument or method new "
        f"since {BASELINE_RELEASE} has `({NEW_LABEL})` after it when {PRE_RELEASE} has it, "
        f"or `({NEXT_LABEL})` when no release has it yet.",
        "",
        _GENERATED.format(
            source="docstring or signature in the code", regenerate=REGENERATE
        ),
        "",
        "Import the package as `import benchflow as bf`. [Run a benchmark](../evaluate/run-a-benchmark.md) "
        "and [Read the results](../evaluate/read-results.md) have worked examples, and "
        "[CLI and Python](./cli-python.md) gives the Python argument of each `bench` flag.",
    ]
    groups: dict[str, list[tuple[str, object]]] = {
        "Functions": [],
        "Classes": [],
        "Values": [],
    }
    for name in bf.__all__:
        obj = getattr(bf, name)
        if inspect.isclass(obj):
            groups["Classes"].append((name, obj))
        elif callable(obj):
            groups["Functions"].append((name, obj))
        else:
            groups["Values"].append((name, obj))
    for title, items in groups.items():
        out += ["", f"## {title}"]
        for name, obj in sorted(items, key=lambda item: item[0].lower()):
            # The name as code: `__version__` would otherwise render as a bold "version".
            out += ["", f"### `{name}`{since.python(name)}", ""]
            summary = _summary(obj) if title != "Values" else ""
            if title == "Values":
                value = (
                    repr(obj)
                    if isinstance(obj, (str, int, float, bool))
                    else type(obj).__name__
                )
                out += [f"`bf.{name}`: {_code(_clean(value))}"]
                continue
            if title == "Functions":
                try:
                    signature = inspect.signature(cast(Callable[..., Any], obj))
                    sig, params = str(signature), list(signature.parameters)
                except (TypeError, ValueError):
                    sig, params = "(...)", []
                out += ["```python", _clean(f"bf.{name}{sig}"), "```"]
                if inspect.iscoroutinefunction(obj):
                    out += [
                        "",
                        "A coroutine function: `await` it, or run it with `asyncio.run(...)`.",
                    ]
                out += _new_arguments(since, name, params)
                if summary:
                    out += ["", summary]
                continue
            cls = cast("type", obj)
            out += [f"Defined in `{cls.__module__}`."]
            if summary:
                out += ["", summary]
            rows = _fields(cls)
            if rows:
                out += ["", "| Field | Type | Default |", "|---|---|---|"]
                out += [
                    f"| `{n}`{since.python(name, n)} | {_code(_clean(t))} | {_code(_clean(d))}"
                    + (
                        f" ({BASELINE_RELEASE}: `{_RELEASE_DEFAULTS[name, n]}`)"
                        if (name, n) in _RELEASE_DEFAULTS
                        else ""
                    )
                    + " |"
                    for n, t, d in rows
                ]
            elif not issubclass(cls, BaseException):
                try:
                    signature = inspect.signature(cls)
                except (TypeError, ValueError):
                    signature = None
                if signature is not None:
                    out += ["", "```python", _clean(f"bf.{name}{signature}"), "```"]
                    out += _new_arguments(since, name, signature.parameters)
                    notes = _release_default_lines(name, signature.parameters)
                    if notes:
                        out += ["", " ".join(notes)]
            members = _methods(cls)
            if members:
                out += [""]
                for member, sig, text in members:
                    shown = f"{member}{sig}" if sig is not None else member
                    label = since.method(name, member)
                    tail = f": {text}" if text else ""
                    out.append(f"- `.{_clean(shown)}`{label}{tail}")
    return "\n".join(out) + "\n"


# --- cli and python -----------------------------------------------------------


def python_in_note(line: str) -> str:
    """A flag's table row with the Python its note writes (``Evaluation(budget=Budget(...))``) as code."""
    if not re.match(r"\| `(--[^`]+)` \|", line):
        return line
    cells = line.split(" | ")
    cells[-1] = re.sub(
        r"(?<![`\w])([A-Za-z_][\w.]*\([^\s|`]*\)[\w.()]*)", r"`\1`", cells[-1]
    )
    return " | ".join(cells)


def _cli_python_page() -> str:
    """``benchflow.cli_parity``'s tables, under this page's title and intro."""
    from benchflow import cli_parity

    commands = ", ".join(f"`{c}`" for c in cli_parity.TABLES)
    text = cli_parity.markdown()
    # markdown() is a title, its own intro, then one section per command.
    first = text.index("\n## ")
    labels = _labels()
    lines, command = [], None
    for line in text[first:].split("\n"):
        # The Python a note writes is set as code before a label can shift the row.
        line = python_in_note(line)
        heading = re.match(r"## `([^`]+)`$", line)
        row = re.match(r"\| `(--[^`]+)` \|", line)
        if heading:
            command = heading.group(1)
            line += _tier(command in labels.baseline, command in labels.prerelease)
        elif row and command is not None:
            flag = row.group(1)
            if command in labels.baseline:
                label = labels.flag(command, {flag})
            else:
                # A new command is labelled itself; only a flag added after
                # the prerelease that had the command is labelled again.
                pre = labels.prerelease.get(command)
                label = "" if pre is None or flag in pre else f" ({NEXT_LABEL})"
            if label:
                line = line.replace(f"`{flag}`", f"`{flag}`{label}", 1)
        lines.append(line)
    return "\n".join(
        [
            "# CLI and Python",
            "",
            f"Each flag of {commands}, and the Python that does the same. "
            "`sdk` is a name in the public `benchflow` package, `module` a function or "
            "field in one of its modules, and `cli` a flag with no Python equivalent, "
            "for the reason given. `bench run` takes the flags of `bench eval run`, some "
            "under shorter names: [Commands](./commands.md) has both, and "
            "[Python API](./python.md) every public name. A command or flag new since "
            f"{BASELINE_RELEASE} has `({NEW_LABEL})` after it when {PRE_RELEASE} has it, "
            f"or `({NEXT_LABEL})` when no release has it yet.",
            "",
            "<!-- This page is generated from src/benchflow/cli_parity.py by "
            "tools/docs_reference.py; tests/test_cli_python_parity.py fails when a "
            "flag has no row or a Python target stops existing. To change it, change "
            f"the table there, then run `{REGENERATE}`. -->",
            "",
        ]
    ) + "\n".join(lines)


# --- files --------------------------------------------------------------------

# The pieces of a task.md header that are not TaskConfig fields: the parser
# reads them itself (benchflow.task._document_parse), into these types.
_DOCUMENT_KEYS = ("agents", "scenes", "user", "benchflow")
_SCALARS = {
    "str": "text",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "Path": "path",
    "Any": "any",
}


def _yaml_value(value: Any) -> str:
    """A default as the YAML a task author writes."""
    import enum

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, enum.Enum):
        value = value.value
    if isinstance(value, str):
        return value if value else '""'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_yaml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return json.dumps(value) if value else "{}"
    return str(value)


def _models_in(annotation: Any) -> list[tuple[type, bool]]:
    """The pydantic models an annotation holds, each with whether in a list."""
    from typing import get_args, get_origin

    from pydantic import BaseModel

    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return [(annotation, False)]
    found: list[tuple[type, bool]] = []
    many = get_origin(annotation) in (list, tuple, set, frozenset)
    for arg in get_args(annotation):
        found += [(model, many or inner) for model, inner in _models_in(arg)]
    return found


_PLURALS = {
    "table": "tables",
    "turn": "turns",
    "role name": "role names",
    "number": "numbers",
    "integer": "integers",
    "boolean": "booleans",
    "path": "paths",
    "any": "values",
}


def _plural(text: str) -> str:
    """``table ([...])`` as ``tables ([...])``: each word before the first
    bracket or comma, in the plural."""
    words = re.split(r"[(\[,]", text, maxsplit=1)[0]
    rest = text[len(words) :]
    plural = " or ".join(
        _PLURALS.get(w.strip(), w.strip()) for w in words.split(" or ")
    )
    return plural + (" " if rest and words.endswith(" ") else "") + rest


def _annotation_text(
    annotation: Any,
    link: Callable[[type], str],
    names: dict[type, str] | None = None,
) -> str:
    """An annotation in a task author's words; ``link`` names a model's
    section, ``names`` what another class is called."""
    import enum
    import types
    from typing import Literal, Union, get_args, get_origin

    from pydantic import BaseModel

    def inner(arg: Any) -> str:
        return _annotation_text(arg, link, names)

    origin, args = get_origin(annotation), get_args(annotation)
    if annotation is type(None):
        return "null"
    if names and annotation in names:
        return names[annotation]
    if origin in (Union, types.UnionType):
        return " or ".join(inner(a) for a in args if a is not type(None))
    if origin is Literal:
        return " \\| ".join(_code(str(a)) for a in args)
    if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
        return " \\| ".join(_code(str(m.value)) for m in annotation)
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return link(annotation)
    if origin is tuple and args and args[-1] is not Ellipsis:
        return "[" + ", ".join(inner(a) for a in args) + "]"
    if origin in (list, tuple, set, frozenset):
        return "list of " + (_plural(inner(args[0])) if args else "values")
    if origin is dict:
        if len(args) == 2 and args[1] is not Any:
            return "mapping of names to " + _plural(inner(args[1]))
        return "mapping"
    name = getattr(annotation, "__name__", None) or str(annotation)
    return _SCALARS.get(name, name)


def _constraints(field) -> str:
    """Bounds pydantic enforces on a field (``> 0``, ``>= 0``, a length)."""
    words = []
    for item in getattr(field, "metadata", []):
        for attr, sign in (("gt", ">"), ("ge", ">="), ("lt", "<"), ("le", "<=")):
            value = getattr(item, attr, None)
            if value is not None:
                words.append(f"{sign} {value}")
        if getattr(item, "min_length", None):
            words.append(f"at least {item.min_length} long")
        if getattr(item, "max_length", None):
            words.append(f"at most {item.max_length} long")
    return f" ({', '.join(words)})" if words else ""


def _field_key(name: str, field) -> str:
    return field.serialization_alias or field.alias or name


def _other_keys(name: str, field, key: str) -> list[str]:
    from pydantic import AliasChoices

    alias = field.validation_alias
    choices = alias.choices if isinstance(alias, AliasChoices) else [alias]
    return [c for c in choices if isinstance(c, str) and c != key]


def _field_default(field) -> str:
    from pydantic import BaseModel

    if field.is_required():
        return "required"
    if field.default_factory is not None:
        try:
            value = field.default_factory()
        except TypeError:
            return ""
        return "" if isinstance(value, BaseModel) else _code(_yaml_value(value))
    if field.default is None:
        return ""
    return _code(_yaml_value(field.default))


def _model_sections(root: type) -> tuple[list[tuple[str, type]], dict[type, str]]:
    """Each model under ``root`` and the key path it is described at: the
    shortest path to it (``sandbox``, not ``verifier.sandbox``), in the order
    a reader meets them (depth first)."""
    canonical: dict[type, str] = {}
    queue: deque[tuple[str, type]] = deque([("", root)])
    while queue:
        prefix, model = queue.popleft()
        for name, field in model.model_fields.items():  # ty: ignore[unresolved-attribute]
            key = _field_key(name, field)
            for sub, many in _models_in(field.annotation):
                if sub in canonical or sub is root:
                    continue
                path = (f"{prefix}.{key}" if prefix else key) + ("[]" if many else "")
                canonical[sub] = path
                queue.append((path, sub))
    order: list[tuple[str, type]] = []

    def visit(prefix: str, model: type) -> None:
        for name, field in model.model_fields.items():  # ty: ignore[unresolved-attribute]
            key = _field_key(name, field)
            for sub, many in _models_in(field.annotation):
                path = (f"{prefix}.{key}" if prefix else key) + ("[]" if many else "")
                if canonical.get(sub) == path and (path, sub) not in order:
                    order.append((path, sub))
                    visit(path, sub)

    visit("", root)
    return order, canonical


def _section_heading(path: str) -> str:
    return f"#### `{path}`"


# What a release reads but does not run: said beside the key, since the
# key itself is not new. Checked by running a multi-step task on 0.9.2 and
# 0.10.0rc2, and by checking a task.md that sets `verifier.type` on 0.9.2.
_RUNS_FROM = {
    ("", "steps"): "0.9.2 reads this key but refuses to run a task that has it "
    "(`multi-step tasks are not supported`); multi-step tasks run on 0.10.0rc2.",
    ("", "multi_step_reward_strategy"): "Used only by a task with `steps`, which "
    "runs on 0.10.0rc2.",
    ("verifier", "type"): "In `task.md`, 0.9.2's `bench check` and `bench run` refuse "
    "`llm-judge` (no runnable verifier entrypoint); a verifier document grades with a model "
    "instead ([LLM judge](../tasks/llm-judge.md)).",
    ("verifier", "judge"): "Read only with `type: llm-judge`, which 0.9.2 refuses in "
    "`task.md` ([LLM judge](../tasks/llm-judge.md)).",
    ("verifier.judge", "model"): "0.9.2 defaults to `claude-sonnet-4-6`.",
}


def _model_rows(
    model: type, here: str, canonical: dict[type, str], since: _Since
) -> list[str]:
    def link(sub: type) -> str:
        path = canonical[sub]
        target = f"[`{path}`](#{_slug(_section_heading(path).lstrip('# '))})"
        if not here or path.startswith(f"{here}.") or path.startswith(f"{here}["):
            return f"table ({target})"
        return f"table, the keys of {target}"

    rows = []
    for name, field in model.model_fields.items():  # ty: ignore[unresolved-attribute]
        key = _field_key(name, field)
        kind = _annotation_text(field.annotation, link) + _constraints(field)
        description = _one_line(field.description)
        if (here, key) in _RUNS_FROM:
            description = f"{description} {_RUNS_FROM[here, key]}".strip()
        others = _other_keys(name, field, key)
        if others:
            also = ", ".join(_code(o) for o in others)
            description = f"{description} Also read as {also}.".strip()
        label = since.key("task_md", key, here)
        rows.append(
            f"| `{key}`{label} | {kind} | {_field_default(field)} | {description} |"
        )
    return rows


def _trailing_comments(cls: type) -> dict[str, str]:
    """A dataclass's field comments (``timeout_sec: int | None = None  # ...``)."""
    import io
    import tokenize

    source = textwrap.dedent(inspect.getsource(cls))
    comments: dict[int, str] = {}
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            comments[token.start[0]] = token.string.lstrip("#").strip()
    out = {}
    definition = ast.parse(source).body[0]
    assert isinstance(definition, ast.ClassDef), cls
    for node in definition.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.lineno in comments
        ):
            out[node.target.id] = comments[node.lineno]
    return out


def _dataclass_rows(
    cls: type, here: str, since: _Since, *, skip: tuple[str, ...] = ()
) -> list[str]:
    """A parsed task.md block's keys: the fields of the type the parser fills."""
    import dataclasses
    import typing

    from benchflow._types import Role, Turn

    hints = typing.get_type_hints(cls)
    comments = _trailing_comments(cls)
    # The parser takes a scene's roles by name, and each turn as a table (or
    # the name of its role).
    names: dict[type, str] = {Role: "role name", Turn: "turn"}

    def kind(annotation: Any) -> str:
        return _annotation_text(annotation, lambda _model: "table", names)

    rows = []
    for f in dataclasses.fields(cls):
        if f.name in skip:
            continue
        required = (
            f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING  # type: ignore[misc]
        )
        note = _one_line(comments.get(f.name, ""))
        if required:
            note = f"Required. {note}".strip()
        label = since.key("task_md", f.name, here)
        rows.append(f"| `{f.name}`{label} | {kind(hints[f.name])} | {note} |")
    return rows


def _document_keys_read(
    block: str, package: Path = REPO / "src" / "benchflow"
) -> list[str]:
    """The keys BenchFlow reads from a task.md block it keeps as a mapping
    (``document.user.get("model")``), from the source."""
    found: set[str] = set()
    for path in sorted(package.rglob("*.py")):
        if "_vendor" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == block
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "document"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                found.add(node.args[0].value)
    return sorted(found)


def _task_md_part() -> list[str]:
    from benchflow._types import Role, Scene, Turn
    from benchflow.task._document_parse import _DOCUMENT_ONLY_FRONTMATTER_KEYS
    from benchflow.task.config import TaskConfig

    assert set(_DOCUMENT_KEYS) == set(_DOCUMENT_ONLY_FRONTMATTER_KEYS), (
        "the task.md parser reads other keys itself now; describe them here",
        sorted(_DOCUMENT_ONLY_FRONTMATTER_KEYS),
    )
    sections, canonical = _model_sections(TaskConfig)
    since = _Since()

    def anchor(path: str) -> str:
        return "#" + _slug(_section_heading(path).lstrip("# "))

    turns = "scenes[].turns[]"
    document_rows = {
        "agents": (
            f"table: `roles`, each role by name ([`agents.roles.*`]({anchor('agents.roles.*')}))",
            "A multi-turn task's roles: which agent plays each.",
        ),
        "scenes": (
            f"list of tables ([`scenes[]`]({anchor('scenes[]')}))",
            "A multi-turn task's scenes, run in order; each has its turns.",
        ),
        "user": (
            f"table ([`user`]({anchor('user')}))",
            "A simulated user that answers the agent; its persona is the body's "
            "`## user-persona` section.",
        ),
        "benchflow": (
            f"table ([`benchflow`]({anchor('benchflow')}))",
            "BenchFlow's own settings, read by the features that use them.",
        ),
    }
    out = [
        "### task.md",
        "",
        "The header is YAML between two `---` lines at the top of `task.md`; the "
        "instruction is below it. Each table below is one key's settings, with the type, "
        "default and description the code gives them. A key is optional unless the table "
        "says required, and a key the header doesn't know is refused. A key new since "
        f"{BASELINE_RELEASE} has `({NEW_LABEL})` after it when {PRE_RELEASE} has it, "
        f"or `({NEXT_LABEL})` when no release has it yet.",
        "",
        "| Key | Type | Default | Description |",
        "|---|---|---|---|",
        *_model_rows(TaskConfig, "", canonical, since),
        *(
            f"| `{key}`{since.key('task_md', key, '')} | {kind} |  | {description} |"
            for key, (kind, description) in document_rows.items()
        ),
    ]
    for path, model in sections:
        out += [
            "",
            _section_heading(path),
            "",
            "| Key | Type | Default | Description |",
            "|---|---|---|---|",
            *_model_rows(model, path, canonical, since),
        ]
    out += [
        "",
        _section_heading("agents.roles.*"),
        "",
        "Each role of `agents.roles`, under its name.",
        "",
        "| Key | Type | Description |",
        "|---|---|---|",
        *_dataclass_rows(Role, "agents.roles.*", since, skip=("name",)),
        "",
        _section_heading("scenes[]"),
        "",
        "| Key | Type | Description |",
        "|---|---|---|",
        *_dataclass_rows(Scene, "scenes[]", since),
        "",
        _section_heading(turns),
        "",
        "A turn is a table, or the name of the role that takes it.",
        "",
        "| Key | Type | Description |",
        "|---|---|---|",
        *_dataclass_rows(Turn, turns, since),
    ]
    for block in ("user", "benchflow"):
        keys = ", ".join(
            f"`{k}`{since.key('task_md', k, block)}" for k in _document_keys_read(block)
        )
        out += ["", _section_heading(block), "", f"The keys BenchFlow reads: {keys}."]
    return out


class _Loader:
    """What ``Evaluation._from_native_yaml`` does with each run config key,
    read from its source: the key's default there, the setting it is passed
    to (``EvaluationConfig.concurrency``, ``Evaluation.__init__(jobs_dir)``,
    ``RetryConfig.max_retries``), and the class a key holds a table of
    (``Budget.coerce(raw.get("budget"))``). Only a key read in the argument
    itself counts, so a value computed from several keys names none."""

    def __init__(self) -> None:
        import benchflow.evaluation as ev

        self.module = ev
        source = textwrap.dedent(inspect.getsource(ev.Evaluation._from_native_yaml))
        self.tree = ast.parse(source)
        self.parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(self.tree):
            for child in ast.iter_child_nodes(node):
                self.parents[child] = node
        self.defaults: dict[str, Any] = {}
        self.targets: dict[str, set[str]] = {}
        self.tables: dict[str, type] = {}
        for node in ast.walk(self.tree):
            key = self._key(node)
            if key is not None and key not in self.defaults:
                try:
                    default = self._default(node)
                except (ValueError, SyntaxError):
                    default = None
                if default is not None:
                    self.defaults[key] = default
            if isinstance(node, ast.Call):
                self._call(node)

    @staticmethod
    def _key(node: ast.AST) -> str | None:
        """The key of ``raw.get("key", ...)`` or ``raw["key"]``."""
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "raw"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            return node.args[0].value
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "raw"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            return node.slice.value
        return None

    def _value(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Name) and hasattr(self.module, node.id):
            return getattr(self.module, node.id)
        if self._key(node) is not None:
            return self._default(node)
        return ast.literal_eval(node)

    def _default(self, get: ast.AST) -> Any:
        if isinstance(get, ast.Call) and len(get.args) > 1:
            return self._value(get.args[1])
        parent = self.parents.get(get)
        # raw.get("keep_sandbox") or "never"
        if (
            isinstance(parent, ast.BoolOp)
            and isinstance(parent.op, ast.Or)
            and parent.values[-1] is not get
        ):
            return self._value(parent.values[-1])
        return None

    def _keys_in(self, node: ast.AST) -> set[str]:
        keys = {self._key(sub) for sub in ast.walk(node)}
        return {key for key in keys if key is not None}

    def _call(self, call: ast.Call) -> None:
        func = call.func
        if isinstance(func, ast.Name) and func.id == "cls":
            setting = "benchflow.Evaluation.__init__({})"
        elif isinstance(func, ast.Name) and inspect.isclass(
            getattr(self.module, func.id, None)
        ):
            setting = f"benchflow.{func.id}.{{}}"
        elif (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and inspect.isclass(getattr(self.module, func.value.id, None))
        ):
            # Budget.coerce(raw.get("budget")): the key is a table of Budget.
            for arg in call.args:
                key = self._key(arg)
                if key is not None:
                    self.tables[key] = getattr(self.module, func.value.id)
            return
        else:
            return
        for keyword in call.keywords:
            if keyword.arg is None:
                continue
            target = setting.format(keyword.arg)
            keys = self._keys_in(keyword.value)
            if isinstance(keyword.value, ast.Name):
                # prompts=prompts: the key the argument is named for.
                keys.add(keyword.arg)
            for key in keys:
                self.targets.setdefault(key, set()).add(target)


def _table_keys(cls: type) -> list[str]:
    import dataclasses

    if dataclasses.is_dataclass(cls):
        return [f.name for f in dataclasses.fields(cls) if not f.name.startswith("_")]
    return [
        name for name in getattr(cls, "model_fields", {}) if not name.startswith("_")
    ]


def _eval_run_flags() -> dict[str, click.Option]:
    """``bench eval run``'s options, by their long name."""
    import click

    root = _root_command()
    group = cast("click.Group", root).get_command(click.Context(root), "eval")
    assert group is not None
    command = cast("click.Group", group).get_command(click.Context(group), "run")
    assert command is not None
    flags = {}
    for param in command.params:
        if isinstance(param, click.Option):
            long = next((o for o in param.opts if o.startswith("--")), param.opts[0])
            flags[long] = param
    return flags


def _run_config_part() -> list[str]:
    from benchflow import cli_parity
    from benchflow.evaluation import _SOURCE_KEYS, RUN_CONFIG_KEYS
    from benchflow.usage_tracking import _LEGACY_USAGE_PROXY_KEYS

    flags = _eval_run_flags()
    by_name = {p.name: long for long, p in flags.items()}
    # Each bench eval run flag by the Python setting it stands for.
    by_target: dict[str, list[str]] = {}
    for flag, eq in cli_parity.TABLES["bench eval run"].items():
        if flag in flags and eq.target:
            by_target.setdefault(eq.target, []).append(flag)
    loader = _Loader()
    since = _Since()
    run_flags = json.loads(BASELINE.read_text(encoding="utf-8")).get("bench eval run")
    pre_flags = json.loads(PRE_BASELINE.read_text(encoding="utf-8")).get(
        "bench eval run"
    )

    def new_flag(flag: str) -> str:
        return since._label(run_flags, pre_flags, flag)

    def flags_for(key: str) -> list[str]:
        """The flag that sets what ``key`` sets: the one named for it, else
        the one for the setting the loader passes it to, else every flag of
        the class it is a table of."""
        if key in by_name:
            return [by_name[key]]
        found = {
            flag
            for target in loader.targets.get(key, ())
            for flag in by_target.get(target, ())
        }
        if not found and key in loader.tables:
            prefix = f"benchflow.{loader.tables[key].__name__}."
            found = {
                flag
                for target, names in by_target.items()
                if target.startswith(prefix)
                for flag in names
            }
        return sorted(found, key=list(flags).index)

    rows = []
    for key in sorted(RUN_CONFIG_KEYS):
        names = flags_for(key)
        default = loader.defaults.get(key)
        cell = _code(_yaml_value(default)) if default is not None else ""
        if key == "source":
            # A table of the --source-repo, --source-path, --source-ref values.
            names = [
                f"--source-{k}"
                for k in sorted(_SOURCE_KEYS)
                if f"--source-{k}" in flags
            ]
            description = (
                "A GitHub folder, a table of "
                + ", ".join(f"`{k}`" for k in sorted(_SOURCE_KEYS))
                + "."
            )
        elif key in loader.tables:
            description = (
                "A table of "
                + ", ".join(f"`{k}`" for k in _table_keys(loader.tables[key]))
                + "."
            )
        elif key in _LEGACY_USAGE_PROXY_KEYS:
            # UsageTrackingConfig.from_mapping refuses them by name.
            description = "No longer supported: a config with it is refused. Use `usage_tracking`."
        elif len(names) == 1:
            description = _one_line(flags[names[0]].help)
        else:
            description = ""
        flag_cell = ", ".join(_code(n) + new_flag(n) for n in names)
        label = since.key("run_config", key)
        rows.append(f"| `{key}`{label} | {flag_cell} | {cell} | {description} |")
    return [
        "### Run config",
        "",
        "`bench run --config FILE` reads a whole run from a YAML file of these keys. "
        "It needs `tasks_dir` or `source`; a flag given beside it wins over its key; "
        "`sandbox` is read as `environment`; and a key not in this table is refused, "
        "with the closest one. The flag column names the flag that sets the same "
        "thing, and the help is that flag's. A key new since "
        f"{BASELINE_RELEASE} has `({NEW_LABEL})` after it when {PRE_RELEASE} has it, "
        f"or `({NEXT_LABEL})` when no release has it yet.",
        "",
        "| Key | Flag | Default | Help |",
        "|---|---|---|---|",
        *rows,
    ]


def _job_config_part() -> list[str]:
    from benchflow import job_config
    from benchflow.evaluation import LEGACY_RUN_CONFIG_KEYS

    refused = job_config._JOB_ADVICE
    keys = sorted((LEGACY_RUN_CONFIG_KEYS | job_config.JOB_KEYS) - set(refused))
    since = _Since()

    def listed(part: str, names: Iterator[str] | list[str]) -> str:
        return ", ".join(f"`{k}`{since.key(part, k)}" for k in names)

    return [
        "### YAML job config",
        "",
        "A run config with `agents:` and `datasets:` (or a `tasks:` list) is a YAML job "
        "config: it runs every agent on every dataset, `n_attempts` times each. Its "
        "top-level keys: " + listed("job_config", keys) + ". "
        "An agent entry takes "
        + listed("job_agent", sorted(job_config._AGENT_HONOURED))
        + "; a dataset entry "
        + listed("job_dataset", sorted(job_config._DATASET_HONOURED))
        + ". A setting BenchFlow can't honour exactly is refused when the file loads, "
        "unless it holds its default:",
        "",
        "| Key | Instead |",
        "|---|---|",
        *(f"| `{key}` | {_one_line(advice)} |" for key, advice in refused.items()),
    ]


def _schema_type(node: dict) -> str:
    if "$ref" in node:
        return "object (" + node["$ref"].rsplit("/", 1)[-1] + ")"
    kind = node.get("type")
    if isinstance(kind, list):
        return " or ".join(kind)
    if kind == "array":
        items = node.get("items")
        return "array of " + (
            _schema_type(items) if isinstance(items, dict) else "values"
        )
    if kind:
        return str(kind)
    for key in ("anyOf", "oneOf"):
        if key in node:
            return " or ".join(
                _schema_type(n) for n in node[key] if isinstance(n, dict)
            )
    if "const" in node:
        return "constant " + json.dumps(node["const"])
    if "enum" in node:
        return "one of " + ", ".join(json.dumps(v) for v in node["enum"])
    return "any"


def _schemas_part() -> list[str]:
    since = _Since()
    out = [
        "## Files BenchFlow writes",
        "",
        "The JSON documents BenchFlow writes, one section per schema file in "
        "[`schemas/`](./schemas/), with each document's top-level fields. A schema or "
        f"field new since {BASELINE_RELEASE} has `({NEW_LABEL})` after it when {PRE_RELEASE} "
        f"has it, or `({NEXT_LABEL})` when no release has it yet.",
    ]
    for path in sorted(SCHEMAS.glob("*.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        title = schema.get("title") or path.stem
        out += [
            "",
            f"### {path.name}{since.schema(path.name)}",
            "",
            f"**{_text(str(title))}**",
        ]
        if schema.get("description"):
            out += ["", _one_line(schema["description"])]
        if schema.get("$id"):
            out += [
                "",
                f"Identifier: `{schema['$id']}`. File: [`schemas/{path.name}`](./schemas/{path.name}).",
            ]
        props = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        if props:
            out += [
                "",
                "| Field | Type | Required | Description |",
                "|---|---|---|---|",
            ]
            for name, node in props.items():
                node = node if isinstance(node, dict) else {}
                out.append(
                    f"| `{name}`{since.schema(path.name, name)} | {_one_line(_schema_type(node))} "
                    f"| {'yes' if name in required else 'no'} "
                    f"| {_one_line(node.get('description', ''))} |"
                )
    return out


def _files_page() -> str:
    out = [
        "# Files BenchFlow reads and writes",
        "",
        "You write two kinds of file for BenchFlow, a task's `task.md` header and a run "
        "config; every key they take is here, with its type and default. BenchFlow writes "
        "JSON documents for each job and trial; the ones with a schema file are here, with "
        "their top-level fields. Every file in a trial's folder, and every key of its "
        "`result.json`, is in "
        "[Read the results](../evaluate/read-results.md#everything-in-a-trials-folder).",
        "",
        _GENERATED.format(
            source="field, flag help or schema file it comes from",
            regenerate=REGENERATE,
        ),
        "",
        "## Files you write",
        "",
        *_task_md_part(),
        "",
        *_run_config_part(),
        "",
        *_job_config_part(),
        "",
        *_schemas_part(),
    ]
    return "\n".join(out) + "\n"


# --- changelog ------------------------------------------------------------------


def _redirected(target: str) -> str | None:
    """Where the old docs page ``target`` (``docs/jobs/resume.md``) moved, from
    the repository root and with the anchor ``docs/redirects.json`` gives it;
    None if it was no docs page. A ``github:`` page stayed in ``docs/``."""
    if not (target.startswith("docs/") and target.endswith(".md")):
        return None
    moved = json.loads(REDIRECTS.read_text(encoding="utf-8"))["redirects"].get(
        target.removeprefix("docs/").removesuffix(".md")
    )
    if moved is None:
        return None
    page, _, anchor = moved.removeprefix("github:").partition("#")
    return f"docs/{page}.md" + (f"#{anchor}" if anchor else "")


def _moved_link(match: re.Match[str], page: Path) -> str:
    """A link written from the repository root (``docs/start/install.md``),
    from ``page``'s folder. A link to an old docs page goes to the page it
    moved to, at that page's anchor in ``docs/redirects.json`` (the old
    page's own anchors name its sections, not the new page's); a link to a
    file the repository doesn't have becomes its text."""
    text, target = match.group("text"), match.group("target")
    fragment = match.group("fragment") or ""
    if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I):
        return match.group(0)
    if not (REPO / target).exists():
        moved = _redirected(target)
        if moved is None:
            return text[1:-1]
        target, _, anchor = moved.partition("#")
        fragment = f"#{anchor}" if anchor else ""
    relative = os.path.relpath(REPO / target, page.parent)
    return f"{text}({Path(relative).as_posix()}{fragment})"


# A code span, or a link whose text may hold code spans (``[`docs/x.md`](...)``).
_CODE_OR_LINK = re.compile(
    r"(?P<code>`+[^`]*?`+)"
    r"|(?P<text>\[(?:[^\[\]`]|`[^`]*`)*\])\((?P<target>[^)\s#]+)(?P<fragment>#[^)\s]*)?\)"
)


def _mdx_safe(text: str, page: Path) -> str:
    """``<``, ``>``, ``{`` and ``}`` escaped outside code (fenced blocks and
    spans, which may wrap lines), so the site's MDX build reads them as text;
    and links written from the repository root rewritten from ``page``."""
    out, fence, prose = [], None, []

    def escape(part: str) -> str:
        return (
            part.replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("{", "&#123;")
            .replace("}", "&#125;")
        )

    def flush() -> None:
        joined = _CODE_OR_LINK.sub(
            lambda m: m.group(0) if m.group("code") else _moved_link(m, page),
            "\n".join(prose),
        )
        parts = re.split(r"(`+[^`]*?`+)", joined)
        out.extend(
            "".join(
                part if i % 2 else escape(part) for i, part in enumerate(parts)
            ).split("\n")
        )
        prose.clear()

    for line in text.split("\n"):
        m = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence is None and m:
            flush()
            fence = m.group(1)
            out.append(line)
        elif fence is not None:
            out.append(line)
            if line.strip().startswith(fence):
                fence = None
        else:
            prose.append(line)
    flush()
    return "\n".join(out)


# The [0.10.0rc1] section's entries at the commit the 0.10.0rc1 wheel was built from (bold titles, one a line);
# the page marks the section's other entries, which were added after that wheel.
RC1_ENTRIES = REPO / "tools" / "changelog_0.10.0rc1_entries.txt"


def _section_above_rc1(lines: list[str]) -> tuple[int | None, str]:
    """The section shown right above [0.10.0rc1]: its heading's line and its name, or (None, "Unreleased")
    when [0.10.0rc1] is the newest section."""
    start = next(i for i, line in enumerate(lines) if line.startswith("## [0.10.0rc1]"))
    above = next(
        (i for i in range(start - 1, -1, -1) if lines[i].startswith("## [")), None
    )
    if above is None:
        return None, "Unreleased"
    return above, lines[above][len("## [") :].split("]", 1)[0]


def _move_after_rc1(body: str) -> str:
    """Show the [0.10.0rc1] entries the wheel's own changelog didn't have in the section above it, each under
    its subsection's heading: [Unreleased], or the next release candidate once CHANGELOG.md has one. The
    wheel's entries stay where they are. An entry's indented lines move with it."""
    known = [
        line.strip()
        for line in RC1_ENTRIES.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    def after(line: str) -> bool:
        m = re.match(r"^- \*\*(.+?)\*\*", line)
        return bool(m) and not any(
            difflib.SequenceMatcher(None, m.group(1).strip(), k).ratio() > 0.85
            for k in known
        )

    lines = body.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith("## [0.10.0rc1]"))
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")),
        len(lines),
    )
    kept: list[str] = []
    moved: dict[str, list[str]] = {}
    heading = ""
    i = start
    while i < end:
        line = lines[i]
        if line.startswith("### "):
            heading = line
        if line.startswith("- ") and after(line):
            block = [line]
            i += 1
            while i < end and lines[i].startswith("  "):
                block.append(lines[i])
                i += 1
            moved.setdefault(heading, []).extend(block)
            continue
        kept.append(line)
        i += 1
    if not moved:
        return body
    above, name = _section_above_rc1(lines)
    count = len([b for blk in moved.values() for b in blk if b.startswith("- ")])
    if name == "Unreleased":
        note = [
            "",
            f"{count} changes on the main line after the 0.10.0rc1 wheel, which CHANGELOG.md lists under "
            "`[0.10.0rc1]`: no release has them yet.",
        ]
        for head, block in moved.items():
            note += ["", head or "### Other", "", *block]
    else:
        # Under a release candidate's own sub-headings, the moved entries get one heading of their own.
        note = [
            "",
            "### After the 0.10.0rc1 wheel",
            "",
            f"{count} more changes came after the 0.10.0rc1 wheel, though CHANGELOG.md lists them under "
            "`[0.10.0rc1]`.",
        ]
        for head, block in moved.items():
            note += ["", "#" + (head or "### Other"), "", *block]
    # Drop sub-headings the move left empty in the rc1 section.
    out_section: list[str] = []
    for j, line in enumerate(kept):
        if line.startswith("### "):
            rest = kept[j + 1 :]
            nxt = next(
                (k for k, l2 in enumerate(rest) if l2.startswith("### ")), len(rest)
            )
            if not any(l2.strip() for l2 in rest[:nxt]):
                continue
        out_section.append(line)
    if above is None:
        # No section above [0.10.0rc1]: the moved entries get an [Unreleased] of their own.
        return "\n".join(
            [*lines[:start], "## [Unreleased]", *note, "", *out_section, *lines[end:]]
        )
    # The moved entries close the section above, after its own entries.
    before = lines[:start]
    while before and not before[-1].strip():
        before.pop()
    return "\n".join(before + note + [""] + out_section + lines[end:])


def _changelog_page() -> str:
    text = CHANGELOG.read_text(encoding="utf-8")
    lines = text.split("\n")
    assert lines[0].startswith("# "), "CHANGELOG.md starts with its title"
    body = _move_after_rc1("\n".join(lines[1:]).strip("\n"))
    _, above = _section_above_rc1(lines)
    where = (
        "under `[Unreleased]`, as no release has them yet"
        if above == "Unreleased"
        else f"at the end of `[{above}]`"
    )
    return (
        "# Changelog\n\n"
        f"What changed in each BenchFlow release: {BASELINE_RELEASE} is the current release, and "
        f"{PRE_RELEASE} the current release candidate. Under `[0.10.0rc1]` are the changes its wheel has; the "
        f"ones CHANGELOG.md also lists there but that came after the wheel are {where}. "
        f"[What's in {PRE_RELEASE}](./0.10.md) walks through what 0.10 adds.\n\n"
        "<!-- This page is generated from the repository's CHANGELOG.md by "
        "tools/docs_reference.py; to change it, change that file, then run "
        f"`{REGENERATE}`. -->\n\n" + _mdx_safe(body, PAGES["changelog"]) + "\n"
    )


# --- driver -----------------------------------------------------------------


def _render_here() -> dict[str, str]:
    return {
        "commands": _commands_page(),
        "python": _python_page(),
        "cli-python": _cli_python_page(),
        "files": _files_page(),
        "changelog": _changelog_page(),
    }


def _child(mode: str, extra: dict[str, str] | None = None) -> str:
    """Run this file in a fresh interpreter with a fixed environment."""
    with tempfile.TemporaryDirectory(prefix="bench-docs-reference-") as home:
        env = _child_env(home)
        env[_CHILD] = mode
        env.update(extra or {})
        env.pop("_BENCHFLOW_HELP_GOLDEN_CHILD", None)
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            env=env,
            cwd=home,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if proc.returncode != 0:
            raise SystemExit(
                f"rendering the reference failed ({proc.returncode}):\n{proc.stderr}"
            )
        return proc.stdout.replace(home, "~")


_DASH = re.compile(r"\s+[\u2014\u2013]\s+|\u2014")


def _undash_line(line: str) -> str:
    # A release heading's date goes in parentheses ("0.7.11 (2026-10-16)"): a semicolon
    # there read as two headings in the agents' index.
    dated = re.match(r"^(#{2,6} )(.+?) \u2014 (\d{4}-\d{2}-\d{2})\s*$", line)
    if dated:
        return f"{dated.group(1)}{dated.group(2)} ({dated.group(3)})"
    parts = re.split(r"(`[^`]*`)", line)
    for i in range(0, len(parts), 2):
        # Three dots in prose are an ellipsis; a quoted string that shows a symbol is
        # output, so it is code; an arrow in prose is the word "to". Code keeps all three.
        piece = parts[i].replace("...", "\u2026")
        piece = re.sub(
            r'"([^"`]*[\u2717\u2713\u2714\u2718\u2192][^"`]*)"', r"`\1`", piece
        )
        piece = re.sub(r"\s*\u2192\s*", " to ", piece)
        # A range written with an en dash (F1\u2013F6, #842\u2013#865) reads "F1 to F6".
        parts[i] = re.sub(r"(?<=[\w)])\u2013(?=[\w#(])", " to ", piece)
    if any(i % 2 == 0 and "`" in p for i, p in enumerate(parts)):
        parts = re.split(r"(`[^`]*`)", "".join(parts))
    hits = [(i, m) for i in range(0, len(parts), 2) for m in _DASH.finditer(parts[i])]
    if not hits:
        return "".join(parts)
    if len(hits) == 2:
        (i1, m1), (i2, m2) = hits
        if i1 == i2:
            between = parts[i1][m1.end() : m2.start()]
        else:
            between = (
                parts[i1][m1.end() :]
                + "".join(parts[i1 + 1 : i2])
                + parts[i2][: m2.start()]
            )
        if ". " not in between:
            # A pair around an aside: parentheses.
            tail = parts[i2][m2.end() :]
            close = ")" + ("" if not tail or tail[0] in ".,;:)" else " ")
            if i1 == i2:
                p = parts[i1]
                parts[i1] = (
                    p[: m1.start()]
                    + " ("
                    + p[m1.end() : m2.start()]
                    + close
                    + p[m2.end() :]
                )
            else:
                parts[i2] = parts[i2][: m2.start()] + close + tail
                parts[i1] = parts[i1][: m1.start()] + " (" + parts[i1][m1.end() :]
            return "".join(parts)
    for i in range(0, len(parts), 2):
        piece = parts[i]

        def mark(m: re.Match[str], i: int = i, piece: str = piece) -> str:
            before = piece[: m.start()]
            if not before.strip() and i:
                before = parts[i - 1]
            # After a term (a bold name or a code span) a colon; otherwise a semicolon.
            return ": " if before.rstrip().endswith(("**", "`")) else "; "

        parts[i] = _DASH.sub(mark, piece)
    return "".join(parts)


def _undash(page: str) -> str:
    """The pages' prose without em dashes, which read as machine-written: a pair
    around an aside becomes parentheses, one after a term a colon, any other a
    semicolon; three dots become an ellipsis. Code blocks, code spans and comments
    keep theirs."""
    lines, fence = [], False
    for line in page.split("\n"):
        if line.lstrip().startswith("```"):
            fence = not fence
        elif not fence and not line.lstrip().startswith("<!--"):
            line = _undash_line(line)
        lines.append(line)
    return "\n".join(lines)


def render() -> dict[str, str]:
    """Every page, rendered in a fresh interpreter with a fixed environment."""
    return {name: _undash(text) for name, text in json.loads(_child("pages")).items()}


def diff(name: str, expected: str, actual: str) -> str:
    return "".join(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile=str(PAGES[name].relative_to(REPO)),
            tofile=f"generated {name}",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    mode = os.environ.get(_CHILD)
    if mode == "inventory":
        sys.stdout.write(json.dumps(inventory(), indent=2) + "\n")
        return 0
    if mode == "names":
        names = names_inventory(Path(os.environ[_SCHEMAS_FROM]))
        sys.stdout.write(json.dumps(names, indent=2) + "\n")
        return 0
    if mode:
        sys.stdout.write(json.dumps(_render_here()))
        return 0
    parser = argparse.ArgumentParser(
        description="Write or check the generated reference pages."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="write the pages")
    group.add_argument("--check", action="store_true", help="exit 1 if a page is stale")
    group.add_argument(
        "--dump-inventory",
        metavar="PATH",
        type=Path,
        help="write this interpreter's CLI commands and option names as JSON",
    )
    group.add_argument(
        "--dump-names",
        metavar="PATH",
        type=Path,
        help="write this interpreter's public Python names, file keys and, with "
        "--schemas, schema fields as JSON",
    )
    parser.add_argument(
        "--schemas",
        metavar="DIR",
        type=Path,
        default=SCHEMAS,
        help="the schema files of the release --dump-names describes",
    )
    args = parser.parse_args(argv)
    if args.dump_inventory:
        args.dump_inventory.write_text(_child("inventory"), encoding="utf-8")
        return 0
    if args.dump_names:
        names = _child("names", {_SCHEMAS_FROM: str(args.schemas.resolve())})
        args.dump_names.write_text(names, encoding="utf-8")
        return 0
    pages = render()
    stale = []
    for name, text in pages.items():
        path = PAGES[name]
        if args.write:
            path.write_text(text, encoding="utf-8")
        elif not path.exists() or path.read_text(encoding="utf-8") != text:
            stale.append(name)
            print(
                diff(
                    name,
                    path.read_text(encoding="utf-8") if path.exists() else "",
                    text,
                )
            )
    if stale:
        print(f"stale: {', '.join(stale)}; regenerate with: {REGENERATE}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
