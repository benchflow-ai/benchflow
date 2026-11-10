"""Generate docs/reference/errors.md, the error-message index, from the code.

    python tools/docs_errors.py --write   # rewrite the page
    python tools/docs_errors.py --check   # exit 1 when the page is stale

The page lists every message BenchFlow can print that has a fixed text in the
source, with what to do where the code says:

- the causes a run names for a trial with no score (``bench explain``);
- the warn and fail rows of ``bench doctor``;
- the error classes that carry a next step (``benchflow.errors.UserError``),
  each message they are raised with and its ``hint``, and the file errors the
  CLI prints the same way (``cli/_errors.py``'s ``_FILE_ERRORS``);
- the CLI's errors and their ``next:`` lines (``print_error``, the usage
  errors that name a next step, ``typer.BadParameter``, and the steps
  ``cli/verbs.py`` gives click's own usage errors);
- the lines that refuse a task.md or run config value;
- the refusals before a sandbox starts (``R401``);
- what the verifier step reports;
- the warnings a run logs.

Only the causes come from importing BenchFlow; the rest is read from the
source with ``ast``, so the page names a message by module, not by line, and
changes only when a message does. ``tests/test_docs_errors.py`` fails when the
committed page differs.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import copy
import difflib
import functools
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "benchflow"
PAGE = ROOT / "docs" / "reference" / "errors.md"
# The fix for each message whose code gives none: section, key, message (as the page shows it), fix.
FIXES = ROOT / "tools" / "docs_errors_fixes.tsv"
REGENERATE = "uv run python tools/docs_errors.py --write"
# Each release's messages, by section, as --dump-texts writes them from that release's source with that
# release's interpreter: a message 0.9.2 lacks is labelled (0.10) when 0.10.0rc3 prints it, else (unreleased).
BASELINES = {
    "0.9.2": ROOT / "tools" / "errors_baseline_0.9.2.json",
    "0.10.0rc3": ROOT / "tools" / "errors_baseline_0.10.0rc3.json",
}

_HOLE = "…"


_URL = re.compile(r"https?://[^\s`()]+[^\s`().,;:]")


def _version() -> str:
    """The version in pyproject.toml: the release whose source this page is generated from."""
    m = re.search(
        r'^version\s*=\s*"([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M
    )
    return m.group(1) if m else "this source tree"


def _cell(text: str) -> str:
    """One Markdown table cell, safe for the site's MDX build.

    A URL a message prints is shown as code: it is text to search for, and a
    link to a page this branch adds would not resolve until it merges.
    """
    out = []
    for i, part in enumerate(re.split(r"(`[^`]*`)", " ".join(text.split()))):
        part = part.replace("|", "\\|")
        if not i % 2:
            pieces = re.split(f"({_URL.pattern})", part)
            part = "".join(
                f"`{piece}`"
                if j % 2
                else piece.replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("{", "&#123;")
                .replace("}", "&#125;")
                for j, piece in enumerate(pieces)
            )
        out.append(part)
    return "".join(out)


def _norm(text: str) -> str:
    return " ".join(text.split()).strip()


def curated() -> dict[tuple[str, str, str], str]:
    """The fixes written in ``tools/docs_errors_fixes.tsv``, keyed by section, key and message."""
    out: dict[tuple[str, str, str], str] = {}
    if not FIXES.exists():
        return out
    for line in FIXES.read_text(encoding="utf-8").splitlines()[1:]:
        if line.strip():
            section, key, message, fix = [*line.split("\t"), "", "", "", ""][:4]
            out[(section, key, _norm(message))] = fix.strip()
    return out


class _Fixes:
    """Looks up a written fix and records which ones the page used, and which messages have none."""

    def __init__(self) -> None:
        self.table = curated()
        self.used: set[tuple[str, str, str]] = set()
        self.missing: list[tuple[str, str, str]] = []

    def __call__(self, section: str, key: str, message: str, given: str = "") -> str:
        k = (section, key, _norm(message))
        if given.strip():
            return given
        fix = self.table.get(k, "")
        if fix:
            self.used.add(k)
        else:
            self.missing.append(k)
        return fix


_COMMAND = re.compile(
    r"(export|uv|bench|pip|docker|colima|claude|codex|gh|git|npm|curl|sudo) \S"
)


def _fix(text: str) -> str:
    """A fix cell: a fix that is a shell command is shown as code, to copy."""
    text = " ".join(text.split())
    if (
        _COMMAND.fullmatch(text[: text.find(" ") + 2])
        and "`" not in text
        and not text.endswith(".")
    ):
        return f"`{text}`"
    return _cell(text)


def _code(text: str) -> str:
    """A message as inline code; ``…`` marks text filled in at run time."""
    text = " ".join(text.split()).strip()
    if not text:
        return ""
    text = text.replace("|", "\\|")
    fence = "``" if "`" in text else "`"
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"


# ── Static text of a message expression ─────────────────────────────────


@dataclass
class Module:
    path: Path
    tree: ast.Module
    strings: dict[str, str]

    @property
    def name(self) -> str:
        rel = self.path.relative_to(SRC.parent).with_suffix("")
        return ".".join(rel.parts)


@functools.cache
def _load(path: Path) -> Module:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    strings: dict[str, str] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            text = _static(node.value, {})
            if text.strip(" " + _HOLE):
                strings[node.targets[0].id] = text
    return Module(path, tree, strings)


def _static(node: ast.AST | None, names: dict[str, str]) -> str:
    """The text the source fixes in a string expression; ``…`` for the rest."""
    if node is None:
        return ""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else ""
    if isinstance(node, ast.JoinedStr):
        out = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                out.append(part.value)
            elif (
                # f"...{docs}#keys": a name the source fixes to words is its text.
                isinstance(part, ast.FormattedValue)
                and isinstance(part.value, ast.Name)
                and part.conversion == -1
                and part.format_spec is None
                and _fixed(names.get(part.value.id, ""))
            ):
                out.append(names[part.value.id])
            elif not out or out[-1] != _HOLE:
                out.append(_HOLE)
        return "".join(out)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        # A part with no fixed text is filled in at run time.
        parts = [
            side.value
            if isinstance(side, ast.Constant) and isinstance(side.value, str)
            else _static(side, names) or _HOLE
            for side in (node.left, node.right)
        ]
        text = "".join(parts)
        while _HOLE * 2 in text:
            text = text.replace(_HOLE * 2, _HOLE)
        return "" if text == _HOLE else text
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return _holes(_static(node.left, names))
    if isinstance(node, ast.Name):
        return names.get(node.id, "")
    if isinstance(node, ast.IfExp):
        return _static(node.body, names) or _static(node.orelse, names)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        # hint or "`bench doctor` lists ...": the fixed text is the fallback.
        texts = [_static(v, names) for v in node.values]
        return next((t for t in reversed(texts) if t), "")
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
    ):
        return _static(node.func.value, names).replace("{}", _HOLE)
    return ""


_PERCENT = re.compile(r"%[-#0 +]*\d*(?:\.\d+)?[sdrfxi]")


def _holes(text: str) -> str:
    """printf-style fields (``%s``, ``%.1f``) are filled at run time, like f-string ones."""
    return _PERCENT.sub(_HOLE, text)


def _fixed(text: str) -> bool:
    """A message worth listing: some fixed words, not only filled-in values."""
    return sum(ch.isalpha() for ch in text.replace(_HOLE, "")) >= 12


def _modules(*subdirs: str) -> list[Module]:
    roots = [SRC / d for d in subdirs] if subdirs else [SRC]
    out = []
    for root in roots:
        paths = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for path in paths:
            if "_vendor" in path.parts:
                continue
            out.append(_load(path))
    return out


def _functions(tree: ast.AST) -> Iterator[tuple[str, ast.AST]]:
    """(qualified function name, node) for every function, nested ones too."""

    def walk(node: ast.AST, prefix: str) -> Iterator[tuple[str, ast.AST]]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                name = f"{prefix}.{child.name}" if prefix else child.name
                yield name, child
                yield from walk(child, name)
            elif isinstance(child, ast.ClassDef):
                name = f"{prefix}.{child.name}" if prefix else child.name
                yield from walk(child, name)
            else:
                yield from walk(child, prefix)

    yield "", tree
    yield from walk(tree, "")


def _own_nodes(fn: ast.AST) -> Iterator[ast.AST]:
    """The nodes of a function's body, not of the functions it defines."""
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        if isinstance(
            node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda
        ):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _own_calls(fn: ast.AST) -> Iterator[ast.Call]:
    """Calls in a function's body, not in the functions it defines."""
    for node in _own_nodes(fn):
        if isinstance(node, ast.Call):
            yield node


@functools.cache
def _bodies(path: Path) -> tuple[tuple[str, ast.AST, tuple[ast.AST, ...]], ...]:
    """Each function of a module (nested ones and methods too, and the module
    itself as ``""``) with the nodes of its own body, as :func:`_functions`
    and :func:`_own_nodes` give them, in one walk the families share."""
    out: list[tuple[str, ast.AST, tuple[ast.AST, ...]]] = []

    def scope(prefix: str, name: str) -> str:
        return f"{prefix}.{name}" if prefix else name

    def classes(prefix: str, cls: ast.AST) -> None:
        stack = list(ast.iter_child_nodes(cls))
        while stack:
            node = stack.pop()
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                body(scope(prefix, node.name), node)
            elif isinstance(node, ast.ClassDef):
                classes(scope(prefix, node.name), node)
            else:
                stack.extend(ast.iter_child_nodes(node))

    def body(name: str, fn: ast.AST) -> None:
        own: list[ast.AST] = []
        stack = list(ast.iter_child_nodes(fn))
        while stack:
            node = stack.pop()
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                body(scope(name, node.name), node)
            elif isinstance(node, ast.ClassDef):
                classes(scope(name, node.name), node)
            elif not isinstance(node, ast.Lambda):
                own.append(node)
                stack.extend(ast.iter_child_nodes(node))
        out.append((name, fn, tuple(own)))

    body("", _load(path).tree)
    return tuple(out)


@functools.cache
def _assignments(fn: ast.AST) -> tuple[ast.Assign, ...]:
    """A function's own ``name = ...`` statements, in source order."""
    return tuple(
        sorted(
            (
                node
                for node in _own_nodes(fn)
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ),
            key=lambda node: node.lineno,
        )
    )


def _names_at(fn: ast.AST, strings: dict[str, str], line: int) -> dict[str, str]:
    """The module's strings, and the strings the function assigns to a name
    before ``line`` (the last assignment wins): ``hint = f"fix {path.name}; ..."``
    then ``raise ...(hint=hint)``."""
    names = dict(strings)
    if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
        return names
    for node in _assignments(fn):
        if node.lineno >= line:
            break
        # x = a if c else b holds one of two texts: neither is the name's.
        text = "" if isinstance(node.value, ast.IfExp) else _static(node.value, names)
        target = node.targets[0]
        assert isinstance(target, ast.Name)
        if text.strip(" " + _HOLE):
            names[target.id] = text
        else:
            names.pop(target.id, None)
    return names


def _variants(node: ast.AST | None, names: dict[str, str]) -> list[str]:
    """Each text a message argument can be: both sides of ``a if c else b``."""
    if isinstance(node, ast.IfExp):
        return [*_variants(node.body, names), *_variants(node.orelse, names)]
    return [_static(node, names)]


def _call_name(call: ast.Call) -> str:
    func = call.func
    parts = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    return ".".join(reversed(parts))


def _kw(call: ast.Call, name: str) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _arg(call: ast.Call, index: int, name: str) -> ast.expr | None:
    """A call's argument by position or keyword."""
    if len(call.args) > index:
        return call.args[index]
    return _kw(call, name)


# ── The families ─────────────────────────────────────────────────────────


def causes() -> list[dict[str, str]]:
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from benchflow.cli import explain

    return [
        {
            "code": page.code or "",
            "key": page.key,
            "title": page.title,
            "owner": explain.OWNER_LABELS.get(page.owner, page.owner),
            "body": page.body,
            "next": page.next,
        }
        for page in explain.pages()
    ]


def doctor_rows() -> list[dict[str, str]]:
    module = _load(SRC / "doctor.py")
    rows = []
    for _fname, fn in _functions(module.tree):
        checkers: dict[str, tuple[str, str]] = {}
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
                and _call_name(node.value) == "_checker"
                and len(node.value.args) >= 2
            ):
                group = _static(node.value.args[0], module.strings)
                name = _static(node.value.args[1], module.strings) or _HOLE
                checkers[node.targets[0].id] = (group, name)
        for call in _own_calls(fn):
            callee = _call_name(call)
            if callee not in checkers or not call.args:
                continue
            names = _names_at(fn, module.strings, call.lineno)
            status = _static(call.args[0], names)
            if status not in ("warn", "fail"):
                continue
            summary = _static(_arg(call, 1, "summary"), names)
            fix = _static(_arg(call, 2, "fix"), names)
            group, name = checkers[callee]
            if not summary.strip(" " + _HOLE):
                continue
            rows.append(
                {
                    "check": f"{group}: {name}",
                    "status": status,
                    "summary": summary,
                    "fix": fix,
                }
            )
    uniq = {(r["check"], r["status"], r["summary"], r["fix"]): r for r in rows}
    return sorted(uniq.values(), key=lambda r: (r["check"], r["status"], r["summary"]))


def _init_texts(
    node: ast.ClassDef, strings: dict[str, str]
) -> tuple[list[tuple[str, str]], str]:
    """What a class's ``__init__`` says itself: the messages it passes to
    ``super().__init__`` with the hint it sets after each, and the hint it
    sets for a caller's message (``self.hint = hint or "..."``)."""
    messages: list[tuple[str, str]] = []
    default = ""
    for stmt in node.body:
        if not (isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__"):
            continue
        for block in ast.walk(stmt):
            body = getattr(block, "body", None)
            if not isinstance(body, list):
                continue
            message: str | None = None
            for sub in body:
                if (
                    isinstance(sub, ast.Expr)
                    and isinstance(sub.value, ast.Call)
                    and isinstance(sub.value.func, ast.Attribute)
                    and sub.value.func.attr == "__init__"
                    and isinstance(sub.value.func.value, ast.Call)
                    and _call_name(sub.value.func.value) == "super"
                    and sub.value.args
                ):
                    message = _holes(_static(sub.value.args[0], strings))
                    if _fixed(message):
                        messages.append((message, ""))
                elif (
                    isinstance(sub, ast.Assign)
                    and len(sub.targets) == 1
                    and isinstance(sub.targets[0], ast.Attribute)
                    and sub.targets[0].attr == "hint"
                    and isinstance(sub.targets[0].value, ast.Name)
                    and sub.targets[0].value.id == "self"
                ):
                    hint = _holes(_static(sub.value, strings))
                    if message is not None and _fixed(message):
                        messages[-1] = (message, hint)
                    elif _fixed(hint) and not default:
                        default = hint
    return messages, default


def error_classes() -> list[dict[str, str]]:
    bases: dict[str, list[str]] = {}
    info: dict[str, dict[str, str]] = {}
    # Messages a class builds in its own __init__, with their hints.
    built: dict[str, set[tuple[str, str]]] = {}
    for module in _modules():
        for node in ast.walk(module.tree):
            if not isinstance(node, ast.ClassDef):
                continue
            names = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    names.append(base.id)
                elif isinstance(base, ast.Attribute):
                    names.append(base.attr)
            bases.setdefault(node.name, names)
            own, default = _init_texts(node, module.strings)
            fields = {
                "module": module.name,
                "doc": (ast.get_docstring(node) or "").split("\n\n")[0],
                "hint": default,
            }
            if own:
                built.setdefault(node.name, set()).update(own)
            for stmt in node.body:
                target = None
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                ):
                    target, value = stmt.targets[0].id, stmt.value
                elif (
                    isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Name)
                    and stmt.value is not None
                ):
                    target, value = stmt.target.id, stmt.value
                if target in ("hint", "fault"):
                    text = _static(value, module.strings)
                    if text:
                        fields[target] = text
            info.setdefault(node.name, fields)

    def is_user_error(name: str, seen: frozenset[str] = frozenset()) -> bool:
        if name == "UserError":
            return True
        if name in seen:
            return False
        return any(is_user_error(b, seen | {name}) for b in bases.get(name, []))

    classes = {name for name in bases if name != "UserError" and is_user_error(name)}
    # The CLI prints a missing or unreadable file as expected too, whatever
    # raised it, with one next step (cli/_errors.py).
    file_errors, file_step = _file_errors()
    for name in file_errors:
        classes.add(name)
        info[name] = {
            "module": "builtins",
            "doc": (getattr(builtins, name).__doc__ or "").split("\n\n")[0],
            "hint": file_step,
        }
    raised: dict[str, set[tuple[str, str]]] = {
        name: set(built.get(name, ())) for name in classes
    }
    for module in _modules():
        for _fname, fn, nodes in _bodies(module.path):
            # error.hint = ... after error = RunCardError(...), by variable.
            later_hints: dict[str, list[ast.Assign]] = {}
            for node in nodes:
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Attribute)
                    and node.targets[0].attr == "hint"
                    and isinstance(node.targets[0].value, ast.Name)
                ):
                    later_hints.setdefault(node.targets[0].value.id, []).append(node)
            for node in nodes:
                variable = None
                if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                    call = node.exc
                elif (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and isinstance(node.value, ast.Call)
                ):
                    # error = RunCardError(...); error.hint = hint; raise error
                    call, variable = node.value, node.targets[0].id
                else:
                    continue
                name = _call_name(call).rsplit(".", 1)[-1]
                if name not in classes or not call.args:
                    continue
                names = _names_at(fn, module.strings, node.lineno)
                texts = _variants(call.args[0], names)
                hint_node: ast.expr | None = _kw(call, "hint")
                hint_names = names
                if hint_node is None and variable is not None:
                    after = [
                        a
                        for a in later_hints.get(variable, [])
                        if a.lineno > node.lineno
                    ]
                    if after:
                        first = min(after, key=lambda a: a.lineno)
                        hint_node = first.value
                        hint_names = _names_at(fn, module.strings, first.lineno)
                for text in texts:
                    for hint in _variants(hint_node, hint_names):
                        text, hint = _holes(text), _holes(hint)
                        # A message built at run time can still have a fixed step.
                        if _fixed(text) or _fixed(hint):
                            raised[name].add((text if _fixed(text) else "", hint))
    rows = []
    for name in sorted(classes):
        fields = info[name]
        for message, hint in sorted(raised[name]) or [("", "")]:
            fix = hint or fields.get("hint", "")
            if not fix and name == "EvalPlanError":
                # The CLI prints these as usage errors, with this next step.
                fix = "correct the flag the message names; `bench run --help-all` lists every flag"
            rows.append({**fields, "name": name, "message": message, "fix": fix})
    return rows


def _file_errors() -> tuple[list[str], str]:
    """The built-in errors cli/_errors.py prints as expected (``_FILE_ERRORS``),
    and the next step it gives them."""
    module = _load(SRC / "cli" / "_errors.py")
    names: list[str] = []
    for node in module.tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_FILE_ERRORS"
            and isinstance(node.value, ast.Tuple)
        ):
            names = [e.id for e in node.value.elts if isinstance(e, ast.Name)]
    step = ""
    for node in ast.walk(module.tree):
        if (
            isinstance(node, ast.If)
            and "_FILE_ERRORS" in ast.unparse(node.test)
            and "UserError" in ast.unparse(node.test)
        ):
            for sub in node.body:
                if (
                    isinstance(sub, ast.Assign)
                    and isinstance(sub.targets[0], ast.Name)
                    and sub.targets[0].id == "step"
                ):
                    step = _static(sub.value, module.strings)
    if not names or not step:
        raise LookupError(
            "cli/_errors.py no longer names its file errors and their step"
        )
    return names, step


def _usage_steps() -> dict[str, list[str]]:
    """The next steps cli/verbs.py gives click's own usage errors: an unknown
    command, an unknown flag, a value click refused (each function's answers,
    the one for any command last)."""
    module = _load(SRC / "cli" / "verbs.py")
    steps: dict[str, list[str]] = {}
    for node in module.tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "unknown_command_next",
            "unknown_flag_next",
            "bad_value_next",
        ):
            steps[node.name] = [
                _static(sub.value, module.strings)
                for sub in ast.walk(node)
                if isinstance(sub, ast.Return)
            ]
    if len(steps) != 3:
        raise LookupError(f"cli/verbs.py's usage steps moved: {sorted(steps)}")
    return steps


class _Fill(ast.NodeTransformer):
    """A helper's message with the caller's texts in place of its parameters."""

    def __init__(self, texts: dict[str, str]) -> None:
        self.texts = texts

    def visit_FormattedValue(self, node: ast.FormattedValue) -> ast.AST:
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in self.texts
            and node.conversion == -1
            and node.format_spec is None
        ):
            return ast.Constant(self.texts[node.value.id])
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.Constant(self.texts[node.id]) if node.id in self.texts else node


@dataclass
class _Wrapper:
    """A CLI helper that prints an error with its caller's next step
    (``exit_if_missing(path, next_step=...)``)."""

    fn: ast.FunctionDef | ast.AsyncFunctionDef
    module: Module
    call: ast.Call
    step: str

    def texts(
        self, call: ast.Call, names: dict[str, str]
    ) -> tuple[list[str], list[str]]:
        args = self.fn.args
        params = [a.arg for a in (*args.posonlyargs, *args.args)]
        given: dict[str, ast.AST] = dict(zip(params, call.args, strict=False))
        given.update({kw.arg: kw.value for kw in call.keywords if kw.arg})
        filled = {
            name: text
            for name, node in given.items()
            if name != self.step and (text := _static(node, names))
        }
        message = _arg(self.call, 0, "message")
        assert message is not None, ast.unparse(self.call)
        template = _Fill(filled).visit(copy.deepcopy(message))
        messages = _variants(template, self.module.strings)
        return messages, _variants(given.get(self.step), names)


def _wrappers() -> dict[str, _Wrapper]:
    """Every CLI function that passes a parameter of its own as print_error's
    ``next_step``, by name."""
    found: dict[str, _Wrapper] = {}
    for module in _modules("cli"):
        for _fname, fn, nodes in _bodies(module.path):
            if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            params = {
                a.arg
                for a in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs)
            }
            for call in nodes:
                if isinstance(call, ast.Call) and _call_name(call) == "print_error":
                    step = _kw(call, "next_step")
                    if isinstance(step, ast.Name) and step.id in params and call.args:
                        found[fn.name] = _Wrapper(fn, module, call, step.id)
    return found


def cli_errors() -> list[dict[str, str]]:
    shared = _load(SRC / "cli" / "_shared.py").strings
    usage = _usage_steps()
    bad_value = usage["bad_value_next"][-1]
    rows = []
    # click's own messages (No such command, No such option, Invalid value)
    # get their next step from cli/verbs.py.
    what = {
        "unknown_command_next": "click's error for a command bench doesn't have",
        "unknown_flag_next": "click's error for a flag the command doesn't have",
        "bad_value_next": "click's error for a value a flag can't take",
    }
    for function, steps in usage.items():
        for step in steps:
            rows.append(
                {
                    "command": "verbs",
                    "message": "",
                    "what": what[function],
                    "next": _holes(step),
                }
            )
    wrappers = _wrappers()
    for module in _modules("cli"):
        strings = {**shared, **module.strings}
        for _fname, fn, nodes in _bodies(module.path):
            for call in nodes:
                if not isinstance(call, ast.Call):
                    continue
                callee = _call_name(call)
                helper = wrappers.get(callee.rsplit(".", 1)[-1])
                if helper is not None and fn is helper.fn:
                    continue
                if helper is None and callee not in (
                    "print_error",
                    "UsageErrorWithNext",
                    "_usage",
                    "typer.BadParameter",
                ):
                    continue
                names = _names_at(fn, strings, call.lineno)
                if helper is not None:
                    # exit_if_missing(path, next_step=...): the helper's own
                    # message, filled with what the caller passes it.
                    messages, steps = helper.texts(call, names)
                else:
                    messages = _variants(_arg(call, 0, "message"), names)
                    if callee == "print_error":
                        steps = _variants(_kw(call, "next_step"), names)
                    elif callee in ("UsageErrorWithNext", "_usage"):
                        # A usage error that names its next step (exit 2).
                        steps = _variants(_arg(call, 1, "next_step"), names)
                    else:  # typer.BadParameter: click's usage error for a value
                        steps = [bad_value]
                for message in messages:
                    for step in steps:
                        if not (_fixed(message) or _fixed(step)):
                            continue
                        rows.append(
                            {
                                "command": module.name.rsplit(".", 1)[-1],
                                "message": _holes(message) if _fixed(message) else "",
                                "next": _holes(step),
                            }
                        )
    uniq = {(r["command"], r["message"], r["next"]): r for r in rows}
    return sorted(uniq.values(), key=lambda r: (r["command"], r["message"], r["next"]))


# Where a task.md value is checked: the TaskConfig models and their helpers.
_TASK_VALUE_FILES = ("task/config.py", "task/keep_services.py")


def _key_path(node: ast.AST) -> str:
    """``("orchestrator", "kwargs")`` as ``orchestrator.kwargs``; ``…`` for a
    part filled in at run time."""
    if not isinstance(node, ast.Tuple):
        return ""
    parts = [
        e.value if isinstance(e, ast.Constant) and isinstance(e.value, str) else _HOLE
        for e in node.elts
    ]
    return ".".join(parts)


def file_problems() -> list[dict[str, str]]:
    """The lines that refuse a task.md or run config value: the checks of the
    task's fields (pydantic shows their ``ValueError`` text), a YAML job
    config's problems, and the unknown-key lines both kinds of file get."""
    rows = []

    def add(
        file: str, node: ast.AST, fn: ast.AST, module: Module, key: str = ""
    ) -> None:
        names = _names_at(fn, module.strings, getattr(node, "lineno", 0))
        for text in _variants(node, names):
            text = _holes(text)
            if _fixed(text):
                rows.append(
                    {"file": file, "message": f"{key}: {text}" if key else text}
                )

    for rel in _TASK_VALUE_FILES:
        module = _load(SRC / rel)
        for _fname, fn, nodes in _bodies(module.path):
            for node in nodes:
                if (
                    isinstance(node, ast.Raise)
                    and isinstance(node.exc, ast.Call)
                    and _call_name(node.exc) == "ValueError"
                    and node.exc.args
                ):
                    add("task.md", node.exc.args[0], fn, module)
    module = _load(SRC / "job_config.py")
    for _fname, fn, nodes in _bodies(module.path):
        for node in nodes:
            if (
                isinstance(node, ast.Call)
                and _call_name(node) in ("problems.add", "self.add")
                and len(node.args) >= 2
            ):
                add("run config", node.args[1], fn, module, _key_path(node.args[0]))
    module = _load(SRC / "_utils" / "config_keys.py")
    for _fname, fn, nodes in _bodies(module.path):
        for node in nodes:
            if isinstance(node, ast.Call) and _call_name(node).endswith("append"):
                for arg in node.args[:1]:
                    add("task.md or run config", arg, fn, module)
            elif (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "message"
            ):
                add("task.md or run config", node.value, fn, module)
    uniq = {(r["file"], r["message"]): r for r in rows}
    return sorted(uniq.values(), key=lambda r: (r["file"], r["message"]))


def refusals() -> list[dict[str, str]]:
    module = _load(SRC / "task" / "runtime_capabilities.py")
    rows = []
    for fname, fn in _functions(module.tree):
        for call in _own_calls(fn):
            callee = _call_name(call)
            if callee not in ("_issue", "UnsupportedTaskFeature"):
                continue
            reason = _kw(call, "reason")
            if reason is None and len(call.args) >= 2:
                reason = call.args[1]
            names = _names_at(fn, module.strings, call.lineno)
            text = _static(reason, names)
            if _fixed(text):
                rows.append(
                    {
                        "function": fname,
                        "path": _static(_kw(call, "path"), names),
                        "reason": text,
                    }
                )
    uniq = {(r["path"], r["reason"]): r for r in rows}
    return sorted(uniq.values(), key=lambda r: (r["reason"], r["path"]))


_VERIFIER_FILES = (
    "task/verifier_core.py",
    "rollout/_setup.py",
    "rollout/_verifier_recovery.py",
    "rollout/_separate_verifier.py",
)


def _verifier_text(text: str) -> bool:
    low = text.lower()
    return (
        _fixed(text)
        and ("verifier" in low or "reward" in low)
        and " " in text.strip()
        and not text.startswith(("_", "/", "{"))
    )


def verifier_messages() -> list[dict[str, str]]:
    """What the verifier step reports: texts about the verifier or the reward
    that the code builds into an error or a warning (often through a variable,
    so they are found by their words, not by the call that prints them)."""
    rows = []
    for rel in _VERIFIER_FILES:
        module = _load(SRC / rel)
        for fname, fn in _functions(module.tree):
            if not fname:
                continue
            for node in ast.walk(fn):
                if (
                    isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                    and node is not fn
                ):
                    continue
                values: list[ast.AST] = []
                if isinstance(node, ast.Assign | ast.AugAssign):
                    values = [node.value]
                elif isinstance(node, ast.Call):
                    values = [*node.args, *(kw.value for kw in node.keywords)]
                for value in values:
                    if isinstance(value, ast.JoinedStr | ast.Constant | ast.BinOp):
                        text = _holes(_static(value, module.strings))
                        if _verifier_text(text):
                            rows.append({"module": module.name, "message": text})
    uniq = {(r["module"], r["message"]): r for r in rows}
    return sorted(uniq.values(), key=lambda r: (r["module"], r["message"]))


def warnings_logged() -> list[dict[str, str]]:
    rows = []
    for module in _modules():
        if module.name.startswith("benchflow.cli"):
            continue
        for _fname, fn, nodes in _bodies(module.path):
            for call in nodes:
                if not isinstance(call, ast.Call):
                    continue
                callee = _call_name(call)
                if not (
                    callee.endswith("logger.warning")
                    or callee.endswith("log.warning")
                    or callee == "warnings.warn"
                ):
                    continue
                if not call.args:
                    continue
                names = _names_at(fn, module.strings, call.lineno)
                text = _holes(_static(call.args[0], names))
                if _fixed(text):
                    rows.append({"module": module.name, "message": text})
    uniq = {(r["module"], r["message"]): r for r in rows}
    return sorted(uniq.values(), key=lambda r: (r["module"], r["message"]))


# ── The page ─────────────────────────────────────────────────────────────

_SECTIONS = (
    ("causes", "causes", "key"),
    ("doctor", "doctor_rows", "summary"),
    ("errors", "error_classes", "message"),
    ("cli", "cli_errors", "message"),
    ("files", "file_problems", "message"),
    ("refusals", "refusals", "reason"),
    ("verifier", "verifier_messages", "message"),
    ("warnings", "warnings_logged", "message"),
)


def texts() -> dict[str, list[str] | None]:
    """Each section's messages for this source tree; None for a section that can't be read here."""
    out: dict[str, list[str] | None] = {}
    for section, fn, field in _SECTIONS:
        try:
            out[section] = sorted({_norm(r.get(field) or "") for r in globals()[fn]()})
        except Exception as exc:  # an older tree can lack a module a section reads
            print(f"{section}: {type(exc).__name__}: {exc}", file=sys.stderr)
            out[section] = None
    return out


@functools.cache
def _baselines() -> tuple[dict, dict] | None:
    import json

    if not all(path.exists() for path in BASELINES.values()):
        return None
    return tuple(
        json.loads(path.read_text(encoding="utf-8")) for path in BASELINES.values()
    )


def _since(section: str, text: str) -> str:
    """` (0.10)` or ` (unreleased)` for a message 0.9.2 doesn't print; nothing for one it does."""
    pair = _baselines()
    if pair is None or pair[0].get(section) is None or pair[1].get(section) is None:
        return ""
    t = _norm(text or "")
    if t in pair[0][section]:
        return ""
    return " (0.10)" if t in pair[1][section] else " (unreleased)"


def render(fixes: _Fixes | None = None) -> str:
    fixes = fixes if fixes is not None else _Fixes()
    c = causes()
    d = doctor_rows()
    e = error_classes()
    cli = cli_errors()
    f = file_problems()
    r = refusals()
    v = verifier_messages()
    w = warnings_logged()
    out = [
        "# Error messages",
        "",
        "Every message BenchFlow can print that has a fixed text in its source, and what to do about "
        "it. Search this page for the words of your message: `…` marks a part "
        "filled in at run time. To find out why a trial failed, start with "
        "[When a trial fails](../evaluate/failed-trials.md).",
        "",
        "These messages come from BenchFlow's source on the 0.10 line. A message 0.9.2 doesn't print "
        "has `(0.10)` after it when 0.10.0rc3 prints it, or `(unreleased)` when no release does yet.",
        "",
        "<!-- This page is generated from the code by tools/docs_errors.py, so it lists what this "
        f"version prints. To change it, change the message in the code, then run `{REGENERATE}`. -->",
        "",
        "| Section | Messages |",
        "|---|---|",
        f"| [Causes a run names](#causes-a-run-names) | {len(c)} |",
        f"| [`bench doctor` warnings and failures](#bench-doctor-warnings-and-failures) | {len(d)} |",
        f"| [Errors that say what to do](#errors-that-say-what-to-do) | {len(e)} |",
        f"| [Command errors](#command-errors) | {len(cli)} |",
        f"| [Problems in a task or config file](#problems-in-a-task-or-config-file) | {len(f)} |",
        f"| [Tasks a sandbox can't run](#tasks-a-sandbox-cant-run) | {len(r)} |",
        f"| [What the verifier step reports](#what-the-verifier-step-reports) | {len(v)} |",
        f"| [Warnings a run logs](#warnings-a-run-logs) | {len(w)} |",
        "",
        "## Causes a run names",
        "",
        "The end of a run names a cause for each trial that failed without a usable reward, and "
        "`bench explain <code>` prints its entry. Whose problem it is says where the fix belongs.",
        "",
        "A test script that exits with an error, or runs past `verifier.timeout_sec`, without "
        "writing a reward is not filed under `verifier_failure` or `verifier_timeout` in 0.9.1 and "
        "0.10.0rc1, whatever their entries say: the trial is scored 0, a failure, with a "
        "`Verifier scored 0:` warning in its `result.json` "
        "([When the test itself fails](../evaluate/failed-trials.md#when-the-test-itself-fails)). "
        "Those two causes name a verifier step that ended with no verdict, such as one the pytest "
        "plugin guard refused.",
        "",
        "| Cause | Code | What happened | Whose problem | What to do |",
        "|---|---|---|---|---|",
    ]
    for row in c:
        code = row["code"] if row["code"] != row["key"] else ""
        out.append(
            f"| `{row['key']}` | {code} | {_cell(row['title'])}{_since('causes', row['key'])}. {_cell(row['body'])} "
            f"| {_cell(row['owner'])} | {_cell(row['next'])} |"
        )
    out += [
        "",
        "## `bench doctor` warnings and failures",
        "",
        "`bench doctor` checks this machine and each agent's login. A `warn` row lets runs go "
        "ahead; a `fail` row stops the runs it names. Each row prints a `fix:` line.",
        "",
        "| Check | Status | Message | Fix |",
        "|---|---|---|---|",
    ]
    for row in d:
        out.append(
            f"| {_cell(row['check'])} | {row['status']} | {_code(row['summary'])}{_since('doctor', row['summary'])} | "
            f"{_fix(fixes('doctor', _cell(row['check']), row['summary'], row['fix']))} |"
        )
    out += [
        "",
        "## Errors that say what to do",
        "",
        "These errors stop a command or a run. Each row is one message an error class is raised "
        "with, and what to do: the step it prints as its `next:` line, or, where it prints none, "
        "the fix written here. What each class covers is below the table.",
        "",
        "| Error | Message | What to do |",
        "|---|---|---|",
    ]
    for row in e:
        out.append(
            f"| `{row['name']}` | {_code(row['message']) or '(built at run time)'}{_since('errors', row['message'])} | "
            f"{_fix(fixes('error', row['name'], row['message'], row['fix']))} |"
        )
    out += ["", "| Error class | Where | What it covers |", "|---|---|---|"]
    for name in sorted({row["name"] for row in e}):
        row = next(r for r in e if r["name"] == name)
        out.append(f"| `{name}` | `{row['module']}` | {_cell(row['doc'])} |")
    out += [
        "",
        "## Command errors",
        "",
        "Errors a `bench` command prints before it exits, with the `next:` line it prints "
        "after them. The command module is the file under `benchflow/cli/` that prints it.",
        "",
        "| Command module | Message | What to do |",
        "|---|---|---|",
    ]
    for row in cli:
        message = _code(row["message"]) or _cell(
            row.get("what") or "(built at run time)"
        )
        if row["message"]:
            message += _since("cli", row["message"])
        out.append(
            f"| `{row['command']}` | {message} | "
            f"{_cell(fixes('command', row['command'], row['message'] or '', row['next']))} |"
        )
    out += [
        "",
        "## Problems in a task or config file",
        "",
        "A value in a `task.md` header or a run config that BenchFlow refuses gets one line: "
        "the file, the line and the key, then the problem, as in "
        "`run.yaml:2: concurrenzy: unknown key (did you mean 'concurrency'?)`. Fix the value the "
        "line names; each key's type and default are in "
        "[Files BenchFlow reads and writes](./files.md).",
        "",
        "| File | Problem | What to do |",
        "|---|---|---|",
    ]
    for row in f:
        out.append(
            f"| {row['file']} | {_code(row['message'])}{_since('files', row['message'])} | {_fix(fixes('file', row['file'], row['message']))} |"
        )
    out += [
        "",
        "## Tasks a sandbox can't run",
        "",
        "`bench check` and `bench run` refuse a task feature the chosen sandbox can't provide, "
        "before any sandbox starts, with code `R401`; the line names the task's key and the reason. "
        "Run the task on a sandbox that has the feature (usually `--sandbox docker`), or change "
        "what the task asks for.",
        "",
        "| Task key | Reason | What to do |",
        "|---|---|---|",
    ]
    for row in r:
        out.append(
            f"| {_code(row['path'])} | {_code(row['reason'])}{_since('refusals', row['reason'])} | "
            f"{_fix(fixes('refusal', row['path'], row['reason']))} |"
        )
    out += [
        "",
        "## What the verifier step reports",
        "",
        "Messages about the verifier and the reward, as a trial's `error`, its `warnings` or a "
        "run's log line shows them; a reward of 0 the verifier did not write starts with "
        "`Verifier scored 0:`. The verifier's own output is in the trial's "
        "`verifier/test-stdout.txt`.",
        "",
        "| Module | Message | What to do |",
        "|---|---|---|",
    ]
    for row in v:
        out.append(
            f"| `{row['module']}` | {_code(row['message'])}{_since('verifier', row['message'])} | "
            f"{_fix(fixes('verifier', row['module'], row['message']))} |"
        )
    out += [
        "",
        "## Warnings a run logs",
        "",
        "A run prints these as it works, prefixed with the trial's task. A warning doesn't stop "
        "the run; the trial's `warnings` in `result.json` keeps the ones about it.",
        "",
        "| Module | Message |",
        "|---|---|",
    ]
    for row in w:
        out.append(
            f"| `{row['module']}` | {_code(row['message'])}{_since('warnings', row['message'])} |"
        )
    out.append("")
    return _without_review_tags("\n".join(out))


# An internal review tag in a docstring, such as "(K1-R9-1)", names nothing a reader can
# look up; tools/docs_leaks.py refuses it on a published page, so the page drops it.
_REVIEW_TAG = r"(?:PY\d+-D\d+(?:-[A-Za-z0-9-]+)?|K\d+-R\d+-\d+)"
_REVIEW_TAGS = re.compile(rf" ?\({_REVIEW_TAG}(?:[,;] ?{_REVIEW_TAG})*\)")


def _without_review_tags(text: str) -> str:
    return _REVIEW_TAGS.sub("", text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--write", action="store_true", help="rewrite docs/reference/errors.md"
    )
    mode.add_argument(
        "--check", action="store_true", help="exit 1 when the page is stale"
    )
    mode.add_argument(
        "--dump-texts",
        metavar="PATH",
        help="write each section's messages as JSON (a release baseline)",
    )
    args = parser.parse_args(argv)
    if args.dump_texts:
        import json

        Path(args.dump_texts).write_text(
            json.dumps(texts(), indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 0
    page = render()
    if args.write:
        PAGE.write_text(page, encoding="utf-8")
        print(f"wrote {PAGE.relative_to(ROOT)}")
        return 0
    current = PAGE.read_text(encoding="utf-8") if PAGE.exists() else ""
    if current == page:
        return 0
    sys.stdout.writelines(
        difflib.unified_diff(
            current.splitlines(keepends=True),
            page.splitlines(keepends=True),
            "docs/reference/errors.md (committed)",
            "docs/reference/errors.md (generated)",
        )
    )
    print(f"stale: docs/reference/errors.md; regenerate with: {REGENERATE}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
