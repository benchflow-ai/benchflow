"""Answer variants for the equivalence battery (pure, no sandbox).

Given the files an oracle wrote, build two kinds of variant:

* ``reject`` — wrong answers a verifier must score below the oracle: the output
  missing, an answer file emptied or cut in half, wrong numbers, an off-by-one,
  a wrong unit scale, the values of two rows swapped.
* ``accept`` — the same answer written differently, which a verifier must score
  like the oracle: JSON re-indented or with keys reordered, floats moved by a
  one ulp (float rounding noise), rows or evidence lists in another order, a graded prose
  literal paraphrased, an evidence file cited by full path instead of bare name
  (or the reverse), the result list restricted to the ids of the task's
  request file, a
  trailing newline added or removed.

When the instruction itself fixes the form an accept variant changes (an
exact phrase, an order, a precision, a log count), the variant is marked
``instruction_fixes_form`` and a drop is a warning, not a false negative.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Literal

Expect = Literal["reject", "accept"]

REJECT_FAMILIES = (
    "empty-output",
    "empty-file",
    "truncated",
    "wrong-number",
    "off-by-one",
    "wrong-unit",
    "swapped-rows",
)
ACCEPT_FAMILIES = (
    "reformat",
    "key-order",
    "whitespace",
    "precision",
    "row-order",
    "list-order",
    "paraphrase",
    "path-form",
    "log-shape",
    "request-ids",
)
FAMILIES = REJECT_FAMILIES + ACCEPT_FAMILIES


@dataclass(frozen=True)
class Variant:
    """One perturbed answer: file changes over the oracle output.

    ``changes`` maps a sandbox path to its new bytes, or to ``None`` to delete
    the file. ``label`` names the exact change (field, old and new value).
    """

    family: str
    expect: Expect
    label: str
    changes: Mapping[str, bytes | None]
    instruction_fixes_form: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "expect": self.expect,
            "label": self.label,
            "files": sorted(self.changes),
            "instruction_fixes_form": self.instruction_fixes_form,
        }


@dataclass(frozen=True)
class GradedLiteral:
    """A phrase the verifier requires by substring inside a prose field."""

    file: str
    line: int
    field: str
    literal: str


# ------------------------------------------------------------ instruction

_EXACT_MANDATE = re.compile(
    r"\b(exact(ly)?|verbatim|literal(ly)?|word[- ]for[- ]word|"
    r"must (contain|include|state|say|use)|include the (phrase|string|words?)|"
    r"contain the (phrase|string|words?)|the (phrase|string) )",
    re.I,
)
_ORDER_MANDATE = re.compile(
    r"\b(in (the )?(same |request |input |given |following )?order|ordered|sorted|"
    r"sort (them|rows|results|by)|ascending|descending|in sequence)\b",
    re.I,
)
_PRECISION_MANDATE = re.compile(
    r"\b(decimals?|decimal places?|significant (figures|digits)|round(ed|ing)?|"
    r"exact(ly)?\b[^.\n]{0,40}\b(values?|numbers?|figures?)|precision)\b",
    re.I,
)
_NEWLINE_MANDATE = re.compile(r"\bnew ?lines?\b|\blinefeed\b|\\n", re.I)
_KEY_ORDER_MANDATE = re.compile(
    r"\bkeys?\b[^.\n]{0,40}\bin (this |that |the )?order\b|\bordered keys\b|"
    r"\bkeys? order\b|\border of (the )?keys\b",
    re.I,
)
_EVIDENCE_ORDER_MANDATE = re.compile(
    r"evidence[^.\n]{0,100}\b(order|ordered|sorted|sequence|first|exactly)\b|"
    r"\b(order|ordered|sorted)\b[^.\n]{0,100}evidence",
    re.I,
)
_PATH_FORM_MANDATE = re.compile(
    r"(bare|base ?name|file ?name only|full path|absolute path|exact path)", re.I
)
_LOG_COUNT_MANDATE = re.compile(
    r"(\blog\b|action_log|action log|log entries|calls?)[^.\n]{0,120}\bexactly\b|"
    r"\bexactly\b[^.\n]{0,120}(\blog\b|action_log|action log|log entries|calls?)",
    re.I,
)


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", " ", (text or "").lower()).strip()


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def instruction_mandates_phrase(instruction: str, literal: str) -> bool:
    """True when a sentence of the instruction names ``literal`` and demands it exactly."""
    needle = _norm(literal)
    if not needle:
        return False
    padded = f" {needle} "
    return any(
        padded in f" {_norm(s)} " and _EXACT_MANDATE.search(s)
        for s in _sentences(instruction)
    )


# ------------------------------------------------------- graded literals

_PROSE_FIELDS = re.compile(
    r"\b(methods?|methodology|finding|findings|summary|notes?|purpose|evidence|"
    r"explanation|rationale|narrative|justification|comment|observation|"
    r"conclusion|description|interpretation|caveats?|remarks?)\b",
    re.I,
)


def _const_strs(node: ast.AST) -> list[str]:
    """String constants in a tuple/list/set literal (or set()/tuple()/... of one)."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("set", "frozenset", "tuple", "list")
        and node.args
    ):
        return _const_strs(node.args[0])
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        out: list[str] = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                out.append(elt.value)
            elif isinstance(elt, (ast.Tuple, ast.List, ast.Set, ast.Call)):
                out.extend(_const_strs(elt))
        return out
    return []


def _literal_bindings(tree: ast.AST) -> dict[str, list[str]]:
    binds: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            values = _const_strs(node.value)
            if values:
                binds[node.targets[0].id] = values
    return binds


def extract_graded_literals(sources: Mapping[str, str]) -> list[GradedLiteral]:
    """Phrases graded by substring membership inside a prose field.

    Matches ``"lit" in <prose>``, ``lit in m for lit in (...)`` (also through a
    named tuple and one level of nested groups) and ``{...} & words``, when the
    statement, its enclosing function or the two lines before it name a prose
    field (methods, finding, summary, evidence, ...). ``sources`` maps a
    display path to Python source.
    """
    hits: list[GradedLiteral] = []
    for rel, src in sources.items():
        hits.extend(_graded_literals_in(rel, src))
    return hits


def _graded_literals_in(rel: str, src: str) -> list[GradedLiteral]:
    hits: list[GradedLiteral] = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return hits
    lines = src.splitlines()
    binds = _literal_bindings(tree)
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def stmt_of(node: ast.AST) -> ast.AST:
        while node in parents and not isinstance(node, ast.stmt):
            node = parents[node]
        return node

    def generator_iter(name: ast.Name) -> ast.AST | None:
        node: ast.AST = name
        while node in parents:
            parent = parents[node]
            if isinstance(parent, (ast.GeneratorExp, ast.ListComp, ast.SetComp)):
                for gen in parent.generators:
                    if isinstance(gen.target, ast.Name) and gen.target.id == name.id:
                        return gen.iter
            node = parent
        return None

    def strings_of(node: ast.AST | None) -> list[str]:
        if node is None:
            return []
        if isinstance(node, ast.Name):
            return binds.get(node.id, [])
        return _const_strs(node)

    def stmt_text(stmt: ast.AST) -> str:
        lo = getattr(stmt, "lineno", None)
        if lo is None:
            return ""
        hi = getattr(stmt, "end_lineno", lo) or lo
        return "\n".join(lines[max(0, lo - 1) : hi])

    def add(node: ast.AST, literals: Iterable[str], context: str) -> None:
        kept = sorted(
            {
                lit
                for lit in literals
                if len(lit) >= 2
                and re.search(r"[A-Za-z0-9]", lit)
                and not lit.startswith("/")
            }
        )
        match = _PROSE_FIELDS.search(context.replace("_", " "))
        if not kept or not match:
            return
        for lit in kept:
            hits.append(
                GradedLiteral(
                    rel, getattr(node, "lineno", 0), match.group(0).lower(), lit
                )
            )

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
            lits: list[str] = []
            for side in (node.left, node.right):
                lits += _const_strs(side)
                if isinstance(side, ast.Name):
                    lits += strings_of(generator_iter(side))
                    lits += binds.get(side.id, [])
            stmt = stmt_of(node)
            func: ast.AST = stmt
            while func in parents and not isinstance(func, ast.FunctionDef):
                func = parents[func]
            fname = func.name if isinstance(func, ast.FunctionDef) else ""
            add(
                node,
                [x for x in lits if re.search(r"[A-Za-z]", x)],
                fname + "\n" + stmt_text(stmt),
            )
            continue
        if not (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], (ast.In, ast.NotIn))
        ):
            continue
        lits = []
        if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
            lits = [node.left.value]
        elif isinstance(node.left, ast.Name):
            it = generator_iter(node.left)
            lits = strings_of(it)
            if not lits and isinstance(it, ast.Name):
                lits = strings_of(generator_iter(it))
        if not lits:
            continue
        comparator = ast.get_source_segment(src, node.comparators[0]) or ""
        stmt = stmt_of(node)
        ctx = stmt_text(stmt)
        if "/tests" in comparator or "leak" in ctx.lower():
            continue
        lineno = getattr(stmt, "lineno", 1)
        before = "\n".join(lines[max(0, lineno - 4) : lineno - 1])
        add(node, lits, comparator + "\n" + ctx + "\n" + before)
    return hits


# ------------------------------------------------------------ paraphrase

_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December"
)
_MONTH_NAMES = _MONTHS.split("|")


def _iso_to_words(m: re.Match[str]) -> str:
    year, month, day = m.group(1), int(m.group(2)), int(m.group(3))
    if 1 <= month <= 12:
        return f"{day} {_MONTH_NAMES[month - 1]} {year}"
    return m.group(0)


_PARAPHRASE_RULES: list[
    tuple[re.Pattern[str], str | Callable[[re.Match[str]], str]]
] = [
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), _iso_to_words),
    (re.compile(r"\b(\w+) decimals\b", re.I), r"\1 decimal places"),
    (re.compile(r"\b(" + _MONTHS + r") (\d{4})\b", re.I), r"\1 of \2"),
    (re.compile(r"\bnot a (\w+)\b", re.I), r"not any \1"),
    (re.compile(r"\bordering\b", re.I), "order"),
    (re.compile(r"\barithmetic mean\b", re.I), "equally weighted mean"),
    (re.compile(r"\bpdf page\b", re.I), "page"),
    (re.compile(r"\binterchangeable\b", re.I), "substitutable"),
    (re.compile(r"\bhalf-unit\b", re.I), "half unit"),
    (re.compile(r"\b(\d+)/(\d+)\b"), r"\1 and \2"),
    (re.compile(r"(\w)-(\w)"), r"\1 \2"),
    (re.compile(r"\bmean\b", re.I), "average"),
    (re.compile(r"\bannual\b", re.I), "yearly"),
]
_SYNONYMS = {
    "table": "tabulation",
    "missing": "absent",
    "round": "rounded",
    "projection": "projected values",
    "sum": "total",
    "difference": "subtraction",
    "ratio": "quotient",
    "methodology": "method",
    "classification": "categorisation",
    "absolute": "in absolute terms",
    "range": "span",
    "mixed": "combined",
    "percent": "per cent",
    "estimate": "estimated value",
    "residual": "remainder",
    "figure": "fig.",
    "order": "sequence",
    "page": "p.",
    "interval": "band",
    "decimal": "decimal place",
    "unrounded": "not rounded",
    "annual": "yearly",
}
_NUMBER_WORDS = {
    w: str(i)
    for i, w in enumerate(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
        ]
    )
}
_DIGIT_WORDS = {v: k for k, v in _NUMBER_WORDS.items()}


def _swap_word(word: str) -> str | None:
    w = word.lower()
    if w in _NUMBER_WORDS:
        return _NUMBER_WORDS[w]
    if w in _DIGIT_WORDS and w != "0":
        return _DIGIT_WORDS[w]
    return _SYNONYMS.get(w)


def paraphrase(literal: str) -> str | None:
    """A meaning-preserving rewrite of ``literal`` that no longer contains it.

    Never inserts filler: a rejected paraphrase has to be an answer a
    reviewer would accept, or the drop proves nothing.
    """
    for rx, rep in _PARAPHRASE_RULES:
        new = rx.sub(rep, literal)
        if new.lower() != literal.lower():
            return new
    words = literal.split()
    for i in range(len(words) - 1, -1, -1):
        rep = _swap_word(words[i])
        if rep is not None:
            new = " ".join([*words[:i], rep, *words[i + 1 :]])
            if new.lower() != literal.lower():
                return new
    return None


# ------------------------------------------------------------- JSON walks

_ID_LIKE = re.compile(
    r"(^|_)(id|ids|year|yr|fy|page|pages|step|call|index|idx|rank|line|row|col|"
    r"n|num|number|quarter|month|day|code|version)($|_)",
    re.I,
)
_ROW_LISTS = (
    "results",
    "records",
    "audits",
    "cases",
    "items",
    "rows",
    "series",
    "entries",
    "answers",
    "findings",
)
# Answer keys holding an ordered log of the agent's steps.
LOG_KEYS = ("log", "action_log")
_SKIP_SUBTREES = (*LOG_KEYS, "evidence", "requests", "canary")
PathT = tuple[str | int, ...]


def _fmt_path(path: PathT) -> str:
    out = ""
    for part in path:
        out += (
            f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
        )
    return out or "<root>"


def _numeric_leaves(obj: Any, path: PathT = ()) -> Iterable[tuple[PathT, int | float]]:
    """Substantive numbers: floats anywhere, ints unless the key looks like an id."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        key = str(path[-1]) if path else ""
        if isinstance(obj, float) or not _ID_LIKE.search(key):
            yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k not in _SKIP_SUBTREES:
                yield from _numeric_leaves(v, (*path, k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _numeric_leaves(v, (*path, i))


def _set_path(obj: Any, path: PathT, value: Any) -> None:
    for part in path[:-1]:
        obj = obj[part]
    obj[path[-1]] = value


def _row_list(answer: Any) -> tuple[str | None, list[Any] | None]:
    """The answer's main list of row objects: a conventional name first
    (results, records, ...), else the longest top-level list of objects."""
    if not isinstance(answer, dict):
        return None, None
    for key in _ROW_LISTS:
        rows = answer.get(key)
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            return key, rows
    candidates = [
        (len(v), k)
        for k, v in answer.items()
        if k not in _SKIP_SUBTREES
        and isinstance(v, list)
        and len(v) >= 2
        and all(isinstance(r, dict) for r in v)
    ]
    if not candidates:
        return None, None
    _, key = max(candidates)
    return key, answer[key]


def _walk_strings(obj: Any, fn: Callable[[PathT, str], str], path: PathT = ()) -> int:
    changed = 0
    items: Iterable[tuple[str | int, Any]]
    if isinstance(obj, dict):
        items = [(str(k), v) for k, v in obj.items()]
    elif isinstance(obj, list):
        items = list(enumerate(obj))
    else:
        return 0
    for k, v in items:
        if isinstance(v, str):
            new = fn((*path, k), v)
            if new != v:
                obj[k] = new
                changed += 1
        else:
            changed += _walk_strings(v, fn, (*path, k))
    return changed


def _reverse_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _reverse_keys(obj[k]) for k in reversed(list(obj))}
    if isinstance(obj, list):
        return [_reverse_keys(v) for v in obj]
    return obj


def _num(v: int | float) -> str:
    return repr(v) if isinstance(v, float) else str(v)


# -------------------------------------------------------- variant builders


@dataclass
class _Builder:
    instruction: str
    literals: list[GradedLiteral]
    request_ids: set[str]
    out: list[Variant] = field(default_factory=list)

    def add(
        self,
        family: str,
        expect: Expect,
        label: str,
        changes: Mapping[str, bytes | None],
        fixed: bool = False,
    ) -> None:
        self.out.append(Variant(family, expect, label, dict(changes), fixed))


def _dump_like(original: bytes, obj: Any) -> bytes:
    """Serialize ``obj`` in the oracle file's own layout (indented or compact)."""
    text = original.decode("utf-8", errors="replace")
    indent = 2 if "\n " in text else None
    return (
        json.dumps(obj, ensure_ascii=False, indent=indent)
        + ("\n" if text.endswith("\n") else "")
    ).encode()


def _json_variants(b: _Builder, path: str, raw: bytes, answer: Any) -> None:
    name = PurePosixPath(path).name
    dump = lambda obj: _dump_like(raw, obj)  # noqa: E731

    # --- reject: truncated row list, wrong numbers, off-by-one, unit, swap
    key, rows = _row_list(answer)
    if key and rows and len(rows) >= 2:
        mut = copy.deepcopy(answer)
        keep = len(rows) // 2
        mut[key] = rows[:keep]
        b.add(
            "truncated",
            "reject",
            f"{name}: {key} cut from {len(rows)} to {keep} rows",
            {path: dump(mut)},
        )
    elif len(raw) >= 8:
        b.add(
            "truncated",
            "reject",
            f"{name}: cut to its first {len(raw) // 2} of {len(raw)} bytes",
            {path: raw[: len(raw) // 2]},
        )
    scope: PathT = (key, 0) if key else ()
    target = rows[0] if key and rows else answer
    leaves = list(_numeric_leaves(target, scope))
    if not leaves and key:
        leaves = list(_numeric_leaves(answer))
    if leaves:
        for pct in (10, 1):
            mut = copy.deepcopy(answer)
            changed = []
            for p, v in leaves:
                nv: int | float = round(v * (1 + pct / 100.0), 10) if v else pct / 100.0
                if isinstance(v, int):
                    nv = round(nv) if round(nv) != v else v + 1
                _set_path(mut, p, nv)
                changed.append(f"{_fmt_path(p)} {_num(v)} -> {_num(nv)}")
            b.add(
                "wrong-number",
                "reject",
                f"{name}: +{pct}% on {len(changed)} field(s): "
                + "; ".join(changed[:4]),
                {path: dump(mut)},
            )
        ints = [(p, v) for p, v in leaves if isinstance(v, int)]
        if ints:
            mut = copy.deepcopy(answer)
            p, v = ints[0]
            _set_path(mut, p, v + 1)
            b.add(
                "off-by-one",
                "reject",
                f"{name}: {_fmt_path(p)} {v} -> {v + 1}",
                {path: dump(mut)},
            )
        floats = [(p, v) for p, v in leaves if isinstance(v, float) and v]
        if floats:
            fraction = all(abs(v) <= 1 for _, v in floats)
            scale = 100 if fraction else 1000
            why = "fraction written as percent" if fraction else "thousands vs units"
            mut = copy.deepcopy(answer)
            changed = []
            for p, v in floats:
                _set_path(mut, p, round(v * scale, 10))
                changed.append(
                    f"{_fmt_path(p)} {_num(v)} -> {_num(round(v * scale, 10))}"
                )
            b.add(
                "wrong-unit",
                "reject",
                f"{name}: x{scale} ({why}) on " + "; ".join(changed[:4]),
                {path: dump(mut)},
            )
    if key and rows and len(rows) >= 2:
        mut = copy.deepcopy(answer)
        a, c = mut[key][0], mut[key][1]
        swapped = []
        for k in list(a):
            if (
                k in c
                and not _ID_LIKE.search(str(k))
                and k not in ("id", "evidence", "source", "label", "name")
                and isinstance(a[k], (int, float))
                and not isinstance(a[k], bool)
                and a[k] != c[k]
            ):
                a[k], c[k] = c[k], a[k]
                swapped.append(str(k))
        if swapped:
            b.add(
                "swapped-rows",
                "reject",
                f"{name}: {key}[0] and {key}[1] exchange {', '.join(swapped[:6])} (ids kept)",
                {path: dump(mut)},
            )

    # --- accept: layout, key order, precision
    text = raw.decode("utf-8", errors="replace")
    indented = "\n " in text
    reformatted = json.dumps(answer, ensure_ascii=False, indent=None if indented else 4)
    if text.endswith("\n"):
        reformatted += "\n"
    b.add(
        "reformat",
        "accept",
        f"{name}: re-serialized {'compact' if indented else 'with 4-space indent'}",
        {path: reformatted.encode()},
    )
    _newline_variant(b, path, text)
    if isinstance(answer, (dict, list)) and json.dumps(
        _reverse_keys(answer)
    ) != json.dumps(answer):
        b.add(
            "key-order",
            "accept",
            f"{name}: every object's keys in reverse order",
            {path: dump(_reverse_keys(answer))},
            fixed=bool(_KEY_ORDER_MANDATE.search(b.instruction)),
        )
    floats_all = [
        (p, v) for p, v in _numeric_leaves(answer) if isinstance(v, float) and v
    ]
    if floats_all:
        mut = copy.deepcopy(answer)
        for p, v in floats_all:
            _set_path(mut, p, math.nextafter(v, math.inf))
        p0, v0 = floats_all[0]
        b.add(
            "precision",
            "accept",
            f"{name}: {len(floats_all)} float(s) moved by one ulp (float rounding noise, as in 0.1 + 0.2), e.g. {_fmt_path(p0)} {_num(v0)} -> {_num(math.nextafter(v0, math.inf))}",
            {path: dump(mut)},
            fixed=bool(_PRECISION_MANDATE.search(b.instruction)),
        )
    if key and rows and len(rows) >= 2:
        mut = copy.deepcopy(answer)
        mut[key] = list(reversed(rows))
        b.add(
            "row-order",
            "accept",
            f"{name}: {key} in reverse order ({len(rows)} rows)",
            {path: dump(mut)},
            fixed=bool(_ORDER_MANDATE.search(b.instruction)),
        )

    # --- accept: evidence lists reversed, citation form, log shape
    if isinstance(answer, dict):
        mut = copy.deepcopy(answer)
        n = 0
        for rk in _ROW_LISTS:
            for row in mut.get(rk, []) if isinstance(mut.get(rk), list) else []:
                ev = row.get("evidence") if isinstance(row, dict) else None
                if isinstance(ev, list) and len(ev) > 1:
                    row["evidence"] = list(reversed(ev))
                    n += 1
        if n:
            b.add(
                "list-order",
                "accept",
                f"{name}: evidence lists reversed in {n} row(s)",
                {path: dump(mut)},
                fixed=bool(_EVIDENCE_ORDER_MANDATE.search(b.instruction)),
            )
        for direction in ("full path", "bare name"):
            mut = copy.deepcopy(answer)
            count = [0]

            def cite(
                p: PathT, s: str, direction: str = direction, count: list[int] = count
            ) -> str:
                cited = bool(p) and (
                    str(p[-1]) in ("evidence", "source", "source_file", "artifacts")
                    or (len(p) >= 2 and str(p[-2]) == "evidence")
                )
                if not cited:
                    return s
                if direction == "full path" and re.fullmatch(
                    r"[A-Za-z0-9_.-]+\.(json|xlsx|xls|csv|pdf|html|txt|png|svg)", s
                ):
                    count[0] += 1
                    return "/data/" + s
                if direction == "bare name" and re.fullmatch(
                    r"/(?:[A-Za-z0-9_.-]+/)*data/[A-Za-z0-9_./-]+",
                    s,
                ):
                    count[0] += 1
                    return PurePosixPath(s).name
                return s

            _walk_strings(mut, cite)
            if count[0]:
                b.add(
                    "path-form",
                    "accept",
                    f"{name}: {count[0]} evidence citation(s) rewritten as {direction}",
                    {path: dump(mut)},
                    fixed=bool(_PATH_FORM_MANDATE.search(b.instruction)),
                )
        log_key = next((k for k in LOG_KEYS if isinstance(answer.get(k), list)), None)
        log = answer.get(log_key) if log_key else None
        if log_key and isinstance(log, list) and log and isinstance(log[-1], dict):
            fixed = bool(_LOG_COUNT_MANDATE.search(b.instruction))
            mut = copy.deepcopy(answer)
            extra = copy.deepcopy(log[-1])
            for k in ("call", "step"):
                if isinstance(extra.get(k), int):
                    extra[k] += 1
            mut[log_key].append(extra)
            b.add(
                "log-shape",
                "accept",
                f"{name}: {log_key} {len(log)} -> {len(log) + 1} entries (last one repeated)",
                {path: dump(mut)},
                fixed,
            )

    # --- accept: result list restricted to the ids of the task's request file
    if b.request_ids and isinstance(answer, dict):
        for rk in ("results", "records", "audits", "cases", "items"):
            lst = answer.get(rk)
            if (
                isinstance(lst, list)
                and lst
                and isinstance(lst[0], dict)
                and "id" in lst[0]
            ):
                kept = [r for r in lst if r.get("id") in b.request_ids]
                if len(kept) != len(lst):
                    mut = copy.deepcopy(answer)
                    mut[rk] = kept
                    b.add(
                        "request-ids",
                        "accept",
                        f"{name}: {rk} restricted to the {len(kept)} request-file id(s) of {len(lst)}",
                        {path: dump(mut)},
                    )
                break

    # --- accept: graded prose literal paraphrased
    seen: set[str] = set()
    for graded in b.literals:
        lit = graded.literal
        if lit.lower() in seen:
            continue
        seen.add(lit.lower())
        rep = paraphrase(lit)
        if rep is None:
            continue
        rx = re.compile(re.escape(lit), re.I)
        mut = copy.deepcopy(answer)
        n = _walk_strings(
            mut,
            lambda _p, s, rx=rx, rep=rep: (
                rx.sub(lambda m: paraphrase(m.group(0)) or rep, s)
                if rx.search(s)
                else s
            ),
        )
        if n:
            b.add(
                "paraphrase",
                "accept",
                f"{name}: {lit!r} -> {rep!r} in {n} string(s) ({graded.field} graded at {graded.file}:{graded.line})",
                {path: dump(mut)},
                fixed=instruction_mandates_phrase(b.instruction, lit),
            )


_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")


def _newline_variant(b: _Builder, path: str, text: str) -> None:
    """Toggle one trailing newline; fixed when the instruction talks about newlines."""
    name = PurePosixPath(path).name
    if text.endswith("\n"):
        toggled, label = text[:-1], f"{name}: trailing newline removed"
    else:
        toggled, label = text + "\n", f"{name}: trailing newline added"
    b.add(
        "whitespace",
        "accept",
        label,
        {path: toggled.encode()},
        fixed=bool(_NEWLINE_MANDATE.search(b.instruction)),
    )


def _text_variants(b: _Builder, path: str, raw: bytes) -> None:
    name = PurePosixPath(path).name
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    if len(lines) >= 2:
        b.add(
            "truncated",
            "reject",
            f"{name}: cut to its first {len(lines) // 2} of {len(lines)} lines",
            {path: "".join(lines[: len(lines) // 2]).encode()},
        )
    elif len(text.strip()) >= 8:
        b.add(
            "truncated",
            "reject",
            f"{name}: cut to its first {len(text) // 2} of {len(text)} characters",
            {path: text[: len(text) // 2].encode()},
        )
    first = next((m for m in _NUMBER.finditer(text)), None)
    if first is not None:
        tok = first.group(0)
        value = float(tok)
        if "." in tok:
            decimals = len(tok.split(".")[1])
            new = f"{value * 1.1:.{decimals}f}"
            if new == tok:
                new = f"{value + 10**-decimals:.{decimals}f}"
        else:
            bumped = round(value * 1.1)
            new = str(bumped if bumped != int(value) else int(value) + 1)
        span = first.span()
        b.add(
            "wrong-number",
            "reject",
            f"{name}: first number {tok} -> {new} (+10%)",
            {path: (text[: span[0]] + new + text[span[1] :]).encode()},
        )
        ints = [m for m in _NUMBER.finditer(text) if "." not in m.group(0)]
        if ints:
            m = ints[0]
            bumped = str(int(m.group(0)) + 1)
            b.add(
                "off-by-one",
                "reject",
                f"{name}: {m.group(0)} -> {bumped}",
                {path: (text[: m.start()] + bumped + text[m.end() :]).encode()},
            )
    _newline_variant(b, path, text)


def build_variants(
    answers: Mapping[str, bytes],
    *,
    instruction: str,
    verifier_sources: Mapping[str, str] | None = None,
    request_ids: Iterable[str] = (),
    families: Iterable[str] | None = None,
) -> list[Variant]:
    """Every variant for the oracle's answer files, in a stable order.

    ``answers`` maps sandbox paths to the bytes the oracle wrote. JSON files get
    the structural variants; any other text file gets the line/number ones.
    """
    b = _Builder(
        instruction=instruction,
        literals=extract_graded_literals(verifier_sources or {}),
        request_ids={str(i) for i in request_ids},
    )
    if answers:
        b.add(
            "empty-output",
            "reject",
            "every answer file removed: " + ", ".join(sorted(answers)),
            {p: None for p in answers},
        )
    for path in sorted(answers):
        raw = answers[path]
        b.add(
            "empty-file",
            "reject",
            f"{PurePosixPath(path).name}: emptied to 0 bytes",
            {path: b""},
        )
        answer: Any = None
        is_json = False
        if path.endswith(".json"):
            try:
                answer = json.loads(raw)
                is_json = True
            except (UnicodeDecodeError, json.JSONDecodeError):
                is_json = False
        if is_json:
            _json_variants(b, path, raw, answer)
        else:
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            _text_variants(b, path, raw)
    wanted = set(families) if families is not None else None
    return [v for v in b.out if wanted is None or v.family in wanted]
