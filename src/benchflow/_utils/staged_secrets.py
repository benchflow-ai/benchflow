"""BenchFlow masks credentials it staged in everything it records.

BenchFlow knows the exact values it puts into a sandbox: the provider keys and
the proxy key in the agent's environment, and the secret fields of each
credentials file it stages. Each is registered where it is staged
(:func:`register`), into the trial's registry (:func:`trial_registry`), or the
run's for a value every trial may see (:func:`register_run`: a proxy's master
key). Whatever an agent prints, the records BenchFlow writes through its
redaction layer (``trajectories.types``: the trajectory, result.json's text,
the session-log copies) replace each registered value with
``[masked:<name>]``, in each form it can take:

- as it is, JSON-escaped (up to three times) and %-encoded, as each URL
  encoder writes it (:func:`secret_forms`);
- base64, standard and URL-safe, whole and at each byte offset inside a
  larger buffer (the characters that depend on the value alone);
- any of those broken across lines (``base64 -w 76``, a 64-column block, a
  terminal's wrap);
- each line of a multi-line value on its own;
- for a JSON value, each secret field's value on its own; a JSON value with
  no field on the secret-name list, or an array, whole, and each string in it.

Values that overlap in a text are masked as one run, named for the longest.
A value shorter than :data:`MIN_LENGTH` is never masked (it would wipe
ordinary text); it is named in the record instead. A file that is not text
(it holds a NUL, or is not UTF-8) keeps its length: each registered value's
bytes are filled in place (:func:`mask_bytes`), so a tar or sqlite file stays
valid; a file the filler can't fill safely (a sqlite index over a filled
value) is withheld, a stub in its place (:meth:`StagedSecrets.withhold`).
Only what BenchFlow staged is masked: this is no general scrubber, and the
pattern redaction still runs after it.

A rollout claims its registry with its first step and binds it around each
of its steps (``rollout._claim_staged_registry``): the one a verb that
drives the trial itself bound for it (``bench eval continue``,
``bench eval branch``), its trial's when it is part of one (a branch child,
a verifier's child), else its own, so one task can drive several rollouts
and each is masked and recorded with its own. The trial's ``integrity/masking.json``
(:func:`write_trial_record`) holds names and counts, never a value, and names
what this version does not sweep: the folders the sandbox writes itself and
the job's own files.
"""

from __future__ import annotations

import base64
import contextlib
import functools
import json
import logging
import re
import threading
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any
from urllib.parse import quote, quote_plus

logger = logging.getLogger(__name__)

#: Shorter values are never masked.
MIN_LENGTH = 16
#: The record, under the trial's ``integrity/``.
RECORD = "masking.json"
#: What this version masks: the trial's records BenchFlow writes through its
#: redaction layer, its own files under ``agent/`` among them.
APPLIED_TO = (
    "trajectory/",
    "result.json",
    "results.jsonl",
    "trainer/",
    "agent/acp_trajectory.jsonl",
    "agent/output.txt",
    "agent/claude-sessions/",
    "agent/codex-sessions/",
    # The stop's records of the processes it ended, and the integrity check's
    # process rows: command lines.
    "kill-before-grading.json",
    "kill-after-grading.json",
    "integrity/verifier-inputs.json",
)
#: What it does not sweep yet: what the sandbox writes itself into these
#: folders, and the job's own files, written after its trials.
NOT_SWEPT_FOLDERS = ("verifier/", "agent/", "artifacts/")
NOT_SWEPT_JOB_FILES = (
    "run-card.json",
    "summary.json",
    "results.jsonl",
    "adp.jsonl",
    "verifiers.jsonl",
    "report.md",
    "evaluation.json",
)
#: The record's note on a staged JSON value with no field it knows a secret.
WHOLE_VALUE = (
    "no secret fields recognised; whole value registered, and each string in "
    f"it of {MIN_LENGTH} characters or more except its type, URLs, URNs and "
    "file paths"
)
WHOLE_ARRAY = (
    f"a JSON array; whole value registered, and each string in it of {MIN_LENGTH} "
    "characters or more except its type, URLs, URNs and file paths"
)


def marker(name: str) -> str:
    return f"[masked:{name}]"


# --- the forms a value takes -------------------------------------------------

#: The characters JavaScript's ``encodeURIComponent`` leaves as they are,
#: and ``encodeURI`` (ECMA-262, 19.2.6); Python's ``quote`` keeps letters,
#: digits and ``_.-~`` whatever ``safe`` says.
_URI_COMPONENT_KEEPS = "!*'()"
_URI_KEEPS = _URI_COMPONENT_KEEPS + ";/?:@&=+$,#"


def url_forms(value: str) -> set[str]:
    """*value* as each URL encoder writes it, once and twice: Python's
    ``quote`` (``/`` kept, and nothing kept) and ``quote_plus`` (nothing and
    ``/`` kept), JavaScript's ``encodeURIComponent`` and ``encodeURI``, and a
    form body or ``URLSearchParams`` (WHATWG urlencoded: ``*`` kept, ``~``
    escaped, a space as ``+``)."""
    encoders = (
        lambda v: quote(v, safe=""),
        quote,
        quote_plus,
        lambda v: quote_plus(v, safe="/"),
        lambda v: quote(v, safe=_URI_COMPONENT_KEEPS),
        lambda v: quote(v, safe=_URI_KEEPS),
        lambda v: quote_plus(v, safe="*").replace("~", "%7E"),
    )
    forms = set()
    for encode in encoders:
        once = encode(value)
        forms.update({once, encode(once)})
    return forms


def secret_forms(value: str) -> list[str]:
    """Every form a value takes in a copied log, as the encoders write it,
    never a list of escapes of our own: as given; JSON-escaped
    (``json.dumps``, ASCII-only and not, up to three times: a JSON line held
    in a string field of a JSON line); each of those %-encoded (``quote``,
    once and again: ``%25XX``); the value as each URL encoder writes it
    (:func:`url_forms`). Longest first."""
    forms: set[str] = {value}
    for ascii_only in (True, False):
        escaped = value
        for _ in range(3):
            escaped = json.dumps(escaped, ensure_ascii=ascii_only)[1:-1]
            forms.add(escaped)
    for form in list(forms):
        encoded = form
        for _ in range(2):
            encoded = quote(encoded, safe="")
            forms.add(encoded)
    forms.update(url_forms(value))
    return sorted(forms, key=lambda form: -len(form))


def base64_forms(value: str) -> set[str]:
    """*value* base64-encoded (standard and URL-safe): whole, with and
    without padding, and, at each byte offset inside a larger buffer, the
    characters that depend on the value alone (the first group, which holds
    bytes before it, and the last, which may hold bytes after it, dropped)."""
    data = value.encode()
    forms: set[str] = set()
    for encode in (base64.b64encode, base64.urlsafe_b64encode):
        whole = encode(data).decode()
        forms.update({whole, whole.rstrip("=")})
        for offset in range(3):
            encoded = encode(b"\0" * offset + data).decode()
            groups = (offset + len(data)) // 3
            core = encoded[4 if offset else 0 : groups * 4]
            if len(core) >= MIN_LENGTH:
                forms.add(core)
    return forms


# --- one pattern for many forms ----------------------------------------------

#: One unit of a form: an escape whose hex digits come in either case
#: (``%XX``, ``\\uXXXX``), or one character.
_UNIT = re.compile(r"%[0-9A-Fa-f]{2}|\\u[0-9A-Fa-f]{4}|.", re.DOTALL)


def unit_key(unit: str) -> str:
    if len(unit) == 3 and unit[0] == "%":
        return unit.upper()
    if len(unit) == 6 and unit.startswith("\\u"):
        return unit.lower()
    return unit


def _unit_pattern(unit: str) -> str:
    if len(unit) == 3 and unit[0] == "%":
        return f"%(?i:{unit[1:]})"
    if len(unit) == 6 and unit.startswith("\\u"):
        return f"\\\\u(?i:{unit[2:]})"
    return re.escape(unit)


#: A line break a wrapped form may hold before any of its characters: as
#: written, or JSON-escaped once or more (the redaction layer reads a field's
#: value inside a JSON copy, and a request body is JSON inside JSON).
_WRAP = r"(?:\r?\n|(?:\\+r)?\\+n)?"
_BREAKS = re.compile(r"\r|\n|\\+r|\\+n")


def _wrapped_unit_pattern(unit: str) -> str:
    return _WRAP + re.escape(unit)


class _Node:
    """A node of the forms' trie: its edges by their first unit, each the
    run of units no other form shares there and the node it leads to, and
    whether a form ends here."""

    __slots__ = ("edges", "end")

    def __init__(self) -> None:
        self.edges: dict[str, list[Any]] = {}
        self.end = False


def _insert(root: _Node, units: list[str]) -> None:
    """Add one form's units to the trie, splitting an edge where it parts
    from another form's. A loop, not recursion: a form is as deep as it is
    long, and a credential can be kilobytes."""
    node, i = root, 0
    while i < len(units):
        edge = node.edges.get(units[i])
        if edge is None:
            leaf = _Node()
            leaf.end = True
            node.edges[units[i]] = [units[i:], leaf]
            return
        label, child = edge
        shared = 1
        while (
            shared < len(label)
            and i + shared < len(units)
            and label[shared] == units[i + shared]
        ):
            shared += 1
        if shared < len(label):
            middle = _Node()
            middle.edges[label[shared]] = [label[shared:], child]
            edge[0], edge[1] = label[:shared], middle
            child = middle
        node, i = child, i + shared
    node.end = True


def _trie_pattern(
    root: _Node, unit_pattern: Callable[[str], str] = _unit_pattern
) -> str:
    """The forms in the trie as one regex: a branch per edge, and a form
    that ends at a node only when no longer one goes on (greedy ``?``).
    Built bottom up with a stack, so no form is too long for it."""
    built: dict[int, str] = {}
    stack: list[tuple[_Node, bool]] = [(root, False)]
    while stack:
        node, ready = stack.pop()
        if not ready:
            stack.append((node, True))
            stack.extend((child, False) for _, child in node.edges.values())
            continue
        branches = [
            "".join(map(unit_pattern, label)) + built.pop(id(child))
            for label, child in node.edges.values()
        ]
        if not branches:
            built[id(node)] = ""
            continue
        body = branches[0] if len(branches) == 1 else "(?:" + "|".join(branches) + ")"
        built[id(node)] = f"(?:{body})?" if node.end else body
    return built[id(root)]


def forms_pattern(
    forms: list[str], unit_pattern: Callable[[str], str] = _unit_pattern
) -> re.Pattern[str]:
    """One pattern for *forms*: a trie of their units, so a match costs the
    form's length, not the number of forms, and the longest form wins; an
    escape's hex digits in either case."""
    root = _Node()
    for form in forms:
        _insert(root, [unit_key(unit) for unit in _UNIT.findall(form)])
    return re.compile(_trie_pattern(root, unit_pattern))


@functools.lru_cache(maxsize=64)
def values_pattern(values: tuple[str, ...]) -> re.Pattern[str]:
    """One pattern for every :func:`secret_forms` form of each of *values*."""
    return forms_pattern([form for value in values for form in secret_forms(value)])


def _key(text: str) -> str:
    return "".join(unit_key(unit) for unit in _UNIT.findall(text))


# --- the registry --------------------------------------------------------------


class StagedSecrets:
    """The values staged in one trial (or the run), by name, and what masking
    them did. A run's registry is the *parent* of each trial's: a trial masks
    both. Never written: :meth:`record` holds names and counts only."""

    def __init__(
        self, parent: StagedSecrets | None = None, owner: object = None
    ) -> None:
        self.parent = parent
        #: The rollout whose trial this is, when a driver bound it for one.
        self.owner = owner
        self._lock = threading.Lock()
        self._values: dict[str, str] = {}  # value -> name
        self._too_short: set[str] = set()
        self._notes: dict[str, str] = {}
        self._withheld: dict[Path, str] = {}  # the file's full path -> why
        self._masked: Counter[str] = Counter()
        self._compiled: tuple[Any, ...] | None = None

    # registering

    def register(self, name: str, value: object) -> None:
        """Register *value*, staged under *name*. A JSON object registers
        each secret field's value instead (``name.field.sub``). A JSON value
        with no secret field, or an array, registers its whole value and each
        string in it (``name.0``, ``name.field``), and the record says so. A
        multi-line value registers each of its lines too."""
        if not isinstance(value, str) or not value.strip():
            return
        parsed = _json_strings(value)
        if parsed is None:
            self._register_value(name, value)
            return
        is_array, leaves = parsed
        secret = [(path, leaf) for path, leaf, is_secret in leaves if is_secret]
        if secret and not is_array:
            for path, leaf in secret:
                self._register_value(f"{name}.{path}", leaf)
            return
        # Which part is the secret is unknown here, so all of it is, but for
        # the strings any such file holds in the open (its type, a URL, a URN,
        # a file's path): those stay in the whole value only.
        with self._lock:
            self._notes[name] = WHOLE_ARRAY if is_array else WHOLE_VALUE
        self._register_value(name, value)
        for path, leaf, _ in leaves:
            if len(leaf.strip()) >= MIN_LENGTH and not _public(path, leaf.strip()):
                self._register_value(f"{name}.{path}", leaf)

    def _register_value(self, name: str, value: str) -> None:
        if not value.strip():
            return
        self._add(name, value.strip())
        lines = [line.strip() for line in value.splitlines()]
        if len(lines) > 1:
            for line in lines:
                # A key block's armour (``-----BEGIN ...-----``) is no secret.
                if len(line) >= MIN_LENGTH and not line.startswith("-----"):
                    self._add(name, line)

    def _add(self, name: str, value: str) -> None:
        with self._lock:
            if len(value) < MIN_LENGTH:
                self._too_short.add(name)
                return
            if value not in self._values:
                self._values[value] = name
                self._compiled = None

    def names(self) -> list[str]:
        with self._lock:
            return sorted(set(self._values.values()))

    def _entries(self) -> dict[str, str]:
        entries = dict(self.parent._entries()) if self.parent is not None else {}
        with self._lock:
            entries.update(self._values)
        return entries

    # masking

    def _patterns(self) -> tuple[Any, ...] | None:
        entries = self._entries()
        if not entries:
            return None
        with self._lock:
            compiled = self._compiled
        if compiled is not None and compiled[0] == entries:
            return compiled
        names: dict[str, str] = {}
        wrapped_names: dict[str, str] = {}
        for value, name in entries.items():
            for form in secret_forms(value):
                names.setdefault(_key(form), name)
            for form in base64_forms(value):
                names.setdefault(_key(form), name)
            for form in [value, *base64_forms(value)]:
                wrapped_names.setdefault(form, name)
        compiled = (
            entries,
            forms_pattern(list(names)),
            names,
            forms_pattern(list(wrapped_names), _wrapped_unit_pattern),
            wrapped_names,
        )
        with self._lock:
            self._compiled = compiled
        return compiled

    def mask(self, text: str) -> str:
        """*text* with every form of every registered value replaced by a
        marker, counted. Values that overlap are one run, named for the
        longest (:func:`_runs`), so no part of either is left."""
        if not text:
            return text
        compiled = self._patterns()
        if compiled is None:
            return text
        _, pattern, names, wrapped, wrapped_names = compiled
        text = self._replace(
            pattern, text, lambda found: names.get(_key(found), "staged")
        )
        if "\n" in text or "\\n" in text:  # a line break, as written or escaped
            text = self._replace(
                wrapped,
                text,
                lambda found: wrapped_names.get(_BREAKS.sub("", found), "staged"),
            )
        return text

    def _replace(
        self, pattern: re.Pattern[str], text: str, name_of: Callable[[str], str]
    ) -> str:
        out: list[str] = []
        last = 0
        for start, end, parts in _runs(pattern, text):
            out.append(text[last:start])
            for name in {name_of(part) for part in parts}:
                self._count(name)
            out.append(marker(name_of(max(parts, key=lambda part: len(part)))))
            last = end
        if not out:
            return text
        out.append(text[last:])
        return "".join(out)

    def mask_bytes(self, data: bytes) -> bytes:
        """A file's bytes masked: text (UTF-8, no NUL) as :meth:`mask` masks
        it; anything else keeps its length, each registered value's bytes (as
        it is, JSON-escaped, base64) filled with ``*`` in place, overlapping
        values as one run."""
        if b"\0" not in data:
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                pass
            else:
                return self.mask(text).encode("utf-8")
        spans: list[tuple[int, int, str]] = []
        for value, name in self._entries().items():
            for form in {*secret_forms(value), *base64_forms(value)}:
                needle = form.encode("utf-8")
                start = data.find(needle)
                while start >= 0:
                    spans.append((start, start + len(needle), name))
                    start = data.find(needle, start + 1)
        filled = bytearray(data)
        for start, end, names in _merged(spans):
            filled[start:end] = b"*" * (end - start)
            for name in names:
                self._count(name)
        return bytes(filled)

    def _count(self, name: str) -> None:
        with self._lock:
            self._masked[name] += 1

    def holds_anything(self) -> bool:
        """Whether anything was registered here (or in the run's registry),
        masked, named too short or withheld: whether a record has anything
        to say about this trial beyond what is not swept."""
        if self._entries():
            return True
        with self._lock:
            return bool(self._too_short or self._masked or self._withheld)

    def withhold(self, path: str | Path, why: str) -> None:
        """Record a file left out of the trial's record, a stub in its place,
        because what it holds could not be masked safely. *path* is where the
        file is in the trial folder, in full."""
        with self._lock:
            self._withheld[Path(path)] = why

    # the record

    def _notes_all(self) -> dict[str, str]:
        notes = dict(self.parent._notes_all()) if self.parent is not None else {}
        with self._lock:
            notes.update(self._notes)
        return notes

    def record(
        self,
        agent_log: str | None = None,
        *,
        folder: str | Path | None = None,
        root: str | Path | None = None,
        branch_total: bool = False,
        others: tuple[Path, ...] = (),
    ) -> dict[str, Any]:
        """Names and counts, never a value, for the record in *folder*.
        *agent_log* is the trial's log of the agent's own output
        (``agent/<name>``), which BenchFlow writes through its redaction
        layer. ``withheld`` holds only the files under *folder* and not
        under one of *others* (the child folders a branch made under it),
        each by its path from *root*, the trial's folder. *branch_total*: the
        counts are a branch trial's, its parent's and every child's, and say
        so."""
        registered = sorted(set(self._entries().values()))
        notes = self._notes_all()
        applied_to = [*APPLIED_TO, *([agent_log] if agent_log else [])]
        with self._lock:
            withheld = [
                {"path": _shown(path, root), "why": why}
                for path, why in sorted(self._withheld.items())
                if (folder is None or path.is_relative_to(folder))
                and not any(path.is_relative_to(other) for other in others)
            ]
            return {
                "registered": registered,
                "branch_total" if branch_total else "masked": dict(
                    sorted(self._masked.items())
                ),
                "too_short": sorted(self._too_short),
                "min_length": MIN_LENGTH,
                "notes": dict(sorted(notes.items())),
                "withheld": withheld,
                "paths": (
                    "applied_to and not_swept are relative to this folder; "
                    "withheld paths are from the trial's folder (a branch's "
                    "parent's)"
                ),
                "applied_to": applied_to,
                "not_swept": {
                    "folders": list(NOT_SWEPT_FOLDERS),
                    "job_files": list(NOT_SWEPT_JOB_FILES),
                    "why": (
                        "the sandbox writes into these folders itself (BenchFlow's "
                        "own files there, named under applied_to, are masked), and "
                        "the job's files are written after its trials"
                    ),
                },
            }


_URL = re.compile(r"(?:[a-z][a-z0-9+.-]*:)?//", re.IGNORECASE)
_ABSOLUTE_PATH = re.compile(r"/[\w.@~-]+(?:/[\w.@~ -]*)+")


def _public(path: str, leaf: str) -> bool:
    """A string a credentials file holds in the open: the JSON's ``type``,
    a URL, a URN, or an absolute file path."""
    return (
        path == "type"
        or _URL.match(leaf) is not None
        or leaf.lower().startswith("urn:")
        or _ABSOLUTE_PATH.fullmatch(leaf) is not None
    )


def _shown(path: Path, root: str | Path | None) -> str:
    if root is not None and path.is_relative_to(root):
        return path.relative_to(root).as_posix()
    return path.as_posix()


def _json_strings(value: str) -> tuple[bool, list[tuple[str, str, bool]]] | None:
    """For a JSON object or array: whether it is an array, and each string
    leaf by its dotted path, with whether it sits under a secret field's
    name (``trajectories.types.is_secret_field``); None for any other value."""
    text = value.strip()
    if not text or text[0] not in "{[":
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    from benchflow.trajectories.types import is_secret_field

    # Each leaf by its own field's name: ``tokens.account_id`` is no secret
    # for sitting beside ``tokens.refresh_token``.
    found: list[tuple[str, str, bool]] = []
    stack: list[tuple[str, Any, bool]] = [("", data, False)]
    while stack:
        path, node, secret = stack.pop()
        if isinstance(node, dict):
            for key, item in node.items():
                child = f"{path}.{key}" if path else str(key)
                stack.append((child, item, is_secret_field(str(key), item)))
        elif isinstance(node, list):
            for index, item in enumerate(node):
                stack.append((f"{path}.{index}" if path else str(index), item, secret))
        elif isinstance(node, str):
            found.append((path, node, secret))
    return isinstance(data, list), sorted(found)


def _runs(pattern: re.Pattern[str], text: str) -> Iterator[tuple[int, int, list[str]]]:
    """Each run of *text* that *pattern* matches, as (start, end, the
    matched texts): a match that starts inside one and runs past it joins
    it, so two overlapping values leave no tail. The extra tries are at
    matched characters only."""
    pos = 0
    while (found := pattern.search(text, pos)) is not None:
        start, end = found.span()
        parts = [found[0]]
        at = start + 1
        while at < end:
            inner = pattern.match(text, at)
            if inner is not None and inner.end() > end:
                end = inner.end()
                parts.append(inner[0])
            at += 1
        yield start, end, parts
        pos = end if end > start else end + 1


def _merged(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, set[str]]]:
    """Overlapping spans as one, with the names of the values in it: each
    form of a value found at one place (its base64 whole and its core) is
    one mask."""
    merged: list[tuple[int, int, set[str]]] = []
    for start, end, name in sorted(spans):
        if merged and start < merged[-1][1]:
            last_start, last_end, names = merged[-1]
            names.add(name)
            merged[-1] = (last_start, max(last_end, end), names)
        else:
            merged.append((start, end, {name}))
    return merged


# --- the run's and the trial's registries --------------------------------------

_RUN = StagedSecrets()
_trial: ContextVar[StagedSecrets | None] = ContextVar("staged_secrets", default=None)


def bind_trial(
    owner: object = None,
) -> tuple[StagedSecrets, Token[StagedSecrets | None] | None]:
    """Bind a registry for one trial (the run's is its parent), and the
    token :func:`release_trial` takes. The one already bound for *owner*
    (the trial's rollout) is kept, so a verb that drives a trial binds it
    once and the rollout's own ``run()`` inside adds to it; any other trial
    gets its own."""
    bound = _trial.get()
    if owner is not None and bound is not None and bound.owner is owner:
        return bound, None
    registry = StagedSecrets(parent=_RUN, owner=owner)
    return registry, _trial.set(registry)


def release_trial(token: Token[StagedSecrets | None] | None) -> None:
    """Undo :func:`bind_trial`'s binding, in the context that made it."""
    if token is not None:
        _trial.reset(token)


def new_trial_registry(owner: object = None) -> StagedSecrets:
    """A registry for one trial (the run's is its parent), not yet bound."""
    return StagedSecrets(parent=_RUN, owner=owner)


@contextlib.contextmanager
def bound(registry: StagedSecrets) -> Iterator[StagedSecrets]:
    """*registry* bound for the block, as it is when it is already the one
    bound (a rollout's own, around each of its steps)."""
    if _trial.get() is registry:
        yield registry
        return
    token = _trial.set(registry)
    try:
        yield registry
    finally:
        _trial.reset(token)


@contextlib.contextmanager
def trial_registry(owner: object = None) -> Iterator[StagedSecrets]:
    """:func:`bind_trial` for the block."""
    registry, token = bind_trial(owner)
    try:
        yield registry
    finally:
        release_trial(token)


def current() -> StagedSecrets | None:
    """The trial's registry, or None outside a trial."""
    return _trial.get()


def register(name: str, value: object) -> None:
    """Register a value staged in this trial; nothing outside a trial."""
    registry = _trial.get()
    if registry is not None:
        registry.register(name, value)


def register_env(env: Mapping[str, object]) -> None:
    """Register each credential-bearing entry of an environment staged in
    this trial, by the rule that keeps it out of ``config.json``: a model
    name or a URL is never registered."""
    from benchflow._utils.config_redaction import _should_record_env_entry

    for name, value in env.items():
        if isinstance(value, str) and not _should_record_env_entry(str(name), value):
            register(str(name), value)


def register_run(name: str, value: object) -> None:
    """Register a value every trial of this run may see. A value staged for
    one launch (a proxy's master key) is the trial's (:func:`register`): the
    run's registry is never cleared, and each trial's first mask compiles
    all of it."""
    _RUN.register(name, value)


def mask(text: str) -> str:
    """*text* masked in this trial; as it is outside a trial."""
    registry = _trial.get()
    return registry.mask(text) if registry is not None else text


def mask_bytes(data: bytes) -> bytes:
    """A file's bytes masked in this trial (:meth:`StagedSecrets.mask_bytes`);
    as they are outside a trial."""
    registry = _trial.get()
    return registry.mask_bytes(data) if registry is not None else data


def withhold(path: str | Path, why: str) -> None:
    """Name a file left out of this trial's record (a stub in its place);
    nothing outside a trial."""
    registry = _trial.get()
    if registry is not None:
        registry.withhold(path, why)


def write_record(
    rollout_dir: str | Path,
    registry: StagedSecrets,
    *,
    agent_log: str | None = None,
    root: str | Path | None = None,
    branch_total: bool = False,
    others: tuple[Path, ...] = (),
) -> Path:
    """``integrity/masking.json`` in *rollout_dir*, a trial's folder or a
    branch child's under *root* (:meth:`StagedSecrets.record`)."""
    path = Path(rollout_dir) / "integrity" / RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    record = registry.record(
        agent_log,
        folder=rollout_dir,
        root=root if root is not None else rollout_dir,
        branch_total=branch_total,
        others=others,
    )
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def write_trial_record(
    folder: str | Path | None, registry: StagedSecrets, *, agent: object = None
) -> None:
    """The trial's record (:func:`write_record`) in its *folder*, and in each
    folder a branch made under it for a child
    (``branches/<fork>/children/<child>/``, masked by the same registry: its
    counts, in each, are the branch's ``branch_total``). *agent* names the
    trial's agent log under ``applied_to``. Nothing without a folder; a
    failed write is logged, never raised."""
    from benchflow._utils.artifact_names import agent_log_name

    if folder is None or not Path(folder).is_dir():
        return
    agent_log = f"agent/{agent_log_name(agent)}" if isinstance(agent, str) else None
    root = Path(folder)
    children = sorted(
        child
        for child in root.glob("branches/*/children/*")
        if child.is_dir() and not child.is_symlink()
    )
    for each in (root, *children):
        try:
            write_record(
                each,
                registry,
                agent_log=agent_log,
                root=root,
                branch_total=bool(children),
                others=tuple(child for child in children if child != each),
            )
        except OSError as exc:
            logger.warning(f"The masking record was not written: {exc}")
