"""Config axis (``C``) overlay — patch a task's resolved config per run.

``C`` in Han Lee's ``I_eval = {T, A, M, S, C, R, τ}`` is configuration: budgets,
tools, skills, stopping rules. This is the config twin of
:mod:`benchflow._utils.env_registry` (the ``S`` axis): the task ships sensible
defaults, and a run *overlays* a patch on top — so one knob (e.g.
``agent.timeout_sec`` — Han's "lower turn budget" branch) can be varied while
``T/A/M/S/R`` are held fixed.

The overlay is supplied by ``--config-override`` as **inline JSON/YAML/TOML or an
``@file`` ref** — the same dual "value or ref" form as ``--environment``. It is
parsed once at plan time (:func:`load_config_override`) into a dict, threaded as
typed data through ``EvalCreateRequest → EvalPlan → EvaluationConfig →
RolloutConfig``, then deep-merged into the resolved :class:`TaskConfig` at the
rollout layer (:func:`apply_config_override`) and re-validated — so a bad overlay
fails loudly. It is persisted to ``config.json`` and ``evaluation.json``
without the values that may be a credential (:func:`overlay_record`), so the
run records which configuration it bound.
"""

from __future__ import annotations

import json
import logging
import math
import re
import typing
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from benchflow._utils.content_address import sha256_prefixed
from benchflow._utils.credential_names import _words, entry_kind, name_kind
from benchflow.errors import UserError

#: Top-level config sections a run-time overlay MAY patch (fail-closed allowlist).
#: The C axis is agent/budgets/sandbox/config — never the scorer, so anything
#: outside this set (verifier, reward, solution/oracle, …) is rejected.
#: ``artifacts`` is additive: an overlay's entries are collected besides the
#: task's own, which a separate verifier may need, and never replace them.
_PATCHABLE_SECTIONS = frozenset(
    {"agent", "sandbox", "metadata", "artifacts", "timeout_multipliers"}
)
_ADDITIVE_SECTIONS = frozenset({"artifacts"})
#: ``timeout_multipliers`` scales the task's own budgets instead of replacing
#: them, so one overlay fits every task of a run: ``verifier`` multiplies
#: ``verifier.timeout_sec`` and ``build`` multiplies ``sandbox.build_timeout_sec``
#: (a YAML job config's timeout multipliers). It changes
#: only how long a phase may run, never what the verifier checks.
_TIMEOUT_MULTIPLIERS = {
    "verifier": ("verifier", "timeout_sec"),
    "build": ("sandbox", "build_timeout_sec"),
}
_COMPUTED_SECTIONS = _ADDITIVE_SECTIONS | {"timeout_multipliers"}

logger = logging.getLogger(__name__)


def _parse_overlay(raw: str) -> dict[str, Any]:
    """Parse an overlay value: ``@file`` ref or inline JSON/YAML/TOML.

    Tries JSON, then YAML, then TOML, and on total failure raises one clear
    error that names the source and echoes a snippet — rather than surfacing
    only the last (TOML) parser's confusing message.
    """
    import tomllib

    source = "inline --config-override"
    text = raw
    path = _overlay_file(raw)
    if path is not None:
        text, source = path.read_text(), f"file {path}"

    def _load_yaml(s: str) -> Any:
        import yaml

        return yaml.safe_load(s)

    data: Any = None
    last: Exception | None = None
    for loader in (json.loads, _load_yaml, tomllib.loads):
        try:
            data = loader(text)
            break
        except Exception as exc:
            last = exc
    else:
        # Where, never what: the text may hold a credential.
        raise ValueError(
            f"could not parse config override ({source}) as JSON, YAML, or TOML: "
            f"{type(last).__name__}: {last}"
        )

    if not isinstance(data, dict):
        raise ValueError(
            f"config override ({source}) must be a mapping (table), got "
            f"{type(data).__name__}"
        )
    return data


def _overlay_file(raw: str) -> Path | None:
    """The file a ``--config-override`` value names (``@file``, or a path
    to a .json, .yaml, .yml or .toml file), or None for inline text."""
    if raw.startswith("@"):
        return Path(raw[1:]).expanduser()
    candidate = Path(raw)
    if candidate.suffix in {".json", ".yaml", ".yml", ".toml"} and candidate.is_file():
        return candidate
    return None


def load_config_override(value: str | None) -> dict[str, Any] | None:
    """Parse a raw ``--config-override`` value into an overlay dict (or ``None``)."""
    if not value:
        return None
    return _parse_overlay(value)


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base``.

    Nested tables (dicts) deep-merge; all other values — scalars *and lists* —
    replace the base value wholesale.
    """
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def overlay_hash(overlay: dict[str, Any]) -> str:
    """Content address of an overlay, stable across key order."""
    payload = json.dumps(overlay, sort_keys=True, separators=(",", ":"))
    return sha256_prefixed(payload.encode())


#: Where a command that runs part of a recorded job again (regrade, verifier
#: recovery, continue, resume) reads the overlay the job ran with, when it is
#: not passed to it: inline or an ``@file`` ref, as ``--config-override`` takes.
CONFIG_OVERRIDE_ENV = "BENCHFLOW_CONFIG_OVERRIDE"


class ConfigOverrideNeededError(ValueError, UserError):
    """A recorded job, or a saved config, holds a config overlay whose record
    left values out, and the command that runs it again (or loads it) was
    not given them, or was given an overlay that is not the recorded one.

    An overlay is free-form (a sandbox env, MCP headers), so a value in it may
    be a credential. The records keep every name, and the values whose name
    and shape both look harmless; the others are supplied again, as
    ``agent_env``'s are.
    """

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


class ConfigOverrideInvalidError(ValidationError, UserError):
    """A config overlay the task config refuses: a key it does not have, or
    a value of another type. Each problem is named by its field path and
    what the field takes, never by the value given (its errors hold no
    input): the overlay may hold a credential (a sandbox env, MCP headers).
    Still a pydantic ValidationError, as the task config's own is."""

    hint = (
        "correct the key or the value at that path in --config-override; "
        "`bench run <tasks> --config-override ... --dry-run` checks it "
        "against each task without running one"
    )

    def __str__(self) -> str:
        problems = (
            f"{'.'.join(map(str, error['loc'])) or 'the config'}: {error['msg']}"
            for error in self.errors(include_input=False, include_url=False)
        )
        return "the config override does not fit the task's config: " + "; ".join(
            problems
        )


# What a field takes, by pydantic's error type, for a problem at a path whose
# value the record withholds (its own message may quote the value).
_TAKES = {
    "extra_forbidden": "is not a field there",
    "missing": "is required",
    "string_type": "takes a string",
    "int_type": "takes an integer",
    "int_parsing": "takes an integer",
    "int_from_float": "takes an integer",
    "float_type": "takes a number",
    "float_parsing": "takes a number",
    "bool_type": "takes true or false",
    "bool_parsing": "takes true or false",
    "dict_type": "takes a mapping",
    "model_type": "takes a mapping",
    "model_attributes_type": "takes a mapping",
    "list_type": "takes a list",
    "literal_error": "takes one of the values it allows",
    "enum": "takes one of the values it allows",
}


def _invalid(exc: Any, overlay: dict[str, Any]) -> ConfigOverrideInvalidError:
    """*exc* (the task config's ValidationError for an overlaid config) as
    an error that names each problem by its path, with pydantic's own words
    only where the record keeps every value under and over that path, and
    otherwise what the field takes, so no withheld value can appear (and no
    ``input_value``: the errors are read without their input)."""
    from pydantic_core import PydanticCustomError

    withheld: list[str] = []
    _split(overlay, _comparable(overlay), "", withheld)
    problems = []
    for error in exc.errors(include_input=False, include_url=False):
        path = ".".join(str(part) for part in error["loc"])
        hidden = any(
            not path
            or path == w
            or path.startswith(f"{w}.")
            or w.startswith(f"{path}.")
            for w in withheld
        )
        said = _TAKES.get(error["type"], "is not valid") if hidden else error["msg"]
        problems.append(
            {
                "type": PydanticCustomError(
                    "config_override", "{said}", {"said": said}
                ),
                "loc": error["loc"],
                "input": None,
            }
        )
    return ConfigOverrideInvalidError.from_exception_data(
        exc.title, problems, hide_input=True
    )


def overlay_paths(overlay: dict[str, Any], prefix: str = "") -> list[str]:
    """The dotted path of every value in an overlay (a list is one value)."""
    paths: list[str] = []
    for key in sorted(overlay, key=lambda k: str(k)):
        value = overlay[key]
        path = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            paths.extend(overlay_paths(value, f"{path}."))
        else:
            paths.append(path)
    return paths


#: A mapping of free-form names (a sandbox's ``env``, an MCP server's
#: ``headers``): nothing under it is stored either, though a change to what
#: a resume compares there is still a change (see _blind).
_FREE_FORM_NAME_WORDS = frozenset({"env", "header"})
_DROPPED = object()
_NO_VALUE = object()  # a key path's or a flag's name asked about alone
#: What ``compare`` holds in place of a value it does not compare (under a
#: credential's name, after ``--api-key``), so the name and the place stay
#: and a rotated key is no change; and what an echoed command line shows in
#: place of a withheld value.
_WITHHELD_ITEM = "<withheld>"
# A list item that is a long flag: --api-key, or --api-key=<value>.
_FLAG = re.compile(r"--([^=\s]+)(=.*)?", re.DOTALL)
# A value that is a bearer credential by itself, as an Authorization header
# holds one, under any key.
_BEARER_VALUE = re.compile(r"\s*bearer\s+\S+\s*", re.IGNORECASE)


def _key_words(name: str) -> set[str]:
    """The words of a key name (credential_names' split), each also without
    a plural s."""
    words = set(_words(name))
    return words | {word[:-1] for word in words if word.endswith("s")}


def _held(annotation: Any) -> Any:
    """What a field of the task config models holds: a model (itself, or in
    a list or an optional), ``dict`` for a mapping, or None for a value."""
    from pydantic import BaseModel

    if typing.get_origin(annotation) is dict:
        return dict
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in typing.get_args(annotation):
        if (held := _held(arg)) is not None:
            return held
    return None


def _declared(names: tuple[str, ...]) -> list[bool]:
    """For each key of the path *names*, whether the task config models
    declare it there as a field that is not a mapping (``agent.token_budget``,
    a setup command's ``capture_dir_b64_env``).

    Such a key is the schema's name, not a credential's, so its words are not
    matched, and a field the models gain later is covered with no list to
    keep. A declared mapping (``env``, ``headers``, ``metadata``) holds
    free-form names: it and every key under it are matched as any key is.
    """
    from benchflow.task.config import TaskConfig

    model: Any = TaskConfig
    declared: list[bool] = []
    for name in names:
        field = model.model_fields.get(name) if model is not None else None
        held = _held(field.annotation) if field is not None else None
        declared.append(field is not None and held is not dict)
        model = None if held is dict else held
    return declared


def _kind(name: str, value: Any = _NO_VALUE) -> str | None:
    """``secret`` when one key or flag name is a credential's, holding
    *value* when it is given
    (:func:`~benchflow._utils.credential_names.entry_kind`: TESTS_PASS
    holding a password is one), ``free-form`` when it names a free-form
    mapping (``env``, ``headers``), None otherwise, a setting's name among
    them (``max_tokens``). A credential's word comes before a free-form
    one: ``env_token`` is a credential's."""
    kind = name_kind(name) if value is _NO_VALUE else entry_kind(name, value)
    if kind is not None:
        return "secret" if kind == "credential" else None
    return "free-form" if _key_words(name) & _FREE_FORM_NAME_WORDS else None


def _name_rule(names: tuple[str, ...], value: Any = _NO_VALUE) -> str | None:
    """``free-form`` or ``secret`` when a key of the path *names* is such a
    name (:func:`_kind`), leaving out the fields the task config models
    declare (:func:`_declared`); None otherwise.

    A free-form mapping comes first: what is left under it has already been
    through the rule for such mappings (``_without_secrets``), which leaves
    out an entry with a secret name unless it names a setting or is a
    recognized one, and a change to such a setting must stay a change. An
    entry whose own name is a setting's (``MAX_OUTPUT_TOKENS``) is a setting
    there too, stored and compared as given, unless a credential's name is
    above it; any other entry is compared as a digest and never stored.
    The last key holds *value*, when it is given, and is judged with it
    (:func:`~benchflow._utils.credential_names.entry_kind`).
    """
    declared = _declared(names)
    last = len(names) - 1
    kinds = [
        _kind(name, value if index == last else _NO_VALUE)
        for index, (name, kept) in enumerate(zip(names, declared, strict=True))
        if not kept
    ]
    if "free-form" in kinds:
        kind = None
        if names:
            kind = (
                name_kind(names[-1])
                if value is _NO_VALUE
                else entry_kind(names[-1], value)
            )
        setting = bool(names) and not declared[-1] and kind == "setting"
        return None if setting and "secret" not in kinds else "free-form"
    return "secret" if "secret" in kinds else None


def _digest(value: Any) -> str:
    """A short digest of *value* (``evaluation._redacted``)."""
    from benchflow.evaluation import _redacted

    return _redacted(json.dumps(value, sort_keys=True, default=str))


def _hidden(value: Any, rule: str) -> Any:
    """*value* as compared when its name is a credential's (rule ``secret``)
    or a free-form mapping's. Under a credential's name it is not compared,
    whatever it is (a number too: a short digest of one could be guessed),
    so a rotated key is no change; dropping or adding it still is
    (:func:`overlay_changes`). Under a free-form mapping it is a digest, so
    a changed setting there (MAX_THINKING_TOKENS) is a change and its value
    is not kept."""
    if rule == "secret":
        return _DROPPED
    return _digest(value)


# A NAME=VALUE or ``Name: value`` item after a free-form flag (--env,
# --header): an entry of such a mapping, in one word.
_ENTRY = re.compile(r"([^\s=:]+)\s*[=:]\s*(.*)", re.DOTALL)


def _hidden_item(value: Any, rule: str, given: Any) -> Any:
    """:func:`_hidden` for a list item, which keeps its place; *given* is
    the item as given. After a free-form flag, an entry as one word
    (``--env DB_PASSWORD=<v>``, ``--header "Authorization: <v>"``) is left
    out as the entry of an env or headers mapping would be, unless that rule
    compares it (``MY_FLAG=on``, a digest), and one named like a setting
    (``MAX_OUTPUT_TOKENS=8192``) is kept as given."""
    from benchflow._utils.config_redaction import _should_compare_env_entry

    entry = _ENTRY.fullmatch(given) if isinstance(given, str) else None
    if rule == "free-form" and entry:
        if not _should_compare_env_entry(*entry.groups()):
            return _WITHHELD_ITEM
        if entry_kind(entry[1], entry[2]) == "setting":
            return value  # MAX_OUTPUT_TOKENS=8192: a setting, as given
    kept = _hidden(value, rule)
    return _WITHHELD_ITEM if kept is _DROPPED else kept


def _blind_list(
    items: list[Any], names: tuple[str, ...], given: list[Any]
) -> list[Any]:
    """A list as compared, item by item, beside the list as *given* (before
    ``_without_secrets``, which digests an item shaped like a secret whole).
    A mapping in it (an MCP server) is gone through by its key path. The
    value of a flag whose name is a credential's or a free-form mapping's is
    a value under that name (:func:`_hidden_item`): the item after
    ``--api-key`` or ``--header``, and what follows the first ``=`` of
    ``--api-key=<v>`` as given. When that value is itself such a flag
    (``--no-auth --api-key <v>``: the first was a switch), the item after it
    is its value in turn."""
    blinded: list[Any] = []
    follows: str | None = None  # the rule of the flag just before
    for index, (item, as_given) in enumerate(zip(items, given, strict=True)):
        flag = _FLAG.fullmatch(as_given) if isinstance(as_given, str) else None
        rule = None
        if flag:
            # The flag's value: after its first ``=``, or the next item.
            held = flag[2][1:] if flag[2] is not None else None
            if flag[2] is None and index + 1 < len(given):
                held = given[index + 1]
            rule = _kind(flag[1], held)
        if follows is not None:
            blinded.append(_hidden_item(item, follows, as_given))
            follows = rule if flag and flag[2] is None else None
        elif flag and rule and flag[2] is not None:
            value = flag[2][1:]
            blinded.append(f"--{flag[1]}={_hidden_item(value, rule, value)}")
        else:
            blinded.append(_blind(item, names, as_given))
            follows = rule
    return blinded


def _blind(value: Any, names: tuple[str, ...], given: Any) -> Any:
    """*value* (what ``_without_secrets`` left of the overlay) as compared,
    by its key path *names* (:func:`_name_rule`), beside the same part of
    the overlay as *given*: under a credential's name a string is
    ``<withheld>`` and anything else is a digest, and under a free-form
    mapping it is a digest (:func:`_hidden`). A mapping keeps every name it
    was given, one whose entry ``_without_secrets`` left out as
    ``<withheld>`` too, so dropping or adding a key is a change wherever it
    is, inside a list too. A list is gone through item by item
    (:func:`_blind_list`), and a bare ``Bearer <token>`` is a digest under
    any name."""
    if isinstance(given, dict) and given and isinstance(value, dict):
        blinded = {}
        for key, item in given.items():
            kept = (
                _blind(value[key], (*names, str(key)), item)
                if key in value
                else _DROPPED
            )
            blinded[key] = _WITHHELD_ITEM if kept is _DROPPED else kept
        return blinded
    rule = _name_rule(names, given)
    if rule is not None:
        return _hidden(value, rule)
    if isinstance(value, list) and isinstance(given, list):
        return _blind_list(value, names, given)
    if isinstance(value, str) and _BEARER_VALUE.fullmatch(value):
        return _digest(value)
    return value


def _comparable(overlay: dict[str, Any]) -> dict[str, Any]:
    """The overlay as a resume compares it, in the form JSON gives back, with
    no value that may be a credential. It names every key the overlay sets.

    By its value (``benchflow.evaluation._without_secrets``, the rule for
    agent_env): a secret-looking entry of an env or headers mapping is
    ``<withheld>``, any other secret-looking string is a digest. By its key
    path (:func:`_blind`): a string under a secret name is ``<withheld>``,
    any other value there and one under a free-form mapping is a digest, and
    in a list the value of a flag with such a name is treated as one under
    it.
    """
    from benchflow.evaluation import _without_secrets

    given = json.loads(json.dumps(overlay, default=str))
    safe = json.loads(json.dumps(_without_secrets(given), default=str))
    return _blind(safe, (), given)


def _split(
    overlay: Any,
    comparable: Any,
    prefix: str,
    withheld: list[str],
    shown: Any = None,
) -> Any:
    """*overlay* without the values *comparable* left out or replaced (or,
    given *shown*, with ``shown(<the value in comparable>)`` in their place),
    whose paths are added to *withheld*."""
    kept: dict[str, Any] = {}
    for key, value in overlay.items():
        path = f"{prefix}{key}"
        known = isinstance(comparable, dict) and key in comparable
        if known and isinstance(value, dict) and value:
            kept[key] = _split(value, comparable[key], f"{path}.", withheld, shown)
        elif known and json.loads(json.dumps(value, default=str)) == comparable[key]:
            kept[key] = value
        else:
            withheld.append(path)
            if shown is not None:
                kept[key] = shown(comparable[key]) if known else _WITHHELD_ITEM
    return kept


def _plain_json(text: str) -> bool:
    """Whether *text* is JSON with no key given twice (all of it is in what
    it parses to)."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        if len({key for key, _ in items}) < len(items):
            raise ValueError("a key given twice")
        return dict(items)

    try:
        json.loads(text, object_pairs_hook=pairs)
    except ValueError:
        return False
    return True


_SHOWN_DIGEST = re.compile(r"<redacted sha256:[0-9a-f]+>")


def _echoed(compared: Any) -> Any:
    """A withheld value as an echoed command line shows it: ``<withheld>``;
    a list (MCP args, an MCP server) as ``compare`` has it, with
    ``<withheld>`` for each digest too, so its other items stay as given."""

    def items(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: items(item) for key, item in value.items()}
        if isinstance(value, list):
            return [items(item) for item in value]
        if isinstance(value, str):
            return _SHOWN_DIGEST.sub(_WITHHELD_ITEM, value)
        return value

    return items(compared) if isinstance(compared, list) else _WITHHELD_ITEM


def echoed_override(raw: str) -> str | None:
    """A ``--config-override`` value as a command line BenchFlow prints may
    show it: a file reference as given; inline JSON with no withheld value
    as given; anything else as JSON with ``<withheld>`` in place of each
    value :func:`overlay_record` would not store (in a list, of each item it
    does not show). None when it does not parse, so that no part of it is
    shown."""
    if _overlay_file(raw) is not None:
        return raw
    try:
        overlay = _parse_overlay(raw)
    except ValueError:
        return None
    withheld: list[str] = []
    shown = _split(overlay, _comparable(overlay), "", withheld, _echoed)
    if not withheld and _plain_json(raw):
        return raw
    return json.dumps(shown, default=str)


def overlay_record(overlay: dict[str, Any]) -> dict[str, Any]:
    """What config.json and evaluation.json keep of an overlay.

    - ``keys``: the sections it sets;
    - ``patch``: the overlay without the values that may be a credential,
      which is what a later command applies again;
    - ``withheld``: the paths of the values left out of ``patch``, which that
      command must be given again (:func:`recorded_overlay`);
    - ``compare``: the form a resume compares (:func:`_comparable`);
    - ``sha256``: the content address of ``patch``, ``withheld`` and
      ``compare``, what the record stores, so it confirms no value the
      record left out (a guessed password, checked against it). A record
      from before ``withheld`` and ``compare`` has the whole overlay's;
      nothing compares one record's hash with another's.

    A value is left out when a word of its key path names a credential or a
    free-form mapping (:func:`_name_rule`: ``env``, ``token``, ``password``,
    ``auth``, ``session`` and the like, at any level, other than a field the
    task config models declare, such as ``agent.token_budget``), or when the
    value itself has a secret's shape (a bare ``Bearer <token>`` too). A
    list is left out whole when it holds such a value, a mapping with one
    (an MCP server's env) or a flag that names one (``--api-key <v>``).
    Until this record, ``patch`` was the whole overlay.
    """
    comparable = _comparable(overlay)
    withheld: list[str] = []
    patch = _split(overlay, comparable, "", withheld)
    stored = {"patch": patch, "withheld": sorted(withheld), "compare": comparable}
    return {"keys": sorted(overlay), "sha256": overlay_hash(stored), **stored}


def exported_override(overlay: dict[str, Any] | None, *, include: bool) -> Any:
    """What a saved config (``Evaluation.to_dict``, ``RolloutConfig.to_dict``)
    writes of an overlay: the overlay itself with ``include``, or when
    nothing in it is withheld; otherwise its record (:func:`overlay_record`),
    which a load reads back through :func:`recorded_overlay`, so the
    withheld values are supplied again (``BENCHFLOW_CONFIG_OVERRIDE``) or
    the load stops with :class:`ConfigOverrideNeededError`."""
    if not overlay or include:
        return overlay
    record = overlay_record(overlay)
    return record if record["withheld"] else overlay


def is_overlay_record(value: Any) -> bool:
    """Whether *value* is a stored record of an overlay, not an overlay
    (whose sections are never named ``patch`` or ``sha256``)."""
    return isinstance(value, dict) and ("patch" in value or "sha256" in value)


def _as_record(stored: Any) -> dict[str, Any] | None:
    """What a job stored of its overlay, as a record of this version: a
    record itself, or, for an older record (whose ``patch`` is the whole
    overlay) or an older evaluation.json's overlay, the record made of it
    now; None for none."""
    if not stored:
        return None
    if is_overlay_record(stored):
        if "compare" in stored:
            return stored
        stored = stored.get("patch")
    return overlay_record(stored) if isinstance(stored, dict) else None


def _patch_paths(record: dict[str, Any]) -> set[str]:
    patch = record.get("patch")
    return set(overlay_paths(patch)) if isinstance(patch, dict) else set()


def _changed_paths(old: Any, new: Any, prefix: str = "") -> list[str]:
    """The dotted paths below which ``old`` and ``new`` differ (a list or a
    scalar is one leaf), e.g. ["sandbox.env.MY_FLAG"]."""
    if not isinstance(old, dict) and not isinstance(new, dict):
        return [prefix] if prefix and old != new else []
    old = old if isinstance(old, dict) else {}
    new = new if isinstance(new, dict) else {}
    paths: list[str] = []
    for sub in sorted(old.keys() | new.keys(), key=str):
        if old.get(sub) != new.get(sub):
            paths.extend(_changed_paths(old.get(sub), new.get(sub), f"{prefix}{sub}."))
    return [p.rstrip(".") for p in paths]


def overlay_changes(recorded: Any, current: Any) -> str | None:
    """How the overlay a run stores (*current*) differs from the one a job
    stored (*recorded*) as a resume compares them, by path only, never a
    value (``sandbox.env.MY_FLAG changed; sandbox.env.GH_TOKEN removed``),
    or None when they match.

    Both are compared in their ``compare`` form, where a rotated key is no
    change, and by their ``withheld`` paths, which must be the same: a
    withheld value the job's overlay set and this one does not is removed,
    and one this one sets and the job's did not is added, however equal the
    rest is. An older record is read by the rules of this version
    (:func:`_as_record`); no hash is compared.
    """
    old, new = _as_record(recorded) or {}, _as_record(current) or {}
    old_held = set(old.get("withheld") or [])
    new_held = set(new.get("withheld") or [])
    if old.get("compare") == new.get("compare") and old_held == new_held:
        return None
    removed = sorted(old_held - new_held - _patch_paths(new))
    added = sorted(new_held - old_held - _patch_paths(old))
    named = {*removed, *added}
    changed = [
        path
        for path in _changed_paths(old.get("compare"), new.get("compare"))
        if path not in named
    ]
    changed += sorted((old_held ^ new_held) - named - set(changed))
    groups = (("changed", changed), ("removed", removed), ("added", added))
    shown = "; ".join(f"{', '.join(paths)} {what}" for what, paths in groups if paths)
    return shown or "changed"


def recorded_overlay(
    recorded: Any, supplied: dict[str, Any] | None = None, *, what: str
) -> dict[str, Any] | None:
    """The overlay a recorded job ran with, for *what* (a command that runs
    part of the job again), or None when it ran with none.

    *recorded* is what the job stored. *supplied* is the overlay the caller
    passes; without one, :data:`CONFIG_OVERRIDE_ENV` is read. A record that
    withheld nothing (and any older record, which kept every value) needs
    nothing supplied. One that withheld values needs the overlay again, and
    a supplied overlay must be the job's in everything a resume compares
    (:func:`overlay_changes`): a rotated key is accepted; another setting,
    or a withheld key dropped or added, is not. Otherwise this raises
    :class:`ConfigOverrideNeededError`.
    """
    import os

    if not recorded:
        return None
    if not is_overlay_record(recorded):
        # An older evaluation.json kept the overlay itself.
        return recorded if isinstance(recorded, dict) else None
    if supplied is None:
        supplied = load_config_override(os.environ.get(CONFIG_OVERRIDE_ENV))
    withheld = list(recorded.get("withheld") or [])
    patch = recorded.get("patch")
    patch = patch if isinstance(patch, dict) else {}
    how = (
        f"set {CONFIG_OVERRIDE_ENV} to the --config-override this was recorded "
        "with (inline or @file); a rotated key is fine"
    )
    if supplied is None:
        if withheld:
            raise ConfigOverrideNeededError(
                f"{what} needs the config override values its record left out, "
                "because their name or their value looks secret: "
                f"{', '.join(withheld)}",
                hint=how,
            )
        return patch or None
    changes = overlay_changes(recorded, overlay_record(supplied))
    if changes is not None:
        names = sorted({*overlay_paths(patch), *withheld})
        raise ConfigOverrideNeededError(
            f"the config override given to {what} is not the recorded one "
            f"({changes}); the record sets {', '.join(names)}",
            hint=how,
        )
    return supplied


def validate_overlay(overlay: dict[str, Any]) -> dict[str, Any]:
    """Reject overlay sections outside the patchable allowlist (fail-closed).

    Scoring sections (``verifier``/``reward``/``solution``/``oracle``) and any
    other non-config section are disallowed so a per-run overlay can never weaken
    how the run is graded. ``artifacts`` entries are added to the task's own.
    Returns the overlay unchanged when it is legal.
    """
    illegal = set(overlay) - _PATCHABLE_SECTIONS
    if illegal:
        raise ValueError(
            f"config override may only patch {sorted(_PATCHABLE_SECTIONS)}; got "
            f"disallowed section(s) {sorted(illegal)} — the C axis is "
            "agent/sandbox/config, not the verifier/reward/scorer"
        )
    scales = overlay.get("timeout_multipliers")
    if scales is not None:
        if not isinstance(scales, dict) or set(scales) - set(_TIMEOUT_MULTIPLIERS):
            # Names only: the override's values are never in a message.
            given = sorted(map(str, scales)) if isinstance(scales, dict) else None
            raise ValueError(
                "config override timeout_multipliers takes "
                f"{sorted(_TIMEOUT_MULTIPLIERS)}, each a number above 0; got "
                + (f"keys {given}" if given is not None else type(scales).__name__)
            )
        for key, value in scales.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0
            ):
                # A number is shown; anything else by its type only.
                shown = (
                    value if isinstance(value, int | float) else type(value).__name__
                )
                raise ValueError(
                    f"config override timeout_multipliers.{key} must be a number "
                    f"above 0, got {shown}"
                )
    return overlay


#: The sandbox keys that make an overlay the operator's network policy.
_SANDBOX_NETWORK_KEYS = (
    "network_mode",
    "allow_internet",
    "allowed_hosts",
    "blocked_hosts",
    "blocked_urls",
)
#: Each network mode's openness: a phase key above the operator's is replaced.
_NETWORK_RANK = {"no-network": 0, "allowlist": 1, "denylist": 2, "public": 3}


def _mode(section: dict[str, Any]) -> str:
    """A merged section's network mode, ``allow_internet = false`` first."""
    if section.get("allow_internet") is False:
        return "no-network"
    mode = section.get("network_mode")
    mode = getattr(mode, "value", mode)
    return str(mode) if mode in _NETWORK_RANK else "public"


def _replace_phase_networks(
    overlay: dict[str, Any], merged: dict[str, Any]
) -> list[str]:
    """The operator's network wins over the task's phase keys: when *overlay*
    sets the sandbox's network, each task key of a phase (the verifier's
    ``network_mode``/``allowed_hosts``, a separate verifier sandbox's network,
    the agent's ``network_mode``/``allowed_hosts`` the sandbox folds in, and the
    same in each step) that is looser than the operator's policy, or a
    different allowlist, is set to the operator's in *merged*. A stricter phase
    key (an agent's own no-network) stays. Returns what was replaced, for the
    record; empty when the overlay sets no network."""
    asked = overlay.get("sandbox")
    if not isinstance(asked, dict) or not any(
        k in asked for k in _SANDBOX_NETWORK_KEYS
    ):
        return []
    sandbox = merged.get("sandbox") or {}
    policy = _mode(sandbox)
    hosts = list(sandbox.get("allowed_hosts") or [])
    set_by_operator = {
        section: set(overlay.get(section) or {})
        if isinstance(overlay.get(section), dict)
        else set()
        for section in ("agent", "verifier")
    }
    replaced: list[str] = []

    def hold(
        section: dict[str, Any] | None, where: str, operator_keys: set[str]
    ) -> None:
        if not isinstance(section, dict) or "network_mode" in operator_keys:
            return
        mode = section.get("network_mode")
        mode = getattr(mode, "value", mode)
        if mode is None and section.get("allow_internet") is not False:
            return
        own = _mode(section)
        looser = _NETWORK_RANK[own] > _NETWORK_RANK[policy]
        other_list = (
            own == policy == "allowlist"
            and list(section.get("allowed_hosts") or []) != hosts
        )
        if not (looser or other_list):
            return
        replaced.append(f"{where}.network_mode = {own}")
        section["network_mode"] = policy if mode is not None else None
        if "allow_internet" in section and section.get("allow_internet") is False:
            section["allow_internet"] = sandbox.get("allow_internet", True)
        if "allowed_hosts" in section:
            section["allowed_hosts"] = (
                hosts if policy == "allowlist" and mode is not None else None
            )
        for key in ("blocked_hosts", "blocked_urls"):
            if key in section:
                section[key] = (
                    list(sandbox.get(key) or []) if policy == "denylist" else []
                )

    hold(merged.get("agent"), "agent", set_by_operator["agent"])
    verifier = merged.get("verifier")
    hold(verifier, "verifier", set_by_operator["verifier"])
    if isinstance(verifier, dict):
        own_sandbox = verifier.get("sandbox")
        if isinstance(own_sandbox, dict):
            hold(own_sandbox, "verifier.sandbox", set())
    for index, step in enumerate(merged.get("steps") or ()):
        if isinstance(step, dict):
            hold(step.get("agent"), f"steps[{index}].agent", set())
            hold(step.get("verifier"), f"steps[{index}].verifier", set())
    return replaced


def operator_network_replacements(
    config: Any, overlay: dict[str, Any] | None
) -> list[str]:
    """What :func:`apply_config_override` replaces of *config*'s phase network
    keys under *overlay* (empty when the overlay sets no sandbox network):
    result.json's and the run card's record."""
    if not overlay:
        return []
    base = config.model_dump(by_alias=False)
    merged = deep_merge(
        base, {k: v for k, v in overlay.items() if k not in _COMPUTED_SECTIONS}
    )
    return _replace_phase_networks(overlay, merged)


def apply_config_override(config: Any, overlay: dict[str, Any] | None) -> Any:
    """Deep-merge a validated ``C``-axis ``overlay`` into a ``TaskConfig``.

    A no-op when ``overlay`` is falsy. Otherwise the overlay is allowlist-checked,
    deep-merged into ``config``, and the result is **re-validated** through
    ``TaskConfig`` so a bad overlay raises rather than silently corrupting the run.
    """
    if not overlay:
        return config

    validate_overlay(overlay)

    from benchflow.task.config import TaskConfig

    # Merge against the FIELD-NAME dump (``by_alias=False``) so overlays use the
    # canonical field names (``agent``, ``sandbox``, …); ``populate_by_name`` lets
    # re-validation accept them. (``sandbox`` is now the only spelling — the
    # legacy ``environment`` alias was removed in the rename — but ``oracle``
    # still serializes via alias, so the field-name dump stays load-bearing.)
    base = config.model_dump(by_alias=False)
    merged = deep_merge(
        base, {k: v for k, v in overlay.items() if k not in _COMPUTED_SECTIONS}
    )
    # An operator's sandbox network holds in every phase (operator_network_replacements).
    _replace_phase_networks(overlay, merged)
    for key in _ADDITIVE_SECTIONS & set(overlay):
        added = overlay[key]
        if not isinstance(added, list):
            raise ValueError(
                f"config override {key!r} must be a list, got {type(added).__name__}"
            )
        kept = list(base.get(key) or [])
        merged[key] = kept + [item for item in added if item not in kept]
    for key, scale in (overlay.get("timeout_multipliers") or {}).items():
        section, field = _TIMEOUT_MULTIPLIERS[key]
        merged[section] = dict(merged.get(section) or {})
        # The dump always holds the field (its default when the task sets none).
        merged[section][field] = float(merged[section][field]) * float(scale)
    try:
        patched = TaskConfig.model_validate(merged)
    except ValidationError as exc:
        invalid = _invalid(exc, overlay)
    else:
        invalid = None
    if invalid is not None:
        # Raised here, not in the except: pydantic's text, which quotes the
        # input, is neither the cause nor the context a traceback shows.
        raise invalid
    if "build" in (overlay.get("timeout_multipliers") or {}):
        # The scaled budget bounds the whole environment start
        # (review B2-M1); the image build keeps its own budget inside it.
        patched.sandbox._start_deadline_sec = patched.sandbox.build_timeout_sec
    logger.debug("config override applied: keys=%s", sorted(overlay))
    return patched
