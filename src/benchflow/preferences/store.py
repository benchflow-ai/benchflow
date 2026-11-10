"""A preference store: blind pairs of completed trials, the human votes on
them, and the ratings computed from those votes.

``issue`` reads finished trials, makes one item per trial output, adds the
outputs made outside BenchFlow that an items file names (a golden, a remix),
and pairs items of the same task version from different arms (an arm is an
agent and model, and the condition when an item has one), and writes the
store. The store keeps the private mapping from item to trial and arm;
``assignment.json`` shows a rater's tool only opaque ids, each task and each
output. ``import_votes`` checks a whole file of votes against the store and
appends them to ``votes.v1.jsonl``, one canonical JSON line each (RFC 8785),
each naming the hash of the line before it. A vote names a pair ``issue``
made, or the two outputs as a rater's tool showed them; the FDB vote site's
export is read as it comes. ``rank`` verifies that log and computes the
ratings (:mod:`.ratings`).

Votes are labels beside results, never rewards: nothing here writes a
trial's files, and no other command reads a store.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import random
import re
import shutil
import stat
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from benchflow.assessment.evidence import file_sha256
from benchflow.assessment.jcs import CanonicalizationError, canonicalize, loads_strict
from benchflow.preferences import ratings

STORE = "store.v1.json"
ITEMS = "items.v1.jsonl"
PAIRS = "pairs.v1.jsonl"
VOTES = "votes.v1.jsonl"
RUBRIC = "rubric.v1.json"
RATINGS = "ratings.v1.json"
# The outputs as the rater's tool sees them, each under its opaque id.
BUNDLE = "outputs"
LOCK = ".lock"
CHOICES = ("left", "right", "tie", "abstain")
_SCORE = {"left": 1.0, "right": 0.0, "tie": 0.5}
# Who cast a vote. A vote that doesn't say was cast by a person.
RATER_KINDS = ("person", "guest", "model")
# What a rater's tool showed beside the pair: the golden as the bar, or the
# brief alone.
AGAINST = ("golden", "brief")
# An item from an items file is an output, or a golden (a side, or the bar
# that is never paired: issue(golden=...)).
ITEM_KINDS = ("output", "golden")
GOLDEN_MODES = ("side", "bar")
_OUTSIDE_KEYS = frozenset(
    {
        "task",
        "task_digest",
        "output",
        "kind",
        "condition",
        "agent",
        "model",
        "harness_versions",
        "external_id",
        "source",
    }
)
# The Frontier Design Benchmark vote site's export. Its older exports carry
# the same fields under an earlier name, fdb-<name>-votes/v1, and those from
# before guest voting have no `invited`.
EXPORT_SCHEMA = "fdb-votes/v1"
_EXPORT_SCHEMA = re.compile(r"fdb-(?:[a-z0-9]+-)?votes/v1")
BY = ("agent", "model", "agent+model", "condition")
# Keys of a task.md [preference] table that issue() applies; the rest are
# recorded in the store as written.
APPLIED_PREFERENCE_KEYS = (
    "protocol",
    "randomize_sides",
    "raters_per_pair",
    "min_agreement",
)


class PreferenceError(ValueError):
    """A refusal: nothing was written."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


class Conflict(PreferenceError):
    """A vote id already recorded with other content."""


class UnknownOutput(PreferenceError):
    """A vote names an output the store doesn't have (skippable on import)."""


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    try:
        return canonicalize(value)
    except CanonicalizationError as exc:
        raise PreferenceError(f"not storable as canonical JSON: {exc}") from None


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _parse_time(value: Any, what: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise PreferenceError(f"{what} needs a time (ISO 8601, with a zone)")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        raise PreferenceError(f"{what} {value!r} is not an ISO 8601 time") from None
    if parsed.tzinfo is None:
        raise PreferenceError(f"{what} {value!r} has no time zone")
    return parsed


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return data if isinstance(data, dict) else {}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def output_manifest(output: Path) -> tuple[str, int]:
    """sha256 over an output folder's files (path, sha256 and size of each,
    sorted) and how many there are. Links and special files are refused."""
    members = []
    for root, dirs, files in os.walk(output):
        dirs.sort()
        for name in sorted(files):
            path = Path(root) / name
            relative = path.relative_to(output).as_posix()
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise PreferenceError(f"output {output} holds {relative}, not a file")
            digest, size = file_sha256(path, "output file")
            members.append({"path": relative, "sha256": digest, "bytes": size})
        for name in dirs:
            if (Path(root) / name).is_symlink():
                raise PreferenceError(f"output {output} holds a link: {name}")
    members.sort(key=lambda m: m["path"].encode("utf-8"))
    return _sha256(_canonical(members)), len(members)


def _trials(job: Path) -> list[Path]:
    from benchflow.eval_regrade import find_trials

    try:
        return find_trials(job)
    except FileNotFoundError:
        found = [
            trial
            for child in sorted(Path(job).iterdir())
            if child.is_dir()
            for trial in _trials_or_none(child)
        ]
        if not found:
            raise PreferenceError(f"no trials in {job}") from None
        return found


def _trials_or_none(path: Path) -> list[Path]:
    from benchflow.eval_regrade import find_trials

    try:
        return find_trials(path)
    except FileNotFoundError:
        return []


def _task_preference(
    trial: Path, task_name: str, digest: str
) -> tuple[dict[str, Any], str | None]:
    """The task's preference table, and why it was not used when it wasn't.

    Found through the job's recorded tasks folder: ``<tasks_dir>/<task>`` for
    a run of a folder of tasks, ``<tasks_dir>`` itself for a run of one task.
    It is used only while the folder is the version the trial ran (its task
    digest): a table edited since says nothing about that version. A task.md
    package carries it as ``[preference]``, a task.md or task.toml as
    ``preference:``.
    """
    from benchflow.task.discovery import is_task_dir
    from benchflow.taskmd import is_taskmd_package, parse_package

    job = trial.parent
    run = next(
        (p for p in (job, *job.parents) if (p / "evaluation.json").is_file()), None
    )
    tasks_dir = _read_json(run / "evaluation.json").get("tasks_dir") if run else None
    if not tasks_dir:
        return {}, None
    root = Path(tasks_dir)
    folder = next(
        (
            p
            for p in (root / task_name, root)
            if p.name == task_name and (is_taskmd_package(p) or is_task_dir(p))
        ),
        None,
    )
    if folder is None:
        return {}, None
    if is_taskmd_package(folder):
        table = parse_package(folder).config.get("preference")
    else:
        from benchflow.task.runtime_view import TaskRuntimeView

        try:
            table = TaskRuntimeView.from_task_dir(folder).config.preference
        except Exception as exc:  # a task that no longer loads has no table to use
            return {}, f"{task_name}: its task folder no longer loads ({exc})"
    if not table:
        return {}, None
    from benchflow._utils.task_authoring import task_digest

    if digest and task_digest(folder) != digest:
        return {}, (
            f"{task_name}: the task folder changed after the trial ran, so its "
            "preference table was not used"
        )
    return (dict(table) if isinstance(table, dict) else {}), None


def _rubric(path: Path) -> tuple[dict[str, Any], bytes]:
    data = path.read_bytes()
    try:
        rubric = loads_strict(data)
    except CanonicalizationError as exc:
        raise PreferenceError(f"rubric {path}: {exc}") from None
    criteria = rubric.get("criteria") if isinstance(rubric, dict) else None
    if not isinstance(criteria, list) or not criteria:
        raise PreferenceError(
            f"rubric {path} needs a list of criteria",
            hint='{"id": "...", "version": "...", "criteria": [{"id": "layout", "scale": [1, 5]}]}',
        )
    seen = set()
    for criterion in criteria:
        cid = criterion.get("id") if isinstance(criterion, dict) else None
        scale = criterion.get("scale") if isinstance(criterion, dict) else None
        if not isinstance(cid, str) or not cid or cid in seen:
            raise PreferenceError(f"rubric {path}: every criterion needs its own id")
        if not (
            isinstance(scale, list)
            and len(scale) == 2
            and all(
                isinstance(v, int | float) and not isinstance(v, bool) for v in scale
            )
            and scale[0] < scale[1]
        ):
            raise PreferenceError(
                f"rubric {path}: criterion {cid!r} needs a scale [low, high]"
            )
        seen.add(cid)
    return rubric, data


def _outside_items(path: Path) -> list[dict[str, Any]]:
    """The rows of an items file (JSON lines): each an output made outside
    BenchFlow, such as a golden or a remix, with the task it answers and the
    labels its arm is rated under. Every row is checked before anything is
    made."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise PreferenceError(f"items file {path}: {exc}") from None
    rows: list[dict[str, Any]] = []
    externals: dict[str, int] = {}
    outputs: dict[str, int] = {}
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        where = f"{path} line {number}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PreferenceError(f"{where}: not JSON ({exc.msg})") from None
        if not isinstance(row, dict):
            raise PreferenceError(f"{where}: not a JSON object")
        unknown = sorted(set(row) - _OUTSIDE_KEYS)
        if unknown:
            raise PreferenceError(
                f"{where}: unknown key {unknown[0]!r}",
                hint="an item takes " + ", ".join(sorted(_OUTSIDE_KEYS)),
            )
        for key in (
            "task",
            "task_digest",
            "condition",
            "agent",
            "model",
            "external_id",
            "source",
        ):
            if row.get(key) is not None and not isinstance(row[key], str):
                raise PreferenceError(f"{where}: {key} must be a string")
        if not (row.get("task") or "").strip():
            raise PreferenceError(f"{where}: an item needs a task, the task it answers")
        if not (row.get("agent") or "").strip():
            raise PreferenceError(
                f"{where}: an item needs an agent, the label its arm is rated under"
            )
        kind = row.get("kind", "output")
        if kind not in ITEM_KINDS:
            raise PreferenceError(
                f"{where}: kind must be output or golden, not {kind!r}"
            )
        harness = row.get("harness_versions")
        if harness is not None and not isinstance(harness, dict):
            raise PreferenceError(f"{where}: harness_versions must be an object")
        output = row.get("output")
        if not isinstance(output, str) or not output or not Path(output).is_dir():
            raise PreferenceError(f"{where}: no folder at output {output!r}")
        folder = str(Path(output).resolve())
        if folder in outputs:
            raise PreferenceError(
                f"{where}: output {output!r} is listed twice (first on line {outputs[folder]})"
            )
        outputs[folder] = number
        external = row.get("external_id") or None
        if external is not None:
            if external in externals:
                raise PreferenceError(
                    f"{where}: external_id {external!r} is used twice (first on "
                    f"line {externals[external]})"
                )
            externals[external] = number
        rows.append(
            {
                "line": number,
                "task": row["task"].strip(),
                "task_digest": row.get("task_digest"),
                "output": folder,
                "kind": kind,
                "condition": (row.get("condition") or "").strip(),
                "agent": row["agent"].strip(),
                "model": (row.get("model") or "").strip(),
                "harness_versions": harness,
                "external_id": external,
                "source": row.get("source") or "",
            }
        )
    return rows


def issue(
    jobs: Sequence[Path],
    store: Path,
    *,
    output: str = "artifacts",
    per_task: int | None = None,
    seed: int | None = None,
    rubric: Path | None = None,
    bundle: Path | None = None,
    randomize_sides: bool | None = None,
    items: Path | None = None,
    condition: str | None = None,
    entry: str | None = None,
    golden: str = "side",
) -> dict[str, Any]:
    """Make the store and return the blind assignment for a rater's tool.

    ``items`` names a JSON-lines file of outputs made outside BenchFlow
    (:func:`_outside_items`); each joins the trials of its task's version, or
    pairs with the other outside outputs of a task no trial ran. ``condition``
    labels every trial's output (``one-shot``, say), ``entry`` is the page to
    open in each output (an output without it is left out), and ``golden``
    says whether a golden is a side (``side``) or only shown as the bar
    (``bar``: an item, never in a pair)."""
    store = Path(store)
    if (store / STORE).exists():
        raise PreferenceError(
            f"{store} already holds an assignment",
            hint="issue a new assignment into a new --store folder",
        )
    if golden not in GOLDEN_MODES:
        raise PreferenceError(f"golden must be side or bar, not {golden!r}")
    if entry is not None:
        page = PurePosixPath(entry)
        if not entry or page.is_absolute() or ".." in page.parts:
            raise PreferenceError(
                f"entry {entry!r} must be a path inside each output, such as index.html"
            )
    if not jobs and items is None:
        raise PreferenceError("name job folders, an items file, or both")
    outside_rows = _outside_items(Path(items)) if items is not None else []
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big")
    rng = random.Random(seed)
    made: list[dict[str, Any]] = []
    left_out: list[dict[str, Any]] = []
    protocols: dict[tuple[str, str], dict[str, Any]] = {}
    protocol_notes: list[str] = []
    for job in jobs:
        job = Path(job).resolve()
        for trial in _trials(job):
            result = _read_json(trial / "result.json")
            out = trial / output
            files = output_manifest(out)[1] if out.is_dir() else 0
            reason = (
                "unfinished"
                if not result.get("finished_at")
                # An agent that never ran left whatever was there before it.
                else "errored before a reward"
                if result.get("error") and result.get("rewards") is None
                else f"no files in {output}/"
                if not files
                else f"no {entry} in the output"
                if entry is not None and not (out / entry).is_file()
                # Without its digest a trial's task version can't be told,
                # and pairs only compare outputs of one version.
                else "no task digest recorded"
                if not result.get("task_digest")
                else None
            )
            if reason is not None:
                left_out.append(
                    {
                        "trial": str(trial),
                        "agent": str(
                            result.get("agent") or result.get("agent_name") or ""
                        ),
                        "model": str(result.get("model") or ""),
                        "reason": reason,
                    }
                )
                continue
            task = str(result.get("task_name") or trial.name)
            digest, count = output_manifest(out)
            # The registry name (what -a takes), not the name the agent reports.
            agent = str(result.get("agent") or result.get("agent_name") or "")
            model = str(result.get("model") or "")
            made.append(
                {
                    "item_id": "i-"
                    + hashlib.sha256(f"{seed}:{trial}".encode()).hexdigest()[:16],
                    "task": task,
                    "task_digest": str(result.get("task_digest") or ""),
                    "trial": str(trial),
                    "origin": "trial",
                    "kind": "output",
                    "condition": condition or "",
                    "agent": agent,
                    "harness_versions": result.get("harness_versions"),
                    "model": model,
                    "output": str(out),
                    "output_sha256": digest,
                    "files": count,
                }
            )
            version = (task, str(result.get("task_digest") or ""))
            if version not in protocols:
                protocols[version], note = _task_preference(trial, *version)
                if note:
                    protocol_notes.append(note)
    # An outside output answers the task version its task's trials ran; a task
    # no trial ran is a version of its own, shared by its outside outputs.
    versions_of: dict[str, set[str]] = defaultdict(set)
    for item in made:
        versions_of[item["task"]].add(item["task_digest"])
    for row in outside_rows:
        where = f"{items} line {row['line']}"
        digest = row["task_digest"]
        if digest is None:
            known = versions_of.get(row["task"], set())
            if len(known) > 1:
                raise PreferenceError(
                    f"{where}: the trials of task {row['task']!r} ran {len(known)} "
                    "task versions; say which one this output answers with task_digest"
                )
            digest = next(iter(known), "")
        out = Path(row["output"])
        out_digest, count = output_manifest(out)
        reason = (
            "no files in the output"
            if not count
            else f"no {entry} in the output"
            if entry is not None and not (out / entry).is_file()
            else None
        )
        if reason is not None:
            left_out.append(
                {
                    "output": str(out),
                    "agent": row["agent"],
                    "model": row["model"],
                    "reason": reason,
                }
            )
            continue
        key = row["external_id"] or f"{row['line']}:{out}"
        made.append(
            {
                "item_id": "i-"
                + hashlib.sha256(f"{seed}:outside:{key}".encode()).hexdigest()[:16],
                "task": row["task"],
                "task_digest": digest,
                "trial": None,
                "origin": "outside",
                "kind": row["kind"],
                "condition": row["condition"],
                "agent": row["agent"],
                "harness_versions": row["harness_versions"],
                "model": row["model"],
                "external_id": row["external_id"],
                "source": row["source"],
                "output": str(out),
                "output_sha256": out_digest,
                "files": count,
            }
        )
        protocols.setdefault((row["task"], digest), {})
    if not made:
        raise PreferenceError(
            "no finished trial or listed output has files to compare",
            hint="name the folder that holds each trial's output with --output",
        )
    for (task, _digest), table in protocols.items():
        protocol = table.get("protocol", "pairwise")
        if protocol not in ("pairwise", "rubric"):
            raise PreferenceError(
                f"task {task!r} asks for the {protocol!r} preference protocol; "
                "bench preferences runs pairwise and rubric"
            )
    by_version: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in sorted(made, key=lambda i: i["item_id"]):
        by_version[(item["task"], item["task_digest"])].append(item)
    issued_at = _now()
    pairs: list[dict[str, Any]] = []
    rubric_only = {
        version
        for version, table in protocols.items()
        if table.get("protocol") == "rubric"
    }

    def bar(item: dict[str, Any]) -> bool:
        return golden == "bar" and item.get("kind") == "golden"

    sides: set[bool] = set()
    for (task, digest), group in sorted(by_version.items()):
        table = protocols.get((task, digest), {})
        if (task, digest) in rubric_only:
            # A rubric task's outputs are scored one at a time: no pairs.
            continue
        shuffle = (
            table.get("randomize_sides", True)
            if randomize_sides is None
            else randomize_sides
        )
        sides_of = [item for item in group if not bar(item)]
        candidates = [
            (a, b)
            for i, a in enumerate(sides_of)
            for b in sides_of[i + 1 :]
            if _arm(a, "agent+model") != _arm(b, "agent+model")
        ]
        if per_task is not None and len(candidates) > per_task:
            candidates = rng.sample(candidates, per_task)
        if candidates:
            sides.add(bool(shuffle))
        for a, b in candidates:
            left, right = (b, a) if shuffle and rng.random() < 0.5 else (a, b)
            pair_id = (
                "p-"
                + hashlib.sha256(
                    f"{seed}:{left['item_id']}:{right['item_id']}".encode()
                ).hexdigest()[:16]
            )
            pairs.append(
                {
                    "pair_id": pair_id,
                    "task": task,
                    "task_digest": digest,
                    "left": left["item_id"],
                    "right": right["item_id"],
                    "issued_at": issued_at,
                }
            )
    paired = {pair[side] for pair in pairs for side in ("left", "right")}
    left_out += [
        {
            **(
                {"trial": item["trial"]}
                if item["trial"] is not None
                else {"output": item["output"]}
            ),
            "agent": item["agent"],
            "model": item["model"],
            "reason": "no output of another agent or model on its task version",
        }
        for item in made
        if item["item_id"] not in paired
        and not bar(item)
        and (item["task"], item["task_digest"]) not in rubric_only
    ]
    frozen = None
    if rubric is not None:
        frozen, rubric_bytes = _rubric(Path(rubric))
    store.mkdir(parents=True, exist_ok=True)
    if frozen is not None:
        (store / RUBRIC).write_bytes(rubric_bytes)
    _write_jsonl(store / ITEMS, made)
    _write_jsonl(store / PAIRS, pairs)
    # The rater's tool gets copies under opaque ids: a trial's own folder
    # names its job, and its result.json the agent.
    bundle = (
        Path(bundle) if bundle is not None else store.parent / f"{store.name}-{BUNDLE}"
    )
    paths: dict[str, str] = {}
    # One time on every copied file: a trial's own file times date its run,
    # and so can tell an arm apart.
    stamp = datetime.fromisoformat(issued_at.replace("Z", "+00:00")).timestamp()
    for item in made:
        copy = bundle / item["item_id"]
        shutil.copytree(item["output"], copy, symlinks=True)
        for root, dirs, files in os.walk(copy, topdown=False):
            for name in [*files, *dirs]:
                os.utime(Path(root) / name, (stamp, stamp), follow_symlinks=False)
        os.utime(copy, (stamp, stamp))
        paths[item["item_id"]] = str(copy.resolve())
    manifest = {
        "schema": "benchflow.preference-store.v1",
        "created_at": issued_at,
        "seed": seed,
        "jobs": [str(Path(j).resolve()) for j in jobs],
        "output": output,
        "items_sha256": _sha256((store / ITEMS).read_bytes()),
        "pairs_sha256": _sha256((store / PAIRS).read_bytes()),
        "rubric_sha256": _sha256((store / RUBRIC).read_bytes()) if frozen else None,
        "protocol": {
            f"{task} {digest}": table
            for (task, digest), table in sorted(protocols.items())
            if table
        },
        "protocol_notes": protocol_notes,
        "rubric_only_tasks": sorted(task for task, _digest in rubric_only),
        # "fixed" when randomize_sides is false everywhere: the left-side
        # check then says nothing about the raters.
        "sides": "shuffled"
        if sides == {True}
        else "fixed"
        if sides == {False}
        else "mixed"
        if sides
        else None,
        "bundle": str(bundle),
        "left_out": left_out,
        "origins": {
            "trial": sum(item["origin"] == "trial" for item in made),
            "outside": sum(item["origin"] == "outside" for item in made),
        },
        "items_file": str(Path(items).resolve()) if items is not None else None,
        "items_file_sha256": _sha256(Path(items).read_bytes())
        if items is not None
        else None,
        "condition": condition or None,
        "entry": entry,
        "golden": golden,
    }
    (store / STORE).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    shown = {item["item_id"]: item for item in made}
    return {
        "schema": "benchflow.preference-assignment.v1",
        "store": manifest["pairs_sha256"],
        "rubric": frozen,
        "items": [
            {
                "item_id": item["item_id"],
                "task": item["task"],
                "path": paths[item["item_id"]],
                "output_sha256": item["output_sha256"],
                **(
                    {
                        "entry": entry,
                        "page": str(Path(paths[item["item_id"]]) / entry),
                    }
                    if entry is not None
                    else {}
                ),
            }
            for item in sorted(shown.values(), key=lambda i: i["item_id"])
        ],
        "pairs": [
            {k: pair[k] for k in ("pair_id", "task", "left", "right")} for pair in pairs
        ],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


class _Store:
    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        manifest = self.dir / STORE
        if not manifest.is_file():
            raise PreferenceError(
                f"{self.dir} is not a preference store",
                hint="`bench preferences pairs` makes one",
            )
        self.manifest = _read_json(manifest)
        edited = "the store was edited after `bench preferences pairs` made it: make a new one"
        for name, key in ((ITEMS, "items_sha256"), (PAIRS, "pairs_sha256")):
            if _sha256((self.dir / name).read_bytes()) != self.manifest.get(key):
                raise PreferenceError(
                    f"{self.dir / name} changed after it was issued", hint=edited
                )
        if self.manifest.get("rubric_sha256") and (
            not (self.dir / RUBRIC).is_file()
            or _sha256((self.dir / RUBRIC).read_bytes())
            != self.manifest["rubric_sha256"]
        ):
            raise PreferenceError(
                f"{self.dir / RUBRIC} changed after it was issued", hint=edited
            )
        self.items = {i["item_id"]: i for i in _jsonl(self.dir / ITEMS)}
        self.pairs = {p["pair_id"]: p for p in _jsonl(self.dir / PAIRS)}
        # An outside output can be named by the id its own tool gave it.
        self.by_external = {
            i["external_id"]: i["item_id"]
            for i in self.items.values()
            if i.get("external_id")
        }
        # The pair issue() made of two items, whichever side each was shown on.
        self.pair_of = {
            frozenset((p["left"], p["right"])): p for p in self.pairs.values()
        }
        self.rubric = (
            json.loads((self.dir / RUBRIC).read_text())
            if self.manifest.get("rubric_sha256")
            else None
        )
        self.criteria = (
            {c["id"]: c["scale"] for c in self.rubric["criteria"]}
            if self.rubric
            else {}
        )

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        fd = os.open(self.dir / LOCK, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
        try:
            fcntl.lockf(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def log(self) -> list[dict[str, Any]]:
        """Every vote, after checking each line's form, hash and link."""
        path = self.dir / VOTES
        if not path.is_file():
            return []
        data = path.read_bytes()
        hint = f"{VOTES} was changed outside `bench preferences import`: restore it from a copy"
        if data and not data.endswith(b"\n"):
            raise PreferenceError(
                f"{path}: the last line is torn (no newline)", hint=hint
            )
        votes, previous = [], None
        for number, line in enumerate(data.splitlines(), 1):
            try:
                vote = loads_strict(line)
            except CanonicalizationError as exc:
                raise PreferenceError(
                    f"{path} line {number}: {exc}", hint=hint
                ) from None
            body = {k: v for k, v in vote.items() if k != "vote_sha256"}
            if canonicalize(vote) != line:
                raise PreferenceError(
                    f"{path} line {number} is not canonical JSON", hint=hint
                )
            if vote.get("vote_sha256") != _sha256(canonicalize(body)):
                raise PreferenceError(
                    f"{path} line {number}: its hash does not match", hint=hint
                )
            if vote.get("seq") != number or vote.get("previous_sha256") != previous:
                raise PreferenceError(
                    f"{path} line {number} breaks the chain", hint=hint
                )
            previous = vote["vote_sha256"]
            votes.append(vote)
        return votes


def _sides(store: _Store, vote: dict[str, Any]) -> tuple[str, str]:
    """The items a pairwise vote's left and right were: the pair's sides for
    a vote on a pair issue() made, the sides as shown for a vote naming them."""
    if vote.get("pair_id"):
        pair = store.pairs[vote["pair_id"]]
        return pair["left"], pair["right"]
    return vote["shown_left"], vote["shown_right"]


def _export_rows(
    data: dict[str, Any], path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The vote rows of an FDB vote site export as import rows, and what the
    export says about itself: the site and surface it came from, and when it
    was made."""
    schema = data["schema"]
    collector = data.get("collector")
    cases = collector.get("cases") if isinstance(collector, dict) else None
    if not isinstance(cases, list):
        raise PreferenceError(
            f"{path}: an {schema} export needs collector.cases, a list"
        )
    named = data.get("source")
    site: dict[str, Any] = named if isinstance(named, dict) else {}
    # One site, whichever tool exported it: "https://x/" and "https://x" alike.
    label = " ".join(
        v
        for v in (
            str(site.get("site") or "").rstrip("/"),
            str(site.get("surface") or ""),
        )
        if v
    )
    rows: list[dict[str, Any]] = []
    for case in cases:
        votes = case.get("votes") if isinstance(case, dict) else None
        if not isinstance(votes, list):
            raise PreferenceError(
                f"{path}: a case in collector.cases has no list of votes"
            )
        for vote in votes:
            if not isinstance(vote, dict):
                raise PreferenceError(
                    f"{path}: a vote in collector.cases is not an object"
                )
            choice = str(vote.get("choice") or "")
            row: dict[str, Any] = {
                "left_item": vote.get("leftId"),
                "right_item": vote.get("rightId"),
                # A skip on the site is an abstention here: recorded, never counted.
                "choice": "abstain" if choice == "skip" else choice,
                "rater": vote.get("reviewerId") or vote.get("reviewer"),
                "rated_at": vote.get("createdAt"),
                "source": label or str(schema),
                # An export from before guest voting has no `invited`: every
                # voter then came through a personal link.
                "rater_kind": "person"
                if vote.get("invited", True) is not False
                else "guest",
            }
            # The site keeps one vote per voter and pair under one id; a
            # changed vote keeps the id and gets a later time, whole seconds,
            # so the choice tells apart an Undo and a new answer in one second.
            if vote.get("id"):
                row["vote_id"] = f"{vote['id']}@{vote.get('createdAt')}/{choice}"
            if "invited" in vote:
                row["invited"] = vote["invited"]
            if vote.get("judgeAgainst"):
                row["against"] = vote["judgeAgainst"]
            if vote.get("reason"):
                row["reason"] = vote["reason"]
            if vote.get("tags"):
                row["tags"] = vote["tags"]
            rows.append({k: v for k, v in row.items() if v is not None})
    return rows, {"source": label or str(schema), "exported_at": data.get("exportedAt")}


def _rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """The rows of a votes file: JSON lines, CSV, or the FDB vote site's
    export (one JSON document; then the second value says where it came
    from and when, else it is None)."""
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() == ".csv":
        import csv

        return [
            {k: v for k, v in row.items() if v not in (None, "")}
            for row in csv.DictReader(text.splitlines())
        ], None
    try:
        whole = json.loads(text)
    except json.JSONDecodeError:
        whole = None
    if (
        isinstance(whole, dict)
        and isinstance(whole.get("schema"), str)
        and _EXPORT_SCHEMA.fullmatch(whole["schema"])
    ):
        return _export_rows(whole, Path(path))
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PreferenceError(
                f"{path} line {number} is not JSON ({exc.msg})",
                hint="votes are JSON lines, a CSV file, or a vote site export "
                f"({EXPORT_SCHEMA})",
            ) from None
        if not isinstance(row, dict):
            raise PreferenceError(f"{path} line {number} is not a JSON object")
        rows.append(row)
    return rows, None


def _item_named(store: _Store, name: Any, where: str, what: str) -> dict[str, Any]:
    """The item an id or an outside output's own id names."""
    key = str(name) if name is not None else ""
    item = store.items.get(key) or store.items.get(store.by_external.get(key, ""))
    if item is None:
        raise UnknownOutput(f"{where}: {what} {name!r} was not issued")
    return item


def _vote(
    store: _Store, row: dict[str, Any], where: str, source: str | None
) -> dict[str, Any]:
    rater = str(row.get("rater") or "").strip()
    if not rater:
        raise PreferenceError(f"{where}: no rater")
    rated = _parse_time(row.get("rated_at"), f"{where}: rated_at")
    shown = "left_item" in row or "right_item" in row
    if shown and row.get("pair_id"):
        raise PreferenceError(
            f"{where}: a vote names either a pair_id or the left_item and "
            "right_item it showed, not both"
        )
    kind = row.get("kind") or ("pairwise" if "pair_id" in row or shown else "rubric")
    body: dict[str, Any] = {
        "kind": kind,
        "rater": rater,
        "rated_at": rated.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    }
    if kind == "pairwise":
        choice = str(row.get("choice") or "").strip().lower()
        if choice not in CHOICES:
            raise PreferenceError(
                f"{where}: choice must be left, right, tie or abstain"
            )
        aspect = row.get("aspect") or None
        if aspect is not None and aspect not in store.criteria:
            raise PreferenceError(
                f"{where}: aspect {aspect!r} is not a criterion of the rubric"
            )
        if shown:
            left = _item_named(store, row.get("left_item"), where, "output")
            right = _item_named(store, row.get("right_item"), where, "output")
            if left["item_id"] == right["item_id"]:
                raise PreferenceError(
                    f"{where}: output {left['item_id']} can't be compared with itself"
                )
            if (left["task"], left["task_digest"]) != (
                right["task"],
                right["task_digest"],
            ):
                raise PreferenceError(
                    f"{where}: outputs {left['item_id']} and {right['item_id']} "
                    "answer different task versions, so they are not a pair"
                )
            if rated < _parse_time(store.manifest["created_at"], "created_at"):
                raise PreferenceError(f"{where}: rated before the items were issued")
            body.update(
                shown_left=left["item_id"],
                shown_right=right["item_id"],
                aspect=aspect,
                choice=choice,
            )
            identity = {
                k: body[k]
                for k in (
                    "kind",
                    "rater",
                    "rated_at",
                    "shown_left",
                    "shown_right",
                    "aspect",
                )
            }
        else:
            pair = store.pairs.get(str(row.get("pair_id")))
            if pair is None:
                raise PreferenceError(
                    f"{where}: pair {row.get('pair_id')!r} was not issued"
                )
            if rated < _parse_time(pair["issued_at"], "issued_at"):
                raise PreferenceError(f"{where}: rated before the pair was issued")
            body.update(pair_id=pair["pair_id"], aspect=aspect, choice=choice)
            # The fields 0.10.0rc2 made a vote's id from, so its votes files
            # import again without adding anything.
            identity = {
                k: body[k] for k in ("kind", "rater", "rated_at", "pair_id", "aspect")
            }
    elif kind == "rubric":
        item = _item_named(store, row.get("item_id"), where, "item")
        criterion = row.get("criterion")
        if criterion not in store.criteria:
            raise PreferenceError(
                f"{where}: criterion {criterion!r} is not in the rubric"
            )
        raw = row.get("score")
        try:
            if isinstance(raw, bool) or not isinstance(raw, int | float | str):
                raise ValueError
            score = float(raw)
        except ValueError:
            raise PreferenceError(f"{where}: score must be a number") from None
        if not math.isfinite(score):
            raise PreferenceError(f"{where}: score must be a finite number")
        low, high = store.criteria[criterion]
        if not low <= score <= high:
            raise PreferenceError(f"{where}: score {score} is outside [{low}, {high}]")
        if rated < _parse_time(store.manifest["created_at"], "created_at"):
            raise PreferenceError(f"{where}: rated before the items were issued")
        body.update(item_id=item["item_id"], criterion=criterion, score=score)
        identity = {
            k: body[k] for k in ("kind", "rater", "rated_at", "item_id", "criterion")
        }
    else:
        raise PreferenceError(f"{where}: kind must be pairwise or rubric")
    # Who cast the vote and what the rater's tool showed: recorded with the
    # vote, outside its id.
    if "rater_kind" in row:
        if row["rater_kind"] not in RATER_KINDS:
            raise PreferenceError(
                f"{where}: rater_kind must be person, guest or model, not "
                f"{row['rater_kind']!r}"
            )
        body["rater_kind"] = row["rater_kind"]
    if "invited" in row:
        if not isinstance(row["invited"], bool):
            raise PreferenceError(f"{where}: invited must be true or false")
        body["invited"] = row["invited"]
    if "against" in row:
        if row["against"] not in AGAINST:
            raise PreferenceError(
                f"{where}: against must be golden or brief, not {row['against']!r}"
            )
        body["against"] = row["against"]
    if row.get("tags"):
        tags = row["tags"]
        if not (
            isinstance(tags, list)
            and all(isinstance(t, str) and 0 < len(t) <= 80 for t in tags)
        ):
            raise PreferenceError(f"{where}: tags must be a list of short words")
        body["tags"] = list(tags)
    if row.get("reason"):
        body["reason"] = str(row["reason"])
    body["source"] = str(row.get("source") or source or "")
    body["vote_id"] = str(row.get("vote_id") or _sha256(_canonical(identity))[7:23])
    return body


def _withdrawals(
    store: _Store,
    log: list[dict[str, Any]],
    checked: list[dict[str, Any]],
    export: dict[str, Any],
) -> list[dict[str, Any]]:
    """A vote the site no longer has (its voter pressed Undo) stops counting:
    for each rater and pair whose latest vote came from this export's site,
    was cast before the export was made, and is missing from it, an
    abstention dated when the export was made. An export without its time,
    or with no vote on the store's outputs (a site whose store was emptied or
    replaced), withdraws nothing; a time that can't be read refuses the file."""
    if export.get("exported_at") is None:
        return []
    at = (
        _parse_time(export.get("exported_at"), "exportedAt")
        .astimezone(UTC)
        .strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    )
    if not checked:
        return []
    present = {
        (v["rater"], frozenset(_sides(store, v)), v.get("aspect"))
        for v in checked
        if v["kind"] == "pairwise"
    }
    out = []
    for vote in _effective(log, store):
        if (
            vote["kind"] != "pairwise"
            or vote.get("source") != export["source"]
            or vote["choice"] == "abstain"
            # An older export imported again can't speak for later votes.
            or vote["rated_at"] > at
            or (vote["rater"], frozenset(_sides(store, vote)), vote.get("aspect"))
            in present
        ):
            continue
        left, right = _sides(store, vote)
        withdrawal = {
            "kind": "pairwise",
            "rater": vote["rater"],
            "rated_at": at,
            "shown_left": left,
            "shown_right": right,
            "aspect": vote.get("aspect"),
            "choice": "abstain",
            **{k: vote[k] for k in ("rater_kind", "invited", "against") if k in vote},
            "reason": f"withdrawn: the export of {export.get('exported_at')} "
            "no longer has this vote",
            "source": export["source"],
            "vote_id": f"withdrawn:{vote['vote_id']}",
        }
        out.append(withdrawal)
    return out


def import_votes(
    store_dir: Path,
    votes: Path,
    *,
    source: str | None = None,
    skip_unknown: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Check every row of ``votes``, then append the new ones. Returns how
    many were added and how many were already recorded, and the log's last
    hash (``head_sha256``).

    ``votes`` is JSON lines, CSV, or the FDB vote site's export, whose votes
    name the two outputs each voter saw; importing a later export also
    records, for each vote it no longer has, that the voter withdrew it.
    ``skip_unknown`` leaves out votes on outputs the store lacks instead of
    refusing the file, and ``dry_run`` checks everything and writes
    nothing."""
    store = _Store(store_dir)
    rows, export = _rows(votes)
    if not rows and export is None:
        raise PreferenceError(f"{votes} holds no votes")
    checked: list[dict[str, Any]] = []
    skipped = 0
    for number, row in enumerate(rows, 1):
        try:
            checked.append(_vote(store, row, f"{votes} vote {number}", source))
        except UnknownOutput:
            if not skip_unknown:
                raise
            skipped += 1
    touched: set[str] = set()
    for vote in checked:
        if vote["kind"] == "pairwise":
            touched.update(_sides(store, vote))
        else:
            touched.add(vote["item_id"])
    shown_dir = Path(
        store.manifest.get("bundle")
        or (store.dir.parent / f"{store.dir.name}-{BUNDLE}")
    )
    for item_id in sorted(touched):
        item = store.items[item_id]
        if not Path(item["output"]).is_dir():
            raise PreferenceError(
                f"the output of item {item_id} is missing ({item['output']})",
                hint="pairs reads each output where the trial wrote it: put the jobs"
                " folder back where it was, or make new pairs into a new store;"
                " nothing was added",
            )
        # The copy raters saw must still be the output that was issued.
        shown = shown_dir / item_id
        if not shown.is_dir() or output_manifest(shown)[0] != item["output_sha256"]:
            raise PreferenceError(
                f"the copy of item {item_id} that raters see is missing or changed ({shown})",
                hint="votes cast while it was changed rated what the raters saw, not"
                " the trial's output: make new pairs into a new store with `bench"
                " preferences pairs` and have them rated again (restore the copy from"
                " the trial's output only if every vote came before the change);"
                " nothing was added",
            )
        if output_manifest(Path(item["output"]))[0] != item["output_sha256"]:
            raise PreferenceError(
                f"the output of item {item_id} changed after it was issued ({item['output']})",
                hint="restore that output, or make new pairs into a new store with"
                " `bench preferences pairs` (votes on these pairs no longer apply to"
                " it); nothing was added",
            )
    with store.locked():
        log = store.log()
        recorded = {v["vote_id"]: v for v in log}
        new: list[dict[str, Any]] = []
        repeats = 0
        for number, vote in enumerate(checked, 1):
            known = recorded.get(vote["vote_id"]) or next(
                (n for n in new if n["vote_id"] == vote["vote_id"]), None
            )
            if known is None:
                new.append(vote)
                continue
            # Where a vote came from is provenance, not content: the same
            # vote named by another export tool is already recorded.
            kept = {
                k: v
                for k, v in known.items()
                if k
                not in (
                    "seq",
                    "previous_sha256",
                    "recorded_at",
                    "vote_sha256",
                    "source",
                )
            }
            if kept != {k: v for k, v in vote.items() if k != "source"}:
                target = (
                    vote.get("pair_id")
                    or vote.get("item_id")
                    or (f"{vote.get('shown_left')} and {vote.get('shown_right')}")
                )
                raise Conflict(
                    f"{votes} vote {number} (rater {vote['rater']}, {target}): its "
                    f"vote_id {vote['vote_id']} (the file's, or one made from the "
                    "pair or item, the aspect, the rater and rated_at) is already "
                    "recorded with other content",
                    hint="to change an answer, import it with a later rated_at and "
                    "without that vote_id (or with a new one): a rater's latest vote "
                    "on a pair or item is the one counted; nothing was added",
                )
            repeats += 1
        withdrawn = 0
        if export is not None:
            for withdrawal in _withdrawals(store, log, checked, export):
                if withdrawal["vote_id"] not in recorded:
                    new.append(withdrawal)
                    withdrawn += 1
        previous = log[-1]["vote_sha256"] if log else None
        counts = {
            "added": len(new) - withdrawn,
            "already_recorded": repeats,
            "withdrawn": withdrawn,
            "skipped_unknown": skipped,
            "dry_run": dry_run,
        }
        if dry_run:
            return {**counts, "head_sha256": previous}
        lines = []
        for seq, vote in enumerate(new, len(log) + 1):
            entry = {
                **vote,
                "seq": seq,
                "previous_sha256": previous,
                "recorded_at": _now(),
            }
            entry["vote_sha256"] = _sha256(_canonical(entry))
            previous = entry["vote_sha256"]
            lines.append(_canonical(entry) + b"\n")
        if lines:
            fd = os.open(
                store.dir / VOTES,
                os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC,
                0o600,
            )
            try:
                os.write(fd, b"".join(lines))
                os.fsync(fd)
            finally:
                os.close(fd)
    return {**counts, "head_sha256": previous}


def _effective(votes: list[dict[str, Any]], store: _Store) -> list[dict[str, Any]]:
    """Each rater's latest vote on each pair and aspect (or item and
    criterion); a later vote replaces an earlier one. A pair is its two
    outputs, whichever side each was shown on, except for a model judge:
    its looks at a pair each way round are both kept, so whether it answers
    the same both ways can be read."""
    latest: dict[tuple, dict[str, Any]] = {}
    for vote in votes:
        if vote["kind"] == "pairwise":
            sides = _sides(store, vote)
            key = (
                "pairwise",
                frozenset(sides),
                vote["aspect"],
                vote["rater"],
                sides if _kind(vote) == "model" else None,
            )
        else:
            key = ("rubric", vote["item_id"], vote["criterion"], vote["rater"])
        held = latest.get(key)
        if held is None or (vote["rated_at"], vote["seq"]) > (
            held["rated_at"],
            held["seq"],
        ):
            latest[key] = vote
    return list(latest.values())


def _shared(manifest: dict[str, Any], key: str) -> Any:
    """A [preference] setting every task that sets it agrees on, else None
    (the command line then decides)."""
    values = {
        json.dumps(table[key], sort_keys=True)
        for table in manifest.get("protocol", {}).values()
        if key in table
    }
    return json.loads(values.pop()) if len(values) == 1 else None


def _arm(item: dict[str, Any], by: str) -> str:
    if by == "agent":
        return item["agent"] or "-"
    if by == "model":
        return item["model"] or "-"
    if by == "condition":
        return item.get("condition") or "-"
    label = f"{item['agent'] or '-'} / {item['model'] or '-'}"
    # Two outputs of one agent and model under different conditions (one-shot,
    # with references) are different arms.
    if item.get("condition"):
        label += f" [{item['condition']}]"
    receipt = item.get("harness_versions")
    if receipt is not None:
        label += " / harness-" + hashlib.sha256(_canonical(receipt)).hexdigest()[:16]
    return label


def _kind(vote: dict[str, Any]) -> str:
    """Who cast a vote: as the vote says, else a model judge when its rater is
    named ``model:<name>`` (as the docs' judge script names it), else a
    person."""
    kind = vote.get("rater_kind")
    if kind in RATER_KINDS:
        return kind
    return "model" if str(vote.get("rater", "")).startswith("model:") else "person"


KIND_NAMES = {"person": "people", "guest": "guests", "model": "model judges"}


def _chosen(store: _Store, vote: dict[str, Any]) -> tuple[str, str, str]:
    """A pairwise vote's two items as shown and the item it chose (or tie)."""
    left, right = _sides(store, vote)
    choice = vote["choice"]
    return (
        left,
        right,
        left if choice == "left" else right if choice == "right" else "tie",
    )


def _cluster_for(store: _Store, votes: list[dict[str, Any]], cluster: str) -> str:
    """What the bootstrap resamples: what the votes hold more than one of."""
    if cluster != "auto":
        return cluster
    raters = {v["rater"] for v in votes}
    tasks = {
        (
            store.items[_sides(store, v)[0]]["task"],
            store.items[_sides(store, v)[0]]["task_digest"],
        )
        for v in votes
    }
    if len(raters) >= 2 and len(tasks) >= 2:
        return "rater+task"
    return "rater" if len(raters) >= 2 else "task" if len(tasks) >= 2 else "vote"


def _fit(
    store: _Store,
    votes: list[dict[str, Any]],
    by: str,
    *,
    bootstrap: int,
    seed: int,
    cluster: str,
) -> tuple[dict[str, Any], int, str, list[ratings.Outcome]]:
    """Bradley-Terry tables of one kind of rater's pairwise votes, overall and
    per aspect: the tables, the votes --by put within one arm, what the
    bootstrap resampled, and the overall outcomes."""
    arms = sorted({_arm(item, by) for item in store.items.values()})
    pairwise = [v for v in votes if v["kind"] == "pairwise"]
    aspects = sorted({v["aspect"] for v in pairwise if v["aspect"]})
    overall_votes = [v for v in pairwise if v["aspect"] is None] or pairwise
    used = _cluster_for(store, overall_votes, cluster) if pairwise else "vote"
    tables: dict[str, Any] = {}
    # With --by agent or --by model two arms of a pair can become one: such a
    # vote says nothing between arms, so it is neither fitted nor counted.
    same_arm = 0
    overall: list[ratings.Outcome] = []
    for aspect in [None, *aspects]:
        outcomes, clusters = [], []
        for vote in pairwise:
            if vote["aspect"] != aspect:
                continue
            left_item, right_item = _sides(store, vote)
            left, right = (
                _arm(store.items[left_item], by),
                _arm(store.items[right_item], by),
            )
            if left == right:
                same_arm += 1
                continue
            outcomes.append(ratings.Outcome(left, right, _SCORE[vote["choice"]]))
            item = store.items[left_item]
            clusters.append((vote["rater"], f"{item['task']} {item['task_digest']}"))
        # No pairwise votes (a rubric-only store): no table of prior-only ratings.
        if outcomes:
            tables[aspect or "overall"] = ratings.pairwise_table(
                outcomes,
                arms,
                bootstrap=bootstrap,
                seed=seed,
                clusters=clusters,
                cluster=used,
            )
        if aspect is None:
            overall = outcomes
    return tables, same_arm, used, overall


def _design_effect(
    rows: list[dict[str, Any]], cluster: str, bootstrap: int
) -> float | None:
    """How many times wider the clustered intervals are than the ones that
    redraw single votes, squared: the factor by which clustered votes count
    for less (1 when votes were resampled one by one)."""
    if cluster == "vote":
        return 1.0
    if not bootstrap:
        return None
    ratios = []
    for row in rows:
        clustered, by_vote = row.get("interval"), row.get("interval_votes")
        if not clustered or not by_vote or by_vote[1] <= by_vote[0]:
            continue
        ratios.append(((clustered[1] - clustered[0]) / (by_vote[1] - by_vote[0])) ** 2)
    if not ratios:
        return None
    ratios.sort()
    middle = len(ratios) // 2
    median = (
        ratios[middle] if len(ratios) % 2 else (ratios[middle - 1] + ratios[middle]) / 2
    )
    return round(max(1.0, median), 2)


def _whole(x: float) -> float:
    """A count as a whole number where it is one (a judge's pair seen both
    ways round counts a half when its looks differ)."""
    return int(x) if x == int(x) else round(x, 2)


def _share(agree: float, n: int) -> dict[str, Any]:
    low, high = ratings.wilson(agree, n)
    return {
        "agree": agree,
        "agreement": round(agree / n, 4),
        "interval": [round(low, 4), round(high, 4)],
    }


def _judges(
    store: _Store, counted: list[dict[str, Any]], by: str
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Each model judge against people: how often it chose the output most
    people preferred, on the pairs people voted on (a judge's tie counts as
    a miss), split by the arms in the pair; its answers on a pair seen both
    ways round; and the same count for one person against the others."""
    picks: dict[frozenset[str], dict[str, str]] = defaultdict(dict)
    for vote in counted:
        if (
            _kind(vote) == "person"
            and vote["kind"] == "pairwise"
            and vote["aspect"] is None
        ):
            left, right, chose = _chosen(store, vote)
            picks[frozenset((left, right))][vote["rater"]] = chose

    def majority(key: frozenset[str], leave_out: str | None = None) -> str | None:
        counts = Counter(
            chose
            for rater, chose in picks.get(key, {}).items()
            if rater != leave_out and chose != "tie"
        ).most_common()
        if not counts or (len(counts) > 1 and counts[1][1] == counts[0][1]):
            return None
        return counts[0][0]

    base_n = base_agree = 0
    for key, by_rater in picks.items():
        for rater, chose in by_rater.items():
            preferred = majority(key, leave_out=rater)
            if preferred is not None:
                base_n += 1
                base_agree += chose == preferred
    baseline = {"votes": base_n, **_share(base_agree, base_n)} if base_n else None
    judges = []
    models = [v for v in counted if _kind(v) == "model"]
    for rater in sorted({v["rater"] for v in models}):
        mine = [v for v in models if v["rater"] == rater]
        looks = [v for v in mine if v["kind"] == "pairwise" and v["aspect"] is None]
        answers: list[tuple[str, str, str]] = [_chosen(store, v) for v in looks]
        both_ways: dict[frozenset[str], dict[tuple[str, str], str]] = defaultdict(dict)
        for left, right, chose in answers:
            both_ways[frozenset((left, right))][(left, right)] = chose
        twice = [seen for seen in both_ways.values() if len(seen) == 2]
        consistency = (
            {"pairs": len(twice), "same": sum(len(set(s.values())) == 1 for s in twice)}
            if twice
            else None
        )
        sets: list[tuple[str, str | None, list[tuple[str, str, str]]]] = []
        if answers:
            sets.append(("pairwise", None, answers))
        scores: dict[str, dict[str, float]] = defaultdict(dict)
        for vote in mine:
            if vote["kind"] == "rubric":
                scores[vote["criterion"]][vote["item_id"]] = vote["score"]
        for criterion, by_item in sorted(scores.items()):
            implied = []
            for key in picks:
                first, second = sorted(key)
                if first in by_item and second in by_item:
                    a, b = by_item[first], by_item[second]
                    implied.append(
                        (first, second, first if a > b else second if b > a else "tie")
                    )
            sets.append(("scores", criterion, implied))
        for how, criterion, chosen in sets:
            # A pair seen both ways round is one pair, scored by the share of
            # its looks that chose the people's output: two looks at one pair
            # are not two independent answers.
            looks_at: dict[frozenset[str], list[bool]] = defaultdict(list)
            for left, right, chose in chosen:
                key = frozenset((left, right))
                preferred = majority(key)
                if preferred is not None:
                    looks_at[key].append(chose == preferred)
            n = len(looks_at)
            if not n:
                continue
            agree: float = 0
            by_arm: dict[str, dict[str, float]] = defaultdict(
                lambda: {"pairs": 0, "agree": 0}
            )
            for key, hits in looks_at.items():
                hit = sum(hits) / len(hits)
                agree += hit
                first, second = sorted(key)
                for arm in {
                    _arm(store.items[first], by),
                    _arm(store.items[second], by),
                }:
                    by_arm[arm]["pairs"] += 1
                    by_arm[arm]["agree"] += hit
            agree = _whole(agree)
            for counts in by_arm.values():
                counts["agree"] = _whole(counts["agree"])
            judges.append(
                {
                    "rater": rater,
                    "how": how,
                    "criterion": criterion,
                    "pairs": n,
                    **_share(agree, n),
                    "consistency": consistency if how == "pairwise" else None,
                    "by_arm": dict(sorted(by_arm.items())),
                }
            )
    return judges, baseline


def rank(
    store_dir: Path,
    *,
    by: str = "agent+model",
    bootstrap: int = 1000,
    seed: int = 0,
    min_agreement: float | None = None,
    raters_per_pair: int | None = None,
    raters: Sequence[str] | None = None,
    cluster: str = "auto",
) -> dict[str, Any]:
    """Ratings from the store's votes, written to ``ratings.v1.json``.

    The ratings are the people's: votes by guests and by model judges are
    fitted in tables of their own (``by_kind``), and each model judge is
    measured against people (``judges``). ``raters`` names the kinds to fit
    together instead (``person``, ``guest``, ``model``); with no person's
    vote, the ratings are the first kind that voted, and ``headline`` says
    so. ``cluster`` is what the bootstrap resamples (:func:`ratings.draw`);
    ``auto`` resamples raters and tasks when there are several of each.
    ``ranked`` is true once every pair has a vote."""
    if by not in BY:
        raise PreferenceError("--by must be agent, model, agent+model or condition")
    if cluster != "auto" and cluster not in ratings.CLUSTERS:
        raise PreferenceError(
            f"--cluster must be auto or one of {', '.join(ratings.CLUSTERS)}, not {cluster!r}"
        )
    if raters is not None:
        unknown = sorted(set(raters) - set(RATER_KINDS))
        if unknown or not raters:
            raise PreferenceError(
                f"--raters takes person, guest and model, not {', '.join(unknown) or 'nothing'}"
            )
    store = _Store(store_dir)
    log = store.log()
    latest = _effective(log, store)
    # Abstentions are recorded but never counted.
    abstentions = sum(
        v["kind"] == "pairwise" and v["choice"] == "abstain" for v in latest
    )
    counted = [
        v for v in latest if not (v["kind"] == "pairwise" and v["choice"] == "abstain")
    ]
    present = [k for k in RATER_KINDS if any(_kind(v) == k for v in counted)]
    note = None
    if raters is not None:
        headline = [k for k in RATER_KINDS if k in set(raters)]
    else:
        headline = [present[0]] if present else ["person"]
        if headline != ["person"]:
            note = f"No person voted: the ratings below are the {KIND_NAMES[headline[0]]}'."
    votes = [v for v in counted if _kind(v) in headline]
    if min_agreement is None:
        gate = _shared(store.manifest, "min_agreement")
        min_agreement = gate.get("value") if isinstance(gate, dict) else gate
    if raters_per_pair is None:
        raters_per_pair = _shared(store.manifest, "raters_per_pair")
    pairwise = [v for v in votes if v["kind"] == "pairwise"]
    aspects = sorted({v["aspect"] for v in pairwise if v["aspect"]})
    tables, same_arm, used, overall_outcomes = _fit(
        store, votes, by, bootstrap=bootstrap, seed=seed, cluster=cluster
    )
    design_effect = _design_effect(tables.get("overall", []), used, bootstrap)
    # The table pools the scores of whoever the ratings are (people unless
    # --raters says otherwise); a model judge's scores are otherwise its own,
    # and guests' a table of their own.
    scores: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    judge_scores: dict[str, dict[tuple[str, str], dict[str, list[float]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    kind_scores: dict[str, dict[tuple[str, str], dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for vote in counted:
        if vote["kind"] != "rubric":
            continue
        item = store.items[vote["item_id"]]
        key = (_arm(item, by), vote["criterion"])
        kind = _kind(vote)
        if kind in headline:
            scores[key][vote["item_id"]].append(vote["score"])
        elif kind == "model":
            judge_scores[vote["rater"]][key][vote["item_id"]].append(vote["score"])
        else:
            kind_scores[kind][key][vote["item_id"]].append(vote["score"])
    # The rater checks read the overall votes, one per rater and pair: a
    # rater's per-aspect votes on a pair mostly repeat their overall one, and
    # counting them again would inflate the left-side z and the agreement.
    overall_votes = [v for v in pairwise if v["aspect"] is None]
    # Agreement compares the output each rater chose, so each pair's answers
    # are put in one orientation: the sides issue() gave it, or the two ids in
    # order for a pair only a rater's tool made.
    choices: dict[frozenset[str], dict[str, str]] = defaultdict(dict)
    for vote in overall_votes:
        left_item, right_item, chose = _chosen(store, vote)
        key = frozenset((left_item, right_item))
        issued = store.pair_of.get(key)
        first = issued["left"] if issued else min(key)
        choices[key][vote["rater"]] = (
            "tie" if chose == "tie" else "left" if chose == first else "right"
        )
    agreed = ratings.agreement(choices)
    if min_agreement is not None:
        agreed["min_agreement"] = min_agreement
        # None: no pair had two raters, so the minimum can't be checked.
        agreed["met"] = (
            None
            if agreed["agreement"] is None
            else agreed["agreement"] >= float(min_agreement)
        )
    counts: dict[str, int] = defaultdict(int)
    for vote in votes:
        counts[vote["rater"]] += 1
    needed = int(raters_per_pair) if raters_per_pair else 1
    coverage = {
        "raters_per_pair": needed,
        "pairs": len(store.pairs),
        "covered": sum(len(choices.get(key, {})) >= needed for key in store.pair_of),
    }
    voted = sum(bool(choices.get(key)) for key in store.pair_of)
    by_kind: dict[str, Any] = {}
    for kind in present:
        if kind in headline:
            continue
        theirs = [v for v in counted if _kind(v) == kind]
        kind_tables, _same, kind_cluster, _outcomes = _fit(
            store, theirs, by, bootstrap=bootstrap, seed=seed, cluster=cluster
        )
        by_kind[kind] = {
            "votes": len(theirs),
            "raters": len({v["rater"] for v in theirs}),
            "cluster": kind_cluster,
            "pairwise": kind_tables,
        }
        if kind in kind_scores:
            by_kind[kind]["rubric"] = ratings.rubric_table(
                kind_scores[kind], bootstrap=bootstrap, seed=seed
            )
    judges, people_baseline = _judges(store, counted, by)
    result = {
        "schema": "benchflow.preference-ratings.v1",
        "computed_at": _now(),
        "by": by,
        "votes_head_sha256": log[-1]["vote_sha256"] if log else None,
        "votes": len(log),
        # Counted means fitted: a vote --by puts within one arm is not.
        "effective_votes": len(votes) - same_arm,
        "pairwise_votes": sum(v["kind"] == "pairwise" for v in votes) - same_arm,
        "rubric_scores": sum(v["kind"] == "rubric" for v in votes),
        "abstentions": abstentions,
        "same_arm_votes": same_arm,
        "aspect_raters": {
            aspect: len({v["rater"] for v in pairwise if v["aspect"] == aspect})
            for aspect in aspects
        },
        "replaced_votes": len(log) - len(latest),
        "bootstrap": bootstrap,
        "seed": seed,
        "method": "bradley-terry (MM); ties half a win; one virtual tie per arm with a reference at 1000",
        "headline": {"kinds": headline, "note": note},
        "cluster": used,
        "design_effect": design_effect,
        "pairwise": tables,
        "head_to_head": ratings.head_to_head(
            overall_outcomes, design_effect=design_effect
        ),
        "rubric": ratings.rubric_table(scores, bootstrap=bootstrap, seed=seed),
        "raters": dict(sorted(counts.items())),
        "left_share": ratings.left_share(v["choice"] for v in overall_votes),
        "sides": store.manifest.get("sides"),
        # Trials pairs left out (no output, errored): their arms are rated on
        # the rest, and the checks say so.
        "left_out": store.manifest.get("left_out") or [],
        "agreement": agreed,
        "coverage": coverage,
        "voted_pairs": voted,
        # Ranks wait until every pair has at least one vote.
        "ranked": (voted == len(store.pairs)) if store.pairs else None,
        "by_kind": by_kind,
        "judges": judges,
        "people_baseline": people_baseline,
        "judge_scores": [
            {"rater": rater, **row}
            for rater, table in sorted(judge_scores.items())
            for row in ratings.rubric_table(table, bootstrap=bootstrap, seed=seed)
        ],
    }
    (store.dir / RATINGS).write_text(json.dumps(result, indent=1) + "\n")
    return result
