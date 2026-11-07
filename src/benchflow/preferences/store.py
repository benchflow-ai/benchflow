"""A preference store: blind pairs of completed trials, the human votes on
them, and the ratings computed from those votes.

``issue`` reads finished trials, makes one item per trial output and pairs
items of the same task version from different arms (an arm is an agent and
model), and writes the store. The store keeps the private mapping from item
to trial and arm; ``assignment.json`` shows a rater's tool only opaque ids,
each task and each output. ``import_votes`` checks a whole file of votes
against the store and appends them to ``votes.v1.jsonl``, one canonical JSON
line each (RFC 8785), each naming the hash of the line before it. ``rank``
verifies that log and computes the ratings (:mod:`.ratings`).

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
import shutil
import stat
from collections import defaultdict
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
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
) -> dict[str, Any]:
    """Make the store and return the blind assignment for a rater's tool."""
    store = Path(store)
    if (store / STORE).exists():
        raise PreferenceError(
            f"{store} already holds an assignment",
            hint="issue a new assignment into a new --store folder",
        )
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big")
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
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
            items.append(
                {
                    "item_id": "i-"
                    + hashlib.sha256(f"{seed}:{trial}".encode()).hexdigest()[:16],
                    "task": task,
                    "task_digest": str(result.get("task_digest") or ""),
                    "trial": str(trial),
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
    if not items:
        raise PreferenceError(
            f"no finished trial has files in {output}/",
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
    for item in sorted(items, key=lambda i: i["item_id"]):
        by_version[(item["task"], item["task_digest"])].append(item)
    issued_at = _now()
    pairs: list[dict[str, Any]] = []
    rubric_only = {
        version
        for version, table in protocols.items()
        if table.get("protocol") == "rubric"
    }
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
        candidates = [
            (a, b)
            for i, a in enumerate(group)
            for b in group[i + 1 :]
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
            "trial": item["trial"],
            "agent": item["agent"],
            "model": item["model"],
            "reason": "no output of another agent or model on its task version",
        }
        for item in items
        if item["item_id"] not in paired
        and (item["task"], item["task_digest"]) not in rubric_only
    ]
    frozen = None
    if rubric is not None:
        frozen, rubric_bytes = _rubric(Path(rubric))
    store.mkdir(parents=True, exist_ok=True)
    if frozen is not None:
        (store / RUBRIC).write_bytes(rubric_bytes)
    _write_jsonl(store / ITEMS, items)
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
    for item in items:
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
    }
    (store / STORE).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    shown = {item["item_id"]: item for item in items}
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


def _rows(path: Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() == ".csv":
        import csv

        return [
            {k: v for k, v in row.items() if v not in (None, "")}
            for row in csv.DictReader(text.splitlines())
        ]
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise PreferenceError(f"{path} line {number} is not a JSON object")
        rows.append(row)
    return rows


def _vote(
    store: _Store, row: dict[str, Any], where: str, source: str | None
) -> dict[str, Any]:
    rater = str(row.get("rater") or "").strip()
    if not rater:
        raise PreferenceError(f"{where}: no rater")
    rated = _parse_time(row.get("rated_at"), f"{where}: rated_at")
    kind = row.get("kind") or ("pairwise" if "pair_id" in row else "rubric")
    body: dict[str, Any] = {
        "kind": kind,
        "rater": rater,
        "rated_at": rated.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    }
    if kind == "pairwise":
        pair = store.pairs.get(str(row.get("pair_id")))
        if pair is None:
            raise PreferenceError(
                f"{where}: pair {row.get('pair_id')!r} was not issued"
            )
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
        if rated < _parse_time(pair["issued_at"], "issued_at"):
            raise PreferenceError(f"{where}: rated before the pair was issued")
        body.update(pair_id=pair["pair_id"], aspect=aspect, choice=choice)
    elif kind == "rubric":
        item = store.items.get(str(row.get("item_id")))
        if item is None:
            raise PreferenceError(
                f"{where}: item {row.get('item_id')!r} was not issued"
            )
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
    else:
        raise PreferenceError(f"{where}: kind must be pairwise or rubric")
    if row.get("reason"):
        body["reason"] = str(row["reason"])
    body["source"] = str(row.get("source") or source or "")
    identity = {
        k: body[k] for k in body if k not in ("reason", "source", "choice", "score")
    }
    body["vote_id"] = str(row.get("vote_id") or _sha256(_canonical(identity))[7:23])
    return body


def import_votes(
    store_dir: Path, votes: Path, *, source: str | None = None
) -> dict[str, Any]:
    """Check every row of ``votes``, then append the new ones. Returns how
    many were added and how many were already recorded, and the log's last
    hash (``head_sha256``)."""
    store = _Store(store_dir)
    rows = _rows(votes)
    if not rows:
        raise PreferenceError(f"{votes} holds no votes")
    checked = [
        _vote(store, row, f"{votes} vote {n}", source) for n, row in enumerate(rows, 1)
    ]
    touched = {
        store.pairs[v["pair_id"]][side]
        for v in checked
        if v["kind"] == "pairwise"
        for side in ("left", "right")
    }
    touched |= {v["item_id"] for v in checked if v["kind"] == "rubric"}
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
            kept = {
                k: v
                for k, v in known.items()
                if k not in ("seq", "previous_sha256", "recorded_at", "vote_sha256")
            }
            if kept != vote:
                target = vote.get("pair_id") or vote.get("item_id")
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
        previous = log[-1]["vote_sha256"] if log else None
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
    return {"added": len(new), "already_recorded": repeats, "head_sha256": previous}


def _effective(votes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each rater's latest vote on each pair and aspect (or item and
    criterion); a later vote replaces an earlier one."""
    latest: dict[tuple, dict[str, Any]] = {}
    for vote in votes:
        if vote["kind"] == "pairwise":
            key = ("pairwise", vote["pair_id"], vote["aspect"], vote["rater"])
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
    label = f"{item['agent'] or '-'} / {item['model'] or '-'}"
    receipt = item.get("harness_versions")
    if receipt is not None:
        label += " / harness-" + hashlib.sha256(_canonical(receipt)).hexdigest()[:16]
    return label


def rank(
    store_dir: Path,
    *,
    by: str = "agent+model",
    bootstrap: int = 1000,
    seed: int = 0,
    min_agreement: float | None = None,
    raters_per_pair: int | None = None,
) -> dict[str, Any]:
    """Ratings from the store's votes, written to ``ratings.v1.json``."""
    if by not in ("agent", "model", "agent+model"):
        raise PreferenceError("--by must be agent, model or agent+model")
    store = _Store(store_dir)
    log = store.log()
    latest = _effective(log)
    # Abstentions are recorded but never counted.
    abstentions = sum(
        v["kind"] == "pairwise" and v["choice"] == "abstain" for v in latest
    )
    votes = [
        v for v in latest if not (v["kind"] == "pairwise" and v["choice"] == "abstain")
    ]
    arms = sorted({_arm(item, by) for item in store.items.values()})
    if min_agreement is None:
        gate = _shared(store.manifest, "min_agreement")
        min_agreement = gate.get("value") if isinstance(gate, dict) else gate
    if raters_per_pair is None:
        raters_per_pair = _shared(store.manifest, "raters_per_pair")
    pairwise = [v for v in votes if v["kind"] == "pairwise"]
    aspects = sorted({v["aspect"] for v in pairwise if v["aspect"]})
    tables: dict[str, Any] = {}
    # With --by agent or --by model two arms of a pair can become one: such a
    # vote says nothing between arms, so it is neither fitted nor counted.
    same_arm = 0
    for aspect in [None, *aspects]:
        outcomes = []
        for vote in pairwise:
            if vote["aspect"] != aspect or vote["choice"] == "abstain":
                continue
            pair = store.pairs[vote["pair_id"]]
            left, right = (
                _arm(store.items[pair["left"]], by),
                _arm(store.items[pair["right"]], by),
            )
            if left == right:
                same_arm += 1
                continue
            outcomes.append(ratings.Outcome(left, right, _SCORE[vote["choice"]]))
        # No pairwise votes (a rubric-only store): no table of prior-only ratings.
        if outcomes:
            tables[aspect or "overall"] = ratings.pairwise_table(
                outcomes, arms, bootstrap=bootstrap, seed=seed
            )
    scores: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for vote in votes:
        if vote["kind"] == "rubric":
            item = store.items[vote["item_id"]]
            scores[(_arm(item, by), vote["criterion"])][vote["item_id"]].append(
                vote["score"]
            )
    # The rater checks read the overall votes, one per rater and pair: a
    # rater's per-aspect votes on a pair mostly repeat their overall one, and
    # counting them again would inflate the left-side z and the agreement.
    overall_votes = [v for v in pairwise if v["aspect"] is None]
    choices: dict[str, dict[str, str]] = defaultdict(dict)
    for vote in overall_votes:
        if vote["choice"] != "abstain":
            choices[vote["pair_id"]][vote["rater"]] = vote["choice"]
    agreed = ratings.agreement(choices)
    if min_agreement is not None:
        agreed["min_agreement"] = min_agreement
        # None: no pair had two raters, so the minimum can't be checked.
        agreed["met"] = (
            None
            if agreed["agreement"] is None
            else agreed["agreement"] >= float(min_agreement)
        )
    raters: dict[str, int] = defaultdict(int)
    for vote in votes:
        raters[vote["rater"]] += 1
    coverage = None
    if raters_per_pair:
        covered = sum(
            len(choices.get(p, {})) >= int(raters_per_pair) for p in store.pairs
        )
        coverage = {
            "raters_per_pair": int(raters_per_pair),
            "pairs": len(store.pairs),
            "covered": covered,
        }
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
        "pairwise": tables,
        "rubric": ratings.rubric_table(scores, bootstrap=bootstrap, seed=seed),
        "raters": dict(sorted(raters.items())),
        "left_share": ratings.left_share(v["choice"] for v in overall_votes),
        "sides": store.manifest.get("sides"),
        # Trials pairs left out (no output, errored): their arms are rated on
        # the rest, and the checks say so.
        "left_out": store.manifest.get("left_out") or [],
        "agreement": agreed,
        "coverage": coverage,
    }
    (store.dir / RATINGS).write_text(json.dumps(result, indent=1) + "\n")
    return result
