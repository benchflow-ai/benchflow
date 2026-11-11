"""Reward vectors and group-relative advantages for training exports.

``bench train convert --reward-vector --group-advantage grpo|loo`` (and the
same keyword arguments of the Prime-SFT and TRL exporters) add, per row:

- ``reward_vector``: the rollout's reward components with names, kinds and
  weights. A rubric-scored trial gives the test gate plus one component per
  rubric criterion (normalised to 0-1); any other trial gives the numeric keys
  its verifier wrote.
- ``advantage`` and ``group``: the rollout's reward normalised against the
  other scored rollouts of its group (same task, agent and model in the same
  job by default), GRPO-style ``(r - mean) / (std + eps)`` or leave-one-out
  ``r - mean(others)``, with the grouping key and normalisation recorded.

A group never spans jobs unless ``group_by`` leaves ``job`` out: two jobs may
have run different policy checkpoints under one model name, and their rewards
must not share a baseline. Within an Evaluation job, an attempt that a retry
(or a resume) replaced is not a sample: it enters no group (``excluded:
"retried"``), the rule ``bf.load_job`` uses to keep one attempt per task.
- ``advantage_vector`` (both options): the same normalisation per component.

Unscored rollouts never enter a baseline and never get a number: their
advantage is null with ``excluded: "unscored"``.

A group reaches the trainer whole or not at all. A group is withheld when a
member has no score for its baseline (unscored, or a trial whose
``result.json`` is missing or cut short: its ``config.json`` places it in its
group) or never reached the model (the exporter found no model calls, or every
call failed): none of its rows are written, its manifest entry says why
(``withheld``), and its other members are ``excluded: "group_withheld"`` with a
null advantage. A scored member keeps its group whole: one under
``min_reward`` loses only its own rows (advantages are computed on the whole
group, then rows are filtered), and one that ended with an error (a time-out)
writes its rows with the error in its ``group`` block, so a trainer can mask
them. The committed JSON Schema
(``docs/reference/schemas/benchflow-training-signal.v1.schema.json``) is
generated from :data:`SCHEMA` (``python -m
benchflow.trajectories.training_signal docs/reference/schemas``).
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from benchflow._utils.scoring import extract_reward, finite_reward

GroupAdvantage = Literal["grpo", "loo"]
GROUP_ADVANTAGES: tuple[str, ...] = ("grpo", "loo")
GROUP_BY_KEYS: tuple[str, ...] = ("task", "agent", "model", "task_digest", "job")
DEFAULT_GROUP_BY: tuple[str, ...] = ("task", "agent", "model", "job")
EPS = 1e-4
STD_DDOF = 1

_FORMULAS = {
    "grpo": "advantage = (reward - mean) / (std + eps); mean and sample std "
    "(ddof 1) over the scored rollouts of the group",
    "loo": "advantage = reward - mean(rewards of the other scored rollouts "
    "of the group)",
}
_RUBRIC_FORMULA = (
    "reward = rubric_reward if tests pass and every blocker passes, else 0; "
    "rubric_reward = sum(weight * score) / sum(2 * weight) over scored criteria; "
    "values: gate = verifier reward, blocker pass = 1 / fail = 0, "
    "scored = score / 2, legacy pass = 1 / fail = 0 / not_applicable = null"
)
_VERIFIER_FORMULA = "reward = rewards.reward (as written by the verifier)"


def parse_group_by(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Validate a grouping key (``"task,agent"`` or a sequence of names)."""
    if value is None:
        return DEFAULT_GROUP_BY
    parts = (
        [p.strip() for p in value.split(",")]
        if isinstance(value, str)
        else [str(p).strip() for p in value]
    )
    parts = [p for p in parts if p]
    unknown = [p for p in parts if p not in GROUP_BY_KEYS]
    if not parts or unknown or len(set(parts)) != len(parts):
        raise ValueError(
            f"group_by must name distinct keys from {', '.join(GROUP_BY_KEYS)}"
            + (f" (unknown: {', '.join(unknown)})" if unknown else "")
        )
    return tuple(parts)


def check_options(
    *, group_advantage: str | None, group_by: str | Sequence[str] | None
) -> tuple[str, ...]:
    """Validate the options up front; returns the parsed grouping key."""
    if group_advantage is not None and group_advantage not in GROUP_ADVANTAGES:
        raise ValueError("group_advantage must be 'grpo' or 'loo'")
    return parse_group_by(group_by)


def _read(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def scored_reward(result: dict[str, Any] | None) -> float | None:
    """The rollout's reward when it is scored, else None (never 0)."""
    if not isinstance(result, dict):
        return None
    try:
        return finite_reward(extract_reward(result))
    except (TypeError, ValueError):
        return None


def _flatten(prefix: str, value: Any, out: list[tuple[str, float]]) -> None:
    if isinstance(value, dict):
        for key in value:
            _flatten(f"{prefix}.{key}" if prefix else str(key), value[key], out)
        return
    number = finite_reward(value)
    if number is not None:
        out.append((prefix, number))


def _criterion_value(kind: str, check: dict[str, Any]) -> float | None:
    if kind == "scored":
        score = check.get("score")
        if isinstance(score, int) and not isinstance(score, bool) and 0 <= score <= 2:
            return score / 2
        return None
    outcome = check.get("outcome")
    if outcome == "pass":
        return 1.0
    if outcome == "fail":
        return 0.0
    return None


def _rubric_vector(
    rollout_dir: Path, result: dict[str, Any], scoring: dict[str, Any]
) -> dict[str, Any]:
    tests = finite_reward(scoring.get("verifier_reward"))
    revision = scoring.get("revision")
    details = (
        _read(rollout_dir / revision)
        if isinstance(revision, str) and revision
        else None
    )
    checks = details.get("checks") if details else None
    if not isinstance(checks, dict):
        missing = revision if isinstance(revision, str) else "scoring/<revision>"
        return {
            "source": "scoring",
            "names": ["tests", "rubric_reward"],
            "kinds": ["gate", "quality"],
            "weights": [None, None],
            "values": [tests, finite_reward(scoring.get("rubric_reward"))],
            "formula": _RUBRIC_FORMULA,
            "revision": revision if isinstance(revision, str) else None,
            "note": f"per-criterion verdicts unavailable: {missing} is missing "
            "or has no checks",
        }
    from benchflow.rubric_reviews import _rubric

    snapshot = details.get("rubric_snapshot") if details else None
    definition = (
        _read(rollout_dir / snapshot)
        if isinstance(snapshot, str) and snapshot
        else None
    )
    rubric = _rubric(definition, None) or {"criteria": []}
    known = {c["name"]: c for c in rubric["criteria"]}
    names = list(known) + [n for n in checks if n not in known]
    vector: dict[str, Any] = {
        "source": "rubric",
        "names": ["tests"],
        "kinds": ["gate"],
        "weights": [None],
        "values": [tests],
    }
    for name in names:
        check = checks.get(name)
        check = check if isinstance(check, dict) else {}
        criterion = known.get(name, {})
        kind = criterion.get("kind") or ("scored" if "score" in check else "legacy")
        vector["names"].append(name)
        vector["kinds"].append(kind)
        vector["weights"].append(criterion.get("weight") if kind == "scored" else None)
        vector["values"].append(_criterion_value(kind, check))
    vector["formula"] = _RUBRIC_FORMULA
    vector["revision"] = revision
    sha = details.get("rubric_sha256") if details else None
    vector["rubric_sha256"] = sha if isinstance(sha, str) else None
    return vector


def reward_vector(rollout_dir: Path, result: dict[str, Any]) -> dict[str, Any] | None:
    """The rollout's reward components, or None when it is unscored."""
    if scored_reward(result) is None:
        return None
    scoring = result.get("scoring")
    if isinstance(scoring, dict) and scoring.get("status") == "complete":
        return _rubric_vector(rollout_dir, result, scoring)
    rewards = result.get("rewards")
    pairs: list[tuple[str, float]] = []
    if isinstance(rewards, dict):
        _flatten("", rewards, pairs)
    if not any(name == "reward" for name, _ in pairs):
        pairs.insert(0, ("reward", scored_reward(result) or 0.0))
    pairs.sort(key=lambda p: p[0] != "reward")
    return {
        "source": "verifier",
        "names": [n for n, _ in pairs],
        "kinds": ["verifier"] * len(pairs),
        "weights": [None] * len(pairs),
        "values": [v for _, v in pairs],
        "formula": _VERIFIER_FORMULA,
    }


def _key_value(
    result: dict[str, Any], rollout_dir: Path, name: str, *, job: str | None = None
) -> Any:
    if name == "task":
        return result.get("task_name") or rollout_dir.name.rsplit("__", 1)[0]
    if name == "agent":
        return result.get("agent")
    if name == "model":
        return result.get("model")
    if name == "job":
        return job if job is not None else Path(rollout_dir).parent.name
    return result.get("task_digest")


def job_labels(rollout_dirs: Iterable[Path]) -> dict[Path, str]:
    """A label for each rollout's job folder: its name when all rollouts share
    one job, else its path below the job folders' common parent (job names
    can repeat, e.g. ``trial-01/job`` and ``trial-02/job``)."""
    import os

    folders = sorted({Path(d).parent for d in rollout_dirs})
    if len(folders) <= 1:
        return {f: f.name for f in folders}
    common = Path(os.path.commonpath([str(f) for f in folders]))
    return {f: f.relative_to(common).as_posix() or f.name for f in folders}


def replaced_attempts(
    rollouts: Iterable[tuple[Path, dict[str, Any] | None]],
) -> set[Path]:
    """Rollouts that a later attempt of the same trial replaced.

    In an Evaluation job folder (``evaluation.json`` or ``summary.json``) the
    rollouts of one task, agent and model are attempts of one trial, whose
    result is the scored attempt, then the newest (``bf.load_job``'s rule);
    the others are returned. Rollouts in other folders are samples, never
    replaced.
    """
    from benchflow._utils.result_paths import attempt_rank, holds_attempts

    folders: dict[Path, bool] = {}
    chains: dict[tuple[Any, ...], list[Path]] = {}
    ranks: dict[Path, tuple[bool, float, str]] = {}
    for rollout_dir, result in rollouts:
        if not isinstance(result, dict):
            continue
        root = Path(rollout_dir)
        folder = root.parent
        if folder not in folders:
            folders[folder] = holds_attempts(folder)
        if not folders[folder]:
            continue
        key = (
            folder,
            _key_value(result, root, "task"),
            result.get("agent"),
            result.get("model"),
        )
        chains.setdefault(key, []).append(root)
        ranks[root] = attempt_rank(
            root / "result.json", scored=scored_reward(result) is not None
        )
    replaced: set[Path] = set()
    for chain in chains.values():
        if len(chain) > 1:
            final = max(chain, key=lambda d: ranks[d])
            replaced.update(d for d in chain if d != final)
    return replaced


def _group_id(key: dict[str, Any]) -> str:
    return "|".join(f"{k}={'null' if v is None else v}" for k, v in key.items())


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sample_std(values: list[float]) -> float:
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - STD_DDOF))


def _advantages(values: list[float | None], method: str) -> list[float | None]:
    scored = [v for v in values if v is not None]
    if len(scored) < 2:
        return [None] * len(values)
    if method == "grpo":
        mean, std = _mean(scored), _sample_std(scored)
        if std == 0.0:
            return [None if v is None else 0.0 for v in values]
        return [None if v is None else (v - mean) / (std + EPS) for v in values]
    total, n = sum(scored), len(scored)
    return [None if v is None else v - (total - v) / (n - 1) for v in values]


@dataclass
class _Member:
    rollout_dir: Path
    reward: float | None
    vector: dict[str, Any] | None
    advantage: float | None = None
    advantage_vector: list[float | None] | None = None
    excluded: str | None = None
    # A later attempt of the same trial replaced it: not a sample.
    retried: bool = False
    # Why it has no result to read ("result.json missing"), or None.
    no_result: str | None = None
    # How the rollout ended when it ended with an error, or None.
    error: str | None = None


def ended_with_error(result: dict[str, Any]) -> str | None:
    """The error a rollout ended with, as a short name, or None: its
    ``error_category`` (``error`` without one), a verifier error, or a
    partial trajectory."""
    if result.get("error"):
        return str(result.get("error_category") or "error")
    if result.get("verifier_error"):
        return str(result.get("verifier_error_category") or "verifier_error")
    if result.get("partial_trajectory") is True:
        return "partial_trajectory"
    return None


def _stand_in(rollout_dir: Path) -> dict[str, Any] | None:
    """A result for a trial with no readable ``result.json``, from the
    ``config.json`` the trial wrote when it started: unscored, and enough to
    place it in its group (its task is the task folder's name, as a trial
    names it). None for a folder that is not a trial's."""
    config = _read(Path(rollout_dir) / "config.json")
    if config is None or not config.get("task_path"):
        return None
    return {
        "task_name": Path(str(config["task_path"])).name,
        "agent": config.get("agent"),
        "model": config.get("model"),
        "task_digest": config.get("task_digest"),
        "rewards": None,
    }


def unfinished_rollout_dirs(rollout_dirs: Iterable[Path]) -> list[Path]:
    """Trial folders beside these rollouts that never wrote a ``result.json``
    (a trial killed after it started, or still running): their groups are
    not whole. Set-aside trials and folders that are not a trial's are left
    out."""
    from benchflow._utils.result_paths import SET_ASIDE_DIR

    dirs = [Path(d) for d in rollout_dirs]
    known = {d.resolve() for d in dirs}
    out: list[Path] = []
    for folder in sorted({d.parent for d in dirs}):
        try:
            children = sorted(folder.iterdir())
        except OSError:
            continue
        for child in children:
            if (
                child.name == SET_ASIDE_DIR
                or not child.is_dir()
                or child.resolve() in known
                or (child / "result.json").exists()
            ):
                continue
            if _stand_in(child) is not None:
                out.append(child)
    return out


# Why an exporter wrote no row for a rollout, in the words a withheld group
# gives (codes the Prime-SFT and TRL converters record). A member under
# --min-reward ("reward") is scored and never withholds its group.
_NO_ROW_WORDS = {
    "terminal_error": "ended with an error",
    "reward": "below --min-reward",
    "no_trajectory": "no model calls recorded",
    "provider_error": "every model call failed",
    "control": "a control run",
    "no_result": "no readable result.json",
}


def _rollout_key(rollout_dir: Path) -> str:
    return str(Path(rollout_dir).resolve())


@dataclass
class TrainingSignals:
    """Per-rollout reward vectors and advantages for one conversion."""

    reward_vector: bool
    group_advantage: str | None
    group_by: tuple[str, ...]
    groups: list[dict[str, Any]] = field(default_factory=list)
    rows_withheld: int = 0
    _rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Per group id: its samples' rollout keys, every member's key (retried
    # attempts too), its manifest entry, and why it is withheld.
    _samples: dict[str, list[str]] = field(default_factory=dict)
    _members: dict[str, list[str]] = field(default_factory=dict)
    _entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    _withheld: dict[str, str] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return self.reward_vector or self.group_advantage is not None

    def for_rollout(self, rollout_dir: Path) -> dict[str, Any]:
        """The fields to add to every row of this rollout (empty when off)."""
        return dict(self._rows.get(_rollout_key(rollout_dir), {}))

    def manifest(self) -> dict[str, Any]:
        manifest: dict[str, Any] = {
            "reward_vector": self.reward_vector,
            "group_advantage": self.group_advantage,
            "group_by": list(self.group_by),
            "std_ddof": STD_DDOF,
            "eps": EPS,
            "groups": self.groups,
        }
        if self.group_advantage is not None:
            manifest["withheld_groups"] = len(self._withheld)
            manifest["rows_withheld"] = self.rows_withheld
        return manifest

    def withhold(self, gid: str, reason: str) -> None:
        """Withhold a whole group: no advantage for any member, and why."""
        self._withheld[gid] = reason
        entry = self._entries[gid]
        entry["withheld"] = reason
        for member in entry["members"]:
            member["advantage"] = None
            member.setdefault("excluded", "group_withheld")
        for key in self._members[gid]:
            fields = self._rows.get(key)
            if not fields or "group" not in fields:
                continue
            fields["advantage"] = None
            if "advantage_vector" in fields:
                fields["advantage_vector"] = None
            block = fields["group"]
            block["withheld"] = reason
            block.setdefault("excluded", "group_withheld")

    def keep_whole_groups(
        self,
        rows: list[dict[str, Any]],
        row_keys: list[str],
        dropped: dict[str, str],
    ) -> list[dict[str, Any]]:
        """The rows of whole groups only.

        ``row_keys`` holds each row's rollout key (:func:`_rollout_key`), and
        ``dropped`` the reason code of each rollout the exporter wrote no row
        for. A group one of whose samples wrote no row is withheld, and so is
        every group already withheld for an unscored member: none of their
        rows are kept. A sample ``min_reward`` dropped (``"reward"``) is
        scored: its reward is in the baseline already, so its group keeps
        its other rows. Without ``group_advantage`` there are no groups to
        keep whole, and every row is kept.
        """
        if self.group_advantage is None:
            return rows
        written = set(row_keys)
        for gid, samples in self._samples.items():
            if gid in self._withheld:
                continue
            missing = [
                key
                for key in samples
                if key not in written and dropped.get(key) != "reward"
            ]
            if missing:
                self.withhold(
                    gid,
                    "no row: "
                    + ", ".join(
                        f"{Path(key).name} "
                        f"({_NO_ROW_WORDS.get(dropped.get(key, ''), 'no trainable model call')})"
                        for key in missing
                    ),
                )
        held = {key for gid in self._withheld for key in self._members[gid]}
        kept = [row for row, key in zip(rows, row_keys, strict=True) if key not in held]
        self.rows_withheld = len(rows) - len(kept)
        return kept


def compute_training_signals(
    rollouts: Iterable[tuple[Path, dict[str, Any] | None]],
    *,
    reward_vector: bool = False,
    group_advantage: str | None = None,
    group_by: str | Sequence[str] | None = None,
) -> TrainingSignals:
    """Reward vectors and group advantages for ``(rollout_dir, result)`` pairs."""
    keys = check_options(group_advantage=group_advantage, group_by=group_by)
    signals = TrainingSignals(
        reward_vector=reward_vector, group_advantage=group_advantage, group_by=keys
    )
    if not signals.enabled:
        return signals
    # A rollout without a readable result.json joins its group through the
    # config.json its trial wrote at start (unscored: the group is withheld).
    triples: list[tuple[Path, dict[str, Any], str | None]] = []
    for d, r in rollouts:
        rollout_dir = Path(d)
        if isinstance(r, dict):
            triples.append((rollout_dir, r, None))
            continue
        stand_in = _stand_in(rollout_dir)
        if stand_in is not None:
            gone = not (rollout_dir / "result.json").exists()
            why = "result.json missing" if gone else "result.json unreadable"
            triples.append((rollout_dir, stand_in, why))
    pairs = [(d, r) for d, r, _ in triples]
    labels = job_labels(d for d, _ in pairs)
    replaced = replaced_attempts(pairs)
    grouped: dict[str, tuple[dict[str, Any], list[_Member]]] = {}
    for rollout_dir, result, no_result in triples:
        key = {
            k: _key_value(result, rollout_dir, k, job=labels.get(rollout_dir.parent))
            for k in keys
        }
        member = _Member(
            rollout_dir=rollout_dir,
            reward=scored_reward(result),
            vector=reward_vector_or_none(rollout_dir, result),
            retried=rollout_dir in replaced,
            no_result=no_result,
            error=None if no_result else ended_with_error(result),
        )
        grouped.setdefault(_group_id(key), (key, []))[1].append(member)

    for gid, (key, members) in grouped.items():
        _fill_group(signals, gid, key, members)
    return signals


def reward_vector_or_none(
    rollout_dir: Path, result: dict[str, Any]
) -> dict[str, Any] | None:
    try:
        return reward_vector(Path(rollout_dir), result)
    except (TypeError, ValueError, KeyError):
        return None


def _fill_group(
    signals: TrainingSignals,
    gid: str,
    key: dict[str, Any],
    members: list[_Member],
) -> None:
    method = signals.group_advantage
    samples = [m for m in members if not m.retried]
    rewards = [m.reward for m in samples]
    scored = [r for r in rewards if r is not None]
    names_seen = {tuple(m.vector["names"]) for m in samples if m.vector}
    group_block: dict[str, Any] | None = None
    if method is not None:
        for member, adv in zip(samples, _advantages(rewards, method), strict=True):
            member.advantage = adv
            if member.reward is None:
                member.excluded = "unscored"
            elif adv is None:
                member.excluded = "single_scored_rollout"
        for member in members:
            if member.retried:
                member.excluded = "retried"
        if signals.reward_vector:
            _fill_vector_advantages(samples, method)
        group_block = {
            "id": gid,
            "by": list(signals.group_by),
            "key": key,
            "normalisation": method,
            "formula": _FORMULAS[method],
            "rollouts": len(samples),
            "scored": len(scored),
            "mean": _mean(scored) if scored else None,
            "std": _sample_std(scored) if len(scored) >= 2 else None,
            "std_ddof": STD_DDOF,
            "eps": EPS,
        }
        if len(names_seen) > 1:
            group_block["vector_names_differ"] = True
        entry = {
            **group_block,
            "members": [
                {
                    "rollout": m.rollout_dir.name,
                    "rollout_dir": str(m.rollout_dir),
                    "reward": m.reward,
                    "advantage": m.advantage,
                    **({"excluded": m.excluded} if m.excluded else {}),
                    **({"error": m.error} if m.error else {}),
                }
                for m in members
            ],
        }
        signals.groups.append(entry)
        signals._entries[gid] = entry
        signals._samples[gid] = [_rollout_key(m.rollout_dir) for m in samples]
        signals._members[gid] = [_rollout_key(m.rollout_dir) for m in members]
    for member in members:
        fields: dict[str, Any] = {}
        if signals.reward_vector:
            fields["reward_vector"] = member.vector
        if group_block is not None:
            fields["advantage"] = member.advantage
            block = dict(group_block)
            if member.excluded:
                block["excluded"] = member.excluded
            if member.error:
                block["error"] = member.error
            fields["group"] = block
            if signals.reward_vector:
                fields["advantage_vector"] = member.advantage_vector
        signals._rows[_rollout_key(member.rollout_dir)] = fields
    no_result = [
        f"{m.rollout_dir.name} ({m.no_result})" for m in samples if m.no_result
    ]
    unscored = [
        m.rollout_dir.name for m in samples if m.reward is None and not m.no_result
    ]
    if group_block is not None and (no_result or unscored):
        reasons = []
        if no_result:
            reasons.append("no result: " + ", ".join(no_result))
        if unscored:
            reasons.append("unscored: " + ", ".join(unscored))
        signals.withhold(gid, "; ".join(reasons))


def _fill_vector_advantages(members: list[_Member], method: str) -> None:
    names: list[str] = []
    for m in members:
        for name in (m.vector or {}).get("names", []):
            if name not in names:
                names.append(name)
    per_name: dict[str, list[float | None]] = {}
    for name in names:
        values = []
        for m in members:
            vec = m.vector
            value = None
            if vec and m.reward is not None and name in vec["names"]:
                value = vec["values"][vec["names"].index(name)]
            values.append(value)
        per_name[name] = _advantages(values, method)
    for idx, m in enumerate(members):
        if m.vector is None:
            m.advantage_vector = None
            continue
        m.advantage_vector = [per_name[name][idx] for name in m.vector["names"]]


def rollout_training_signals(
    jobs_dir: str | Path,
    *,
    reward_vector: bool = True,
    group_advantage: str | None = "grpo",
    group_by: str | Sequence[str] | None = None,
    canonical_selection: str | Path | None = None,
) -> TrainingSignals:
    """Reward vectors and group advantages for every rollout under a folder.

    Works without LLM trajectories (oracle runs, subscription runs), so the
    groups of any job can be inspected before converting it.
    """
    from benchflow.trajectories.export_prime_sft import (
        _iter_rollout_dirs,
        _iter_selected_rollout_dirs,
        load_rollout_result,
    )

    dirs = (
        _iter_selected_rollout_dirs(canonical_selection)
        if canonical_selection is not None
        else _iter_rollout_dirs(jobs_dir)
    )
    if canonical_selection is None and group_advantage is not None:
        dirs = [*dirs, *unfinished_rollout_dirs(dirs)]
    return compute_training_signals(
        ((d, load_rollout_result(d)) for d in dirs),
        reward_vector=reward_vector,
        group_advantage=group_advantage,
        group_by=group_by,
    )


# JSON Schema for the row fields (docs/reference/schemas).

_OPT_NUM = {"type": ["number", "null"]}
_NUM_LIST = {"type": "array", "items": _OPT_NUM}

SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://benchflow.ai/schemas/benchflow-training-signal.v1.schema.json",
    "title": "Training-signal fields of a bench train convert row",
    "description": "Fields added by --reward-vector and --group-advantage to "
    "prime-sft and trl-sft rows. Other row fields are described by the "
    "trainer format. Unscored rollouts have a null reward_vector and a null "
    "advantage, never 0.",
    "type": "object",
    "properties": {
        "reward_vector": {
            "oneOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "required": [
                        "source",
                        "names",
                        "kinds",
                        "weights",
                        "values",
                        "formula",
                    ],
                    "properties": {
                        "source": {"enum": ["rubric", "scoring", "verifier"]},
                        "names": {"type": "array", "items": {"type": "string"}},
                        "kinds": {
                            "type": "array",
                            "items": {
                                "enum": [
                                    "gate",
                                    "blocker",
                                    "scored",
                                    "legacy",
                                    "quality",
                                    "verifier",
                                ]
                            },
                        },
                        "weights": {
                            "type": "array",
                            "items": {"type": ["integer", "null"]},
                        },
                        "values": {
                            **_NUM_LIST,
                            "description": "Component values normalised to "
                            "0-1 for rubric criteria; null when the criterion "
                            "has no verdict.",
                        },
                        "formula": {"type": "string"},
                        "revision": {"type": ["string", "null"]},
                        "rubric_sha256": {"type": ["string", "null"]},
                        "note": {"type": "string"},
                    },
                },
            ]
        },
        "advantage": _OPT_NUM,
        "advantage_vector": {"oneOf": [{"type": "null"}, _NUM_LIST]},
        "group": {
            "type": "object",
            "required": [
                "id",
                "by",
                "key",
                "normalisation",
                "formula",
                "rollouts",
                "scored",
                "mean",
                "std",
                "std_ddof",
                "eps",
            ],
            "properties": {
                "id": {"type": "string"},
                "by": {
                    "type": "array",
                    "items": {"enum": list(GROUP_BY_KEYS)},
                },
                "key": {"type": "object"},
                "normalisation": {"enum": list(GROUP_ADVANTAGES)},
                "formula": {"type": "string"},
                "rollouts": {"type": "integer", "minimum": 0},
                "scored": {"type": "integer", "minimum": 0},
                "mean": _OPT_NUM,
                "std": _OPT_NUM,
                "std_ddof": {"const": STD_DDOF},
                "eps": {"type": "number"},
                "vector_names_differ": {"type": "boolean"},
                "withheld": {
                    "type": "string",
                    "description": "Why the whole group is withheld from "
                    "the trainer: a member has no result or is unscored, or "
                    "wrote no row (no model calls, every call failed).",
                },
                "error": {
                    "type": "string",
                    "description": "This rollout ended with an error (its "
                    "error_category, verifier_error or partial_trajectory) "
                    "but was scored, so it stays in its group; a trainer may "
                    "mask its rows.",
                },
                "excluded": {
                    "enum": [
                        "unscored",
                        "single_scored_rollout",
                        "retried",
                        "group_withheld",
                    ]
                },
            },
        },
    },
}

SCHEMA_FILE = "benchflow-training-signal.v1.schema.json"


def write_schema(directory: str | Path) -> Path:
    path = Path(directory) / SCHEMA_FILE
    path.write_text(json.dumps(SCHEMA, indent=2) + "\n")
    return path


if __name__ == "__main__":
    print(write_schema(sys.argv[1] if len(sys.argv) > 1 else "."))
