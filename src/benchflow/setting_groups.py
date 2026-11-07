"""Trials split by what ran, never by what their labels say.

A score means something for one setting only. On a published leaderboard
one model's runs score from 34.85% at low effort to 53.94% at xhigh, and its
max run scores below xhigh: pooled by model, that spread disappears and the
ranking against other models is wrong. Runs that differ in the route their
calls took, the harness version that ran or the BenchFlow version that ran
them measure different things too.

:func:`setting_rows` splits trials by their :class:`Setting`:

- the agent, and whether it ran as an ACP agent or natively;
- the harness's installed version (the adapter, CLI and engine packages the
  trial's ``harness_versions`` receipt names);
- the model that served the agent's main calls (``settings_sent``'s
  ``models_served``), else the model the run named;
- the route those calls took through BenchFlow's model proxy (``routes``);
- the reasoning effort those calls carried; the effort the agent applied
  when no call recorded one; else the effort the run asked for, marked as
  only that;
- the task set: the dataset and its version, and which version of each task
  ran when the same task ran at two versions (its ``task_digest``);
- the BenchFlow version that ran the trial (config.json's
  ``benchflow_version``);
- a control run's kind (oracle, empty).

A value that 1% of the calls or fewer carried does not count, as for the run
checks (``benchflow.run_checks.MISMATCH_SHARE``). Two or more values that
each carried more are joined with ``+``: such a trial ran a mix.

A trial whose calls show no served model, route, effort or harness version
(it errored before its first model call, or before the harness was probed)
takes the one value the other trials of its launch show: the same job
folder, agent, harness mode, requested model, requested effort, control kind
and BenchFlow version. An errored trial so stays in its launch's row, where
the scoring rule counts it as 0, and never leaves a row with a better score.
When its launch shows none or several, the value is unknown, and a trial
whose effort nothing records is in an ``unknown`` row of its own.

When the calls carried another effort than the one asked for, the row is
keyed by both, so it is never merged with trials that asked for the effort
they carried, and its label says both (``effort high (requested max)``).

:class:`Settings` is the result: one :class:`SettingRow` per setting, what
differs between them (:meth:`Settings.differ`), and the ``mixed settings``
warning. A job resumed after a change (summary.json's
``restarted_after_change``, written by ``--restart-changed``) also reads as
mixed: its earlier trials ran with other settings and were set aside.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchflow.jobs import Trial
    from benchflow.scoring import Score

__all__ = [
    "EFFORT_ORDER",
    "UNKNOWN",
    "Setting",
    "SettingRow",
    "Settings",
    "job_summaries",
    "record_of",
    "scoring_document",
    "setting_rows",
    "summary_settings",
    "trial_settings",
]

UNKNOWN = "unknown"
#: The effort of a trial whose main calls BenchFlow's proxy saw, none of
#: which carried an effort: the provider's default applied.
NOT_SENT = "not sent"
#: The order rows are listed in: this ladder, then any other effort, then
#: unknown.
EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")

#: The fields :meth:`Settings.differ` names, in order, with their words.
FIELDS: tuple[tuple[str, str], ...] = (
    ("agent", "agent"),
    ("harness", "harness mode"),
    ("harness_versions", "harness version"),
    ("model", "served model"),
    ("route", "route"),
    ("effort", "effort"),
    ("task_set", "task set"),
    ("benchflow_version", "BenchFlow version"),
    ("control", "control run"),
)


@dataclass(frozen=True)
class Setting:
    """What one row of trials ran (see the module docstring).

    ``model_source`` is ``served`` (the model that answered the calls) or
    ``requested`` (the run's model, when no call recorded one);
    ``effort_source`` is ``calls``, ``applied``, ``requested`` or None.
    ``effort_requested`` is the effort the run asked for, and
    ``effort_mismatch`` True when the calls carried (or the agent applied)
    another one. Neither source is part of the key.
    """

    agent: str
    harness: str | None
    harness_versions: tuple[str, ...] | None
    model: str | None
    route: str | None
    effort: str
    effort_requested: str | None
    effort_mismatch: bool
    task_set: str | None
    benchflow_version: str | None
    control: str | None = None
    model_source: str | None = None
    effort_source: str | None = None

    def key(self) -> tuple[Any, ...]:
        """What makes two trials one setting: the requested effort counts
        only when what ran was another one."""
        return (
            self.agent,
            self.harness,
            self.harness_versions,
            self.model,
            self.route,
            self.effort,
            self.effort_requested if self.effort_mismatch else None,
            self.task_set,
            self.benchflow_version,
            self.control,
        )

    def value(self, name: str) -> str:
        """One field of :data:`FIELDS` in words, for the mixed-settings warning."""
        if name == "harness_versions":
            return " + ".join(_short(v) for v in self.harness_versions or ()) or UNKNOWN
        if name == "effort":
            return self._effort_words()
        value = getattr(self, name)
        return str(value) if value is not None else UNKNOWN

    def _effort_words(self) -> str:
        words = f"effort {self.effort}"
        if self.effort_mismatch:
            words += f" (requested {self.effort_requested})"
        elif self.effort_source == "requested":
            words += " (requested; no call recorded it)"
        return words.removeprefix("effort ")

    def label(self) -> str:
        """``agent (harness; versions) on model via route, effort E, ...``,
        the way reports name a row."""
        inside = [part for part in (self.harness,) if part]
        if self.harness_versions:
            inside.append(", ".join(_short(v) for v in self.harness_versions))
        harness = f" ({'; '.join(inside)})" if inside else ""
        model = self.model or "no model"
        if self.model_source == "requested":
            model += " (requested)"
        parts = [f"{self.agent or 'no agent'}{harness} on {model}"]
        if self.route:
            parts[0] += f" via {self.route}"
        parts.append(f"effort {self._effort_words()}")
        if self.task_set:
            parts.append(f"tasks {self.task_set}")
        if self.benchflow_version:
            parts.append(f"BenchFlow {self.benchflow_version}")
        if self.control:
            parts.append(f"{self.control} control run")
        return ", ".join(parts)

    def sort_key(self) -> tuple[Any, ...]:
        """Agent runs first, by agent, model, route, harness, then effort in
        the order of :data:`EFFORT_ORDER`, others after it, unknown last;
        control runs after every agent run."""
        rank = (
            EFFORT_ORDER.index(self.effort)
            if self.effort in EFFORT_ORDER
            else len(EFFORT_ORDER) + (self.effort == UNKNOWN)
        )
        return (
            self.control is not None,
            self.agent,
            self.model or "",
            self.route or "",
            self.harness or "",
            self.harness_versions or (),
            rank,
            self.effort,
            self.effort_requested or "",
            self.task_set or "",
            self.benchflow_version or "",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "harness": self.harness,
            "harness_versions": list(self.harness_versions)
            if self.harness_versions is not None
            else None,
            "model": self.model,
            "model_source": self.model_source,
            "route": self.route,
            "effort": self.effort,
            "effort_source": self.effort_source,
            "effort_requested": self.effort_requested,
            "effort_mismatch": self.effort_mismatch,
            "task_set": self.task_set,
            "benchflow_version": self.benchflow_version,
            "control": self.control,
        }


def _short(package_version: str) -> str:
    """``@anthropic-ai/claude-code@2.1.257`` as ``claude-code 2.1.257``."""
    name, _, version = package_version.rpartition("@")
    return f"{name.rsplit('/', 1)[-1]} {version}" if name else package_version


@dataclass
class SettingRow:
    """The trials of one :class:`Setting`; ``effort_checked`` counts the
    trials whose own calls carried the row's effort."""

    setting: Setting
    trials: list[Trial] = field(default_factory=list)
    effort_checked: int = 0
    _scored: Score | None = field(default=None, repr=False, compare=False)

    def records(self) -> list[dict[str, Any]]:
        """The rule's record of each trial (``benchflow.scoring.trial_record``)."""
        return [record_of(t) for t in self.trials]

    def score(self) -> Score:
        """The scoring rule over this row's trials (read once)."""
        from benchflow.scoring import score

        if self._scored is None or self._scored.n != len(self.trials):
            self._scored = score(self.records())
        return self._scored

    def line(self) -> str:
        """The row in one line, as every reader prints it: its setting, its
        score by the rule with the interval, and its error-excluding figure."""
        figures = self.score()
        text = (
            f"{self.setting.label()}: {figures.passed} of {figures.n} "
            f"trial{'' if figures.n == 1 else 's'} passed "
            f"({figures.score or 0.0:.1%}; {figures.interval_text()}), "
            f"{figures.n_tasks} task{'' if figures.n_tasks == 1 else 's'}"
        )
        excl = figures.excl_errors_text()
        beside = [
            part
            for part in (
                excl,
                figures.model_refusals_text(),
                figures.budget_stops_text(),
            )
            if part
        ]
        return text + "".join(f"; {part}" for part in beside)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.setting.to_dict(),
            "label": self.setting.label(),
            "trials": len(self.trials),
            "effort_checked": self.effort_checked,
            "scoring": self.score().to_dict(),
        }


def record_of(trial: Trial) -> dict[str, Any]:
    """The scoring rule's record of one trial (``benchflow.scoring.trial_record``),
    with a broken agent integration detected on read counted as one."""
    from benchflow.scoring import trial_record

    stored = trial.raw.get("integration_failure_info")
    detected = None if isinstance(stored, Mapping) else trial.integration_failure
    record = trial_record(trial.raw, integration_failure=detected)
    record["task"] = trial.task_name or record["task"] or str(trial.path)
    record["trial"] = record["trial"] or trial.path.name
    return record


def _effort(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower()


def _major(counts: Any, *, lower: bool = False) -> str | None:
    """The values carried by more than 1% of the calls a ``settings_sent``
    count covers, joined with ``+``; None when it holds none."""
    from benchflow.run_checks import MISMATCH_SHARE

    if not isinstance(counts, Mapping) or not counts:
        return None
    total = sum(int(n) for n in counts.values() if isinstance(n, int))
    kept = sorted(
        {
            (str(value).strip().lower() if lower else str(value))
            for value, n in counts.items()
            if isinstance(n, int) and n > MISMATCH_SHARE * total and str(value).strip()
        }
    )
    return "+".join(kept) if kept else None


def _sent(trial: Trial) -> dict[str, Any] | None:
    """What the trial's main calls carried: result.json's ``settings_sent``,
    or, for a trial written before it, the same count from its per-call
    records (``benchflow.run_checks.settings_sent``)."""
    sent = trial.raw.get("settings_sent")
    if isinstance(sent, dict):
        return sent
    if trial.source != "result.json":
        return None
    from benchflow.run_checks import read_calls, settings_sent

    return settings_sent(read_calls(trial.path), trial.model)


def _versions(trial: Trial) -> tuple[str, ...] | None:
    """The installed packages of the trial's harness receipt, adapter, CLI
    and engine, each once; None when no package's version was read."""
    receipt = trial.raw.get("harness_versions")
    if not isinstance(receipt, Mapping):
        receipt = trial.config.get("harness_versions")
    if not isinstance(receipt, Mapping):
        return None
    found: list[str] = []
    for role in ("adapter", "cli", "engine"):
        part = receipt.get(role)
        if not isinstance(part, Mapping) or part.get("source") != "installed-package":
            continue
        package, version = part.get("package"), part.get("version")
        if package and version and f"{package}@{version}" not in found:
            found.append(f"{package}@{version}")
    return tuple(found) or None


def _requested_effort(trial: Trial) -> str | None:
    block = trial.raw.get("effort")
    if isinstance(block, Mapping) and _effort(block.get("requested")):
        return _effort(block.get("requested"))
    return _effort((trial.config or {}).get("reasoning_effort"))


def _applied_effort(trial: Trial) -> str | None:
    block = trial.raw.get("effort")
    return _effort(block.get("applied")) if isinstance(block, Mapping) else None


def _harness_mode(trial: Trial) -> str | None:
    # As Trial.settings() reads it: older trials predate the option (acp).
    config = trial.config or {}
    if not config:
        return None
    mode = config.get("harness_mode", "acp")
    return mode if isinstance(mode, str) else None


def _dataset(trial: Trial) -> str | None:
    def pick(key: str) -> str | None:
        value = trial.raw.get(key) or (trial.config or {}).get(key)
        return str(value) if value else None

    name, version = pick("dataset_name"), pick("dataset_version")
    if name and version:
        return f"{name}@{version}"
    return name


def _digest(trial: Trial) -> str | None:
    value = trial.raw.get("task_digest") or (trial.config or {}).get("task_digest")
    return str(value) if value else None


def _version(trial: Trial) -> str | None:
    value = (trial.config or {}).get("benchflow_version") or trial.raw.get(
        "benchflow_version"
    )
    return str(value) if value else None


@dataclass
class _Seen:
    """One trial's own observations and its launch."""

    trial: Trial
    launch: tuple[Any, ...]
    model: str | None
    route: str | None
    effort: str | None
    versions: tuple[str, ...] | None
    proxy_saw_calls: bool


def _observe(trial: Trial) -> _Seen:
    sent = _sent(trial)
    model = route = effort = None
    proxy_saw = False
    if sent is not None:
        model = _major(sent.get("models_served"))
        route = _major(sent.get("routes"))
        effort = _major(sent.get("effort"), lower=True)
        source = sent.get("source")
        proxy_saw = int(sent.get("main_calls") or 0) > 0 and source == "proxy"
    launch = (
        str(trial.path.parent),
        trial.agent or "",
        _harness_mode(trial),
        trial.model,
        _requested_effort(trial),
        trial.control,
        _version(trial),
    )
    return _Seen(trial, launch, model, route, effort, _versions(trial), proxy_saw)


def _one(values: set[Any]) -> Any:
    return next(iter(values)) if len(values) == 1 else None


def _task_versions(seen: list[_Seen]) -> dict[tuple[Any, ...], str | None]:
    """Per launch, a short id of the versions it ran of the tasks that ran
    at more than one version among ``seen``; None for every launch when no
    task did."""
    digests: dict[str, set[str]] = {}
    for s in seen:
        digest = _digest(s.trial)
        if digest:
            digests.setdefault(s.trial.task_name, set()).add(digest)
    varying = {task for task, found in digests.items() if len(found) > 1}
    out: dict[tuple[Any, ...], str | None] = {}
    if not varying:
        return dict.fromkeys((s.launch for s in seen), None)
    pairs: dict[tuple[Any, ...], set[tuple[str, str]]] = {}
    for s in seen:
        digest = _digest(s.trial)
        entry = pairs.setdefault(s.launch, set())
        if s.trial.task_name in varying and digest:
            entry.add((s.trial.task_name, digest))
    for launch, found in pairs.items():
        out[launch] = (
            hashlib.sha256(json.dumps(sorted(found)).encode()).hexdigest()[:8]
            if found
            else None
        )
    return out


def trial_settings(trials: Iterable[Trial]) -> list[tuple[Trial, Setting, bool]]:
    """Each trial, in the order given, with its :class:`Setting` and whether
    its own calls showed that effort (see the module docstring)."""
    seen = [_observe(t) for t in trials]
    launches: dict[tuple[Any, ...], dict[str, set[Any]]] = {}
    for s in seen:
        found = launches.setdefault(
            s.launch,
            {"model": set(), "route": set(), "effort": set(), "versions": set()},
        )
        for name, value in (
            ("model", s.model),
            ("route", s.route),
            ("effort", s.effort),
            ("versions", s.versions),
        ):
            if value is not None:
                found[name].add(value)
    task_versions = _task_versions(seen)
    out: list[tuple[Trial, Setting, bool]] = []
    for s in seen:
        t, found = s.trial, launches[s.launch]
        requested = _requested_effort(t)
        applied = _applied_effort(t)
        launch_effort = _one(found["effort"])
        effort: str
        source: str | None
        if s.effort is not None:
            effort, source = s.effort, "calls"
        elif s.proxy_saw_calls:
            # The proxy saw every main call and none carried an effort: that
            # is what ran, not a gap to fill from the launch.
            effort, source = NOT_SENT, "calls"
        elif isinstance(launch_effort, str):
            effort, source = launch_effort, "calls"
        elif applied is not None:
            effort, source = applied, "applied"
        elif requested is not None:
            effort, source = requested, "requested"
        else:
            effort, source = UNKNOWN, None
        served = s.model if s.model is not None else _one(found["model"])
        dataset = _dataset(t)
        version_id = task_versions.get(s.launch)
        task_set = (
            " ".join(
                part
                for part in (
                    dataset,
                    f"(task versions {version_id})" if version_id else "",
                )
                if part
            )
            or None
        )
        setting = Setting(
            agent=t.agent or "",
            harness=_harness_mode(t),
            harness_versions=s.versions
            if s.versions is not None
            else _one(found["versions"]),
            model=served if served is not None else t.model,
            model_source="served" if served is not None else "requested",
            route=s.route if s.route is not None else _one(found["route"]),
            effort=effort,
            effort_source=source,
            effort_requested=requested,
            effort_mismatch=source in ("calls", "applied")
            and requested is not None
            and effort != requested,
            task_set=task_set,
            benchflow_version=_version(t),
            control=t.control,
        )
        out.append((t, setting, s.effort is not None and s.effort == effort))
    return out


@dataclass
class Settings:
    """Trials split by setting (:func:`setting_rows`).

    ``rows`` are agent runs first, then control runs; ``restarted`` holds
    the ``restarted_after_change`` blocks of the jobs the trials came from.
    """

    rows: list[SettingRow]
    restarted: list[dict[str, Any]] = field(default_factory=list)
    # Trial folders the job reader could not read: in no row, and warned of.
    unreadable: list[Path] = field(default_factory=list)

    def counted(self) -> list[SettingRow]:
        """The rows a headline would count: agent runs, or the control runs
        when there are no agent runs."""
        agents = [r for r in self.rows if r.setting.control is None]
        return agents or self.rows

    @property
    def split(self) -> bool:
        """Whether the counted rows are more than one: no figure may pool
        them, each row has its own."""
        return len(self.counted()) > 1

    @property
    def mixed(self) -> bool:
        """Whether the trials read as mixed settings: they ran several
        (:attr:`split`), or a job was restarted after a change, so its
        earlier trials ran other settings and are set aside."""
        return self.split or bool(self.restarted)

    def differ(self) -> list[tuple[str, list[str]]]:
        """Each field the counted rows differ in, with its values in words."""
        rows = self.counted()
        out: list[tuple[str, list[str]]] = []
        if len(rows) < 2:
            return out
        for name, words in FIELDS:
            values = list(dict.fromkeys(r.setting.value(name) for r in rows))
            if len(values) > 1:
                out.append((words, values))
        return out

    def effort_warnings(self) -> list[str]:
        """One warning per row whose calls carried (or whose agent applied)
        another effort than its trials were labelled with: the row is
        reported under the effort that ran."""
        out = []
        for row in self.rows:
            setting = row.setting
            if not setting.effort_mismatch:
                continue
            carried = (
                "the calls carried no effort"
                if setting.effort == NOT_SENT
                else f"the agent applied {setting.effort}"
                if setting.effort_source == "applied"
                else f"the calls carried {setting.effort}"
            )
            out.append(
                f"effort label: {len(row.trials)} trial"
                f"{'' if len(row.trials) == 1 else 's'} labelled "
                f"{setting.effort_requested}, but {carried}; they are reported "
                f"under the effort that ran ({setting.label()})"
            )
        return out

    def warning(self) -> str | None:
        """The ``mixed settings`` warning, or None when nothing is mixed."""
        if not self.mixed:
            return None
        parts: list[str] = []
        rows = self.counted()
        if len(rows) > 1:
            differ = "; ".join(
                f"{name} ({', '.join(values)})" for name, values in self.differ()
            )
            parts.append(
                f"these trials ran {len(rows)} settings"
                + (f", which differ in {differ}" if differ else "")
                + "; each setting is its own row, and no figure pools them"
            )
        for block in self.restarted:
            changed = ", ".join(str(c) for c in block.get("changed") or []) or (
                "its settings"
            )
            trials = block.get("trials")
            finished = block.get("finished_trials")
            parts.append(
                f"the job was restarted after a change ({changed}): "
                + (
                    f"{trials} earlier trial{'' if trials == 1 else 's'}"
                    + (f" ({finished} finished)" if isinstance(finished, int) else "")
                    + " ran with the old settings"
                    if isinstance(trials, int)
                    else "its earlier trials ran with the old settings"
                )
                + f" and are set aside in {block.get('set_aside') or '.set-aside'}, "
                "not counted"
            )
        return "mixed settings: " + "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mixed": self.mixed,
            "split": self.split,
            "differ": [
                {"field": name, "values": values} for name, values in self.differ()
            ],
            "warning": self.warning(),
            "effort_warnings": self.effort_warnings(),
            "restarted_after_change": self.restarted or None,
            "unreadable": [str(path) for path in self.unreadable],
            "rows": [row.to_dict() for row in self.rows],
        }


def setting_rows(
    trials: Iterable[Trial], *, summaries: Iterable[Mapping[str, Any] | None] = ()
) -> Settings:
    """``trials`` (from ``bf.load_job``) split by :class:`Setting`.

    ``summaries`` are the summary.json files of the jobs the trials came
    from, read for ``restarted_after_change``. Agent runs come first, by
    agent, model, route, harness and effort (in the order of
    :data:`EFFORT_ORDER`, then any other effort, then unknown), then control
    runs.
    """
    rows: dict[tuple[Any, ...], SettingRow] = {}
    for trial, setting, checked in trial_settings(trials):
        row = rows.setdefault(setting.key(), SettingRow(setting=setting))
        row.trials.append(trial)
        row.effort_checked += checked
    # The jobs folder keeps a copy of its last job's summary.json: a block
    # read twice is one restart.
    restarted: list[dict[str, Any]] = []
    for summary in summaries:
        block = (
            summary.get("restarted_after_change")
            if isinstance(summary, Mapping)
            else None
        )
        if isinstance(block, Mapping) and dict(block) not in restarted:
            restarted.append(dict(block))
    return Settings(
        rows=sorted(rows.values(), key=lambda r: r.setting.sort_key()),
        restarted=restarted,
    )


def job_summaries(root: Path, trials: Iterable[Trial]) -> list[dict[str, Any]]:
    """The summary.json of each job folder ``trials`` came from (and of
    ``root``), each once: where :func:`setting_rows` reads a restart."""
    folders = {root, *(t.path.parent for t in trials if t.source == "result.json")}
    out = []
    for folder in sorted(folders):
        try:
            data = json.loads((folder / "summary.json").read_text())
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            out.append(data)
    return out


def summary_settings(
    job_dir: Path, records: Iterable[Mapping[str, Any] | None] = ()
) -> dict[str, Any]:
    """summary.json's ``scoring.settings``: the job's trials split by
    setting, each row with the scoring rule's figures, and the mixed-settings
    warning. ``records`` are the job's records that may carry a
    ``restarted_after_change`` block (its summary and run card)."""
    from benchflow.jobs import load_job

    try:
        job = load_job(job_dir)
    except (FileNotFoundError, ValueError):
        return setting_rows([], summaries=records).to_dict()
    settings = setting_rows(job.trials, summaries=records)
    settings.unreadable = job.unreadable
    return settings.to_dict()


#: The file ``bench hub push`` adds to the job it uploads (:func:`scoring_document`).
SCORING_FILE = "scoring.json"
SCORING_SCHEMA = "benchflow.scoring.v1"


def scoring_document(job_dir: Path) -> dict[str, Any] | None:
    """The job counted by BenchFlow's scoring rule, for a reader that counts
    it again (BenchFlow Hub): the rule, each trial's classification
    (``benchflow.scoring.trial_record``, with its setting row), the trials
    split by setting with each row's figures, and the seeds. The same code
    ``bench eval metrics`` and ``bench results`` use. None for a folder with
    no trial."""
    from benchflow.jobs import load_job
    from benchflow.scoring import rule_record, score

    try:
        job = load_job(job_dir)
    except (FileNotFoundError, ValueError):
        return None
    summaries = job_summaries(Path(job_dir), job.trials)
    settings = setting_rows(job.trials, summaries=summaries)
    settings.unreadable = job.unreadable
    trials = []
    for index, row in enumerate(settings.rows):
        for trial, record in zip(row.trials, row.records(), strict=True):
            trials.append({**record, "setting": index, "control": trial.control})
    counted = [r for row in settings.counted() for r in row.records()]
    seeding = next(
        (
            s["scoring"]["seeding"]
            for s in summaries
            if isinstance(s.get("scoring"), dict) and s["scoring"].get("seeding")
        ),
        None,
    )
    return {
        "schema": SCORING_SCHEMA,
        "rule": rule_record(),
        # The headline over the counted rows (agent runs, or the control runs
        # of a job of nothing else); withheld when they are several settings.
        "headline": None if settings.split else score(counted).to_dict(),
        "settings": settings.to_dict(),
        "seeding": seeding,
        "trials": trials,
    }
