"""The report every job writes: ``report.md``, a short page beside ``summary.json``.

First the totals by BenchFlow's scoring rule (``benchflow.scoring``): the
score with its 95% interval over tasks, the error-excluding figure, who ended
the trials with no score, and the rule's warnings; one line per setting
instead when the trials ran more than one (``benchflow.setting_groups``).
Then per task, the trials, the passes, a 95% interval on the task's pass rate
(Wilson over its trials), the spread of rewards across tries and a bar; then
the time and tokens, and the sampling the model requests carried, read from
each trial's ``trajectory/llm_trajectory.jsonl``, next to the vendor's setting
the run card showed. A ``--trials`` run also writes one beside its
``matrix-summary.json`` that counts each task's tries across the repeats.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPORT = "report.md"
_BAR = 10
# Requests read per trial for the sampling line: the sampling rarely changes
# within a trial, and a long trial's log can run to tens of megabytes.
_REQUESTS_READ = 20
_SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "max_tokens",
    "max_completion_tokens",
    "reasoning_effort",
    "thinking",
    "chat_template_kwargs",
)


def wilson(passed: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval for ``passed`` of ``n`` (0 to 1 when n is 0)."""
    if n <= 0:
        return 0.0, 1.0
    p = passed / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _trials(job_dirs: list[Path]) -> tuple[list[dict[str, Any]], list[str], Any]:
    """The job's trials as ``bf.load_job`` counts them, the planned tasks
    with no trial, and the trials split by setting.

    A retried task is one trial, its best attempt, and a trial passes by the
    scoring rule's verdict, the one the job's summary uses (a rubric review
    that passes its gates passes below a reward of 1), so the report and
    ``summary.json`` agree.
    """
    from benchflow.jobs import load_job
    from benchflow.setting_groups import job_summaries, setting_rows

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    trials: list[Any] = []
    summaries: list[dict[str, Any]] = []
    unreadable: list[Path] = []
    for job_dir in job_dirs:
        try:
            job = load_job(job_dir)
        except Exception as exc:  # a report is never worth a failed run
            logger.debug("report: cannot read %s: %s", job_dir, exc)
            continue
        interrupted, not_started = job.unfinished()
        missing += [*interrupted, *not_started]
        trials += job.trials
        unreadable += job.unreadable
        summaries += job_summaries(Path(job_dir), job.trials)
    settings = setting_rows(trials, summaries=summaries)
    settings.unreadable = unreadable
    for row in settings.rows:
        for trial, record in zip(row.trials, row.records(), strict=True):
            rows.append(
                {
                    "task": trial.task_name,
                    "reward": trial.reward,
                    "passed": record["outcome"] == "passed",
                    "rule": record,
                    "scored": trial.assessment == "scored",
                    "tokens": trial.total_tokens or 0,
                    "cost": trial.cost_usd,
                    "seconds": trial.duration_sec,
                    "dir": trial.path if trial.path.is_dir() else None,
                    "outcome": _read_outcome(trial.path),
                }
            )
    rows.sort(key=lambda row: (row["task"], str(row["dir"] or "")))
    return rows, missing, settings


def _read_outcome(path: Path) -> dict[str, Any]:
    try:
        data = json.loads((path / "result.json").read_text())
    except (OSError, ValueError):
        return {}
    outcome = data.get("agent_outcome") if isinstance(data, dict) else None
    return outcome if isinstance(outcome, dict) else {}


def _outcome_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not any(row.get("outcome") for row in rows):
        return []
    lines = [
        "",
        "Model outcomes beside verifier scores (unknown counts were not recorded):",
        "",
        "| Trial | Verifier reward | Model outcome | HTTP errors | Refused calls |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        outcome = row.get("outcome") or {}
        reward = "no score" if row["reward"] is None else f"{row['reward']:g}"
        trial = row["dir"].name if row["dir"] is not None else row["task"]
        values = [
            outcome.get("status", "unknown"),
            outcome.get("http_errors"),
            outcome.get("refused_calls"),
        ]
        rendered = ["unknown" if value is None else str(value) for value in values]
        lines.append(f"| {trial} | {reward} | " + " | ".join(rendered) + " |")
    return lines


def _sampling(trial_dirs: list[Path]) -> dict[str, Counter]:
    """Each sampling key the requests carried, and how many requests sent each value."""
    seen: dict[str, Counter] = defaultdict(Counter)
    for trial in trial_dirs:
        path = trial / "trajectory" / "llm_trajectory.jsonl"
        try:
            with path.open() as lines:
                for count, line in enumerate(lines):
                    if count >= _REQUESTS_READ:
                        break
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    request = (
                        record.get("request") if isinstance(record, dict) else None
                    )
                    body = request.get("body") if isinstance(request, dict) else None
                    if not isinstance(body, dict):
                        continue
                    for key in _SAMPLING_KEYS:
                        if key in body:
                            seen[key][json.dumps(body[key], sort_keys=True)] += 1
        except OSError:
            continue
    return seen


def _value(encoded: str) -> str:
    """A sampling value as JSON spells it, a string without its quotes."""
    value = json.loads(encoded)
    return value if isinstance(value, str) else encoded


def _bar(rate: float) -> str:
    filled = round(rate * _BAR)
    return "█" * filled + "░" * (_BAR - filled)


def _totals(rows: list[dict[str, Any]], figures: Any) -> str:
    """``P passed (S; interval), mean reward M (...), U with no score``."""
    rewards = [row["reward"] or 0.0 for row in rows]
    return (
        f"{figures.passed} passed ({figures.score or 0.0:.1%}; "
        f"{figures.interval_text()}), mean reward "
        f"{sum(rewards) / len(rewards):.2f} (a trial with no score counts 0), "
        f"{sum(not row['scored'] for row in rows)} with no score"
    )


def _seeding(job_dirs: list[Path]) -> dict[str, Any] | None:
    """The jobs' seeds (summary.json's ``scoring.seeding``): one record when
    every job's says the same, else one that says they differ."""
    found = []
    for job_dir in job_dirs:
        try:
            summary = json.loads((job_dir / "summary.json").read_text())
        except (OSError, ValueError):
            continue
        record = ((summary or {}).get("scoring") or {}).get("seeding")
        if isinstance(record, dict) and record.get("text"):
            found.append(record)
    if not found:
        return None
    if all(record["text"] == found[0]["text"] for record in found):
        return found[0]
    return {"text": "the jobs differ in their seeds (see each job's summary.json)"}


def _card(job_dirs: list[Path]) -> dict[str, Any] | None:
    for job_dir in job_dirs:
        try:
            return json.loads((job_dir / "run-card.json").read_text())
        except (OSError, ValueError):
            continue
    return None


def build_report(job_dirs: list[Path], title: str) -> str:
    """The report's Markdown over the trials of ``job_dirs``."""
    from benchflow.run_card import describe_vendor_record
    from benchflow.scoring import RULE_ID, RULE_TEXT, score

    rows, missing, settings = _trials(job_dirs)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    card = _card(job_dirs) or {}
    agent = (card.get("agent") or {}).get("name")
    model = (card.get("model") or {}).get("id")
    lines = [f"# Report: {title}", ""]
    if not rows:
        return "\n".join([*lines, "No finished trials.", ""])
    head = f"{len(rows)} trials of {len(by_task)} tasks"
    if not settings.split:
        whole = score(row["rule"] for row in rows)
        lines.append(
            head
            + (f", {agent}" if agent else "")
            + (f" on {model}" if model else "")
            + f": {_totals(rows, whole)}."
        )
        excl = whole.excl_errors_text()
        lines.append(
            f"Counted by the scoring rule {RULE_ID}: {RULE_TEXT}"
            + (f" {excl[0].upper()}{excl[1:]}." if excl else "")
        )
        endings = whole.endings_text()
        if endings:
            lines.append(f"{endings[0].upper()}{endings[1:]}.")
        lines += [
            f"{text[0].upper()}{text[1:]}."
            for text in (whole.model_refusals_text(), whole.budget_stops_text())
            if text
        ]
        lines += [f"Warning: {warning}." for warning in whole.warnings()]
        if settings.mixed:
            lines.append(f"Warning: {settings.warning()}.")
    else:
        # The trials ran several settings: one number over them would hide
        # the difference, so each setting gets its own line.
        counted = settings.counted()
        lines.append(
            f"{head} in {len(counted)} settings. One number over them would mix "
            "them, so each setting has its own line, counted by the scoring rule "
            f"{RULE_ID} ({RULE_TEXT})"
        )
        lines.append("")
        for row in settings.rows:
            lines.append(f"- {row.line()}.")
            lines += [f"  - Warning: {w}." for w in row.score().warnings()]
        lines += ["", f"Warning: {settings.warning()}."]
    lines += [f"Warning: {warning}." for warning in settings.effort_warnings()]
    if settings.unreadable:
        from benchflow.jobs import unreadable_warning

        lines.append(f"Warning: {unreadable_warning(settings.unreadable)}.")
    seeds = _seeding(job_dirs)
    if seeds is not None:
        lines.append(f"Seeds: {seeds['text']}.")
        lines += [f"Warning: {warning}." for warning in seeds.get("warnings") or []]
    if missing:
        lines.append(
            f"Not in these numbers: {len(missing)} planned "
            f"task{'' if len(missing) == 1 else 's'} with no finished trial (the job "
            f"stopped or was interrupted): {', '.join(sorted(missing)[:10])}"
            + (", ..." if len(missing) > 10 else "")
            + "."
        )
    lines += [
        "",
        "| Task | Trials | Passed | Pass rate | 95% interval | Rewards | |",
        "|---|---|---|---|---|---|---|",
    ]
    for task in sorted(by_task):
        trials = by_task[task]
        n, ok = len(trials), sum(1 for row in trials if row["passed"])
        lo, hi = wilson(ok, n)
        spread = ", ".join(
            "no score" if row["reward"] is None else f"{row['reward']:g}"
            for row in trials
        )
        lines.append(
            f"| {task} | {n} | {ok} | {ok / n:.0%} | {lo:.0%} to {hi:.0%} | {spread} | "
            f"{_bar(ok / n)} |"
        )
    lines += _outcome_lines(rows)
    tokens = sum(row["tokens"] for row in rows)
    costs = [row["cost"] for row in rows if isinstance(row["cost"], int | float)]
    seconds = [
        row["seconds"] for row in rows if isinstance(row["seconds"], int | float)
    ]
    lines += [
        "",
        (
            f"Tokens {tokens:,}; cost "
            + (f"${sum(costs):.2f}" if costs else "not known (no price for the model)")
            if tokens
            else "No model tokens were used"
        )
        + (
            f"; trial time {min(seconds):.0f} to {max(seconds):.0f} s, "
            f"{sum(seconds) / 60:.0f} trial-minutes in all"
            if seconds
            else ""
        )
        + ".",
        "",
    ]
    sampling = _sampling([row["dir"] for row in rows if row["dir"] is not None])
    if sampling:
        sent = "; ".join(
            f"{key} " + ", ".join(f"{_value(v)} ({n})" for v, n in values.most_common())
            for key, values in sampling.items()
        )
        unsent = [key for key in ("temperature", "top_p") if key not in sampling]
        lines.append(
            f"Sampling the model requests carried (the first {_REQUESTS_READ} of each "
            f"trial, with how many sent each value): {sent}"
            + (
                f"; no {' or '.join(unsent)} sent, so the model server's default applies"
                if unsent
                else ""
            )
            + "."
        )
    else:
        lines.append(
            "Sampling the model requests carried: not recorded (no requests through "
            "BenchFlow's proxy)."
        )
    vendor = (card.get("model") or {}).get("vendor_setting")
    if vendor:
        lines.append(f"The run card's Vendor line: {describe_vendor_record(vendor)}.")
    lines.append("")
    return "\n".join(lines)


def write_report(job_dirs: list[Path], out_dir: Path, title: str) -> Path | None:
    """Write ``out_dir/report.md``; None when it could not be written."""
    try:
        path = out_dir / REPORT
        path.write_text(build_report(job_dirs, title))
        return path
    except Exception as exc:  # a report is never worth a failed run
        logger.warning("Could not write %s: %s", out_dir / REPORT, exc)
        return None
