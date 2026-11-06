"""BenchFlow's scoring rule: one headline number, counted the same way by every reader.

The rule is ``benchflow.score.v1`` (:data:`RULE_ID`, in one sentence in
:data:`RULE_TEXT`):

- Every finished trial is in the denominator: ``n`` is the number of
  finished trials (a retried task's best attempt is one trial). A planned
  trial that never finished is not; summary.json's ``score_of_plan`` counts
  it, and readers name it.
- A trial counts 1 when it was scored and passed (reward 1, or the rubric
  review's pass verdict) and 0 otherwise: a wrong answer, a refusal, an
  error and a trial left with no score all count 0. ``score = passed / n``.
  A trial the model provider or BenchFlow's own limits ended without a score
  (an API error or a rate limit left after retries, a timeout, the idle
  watchdog) counts 0 like any error, and is counted by its ending
  (:data:`ENDINGS`).
- The error-excluding figure, ``score_excl_errors``, is secondary and always
  named as such. It leaves out the trials with an error or no score, and
  keeps a refusal (the plugin guard refused a pytest plugin, or the final
  verify found no usable reward) as a failure, since the agent can cause it.
- A trial whose model refused (a ``refusal`` stop, or a content filter that
  stopped it: result.json's ``model_refusal``) and one a trial budget stopped
  (``trial_budget.stopped_by``) are graded on what the agent left, so each
  counts by that grade, 0 unless the tests passed. Neither changes what the
  rule counts; both are counted apart beside the score (``model_refusals``,
  ``budget_stops``) with their rates, as timeouts are.
- The interval is over tasks, because a task's trials are not independent.
  ``se`` is the cluster-robust standard error of ``score`` with each task's
  trials one cluster and the small-sample factor T/(T-1) for T tasks
  (Miller, "Adding Error Bars to Evals", 2024); None with one task. The 95%
  interval is the one the solve rates give
  (:func:`benchflow.pass_at_k.solve_rate_interval`): Wilson's score interval
  over the trials when every task has one trial or there is one task
  (``wilson``), and otherwise Wilson's interval on the effective sample size
  n / max(1, se² / (p(1-p)/n)), or the number of tasks when every trial
  passed or none did (``wilson-clustered``). It is computed, not resampled,
  so it needs no seed.

:func:`outcome` gives a trial's outcome from its ``result.json``, and
:func:`trial_record` its whole classification (outcome, ending, error
class, retries): the record ``bench hub push`` uploads so the Hub can count
the same way. :func:`score` applies the rule to such records. BenchFlow's
``summary.json``, ``report.md``, ``bench eval metrics``, ``bench results``,
the end-of-run line and the run card's after-run lines all use them.

The conformance fixture ``tests/fixtures/scoring/benchflow.score.v1.json``
lists trial outcomes and the figures this rule gives them. A reader that
reproduces it (BenchFlow Hub, a notebook, another language) counts the same
way. See docs/reference/scoring-rule.md.

>>> s = score([{"task": "a", "outcome": "passed"}, {"task": "a", "outcome": "errored"},
...            {"task": "b", "outcome": "failed"}])
>>> round(s.score, 4), s.n, s.n_tasks, round(s.score_excl_errors, 4), s.ci_method
(0.3333, 3, 2, 0.5, 'wilson-clustered')
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CI_TEXT",
    "ENDINGS",
    "GAP_POINTS",
    "Gap",
    "MIN_TRIES",
    "OUTCOMES",
    "RULE_ID",
    "RULE_TEXT",
    "Score",
    "WARN_RATE",
    "card_lines",
    "cluster_se",
    "gap",
    "no_score_text",
    "outcome",
    "rule_record",
    "score",
    "t975",
    "trial_record",
]

RULE_ID = "benchflow.score.v1"
RULE_TEXT = (
    "Every finished trial is in n and counts 1 only when it was scored and "
    "passed, so a wrong answer, an error, a timeout left with no score and a "
    "trial with no score at all each count 0."
)
#: How the interval is computed, in words (the method name is ``ci_method``).
CI_TEXT = (
    "95% interval over tasks: Wilson's score interval over the trials with one "
    "trial per task or one task (wilson); else Wilson's interval on the "
    "effective sample size n / max(1, se^2 / (p(1-p)/n)), with se the "
    "cluster-robust standard error over tasks (factor T/(T-1)), or on the "
    "number of tasks when every trial passed or none did (wilson-clustered)"
)

#: How a trial counts. ``passed`` counts 1, every other outcome 0; the
#: error-excluding figure keeps ``passed``, ``failed`` and ``refused``.
OUTCOMES = ("passed", "failed", "refused", "errored", "verifier_errored", "no_verdict")
_IN_EXCL_ERRORS = frozenset({"passed", "failed", "refused"})
_NO_SCORE = frozenset({"errored", "verifier_errored", "no_verdict"})

#: Who ended a trial that has no score, when it was not the agent's own
#: doing: the model ``provider`` (its API failed, it rate-limited the run, it
#: refused the credential or the request, the login's usage limit ran out),
#: or the ``harness``, BenchFlow's own limits (the time limit, the idle
#: watchdog, a trial budget). Each is left after the retries the run allows.
ENDINGS = ("provider", "harness")
_PROVIDER_CAUSES = frozenset(
    {
        "api_error",
        "invalid_provider_response",
        "provider_auth",
        "provider_rate_limit",
        "provider_rejected",
        "provider_unavailable",
        "usage_limit",
    }
)
_TIMEOUT_CAUSES = frozenset({"idle_timeout", "timeout"})
_HARNESS_CAUSES = _TIMEOUT_CAUSES | {"trial_budget"}
_RATE_LIMIT_CAUSES = frozenset({"provider_rate_limit", "usage_limit"})

#: Above this share of trials, the error, timeout and rate-limit rates warn.
WARN_RATE = 0.05
#: Fewer trials per task than this warns: one or two tries per task say
#: little about a task's pass rate, and pass@k needs k tries.
MIN_TRIES = 3


@dataclass(frozen=True)
class Score:
    """The rule applied to a set of trials (see the module docstring).

    ``score`` and ``score_excl_errors`` are None when their denominator is
    empty; ``se`` is None with fewer than two tasks; ``ci95`` is None with
    no trial. ``outcomes`` counts the trials per outcome (every outcome
    listed); ``endings`` the trials with no score by who ended them
    (:data:`ENDINGS`), ``error_classes`` by error class; ``timeouts`` and
    ``rate_limited`` the trials that ran out of time (scored or not) and that
    met a rate limit; ``tries`` the fewest and most trials of a task;
    ``model_refusals`` the trials whose model refused (or a content filter
    stopped it) and ``budget_stops`` those a trial budget stopped, scored or
    not.
    """

    n: int
    n_tasks: int
    passed: int
    score: float | None
    n_excl_errors: int
    score_excl_errors: float | None
    se: float | None
    ci95: tuple[float, float] | None
    ci_method: str | None
    outcomes: dict[str, int] = field(default_factory=dict)
    endings: dict[str, int] = field(default_factory=dict)
    error_classes: dict[str, int] = field(default_factory=dict)
    timeouts: int = 0
    rate_limited: int = 0
    tries: tuple[int, int] = (0, 0)
    rule: str = RULE_ID
    # Trials whose model refused or a content filter stopped (a model
    # refusal, not the plugin guard's), and trials a trial budget stopped.
    model_refusals: int = 0
    budget_stops: int = 0

    @property
    def errors(self) -> int:
        """Trials with an error or no score (each counts 0)."""
        return self.n - self.n_excl_errors

    def rates(self) -> dict[str, float | None]:
        """The error, timeout, rate-limit, model-refusal and budget-stop
        rates over every trial."""
        if not self.n:
            return dict.fromkeys(
                ("error", "timeout", "rate_limit", "model_refusal", "budget_stop")
            )
        return {
            "error": self.errors / self.n,
            "timeout": self.timeouts / self.n,
            "rate_limit": self.rate_limited / self.n,
            "model_refusal": self.model_refusals / self.n,
            "budget_stop": self.budget_stops / self.n,
        }

    def warnings(self) -> list[str]:
        """What the figures warn about: few tries per task, and an error,
        timeout, rate-limit, model-refusal or budget-stop rate above
        :data:`WARN_RATE`."""
        out: list[str] = []
        if self.n and self.tries[0] < MIN_TRIES:
            fewest = self.tries[0]
            out.append(
                f"k < {MIN_TRIES}: {'every' if self.tries[1] == fewest else 'some'} "
                f"task has {fewest} {'trial' if fewest == 1 else 'trials'}; with "
                f"fewer than {MIN_TRIES} tries per task the interval and pass@k say "
                f"little about each task (run {MIN_TRIES} or more trials per task)"
            )
        for name, rate in self.rates().items():
            if rate is None or rate <= WARN_RATE:
                continue
            count = {
                "error": self.errors,
                "timeout": self.timeouts,
                "rate_limit": self.rate_limited,
                "model_refusal": self.model_refusals,
                "budget_stop": self.budget_stops,
            }[name]
            what = {
                "error": "ended with an error or no score (each counts 0)",
                "timeout": "ran out of time",
                "rate_limit": "met a rate limit or a usage limit",
                "model_refusal": "ended on a model refusal or a content filter",
                "budget_stop": "were stopped at their token or turn budget",
            }[name]
            out.append(
                f"{name.replace('_', '-')} rate {rate:.1%}: {count} of {self.n} trials "
                f"{what}, above {WARN_RATE:.0%}"
            )
        return out

    def model_refusals_text(self) -> str:
        """The trials whose model refused, beside the score: ``2 of 50
        trials (4.0%) ended on a model refusal or a content filter ...``;
        empty when none did."""
        if not self.n or not self.model_refusals:
            return ""
        return (
            f"{self.model_refusals} of {self.n} trials ({self.model_refusals / self.n:.1%}) "
            "ended on a model refusal or a content filter (each counts 0 unless "
            "its tests passed)"
        )

    def budget_stops_text(self) -> str:
        """The trials a trial budget stopped, beside the score; empty when
        none was."""
        if not self.n or not self.budget_stops:
            return ""
        return (
            f"{self.budget_stops} of {self.n} trials ({self.budget_stops / self.n:.1%}) "
            "were stopped at their token or turn budget (each counts 0 unless its "
            "tests passed)"
        )

    def interval_text(self) -> str:
        """The interval in words, as reports print it."""
        if self.ci95 is None:
            return "no interval: no trial"
        low, high = self.ci95
        how = (
            "clustered by task"
            if self.ci_method == "wilson-clustered"
            else "one trial per task"
            if self.n_tasks > 1
            else "one task, its trials as the sample"
        )
        return f"95% interval {low:.1%} to {high:.1%}, {how}"

    def excl_errors_text(self) -> str:
        """The error-excluding figure in words, named as such; empty when no
        trial had an error or went unscored."""
        left_out = self.errors
        if not left_out:
            return ""
        figure = (
            f"{self.passed} of {self.n_excl_errors} passed ({self.score_excl_errors:.1%})"
            if self.score_excl_errors is not None
            else "no trial is left"
        )
        return (
            f"leaving out the {left_out} trial{'' if left_out == 1 else 's'} with an "
            f"error or no score, {figure}"
        )

    def endings_text(self) -> str:
        """The trials with no score by who ended them and by error class,
        e.g. ``2 ended by the provider (api_error 1, provider_rate_limit 1)``;
        empty when every trial was scored."""
        if not self.errors:
            return ""
        parts = []
        limits = (
            "by BenchFlow's limits"
            if "trial_budget" in self.error_classes
            else "by BenchFlow's time limits"
        )
        for name, words in (
            ("provider", "by the model provider"),
            ("harness", limits),
        ):
            count = self.endings.get(name, 0)
            if count:
                classes = ", ".join(
                    f"{cls} {k}"
                    for cls, k in sorted(self.error_classes.items())
                    if _ending_of_class(cls) == name
                )
                parts.append(f"{count} ended {words} ({classes})")
        other = self.errors - sum(self.endings.values())
        if other:
            classes = ", ".join(
                f"{cls} {k}"
                for cls, k in sorted(self.error_classes.items())
                if _ending_of_class(cls) is None
            )
            parts.append(f"{other} other" + (f" ({classes})" if classes else ""))
        return (
            f"{self.errors} with no score, each counted 0: " + "; ".join(parts)
            if parts
            else ""
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Score:
        """The :class:`Score` a :meth:`to_dict` (summary.json's ``scoring``) holds."""
        ci95 = data.get("ci95")
        tries = data.get("tries") or {}
        return cls(
            n=int(data.get("n") or 0),
            n_tasks=int(data.get("n_tasks") or 0),
            passed=int(data.get("passed") or 0),
            score=data.get("score"),
            n_excl_errors=int(data.get("n_excl_errors") or 0),
            score_excl_errors=data.get("score_excl_errors"),
            se=data.get("se"),
            ci95=(float(ci95[0]), float(ci95[1])) if ci95 else None,
            ci_method=data.get("ci_method"),
            outcomes=dict(data.get("outcomes") or {}),
            endings=dict(data.get("endings") or {}),
            error_classes=dict(data.get("error_classes") or {}),
            timeouts=int(data.get("timeouts") or 0),
            rate_limited=int(data.get("rate_limited") or 0),
            tries=(int(tries.get("min") or 0), int(tries.get("max") or 0)),
            rule=str(data.get("rule") or RULE_ID),
            model_refusals=int(data.get("model_refusals") or 0),
            budget_stops=int(data.get("budget_stops") or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form, the shape the conformance fixture's ``expected`` uses."""
        return {
            "rule": self.rule,
            "n": self.n,
            "n_tasks": self.n_tasks,
            "passed": self.passed,
            "score": self.score,
            "n_excl_errors": self.n_excl_errors,
            "score_excl_errors": self.score_excl_errors,
            "se": self.se,
            "ci95": list(self.ci95) if self.ci95 is not None else None,
            "ci_method": self.ci_method,
            "outcomes": dict(self.outcomes),
            "endings": dict(self.endings),
            "error_classes": dict(sorted(self.error_classes.items())),
            "timeouts": self.timeouts,
            "rate_limited": self.rate_limited,
            "model_refusals": self.model_refusals,
            "budget_stops": self.budget_stops,
            "tries": {"min": self.tries[0], "max": self.tries[1]},
            "rates": self.rates(),
            "warnings": self.warnings(),
        }


def rule_record() -> dict[str, str]:
    """``{"id", "text", "interval"}``: how summary.json names the rule it applied."""
    return {"id": RULE_ID, "text": RULE_TEXT, "interval": CI_TEXT}


def _ending_of_class(cls: str) -> str | None:
    if cls in _PROVIDER_CAUSES:
        return "provider"
    if cls in _HARNESS_CAUSES:
        return "harness"
    return None


def outcome(result: Mapping[str, Any], *, integration_failed: bool = False) -> str:
    """How the rule counts one trial, from its ``result.json``.

    The score bucket is BenchFlow's own (``classify_score_outcome``: an
    explicit reward or a complete review verdict decides passed or failed);
    a trial with no score is then ``refused`` when its verifier error is a
    plugin guard refusal or an unusable reward file, ``verifier_errored``
    for any other verifier error, ``errored`` when the agent, its sandbox or
    its model provider errored, and ``no_verdict`` when nothing recorded why
    (an assessment still pending). ``integration_failed``: the agent's
    integration broke, which a reader may detect after the fact
    (``Trial.integration_failure``); the trial is then ``errored`` whatever
    reward its verifier wrote.
    """
    from benchflow._utils.scoring import (
        classify_score_outcome,
        is_guard_refusal,
        is_unusable_reward,
    )

    if integration_failed:
        return "errored"
    bucket = classify_score_outcome(result)
    if bucket in ("passed", "failed"):
        return bucket
    if bucket == "verifier_errored":
        verifier_error = result.get("verifier_error")
        if is_guard_refusal(verifier_error) or is_unusable_reward(verifier_error):
            return "refused"
        return "verifier_errored"
    return "errored" if result.get("error") else "no_verdict"


def _error_class(result: Mapping[str, Any], kind: str) -> str | None:
    """The cause of a trial with no score (``failures.cause_of``'s key), or
    for a scored one the category its run's error had (a graded timeout)."""
    from benchflow._utils.scoring import classify_error

    if kind in _NO_SCORE or kind == "refused":
        from benchflow.failures import cause_of

        cause = cause_of(dict(result))
        if cause is not None:
            return cause.key
    category = result.get("error_category") or classify_error(result.get("error"))
    return category if isinstance(category, str) and category else None


def trial_record(
    result: Mapping[str, Any],
    *,
    integration_failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One trial's classification by the rule: ``task``, ``trial`` (its
    folder's name), ``outcome``, ``ending`` (:data:`ENDINGS`, or None), ``error_class``
    (the cause of a trial with no score, as ``bench explain`` names it, or a
    scored trial's error category), ``timed_out`` (it ran out of time, scored
    or not), ``rate_limited`` (a rate limit or usage limit ended it, or one of
    its calls got HTTP 429), ``model_refusal`` (its model refused or a content
    filter stopped it: result.json's ``model_refusal``, or for a result
    written before that field its ``agent_outcome`` status), ``budget_stop``
    (the measure of its trial budget that stopped it, ``tokens`` or
    ``turns``, or None) and the retries its calls show
    (``agent_retries``: failed calls the agent tried again;
    ``upstream_retries``: the model proxy's own extra attempts).
    ``integration_failure`` is the trial's broken agent integration
    (``Trial.integration_failure``, stored or detected on read): the trial
    then counts as errored whatever reward its verifier wrote.
    """
    data: dict[str, Any] = dict(result)
    if integration_failure is not None:
        data.update(
            rewards=None,
            error=data.get("error") or "agent integration failure",
            error_category="agent_integration",
            integration_failure_info=dict(integration_failure),
        )
    kind = outcome(data, integration_failed=integration_failure is not None)
    cls = _error_class(data, kind)
    category = data.get("error_category")
    found = data.get("agent_outcome")
    agent_outcome: dict[str, Any] = found if isinstance(found, dict) else {}
    statuses = agent_outcome.get("http_status_counts")
    hit_429 = isinstance(statuses, Mapping) and bool(statuses.get("429"))
    end = _ending_of_class(cls) if cls and kind in _NO_SCORE else None

    def count(key: str) -> int | None:
        value = agent_outcome.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    if "model_refusal" in data:
        refused = isinstance(data.get("model_refusal"), Mapping)
    else:
        refused = agent_outcome.get("status") == "refusal"
    budget = data.get("trial_budget")
    stopped_by = budget.get("stopped_by") if isinstance(budget, Mapping) else None

    return {
        "task": str(data.get("task_name") or ""),
        "trial": str(data.get("rollout_name") or ""),
        "outcome": kind,
        "ending": end,
        "error_class": cls,
        "timed_out": category in _TIMEOUT_CAUSES or cls in _TIMEOUT_CAUSES,
        "rate_limited": hit_429
        or category in _RATE_LIMIT_CAUSES
        or cls in _RATE_LIMIT_CAUSES,
        "model_refusal": refused,
        "budget_stop": stopped_by if stopped_by in ("tokens", "turns") else None,
        "agent_retries": count("agent_retries"),
        "upstream_retries": count("upstream_retries"),
    }


def _records(trials: Iterable[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, trial in enumerate(trials):
        if isinstance(trial, Mapping):
            record = dict(trial)
        else:
            task, kind = trial
            record = {"task": task, "outcome": kind}
        task, kind = record.get("task"), record.get("outcome")
        if not isinstance(task, str) or not task:
            raise ValueError(f"trial {index} has no task name: {trial!r}")
        if kind not in OUTCOMES:
            raise ValueError(
                f"trial {index} ({task}) has outcome {kind!r}; "
                f"an outcome is one of {', '.join(OUTCOMES)}"
            )
        end = record.get("ending")
        if end is not None and end not in ENDINGS:
            raise ValueError(
                f"trial {index} ({task}) has ending {end!r}; an ending is one of "
                f"{', '.join(ENDINGS)}, or null"
            )
        records.append(record)
    return records


def cluster_se(counts: list[tuple[int, int]]) -> float | None:
    """The cluster-robust standard error of the pooled pass rate over
    ``(trials, passes)`` per task, with the factor T/(T-1); None with fewer
    than two tasks.

    >>> round(cluster_se([(1, 1), (1, 0)]), 4)
    0.5
    """
    counts = [(n, c) for n, c in counts if n > 0]
    total = sum(n for n, _ in counts)
    tasks = len(counts)
    if tasks < 2 or not total:
        return None
    p = sum(c for _, c in counts) / total
    spread = math.fsum((c - p * n) ** 2 for n, c in counts)
    return math.sqrt(tasks / (tasks - 1) * spread) / total


def score(trials: Iterable[Mapping[str, Any] | tuple[str, str]]) -> Score:
    """Apply ``benchflow.score.v1`` to ``trials``.

    Each trial is a mapping with ``task`` (its task's name, the cluster) and
    ``outcome`` (one of :data:`OUTCOMES`; :func:`outcome` gives it from a
    ``result.json``), and optionally the rest of a :func:`trial_record`
    (``ending``, ``error_class``, ``timed_out``, ``rate_limited``), or a
    ``(task, outcome)`` pair. An unknown outcome is a ``ValueError``, never
    counted as anything.
    """
    from benchflow.pass_at_k import solve_rate_interval

    records = _records(trials)
    counts = Counter(record["outcome"] for record in records)
    n = len(records)
    passed = counts["passed"]
    per_task: dict[str, list[int]] = {}
    endings: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    for record in records:
        entry = per_task.setdefault(record["task"], [0, 0])
        entry[0] += 1
        entry[1] += record["outcome"] == "passed"
        if record["outcome"] in _NO_SCORE:
            if record.get("ending"):
                endings[record["ending"]] += 1
            if record.get("error_class"):
                classes[str(record["error_class"])] += 1
    pairs = [(k, c) for k, c in per_task.values()]
    interval = solve_rate_interval(pairs)
    excl_n = sum(counts[kind] for kind in _IN_EXCL_ERRORS)
    tries = [k for k, _ in pairs]
    return Score(
        n=n,
        n_tasks=len(per_task),
        passed=passed,
        score=passed / n if n else None,
        n_excl_errors=excl_n,
        score_excl_errors=passed / excl_n if excl_n else None,
        se=cluster_se(pairs),
        ci95=interval[0] if interval else None,
        ci_method=interval[1] if interval else None,
        outcomes={kind: counts[kind] for kind in OUTCOMES},
        endings={name: endings[name] for name in ENDINGS},
        error_classes=dict(classes),
        timeouts=sum(1 for record in records if record.get("timed_out") is True),
        rate_limited=sum(1 for record in records if record.get("rate_limited") is True),
        tries=(min(tries), max(tries)) if tries else (0, 0),
        model_refusals=sum(
            1 for record in records if record.get("model_refusal") is True
        ),
        budget_stops=sum(1 for record in records if record.get("budget_stop")),
    )


def card_lines(block: Mapping[str, Any]) -> list[str]:
    """The run card's after-run lines for summary.json's ``scoring`` block:
    the score with its interval, the trials with no score by who ended
    them, the rule's warnings, and the setting or settings the trials ran."""
    figures = Score.from_dict(block)
    if not figures.n:
        return []
    lines = [
        f"  Score:     {figures.passed} of {figures.n} trials passed "
        f"({figures.score or 0.0:.1%}; {figures.interval_text()}) by {figures.rule}"
    ]
    refusals = figures.model_refusals_text()
    if not refusals and "model_refusals" in block:
        # The card gives the refusal rate beside the score when it is 0 too;
        # a block from before the count was kept says nothing of it.
        refusals = (
            f"0 of {figures.n} trials (0.0%) ended on a model refusal or a "
            "content filter"
        )
    if refusals:
        lines.append(f"  Refusals:  {refusals}")
    if figures.budget_stops_text():
        lines.append(f"  Budgets:   {figures.budget_stops_text()}")
    endings = figures.endings_text()
    if endings:
        lines.append(f"  No score:  {endings}")
    lines += [f"  Warning:   {warning}" for warning in figures.warnings()]
    seeding = block.get("seeding")
    if isinstance(seeding, Mapping) and seeding.get("text"):
        lines.append(f"  Seeds:     {seeding['text']}")
        lines += [f"  Warning:   {w}" for w in seeding.get("warnings") or []]
    settings = block.get("settings")
    if not isinstance(settings, Mapping):
        return lines
    rows = [r for r in settings.get("rows") or [] if isinstance(r, Mapping)]
    lines += [f"  Warning:   {w}" for w in settings.get("effort_warnings") or []]
    if settings.get("mixed"):
        lines.append(f"  Settings:  {settings.get('warning')}")
        for row in rows:
            each = Score.from_dict(row.get("scoring") or {})
            lines.append(
                f"    {row.get('label')}: {each.passed} of {each.n} passed "
                f"({each.score or 0.0:.1%}; {each.interval_text()})"
            )
    elif rows:
        lines.append(f"  Setting:   {rows[0].get('label')}")
    return lines


#: A gap between two scores under this many points warns, whatever its
#: interval: a published study of infrastructure noise found gaps below 3
#: points deserve skepticism.
GAP_POINTS = 0.03


# Student t's 97.5% quantile for 1 to 30 degrees of freedom: the gap's
# interval over few tasks is wider than 1.96 standard errors.
_T975 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)  # fmt: skip


def t975(df: int) -> float:
    """Student t's 97.5% quantile for ``df`` degrees of freedom: the table
    for 1 to 30, the Cornish-Fisher expansion beyond (2.042 at 30, 1.997 at
    65, toward 1.96).

    >>> t975(2), round(t975(65), 3)
    (4.303, 1.997)
    """
    if df < 1:
        raise ValueError("a t quantile needs at least 1 degree of freedom")
    if df <= len(_T975):
        return _T975[df - 1]
    z = 1.959964
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


@dataclass(frozen=True)
class Gap:
    """Side B's score minus side A's over the tasks both ran, by the rule.

    ``se`` is the cluster-robust standard error of the difference with each
    task one cluster (the task's residuals on both sides together, factor
    T/(T-1)), so a task both sides find easy or hard cancels out; ``ci95``
    is ``gap +/- t se`` with Student t's 97.5% quantile for T-1 degrees of
    freedom (:func:`t975`). With the same trials per task this is the t
    interval over tasks of the per-task differences. None with fewer than
    two paired tasks.
    """

    a: Score
    b: Score
    gap: float | None
    se: float | None
    ci95: tuple[float, float] | None
    tasks: int

    @property
    def inside(self) -> bool:
        """The interval of the gap holds 0: the order is not measured."""
        return self.ci95 is not None and self.ci95[0] <= 0.0 <= self.ci95[1]

    @property
    def small(self) -> bool:
        """The gap is under :data:`GAP_POINTS`."""
        return self.gap is not None and abs(self.gap) < GAP_POINTS

    def text(self, labels: tuple[str, str] = ("A", "B")) -> str:
        la, lb = labels
        if self.gap is None:
            return f"{lb} - {la}: no gap (a side has no trial on the paired tasks)"
        interval = (
            f"95% interval {self.ci95[0] * 100:+.1f} to {self.ci95[1] * 100:+.1f} "
            "points, paired by task"
            if self.ci95 is not None
            else "no interval: fewer than two paired tasks"
        )
        return f"{lb} - {la} = {self.gap * 100:+.1f} points ({interval})"

    def warnings(self, labels: tuple[str, str] = ("A", "B")) -> list[str]:
        """The gap's warning: inside its interval, or under 3 points."""
        if self.gap is None:
            return []
        why = []
        if self.inside:
            why.append("its 95% interval holds 0")
        if self.small:
            why.append(f"it is under {GAP_POINTS * 100:.0f} points")
        if not why:
            return []
        la, lb = labels
        return [
            f"the gap {self.text(labels)}: {' and '.join(why)}, so which of {la} "
            f"and {lb} scores higher is not measured"
        ]

    def to_dict(self, labels: tuple[str, str] = ("A", "B")) -> dict[str, Any]:
        return {
            "rule": RULE_ID,
            "tasks": self.tasks,
            "a": self.a.to_dict(),
            "b": self.b.to_dict(),
            "gap": self.gap,
            "gap_se": self.se,
            "gap_ci95": list(self.ci95) if self.ci95 is not None else None,
            "gap_inside_ci": self.inside,
            "gap_under_3_points": self.small,
            "warnings": [
                *(f"{labels[0]}: {w}" for w in self.a.warnings()),
                *(f"{labels[1]}: {w}" for w in self.b.warnings()),
                *self.warnings(labels),
            ],
        }


def gap(a: Iterable[Mapping[str, Any]], b: Iterable[Mapping[str, Any]]) -> Gap:
    """The rule over two sides' trial records (:func:`trial_record`), and the
    gap B - A. Both are best kept to the tasks both sides ran.

    >>> g = gap([{"task": "t", "outcome": "failed"}, {"task": "u", "outcome": "passed"}],
    ...         [{"task": "t", "outcome": "passed"}, {"task": "u", "outcome": "passed"}])
    >>> round(g.gap, 2), g.tasks
    (0.5, 2)
    """
    ra, rb = _records(a), _records(b)
    sa, sb = score(ra), score(rb)
    if sa.score is None or sb.score is None:
        return Gap(sa, sb, None, None, None, 0)
    diff = sb.score - sa.score
    residual: dict[str, float] = {}
    for records, figures, sign in ((ra, sa, -1.0), (rb, sb, 1.0)):
        p, n = figures.score or 0.0, figures.n
        for record in records:
            passed = 1.0 if record["outcome"] == "passed" else 0.0
            residual[record["task"]] = (
                residual.get(record["task"], 0.0) + sign * (passed - p) / n
            )
    tasks = len(residual)
    if tasks < 2:
        return Gap(sa, sb, diff, None, None, tasks)
    se = math.sqrt(tasks / (tasks - 1) * math.fsum(r * r for r in residual.values()))
    half = t975(tasks - 1) * se
    return Gap(sa, sb, diff, se, (diff - half, diff + half), tasks)


def _counted(n: int, word: str, plural: str | None = None) -> str:
    return f"{n} {word if n == 1 else (plural or word + 's')}"


def no_score_text(
    outcomes: Mapping[str, int], *, errored_notes: Iterable[str] = ()
) -> str:
    """The trials with no score in the words every reader uses, from the
    rule's counts by outcome: ``U unscored (V verifier errors, R refused), E
    errored (N with no verdict)``. Unscored trials finished and the verifier
    gave them no score; errored ones stopped before a verdict, as the Score
    line's ``errors=`` counts them. Every one of them counts 0.
    ``errored_notes`` are more words for the errored ones (the integration
    failures among them)."""
    verifier = outcomes.get("verifier_errored", 0)
    refused = outcomes.get("refused", 0)
    pending = outcomes.get("no_verdict", 0)
    errored = outcomes.get("errored", 0) + pending
    inside = ", ".join(
        part
        for part in (
            _counted(verifier, "verifier error") if verifier else "",
            f"{refused} refused" if refused else "",
        )
        if part
    )
    notes = [*errored_notes, f"{pending} with no verdict" if pending else ""]
    after = "; ".join(note for note in notes if note)
    return (
        f"{verifier + refused} unscored"
        + (f" ({inside})" if inside else "")
        + f", {errored} errored"
        + (f" ({after})" if after else "")
    )
