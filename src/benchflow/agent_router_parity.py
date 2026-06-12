"""Parity gate for ``bench agent verify`` — parsers, scoring, and verdict.

The ``verify`` subcommand in :mod:`benchflow.agent_router` scores an adopted
benchmark's recorded ``parity_experiment.json`` and assigns a confidence
verdict. That parser/scorer logic lives here so the router module stays focused
on the create/run/verify CLI wiring; the public names are re-exported from
:mod:`benchflow.agent_router` for backwards compatibility.

The gate is *parity only*: a faithful translation must reproduce the original's
behavior on identical inputs (including any reward-hackability the original has)
— parity never "improves" or sanitizes the source. It is also fail-closed: a
half-recorded reward sample, a malformed record, or an unknown schema can never
silently confirm parity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

# ── verify: parity gate + confidence verdict ──────────────────────────

DEFAULT_REWARD_TOLERANCE = 0.02

Verdict = Literal["parity-confirmed", "parity-divergent", "insufficient-evidence"]


@dataclass(frozen=True)
class CriterionComparison:
    """One per-criterion verdict pair from the deterministic parity floor."""

    task_id: str
    criterion_id: str
    original_verdict: str
    adapted_verdict: str
    agreement: bool


@dataclass(frozen=True)
class RewardSample:
    """One legacy-vs-converted reward delta from the statistical layer."""

    task_id: str
    legacy_reward: float | None
    converted_reward: float | None
    delta: float


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


# Deterministic-summary parity blocks the in-repo benchmarks ship: each is a
# pass/total summary rather than a per-criterion verdict list. They form the
# deterministic floor without fabricating reward samples.
_SUMMARY_BLOCK_KEYS = (
    "structural_parity",
    "eval_parity",
    "live_parity",
    "e2e_parity",
    "pipeline_parity",
    "security_parity",
)


def _as_int(value: Any) -> int | None:
    """Return ``value`` as an int only when it is a real, non-bool integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _summary_pass_total(block: Mapping[str, Any]) -> tuple[int, int] | None:
    """Read (passed, total) from a deterministic-summary block.

    Accepts ``tasks_tested``/``passed`` directly, a ``total_tasks``/``passed``/
    ``failed`` results triple, or a nested ``results_summary`` carrying either.
    Returns ``None`` when the block has no integer pass/total signal — so an
    unknown shape contributes no comparison rather than a false confirmation.
    """
    for source in (block, block.get("results_summary"), block.get("results")):
        if not isinstance(source, Mapping):
            continue
        passed = _as_int(source.get("passed"))
        if passed is None:
            continue
        total = _as_int(source.get("tasks_tested"))
        if total is None:
            total = _as_int(source.get("total_tasks"))
        if total is None:
            failed = _as_int(source.get("failed"))
            if failed is not None:
                total = passed + failed
        if total is not None:
            return passed, total
    return None


def _summary_comparisons(data: Mapping[str, Any]) -> list[CriterionComparison]:
    """Synthesize one comparison per deterministic-summary block.

    Agreement is recorded only when the summary explicitly reports
    ``passed == total > 0`` — a partial summary surfaces as a divergence, and a
    zero-tested summary is dropped (never a false confirmation).
    """
    out: list[CriterionComparison] = []
    for key in _SUMMARY_BLOCK_KEYS:
        block = data.get(key)
        if not isinstance(block, Mapping):
            continue
        pass_total = _summary_pass_total(block)
        if pass_total is None:
            continue
        passed, total = pass_total
        if total <= 0:
            continue
        agreement = passed == total
        out.append(
            CriterionComparison(
                task_id=str(data.get("benchmark", "")),
                criterion_id=key,
                original_verdict=f"{total} tasks",
                adapted_verdict=f"{passed} passed",
                agreement=agreement,
            )
        )
    # A top-level deterministic ``results`` summary (no parity sub-block) also
    # counts as the floor for benchmarks that record it bare.
    if not out:
        results = data.get("results")
        if isinstance(results, Mapping):
            pass_total = _summary_pass_total({"results": results})
            if pass_total is not None and pass_total[1] > 0:
                passed, total = pass_total
                out.append(
                    CriterionComparison(
                        task_id=str(data.get("benchmark", "")),
                        criterion_id="results",
                        original_verdict=f"{total} tasks",
                        adapted_verdict=f"{passed} passed",
                        agreement=passed == total,
                    )
                )
    return out


def _normalize_metric_value(value: Any) -> str | None:
    """Normalize a harvey-lab metric value for original-vs-converted comparison.

    Returns a stripped string for an exact-comparison scalar/string (e.g.
    ``"100%"``), or ``None`` for a non-comparable value — including a
    distributional ``mean ± std`` field, whose run noise makes string equality
    meaningless, so it is neither auto-confirmed nor falsely flagged divergent.
    """
    if value is None or isinstance(value, (list, dict)):
        return None
    text = str(value).strip()
    if "±" in text or "+/-" in text:
        return None
    return text


def _metric_comparisons(records: Sequence[Any]) -> list[CriterionComparison]:
    """Read original-vs-converted metric pairs from a top-level record array.

    Only a metric carrying both ``original`` and ``converted`` exact-comparison
    values becomes a comparison; distributional ``mean ± std`` fields with run
    noise are skipped (no equality without an explicit, noise-free match).
    """
    out: list[CriterionComparison] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        benchmark = str(record.get("benchmark", ""))
        for metric in _as_list(record.get("metrics")):
            if not isinstance(metric, Mapping):
                continue
            original = _normalize_metric_value(metric.get("original"))
            converted = _normalize_metric_value(metric.get("converted"))
            if original is None or converted is None:
                continue
            out.append(
                CriterionComparison(
                    task_id=benchmark,
                    criterion_id=str(metric.get("name", "")),
                    original_verdict=original,
                    adapted_verdict=converted,
                    agreement=original == converted,
                )
            )
    return out


def extract_criterion_comparisons(data: Any) -> list[CriterionComparison]:
    """Pull per-criterion verdict pairs from a parity_experiment.json payload.

    Tolerant of the scaffold shape (``conversion_parity.tasks``), the CONVERT.md
    example shape (top-level ``tasks``), the deterministic-summary blocks the
    in-repo benchmarks ship (``structural_parity`` / ``eval_parity`` /
    ``live_parity`` / ``e2e_parity`` / ``pipeline_parity`` / ``security_parity``
    and a bare top-level ``results`` summary, each read as a pass/total floor),
    and a top-level array of records carrying original-vs-converted ``metrics``.
    An unknown non-mapping/non-array top level is not a supported parity schema:
    return no comparisons so the caller reports ``insufficient-evidence``
    instead of crashing on ``.get``.
    """
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return _metric_comparisons(data)
    if not isinstance(data, Mapping):
        return []
    task_lists: list[Any] = []
    conv = data.get("conversion_parity")
    if isinstance(conv, Mapping):
        task_lists.extend(_as_list(conv.get("tasks")))
    task_lists.extend(_as_list(data.get("tasks")))

    out: list[CriterionComparison] = []
    for task in task_lists:
        if not isinstance(task, Mapping):
            continue
        task_id = str(task.get("task_id", ""))
        for crit in _as_list(task.get("criteria_results")):
            if not isinstance(crit, Mapping):
                continue
            original = str(crit.get("original_verdict", ""))
            adapted = str(crit.get("adapted_verdict", ""))
            agreement = bool(crit.get("agreement", original == adapted))
            out.append(
                CriterionComparison(
                    task_id=task_id,
                    criterion_id=str(crit.get("criterion_id", "")),
                    original_verdict=original,
                    adapted_verdict=adapted,
                    agreement=agreement,
                )
            )
    # No explicit per-criterion list: fall back to the deterministic-summary
    # floor the in-repo benchmarks record.
    if not out:
        out.extend(_summary_comparisons(data))
    return out


def _coerce_reward(value: Any) -> float | None:
    """Coerce a reward field to float, or ``None`` for missing/non-numeric.

    Booleans and non-numeric strings are treated as missing so a malformed
    parity record can never crash ``_reward_pair`` on ``float()`` — the sample
    is reported as one-sided/unmeasured rather than confirming parity.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _reward_pair(result: Mapping[str, Any]) -> tuple[float | None, float | None, float]:
    legacy = _coerce_reward(result.get("legacy_reward", result.get("pb_reward")))
    converted = _coerce_reward(result.get("converted_reward", result.get("bf_reward")))
    if legacy is None and isinstance(result.get("programbench"), Mapping):
        legacy = _coerce_reward(result["programbench"].get("reward"))
    if converted is None and isinstance(result.get("benchflow"), Mapping):
        converted = _coerce_reward(result["benchflow"].get("reward"))

    # An explicit reward_delta is the author's recorded measurement and always
    # wins. Otherwise: both sides present -> real delta; exactly one side present
    # -> the sample is unmeasured, so fail closed with an infinite delta that
    # exceeds any tolerance (a half-recorded sample must never confirm parity).
    if "reward_delta" in result:
        delta = abs(_coerce_reward(result.get("reward_delta")) or 0.0)
    elif legacy is not None and converted is not None:
        delta = abs(converted - legacy)
    elif legacy is not None or converted is not None:
        delta = float("inf")
    else:
        delta = 0.0
    return legacy, converted, delta


def extract_reward_samples(data: Any) -> list[RewardSample]:
    """Pull legacy-vs-converted reward deltas (the statistical parity layer).

    Tolerant of the scaffold shape (``reward_distribution_parity.samples``) and
    the reference ``agent_parity.results`` (programbench/benchflow rewards). A
    non-mapping top level is not a supported parity schema: return no samples
    so the caller reports ``insufficient-evidence`` rather than crashing (and
    never fabricates phantom zero-delta samples from summary records).
    """
    if not isinstance(data, Mapping):
        return []
    result_lists: list[Any] = []
    rdp = data.get("reward_distribution_parity")
    if isinstance(rdp, Mapping):
        result_lists.extend(_as_list(rdp.get("samples")))
    agent = data.get("agent_parity")
    if isinstance(agent, Mapping):
        result_lists.extend(_as_list(agent.get("results")))

    out: list[RewardSample] = []
    for result in result_lists:
        if not isinstance(result, Mapping):
            continue
        legacy, converted, delta = _reward_pair(result)
        # A record with no numeric reward on either side and no explicit
        # reward_delta carries no measurable signal — skip it rather than emit a
        # phantom zero-delta sample (keeps malformed/non-numeric records from
        # confirming parity; the caller then reports insufficient-evidence).
        if legacy is None and converted is None and "reward_delta" not in result:
            continue
        out.append(
            RewardSample(
                task_id=str(result.get("task_id", "")),
                legacy_reward=legacy,
                converted_reward=converted,
                delta=delta,
            )
        )
    return out


@dataclass(frozen=True)
class ConversionParity:
    """Deterministic conversion-faithfulness floor."""

    comparisons: list[CriterionComparison]

    @property
    def compared(self) -> int:
        return len(self.comparisons)

    @property
    def agreed(self) -> int:
        return sum(1 for c in self.comparisons if c.agreement)

    @property
    def all_agree(self) -> bool:
        return self.compared > 0 and self.agreed == self.compared

    @property
    def agreement_rate(self) -> float:
        return self.agreed / self.compared if self.compared else 0.0

    @property
    def disagreements(self) -> list[CriterionComparison]:
        return [c for c in self.comparisons if not c.agreement]


@dataclass(frozen=True)
class RewardDistributionParity:
    """Statistical legacy-vs-converted reward parity layer."""

    samples: list[RewardSample]
    tolerance: float

    @property
    def max_abs_delta(self) -> float:
        return max((s.delta for s in self.samples), default=0.0)

    @property
    def within_tolerance(self) -> bool:
        return self.max_abs_delta <= self.tolerance

    @property
    def exceeding(self) -> list[RewardSample]:
        return [s for s in self.samples if s.delta > self.tolerance]


@dataclass(frozen=True)
class VerifyReport:
    """Parity verdict for an adopted benchmark."""

    name: str
    conversion: ConversionParity
    reward: RewardDistributionParity | None
    verdict: Verdict
    tolerance: float = DEFAULT_REWARD_TOLERANCE

    @property
    def passed(self) -> bool:
        return self.verdict == "parity-confirmed"


def build_verify_report(
    name: str,
    data: Any,
    *,
    tolerance: float = DEFAULT_REWARD_TOLERANCE,
) -> VerifyReport:
    """Score parity and assign a confidence verdict.

    Parity-only gate over two layers:

    * deterministic floor — every compared criterion's converted verdict must
      match the original's verdict on identical inputs;
    * statistical layer — every legacy-vs-converted reward delta must sit within
      ``tolerance``.

    A layer that has no data does not block the verdict. With no data at all the
    verdict is ``insufficient-evidence`` (the support path). The gate never
    "improves" the source: a faithful conversion reproduces the original's
    behavior, including any reward-hackability it has.
    """
    conversion = ConversionParity(extract_criterion_comparisons(data))
    samples = extract_reward_samples(data)
    reward = RewardDistributionParity(samples, tolerance=tolerance) if samples else None

    has_conversion = conversion.compared > 0
    has_reward = reward is not None

    if not has_conversion and not has_reward:
        verdict: Verdict = "insufficient-evidence"
    else:
        conversion_ok = (not has_conversion) or conversion.all_agree
        reward_ok = (reward is None) or reward.within_tolerance
        verdict = (
            "parity-confirmed" if conversion_ok and reward_ok else ("parity-divergent")
        )

    return VerifyReport(
        name=name,
        conversion=conversion,
        reward=reward,
        verdict=verdict,
        tolerance=tolerance,
    )


def confidence_line(report: VerifyReport) -> str:
    """User-facing confidence framing (correctness/parity-based, no fixed %)."""
    if report.verdict == "parity-confirmed":
        if report.conversion.compared:
            return (
                "High-confidence: the converted evaluation reproduces the "
                "original's verdicts on every compared criterion and stays "
                "within reward tolerance."
            )
        return (
            "High-confidence: the converted evaluation's rewards stay within "
            "tolerance of the original across every recorded sample."
        )
    if report.verdict == "parity-divergent":
        return (
            "Divergence found: the conversion does not yet reproduce the "
            "original's behavior — iterate, then open an issue for support."
        )
    return (
        "Insufficient evidence: no recorded parity comparisons. Run "
        "parity_test.py and record results before trusting the conversion."
    )


def render_divergence_issue(report: VerifyReport) -> str:
    """Render a draft GitHub issue body for a non-confirmed verdict.

    Printed/saved for a human to file — never auto-filed.
    """
    lines = [
        f"## Benchmark adoption parity: {report.name}",
        "",
        f"**Verdict:** {report.verdict}",
        "",
        confidence_line(report),
        "",
        "### Conversion parity (deterministic floor)",
        f"- criteria compared: {report.conversion.compared}",
        f"- agreed: {report.conversion.agreed}",
        f"- agreement rate: {report.conversion.agreement_rate:.4f}",
    ]
    for c in report.conversion.disagreements:
        lines.append(
            f"  - {c.task_id}/{c.criterion_id}: original={c.original_verdict} "
            f"converted={c.adapted_verdict}"
        )

    lines.append("")
    lines.append("### Reward-distribution parity (statistical layer)")
    if report.reward is None:
        lines.append("- no reward samples recorded")
    else:
        lines.append(f"- samples: {len(report.reward.samples)}")
        lines.append(f"- max abs delta: {report.reward.max_abs_delta:.4f}")
        lines.append(f"- tolerance: {report.reward.tolerance:.4f}")
        for s in report.reward.exceeding:
            lines.append(
                f"  - {s.task_id}: legacy={s.legacy_reward} "
                f"converted={s.converted_reward} delta={s.delta:.4f}"
            )

    lines += [
        "",
        "### Ask",
        "Parity could not be closed for this conversion. The translation must",
        "reproduce the original's behavior on identical inputs (including any",
        "reward-hackability it has). This draft has NOT been filed — review it,",
        "iterate on the converter, and open it manually if you need support.",
    ]
    return "\n".join(lines)
