"""Binary verdict validation and HealthBench-style plan-score aggregation."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from typing import Any

from benchflow.review.config import ReviewCriterion, ReviewRubric

STATUS_SCORED = "scored"
STATUS_UNSCORED = "unscored"
STATUS_COMPROMISED = "compromised"
STATUS_CONFIG_ERROR = "config_error"
STATUS_ERROR = "error"
STATUS_SKIPPED = "skipped"


@dataclass
class CriterionVerdict:
    """One parsed binary criterion verdict."""

    criterion_id: str
    criterion_met: bool | None
    explanation: str = ""
    evidence: tuple[str, ...] = ()
    score: float | None = None
    unscored_reason: str | None = None


@dataclass
class ReviewOutcome:
    """Aggregated review result, ready for result and reward serialization."""

    status: str
    plan: float | None = None
    passed: bool | None = None
    verdicts: list[CriterionVerdict] = field(default_factory=list)
    failed_gates: list[str] = field(default_factory=list)
    error: str | None = None

    def reward_updates(self, rubric: ReviewRubric | None) -> dict[str, float]:
        if self.status != STATUS_SCORED or self.plan is None or rubric is None:
            return {}
        updates: dict[str, float] = {
            "plan": round(self.plan, 4),
            "plan_passed": 1.0 if self.passed else 0.0,
        }
        for verdict in self.verdicts:
            if verdict.score is not None:
                updates[f"plan/{verdict.criterion_id}"] = round(verdict.score, 4)
        return updates


def extract_verdicts_objects(text: str) -> list[dict[str, Any]]:
    """Return every balanced JSON object carrying a ``verdicts`` key."""

    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    index = text.find("{")
    while index != -1:
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and "verdicts" in obj:
            objects.append(obj)
        index = text.find("{", index + 1)
    return objects


def extract_verdicts_object(text: str) -> dict[str, Any] | None:
    """Return the last candidate, which is normally the reviewer's final answer."""

    objects = extract_verdicts_objects(text)
    return objects[-1] if objects else None


def _trace_json_values(evidence_trace: str) -> list[Any]:
    """Decode adjacent JSON values from an accumulated evidence trace."""

    values: list[Any] = []
    decoder = json.JSONDecoder()
    index = 0
    while index < len(evidence_trace):
        while index < len(evidence_trace) and evidence_trace[index].isspace():
            index += 1
        if index >= len(evidence_trace):
            break
        try:
            value, end = decoder.raw_decode(evidence_trace[index:])
        except json.JSONDecodeError:
            break
        values.append(value)
        index += end
    return values


def _provider_tool_calls(evidence_trace: str) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        if value.get("type") == "provider_tool_call":
            name = value.get("name")
            arguments = value.get("arguments")
            if isinstance(name, str) and isinstance(arguments, dict):
                calls.append((name, arguments))

    for value in _trace_json_values(evidence_trace):
        visit(value)
    return calls


def _cited_tool_call(item: str) -> tuple[str, dict[str, Any]] | None:
    """Parse a reviewer citation such as ``read_file(file_path='...')``."""

    try:
        expression = ast.parse(item, mode="eval").body
    except (SyntaxError, ValueError):
        return None
    if (
        not isinstance(expression, ast.Call)
        or not isinstance(expression.func, ast.Name)
        or expression.args
        or not expression.keywords
    ):
        return None
    arguments: dict[str, Any] = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            return None
        try:
            arguments[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            return None
    return expression.func.id, arguments


def _evidence_is_trace_backed(item: str, evidence_trace: str | None) -> bool:
    if evidence_trace is None:
        return True
    needle = item.strip()
    if not needle:
        return False
    if needle in evidence_trace:
        return True
    # Tool titles often shorten absolute paths.  Require the cited basename to
    # appear in an actual read/search event; a free-form claim alone is not
    # independently checkable.
    path = needle.split(":", 1)[0].strip("`'\"")
    basename = path.rstrip("/").rsplit("/", 1)[-1]
    if basename and len(basename) >= 3 and basename in evidence_trace:
        return True

    cited_call = _cited_tool_call(needle)
    if cited_call is None:
        return False
    cited_name, cited_arguments = cited_call
    return any(
        cited_name == actual_name and cited_arguments == actual_arguments
        for actual_name, actual_arguments in _provider_tool_calls(evidence_trace)
    )


def _parse_candidate(
    obj: dict[str, Any],
    expected: list[ReviewCriterion],
    evidence_trace: str | None,
) -> tuple[list[CriterionVerdict], str | None]:
    raw_verdicts = obj.get("verdicts")
    if not isinstance(raw_verdicts, list):
        return [], "'verdicts' must be a list"

    by_id: dict[str, dict[str, Any]] = {}
    for item in raw_verdicts:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            return [], "every verdicts[] entry must be an object with an 'id'"
        if item["id"] in by_id:
            return [], f"duplicate verdict id {item['id']!r}"
        by_id[item["id"]] = item

    expected_ids = [criterion.id for criterion in expected]
    missing = [
        criterion_id for criterion_id in expected_ids if criterion_id not in by_id
    ]
    extra = [criterion_id for criterion_id in by_id if criterion_id not in expected_ids]
    if missing or extra:
        return [], (
            "verdict ids do not match the requested criteria "
            f"(missing: {missing or 'none'}, unexpected: {extra or 'none'}); "
            f"answer exactly the ids {expected_ids}"
        )

    verdicts: list[CriterionVerdict] = []
    for criterion in expected:
        item = by_id[criterion.id]
        criterion_met = item.get("criterion_met")
        explanation = item.get("explanation")
        evidence_raw = item.get("evidence")
        if not isinstance(criterion_met, bool):
            return [], f"{criterion.id}: criterion_met must be a JSON boolean"
        if not isinstance(explanation, str) or not explanation.strip():
            return [], f"{criterion.id}: explanation must be a non-empty string"
        if (
            not isinstance(evidence_raw, list)
            or not evidence_raw
            or not all(
                isinstance(value, str) and value.strip() for value in evidence_raw
            )
        ):
            return [], f"{criterion.id}: evidence must be a non-empty list of strings"
        evidence = tuple(value.strip() for value in evidence_raw)
        unsupported = [
            value
            for value in evidence
            if not _evidence_is_trace_backed(value, evidence_trace)
        ]
        if unsupported:
            return [], (
                f"{criterion.id}: evidence is not backed by reviewer tool/search "
                f"events: {unsupported}"
            )
        verdicts.append(
            CriterionVerdict(
                criterion_id=criterion.id,
                criterion_met=criterion_met,
                explanation=explanation.strip(),
                evidence=evidence,
                score=1.0 if criterion_met else 0.0,
            )
        )
    return verdicts, None


def parse_reviewer_message(
    text: str,
    expected: list[ReviewCriterion],
    *,
    evidence_trace: str | None = None,
) -> tuple[list[CriterionVerdict], str | None]:
    """Parse the last structurally valid, complete verdict object in ``text``."""

    objects = extract_verdicts_objects(text)
    if not objects:
        return [], (
            "no JSON object with a 'verdicts' key was found in the reply; "
            "return exactly one such object"
        )
    last_error: str | None = None
    for obj in reversed(objects):
        verdicts, error = _parse_candidate(obj, expected, evidence_trace)
        if error is None:
            return verdicts, None
        last_error = error
    return [], last_error or "no complete verdict object was found"


def aggregate(rubric: ReviewRubric, verdicts: list[CriterionVerdict]) -> ReviewOutcome:
    """Aggregate binary verdicts with gates and a positive-only denominator."""

    by_id = {verdict.criterion_id: verdict for verdict in verdicts}
    unscored = [
        verdict
        for verdict in verdicts
        if verdict.score is None
        and not _criterion(rubric, verdict.criterion_id).is_metric
    ]
    if unscored:
        reasons = "; ".join(
            f"{verdict.criterion_id}: {verdict.unscored_reason}" for verdict in unscored
        )
        return ReviewOutcome(
            status=STATUS_UNSCORED,
            verdicts=verdicts,
            error=f"plan review is unscored: {reasons}",
        )

    failed_gates = [
        gate.id for gate in rubric.gates if (by_id[gate.id].score or 0.0) < 1.0
    ]
    if failed_gates:
        plan = 0.0
    else:
        denominator = sum(
            criterion.weight for criterion in rubric.scored if criterion.weight > 0
        )
        numerator = sum(
            criterion.weight * (by_id[criterion.id].score or 0.0)
            for criterion in rubric.scored
        )
        plan = min(1.0, max(0.0, numerator / denominator))

    return ReviewOutcome(
        status=STATUS_SCORED,
        plan=plan,
        passed=plan >= rubric.pass_threshold,
        verdicts=verdicts,
        failed_gates=failed_gates,
    )


def _criterion(rubric: ReviewRubric, criterion_id: str) -> ReviewCriterion:
    for criterion in rubric.criteria:
        if criterion.id == criterion_id:
            return criterion
    raise KeyError(criterion_id)
