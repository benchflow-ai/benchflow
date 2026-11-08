"""The run card's plain "why" sentence and failing-check evidence.

Everything here is derived from the rollout's own artifacts (result.json,
the verifier's ctrf.json, the trajectory's tool calls). Nothing is invented:
when an artifact is missing, the sentence says less rather than guessing.
"""

from __future__ import annotations

import re

from .models import (
    Meta,
    Step,
    TimeoutStep,
    ToolStep,
    Verdict,
    VerdictTone,
    VerifierArtifacts,
    VerifierTest,
    iter_event_steps,
)

# pytest ends a failure trace with the frame that raised it:
# "/tests/test_outputs.py:261: AssertionError".
_LOCATION_RE = re.compile(r"^\s*(?P<path>\S+?\.\w+):(?P<line>\d+): \w+\s*$")
# pytest prefixes the explanation lines of a failure with "E".
_EXPLANATION_RE = re.compile(r"^E\s+(?P<text>\S.*)$")
_SENTENCE_DETAIL_CHARS = 200
_ASSERTION_PREFIX_RE = re.compile(r"^AssertionError\b:?\s*")


def failure_location(trace: str | None) -> str | None:
    """``tests/test_outputs.py:261`` from the last frame line of a trace."""
    if not trace:
        return None
    for line in reversed(trace.splitlines()):
        match = _LOCATION_RE.match(line)
        if match:
            return f"{match['path'].lstrip('/')}:{match['line']}"
    return None


def failure_assertion(trace: str | None, message: str | None = None) -> str | None:
    """The first ``E`` line of a pytest trace, else the report's message."""
    for line in (trace or "").splitlines():
        match = _EXPLANATION_RE.match(line)
        if match:
            return match["text"].strip()
    return message.strip() if message and message.strip() else None


def score_tone(meta: Meta) -> VerdictTone:
    """The pass/fail/unscored bucket the header shows (same rule as viewer.js)."""
    outcome = meta.status.score_outcome if meta.status is not None else None
    if outcome == "passed":
        return "pass"
    if outcome == "failed":
        return "fail"
    if outcome or meta.reward is None:
        return "unscored"
    return "pass" if meta.reward == 1 else "fail"


def format_reward(reward: float) -> str:
    """``1.0``, ``0.0``, ``0.75``: one decimal unless the value needs more."""
    return f"{reward:.1f}" if reward == round(reward, 1) else f"{reward:g}"


def short_test_name(name: str) -> str:
    """``test_value_magnitudes`` from ``test_outputs.py::test_value_magnitudes``."""
    return name.rsplit("::", 1)[-1] or name


def failure_clause(test: VerifierTest) -> str:
    """``test_x asserted 3 == 4 (tests/test_outputs.py:12)``, or the message
    of the failure in place of the bare assertion when the test wrote one."""
    name = short_test_name(test.name)
    where = f" ({test.location})" if test.location else ""
    assertion = _clip(test.assertion) if test.assertion else ""
    assertion = _ASSERTION_PREFIX_RE.sub("", assertion)
    if assertion.startswith("assert "):
        return f"{name} asserted {assertion.removeprefix('assert ')}{where}"
    if assertion:
        return f"{name} failed: {assertion}{where}"
    return f"{name} failed{where}"


def _clip(text: str) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    if len(line) > _SENTENCE_DETAIL_CHARS:
        line = line[: _SENTENCE_DETAIL_CHARS - 1].rstrip() + "…"
    return line.rstrip(".")


def _last_tool(steps: list[Step]) -> ToolStep | None:
    tools = [step for step in iter_event_steps(steps) if isinstance(step, ToolStep)]
    return max(tools, key=lambda step: step.i) if tools else None


def _interrupted_tool(steps: list[Step]) -> ToolStep | None:
    """The call a timeout left pending, else the last call still in progress."""
    events = iter_event_steps(steps)
    tools = [step for step in events if isinstance(step, ToolStep)]
    pending: set[str] = set()
    for step in events:
        if isinstance(step, TimeoutStep):
            pending.update(step.timeout.pending)
    named = [step for step in tools if step.tool.id and step.tool.id in pending]
    if named:
        return max(named, key=lambda step: step.i)
    running = [step for step in tools if step.tool.status in ("pending", "in_progress")]
    return max(running, key=lambda step: step.i) if running else None


def _first_error(meta: Meta) -> str | None:
    for banner in meta.errors:
        if banner.level != "info" and banner.text.strip():
            return _clip(banner.text)
    return None


def _tool_clause(step: ToolStep, *, interrupted: bool) -> str:
    title = _clip(step.tool.title) or step.tool.kind
    if interrupted:
        return f" {title} (#{step.i}) was still running."
    return f" The last tool call was {title} (#{step.i}), {step.tool.status}."


def _execution_verdict(
    meta: Meta, steps: list[Step], tone: VerdictTone
) -> Verdict | None:
    status = meta.status
    if status is None or status.execution == "completed":
        return None
    error = _first_error(meta) or (
        "the agent timed out" if status.execution == "timed_out" else "no error text"
    )
    if status.execution == "timed_out":
        cause = error
    else:
        detail = f" ({status.execution_detail})" if status.execution_detail else ""
        cause = f"the agent stopped with an error{detail}: {error}"
    if meta.reward is not None and tone != "unscored":
        sentence = f"Scored {format_reward(meta.reward)} after the run was cut short. {cause[0].upper()}{cause[1:]}."
    else:
        sentence = f"Not scored. {cause[0].upper()}{cause[1:]}."
    interrupted = _interrupted_tool(steps)
    step = interrupted or _last_tool(steps)
    if step is not None:
        sentence += _tool_clause(step, interrupted=interrupted is not None)
    return Verdict(tone=tone, sentence=sentence, step=step.i if step else None)


def _failing_tests(verifier: VerifierArtifacts) -> list[VerifierTest]:
    return [test for test in verifier.ctrf or [] if test.status == "failed"]


def build_verdict(
    meta: Meta, steps: list[Step], verifier: VerifierArtifacts
) -> Verdict:
    """One plain sentence saying what the run scored and why."""
    tone = score_tone(meta)
    from_execution = _execution_verdict(meta, steps, tone)
    if from_execution is not None:
        return from_execution

    if tone == "unscored" or meta.reward is None:
        detail = (
            meta.status.assessment_detail
            if meta.status is not None and meta.status.assessment_detail
            else "no reward recorded"
        )
        return Verdict(
            tone="unscored",
            sentence=f"The agent finished, but the run was not scored: {detail}.",
        )

    score = format_reward(meta.reward)
    tests = verifier.ctrf or []
    failing = _failing_tests(verifier)
    if failing:
        first = failing[0]
        if tone == "fail":
            sentence = f"Scored {score} because {failure_clause(first)}."
        else:
            sentence = (
                f"Scored {score} with {len(failing)} of {len(tests)} tests failing; "
                f"{failure_clause(first)}."
            )
        others = len(failing) - 1
        if tone == "fail" and others:
            sentence += f" {others} more of {len(tests)} tests failed."
        return Verdict(tone=tone, sentence=sentence, test=first.name)
    if tests:
        passed = sum(test.status == "passed" for test in tests)
        if tone == "pass" and passed == len(tests):
            return Verdict(
                tone=tone, sentence=f"Scored {score}: all {len(tests)} tests passed."
            )
        return Verdict(
            tone=tone,
            sentence=f"Scored {score}; {passed} of {len(tests)} tests passed.",
        )
    detail = (
        f" ({meta.status.assessment_detail})"
        if meta.status is not None and meta.status.assessment_detail
        else ""
    )
    sentence = f"Scored {score}{detail}."
    if verifier.ctrf is None and tone == "fail":
        sentence += " The rollout has no per-test results (verifier/ctrf.json)."
    return Verdict(tone=tone, sentence=sentence)
