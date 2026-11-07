"""Verifier-artifact mining for the eval report's failure lines.

The one file-touching corner of the CLI's final report block: when a failed
task's reason would otherwise be the bare ``reward X`` fallback, mine the
rollout's verifier artifacts for a one-line explanation. Lives in its own
module so ``cli/_shared.py`` stays the side-effect-free display helpers it
advertises. Also home of :func:`metric_breakdown`, the one canonical
rewards-dict flattening — shared by ``_shared.py``'s in-memory metric tier and
the ``reward.json`` probe here, so the two render identically.

Contract (the engine stays file-free — these reads are CLI-side, report-time
only):

- The rollout dir is resolved exactly, from the ``rollout_name`` the engine
  records on every result — never guessed from globs.
- Evidence sources are tried in order (a task.md rubric check that matched no
  test, then the CTRF report, then reward.json metric breakdown, then
  test-stdout tail); the first that yields a line wins.
- JSON reads and the retained stdout tail are bounded (``_ARTIFACT_READ_BYTES``).
  Stdout reads add credential-length overlap, redacted before retaining the
  tail, so a seek cannot expose a credential's suffix. Nothing here raises —
  any surprise degrades to ``None``, i.e. the bare reward reason.
- The report block's ``(details: …)`` pointer is decided separately, by
  :func:`verifier_dir_for` from dir existence alone — every failure block
  with artifacts on disk gets one pointer, evidence or not.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, NamedTuple

from benchflow._diagnostics import _literal_spellings, redact_stream, render_diagnostic

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from benchflow._utils.reachable_write import ReachablePath

from pathlib import Path

# Never retain more than this much evidence, and evidence is only mined
# for the <= _MAX_FAILURE_LINES tasks the report block displays.
_ARTIFACT_READ_BYTES = 64 * 1024
# Pytest final summary, decoration stripped: "1 failed, 2 passed in 40.86s".
_PYTEST_SUMMARY_RE = re.compile(r"\d+ (?:failed|errors?)\b")
# The verifier's pytest plugin guard stopping pytest (sandbox/_pytest_plugin_guard.py).
_GUARD_REFUSAL_MARKER = "Verifier plugin trust rejected: "
# Metric-breakdown cap: one metric-happy verifier can't flood the line.
_BREAKDOWN_METRICS = 3


class FailureLine(NamedTuple):
    """A failure one-liner in two parts with distinct truncation contracts.

    ``body`` is the free-text evidence (test name plus assertion) and competes
    for the display line's char budget; ``suffix`` is the compact multi-failure
    roll-up count (empty when the evidence covers everything) and must survive
    display truncation. Renderers give ``body`` the full line budget and append
    ``suffix`` whole past it — never truncating the concatenation, or a long
    assertion would silently eat the "there is more broken than this" signal.
    """

    body: str
    suffix: str = ""


def _captured_text(text: str) -> str:
    """Protect intact child evidence before any lossy selection or formatting."""
    from benchflow.cli._errors import _secrets

    return "".join(redact_stream((text,), _secrets()))


def _numeric_items(mapping: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """The named numeric metrics of one mapping level, insertion-ordered.

    ``reward`` is excluded at every level: at the top it is the aggregate the
    line already leads with, and a nested ``reward`` would render a confusing
    ``reward 0.8 — reward 0.8``.
    """
    return [
        (name, value)
        for name, value in mapping.items()
        if name != "reward" and isinstance(value, (bool, int, float))
    ]


def metric_breakdown(rewards: Mapping[str, Any]) -> str | None:
    """Compact ``name value`` metric breakdown of a rewards mapping, capped at
    ``_BREAKDOWN_METRICS`` — or ``None`` when it holds no named metrics.

    The numeric evidence may sit flat in the mapping, or nested one level under
    a ``metrics`` sub-dict (env0-style verifiers: ``{"reward": …, "metrics":
    {…}, "details": {…}}``), or under ``details`` when ``metrics`` yields
    nothing. Flat keys win outright — a flat rewards dict renders exactly as
    before.

    Ordering leads with the lowest-signal metrics — they explain the miss:

    - A ``<name>_found`` metric with a matching positive numeric
      ``<name>_total`` anywhere in the mapping renders as ``<name> found/total``
      and sorts by that fraction, so ``deadlines 1/5`` leads even though 1 is
      not 0. A total consumed by a pair is dropped from the shown list (the
      fraction already carries it).
    - Unpaired metrics keep the zero-first rule: zero/failed first, stable
      within each group.
    """
    shown = _numeric_items(rewards)
    if not shown:
        for key in ("metrics", "details"):
            sub = rewards.get(key)
            if isinstance(sub, dict):
                shown = _numeric_items(sub)
                if shown:
                    break
    if not shown:
        return None
    # found/total pairing looks across ALL levels: env0 keeps the found counts
    # under "metrics" and the totals under "details". First-seen wins.
    totals: dict[str, Any] = {}
    for source in (rewards, rewards.get("metrics"), rewards.get("details")):
        if isinstance(source, dict):
            for name, value in _numeric_items(source):
                totals.setdefault(name, value)

    def _pair_total(name: str, value: Any) -> Any | None:
        """The positive numeric ``<base>_total`` for a ``<base>_found`` metric,
        else ``None``. Bools (either side) and non-positive totals never pair."""
        base = name.removesuffix("_found")
        if base == name or isinstance(value, bool):
            return None
        total = totals.get(base + "_total")
        if isinstance(total, bool) or not isinstance(total, (int, float)):
            return None
        return total if total > 0 else None

    consumed = {
        name.removesuffix("_found") + "_total"
        for name, value in shown
        if _pair_total(name, value) is not None
    }
    entries: list[tuple[float, str]] = []
    from benchflow.cli._errors import _secrets

    secrets = _secrets()
    for name, value in shown:
        if name in consumed:
            continue
        total = _pair_total(name, value)
        if total is not None:
            fragment = render_diagnostic(
                (name,),
                secrets,
                lambda fields, value=value, total=total: (
                    f"{fields[0].removesuffix('_found')} {value}/{total}"
                ),
            )
            entries.append((value / total, fragment))
        else:
            entries.append(
                (0.0 if value == 0 else 1.0, _captured_text(f"{name} {value}"))
            )
    entries.sort(key=lambda entry: entry[0])  # stable: ties keep dict order
    return ", ".join(fragment for _, fragment in entries[:_BREAKDOWN_METRICS])


def _display_test_name(raw_name: str) -> str:
    """Test name for display: node-id path segments dropped, param id kept.

    Full pytest node ids ("tests/test_x.py::test_build[param]") waste the
    100-char line budget; keep the function name plus any ``[param]`` id.
    The split must not touch the bracket part — a param id may itself contain
    ``::`` — so only the text before the first ``[`` is segment-split. (When
    the report was written by pytest-json-ctrf — as of 0.5.x — the param id
    is already gone — the plugin stores ``nodeid.split('[')[0]`` as ``name``
    — but other CTRF producers keep the full name, and we must not re-trim
    it.)
    """
    head, bracket, param = raw_name.partition("[")
    return head.rsplit("::", 1)[-1] + bracket + param


def _bounded_json(path: Path | ReachablePath) -> dict[str, Any] | None:
    """The file parsed as a JSON object, or ``None`` when it can't serve.

    JSON only parses whole — an oversized (> ``_ARTIFACT_READ_BYTES``) file is
    skipped, not truncated, and a non-object payload is skipped the same way;
    either way the next evidence source gets its turn.
    """
    if path.stat().st_size > _ARTIFACT_READ_BYTES:
        return None
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    return data if isinstance(data, dict) else None


def _ctrf_failure_line(ctrf_path: Path | ReachablePath) -> FailureLine | None:
    """``<test> failed[: <assertion>]`` from the first failed CTRF test.

    The verifier's pytest run writes a CTRF report (``pytest --ctrf``, see the
    standard path in ``task_authoring/structural_checks.py``). Per test the
    useful failure evidence is the ``E …`` assertion line inside ``trace``;
    ``message`` is only a generic phase description, so it is the fallback.
    When the report holds more than one failed test, the first failure's line
    carries a count suffix (``(+N more failures; P/T checks passed)``) so the
    console never under-reports how much is broken.
    """
    data = _bounded_json(ctrf_path)
    if data is None:
        return None
    raw_tests = (data.get("results") or {}).get("tests") or []
    tests = [test for test in raw_tests if isinstance(test, dict)]
    failed = [test for test in tests if test.get("status") == "failed"]
    if not failed:
        return None
    suffix = ""
    if len(failed) > 1:
        # Counts come from the same rolled-up test list the first-failure pick
        # uses, so the suffix can never disagree with the shown evidence.
        # T is all entries — skips count against the denominator, not as passes.
        extra = len(failed) - 1
        passed = sum(1 for test in tests if test.get("status") == "passed")
        plural = "" if extra == 1 else "s"
        suffix = (
            f" (+{extra} more failure{plural}; {passed}/{len(tests)} checks passed)"
        )
    test = failed[0]
    name = _display_test_name(_captured_text(str(test.get("name") or "test")))
    trace = test.get("trace")
    if isinstance(trace, str):
        for line in _captured_text(trace).splitlines():
            stripped = line.strip()
            if stripped.startswith("E "):
                assertion = stripped[2:].strip()
                # "<test> failed:" already says it's an assertion — drop
                # the prefix to reclaim line budget for the message.
                assertion = assertion.removeprefix("AssertionError: ")
                return FailureLine(f"{name} failed: {assertion}", suffix)
    message = test.get("message")
    if isinstance(message, str) and message.strip():
        return FailureLine(
            f"{name} failed: {' '.join(_captured_text(message).split())}", suffix
        )
    return FailureLine(f"{name} failed", suffix)


# What ``taskmd/verify.py`` wrote as a test-judged verdict's rationale when its
# check matched no test: the note (``check X matched no test in ctrf.json
# ...``), or before it the bare sentence older jobs hold.
_OLD_UNMATCHED_RATIONALE = "no test in ctrf.json matches this check"


def _unmatched_check_failure_line(
    review_path: Path | ReachablePath,
) -> FailureLine | None:
    """``check X matched no test in ctrf.json (...); did you mean Y?``.

    A task.md rubric's test-judged criterion whose ``check`` names no test in
    the verifier's CTRF report fails in every trial, the oracle's included,
    whatever the agent did; a typo in the check reads as an agent failure
    unless the summary says so. Read from the verifier's ``review.json``:
    the verdicts the verifier failed because their check matched no test (it
    read the whole CTRF report, so this needs no second matching here, which
    a report over the read cap would get wrong). For an older job, whose
    rationale is the bare sentence, the note is rebuilt from ``ctrf.json``
    beside it when that can be read. Further unmatched checks are counted in
    the suffix.
    """
    from benchflow.taskmd import grading

    review = _bounded_json(review_path)
    if review is None:
        return None
    unmatched: list[tuple[str, str | None]] = []  # (check, the verifier's note)
    for verdict in review.get("verdicts") or []:
        if not isinstance(verdict, dict) or verdict.get("verdict") != "fail":
            continue
        judge = verdict.get("judge")
        if not isinstance(judge, dict) or judge.get("role") != "test":
            continue
        check = verdict.get("check")
        rationale = verdict.get("rationale")
        if not isinstance(check, str) or not isinstance(rationale, str):
            continue
        if rationale.startswith("check ") and " matched no test in " in rationale:
            unmatched.append((check, rationale))
        elif rationale.strip() == _OLD_UNMATCHED_RATIONALE:
            unmatched.append((check, None))
    if not unmatched:
        return None
    # With no valid CTRF report (none written, not JSON, a test without a
    # status) every test-judged check matches nothing, so its note blames a
    # check that may be right. The verifier recorded why there were no tests:
    # say that instead (PT-R2-A-1). Older jobs have no such record.
    extras = review.get("x-benchflow")
    tests_record = extras.get("tests") if isinstance(extras, dict) else None
    problem = tests_record.get("problem") if isinstance(tests_record, dict) else None
    problem = problem.strip() if isinstance(problem, str) else ""
    if isinstance(tests_record, dict):
        no_tests = tests_record.get("count") == 0
        legacy_tests = None
    else:
        # An older job holds no such record: the ctrf.json beside it, and the
        # notes the verifier wrote from it, say whether it had a test. Such
        # jobs blamed a check for a pytest that collected nothing too
        # (PT-CONF-5). A report over the read cap is not judged empty.
        legacy_tests, missing = _legacy_ctrf_tests(review_path.with_name("ctrf.json"))
        no_tests = (
            missing
            or (
                legacy_tests is not None
                and not any(t.get("name") for t in legacy_tests)
            )
            or any(n is not None and n.endswith("(it has none)") for _, n in unmatched)
        )
    if no_tests or problem:
        # A valid-but-empty report often means pytest failed collection. Its
        # stdout summary/guard refusal outranks both rubric and reward.json.
        try:
            detail = _stdout_tail_failure_line(review_path.with_name("test-stdout.txt"))
        except (OSError, ValueError):
            detail = None
        if detail is not None:
            return detail
        if problem:
            n = len(unmatched)
            return FailureLine(
                f"{n} test-judged check{'' if n == 1 else 's'} failed: {problem}"
            )
        if isinstance(tests_record, dict):
            return FailureLine("the verifier reported no tests in ctrf.json")
    extra = len(unmatched) - 1
    suffix = (
        f" (+{extra} more check{'' if extra == 1 else 's'} matched no test)"
        if extra
        else ""
    )
    check, note = unmatched[0]
    if note is None:
        if legacy_tests is not None:
            note = grading.unmatched_check_note(check, legacy_tests)
        else:
            # The report could not be read: say what the verifier found
            # without claiming the report holds no tests.
            note = f"check {check} matched no test in ctrf.json"
    return FailureLine(note, suffix)


def _legacy_ctrf_tests(
    ctrf_path: Path | ReachablePath,
) -> tuple[list[dict[str, Any]] | None, bool]:
    """An older job's CTRF tests, and whether the report is missing or invalid.

    ``(None, False)`` for a report over the read cap: it cannot be read, but
    it is not shown to hold no tests.
    """
    from benchflow.taskmd import grading

    try:
        if not ctrf_path.is_file():
            return None, True
        if ctrf_path.stat().st_size > _ARTIFACT_READ_BYTES:
            return None, False
        data = ctrf_path.read_bytes()[: _ARTIFACT_READ_BYTES + 1]
    except OSError:
        return None, True
    if len(data) > _ARTIFACT_READ_BYTES:
        return None, False  # it grew past the cap after the size check
    # The grader's own reading (names and statuses checked): a report it
    # holds no tests in is invalid here too, so stdout names the cause
    # (PT-CONF-7).
    report = grading.read_ctrf(data)
    if report.problem is not None:
        return None, True
    return report.tests, False


def _reward_json_failure_line(reward_path: Path | ReachablePath) -> FailureLine | None:
    """Metric breakdown mined from the verifier's ``reward.json``.

    env0-style verifiers write ``{"reward": …, "metrics": {…}, "details":
    {…}}`` (and non-pytest stdout, no CTRF report) — when the result's rewards
    dict was stripped to the bare aggregate (older persisted results, sharded
    aggregation), the numeric evidence still sits on disk. Reuses the same
    one-level flattening as the in-memory reward-dict tier so the two render
    identically. A file with no named metrics repeats nothing the line does
    not already say — it yields to the stdout tail. No count suffix: the
    breakdown is a metric summary, not a first-of-N failure pick.
    """
    data = _bounded_json(reward_path)
    if data is None:
        return None
    shown = metric_breakdown(data)
    return FailureLine(shown) if shown is not None else None


def _redacted_tail(raw: Any) -> str:
    """A bounded tail of an opened capture, credentials masked, from a whole line."""
    from benchflow.cli._errors import _secrets

    secrets = _secrets()
    # Four bytes per code point also covers a seek into a UTF-8 sequence.
    # A partial credential at the read boundary ends within this overlap.
    overlap = 4 * max(map(len, _literal_spellings(secrets)), default=1)
    budget = _ARTIFACT_READ_BYTES + overlap
    raw.seek(0, 2)
    start = max(0, raw.tell() - budget)
    raw.seek(start)
    text = raw.read(budget).decode("utf-8", errors="surrogateescape")
    # Redact before discarding even the overlap. Dropping raw overlap could
    # itself bisect a complete credential present in the captured window.
    tail = "".join(
        redact_stream((text[i : i + 8192] for i in range(0, len(text), 8192)), secrets)
    )
    if start:
        # Discard the uncertain prefix and its partial line. This also drops
        # a URL authority of arbitrary length whose scheme precedes the read.
        tail = "".join(tail[overlap:].splitlines(keepends=True)[1:])
    return tail[-_ARTIFACT_READ_BYTES:]


def _stdout_tail_failure_line(stdout_path: Path | ReachablePath) -> FailureLine | None:
    """Last pytest summary ("N failed, M passed …") — else last ``FAILED …``
    line — from a bounded tail of the verifier's test-stdout capture. No count
    suffix: the summary line already carries the full counts itself. A plugin
    refusal by the verifier's pytest guard, which stops pytest before it runs
    any test, is reported as such."""
    with stdout_path.open("rb") as raw:
        tail = _redacted_tail(raw)
    last_failed: str | None = None
    for line in reversed(tail.splitlines()):
        if _GUARD_REFUSAL_MARKER in line:
            # pytest stopped before collecting: the 0 is the guard's, not a
            # test result, and nothing else in the tail explains it.
            refused = line.split(_GUARD_REFUSAL_MARKER, 1)[1].strip()
            return FailureLine(
                "pytest did not run: the verifier guard refused pytest plugin "
                f"{refused} (see docs/tasks/reward-hacking.md)"
            )
        bare = line.strip().strip("=").strip()
        if _PYTEST_SUMMARY_RE.match(bare):
            return FailureLine(bare)
        if last_failed is None and line.strip().startswith("FAILED "):
            last_failed = line.strip()
    return FailureLine(last_failed) if last_failed is not None else None


def verifier_output_tail(
    job_dir: Path, rollout_name: str, lines: int = 5
) -> tuple[ReachablePath, list[str]] | None:
    """test.sh's last ``lines`` output lines (verifier/test-stdout.txt), and the file.

    For the summary of a crashed verifier (DF-p2-task-author-11): its
    traceback is usually there. Bounded and redacted like the rest of this
    module; None when there is no such file or it cannot be read.
    """
    from benchflow._utils.reachable_write import read_stream
    from benchflow.task.paths import RolloutPaths

    try:
        path = RolloutPaths(rollout_dir=job_dir / rollout_name).test_stdout_path
        if not path.is_file():
            return None
        # Through the helper: the folder is the sandbox's to write.
        with read_stream(path) as raw:
            tail = _redacted_tail(raw)
    except Exception:
        return None
    # The capture is the task's own output: no escape sequence or other
    # control character of it reaches the terminal.
    kept = [
        "".join(ch if ch.isprintable() or ch == "\t" else "?" for ch in line).rstrip()
        for line in tail.encode("utf-8", "replace").decode("utf-8").splitlines()
        if line.strip()
    ]
    return (path, kept[-lines:]) if kept else None


def verifier_dir_for(job_dir: Path, rollout_name: str) -> ReachablePath | None:
    """The rollout's verifier dir when it exists on disk, else ``None``.

    Powers the report block's ``(details: …)`` pointer rule: every failure
    block with artifacts on disk gets one pointer — from dir existence alone,
    independent of which tier supplied the reason line (the pointer is most
    valuable exactly when evidence extraction fails). Never raises.
    """
    # Local import: RolloutPaths pulls in the task package (see
    # artifact_failure_evidence).
    from benchflow.task.paths import RolloutPaths

    try:
        verifier_dir = RolloutPaths(rollout_dir=job_dir / rollout_name).verifier_dir
        return verifier_dir if verifier_dir.is_dir() else None
    except Exception:
        return None


def artifact_failure_evidence(job_dir: Path, rollout_name: str) -> FailureLine | None:
    """``FailureLine`` mined from the rollout's verifier artifacts, or ``None``
    when every probe misses. Never raises.
    """
    # Local import: RolloutPaths pulls in the task package, which the CLI
    # shouldn't pay for until a failure actually needs artifact evidence.
    from benchflow.task.paths import RolloutPaths

    rollout_paths = RolloutPaths(rollout_dir=job_dir / rollout_name)
    verifier_dir = rollout_paths.verifier_dir
    attempts: list[
        tuple[ReachablePath, Callable[[Path | ReachablePath], FailureLine | None]]
    ] = [
        # Unmatched rubric checks precede individual failed tests.
        (verifier_dir / "review.json", _unmatched_check_failure_line),
        # ctrf.json has no RolloutPaths property — the verifier recovers it by
        # literal name (verifier_core.py, `_recover_main_verifier_outputs`).
        (verifier_dir / "ctrf.json", _ctrf_failure_line),
        (rollout_paths.reward_json_path, _reward_json_failure_line),
        (rollout_paths.test_stdout_path, _stdout_tail_failure_line),
    ]
    for path, extract in attempts:
        try:
            if not path.is_file():
                continue
            detail = extract(path)
        except Exception:
            # Unreadable file, bad JSON, permissions … try the next source;
            # the report must never crash over evidence mining.
            continue
        if detail is not None:
            return detail
    return None
