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
- Evidence sources are tried in order (CTRF report, then reward.json metric
  breakdown, then test-stdout tail); the first that yields a line wins.
- Every read is bounded (``_ARTIFACT_READ_BYTES`` per file) and nothing here
  raises — any surprise degrades to ``None``, i.e. the bare reward reason.
- The report block's ``(details: …)`` pointer is decided separately, by
  :func:`verifier_dir_for` from dir existence alone — every failure block
  with artifacts on disk gets one pointer, evidence or not.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

# Never read more than this many bytes per file, and evidence is only mined
# for the <= _MAX_FAILURE_LINES tasks the report block displays.
_ARTIFACT_READ_BYTES = 64 * 1024
# Pytest final summary, decoration stripped: "1 failed, 2 passed in 40.86s".
_PYTEST_SUMMARY_RE = re.compile(r"\d+ failed\b")
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
    for name, value in shown:
        if name in consumed:
            continue
        total = _pair_total(name, value)
        if total is not None:
            base = name.removesuffix("_found")
            entries.append((value / total, f"{base} {value}/{total}"))
        else:
            entries.append((0.0 if value == 0 else 1.0, f"{name} {value}"))
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


def _bounded_json(path: Path) -> dict[str, Any] | None:
    """The file parsed as a JSON object, or ``None`` when it can't serve.

    JSON only parses whole — an oversized (> ``_ARTIFACT_READ_BYTES``) file is
    skipped, not truncated, and a non-object payload is skipped the same way;
    either way the next evidence source gets its turn.
    """
    if path.stat().st_size > _ARTIFACT_READ_BYTES:
        return None
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    return data if isinstance(data, dict) else None


def _ctrf_failure_line(ctrf_path: Path) -> FailureLine | None:
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
    name = _display_test_name(str(test.get("name") or "test"))
    trace = test.get("trace")
    if isinstance(trace, str):
        for line in trace.splitlines():
            stripped = line.strip()
            if stripped.startswith("E "):
                assertion = stripped[2:].strip()
                # "<test> failed:" already says it's an assertion — drop
                # the prefix to reclaim line budget for the message.
                assertion = assertion.removeprefix("AssertionError: ")
                return FailureLine(f"{name} failed: {assertion}", suffix)
    message = test.get("message")
    if isinstance(message, str) and message.strip():
        return FailureLine(f"{name} failed: {' '.join(message.split())}", suffix)
    return FailureLine(f"{name} failed", suffix)


def _reward_json_failure_line(reward_path: Path) -> FailureLine | None:
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


def _stdout_tail_failure_line(stdout_path: Path) -> FailureLine | None:
    """Last pytest summary ("N failed, M passed …") — else last ``FAILED …``
    line — from a bounded tail of the verifier's test-stdout capture. No count
    suffix: the summary line already carries the full counts itself."""
    with stdout_path.open("rb") as fh:
        fh.seek(0, 2)  # SEEK_END
        fh.seek(max(0, fh.tell() - _ARTIFACT_READ_BYTES))
        tail = fh.read(_ARTIFACT_READ_BYTES).decode("utf-8", errors="replace")
    last_failed: str | None = None
    for line in reversed(tail.splitlines()):
        bare = line.strip().strip("=").strip()
        if _PYTEST_SUMMARY_RE.match(bare):
            return FailureLine(bare)
        if last_failed is None and line.strip().startswith("FAILED "):
            last_failed = line.strip()
    return FailureLine(last_failed) if last_failed is not None else None


def verifier_dir_for(job_dir: Path, rollout_name: str) -> Path | None:
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
    attempts: list[tuple[Path, Callable[[Path], FailureLine | None]]] = [
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
