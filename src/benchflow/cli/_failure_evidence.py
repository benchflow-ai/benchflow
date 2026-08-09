"""Verifier-artifact mining for the eval report's failure lines.

The one file-touching corner of the CLI's final report block: when a failed
task's reason would otherwise be the bare ``reward X`` fallback, mine the
rollout's verifier artifacts for a one-line explanation. Lives in its own
module so ``cli/_shared.py`` stays the side-effect-free display helpers it
advertises.

Contract (the engine stays file-free — these reads are CLI-side, report-time
only):

- The rollout dir is resolved exactly, from the ``rollout_name`` the engine
  records on every result — never guessed from globs.
- Evidence sources are tried in order (CTRF report, then test-stdout tail);
  the first that yields a line wins.
- Every read is bounded (``_ARTIFACT_READ_BYTES`` per file) and nothing here
  raises — any surprise degrades to ``None``, i.e. the bare reward reason.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

# Never read more than this many bytes per file, and evidence is only mined
# for the <= _MAX_FAILURE_LINES tasks the report block displays.
_ARTIFACT_READ_BYTES = 64 * 1024
# Pytest final summary, decoration stripped: "1 failed, 2 passed in 40.86s".
_PYTEST_SUMMARY_RE = re.compile(r"\d+ failed\b")


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
    if ctrf_path.stat().st_size > _ARTIFACT_READ_BYTES:
        # JSON only parses whole — an oversized report is skipped (not
        # truncated) and the next evidence source gets its turn.
        return None
    data = json.loads(ctrf_path.read_text(encoding="utf-8", errors="replace"))
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


def artifact_failure_evidence(
    job_dir: Path, rollout_name: str
) -> tuple[FailureLine, Path] | None:
    """``FailureLine`` mined from the rollout's verifier artifacts, or ``None``.

    Returns the structured one-liner plus the verifier dir it came from (for
    the report block's single ``(details: …)`` pointer line). Never raises.
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
            return detail, verifier_dir
    return None
