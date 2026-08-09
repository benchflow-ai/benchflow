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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

# Never read more than this many bytes per file, and evidence is only mined
# for the <= _MAX_FAILURE_LINES tasks the report block displays.
_ARTIFACT_READ_BYTES = 64 * 1024
# Pytest final summary, decoration stripped: "1 failed, 2 passed in 40.86s".
_PYTEST_SUMMARY_RE = re.compile(r"\d+ failed\b")


def _ctrf_failure_line(ctrf_path: Path) -> str | None:
    """``<test> failed[: <assertion>]`` from the first failed CTRF test.

    The verifier's pytest run writes a CTRF report (``pytest --ctrf``, see the
    standard path in ``task_authoring/structural_checks.py``). Per test the
    useful failure evidence is the ``E …`` assertion line inside ``trace``;
    ``message`` is only a generic phase description, so it is the fallback.
    """
    if ctrf_path.stat().st_size > _ARTIFACT_READ_BYTES:
        # JSON only parses whole — an oversized report is skipped (not
        # truncated) and the next evidence source gets its turn.
        return None
    data = json.loads(ctrf_path.read_text(encoding="utf-8", errors="replace"))
    tests = (data.get("results") or {}).get("tests") or []
    for test in tests:
        if not isinstance(test, dict) or test.get("status") != "failed":
            continue
        # Full pytest node ids ("tests/test_x.py::test_build") waste the
        # 100-char line budget; keep the test function name.
        name = str(test.get("name") or "test").rsplit("::", 1)[-1]
        trace = test.get("trace")
        if isinstance(trace, str):
            for line in trace.splitlines():
                stripped = line.strip()
                if stripped.startswith("E "):
                    assertion = stripped[2:].strip()
                    # "<test> failed:" already says it's an assertion — drop
                    # the prefix to reclaim line budget for the message.
                    assertion = assertion.removeprefix("AssertionError: ")
                    return f"{name} failed: {assertion}"
        message = test.get("message")
        if isinstance(message, str) and message.strip():
            return f"{name} failed: {' '.join(message.split())}"
        return f"{name} failed"
    return None


def _stdout_tail_failure_line(stdout_path: Path) -> str | None:
    """Last pytest summary ("N failed, M passed …") — else last ``FAILED …``
    line — from a bounded tail of the verifier's test-stdout capture."""
    with stdout_path.open("rb") as fh:
        fh.seek(0, 2)  # SEEK_END
        fh.seek(max(0, fh.tell() - _ARTIFACT_READ_BYTES))
        tail = fh.read(_ARTIFACT_READ_BYTES).decode("utf-8", errors="replace")
    last_failed: str | None = None
    for line in reversed(tail.splitlines()):
        bare = line.strip().strip("=").strip()
        if _PYTEST_SUMMARY_RE.match(bare):
            return bare
        if last_failed is None and line.strip().startswith("FAILED "):
            last_failed = line.strip()
    return last_failed


def artifact_failure_evidence(
    job_dir: Path, rollout_name: str
) -> tuple[str, Path] | None:
    """One-liner mined from the rollout's verifier artifacts, or ``None``.

    Returns the one-liner plus the verifier dir it came from (for the report
    block's single ``(details: …)`` pointer line). Never raises.
    """
    # Local import: RolloutPaths pulls in the task package, which the CLI
    # shouldn't pay for until a failure actually needs artifact evidence.
    from benchflow.task.paths import RolloutPaths

    rollout_paths = RolloutPaths(rollout_dir=job_dir / rollout_name)
    verifier_dir = rollout_paths.verifier_dir
    attempts: list[tuple[Path, Callable[[Path], str | None]]] = [
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
