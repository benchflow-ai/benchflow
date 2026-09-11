"""Batch orchestration for continuing many timed-out runs."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchflow.continue_run.orchestrator import ContinueResult, continue_run
from benchflow.continue_run.run_folder import (
    ContinueUnsupportedError,
    RunFolderError,
    is_timeout_run,
    load_run_folder,
)

ContinueRunner = Callable[..., Awaitable[ContinueResult]]
#: Notified once per timeout candidate that `benchflow continue` cannot resume.
SkipObserver = Callable[[Path, ContinueUnsupportedError], None]


@dataclass(frozen=True)
class BatchContinueResult:
    """Result for one source folder in a batch continuation.

    ``skipped`` separates *out of scope* from *failed*: a run whose agent has no
    replay ingress, or that was never recorded, did not fail — the batch simply
    could not act on it. Batch callers must not treat a skip as an error.
    """

    folder: Path
    ok: bool
    continued: ContinueResult | None = None
    error: str | None = None
    skipped: bool = False
    reason_code: str | None = None
    recoverable_in_principle: bool | None = None

    @classmethod
    def skip(cls, folder: Path, exc: ContinueUnsupportedError) -> BatchContinueResult:
        """Record an out-of-scope run, carrying the gate's typed reason."""
        return cls(
            folder=folder,
            ok=False,
            error=str(exc),
            skipped=True,
            reason_code=exc.reason_code,
            recoverable_in_principle=exc.recoverable_in_principle,
        )


def discover_timeout_run_folders(
    root: str | Path,
    *,
    limit: int | None = None,
    on_skip: SkipObserver | None = None,
) -> list[Path]:
    """Find OpenHands timeout run folders below ``root``.

    Discovery is intentionally artifact-based: a candidate must have a
    ``config.json`` and a usable ``trajectory/llm_trajectory.jsonl``. Non-timeout
    runs are skipped by ``load_run_folder(require_timeout=True)``.

    ``on_skip`` is notified for each folder that *is* a timeout candidate but
    that ``benchflow continue`` cannot resume (unsupported agent, or no LLM
    recording). Without it those runs vanish silently, and the operator never
    learns that part of the sweep was left behind. Finished runs from other
    agents are not reported — they were never candidates.
    """
    root_path = Path(root).expanduser()
    candidates = [root_path] if (root_path / "config.json").is_file() else []
    candidates.extend(path.parent for path in root_path.rglob("config.json"))

    folders: list[Path] = []
    seen: set[Path] = set()
    for folder in sorted(candidates):
        resolved = folder.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            load_run_folder(folder, require_timeout=True)
        except ContinueUnsupportedError as exc:
            if on_skip is not None and is_timeout_run(folder):
                on_skip(folder, exc)
            continue
        except RunFolderError:
            continue
        folders.append(folder)
        if limit is not None and len(folders) >= limit:
            break
    return folders


async def continue_batch(
    folders: list[Path],
    *,
    concurrency: int,
    tasks_dir: str | Path | None,
    model: str | None,
    timeout: int | None,
    output_dir: str | Path | None,
    require_timeout: bool = True,
    strict_divergence: bool = False,
    proxy_mode: str = "auto",
    runner: ContinueRunner = continue_run,
) -> list[BatchContinueResult]:
    """Run ``benchflow continue`` over folders with rolling concurrency."""
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    semaphore = asyncio.Semaphore(concurrency)

    async def _one(folder: Path) -> BatchContinueResult:
        async with semaphore:
            try:
                result = await runner(
                    folder,
                    tasks_dir=tasks_dir,
                    model=model,
                    timeout=timeout,
                    output_dir=output_dir,
                    require_timeout=require_timeout,
                    strict_divergence=strict_divergence,
                    proxy_mode=proxy_mode,
                )
            except ContinueUnsupportedError as exc:
                # Out of scope, not broken: one unsupported run in a 200-run
                # sweep must not cost the other 199 their continuation.
                return BatchContinueResult.skip(folder, exc)
            except Exception as exc:
                return BatchContinueResult(folder=folder, ok=False, error=str(exc))
            if result.error:
                return BatchContinueResult(
                    folder=folder,
                    ok=False,
                    continued=result,
                    error=result.error,
                )
            return BatchContinueResult(folder=folder, ok=True, continued=result)

    return list(await asyncio.gather(*(_one(folder) for folder in folders)))


def summarize_batch(results: list[BatchContinueResult]) -> dict[str, Any]:
    """Small JSON-serializable summary for CLI output and dashboards.

    ``skipped``/``skips`` are reported apart from ``failed``/``errors`` so a
    sweep containing runs `continue` cannot resume is not scored as a batch of
    failures — and so the operator still sees which runs were left behind, why,
    and whether a future replay ingress could reach them.
    """
    ok = [result for result in results if result.ok]
    skipped = [result for result in results if not result.ok and result.skipped]
    failed = [result for result in results if not result.ok and not result.skipped]
    return {
        "total": len(results),
        "succeeded": len(ok),
        "failed": len(failed),
        "skipped": len(skipped),
        "outputs": [
            str(result.continued.rollout_dir)
            for result in ok
            if result.continued is not None
        ],
        "skips": [
            {
                "folder": str(result.folder),
                "reason": result.reason_code,
                "recoverable_in_principle": result.recoverable_in_principle,
                "detail": result.error,
            }
            for result in skipped
        ],
        "errors": [
            {
                "folder": str(result.folder),
                "output": str(result.continued.rollout_dir)
                if result.continued is not None
                else None,
                "error": result.error,
            }
            for result in failed
        ],
    }
