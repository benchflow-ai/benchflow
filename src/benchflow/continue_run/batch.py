"""Batch orchestration for continuing many timed-out runs."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchflow.continue_run.orchestrator import ContinueResult, continue_run
from benchflow.continue_run.run_folder import RunFolderError, load_run_folder

ContinueRunner = Callable[..., Awaitable[ContinueResult]]


@dataclass(frozen=True)
class BatchContinueResult:
    """Result for one source folder in a batch continuation."""

    folder: Path
    ok: bool
    continued: ContinueResult | None = None
    error: str | None = None


def discover_timeout_run_folders(
    root: str | Path, *, limit: int | None = None
) -> list[Path]:
    """Find OpenHands timeout run folders below ``root``.

    Discovery is intentionally artifact-based: a candidate must have a
    ``config.json`` and a usable ``trajectory/llm_trajectory.jsonl``. Non-timeout
    runs are skipped by ``load_run_folder(require_timeout=True)``.
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
    """Small JSON-serializable summary for CLI output and dashboards."""
    ok = [result for result in results if result.ok]
    failed = [result for result in results if not result.ok]
    return {
        "total": len(results),
        "succeeded": len(ok),
        "failed": len(failed),
        "outputs": [
            str(result.continued.rollout_dir)
            for result in ok
            if result.continued is not None
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
