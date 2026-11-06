"""Pull a Docker job's task images when the job starts, several at once.

Without it each trial's ``compose up`` pulls its own image the first time,
inside the trial's concurrency slot, so a job on a fresh daemon pays every
pull on some trial's critical path, one image per trial start. The job
instead starts pulling every distinct prebuilt task image (``docker_image``)
its trials will use, in queue order, a few at a time, while its first
trials run. A trial whose image is still being pulled when it starts joins
that pull (the daemon shares in-flight layer downloads).

Each pull is the one a trial makes: only an image Docker says it does not
have is pulled, and it is recorded as BenchFlow's pull for ``bench cache
clean --only pulled-images`` exactly as a trial's own pull is
(:meth:`DockerLayers.pull_if_absent`). A failed pull is only logged: the
trial's own start then pulls, and fails, as before.
``BENCHFLOW_PREPULL=off`` turns it off.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

PREPULL_ENV = "BENCHFLOW_PREPULL"
#: Pulls at once: enough to overlap registry round trips, few enough to
#: leave the job's own first starts their bandwidth.
PREPULL_CONCURRENCY = 4


def prepull_requested() -> bool:
    """Whether ``BENCHFLOW_PREPULL`` leaves pre-pulling on (the default)."""
    value = os.environ.get(PREPULL_ENV, "").strip().lower()
    return value not in {"0", "off", "false", "no", "none"}


async def prepull_images(
    refs: Iterable[str], *, concurrency: int = PREPULL_CONCURRENCY
) -> dict[str, str]:
    """Pull each distinct ``ref`` the daemon lacks, ``concurrency`` at a time.

    Returns each reference's outcome: ``present``, ``pulled``, ``unknown``
    (the daemon could not say) or ``failed: <reason>``. Never raises for a
    pull; cancellation stops the pulls still running.
    """
    from benchflow.sandbox._docker_subnets import daemon_key
    from benchflow.sandbox.env_layers import DockerLayers

    ordered = list(dict.fromkeys(ref for ref in refs if ref))
    if not ordered:
        return {}
    layers = DockerLayers(daemon=daemon_key(None))
    gate = asyncio.Semaphore(max(1, concurrency))
    outcomes: dict[str, str] = {}

    async def pull(ref: str) -> None:
        async with gate:
            try:
                outcomes[ref] = await layers.pull_if_absent(ref)
            except Exception as exc:
                outcomes[ref] = f"failed: {exc}"
                logger.warning(
                    "Pre-pull of %s failed; its trial pulls it: %s", ref, exc
                )

    await asyncio.gather(*(pull(ref) for ref in ordered))
    pulled = sum(1 for outcome in outcomes.values() if outcome == "pulled")
    if pulled:
        logger.info("Pre-pulled %d of %d task image(s)", pulled, len(ordered))
    return outcomes


def task_sets_platform(task_dir: Path) -> bool:
    """Whether a service of the task's compose file pins a ``platform``.

    Then the trial's start pulls that platform's variant; a pre-pull (the
    host's default variant) would add a second, unused image. An unreadable
    compose file counts as pinned: the trial pulls as before.
    """
    env_dir = task_dir / "environment"
    for name in ("docker-compose.yaml", "docker-compose.yml"):
        path = env_dir / name
        if not path.exists():
            continue
        try:
            import yaml

            document = yaml.safe_load(path.read_text())
        except Exception:
            return True
        services = document.get("services") if isinstance(document, dict) else None
        if services is None:
            continue
        if not isinstance(services, dict):
            return True
        if any(not isinstance(s, dict) or s.get("platform") for s in services.values()):
            return True
    return False
