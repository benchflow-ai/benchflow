"""Run HILBench — downloads dataset from HuggingFace, generates tasks, runs via Job.

Pre-requisites:
    * ``huggingface_hub`` installed (``pip install huggingface_hub``)
    * ``HF_TOKEN`` env-var set to a token with access to the gated
      ``ScaleAI/hil-bench-swe-images`` bucket on HuggingFace.
    * Docker daemon running.

The runner downloads per-task Docker image tarballs from HuggingFace,
loads them with ``docker load``, and passes the resulting image tag as
the ``BASE_IMAGE`` build arg when building each task's Dockerfile.

Usage:
    python benchmarks/hilbench/run_hilbench.py
    python benchmarks/hilbench/run_hilbench.py path/to/config.yaml
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

from benchflow.job import Job

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_CONVERTER = _SCRIPT_DIR / "benchflow.py"
_DOCKER_LOADED_RE = re.compile(r"Loaded image:\s*(\S+)")
_DOCKER_LOADED_ID_RE = re.compile(r"Loaded image ID:\s*(\S+)")


def _repo_root() -> Path:
    d = Path.cwd()
    while d != d.parent:
        if (d / ".git").exists():
            return d
        d = d.parent
    return Path.cwd()


def _download_and_load_image(download_link: str, cache_dir: Path) -> str:
    """Download a HILBench Docker image tarball and load it.

    Returns the Docker image tag (or image ID) produced by ``docker load``.
    """
    from huggingface_hub import hf_hub_download

    # download_link format: hf://buckets/ScaleAI/hil-bench-swe-images/images/<uid>.tar.zst
    # Extract repo_id and filename from the link.
    match = re.match(r"hf://buckets/([^/]+/[^/]+)/(.+)", download_link)
    if not match:
        raise ValueError(f"Cannot parse download_link: {download_link}")

    repo_id = match.group(1)  # e.g. ScaleAI/hil-bench-swe-images
    filename = match.group(2)  # e.g. images/69bc1094b455a91fa20fb868.tar.zst

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=str(cache_dir),
        token=os.environ.get("HF_TOKEN"),
    )
    logger.info("Downloaded image tarball to %s", local_path)

    # Load the image into Docker
    result = subprocess.run(
        ["docker", "load", "-i", local_path],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker load failed: {result.stderr}")

    output = result.stdout
    m = _DOCKER_LOADED_RE.search(output)
    if m:
        return m.group(1)
    m = _DOCKER_LOADED_ID_RE.search(output)
    if m:
        return m.group(1)
    raise RuntimeError(f"Could not parse image tag from docker load output: {output}")


def load_and_tag_images(tasks_dir: Path, cache_dir: Path) -> dict[str, str]:
    """Load Docker images for all tasks and tag them for Dockerfile use.

    Each loaded image is re-tagged as ``hilbench-base:<task_dir_name>`` so
    that the generated Dockerfile's ``FROM hilbench-base:<task_dir_name>``
    resolves correctly during ``docker build``.

    Returns {task_dir_name: original_image_tag}.
    """
    image_map: dict[str, str] = {}
    for task_dir in sorted(tasks_dir.iterdir()):
        meta_file = task_dir / "tests" / "task_metadata.json"
        if not meta_file.exists():
            continue
        meta = json.loads(meta_file.read_text())
        download_link = meta.get("download_link", "")
        if not download_link:
            logger.warning(
                "No download_link for %s, skipping image load", task_dir.name
            )
            continue

        try:
            tag = _download_and_load_image(download_link, cache_dir)
            image_map[task_dir.name] = tag
            logger.info("Loaded image for %s -> %s", task_dir.name, tag)

            # Tag to the predictable name the Dockerfile expects
            predictable_tag = f"hilbench-base:{task_dir.name}"
            result = subprocess.run(
                ["docker", "tag", tag, predictable_tag],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.error(
                    "docker tag %s -> %s failed: %s",
                    tag,
                    predictable_tag,
                    result.stderr,
                )
            else:
                logger.info("Tagged %s -> %s", tag, predictable_tag)
        except Exception:
            logger.exception("Failed to load image for %s", task_dir.name)
    return image_map


def ensure_converted_tasks() -> Path:
    """Download HILBench dataset from HuggingFace and convert to BenchFlow format."""
    root = _repo_root()
    converted_dir = root / ".cache" / "hilbench-benchflow"

    if converted_dir.exists() and any(converted_dir.iterdir()):
        logger.info("Converted tasks already exist at %s", converted_dir)
        return converted_dir

    logger.info("Converting HILBench SWE tasks to BenchFlow format...")
    result = subprocess.run(
        [
            sys.executable,
            str(_CONVERTER),
            "--output-dir",
            str(converted_dir),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("Conversion failed: %s", result.stderr)
        raise RuntimeError(f"HILBench conversion failed: {result.stderr}")

    logger.info("Converted tasks to %s", converted_dir)
    return converted_dir


async def main():
    config = sys.argv[1] if len(sys.argv) > 1 else None
    tasks_dir = ensure_converted_tasks()
    logger.info("Using tasks from %s", tasks_dir)

    # Download and load Docker images for all tasks
    root = _repo_root()
    cache_dir = root / ".cache" / "hilbench-images"
    image_map = load_and_tag_images(tasks_dir, cache_dir)
    logger.info("Loaded %d Docker images", len(image_map))

    if not image_map:
        logger.warning(
            "No Docker images loaded. Set HF_TOKEN and ensure access to "
            "ScaleAI/hil-bench-swe-images on HuggingFace."
        )

    if config:
        job = Job.from_yaml(config)
    else:
        logger.info("No config specified; tasks generated at %s", tasks_dir)
        logger.info("Use: bench eval create -f <config.yaml> to run evaluations")
        return

    job._tasks_dir = tasks_dir  # type: ignore[attr-defined]
    result = await job.run()
    print(f"\nScore: {result.passed}/{result.total} ({result.score:.1%})")
    print(f"Image map: {image_map}")


if __name__ == "__main__":
    asyncio.run(main())
