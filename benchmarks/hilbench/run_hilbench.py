"""Run HILBench — downloads dataset from HuggingFace, generates tasks, runs via Evaluation.

Pre-requisites:
    * Docker daemon running.

The runner downloads per-task Docker image tarballs from HuggingFace,
loads them with ``docker load``, and passes the resulting image tag as
the ``BASE_IMAGE`` build arg when building each task's Dockerfile.

Usage:
    python benchmarks/hilbench/run_hilbench.py
    python benchmarks/hilbench/run_hilbench.py path/to/config.yaml
"""

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SCRIPT_DIR) in sys.path:
    sys.path.remove(str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_CONVERTER = _SCRIPT_DIR / "benchflow.py"
_DOCKER_LOADED_RE = re.compile(r"Loaded image:\s*(\S+)")
_DOCKER_LOADED_ID_RE = re.compile(r"Loaded image ID:\s*(\S+)")
_HF_BUCKET_RE = re.compile(r"hf://buckets/([^/]+/[^/]+)/(.+)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HILBench via BenchFlow.")
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="BenchFlow evaluation YAML config. Omit to only prepare tasks/images.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Generate tasks and load/tag Docker images, then exit.",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=None,
        help="Use an existing/generated HILBench task directory.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for downloaded HILBench Docker image tarballs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit generated tasks when --tasks-dir is not provided.",
    )
    return parser.parse_args()


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def _download_hf_bucket_file(download_link: str, cache_dir: Path) -> Path:
    """Download an ``hf://buckets/...`` object through HuggingFace bucket URLs."""
    match = _HF_BUCKET_RE.fullmatch(download_link)
    if not match:
        raise ValueError(f"Cannot parse bucket download_link: {download_link}")

    bucket_id = match.group(1)  # e.g. ScaleAI/hil-bench-swe-images
    object_path = match.group(2)  # e.g. images/69bc1094b455a91fa20fb868.tar.zst
    url = (
        f"https://huggingface.co/buckets/{bucket_id}/resolve/"
        f"{quote(object_path, safe='/')}"
    )
    local_path = cache_dir / bucket_id.replace("/", "--") / object_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() and local_path.stat().st_size > 0:
        logger.info("Using cached image tarball %s", local_path)
        return local_path

    headers = {"User-Agent": "benchflow-hilbench-adapter"}
    token = _hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    logger.info("Downloading %s to %s", url, local_path)
    request = Request(url, headers=headers)
    with urlopen(request, timeout=60) as response, local_path.open("wb") as f:
        shutil.copyfileobj(response, f)
    return local_path


def _download_and_load_image(download_link: str, cache_dir: Path) -> str:
    """Download a HILBench Docker image tarball and load it.

    Returns the Docker image tag (or image ID) produced by ``docker load``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = _download_hf_bucket_file(download_link, cache_dir)
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


def ensure_converted_tasks(*, limit: int | None = None) -> Path:
    """Download HILBench dataset from HuggingFace and convert to BenchFlow format."""
    converted_dir = _REPO_ROOT / ".cache" / "hilbench-benchflow"

    if converted_dir.exists() and any(converted_dir.glob("*/task.toml")):
        logger.info("Converted tasks already exist at %s", converted_dir)
        return converted_dir

    logger.info("Converting HILBench SWE tasks to BenchFlow format...")
    cmd = [sys.executable, str(_CONVERTER), "--output-dir", str(converted_dir)]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Conversion failed: %s", result.stderr)
        raise RuntimeError(f"HILBench conversion failed: {result.stderr}")

    logger.info("Converted tasks to %s", converted_dir)
    return converted_dir


async def main():
    args = _parse_args()
    tasks_dir = args.tasks_dir or ensure_converted_tasks(limit=args.limit)
    logger.info("Using tasks from %s", tasks_dir)

    # Download and load Docker images for all tasks
    cache_dir = args.cache_dir or (_REPO_ROOT / ".cache" / "hilbench-images")
    image_map = load_and_tag_images(tasks_dir, cache_dir)
    logger.info("Loaded %d Docker images", len(image_map))

    if not image_map:
        logger.warning(
            "No Docker images loaded. Ensure the HuggingFace bucket URLs in "
            "task_metadata.json are reachable."
        )

    if args.prepare_only:
        print(f"Prepared {len(image_map)} HILBench images for {tasks_dir}")
        return

    if not args.config:
        logger.info("No config specified; tasks generated at %s", tasks_dir)
        logger.info("Use: bench eval create -f <config.yaml> to run evaluations")
        return

    from benchflow.evaluation import Evaluation

    job = Evaluation.from_yaml(args.config)
    job._tasks_dir = tasks_dir  # type: ignore[attr-defined]
    result = await job.run()
    print(f"\nScore: {result.passed}/{result.total} ({result.score:.1%})")
    print(f"Image map: {image_map}")


if __name__ == "__main__":
    asyncio.run(main())
