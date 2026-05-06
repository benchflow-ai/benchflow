"""ProgramBench -> BenchFlow task conversion.

Each upstream ProgramBench instance becomes one BenchFlow task directory:

    <output>/<instance_id>/
        task.toml
        instruction.md
        environment/Dockerfile
        tests/test.sh

The upstream instance directory is just metadata (`task.yaml` + `tests.json`);
the actual reference binary, docs, and per-branch test infrastructure live in
the cleanroom Docker image and the HuggingFace ProgramBench-Tests dataset.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from string import Template

import yaml

logger = logging.getLogger(__name__)

UPSTREAM_DOCKER_ORG = "programbench"
TEMPLATE_DIR = Path(__file__).parent / "templates"

# Difficulty -> agent timeout (seconds). Larger codebases need more time.
DIFFICULTY_TIMEOUT_SEC = {
    "easy": 1800,
    "medium": 3600,
    "hard": 7200,
    "unrated": 3600,
}

DIFFICULTY_RESOURCES = {
    "easy": {"cpus": 2, "memory_mb": 4096, "storage_mb": 20480},
    "medium": {"cpus": 4, "memory_mb": 8192, "storage_mb": 40960},
    "hard": {"cpus": 4, "memory_mb": 16384, "storage_mb": 81920},
    "unrated": {"cpus": 4, "memory_mb": 8192, "storage_mb": 40960},
}

VERIFIER_TIMEOUT_SEC = 3600


def cleanroom_image_name(instance_id: str) -> str:
    """Mirror ProgramBench's `image_name_from_instance_id` ('__' -> '_1776_')."""
    return f"{UPSTREAM_DOCKER_ORG}/{instance_id.replace('__', '_1776_')}"


def sanitize_task_name(instance_id: str) -> str:
    """Convert e.g. `abishekvashok__cmatrix.5c082c6` -> `programbench/abishekvashok-cmatrix-5c082c6`.

    BenchFlow's task registry requires lowercase, hyphenated names with stable
    output across runs.
    """
    safe = instance_id.lower().replace("__", "-").replace(".", "-").replace("_", "-")
    return f"programbench/{safe}"


def _read_template(name: str) -> str:
    return (TEMPLATE_DIR / name).read_text()


@dataclass
class UpstreamInstance:
    instance_id: str
    repository: str
    commit: str
    language: str
    difficulty: str
    tests_json: dict

    @classmethod
    def from_dir(cls, task_dir: Path) -> UpstreamInstance:
        config = yaml.safe_load((task_dir / "task.yaml").read_text())
        tests_path = task_dir / "tests.json"
        tests_json = json.loads(tests_path.read_text()) if tests_path.exists() else {"branches": {}}
        return cls(
            instance_id=task_dir.name,
            repository=config.get("repository", ""),
            commit=config.get("commit", ""),
            language=config.get("language", "unknown"),
            difficulty=config.get("difficulty", "unrated") or "unrated",
            tests_json=tests_json,
        )


def render_task(inst: UpstreamInstance, output_root: Path) -> Path:
    """Materialize one BenchFlow task directory for the upstream instance.

    Returns the path to the generated task directory.
    """
    task_dir = output_root / inst.instance_id
    env_dir = task_dir / "environment"
    tests_dir = task_dir / "tests"
    for d in (task_dir, env_dir, tests_dir):
        d.mkdir(parents=True, exist_ok=True)

    difficulty = inst.difficulty if inst.difficulty in DIFFICULTY_TIMEOUT_SEC else "unrated"
    resources = DIFFICULTY_RESOURCES[difficulty]

    task_toml_vars = {
        "task_name": sanitize_task_name(inst.instance_id),
        "instance_id": inst.instance_id,
        "repository": inst.repository,
        "commit": inst.commit,
        "language": inst.language,
        "difficulty": difficulty,
        "agent_timeout_sec": DIFFICULTY_TIMEOUT_SEC[difficulty],
        "verifier_timeout_sec": VERIFIER_TIMEOUT_SEC,
        "cpus": resources["cpus"],
        "memory_mb": resources["memory_mb"],
        "storage_mb": resources["storage_mb"],
    }
    (task_dir / "task.toml").write_text(
        Template(_read_template("task.toml.tmpl")).substitute(task_toml_vars)
    )

    instruction_vars = {
        "instance_id": inst.instance_id,
        "repository": inst.repository or inst.instance_id,
        "language": inst.language,
    }
    (task_dir / "instruction.md").write_text(
        Template(_read_template("instruction.md.tmpl")).substitute(instruction_vars)
    )

    dockerfile_vars = {
        "instance_id": inst.instance_id,
        "cleanroom_image": cleanroom_image_name(inst.instance_id),
    }
    (env_dir / "Dockerfile").write_text(
        Template(_read_template("Dockerfile.tmpl")).substitute(dockerfile_vars)
    )

    # test.sh is task-agnostic — copy it verbatim.
    shutil.copy(TEMPLATE_DIR / "test.sh", tests_dir / "test.sh")
    (tests_dir / "test.sh").chmod(0o755)
    # Sidecar tests.json — read by test.sh at /tests/tests.json once BenchFlow
    # mounts the per-task tests/ directory into the container.
    (tests_dir / "tests.json").write_text(json.dumps(inst.tests_json, sort_keys=True))

    return task_dir


def convert(
    upstream_tasks_dir: Path,
    output_dir: Path,
    *,
    task_ids: list[str] | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Convert ProgramBench tasks into BenchFlow task directories.

    Args:
        upstream_tasks_dir: Path to ProgramBench's `src/programbench/data/tasks/`
            (or a clone thereof) — one subdirectory per instance.
        output_dir: Where to emit BenchFlow task directories.
        task_ids: If given, only convert these instance IDs.
        limit: If given, convert only the first N matching instances (after
            applying ``task_ids``). Useful for smoke tests.
        overwrite: If True, regenerate task directories that already exist.
    """
    upstream_tasks_dir = Path(upstream_tasks_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(
        d for d in upstream_tasks_dir.iterdir()
        if d.is_dir() and (d / "task.yaml").exists()
    )
    if task_ids is not None:
        wanted = set(task_ids)
        candidates = [d for d in candidates if d.name in wanted]
    if limit is not None:
        candidates = candidates[:limit]

    generated: list[Path] = []
    for task_dir_in in candidates:
        out = output_dir / task_dir_in.name
        if out.exists() and not overwrite:
            logger.info("skipping existing %s (pass --overwrite to regenerate)", out)
            generated.append(out)
            continue
        if out.exists() and overwrite:
            shutil.rmtree(out)
        inst = UpstreamInstance.from_dir(task_dir_in)
        try:
            generated.append(render_task(inst, output_dir))
            logger.info("generated %s", out)
        except Exception as e:
            logger.error("failed to convert %s: %s", task_dir_in.name, e)
            raise
    return generated
