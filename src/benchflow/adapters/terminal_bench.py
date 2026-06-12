"""Inbound adapter for the Terminal-Bench task format.

Terminal-Bench (``laude-institute/terminal-bench``) predates Harbor; Harbor
is itself terminal-bench-derived. A Terminal-Bench task is therefore *almost*
the native format, but with three real differences this adapter resolves:

1. **The instruction is embedded in the config.** ``task.yaml`` carries an
   ``instruction:`` key; BenchFlow keeps the instruction in a separate
   ``instruction.md``. The adapter lifts it out.
2. **Flat metadata vs. structured config.** ``task.yaml`` is a flat YAML
   document — ``author_name`` / ``author_email`` instead of an
   ``[task].authors`` list, ``max_agent_timeout_sec`` /
   ``max_test_timeout_sec`` instead of ``[agent]`` / ``[verifier]`` tables,
   ``difficulty`` / ``category`` / ``tags`` / ``parser_name`` at top level.
3. **The Dockerfile sits at the task root.** Terminal-Bench keeps
   ``Dockerfile`` / ``docker-compose.yaml`` / ``solution.sh`` /
   ``run-tests.sh`` at the task root; BenchFlow expects them under
   ``environment/``, ``solution/``, and ``tests/test.sh``.

This is the work of a real inbound adapter — a pure ``task.yaml`` -> native
:class:`~benchflow.task.config.TaskConfig` translation, with the file map
remapping the foreign layout onto BenchFlow's. Old terminal tasks keep
running (architecture.md, capability #8).
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

import yaml

from benchflow.adapters.inbound import (
    InboundTask,
    carry_native_subtrees,
    manifest_from_task_config,
)
from benchflow.task.config import TaskConfig

# Terminal-Bench top-level keys that become BenchFlow [metadata]. Everything
# carried so the parser_name / difficulty signal survives the translation.
_METADATA_KEYS = (
    "difficulty",
    "category",
    "tags",
    "parser_name",
    "run_tests_in_same_shell",
    "estimated_duration_sec",
    "expert_time_estimate_min",
    "junior_time_estimate_min",
)

# Foreign-root file -> BenchFlow-native relative destination.
_FILE_REMAP = {
    "Dockerfile": "environment/Dockerfile",
    "docker-compose.yaml": "environment/docker-compose.yaml",
    "docker-compose.yml": "environment/docker-compose.yml",
    "solution.sh": "solution/solve.sh",
    "run-tests.sh": "tests/test.sh",
}


class TerminalBenchAdapter:
    """Translate a Terminal-Bench task directory into an :class:`InboundTask`."""

    source = "terminal-bench"

    @classmethod
    def from_task_dir(cls, task_dir: Path | str) -> InboundTask:
        """Translate a Terminal-Bench task directory into BenchFlow-native shape.

        Args:
            task_dir: Path to a Terminal-Bench task directory (contains
                ``task.yaml``).

        Returns:
            An :class:`InboundTask`: instruction lifted out of the YAML, a
            validated native :class:`TaskConfig`, and a ``files`` map that
            remaps the foreign layout (root-level Dockerfile, ``run-tests.sh``)
            onto BenchFlow's directory structure.

        Raises:
            FileNotFoundError: if no ``task.yaml`` / ``task.yml`` is present.
            ValueError: if the YAML carries no ``instruction`` key.
        """
        root = Path(task_dir)

        config_path = cls._locate_config(root)
        raw = yaml.safe_load(config_path.read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError(
                f"Terminal-Bench task.yaml must be a mapping: {config_path}"
            )

        instruction = raw.get("instruction")
        if not instruction:
            raise ValueError(
                f"Terminal-Bench task.yaml has no 'instruction' key: {config_path}"
            )

        # InboundTask.name is the bare task id (the dir name); the native
        # PackageInfo, however, requires the namespaced org/name form.
        namespaced = f"terminal-bench/{root.name}"
        config = cls._build_config(namespaced, raw)
        files = cls._build_file_map(root)

        # Environment-plane seam. Terminal-Bench ships a root Dockerfile
        # and no manifest, so the synthesized manifest names the same
        # image tag the framework's build pipeline will produce.
        manifest = manifest_from_task_config(name=namespaced, config=config)

        return InboundTask(
            name=root.name,
            source=cls.source,
            instruction=str(instruction),
            manifest=manifest,
            config=config,
            files=files,
        )

    @staticmethod
    def _locate_config(root: Path) -> Path:
        for candidate in ("task.yaml", "task.yml"):
            path = root / candidate
            if path.is_file():
                return path
        raise FileNotFoundError(
            f"Terminal-Bench task is missing task.yaml: {root / 'task.yaml'}"
        )

    @staticmethod
    def _build_config(name: str, raw: dict[str, Any]) -> TaskConfig:
        """Map the flat ``task.yaml`` document onto a native ``TaskConfig``."""
        authors: list[dict[str, str]] = []
        author_name = raw.get("author_name")
        if author_name and str(author_name).lower() != "unknown":
            author: dict[str, str] = {"name": str(author_name)}
            author_email = raw.get("author_email")
            if author_email and str(author_email).lower() != "unknown":
                author["email"] = str(author_email)
            authors.append(author)

        metadata = {
            key: raw[key]
            for key in _METADATA_KEYS
            if key in raw and raw[key] is not None
        }

        task_section: dict[str, Any] = {"name": name, "authors": authors}
        tags = raw.get("tags")
        if isinstance(tags, list):
            task_section["keywords"] = [str(t) for t in tags]

        verifier: dict[str, Any] = {}
        if raw.get("max_test_timeout_sec") is not None:
            verifier["timeout_sec"] = float(raw["max_test_timeout_sec"])

        agent: dict[str, Any] = {}
        if raw.get("max_agent_timeout_sec") is not None:
            agent["timeout_sec"] = float(raw["max_agent_timeout_sec"])

        return TaskConfig.model_validate(
            {
                "schema_version": "1.1",
                "task": task_section,
                "metadata": metadata,
                "verifier": verifier,
                "agent": agent,
                "source": "terminal-bench",
            }
        )

    @staticmethod
    def _build_file_map(root: Path) -> dict[str, Path]:
        """Remap the Terminal-Bench layout onto BenchFlow-native paths.

        Raises:
            ValueError: if two source files claim the same native
                destination — e.g. a root ``run-tests.sh`` (remapped to
                ``tests/test.sh``) and a native-shaped ``tests/test.sh``.
                Silently dropping one would be order-dependent data loss.
        """
        files: dict[str, Path] = {}

        def _place(native: str, src: Path) -> None:
            existing = files.get(native)
            if existing is not None and existing != src:
                raise ValueError(
                    f"Terminal-Bench task file-map collision on "
                    f"{native!r}: both {existing} and {src} map to the "
                    f"same BenchFlow-native destination. Keep only one "
                    f"(e.g. drop the root run-tests.sh or the native "
                    f"tests/test.sh)."
                )
            files[native] = src

        # Root-level files that move into a native subtree.
        for foreign, native in _FILE_REMAP.items():
            src = root / foreign
            if src.is_file():
                _place(native, src)

        # The Dockerfile's build context. Terminal-Bench Dockerfiles routinely
        # COPY/ADD task-root files (requirements.txt, app/, etc.) into the
        # image; the legacy adapter only carried the Dockerfile itself, so the
        # converted task built against an empty context and failed at
        # ``docker build`` time. Walk the Dockerfile's COPY/ADD instructions
        # and place every referenced source into the environment/ subtree
        # under its same relative path (so ``COPY requirements.txt /tmp``
        # resolves against environment/requirements.txt, which is where the
        # remapped Dockerfile lives). See issue #362.
        dockerfile = root / "Dockerfile"
        if dockerfile.is_file():
            for src_rel in _dockerfile_context_sources(dockerfile):
                src = root / src_rel
                # Tolerate sources that don't exist on disk (URL ADDs, build
                # args expanded at build time, stage-name --from=builder
                # references already filtered upstream) — the adapter is a
                # pure translator, not a Dockerfile validator.
                if not src.exists():
                    continue
                if src.is_file():
                    _place(f"environment/{src_rel}", src)
                elif src.is_dir():
                    for entry in sorted(p for p in src.rglob("*") if p.is_file()):
                        _place(
                            f"environment/{entry.relative_to(root).as_posix()}",
                            entry,
                        )

        # The tests/ subtree is already native — carry it verbatim, with the
        # same collision check so a native tests/test.sh can't silently
        # collide with a remapped root run-tests.sh.
        carry_native_subtrees(root, _place, subtrees=("tests",))

        return files


# A Dockerfile instruction line — captures the verb and the rest. We only
# care about COPY / ADD; comments and blank lines are filtered separately.
_DOCKERFILE_INSTRUCTION = re.compile(r"^\s*(\w+)\s+(.*)$", re.IGNORECASE)
# Stage references like ``--from=builder`` carry no real source path on the
# build context; same for HTTP(S) URLs in ADD instructions.
_URL_SOURCE = re.compile(r"^https?://", re.IGNORECASE)


def _dockerfile_context_sources(dockerfile: Path) -> list[str]:
    """Extract build-context source paths referenced by COPY/ADD instructions.

    Returns a list of *task-root-relative* paths the Dockerfile expects to
    find in its build context. Designed for the Terminal-Bench case, where
    the Dockerfile sits at the task root and its build context is the task
    root itself.

    Implementation notes:

    * Line continuations (``\\``) are joined before parsing so a multi-line
      ``COPY`` carries all of its sources.
    * Comments (``# ...``), blank lines, and ``--flag=value`` arguments are
      skipped. ``--from=stage`` sources are skipped (they reference an
      earlier build stage, not the build context).
    * Shell-form (``COPY src dst``) and exec-form (``COPY ["src", "dst"]``)
      are both handled.
    * URL sources (``ADD https://...``) are skipped — they aren't part of
      the build context.
    * Sources with ``$VAR`` interpolation are skipped — we can't resolve
      build args without executing the build.
    * Absolute paths and ``..`` escapes are skipped — they don't address a
      file inside the task directory.
    """
    sources: list[str] = []
    try:
        text = dockerfile.read_text(errors="replace")
    except OSError:
        return sources

    # Join line continuations: a trailing ``\`` (optionally followed by
    # whitespace) folds the next line into the current instruction.
    joined = re.sub(r"\\\s*\n", " ", text)

    for raw_line in joined.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = _DOCKERFILE_INSTRUCTION.match(line)
        if not match:
            continue
        verb = match.group(1).upper()
        if verb not in ("COPY", "ADD"):
            continue

        rest = match.group(2).strip()
        # Exec form: ["src1", "src2", "dst"] — parse as JSON-ish list.
        if rest.startswith("["):
            try:
                import json as _json

                parts = _json.loads(rest)
                if not isinstance(parts, list):
                    continue
                tokens = [str(p) for p in parts]
            except (ValueError, TypeError):
                continue
        else:
            try:
                tokens = shlex.split(rest, posix=True)
            except ValueError:
                continue

        # Drop --flag=value arguments (--from=, --chown=, --chmod=, ...).
        # A --from=<stage> source is *not* in the build context — skip the
        # whole instruction in that case.
        positional: list[str] = []
        skip_instruction = False
        for tok in tokens:
            if tok.startswith("--"):
                if tok.startswith("--from="):
                    skip_instruction = True
                continue
            positional.append(tok)
        if skip_instruction:
            continue
        # Need at least one source and one destination.
        if len(positional) < 2:
            continue

        for src in positional[:-1]:
            if _URL_SOURCE.match(src):
                continue
            if "$" in src or "{" in src:
                # Build-arg / env interpolation — can't resolve statically.
                continue
            # Normalize and reject anything that doesn't address a file
            # inside the build context (absolute paths, parent escapes).
            normalized = src.lstrip("./")
            if not normalized:
                continue
            if src.startswith("/") or normalized.startswith("../"):
                continue
            if any(part == ".." for part in normalized.split("/")):
                continue
            sources.append(normalized)

    return sources


def from_terminal_bench_task(task_dir: Path | str) -> InboundTask:
    """Convenience function — translate a Terminal-Bench task directory."""
    return TerminalBenchAdapter.from_task_dir(task_dir)
