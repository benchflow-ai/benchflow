"""Inbound adapter for ``task.toml`` task folders.

A ``task.toml`` task folder has the same layout as a BenchFlow-native
task folder:

::

    task_dir/
    ├── task.toml          # [task] [metadata] [verifier] [agent] [environment]
    ├── instruction.md
    ├── environment/       # Dockerfile + build context
    ├── solution/          # solve.sh — the oracle
    └── tests/             # test.sh — the verifier

This adapter is consequently a thin *normalizer*: it loads the foreign
``task.toml`` through :func:`~benchflow.task.imports.import_task_config_toml`,
whose toml import boundary translates the ``[environment]``-keyed
sandbox section to the native ``sandbox`` key and handles the ``version`` ->
``schema_version`` rename before native :class:`TaskConfig` validation. It
then reads ``instruction.md`` and records the build/solution/test files under
their native relative paths. Beyond that spelling translation no field
remapping is needed: the layout is the native one, which is what lets
Terminal-Bench tasks run through this edge.
"""

from __future__ import annotations

from pathlib import Path

from benchflow.adapters.inbound import (
    InboundCompatibility,
    InboundTask,
    carry_native_subtrees,
    manifest_from_task_config,
)
from benchflow.environment.manifest import EnvironmentManifest
from benchflow.task.config import TaskConfig
from benchflow.task.imports import import_task_config_toml

# Foreign files carried straight through, keyed by their native location.
# The task.toml layout already matches BenchFlow's, so each key equals its source.
_PASSTHROUGH_FILES = (
    "environment/Dockerfile",
    "environment/docker-compose.yaml",
    "solution/solve.sh",
    "tests/test.sh",
)


class TaskTomlAdapter:
    """Translate a ``task.toml`` task directory into an :class:`InboundTask`."""

    source = "task-toml"

    @classmethod
    def from_task_dir(cls, task_dir: Path | str) -> InboundTask:
        """Translate a ``task.toml`` task directory into BenchFlow-native shape.

        Args:
            task_dir: Path to a task directory that contains ``task.toml``.

        Returns:
            An :class:`InboundTask` whose ``config`` is the validated native
            :class:`TaskConfig` and whose ``files`` map carries the build,
            solution, and verifier files.

        Raises:
            FileNotFoundError: if ``task.toml`` or ``instruction.md`` is absent.
        """
        root = Path(task_dir)

        config_path = root / "task.toml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Task is missing task.toml: {config_path}")

        instruction_path = root / "instruction.md"
        if not instruction_path.is_file():
            raise FileNotFoundError(
                f"Task is missing instruction.md: {instruction_path}"
            )

        # The native schema stays strict. Some task.toml producers add extension
        # keys; preserve those in adapter compatibility metadata instead of
        # pretending BenchFlow can execute them.
        imported = import_task_config_toml(config_path.read_text(), source=cls.source)
        config = imported.config
        instruction = instruction_path.read_text()

        name = config.task.name if config.task is not None else root.name

        # Environment-plane seam: prefer a sibling environment.toml (some
        # benchmarks ship one — chi-bench, clawsbench), otherwise
        # derive a baseline manifest from the legacy [environment] section.
        manifest = cls._load_manifest(root, name=name, config=config)

        # The directory layout is already the native one; the file map
        # is an identity mapping over whatever passthrough files exist.
        files: dict[str, Path] = {}
        for rel in _PASSTHROUGH_FILES:
            src = root / rel
            if src.is_file():
                files[rel] = src

        # Carry any extra files in the native subtrees (fixtures, helpers).
        # The layout is the native one, so a setdefault carry is safe —
        # a passthrough file already placed above wins over its subtree copy.
        def _carry(native: str, src: Path) -> None:
            files.setdefault(native, src)

        carry_native_subtrees(root, _carry)

        return InboundTask(
            name=name,
            source=cls.source,
            instruction=instruction,
            manifest=manifest,
            config=config,
            files=files,
            compatibility=InboundCompatibility(
                source=cls.source,
                config_extra=imported.report.extra,
                config_extra_paths=imported.report.extra_paths,
            ),
        )

    @staticmethod
    def _load_manifest(
        root: Path, *, name: str, config: TaskConfig
    ) -> EnvironmentManifest:
        """Load the task's Environment-plane manifest.

        Benchmarks that already follow the Environment-plane shape
        ship a sibling ``environment.toml`` (clawsbench, chi-bench); legacy
        single-container task.toml tasks do not. Either case yields a
        validated manifest so the inbound result is uniformly
        manifest-backed.
        """
        manifest_path = root / "environment.toml"
        if manifest_path.is_file():
            return EnvironmentManifest.model_validate_toml(manifest_path.read_text())
        return manifest_from_task_config(name=name, config=config)


def from_task_toml_dir(task_dir: Path | str) -> InboundTask:
    """Convenience function — translate a ``task.toml`` task directory."""
    return TaskTomlAdapter.from_task_dir(task_dir)
