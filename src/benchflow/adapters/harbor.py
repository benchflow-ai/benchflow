"""Inbound adapter for the Harbor task format.

Harbor (``harbor-framework/harbor``) is "terminal-bench thinking" — the
framework BenchFlow's own :class:`~benchflow.task.config.TaskConfig` was
internalized from. A Harbor task directory is therefore *already* in
BenchFlow-native shape:

::

    task_dir/
    ├── task.toml          # [task] [metadata] [verifier] [agent] [environment]
    ├── instruction.md
    ├── environment/       # Dockerfile + build context
    ├── solution/          # solve.sh — the oracle
    └── tests/             # test.sh — the verifier

This adapter is consequently a thin *normalizer*: it loads the foreign
``task.toml`` through the native :class:`TaskConfig` validator (which already
handles Harbor's ``[environment]``-keyed sandbox section and the
``version`` -> ``schema_version`` rename), reads ``instruction.md``, and
records the build/solution/test files under their native relative paths. No
field remapping is needed — Harbor *is* the native format, which is exactly
what makes Terminal-Bench backward-compatible through this edge.
"""

from __future__ import annotations

from pathlib import Path

from benchflow.adapters.inbound import (
    InboundTask,
    carry_native_subtrees,
    manifest_from_task_config,
)
from benchflow.environment.manifest import EnvironmentManifest
from benchflow.task.config import TaskConfig

# Foreign files carried straight through, keyed by their native location.
# Harbor's layout already matches BenchFlow's, so each key equals its source.
_PASSTHROUGH_FILES = (
    "environment/Dockerfile",
    "environment/docker-compose.yaml",
    "solution/solve.sh",
    "tests/test.sh",
)


class HarborAdapter:
    """Translate a Harbor task directory into an :class:`InboundTask`."""

    source = "harbor"

    @classmethod
    def from_task_dir(cls, task_dir: Path | str) -> InboundTask:
        """Translate a Harbor task directory into BenchFlow-native shape.

        Args:
            task_dir: Path to a Harbor task directory (contains ``task.toml``).

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
            raise FileNotFoundError(f"Harbor task is missing task.toml: {config_path}")

        instruction_path = root / "instruction.md"
        if not instruction_path.is_file():
            raise FileNotFoundError(
                f"Harbor task is missing instruction.md: {instruction_path}"
            )

        # TaskConfig was internalized from Harbor — the validator already
        # accepts the foreign task.toml verbatim.
        config = TaskConfig.model_validate_toml(config_path.read_text())
        instruction = instruction_path.read_text()

        name = config.task.name if config.task is not None else root.name

        # Environment-plane seam: prefer a sibling environment.toml (some
        # Harbor benchmarks ship one — chi-bench, clawsbench), otherwise
        # derive a baseline manifest from the legacy [environment] section.
        manifest = cls._load_manifest(root, name=name, config=config)

        # Harbor's directory layout is already the native one; the file map
        # is an identity mapping over whatever passthrough files exist.
        files: dict[str, Path] = {}
        for rel in _PASSTHROUGH_FILES:
            src = root / rel
            if src.is_file():
                files[rel] = src

        # Carry any extra files in the native subtrees (fixtures, helpers).
        # Harbor's layout is the native one, so a setdefault carry is safe —
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
        )

    @staticmethod
    def _load_manifest(
        root: Path, *, name: str, config: TaskConfig
    ) -> EnvironmentManifest:
        """Load the task's Environment-plane manifest.

        Harbor benchmarks that already follow the Environment-plane shape
        ship a sibling ``environment.toml`` (clawsbench, chi-bench); legacy
        single-container Harbor tasks do not. Either case yields a
        validated manifest so the inbound result is uniformly
        manifest-backed.
        """
        manifest_path = root / "environment.toml"
        if manifest_path.is_file():
            return EnvironmentManifest.model_validate_toml(manifest_path.read_text())
        return manifest_from_task_config(name=name, config=config)


def from_harbor_task(task_dir: Path | str) -> InboundTask:
    """Convenience function — translate a Harbor task directory."""
    return HarborAdapter.from_task_dir(task_dir)
