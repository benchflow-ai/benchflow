from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_RUNTIME_RESOURCES = (
    (
        "src/benchflow/trajectories/redaction.py",
        "benchflow/trajectories/resources/canonical_redaction.py.txt",
    ),
    (
        "src/benchflow/continue_run/sandbox_replay_runtime.py",
        "benchflow/continue_run/resources/sandbox_replay_runtime.py.txt",
    ),
)


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, _version: str, build_data: dict[str, Any]) -> None:
        if self.target_name != "wheel":
            return
        self._generated_paths: list[Path] = []
        for source_name, resource_name in _RUNTIME_RESOURCES:
            source = Path(self.root, source_name)
            with tempfile.NamedTemporaryFile(
                prefix="benchflow-runtime-resource-",
                suffix=".txt",
                delete=False,
            ) as generated:
                generated.write(source.read_bytes())
                generated_path = Path(generated.name)
            self._generated_paths.append(generated_path)
            build_data["force_include"][str(generated_path)] = resource_name

    def finalize(
        self,
        _version: str,
        _build_data: dict[str, Any],
        _artifact_path: str,
    ) -> None:
        self._discard_generated()

    def clean(self, _versions: list[str]) -> None:
        self._discard_generated()

    def _discard_generated(self) -> None:
        for generated in getattr(self, "_generated_paths", []):
            generated.unlink(missing_ok=True)
        self._generated_paths = []
