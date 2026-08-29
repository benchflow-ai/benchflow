from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_REDACTOR_SOURCE = "src/benchflow/trajectories/redaction.py"
_REDACTOR_RESOURCE = "benchflow/trajectories/resources/canonical_redaction.py.txt"


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, _version: str, build_data: dict[str, Any]) -> None:
        if self.target_name != "wheel":
            return
        source = Path(self.root, _REDACTOR_SOURCE)
        with tempfile.NamedTemporaryFile(
            prefix="benchflow-redactor-",
            suffix=".txt",
            delete=False,
        ) as generated:
            generated.write(source.read_bytes())
            self._generated_path = Path(generated.name)
        build_data["force_include"][str(self._generated_path)] = _REDACTOR_RESOURCE

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
        generated = getattr(self, "_generated_path", None)
        if generated is not None:
            generated.unlink(missing_ok=True)
            self._generated_path = None
