"""The decoupled roster — `--agents roster.yaml`, the file form of repeated
`--agent/--model`.

A roster is ONLY the agents list (the A/M axis). The task (`--tasks-dir`), the
shared service (`--environment-manifest`), the sandbox (`--sandbox`), and the run
knobs (`--out`, `--drive`, `--prompt`) all follow the standard single-agent
`bench eval run` flags — they are NOT in this file. This keeps the roster reusable
across any task. Per-seat fields live on each entry (`AgentSpec`), and `count`
fans an entry into `name-0..name-(n-1)` seats.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from benchflow.arena.agents_manifest import AgentSpec, Seat, resolve_spec

__all__ = ["Roster"]

# Keys that USED to live in the old agents.yaml but are now standard
# `bench eval run` flags — reject them with a migration hint, not a bare
# "extra inputs not permitted".
_RUN_LEVEL_KEYS = frozenset({
    "task", "task_path", "tasks_dir", "environment_manifest", "services",
    "sandbox", "out", "drive", "prompt", "deadline_s", "idle_timeout_s",
})


class Roster(BaseModel):
    """A pure list of agent seats (the decoupled `--agents` file)."""

    model_config = ConfigDict(extra="forbid")

    agents: list[AgentSpec] = Field(min_length=1)

    _base_dir: Path = PrivateAttr(default_factory=lambda: Path("."))

    @model_validator(mode="before")
    @classmethod
    def _reject_run_level_keys(cls, data: object) -> object:
        if isinstance(data, dict):
            bad = sorted(_RUN_LEVEL_KEYS & set(data))
            if bad:
                raise ValueError(
                    f"{bad} are run-level config, not roster fields: the roster is "
                    "the `--agents` file (agents only). Pass these as standard "
                    "`bench eval run` flags (--tasks-dir, --environment-manifest, "
                    "--sandbox, --out, --drive, --prompt)."
                )
        return data

    @classmethod
    def from_yaml(cls, path: str | Path) -> Roster:
        path = Path(path)
        roster = cls.model_validate(yaml.safe_load(path.read_text()) or {})
        roster._base_dir = path.resolve().parent
        return roster

    def instructions_path(self, spec: AgentSpec) -> Path | None:
        """Resolve a seat's instruction file relative to the roster's directory."""
        if not spec.instructions:
            return None
        return self._base_dir / spec.instructions

    def seats(self) -> list[Seat]:
        """Resolve + fan out every entry into runnable seats (ids must be unique)."""
        out: list[Seat] = []
        seen: set[str] = set()
        for spec in self.agents:
            cfg = resolve_spec(spec, self._base_dir)
            ids = (
                [spec.name]
                if spec.count == 1
                else [f"{spec.name}-{i}" for i in range(spec.count)]
            )
            for sid in ids:
                if sid in seen:
                    raise ValueError(
                        f"duplicate seat id {sid!r}: agent names (after count "
                        "fan-out) must be unique"
                    )
                seen.add(sid)
                out.append(Seat(seat_id=sid, spec=spec, config=cfg))
        return out
