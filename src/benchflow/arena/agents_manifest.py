"""Parse `--agents agents.yaml` → resolved concurrent-floor seats.

ONE shared task + its ONE service, N agents that run it concurrently. Each
``agents[]`` entry names a PREBUILT benchflow agent (``agent:``) or a BYOA agent
manifest (``manifest:``) — exactly one — plus its model and an optional per-agent
instruction file. ``count`` fans an entry into ``name-0..name-(n-1)`` seats. Every
entry resolves to one :class:`AgentConfig`, so the runner never branches on "which
agent path" (raw ACP / ai-sdk / omnigent all collapse here).

BYOA manifests follow the data-only agent contract (``contract/manifest_schema.json``
in benchflow-ai/agents): ACP-only, strict / no unknown fields. We validate with a
pydantic mirror of that schema (no new dependency) and ``register_agent`` the result.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from benchflow.agents.registry import AgentConfig, register_agent, resolve_agent

__all__ = [
    "AgentSpec",
    "AgentsManifest",
    "SandboxSpec",
    "Seat",
    "Services",
    "agent_model_id",
    "load_byoa",
    "resolve_spec",
]


def agent_model_id(cfg: AgentConfig, model: str | None) -> str:
    """Default seat id: ``<agent>-<model>`` with provider prefixes stripped."""
    model_id = (model or cfg.default_model or "model").split("/")[-1]
    raw = f"{cfg.name}-{model_id}"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-")


class AgentSpec(BaseModel):
    """One declared agent (before count fan-out)."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None  # optional seat-label override; default = agent-model
    agent: str | None = None  # prebuilt registry name (XOR manifest)
    manifest: str | None = None  # path to a BYOA agent manifest.toml (XOR agent)
    model: str | None = None  # omit → AgentConfig.default_model
    instructions: str | None = None  # path to this agent's CLAUDE.md/AGENTS.md body
    count: int = Field(default=1, ge=1)
    env: dict[str, str] = Field(default_factory=dict)
    reasoning_effort: str | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> AgentSpec:
        if bool(self.agent) == bool(self.manifest):
            ident = self.name or self.agent or self.manifest
            raise ValueError(
                f"agent {ident!r}: set exactly one of `agent:` or `manifest:`"
            )
        return self


class Services(BaseModel):
    """The ONE shared service every seat runs against."""

    model_config = ConfigDict(extra="forbid")

    manifest: str | None = None  # name of the [[services]] block (default: the task's)
    shared: bool = True
    url_env: str | None = None  # extra env var to export the service URL under
    # (e.g. CASINO_URL — the var the task's in-sandbox CLI reads).
    seat_env: str | None = None  # extra env var to export THIS seat's id under
    # (e.g. CASINOBENCH_SEAT_ID — how the task identifies the player/seat).
    url: str | None = None  # external service URL → skip bootstrap, use as-is
    command: str | None = None  # host subprocess that starts the shared service
    # ({port} is substituted). Reached from inside the sandbox over the bridge.
    cwd: str | None = None  # working dir for `command`
    port: int | None = None  # fixed service port (0/None → ephemeral)
    health: str = "/health"  # health path polled until the service is up
    env: dict[str, str] = Field(default_factory=dict)  # extra env for `command`
    standings_path: str | None = None  # path → final {seat: score}; if set, the
    # floor writes a per-seat reward vector (SharedEnvReward) into floor.json.
    events_path: str | None = None  # path → {"jsonl": …} event log; if set, the
    # floor snapshots it to events.jsonl (for the town viewer's animated board).


class SandboxSpec(BaseModel):
    """The ONE shared sandbox all seats share (each in /work/<seat>)."""

    model_config = ConfigDict(extra="forbid")

    image_dir: str | None = None  # Dockerfile dir for the shared image
    name: str = "native-floor"


class ByoaManifest(BaseModel):
    """Strict pydantic mirror of the agent manifest contract (v1, ACP-only)."""

    model_config = ConfigDict(extra="forbid")

    contract_version: str
    name: str
    install_cmd: str
    launch_cmd: str
    description: str = ""
    protocol: Literal["acp"] = "acp"
    api_protocol: str = ""
    install_timeout: int = 900
    env_mapping: dict[str, str] = Field(default_factory=dict)
    acp_model_format: Literal["bare", "provider/model", "registered-provider/model"] = (
        "bare"
    )
    supports_acp_set_model: bool = True
    default_model: str = ""
    skill_paths: list[str] = Field(default_factory=list)
    home_dirs: list[str] = Field(default_factory=list)
    requires_env: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)  # discovery-only; ignored here


def load_byoa(path: str | Path) -> AgentConfig:
    """Validate a BYOA manifest.toml (strict) and register it → AgentConfig."""
    data = tomllib.loads(Path(path).read_text())
    m = ByoaManifest.model_validate(data)  # strict: unknown field / non-acp → error
    return register_agent(
        name=m.name,
        install_cmd=m.install_cmd,
        launch_cmd=m.launch_cmd,
        protocol=m.protocol,
        requires_env=m.requires_env,
        description=m.description,
        skill_paths=m.skill_paths,
        install_timeout=m.install_timeout,
        default_model=m.default_model,
        api_protocol=m.api_protocol,
        env_mapping=m.env_mapping,
        home_dirs=m.home_dirs,
        acp_model_format=m.acp_model_format,
        supports_acp_set_model=m.supports_acp_set_model,
    )


def resolve_spec(spec: AgentSpec, base_dir: Path) -> AgentConfig:
    """One AgentConfig for either source. ``manifest:`` is BYOA; else prebuilt."""
    if spec.manifest:
        return load_byoa(base_dir / spec.manifest)
    assert spec.agent is not None  # guaranteed by AgentSpec validator
    return resolve_agent(spec.agent)


@dataclass
class Seat:
    """One resolved, runnable seat — one agent process in its own folder."""

    seat_id: str
    spec: AgentSpec
    config: AgentConfig

    @property
    def agent_cwd(self) -> str:
        return f"/work/{self.seat_id}"

    @property
    def is_byoa(self) -> bool:
        return self.spec.manifest is not None


class AgentsManifest(BaseModel):
    """The whole ``agents.yaml`` — one shared task + its seats."""

    model_config = ConfigDict(extra="forbid")

    task: str | None = None
    task_path: str | None = None
    services: Services = Field(default_factory=Services)
    sandbox: SandboxSpec = Field(default_factory=SandboxSpec)
    out: str = "out/native-floor"
    deadline_s: int = 1200
    idle_timeout_s: int = 300
    drive: Literal["auto-loop", "service-rounds"] = "auto-loop"
    prompt: str | None = None  # shared task prompt sent to each seat (auto-loop)
    agents: list[AgentSpec] = Field(min_length=1)

    _base_dir: Path = PrivateAttr(default_factory=lambda: Path("."))

    @classmethod
    def from_yaml(cls, path: str | Path) -> AgentsManifest:
        path = Path(path)
        m = cls.model_validate(yaml.safe_load(path.read_text()) or {})
        m._base_dir = path.resolve().parent
        return m

    def resolve_path(self, p: str | Path) -> Path:
        """Resolve a manifest-relative path against the yaml's directory."""
        return self._base_dir / p

    def instructions_path(self, spec: AgentSpec) -> Path | None:
        """Resolve a seat's instruction file relative to the yaml dir."""
        if not spec.instructions:
            return None
        return self.resolve_path(spec.instructions)

    def seats(self) -> list[Seat]:
        """Resolve + fan out every entry into runnable seats (ids must be unique)."""
        out: list[Seat] = []
        seen: set[str] = set()
        for spec in self.agents:
            cfg = resolve_spec(spec, self._base_dir)
            base = spec.name or agent_model_id(cfg, spec.model)
            ids = (
                [base]
                if spec.count == 1
                else [f"{base}-{i}" for i in range(spec.count)]
            )
            for sid in ids:
                if sid in seen:  # else two seats share /work/<id> + trajectory dir
                    raise ValueError(
                        f"duplicate seat id {sid!r}: agent names (after count "
                        "fan-out) must be unique"
                    )
                seen.add(sid)
                out.append(Seat(seat_id=sid, spec=spec, config=cfg))
        return out
