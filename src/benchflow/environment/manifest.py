"""EnvironmentManifest — the declarative integration surface of the
Environment plane.

A benchmark ships an ``environment.toml``; this module loads it into a
validated model. Writing this manifest is the entire framework-integration
surface for a stateful benchmark (architecture.md, "The Environment plane
& the manifest").

The schema is honest to the two real stateful-multi-service benchmarks:

* **ClawsBench / smolclaws** — a ``base_image`` plus per-task images that
  bake seed data; the image has no service-starting entrypoint, so the
  framework starts the mock services itself from the ``[[environment.services]]``
  array (this array replaces the hard-coded ``SERVICES`` dict in
  ``benchflow.sandbox.services``). Task selection is ``image``-based.
* **chi-bench** — a single ``image`` whose entrypoint starts the services
  (``owns_lifecycle = true``); the task is chosen at runtime via an env var.
"""

from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class ServiceSpec(BaseModel):
    """One mock service the framework starts inside the environment.

    Used only when ``owns_lifecycle`` is false — the image does not start
    its own services. Mirrors ``benchflow.sandbox.services.ServiceConfig``,
    which this manifest array is designed to replace.
    """

    name: str
    command: str
    port: int
    health_path: str = "/health"

    model_config = {"extra": "forbid"}


class TaskSelection(BaseModel):
    """How the framework tells the environment which task to load.

    * ``image`` — the task's seed data is baked into a per-task image at
      build time (smolclaws); selecting a task means running that image.
    * ``env_var`` — one image serves every task; the task id is passed at
      runtime via an env var (chi-bench).
    """

    mechanism: Literal["image", "env_var"] = "env_var"
    key: str = "BENCHFLOW_TASK_ID"
    # "entrypoint" → set at `docker run` time so it reaches PID 1;
    # "exec" → set per `docker exec` call (does NOT reach the entrypoint).
    inject_into: Literal["entrypoint", "exec"] = "entrypoint"

    model_config = {"extra": "forbid"}


class Readiness(BaseModel):
    """Probes the framework gates on before the agent runs.

    When ``http`` and ``tcp`` are both empty, the framework derives HTTP
    probes from the declared services (``effective_http`` on the manifest).
    """

    http: list[str] = Field(default_factory=list)
    tcp: list[int] = Field(default_factory=list)
    timeout_sec: int = 120

    model_config = {"extra": "forbid"}


class ForwardEnv(BaseModel):
    """Host env vars forwarded into the environment container."""

    keys: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class StateSpec(BaseModel):
    """How the framework snapshots and restores the environment's state.

    The Environment plane's ``snapshot`` / ``restore`` — roll-back, which is
    definitional per the architecture — act on what this declares. For
    ``kind = "sqlite"``, ``paths`` are the database files (globs allowed)
    captured and restored per snapshot. Absent ``[environment.state]``, the
    environment is treated as stateless and snapshot/restore are unsupported.
    """

    kind: Literal["sqlite"] = "sqlite"
    paths: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class EnvironmentManifest(BaseModel):
    """A benchmark's self-describing Environment-plane declaration."""

    name: str
    # Exactly one of image / base_image is the run target. `image` is a
    # ready-to-run image (chi-bench); `base_image` is what per-task images
    # build FROM (smolclaws).
    image: str | None = None
    base_image: str | None = None
    ports: list[int] = Field(default_factory=list)
    services: list[ServiceSpec] = Field(default_factory=list)
    # True  → the image's entrypoint starts the services.
    # False → the framework starts each entry in `services`.
    owns_lifecycle: bool = True
    keep_alive: bool = True
    isolation: Literal["per_task", "persistent"] = "per_task"
    task_selection: TaskSelection = Field(default_factory=TaskSelection)
    readiness: Readiness = Field(default_factory=Readiness)
    forward_env: ForwardEnv = Field(default_factory=ForwardEnv)
    # [environment.state] — present iff the environment is stateful and
    # supports snapshot/restore (roll-back). None => stateless.
    state: StateSpec | None = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_consistency(self) -> EnvironmentManifest:
        if not self.image and not self.base_image:
            raise ValueError("manifest must set either `image` or `base_image`")
        if not self.owns_lifecycle and not self.services:
            raise ValueError(
                "owns_lifecycle = false requires a non-empty [[environment.services]] "
                "array — something has to start the services"
            )
        if self.owns_lifecycle and self.services:
            raise ValueError(
                "owns_lifecycle = true means the image entrypoint starts the "
                "services; remove [[environment.services]] or set owns_lifecycle = false"
            )
        return self

    @property
    def all_ports(self) -> list[int]:
        """Every port the environment exposes — declared + service-derived."""
        return sorted({*self.ports, *(s.port for s in self.services)})

    @property
    def effective_http(self) -> list[str]:
        """The HTTP readiness probes to gate on.

        Explicit ``readiness.http`` wins; otherwise probes are derived from
        the declared services so a smolclaws-style manifest needs no
        hand-written readiness section.
        """
        if self.readiness.http:
            return self.readiness.http
        return [f"http://localhost:{s.port}{s.health_path}" for s in self.services]

    @classmethod
    def model_validate_toml(cls, toml_data: str) -> EnvironmentManifest:
        """Parse a manifest from a TOML string.

        The benchmark-facing keys live under an ``[environment]`` table.
        """
        data = tomllib.loads(toml_data)
        env = data.get("environment")
        if env is None:
            raise ValueError("manifest must have an [environment] table")
        return cls.model_validate(env)


def load_manifest(path: str | Path) -> EnvironmentManifest:
    """Load and validate an environment manifest.

    ``path`` is either a TOML file path (the historical behavior) or a registry
    spec ``name@version`` resolved via ``$BENCHFLOW_ENV_REGISTRY``. The spec form
    lets a run bind its environment (the ``S`` axis) by name at the command line —
    decoupled from the task and swappable per run, like ``--agent`` / ``--model``
    / ``--sandbox``. Resolution is content-addressed so the bound world is
    recorded for replay.
    """
    p = Path(path)
    if not p.is_file():
        from benchflow._utils.env_registry import (
            looks_like_env_spec,
            resolve_environment,
        )

        if looks_like_env_spec(str(path)):
            resolved = resolve_environment(str(path))
            logger.info(
                "environment %s resolved -> %s (%s)",
                resolved.spec,
                resolved.manifest_path,
                resolved.env_hash,
            )
            return EnvironmentManifest.model_validate_toml(
                resolved.manifest_path.read_text()
            )
    return EnvironmentManifest.model_validate_toml(p.read_text())


def resolve_manifest_runtime_env(
    manifest: EnvironmentManifest,
    *,
    task_id: str,
    host_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Resolve the env vars the manifest contributes to the sandbox runtime.

    Combines two manifest control points into a single dict:

    * ``task_selection`` — when ``mechanism = "env_var"`` and
      ``inject_into = "entrypoint"``, the task id is bound under the
      configured key so the image entrypoint (or any in-sandbox service the
      framework starts) sees it at PID 1.
    * ``forward_env`` — each declared key is looked up in ``host_env``
      (defaults to ``os.environ``) and forwarded into the container. Keys
      missing on the host are silently skipped — the benchmark author owns
      the policy of whether absence is an error.

    The returned dict is suitable for merging into ``persistent_env`` so
    every subsequent ``sandbox.exec`` call and the compose-up environment
    both observe the values (see ``BaseSandbox._merge_env`` and Docker's
    compose env injection).
    """
    resolved_env: dict[str, str] = {}
    source_env = host_env if host_env is not None else dict(os.environ)

    sel = manifest.task_selection
    if sel.mechanism == "env_var" and sel.inject_into == "entrypoint":
        resolved_env[sel.key] = task_id

    for key in manifest.forward_env.keys:
        if key in source_env:
            resolved_env[key] = source_env[key]
        else:
            logger.debug("manifest forward_env: host env var %r not set; skipping", key)

    return resolved_env


def resolve_manifest_image(manifest: EnvironmentManifest) -> str | None:
    """Return the image the manifest declares as the sandbox run target.

    ``image`` (a ready-to-run image, chi-bench style) is the run target when
    set. ``base_image`` is what per-task images are built FROM (smolclaws
    style) — it is *not* a runnable target on its own, so this returns
    ``None`` and the existing task-built path remains in effect.
    """
    return manifest.image
