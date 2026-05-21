"""The Environment plane — Han's "S" (the stateful world).

A benchmark author writes a manifest (``environment.toml``); the default
adapter runs it on any Sandbox provider. See ``docs/architecture.md``,
"The Environment plane & the manifest".
"""

from benchflow.environment.manifest import (
    EnvironmentManifest,
    ForwardEnv,
    Readiness,
    ServiceSpec,
    TaskSelection,
    load_manifest,
)
from benchflow.environment.manifest_env import ManifestEnvironment
from benchflow.environment.protocol import (
    EnvHandle,
    Environment,
    EnvState,
    ReadinessProbe,
    StateSnapshot,
)
from benchflow.environment.readiness import wait_for_readiness

__all__ = [
    "EnvHandle",
    "EnvState",
    "Environment",
    "EnvironmentManifest",
    "ForwardEnv",
    "ManifestEnvironment",
    "Readiness",
    "ReadinessProbe",
    "ServiceSpec",
    "StateSnapshot",
    "TaskSelection",
    "load_manifest",
    "wait_for_readiness",
]
