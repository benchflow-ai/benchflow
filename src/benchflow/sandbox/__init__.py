"""benchflow.sandbox — isolated execution environments.

Public API:
    Sandbox, ExecResult      — protocol types
    ImageBuilder/Config/Ref  — image building protocol
    ServiceConfig, SERVICES  — sandbox service registry
    DockerSandbox        — local Docker backend
    DaytonaSandbox       — Daytona cloud backend
"""

from benchflow.sandbox.protocol import (
    ExecResult,
    ImageBuilder,
    ImageConfig,
    ImageRef,
    Sandbox,
    SandboxImage,
    SandboxSnapshotNotSupported,
)
from benchflow.sandbox.services import (
    SERVICES,
    ServiceConfig,
    build_service_hooks,
    detect_services_from_dockerfile,
    register_service,
)

__all__ = [
    "ExecResult",
    "ImageBuilder",
    "ImageConfig",
    "ImageRef",
    "Sandbox",
    "SandboxImage",
    "SandboxSnapshotNotSupported",
    "ServiceConfig",
    "SERVICES",
    "build_service_hooks",
    "detect_services_from_dockerfile",
    "register_service",
]
