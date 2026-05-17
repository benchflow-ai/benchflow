"""benchflow.sandbox — isolated execution environments.

Public API:
    Sandbox, ExecResult      — protocol types
    ImageBuilder/Config/Ref  — image building protocol
    ServiceConfig, SERVICES  — sandbox service registry
    DockerEnvironment        — local Docker backend
    DaytonaEnvironment       — Daytona cloud backend
"""

from benchflow.sandbox.protocol import (
    ExecResult,
    ImageBuilder,
    ImageConfig,
    ImageRef,
    Sandbox,
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
    "ServiceConfig",
    "SERVICES",
    "build_service_hooks",
    "detect_services_from_dockerfile",
    "register_service",
]
