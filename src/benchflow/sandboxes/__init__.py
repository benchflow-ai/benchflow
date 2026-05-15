"""Sandbox protocols and specifications."""

from benchflow.sandboxes.docker import DockerSandbox, DockerSandboxConfig
from benchflow.sandboxes.protocols import ImageBuilder, Sandbox, SandboxProvider
from benchflow.sandboxes.specs import ExecResult, ImageConfig, ImageRef, SandboxSpec

__all__ = [
    "DockerSandbox",
    "DockerSandboxConfig",
    "ExecResult",
    "ImageBuilder",
    "ImageConfig",
    "ImageRef",
    "Sandbox",
    "SandboxProvider",
    "SandboxSpec",
]
