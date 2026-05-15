"""Sandbox protocols and specifications."""

from benchflow.sandboxes.protocols import ImageBuilder, Sandbox, SandboxProvider
from benchflow.sandboxes.specs import ExecResult, ImageConfig, ImageRef, SandboxSpec

__all__ = [
    "ExecResult",
    "ImageBuilder",
    "ImageConfig",
    "ImageRef",
    "Sandbox",
    "SandboxProvider",
    "SandboxSpec",
]
