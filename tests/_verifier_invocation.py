"""In-memory invocation adapter for the historical fake-daemon tests.

Real resolution, plugin paths and transport selection are exercised separately
in test_verifier_resolved_invocation.py. This adapter supplies only the new
record shape to tests whose daemon has no executable toolchain.
"""

from pathlib import Path

from benchflow.sandbox._compose_build import ComposeBuild
from benchflow.sandbox.env_layers import LayerError
from benchflow.sandbox.verifier_layers import compose_builder_identity


async def synthetic_invocation(box):
    env = {**box._docker_compose_env(), **box._compose_transport_env()}
    try:
        builder = compose_builder_identity(env)
        uncertainty = None
    except LayerError as exc:
        builder, uncertainty = "", str(exc)
    return ComposeBuild(
        tuple(box._compose_command(["build"])),
        tuple(sorted(env.items())),
        str(Path.cwd()),
        builder,
        uncertainty,
    )
