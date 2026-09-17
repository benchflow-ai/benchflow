"""Live stdio connections to a process inside a sandbox.

Provides a bidirectional pipe (send lines in, read lines out) needed for
ACP agents running inside containers. Implementations:

- DockerProcess:         `docker compose exec -i` (local Docker)
- ApptainerProcess:      `apptainer exec` (local Apptainer instance)
- AppleContainerProcess: `container exec -i` (Apple Container)
- DaytonaProcess:        SSH to a Daytona sandbox
- DaytonaPtyProcess:     Daytona PTY WebSocket
- AgentCoreProcess:      Bedrock AgentCore shell WebSocket

The first three are local-subprocess transports and share
:class:`SubprocessLiveProcess`; the last two carry bytes over a WebSocket and
implement :class:`LiveProcess` directly.

This package was a single ``process.py`` until it outgrew 1000 lines. Import
paths are unchanged — everything public is re-exported here.
"""

from benchflow.sandbox.process._base import (
    LiveProcess,
    SubprocessLiveProcess,
    drain_oversized_line,
)
from benchflow.sandbox.process.agentcore import AgentCoreProcess
from benchflow.sandbox.process.apple import AppleContainerProcess
from benchflow.sandbox.process.apptainer import ApptainerProcess
from benchflow.sandbox.process.daytona import DaytonaProcess, DaytonaPtyProcess
from benchflow.sandbox.process.docker import DockerProcess

__all__ = [
    "AgentCoreProcess",
    "ApptainerProcess",
    "AppleContainerProcess",
    "DaytonaProcess",
    "DaytonaPtyProcess",
    "DockerProcess",
    "LiveProcess",
    "SubprocessLiveProcess",
    "drain_oversized_line",
]
