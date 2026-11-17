"""A trial whose main stops right after its release ends as sandbox_setup,
unscored and never retried (item 6). The real-Docker trial is in
tests/test_main_early_exit_docker.py."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchflow._utils.scoring import (
    SANDBOX_SETUP,
    classify_error,
    is_unrecoverable_startup_error,
)
from benchflow.rollout._setup import main_stopped_error


class _Main:
    def __init__(self, stopped) -> None:
        self.stopped = stopped

    async def main_stopped(self):
        return self.stopped


STOPPED = {
    "exit_code": 4,
    "entrypoint": ["/usr/local/bin/network_lock.sh"],
    "command": ["sh", "-c", "sleep infinity"],
    "output": [
        "iptables v1.8.9 (nf_tables): Could not fetch rule set generation id: "
        "Permission denied (you must be root)",
        "agent install failed: timed out",
    ],
}


@pytest.mark.asyncio
async def test_main_stopped_names_the_entrypoint_its_status_and_output():
    """The error names the ENTRYPOINT, its exit status and its last output,
    and is sandbox_setup and never retried, whatever its output says (an
    install failure, a timeout)."""
    error = await main_stopped_error(_Main(STOPPED))
    assert error is not None
    text = f"RuntimeError: {error}"
    assert "ENTRYPOINT (/usr/local/bin/network_lock.sh) exited with status 4" in text
    assert "Permission denied (you must be root)" in text
    assert classify_error(text) == SANDBOX_SETUP
    assert is_unrecoverable_startup_error(text)


@pytest.mark.asyncio
async def test_a_running_main_or_another_backend_ends_nothing():
    assert await main_stopped_error(_Main(None)) is None
    assert await main_stopped_error(SimpleNamespace()) is None


@pytest.mark.asyncio
async def test_main_is_found_stopped_on_the_compose_path(monkeypatch):
    """Compose keeps no id for a main that stopped during the release: main
    is found among the project's containers in any state (``ps -a``), and
    what the start read before its teardown stands once main is gone."""
    from benchflow.sandbox.docker import DockerSandbox

    calls: list[list[str]] = []

    class _Box:
        compose_project_name = "p1"
        _main_container = None

        async def _docker_cli(self, args, check=True, raw_output=False):
            calls.append(list(args))
            if args[0] == "ps":
                return SimpleNamespace(return_code=0, stdout="a" * 64 + "\n", stderr="")
            return SimpleNamespace(return_code=0, stdout="", stderr="lock failed\n")

        async def _inspect_container(self, cid):
            return {
                "State": {"Running": False, "ExitCode": 4},
                "Config": {"Entrypoint": ["/entry.sh"], "Cmd": []},
            }

        _service_container_id = DockerSandbox._service_container_id

    box = _Box()
    stopped = await DockerSandbox.main_stopped(box)
    assert stopped is not None and stopped["exit_code"] == 4
    assert stopped["entrypoint"] == ["/entry.sh"]
    assert stopped["output"] == ["lock failed"]
    assert calls[0][:2] == ["ps", "-a"]
    gone = SimpleNamespace(_main_stopped_record=stopped)
    assert await DockerSandbox.main_stopped(gone) is stopped
