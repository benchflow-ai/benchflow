"""The exec env's edges after b173 (main-sync round 12).

- Round 11's check of Daytona's staging folder ran
  ``stat`` from the system PATH as root, and an agent can own a folder on
  it (a workspace at ``/usr/sbin``). Daytona once carried an env
  only inline in the command and refused one too large for it. Since
  DAYTONA-EXEC a command too large for one session command (env included)
  travels as data under the transport contract
  (``benchflow.sandbox._transport``): root's file in root's own folder,
  opened by root's shell, with no tool looked up on any PATH.
- The privacy flag refused Modal, whose exec passes the
  env as a Secret, never in the command.

The backends run through the local-transport harness of
``test_verifier_script_shell``; every value is a dummy marker.
"""

from __future__ import annotations

import pytest

from benchflow.task import RolloutPaths
from tests.test_env_transport_b162 import _non_root_verifier
from tests.test_verifier_script_shell import _DASH, LocalImage, _backend, _stdout


@pytest.fixture(autouse=True)
def _needs_dash():
    if _DASH is None:
        pytest.skip("needs dash, the sh of Debian and Ubuntu images")


def _image_with_planted_tools(tmp_path):
    """An image whose PATH folder holds an agent's ``stat``, ``mkdir`` and
    ``chmod``: each leaves a marker if it ever runs."""
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    image = LocalImage(tmp_path, paths, with_bash=True)
    marker = tmp_path / "planted-tool-ran"
    for tool in ("stat", "mkdir", "chmod", "chown"):
        planted = image.bin / tool
        if planted.exists() or planted.is_symlink():
            planted.unlink()
        planted.write_text(f"#!/bin/sh\necho {tool} >> {marker}\nexit 0\n")
        planted.chmod(0o755)
    return image, marker


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["daytona", "daytona-dind"])
async def test_a_daytona_env_too_large_for_one_command_runs_as_staged_data(
    tmp_path, kind, monkeypatch
):
    """Fails on 57d28d656: the env was refused before anything ran."""
    from benchflow.sandbox import _transport
    from benchflow.sandbox._transport import COMMAND_BOUND

    sent = []
    send = _transport.sent

    def record(backend, what, *args, **kwargs):
        sent.append([what] if isinstance(what, str) else list(what))
        return send(backend, what, *args, **kwargs)

    monkeypatch.setattr(_transport, "sent", record)
    image, marker = _image_with_planted_tools(tmp_path)
    exec_ = _backend(kind, image, monkeypatch)
    await exec_("true", 60)  # the sandbox exists; what it ran to start is not counted
    image.sent.clear()
    sent.clear()
    env = {"PAD": "p" * 16_000, "JUDGE_API_KEY": "bf-dummy-r12"}
    result = await exec_(
        '[ "${#PAD}" -eq 16000 ] && [ "$JUDGE_API_KEY" = bf-dummy-r12 ] && echo ran',
        600,
        env=env,
    )
    assert (result.stdout or "").strip() == "ran", (result.stderr or "")[-300:]
    assert sent or image.sent
    for argv in [*sent, *image.sent]:
        # No value, and no command over the bound, on any command line.
        assert _transport.size(argv) <= COMMAND_BOUND
        assert not any("bf-dummy-r12" in part for part in argv)
        assert not any("\\142\\146\\055" in part for part in argv)
    assert not marker.exists(), marker.read_text()
    # The staged folder is gone once the command ended.
    assert not image.stage.exists() or not any(image.stage.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["daytona", "daytona-dind"])
async def test_a_daytona_env_up_to_the_limit_still_runs_inline(
    tmp_path, kind, monkeypatch
):
    """Every byte up to the limit: the octal form, no file, no planted tool."""
    image, marker = _image_with_planted_tools(tmp_path)
    exec_ = _backend(kind, image, monkeypatch)
    pad = "p" * 12_000  # four bytes each inline: under 48 KiB with the rest
    result = await exec_(
        '[ "${#PAD}" -eq 12000 ] && echo inline', 600, env={"PAD": pad}
    )
    assert (result.stdout or "").strip() == "inline", (result.stderr or "")[-300:]
    assert not marker.exists()


@pytest.mark.asyncio
async def test_a_non_root_verifier_beside_live_agents_gets_its_env_on_modal(
    tmp_path, monkeypatch
):
    """Fails on da9008e0: Modal's Secret channel was taken for a command line."""
    pytest.importorskip("modal")
    verifier = _non_root_verifier(tmp_path, "modal", monkeypatch)
    result = await verifier.verify()
    assert result.rewards == {"reward": 1.0}, _stdout(tmp_path)


def test_modal_and_the_stdin_backends_are_private_and_daytona_is_not():
    from benchflow.sandbox.apple_container import AppleContainerSandbox
    from benchflow.sandbox.daytona import DaytonaSandbox
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.sandbox.remote_docker import RemoteDockerSandbox

    modal = pytest.importorskip("benchflow.sandbox.modal_impl")
    for private in (
        DockerSandbox,
        RemoteDockerSandbox,
        AppleContainerSandbox,
        modal.ModalSandbox,
    ):
        assert private.exec_env_is_private is True, private
    assert DaytonaSandbox.exec_env_is_private is False
