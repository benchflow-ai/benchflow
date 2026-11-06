"""A separate verifier's kit seal admits no image value naming its transfer roots.

Fails on 42bb217b, where that seal admitted an image LD_PRELOAD naming the
workspace (a missing file in a folder of root's), the transfer then unpacked
the agent's library there as root, and root's steps loaded it.
"""

from __future__ import annotations

import os
import subprocess
from types import SimpleNamespace

import pytest

from benchflow import _runtime_tree
from benchflow.sandbox import lockdown


def _names_transfer(roots: list[str], value: str) -> bool:
    script = _runtime_tree._transfer_roots_func(roots) + '\nbf_names_transfer "$1"'
    done = subprocess.run(["/bin/sh", "-c", script, "sh", value], check=False)
    return done.returncode == 0


@pytest.mark.skipif(not os.path.exists("/bin/sh"), reason="needs a POSIX shell")
def test_values_naming_the_transfer_roots_are_found_through_links_and_dots(tmp_path):
    """A path in a root is found however it is spelled; a neighbour's is not."""
    app = tmp_path / "app"
    (app / "lib").mkdir(parents=True)
    (tmp_path / "application").mkdir()
    (tmp_path / "opt").mkdir()
    (tmp_path / "opt" / "link").symlink_to(app)
    roots = [str(app), str(tmp_path / "artifacts")]
    found = {
        f"{app}/gi.so": True,
        f"{app}/missing/dir/x.so": True,
        f"/usr/lib:{app}/lib": True,
        f"{tmp_path}/opt/../app/gi.so": True,
        f"{tmp_path}/opt/link/gi.so": True,
        f"-Dx={app}/x": True,
        f"{tmp_path}/artifacts/gi.so": True,
        f"{tmp_path}/application/gi.so": False,
        "/usr/lib/x86_64-linux-gnu:/lib": False,
        "-O2": False,
    }
    assert {value: _names_transfer(roots, value) for value in found} == found


async def test_the_separate_verifiers_seal_is_given_its_transfer_roots(monkeypatch):
    """The workspace and /logs/artifacts reach the seal of the verifier's kit."""
    calls = []

    async def prepare(env, **kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(lockdown, "prepare_runtime_tree", prepare)
    env = SimpleNamespace(supports_runtime_kit=True)
    task = SimpleNamespace(config=SimpleNamespace(verifier=SimpleNamespace()))
    monkeypatch.setattr(
        "benchflow.sandbox._verifier_runtime.runtime_requested",
        lambda task: ((), False),
    )
    monkeypatch.setattr(
        "benchflow.sandbox._verifier_runtime.runner_requested", lambda task: False
    )
    await lockdown.prepare_separate_verifier_runtime(env, task, workspace="/app")
    assert calls == [{"transfer_roots": ("/app", "/logs/artifacts")}]


def test_the_agents_sandbox_seal_is_unchanged():
    """Without transfer roots the seal is the same text, the filter absent."""
    sealed = _runtime_tree.kit_prepare_command("seal")
    assert sealed == _runtime_tree.kit_prepare_command("seal", transfer_roots=())
    assert "bf_names_transfer" not in sealed
    assert "bf_names_transfer" in _runtime_tree.kit_prepare_command(
        "seal", transfer_roots=("/app",)
    )
