"""PT b179's live lock upload meets FF-install's agents folder, in a real
container (merge wave batch 32).

A live install uploads a Python agent's hash lock (PT b179) into
``/opt/runtime/agents``, which FF-install's recipe checks root's alone
before uv reads anything there. Docker's upload keeps a non-root host's uid
(FF-INSTALL-ON-PT-R1), so the upload is made root's before the recipe runs.
The recipe here is the guard and the lock step the registry's agents use,
with BenchFlow's own ``install_agent``, ``docker cp`` and route. Skipped
without a Docker daemon (it is run on a Docker VM).
"""

from __future__ import annotations

import dataclasses

import pytest

from tests.test_c8_protected_runtime_container import _dx, _sandbox, box  # noqa: F401


@pytest.mark.asyncio
@pytest.mark.parametrize("host_uid", [0, 1000])
async def test_a_live_lock_upload_passes_the_agents_folder_check(
    box,  # noqa: F811
    tmp_path,
    monkeypatch,
    host_uid,
):
    from benchflow._root_only import root_only_tree_check
    from benchflow.agents import install as install_mod
    from benchflow.agents.registry import _AGENT_UV_ROOT, AGENTS, RequirementsFile
    from benchflow.sandbox import daytona_layers

    lock = RequirementsFile(
        f"{_AGENT_UV_ROOT}/merge-wave-requirements.txt",
        ("dummy==1",),
        "/usr/local/bin/python3",
    )
    recipe = (
        f"mkdir -p {_AGENT_UV_ROOT} && "
        + root_only_tree_check(_AGENT_UV_ROOT, "BenchFlow's agent uv tools")
        + " && "
        + lock.command
        + f" && cat {lock.path}"
    )
    agent = "merge-wave-lock"
    config = dataclasses.replace(
        AGENTS["deepagents"], install_cmd=recipe, requirements_files=(lock,)
    )
    monkeypatch.setitem(install_mod.AGENTS, agent, config)
    monkeypatch.setitem(install_mod.AGENT_INSTALLERS, agent, recipe)

    async def no_layer(*args, **kwargs):
        return None

    monkeypatch.setattr(daytona_layers, "cached_layer_in_sandbox", no_layer)

    sandbox = _sandbox(box)
    upload = sandbox.upload_file

    async def as_the_host_left_it(source, target, **kwargs):
        await upload(source, target, **kwargs)
        # What docker cp leaves from a host user with this uid.
        kept = _dx(box, f"chown {host_uid}:{host_uid} {target}")
        assert kept.returncode == 0, kept.stderr

    sandbox.upload_file = as_the_host_left_it

    assert (
        await install_mod.install_agent(sandbox, agent, tmp_path / "rollout") is config
    )
    log = (tmp_path / "rollout/agent/install-stdout.txt").read_text()
    assert "dummy==1" in log
    owner = _dx(box, f"stat -c %u:%g:%a {lock.path}")
    uid, gid, mode = owner.stdout.strip().split(":")
    assert (uid, gid) == ("0", "0") and int(mode, 8) & 0o022 == 0
