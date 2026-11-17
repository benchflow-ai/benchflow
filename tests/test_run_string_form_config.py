"""bf.run("agent", task_path=..., config=...) passes on the same RuntimeConfig
settings as bf.run(Agent(...), env, config).

Guards the audit fix on branch ``audit-cloud-metadata-opt-out``:
the string form built its RolloutConfig without ``allow_cloud_metadata``, so
``RuntimeConfig(allow_cloud_metadata=True)`` was silently ignored there while
``Runtime.execute`` passed it.
"""

from __future__ import annotations

import pytest

import benchflow as bf
import benchflow.runtime as runtime_module
from benchflow.runtime import RuntimeConfig


class _Built(Exception):
    def __init__(self, config):
        self.config = config


async def _string_form_config(tmp_path, monkeypatch, config):
    task = tmp_path / "task"
    task.mkdir(exist_ok=True)
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("Do nothing.\n")

    async def create(rollout_config):
        raise _Built(rollout_config)

    monkeypatch.setattr("benchflow.rollout.Rollout.create", create)
    monkeypatch.setattr(runtime_module, "check_rollout_config", lambda _c: None)
    monkeypatch.setattr(runtime_module, "check_host", lambda _c: None)
    with pytest.raises(_Built) as built:
        await bf.run("oracle", task_path=task, config=config)
    return built.value.config


@pytest.mark.asyncio
async def test_the_string_form_passes_the_cloud_metadata_opt_out(tmp_path, monkeypatch):
    allowed = await _string_form_config(
        tmp_path, monkeypatch, RuntimeConfig(allow_cloud_metadata=True)
    )
    default = await _string_form_config(tmp_path, monkeypatch, RuntimeConfig())

    assert allowed.allow_cloud_metadata is True
    assert default.allow_cloud_metadata is False
