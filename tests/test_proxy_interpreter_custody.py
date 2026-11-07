"""Guard FF-install-C1 against interpreter-target reuse at cd4d492d."""

from pathlib import Path

import pytest

from benchflow import _uv_pin
from benchflow.providers import litellm_runtime as runtime
from benchflow.providers.proxy_requirements import ProxyInstall
from benchflow.sandbox import lockdown
from tests.test_root_only_tools import _layer_venv, _RecordingEnv


@pytest.mark.asyncio
@pytest.mark.parametrize("part", ["binary", "bin", "stdlib", "external-link"])
async def test_cached_proxy_checks_the_interpreter_runtime_tree(
    tmp_path, monkeypatch, part
):
    sandbox = _layer_venv(tmp_path, monkeypatch, open_bin=False)
    python_root = tmp_path / "opt/benchflow/python"
    monkeypatch.setattr(runtime, "PYTHON_INSTALL_DIR", str(python_root), raising=False)
    interpreter = python_root / "python3.12"
    if part == "binary":
        interpreter.chmod(0o777)
    elif part == "bin":
        python_root.chmod(0o777)
    elif part == "stdlib":
        lib = python_root / "lib/python3.12"
        lib.mkdir(parents=True)
        (lib / "site.py").write_text("# dummy agent module\n")
        (lib / "site.py").chmod(0o666)
    else:
        outside = tmp_path / "outside"
        outside.mkdir()
        outside.chmod(0o777)
        target = outside / "python"
        target.write_text("#!/bin/sh\n")
        target.chmod(0o755)
        link = Path(runtime.PROXY_LAYER_VENV) / "bin/python"
        link.unlink()
        link.symlink_to(target)
    python, record = await runtime._proxy_from_layer(
        sandbox, ProxyInstall(group="litellm-proxy")
    )
    assert python is None
    assert record["install"] == "in trial"


@pytest.mark.asyncio
async def test_double_slash_workspace_cannot_own_tools():
    env = _RecordingEnv()
    with pytest.raises(ValueError, match="inside"):
        await lockdown.setup_sandbox_user(
            env, "agent", "/" + _uv_pin.BENCHFLOW_ROOT + "/python/bin"
        )
    assert env.commands == []


@pytest.mark.asyncio
async def test_workspace_alias_is_resolved_before_ownership_changes():
    from tests._receipt_fakes import receipt_answer

    class AliasEnv(_RecordingEnv):
        async def exec(self, command, **kwargs):
            result = await super().exec(command, **kwargs)
            result.stdout = _uv_pin.BENCHFLOW_ROOT + "/python/bin\n"
            # As the sandbox's shell prints it: the read takes a receipt.
            return receipt_answer(command, result)

    env = AliasEnv()
    with pytest.raises(ValueError, match="inside"):
        await lockdown.setup_sandbox_user(env, "agent", "/tmp/workspace-alias")
    assert not any("chown" in c for c in env.commands)
