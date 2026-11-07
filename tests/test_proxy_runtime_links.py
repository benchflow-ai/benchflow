"""Guard the C1 fix in a6f09837 against real CPython terminfo links."""

from pathlib import Path

import pytest

from benchflow.providers import litellm_runtime as runtime
from benchflow.providers.proxy_requirements import ProxyInstall
from tests.test_root_only_tools import _layer_venv


@pytest.mark.asyncio
@pytest.mark.parametrize("directory", [False, True])
async def test_uv_lock_is_data_but_a_writable_runtime_directory_is_not(
    tmp_path, monkeypatch, directory
):
    sandbox = _layer_venv(tmp_path, monkeypatch, open_bin=False)
    lock = Path(runtime.PROXY_LAYER_VENV) / ".lock"
    if directory:
        lock.mkdir()
    else:
        lock.write_text("")
    lock.chmod(0o777 if directory else 0o666)
    python, record = await runtime._proxy_from_layer(
        sandbox, ProxyInstall(group="litellm-proxy")
    )
    assert (python is None) == directory
    assert record["install"] == ("in trial" if directory else "from layer")


@pytest.mark.asyncio
async def test_cached_python_allows_relative_links_within_protected_trees(
    tmp_path, monkeypatch
):
    sandbox = _layer_venv(tmp_path, monkeypatch, open_bin=False)
    terminfo = Path(runtime.PYTHON_INSTALL_DIR) / "share/terminfo"
    (terminfo / "a").mkdir(parents=True)
    (terminfo / "b").mkdir()
    (terminfo / "a/att505").write_text("dummy terminal data")
    (terminfo / "b/gs5430").symlink_to("../a/att505")
    python, record = await runtime._proxy_from_layer(
        sandbox, ProxyInstall(group="litellm-proxy")
    )
    assert python is not None
    assert record == {"install": "from layer"}


@pytest.mark.asyncio
async def test_cached_python_refuses_a_link_that_leaves_then_returns_to_the_tree(
    tmp_path, monkeypatch
):
    sandbox = _layer_venv(tmp_path, monkeypatch, open_bin=False)
    outside = tmp_path / "mutable-hop"
    outside.mkdir()
    outside.chmod(0o777)
    (outside / "back").symlink_to(runtime.PYTHON_INSTALL_DIR)
    link = Path(runtime.PROXY_LAYER_VENV) / "bin/python"
    link.unlink()
    link.symlink_to(outside / "back/python3.12")
    python, record = await runtime._proxy_from_layer(
        sandbox, ProxyInstall(group="litellm-proxy")
    )
    assert python is None
    assert record["install"] == "in trial"
