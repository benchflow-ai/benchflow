"""Guard FF-install-C7 against unresolved extras cached by cd4d492d."""

import os
import shutil
import sys

from benchflow.providers import proxy_requirements as reqs
from tests.test_proxy_requirements import _freeze, _wheel


def test_repeated_host_extras_resolve_again_without_mutating_a_live_venv(
    tmp_path, monkeypatch
):
    uv = os.environ.get("UV") or shutil.which("uv")
    assert uv
    index = tmp_path / "index"
    index.mkdir()
    _, digest = _wheel(index, "lockedbase", "1.0")
    _wheel(index, "extra", "1.0", ("lockedbase>=1",))
    locked = tmp_path / "litellm-proxy.txt"
    locked.write_text(f"lockedbase==1.0 --hash=sha256:{digest}\n")
    monkeypatch.setattr(reqs, "requirements_file", lambda group: locked)
    reqs.requirements_text.cache_clear()
    monkeypatch.setattr(reqs, "PROXY_PYTHON", sys.executable)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("UV_NO_INDEX", "1")
    monkeypatch.setenv("UV_FIND_LINKS", str(index))
    install = reqs.ProxyInstall(reqs.PROXY_GROUP, ("extra>=1",))
    try:
        first = reqs.ensure_host_venv(install, uv=uv)
        _wheel(index, "extra", "2.0", ("lockedbase>=1",))
        second = reqs.ensure_host_venv(install, uv=uv)
        assert _freeze(second / "bin/python")["extra"] == "2.0"
        assert _freeze(first / "bin/python")["extra"] == "1.0"
        assert first != second
    finally:
        reqs.requirements_text.cache_clear()
