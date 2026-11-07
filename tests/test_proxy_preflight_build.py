"""Guard FF-install-C5: cd4d492d executed sdist metadata during preflight."""

import os
import shlex
import shutil
import tarfile

from benchflow.providers import proxy_requirements as reqs
from tests.test_proxy_requirements import _wheel


def _resolver(tmp_path, monkeypatch):
    uv = os.environ.get("UV") or shutil.which("uv")
    assert uv, "run under uv with the pinned tool on PATH"
    index = tmp_path / "index"
    index.mkdir()
    locked = tmp_path / "locked.txt"
    locked.write_text("")
    monkeypatch.setattr(reqs, "requirements_file", lambda group: locked)
    wrapper = tmp_path / "uv-local-index"
    wrapper.write_text(
        f'#!/bin/sh\nexec {shlex.quote(uv)} "$@" --no-index '
        f"--find-links {shlex.quote(str(index))}\n"
    )
    wrapper.chmod(0o755)
    return index, str(wrapper)


def test_preflight_never_executes_sdist_metadata(tmp_path, monkeypatch):
    index, uv = _resolver(tmp_path, monkeypatch)
    source = tmp_path / "guardextra-1.0"
    source.mkdir()
    marker = tmp_path / "backend-ran"
    (source / "pyproject.toml").write_text(
        '[build-system]\nrequires=[]\nbuild-backend="backend"\n'
        'backend-path=["."]\n[project]\nname="guardextra"\n'
        'version="1.0"\ndynamic=["dependencies"]\n'
    )
    (source / "backend.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('dummy backend executed')\n"
        "raise RuntimeError('dummy metadata backend')\n"
    )
    with tarfile.open(index / "guardextra-1.0.tar.gz", "w:gz") as archive:
        archive.add(source, arcname=source.name)
    result = reqs._resolve(("guardextra==1.0",), reqs.PROXY_GROUP, uv)
    assert result is not None
    assert not marker.exists(), "preflight executed an unhashed build backend"


def test_preflight_ignores_workspace_and_ambient_uv_config(tmp_path, monkeypatch):
    index, uv = _resolver(tmp_path, monkeypatch)
    _wheel(index, "guardextra", "1.0")
    (tmp_path / "uv.toml").write_text('resolution="not-a-resolution"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UV_CONFIG_FILE", str(tmp_path / "uv.toml"))
    monkeypatch.setenv("UV_RESOLUTION", "not-a-resolution")
    assert reqs._resolve(("guardextra==1.0",), reqs.PROXY_GROUP, uv) is None
