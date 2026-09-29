"""Fixtures for the review scenarios. See ``support.py``."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from tests.e2e.review import support


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    here = Path(__file__).resolve().parent
    for item in items:
        if here in Path(str(item.fspath)).resolve().parents:
            item.add_marker(pytest.mark.e2e)


@pytest.fixture(scope="session")
def review_sandbox() -> str:
    backend, reason = support.sandbox_or_reason()
    if backend is None:
        pytest.skip(reason)
    return backend


@pytest.fixture(scope="session")
def review_out(review_sandbox: str) -> Path:
    configured = os.environ.get("BENCHFLOW_E2E_OUT")
    root = Path(configured) if configured else Path(tempfile.mkdtemp(prefix="bf-e2e-"))
    out = root / "review"
    out.mkdir(parents=True, exist_ok=True)
    yield out
    # Every sandbox a scenario made carries BENCHFLOW_DAYTONA_OWNER, which
    # ``bench sandbox list`` filters on; none may outlive the session.
    import json

    from benchflow.sandbox.daytona import build_sync_client

    listed = support.bench("sandbox", "list", "--json", log=out / "sandbox-list.log")
    left = (
        json.loads(listed.output[listed.output.index("[") :])
        if "[" in listed.output
        else []
    )
    client = build_sync_client()
    for sandbox in left:
        client.get(sandbox["id"]).delete()
    (out / "sandboxes-left.txt").write_text(f"{len(left)}\n")
