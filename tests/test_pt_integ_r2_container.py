"""PT-integ round 2 in real containers (VM Docker): ``ENV PYTHONPATH=/app``.

PT-INTEG-B-3 (review B, High) at 403505e5: the image's own PYTHONPATH never
reached the hook clean-up. Root's route unset it (``_base.py``'s unset list)
before keeping it as data, and the pre-agent record is taken with ``-I``, so
an image with ``ENV PYTHONPATH=/app`` (``WORKDIR /app``, the agent's
workspace) kept the agent's ``/app/sitecustomize.py``, which test.sh's
Python then ran as root. The earlier control injected the value straight
into the clean-up; this one goes through BenchFlow's real Docker backend,
kit build and clean-up.

The agent's hooks are dummy markers: run as root, they make a folder under
``/tmp/marks`` and nothing else. Each removal case has a live control.
Skipped without a Docker daemon.
"""

from __future__ import annotations

import shutil

import pytest

from tests.test_pt_r16_hooks_container import (
    _ROOT_HOOK,
    _box,
    _build,
    _dx,
    _marks,
    _plant,
    _setup,
)

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
]

APP_PYTHONPATH = """FROM python:3.12-slim
LABEL target="PT-R16"
WORKDIR /app
ENV PYTHONPATH=/app
RUN useradd -m -u 1000 agent && chown agent /app \\
 && printf 'VALUE = "honest"\\n' > /app/honest_module.py
"""


@pytest.fixture(scope="module")
def app_pythonpath(tmp_path_factory):
    return _build(tmp_path_factory, "pt-integ-r2-app-pythonpath:test", APP_PYTHONPATH)


def _root_python(name: str) -> str:
    """test.sh's Python as root, with the image's own environment (docker
    exec passes the image ENV, PYTHONPATH=/app included)."""
    ran = _dx(
        name,
        "cd /tmp && /usr/local/bin/python3 -c 'import honest_module; print(honest_module.VALUE)'",
    )
    return ran.stdout.strip() or ran.stderr


def _plant_hooks(name: str) -> None:
    _plant(name, "/app/sitecustomize.py", _ROOT_HOOK.format(name="site"))
    _plant(name, "/app/usercustomize.py", _ROOT_HOOK.format(name="user"))


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["soft-verify", "scoring-step"])
async def test_image_pythonpath_app_hooks_go_through_the_docker_backend(
    app_pythonpath, path
):
    """PT-INTEG-B-3, fails on 403505e5: the clean-up never saw /app, so the
    agent's /app/sitecustomize.py stayed and ran as root in test.sh's Python.
    ``soft-verify``: ``cleanup_verifier_python_hooks``; ``scoring-step``:
    harden_before_verify's clean-up step, one exec as on a per-step backend."""
    from benchflow.sandbox import lockdown

    with _box(app_pythonpath) as name:
        sandbox = await _setup(name, "/app")
        _plant_hooks(name)
        # Live control: the image's root Python runs the agent's hook.
        assert _root_python(name) == "honest"
        assert _marks(name) == ["site", "user"], _marks(name)
        _dx(name, "rm -rf /tmp/marks")
        if path == "soft-verify":
            await lockdown.cleanup_verifier_python_hooks(
                sandbox, None, workspace="/app", user="root"
            )
        else:
            step = lockdown._cleanup_step(lockdown.HARDENING_DEFAULTS, "/app")
            await lockdown._run_prep_step(sandbox, step)
        assert _marks(name) == [], "a hook ran during the clean-up"
        left = _dx(name, "ls /app").stdout.split()
        assert "sitecustomize.py" not in left
        assert "usercustomize.py" not in left
        # The honest module and test.sh's import of it stay.
        assert _root_python(name) == "honest"
        assert _marks(name) == [], "the agent's hook survived into test.sh's Python"
