"""`bench tasks check` finds plugin installs into any place the agent could write.

Guards the dx/errors fix for SkillsBench's powerlifting-coef-calc, which was
unscored on every trial of the hill-climb climb of 2026-09-30, the reference
solution's included: its test.sh installs pytest-json-ctrf into /root/.venv
after the agent stops, and /root is its workspace (WORKDIR /root), so the
pytest plugin guard refuses the plugin. The check caught that task's
``uv init``/``uv add`` in the working directory, but not the same install
written with an explicit path: ``/root/.venv``, ``~/.venv`` or ``$HOME``
(the verifier's home is /root), ``--python .../bin/python``, an activated
venv, or the agent's home, /logs and /testbed, which the guard also refuses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow._utils.task_authoring import check_task_warnings
from benchflow._utils.task_authoring.structural_checks import (
    task_workspace,
    workspace_plugin_installs,
)
from benchflow.sandbox.lockdown import installed_marker_plugins

CTRF = "pytest-json-ctrf==0.3.5"


@pytest.mark.parametrize(
    ("script", "place"),
    [
        (
            f"uv venv /root/.venv\nuv pip install --python /root/.venv/bin/python {CTRF}\n",
            "uv venv /root/.venv",
        ),
        (
            f"uv pip install --python=/root/.venv/bin/python {CTRF}\n",
            "uv pip install --python /root/.venv/bin/python",
        ),
        (
            f"python3 -m venv ~/.venv && ~/.venv/bin/pip install {CTRF}\n",
            "python -m venv ~/.venv",
        ),
        (
            f"python3 -m venv $HOME/v\n$HOME/v/bin/python -m pip install {CTRF}\n",
            "python -m venv $HOME/v",
        ),
        (
            f"source /root/.venv/bin/activate && pip install {CTRF}\n",
            "source /root/.venv/bin/activate",
        ),
        (f"uv add --project /root/proj {CTRF}\n", "uv add"),
    ],
    ids=["uv-venv", "uv-pip-python", "tilde-venv", "home-venv", "activate", "project"],
)
def test_installs_under_the_workspace_are_found(script, place):
    places, plugins = workspace_plugin_installs(script, workspace="/root")
    assert place in places
    assert plugins == [CTRF]
    # Where /root is not the workspace, the same script installs into a
    # root-only directory the guard trusts.
    places, _ = workspace_plugin_installs(script, workspace="/app")
    assert not places


@pytest.mark.parametrize(
    "script",
    [
        f"pip install --prefix=/home/agent/.local {CTRF}\n",
        f"uv venv /logs/v && uv pip install --python /logs/v {CTRF}\n",
        f"python3 -m venv /testbed/.venv\n/testbed/.venv/bin/pip install {CTRF}\n",
    ],
    ids=["agent-home", "logs", "testbed"],
)
def test_installs_into_the_agents_other_places_are_found_whatever_the_workspace(
    script,
):
    places, plugins = workspace_plugin_installs(script, workspace="/app")
    assert places and plugins == [CTRF]


@pytest.mark.parametrize(
    ("dockerfile", "workspace"),
    [
        ("FROM ubuntu:24.04\nWORKDIR /root\n", "/root"),
        ("FROM ubuntu:24.04\nRUN true\n", "/root"),  # WORKDIR / -> /root
        ("FROM python:3.12-slim\nWORKDIR /app\nWORKDIR src\n", "/app/src"),
        ("FROM ghcr.io/org/image:1\n", None),  # a base image's WORKDIR: unknown
        ("FROM ubuntu:24.04 AS build\nWORKDIR /build\nFROM build\n", None),
        ("FROM ubuntu:24.04\nWORKDIR $APP\n", None),
    ],
)
def test_the_workspace_comes_from_the_dockerfile(tmp_path, dockerfile, workspace):
    (tmp_path / "environment").mkdir()
    (tmp_path / "environment" / "Dockerfile").write_text(dockerfile)
    assert task_workspace(tmp_path) == workspace


def _task(tmp_path: Path, test_sh: str, workdir: str = "/root") -> Path:
    task = tmp_path / "powerlifting"
    (task / "tests").mkdir(parents=True)
    (task / "environment").mkdir()
    (task / "environment" / "Dockerfile").write_text(
        f"FROM ubuntu:24.04\nWORKDIR {workdir}\n"
    )
    (task / "instruction.md").write_text("Compute the Dots coefficient.\n")
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "tests" / "test.sh").write_text(test_sh)
    return task


EXPLICIT = f"""#!/bin/bash
uv venv /root/.venv --python 3.12
uv pip install --python /root/.venv/bin/python pytest==8.4.1 {CTRF}
/root/.venv/bin/pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA
"""


def test_check_names_the_plugin_the_place_and_the_workspace(tmp_path):
    task = _task(tmp_path, EXPLICIT)
    (warning,) = [w for w in check_task_warnings(task) if "pytest plugins" in w]
    assert f"installs pytest plugins ({CTRF})" in warning
    assert "uv venv /root/.venv" in warning
    assert "the workspace is /root" in warning
    assert "every trial ends unscored" in warning
    assert f"uvx --with {CTRF} pytest" in warning
    # The same script in a task whose workspace is /app is fine.
    task = _task(tmp_path / "app", EXPLICIT, workdir="/app")
    assert not [w for w in check_task_warnings(task) if "pytest plugins" in w]


def test_the_refusal_names_the_plugin_and_its_environment():
    marker = (
        "/root/.venv/lib/python3.12/site-packages/pytest_json_ctrf-0.3.5.dist-info"
        "\t/root/.venv/lib/python3.12/site-packages/pytest_json_ctrf-0.3.5.dist-info"
        " is under /root, which the agent could write\n"
        "/root/.venv/lib/python3.12/site-packages/ctrf/__init__.py\tis under /root\n"
    )
    assert installed_marker_plugins(marker) == (
        ["pytest-json-ctrf 0.3.5"],
        ["/root/.venv"],
    )


async def test_the_refusal_says_what_to_do(tmp_path, monkeypatch):
    """The run-time refusal names the plugin, where test.sh put it, the fix,
    and points at `bench tasks check` (the before-state named only a
    dist-info file)."""
    from tests.test_verifier_uv_state import (
        CTRF_PLUGIN,
        PASSING,
        install_plugin,
        score,
    )

    def workspace_venv(env, layout):
        site = layout.workspace / ".venv/lib/python3.12/site-packages"
        install_plugin(site, CTRF_PLUGIN)
        return [site]

    layout, _, rewards, error = await score(
        tmp_path, monkeypatch, PASSING, workspace_venv
    )
    assert rewards is None and error is not None
    assert error.startswith("verifier crashed: PluginGuardLoadError: ")
    assert (
        "pytest plugin ctrf-model 0.3.5 was installed after the agent stopped, "
        f"by the verifier into {layout.workspace}/.venv, where the agent could write"
    ) in error
    assert "This is a task problem: `bench tasks check" in error
    assert "`uvx --with ctrf-model==0.3.5 pytest ...`" in error
    assert "neither is any run of this task" in error
    stdout = layout.paths.test_stdout_path.read_text()
    assert "`bench tasks check` flags this task" in stdout
