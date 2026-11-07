"""A frozen launch reads back to the checked project on Compose 2.40 and 5.0.

Guards the fix for Docker Compose 5.0: it prints a bind whose
``create_host_path`` is true (what every short-syntax bind implies, BenchFlow's
own log binds included) as ``"bind": {}``, and reads a ``bind`` without the
key as false. The frozen launch is that print, so Compose read it back
differently ("Compose reads the checked project back differently, at
services.main.volumes[0].bind.create_host_path") and every Compose task was
refused. The launch file now spells the value out where, and only where,
that is the difference; Compose 2.40 prints true and reads its print back
unchanged, so nothing changes there.
"""

from __future__ import annotations

import json

import pytest

from benchflow.task.compose_env import _create_host_path_spelled_out
from tests.test_compose_launch import _from_task, _launched
from tests.test_compose_oracle import _compose_available, _compose_task


def _project(*binds: dict) -> dict:
    return {
        "name": "p",
        "services": {
            "main": {
                "image": "alpine:3.20",
                "volumes": [
                    {"type": "bind", "source": f"/s{i}", "target": f"/t{i}", **bind}
                    for i, bind in enumerate(binds)
                ],
            }
        },
    }


def test_compose_5s_dropped_true_is_spelled_out():
    printed = _project({"bind": {}}, {"bind": {"create_host_path": False}}, {})
    again = _project(
        {"bind": {"create_host_path": False}},
        {"bind": {"create_host_path": False}},
        {},
    )
    spelled = _create_host_path_spelled_out(printed, again)
    assert spelled == _project(
        {"bind": {"create_host_path": True}},
        {"bind": {"create_host_path": False}},
        {},
    )
    # The other keys of the bind mapping are kept.
    printed = _project({"bind": {"propagation": "rprivate"}})
    again = _project({"bind": {"propagation": "rprivate", "create_host_path": False}})
    assert _create_host_path_spelled_out(printed, again) == _project(
        {"bind": {"propagation": "rprivate", "create_host_path": True}}
    )


@pytest.mark.parametrize(
    "printed, again",
    [
        # Compose 2.40: the same project back, nothing to spell out.
        (_project({"bind": {"create_host_path": True}}),) * 2,
        (_project({"bind": {}}),) * 2,
        # Any other difference is still the refusal's to name.
        (_project({"bind": {}}), _project({"bind": {"create_host_path": True}})),
        (
            _project({"bind": {"create_host_path": True}}),
            _project({"bind": {"create_host_path": False}}),
        ),
        (
            _project({"bind": {}}),
            {
                **_project({"bind": {"create_host_path": False}}),
                "services": {
                    "main": {
                        **_project({"bind": {"create_host_path": False}})["services"][
                            "main"
                        ],
                        "privileged": True,
                    }
                },
            },
        ),
        (_project({}), _project({"bind": {"create_host_path": False}})),
    ],
)
def test_nothing_else_is_tolerated(printed, again):
    assert _create_host_path_spelled_out(printed, again) is None


@pytest.mark.skipif(
    not _compose_available(),
    reason="the freeze runs `docker compose config`, which needs docker compose",
)
def test_a_short_syntax_bind_is_frozen_and_launched_as_checked(tmp_path, monkeypatch):
    """On whichever Compose this host has: the task's short-syntax bind and
    BenchFlow's log binds freeze, and the launch keeps creating a missing
    host folder, as the checked project does."""
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_ENV", "")
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "absent.env"))
    monkeypatch.delenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", raising=False)
    monkeypatch.chdir(tmp_path)
    task = _compose_task(
        tmp_path / "task",
        "services:\n  main:\n    image: alpine:3.20\n    volumes: [./data:/data]\n",
    )
    (task / "environment" / "data").mkdir()
    sandbox = _from_task(task).inner
    frozen = json.loads(sandbox._launch.path.read_text())
    launched = _launched(sandbox)
    for volumes in (
        frozen["services"]["main"]["volumes"],
        launched["services"]["main"]["volumes"],
    ):
        binds = [v for v in volumes if v.get("type") == "bind"]
        assert binds
        # Compose 2.40 prints true; Compose 5 prints nothing for true.
        assert all(
            v.get("bind", {}).get("create_host_path", True) is True for v in binds
        ), volumes
