"""The differential's effect compare (tools/perf/direct_docker_diff.py), B-R3-L1.

Each effect is named by its target's role (the stopped prototype, main, the
network) and the sequence is compared in order, so "start main, remove the
prototype" and "start the prototype, remove main" differ, and a call that
names a container by an ID prefix is still attributed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT = "task__ab12cd34-ef56ab78"
PROTO = "1" * 64
MAIN = "2" * 64
NET = "3" * 64


def _tool():
    path = Path(__file__).parents[1] / "tools/perf/direct_docker_diff.py"
    spec = importlib.util.spec_from_file_location("direct_docker_diff", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _req(method, path, status=204, body=None, reply=None):
    return {
        "kind": "request",
        "method": method,
        "path": f"/v1.51{path}",
        "status": status,
        "body": body,
        "reply": reply,
    }


def _log(*steps):
    return [
        _req(
            "POST", "/networks/create", 201, {"Name": f"{PROJECT}_default"}, {"Id": NET}
        ),
        _req(
            "POST", f"/containers/create?name={PROJECT}-main-1", 201, {}, {"Id": PROTO}
        ),
        _req(
            "POST", f"/containers/create?name={PROJECT}-main-1", 201, {}, {"Id": MAIN}
        ),
        *steps,
    ]


def test_effects_name_their_target_and_keep_their_order():
    tool = _tool()
    right = _log(
        _req("DELETE", f"/containers/{PROTO}"),
        _req("POST", f"/containers/{MAIN}/start"),
        _req("DELETE", f"/networks/{NET}"),
    )
    assert tool.effects(right, PROJECT) == [
        "prototype: remove -> 204",
        "main: start -> 204",
        "network: remove -> 204",
    ]
    swapped = _log(
        _req("DELETE", f"/containers/{MAIN}"),
        _req("POST", f"/containers/{PROTO}/start"),
        _req("DELETE", f"/networks/{NET}"),
    )
    assert tool.effects(swapped, PROJECT) != tool.effects(right, PROJECT)
    reordered = _log(
        _req("POST", f"/containers/{MAIN}/start"),
        _req("DELETE", f"/containers/{PROTO}"),
        _req("DELETE", f"/networks/{NET}"),
    )
    assert tool.effects(reordered, PROJECT) != tool.effects(right, PROJECT)


def test_a_call_by_an_id_prefix_or_compose_s_temporary_name_is_attributed():
    tool = _tool()
    log = _log(
        _req("POST", f"/containers/{MAIN[:20]}/start"),
        _req("POST", f"/containers/{MAIN[:12]}_{PROJECT}-main-1/kill?signal=USR1"),
        _req("POST", f"/containers/{'9' * 64}/start"),  # not the project's
    )
    assert tool.effects(log, PROJECT) == [
        "main: start -> 204",
        "main: kill USR1 -> 204",
    ]


def test_a_stop_answered_304_and_a_rename_are_not_effects():
    tool = _tool()
    log = _log(
        _req("POST", f"/containers/{PROTO}/stop", 304),
        _req("POST", f"/containers/{MAIN}/rename?name={PROJECT}-main-1"),
    )
    assert tool.effects(log, PROJECT) == []
