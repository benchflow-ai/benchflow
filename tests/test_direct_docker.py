"""The direct Docker path's pure parts (PERF-E).

The fixture ``fixtures/direct_docker/compose-2.40.3.json`` is Compose's own
output for a BenchFlow-shaped project (BenchFlow's base and prebuilt files,
a task value with ``$`` and ``<>&``, a pinned subnet, the inert-bootstrap
overlay), recorded on a build VM by ``docker compose config``, ``config
--hash`` and the recording proxy around ``compose create`` / ``up``. Each
test here rebuilds one of those from the printed model and must match it.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from benchflow.sandbox import _direct_docker as dd
from benchflow.sandbox._engine_api import (
    EngineClient,
    EngineError,
    EngineUnavailable,
    negotiated,
)
from benchflow.task.compose_env import decode_dollars

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/direct_docker/compose-2.40.3.json").read_text()
)
ROOT = "/tmp/perf-e-fixture"
FILES = [
    f"{ROOT}/files/docker-compose-base.yaml",
    f"{ROOT}/files/docker-compose-prebuilt.yaml",
    f"{ROOT}/files/env.json",
    f"{ROOT}/files/networks.json",
]
# The overlay file the fixture used, as Compose reads it back ($$ -> $).
OVERLAY = {
    "entrypoint": [
        "/bin/sh",
        "-c",
        "trap 'r=1' USR1; exec \"$@\"",
        "benchflow-bootstrap",
        "PATH=/bin",
        "sh",
        "-c",
        "sleep infinity",
    ],
    "command": [],
    "environment": {"LD_PRELOAD": None, "BASH_ENV": None},
    "healthcheck": {"disable": True},
    "labels": {"io.benchflow.bootstrap.v1": "cmVjZWlwdA=="},
    "depends_on": {},
}


def _project() -> dict:
    return decode_dollars(FIXTURE["printed"])


def _taken() -> dict:
    """The fixture project, which the path takes (two environment variables:
    their order is now canonical)."""
    return _project()


def _requests(kind: str) -> list[dict]:
    return [r["body"] for r in FIXTURE["requests"] if f"/{kind}/create" in r["path"]]


# --- Compose's hashes ---------------------------------------------------------


def test_service_hash_is_composes():
    assert dd.service_hash(_project()["services"]["main"]) == FIXTURE["hash"]


def test_the_printed_dollars_are_not_hashed():
    """Compose prints ``$`` as ``$$``; its hash is over the value it runs."""
    printed = FIXTURE["printed"]["services"]["main"]
    assert "$$" in json.dumps(printed)
    assert dd.service_hash(printed) != FIXTURE["hash"]


def test_bootstrap_merge_is_composes_model_and_hash():
    merged = dd.merge_bootstrap(_project()["services"]["main"], OVERLAY)
    expected = decode_dollars(FIXTURE["printed_bootstrap"])["services"]["main"]
    assert list(merged.items()) == list(expected.items())
    assert dd.service_hash(merged) == FIXTURE["hash_bootstrap"]


def test_network_body_is_composes():
    (expected,) = _requests("networks")
    assert dd.network_body(_project(), "default", "2.40.3") == expected


def test_container_bodies_are_composes():
    project = _project()
    prototype, final = _requests("containers")
    common = {
        "image_id": FIXTURE["image_id"],
        "compose_version": "2.40.3",
        "working_dir": f"{ROOT}/project",
    }
    got = dd.container_body(
        project,
        project["services"]["main"],
        config_files=FILES,
        replace=False,
        **common,
    )
    assert dd.order_rules(got) == dd.order_rules(prototype)
    merged = dd.merge_bootstrap(project["services"]["main"], OVERLAY)
    got = dd.container_body(
        project,
        merged,
        config_files=[*FILES, f"{ROOT}/files/bootstrap.json"],
        replace=True,
        **common,
    )
    assert dd.order_rules(got) == dd.order_rules(final)


def test_go_json_escapes_like_encoding_json():
    assert dd.go_json({"a": "<x>&\u2028é"}) == '{"a":"\\u003cx\\u003e\\u0026\\u2028é"}'


@pytest.mark.parametrize(
    ("cpus", "nano"),
    [(1, 1_000_000_000), (1.5, 1_500_000_000), (0.1, 100_000_000), (0.3, 300_000_000)],
)
def test_nano_cpus_uses_compose_gos_float32(cpus, nano):
    assert dd.nano_cpus(cpus) == nano


# --- What the path takes, and what goes to Compose ---------------------------


def test_the_fixture_project_is_taken():
    dd.check_project(_taken(), "perf-e-fixture")


def _with(**main) -> dict:
    project = _project()
    project["services"]["main"].update(main)
    return project


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda p: p["services"].update(db={"image": "x"}), "more than one service"),
        (lambda p: p.update(volumes={"v": {}}), "top-level volumes"),
        (lambda p: p["services"]["main"].update(ports=["80"]), "service key ports"),
        (
            lambda p: p["services"]["main"].update(
                healthcheck={"test": ["CMD", "true"]}
            ),
            "a healthcheck",
        ),
        (lambda p: p["services"]["main"].update(pull_policy="always"), "pull_policy"),
        (
            lambda p: p["services"]["main"].update(
                labels={"com.docker.compose.x": "1"}
            ),
            "com.docker.compose label",
        ),
        (
            lambda p: p["services"]["main"]["volumes"].append(
                dict(p["services"]["main"]["volumes"][0])
            ),
            "share a destination",
        ),
        (
            lambda p: p["services"]["main"]["volumes"].append(
                {"type": "volume", "source": "v", "target": "/v"}
            ),
            "other than a bind",
        ),
        (
            lambda p: p["services"]["main"]["volumes"][0].update(
                bind={"create_host_path": True, "propagation": "rshared"}
            ),
            "Mount API",
        ),
        (
            lambda p: p["services"]["main"].update(security_opt=["seccomp=unconfined"]),
            "security_opt seccomp",
        ),
        (
            lambda p: p["services"]["main"].update(network_mode="host"),
            "network_mode other than none",
        ),
        (
            lambda p: p["networks"]["default"].update(driver="macvlan"),
            "a network setting",
        ),
        (lambda p: p["services"]["main"].update(cgroup="weird"), "cgroup"),
        (
            lambda p: p["services"]["main"]["deploy"].update(replicas=2),
            "deploy setting",
        ),
    ],
)
def test_what_goes_to_compose(change, reason):
    project = _taken()
    change(project)
    with pytest.raises(dd.Fallback, match=reason):
        dd.check_project(project, "perf-e-fixture")


def test_another_project_name_goes_to_compose():
    with pytest.raises(dd.Fallback, match="project name"):
        dd.check_project(_taken(), "someone-else")


def test_a_bootstrap_overlay_key_the_path_does_not_merge_goes_to_compose():
    with pytest.raises(dd.Fallback):
        dd.merge_bootstrap(_project()["services"]["main"], {"ports": ["1"]})


# --- The endpoint --------------------------------------------------------------


def test_the_default_endpoint_is_the_clis(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"auths": {}}))
    assert dd.endpoint_socket(["--config", str(tmp_path)]) == "/var/run/docker.sock"
    (tmp_path / "config.json").write_text(json.dumps({"currentContext": "default"}))
    assert dd.endpoint_socket(["--config", str(tmp_path)]) == "/var/run/docker.sock"


def test_a_unix_docker_host_is_its_socket(tmp_path):
    args = ["--config", str(tmp_path), "-H", "unix:///run/x/docker.sock"]
    assert dd.endpoint_socket(args) == "/run/x/docker.sock"


@pytest.mark.parametrize(
    "extra",
    [
        ["-H", "tcp://10.0.0.1:2376"],
        ["-H", "ssh://me@host"],
        ["-H", "unix:///x.sock", "--tlsverify"],
        ["--context", "desktop-linux"],
    ],
)
def test_other_endpoints_go_to_compose(tmp_path, extra):
    with pytest.raises(dd.Fallback):
        dd.endpoint_socket(["--config", str(tmp_path), *extra])


def test_a_selected_context_goes_to_compose(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"currentContext": "colima"}))
    with pytest.raises(dd.Fallback, match="context"):
        dd.endpoint_socket(["--config", str(tmp_path)])


def test_only_proven_versions_take_the_path():
    proven = "2.40.3+ds1-0ubuntu1~24.04.1"
    dd.check_versions(proven, "1.51", "1.52")
    with pytest.raises(dd.Fallback, match=re.escape("a daemon speaking API 1.51")):
        dd.check_versions(proven, "1.51", "1.51")
    with pytest.raises(dd.Fallback, match=re.escape("Compose 2.39.4: the direct path")):
        dd.check_versions("2.39.4", "1.51", "1.52")
    with pytest.raises(
        dd.Fallback, match=re.escape("Engine API 1.48: the direct path")
    ):
        dd.check_versions(proven, "1.48", "1.48")


@pytest.mark.parametrize(
    "build",
    ["2.40.3", "v2.40.3", "2.40.3-rc.1", "2.40.3-desktop.1", "2.40.3+ds1-0ubuntu2"],
)
def test_another_build_of_the_proven_release_is_not_the_proven_build(build):
    """PERF-E-r3-A-L2: the gate matches the printed build exactly."""
    with pytest.raises(dd.Fallback, match="the direct path is proven on"):
        dd.check_versions(build, "1.51", "1.52")


def test_the_label_carries_composes_core_version():
    assert dd.compose_core("2.40.3+ds1-0ubuntu1~24.04.1") == "2.40.3"
    assert dd.compose_core("v2.40.3") == "2.40.3"
    assert dd.compose_core("2.40.31") == "2.40.31"
    assert dd.compose_core("not a version") is None


def test_the_setting(monkeypatch):
    assert dd.enabled({})
    assert dd.enabled({dd.SETTING_ENV: "1"})
    for off in ("0", "false", "off", "no"):
        assert not dd.enabled({dd.SETTING_ENV: off})


# --- The published normalisation (D962 (2), D965) ------------------------------


def _inspect(**over) -> dict:
    base = {
        "Path": "/bin/sh",
        "Args": ["-c", "x"],
        "Config": {
            "Env": ["A=1", "B=2"],
            "Entrypoint": ["/bin/sh"],
            "Cmd": ["-c", "x"],
            "User": "",
            "Labels": {"k": "v"},
        },
        "HostConfig": {
            "Binds": ["/h/a:/a:rw", "/h/b:/b:rw"],
            "DnsOptions": None,
            "ExtraHosts": [],
            "CapAdd": None,
            "Privileged": False,
        },
        "Mounts": [{"Destination": "/a"}, {"Destination": "/b"}],
    }
    for path, value in over.items():
        node = base
        *heads, last = path.split("__")
        for head in heads:
            node = node[head]
        node[last] = value
    return base


def _same(a: dict, b: dict) -> bool:
    return dd.normalise_inspect(a) == dd.normalise_inspect(b)


def test_rule_a_empty_is_empty_only_when_empty():
    assert _same(
        _inspect(), _inspect(HostConfig__DnsOptions=[], HostConfig__ExtraHosts=None)
    )
    assert _same(_inspect(), _inspect(HostConfig__CapAdd={}))
    assert not _same(_inspect(), _inspect(HostConfig__CapAdd=["NET_ADMIN"]))


def test_rule_b_env_is_a_multiset_unless_a_key_repeats():
    assert _same(_inspect(), _inspect(Config__Env=["B=2", "A=1"]))
    assert not _same(_inspect(), _inspect(Config__Env=["A=1", "B=3"]))
    twice = _inspect(Config__Env=["A=1", "A=2"])
    assert not _same(twice, _inspect(Config__Env=["A=2", "A=1"]))
    assert _same(twice, _inspect(Config__Env=["A=1", "A=2"]))


def test_rule_c_entrypoint_and_cmd_may_split_only_as_docker_runs_them():
    split = _inspect(Config__Entrypoint=["/bin/sh", "-c"], Config__Cmd=["x"])
    assert _same(_inspect(), split)
    moved = _inspect(
        Config__Entrypoint=["/bin/sh", "-c"], Config__Cmd=["x"], Path="/bin/sh"
    )
    assert _same(_inspect(), moved)
    other = _inspect(Args=["-c", "y"], Config__Cmd=["-c", "y"])
    assert not _same(_inspect(), other)


def test_d965_binds_and_mounts_are_multisets_unless_a_destination_repeats():
    assert _same(
        _inspect(),
        _inspect(
            HostConfig__Binds=["/h/b:/b:rw", "/h/a:/a:rw"],
            Mounts=[{"Destination": "/b"}, {"Destination": "/a"}],
        ),
    )
    assert not _same(
        _inspect(), _inspect(HostConfig__Binds=["/h/a:/a:ro", "/h/b:/b:rw"])
    )
    dup = ["/h/a:/a:rw", "/h/c:/a:rw"]
    assert not _same(
        _inspect(HostConfig__Binds=dup), _inspect(HostConfig__Binds=dup[::-1])
    )


def test_everything_else_is_exact():
    assert not _same(_inspect(), _inspect(Config__User="root"))
    assert not _same(_inspect(), _inspect(HostConfig__Privileged=True))
    assert not _same(_inspect(), _inspect(Config__Labels={"k": "w"}))


def test_order_rules_leave_request_bodies_otherwise_exact():
    body = {
        "Env": ["B=1", "A=1"],
        "Cmd": [],
        "HostConfig": {"Binds": ["/x:/b", "/y:/a"]},
    }
    out = dd.order_rules(body)
    assert out["Env"] == ["A=1", "B=1"]
    assert out["HostConfig"]["Binds"] == sorted(body["HostConfig"]["Binds"])
    assert out["Cmd"] == []  # rule (a) is inspect's, not the request's


# --- The Engine API client, against a fake socket ------------------------------


async def _serve(tmp_path: Path, replies: list[bytes], seen: list[bytes]):
    async def handle(reader, writer):
        head = await reader.readuntil(b"\r\n\r\n")
        length = 0
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                length = int(line.split(b":")[1])
        seen.append(head + (await reader.readexactly(length) if length else b""))
        writer.write(replies.pop(0))
        await writer.drain()
        writer.close()

    path = str(tmp_path / "d.sock")
    server = await asyncio.start_unix_server(handle, path=path)
    return server, path


_PING = b"HTTP/1.1 200 OK\r\nApi-Version: 1.52\r\nContent-Length: 2\r\n\r\nOK"


@pytest.mark.asyncio
async def test_client_negotiates_down_and_reads_chunked_json(tmp_path):
    seen: list[bytes] = []
    chunked = (
        b"HTTP/1.1 201 Created\r\nContent-Type: application/json\r\n"
        b'Transfer-Encoding: chunked\r\n\r\n5\r\n{"Id"\r\n7\r\n:"abc"}\r\n0\r\n\r\n'
    )
    server, path = await _serve(tmp_path, [_PING, chunked], seen)
    async with server:
        client = EngineClient(path)
        reply = await client.call(
            "POST", "/containers/create", query={"name": "n"}, body={"A": "é"}
        )
    assert reply.json() == {"Id": "abc"}
    assert client.version == "1.51"
    assert seen[1].startswith(b"POST /v1.51/containers/create?name=n HTTP/1.1\r\n")
    assert seen[1].endswith('{"A":"é"}'.encode())


@pytest.mark.asyncio
async def test_client_raises_dockers_message(tmp_path):
    error = (
        b"HTTP/1.1 409 Conflict\r\nContent-Type: application/json\r\n"
        b'Content-Length: 22\r\n\r\n{"message":"in use x"}'
    )
    server, path = await _serve(tmp_path, [_PING, error], [])
    async with server:
        with pytest.raises(EngineError, match="409 in use x") as caught:
            await EngineClient(path).call("DELETE", "/networks/n")
    assert caught.value.status == 409
    assert dd.daemon_message(caught.value) == "Error response from daemon: in use x"


@pytest.mark.asyncio
async def test_client_refuses_an_old_daemon_and_a_missing_socket(tmp_path):
    old = b"HTTP/1.1 200 OK\r\nApi-Version: 1.41\r\nContent-Length: 0\r\n\r\n"
    server, path = await _serve(tmp_path, [old], [])
    async with server:
        with pytest.raises(EngineUnavailable, match="below"):
            await EngineClient(path).negotiate()
    with pytest.raises(EngineUnavailable, match="cannot reach"):
        await EngineClient(str(tmp_path / "none.sock")).negotiate()


def test_negotiated():
    assert negotiated("1.52") == "1.51"
    assert negotiated("1.45") == "1.45"


# --- The recording proxy ---------------------------------------------------------


@pytest.mark.asyncio
async def test_recorder_passes_bytes_and_records_bodies(tmp_path):
    from tests._docker_api_recorder import Recorder, read_log

    seen: list[bytes] = []
    head = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nX-Docker-Container-Path-Stat: e30=\r\n\r\n"
    created = b'HTTP/1.1 201 Created\r\nContent-Type: application/json\r\nContent-Length: 11\r\n\r\n{"Id":"ab"}'
    server, upstream = await _serve(tmp_path, [_PING, head, created], seen)
    log = tmp_path / "api.jsonl"
    recorder = Recorder(str(tmp_path / "proxy.sock"), upstream, log)
    async with server:
        await recorder.start()
        client = EngineClient(recorder.listen)
        # A HEAD reply carries a Content-Length and no body: the proxy must
        # not wait for one (docker cp's stat hung on that once).
        reply = await asyncio.wait_for(client.call("HEAD", "/containers/x/archive"), 5)
        assert reply.status == 200
        made = await client.call("POST", "/containers/create", body={"Env": ["A=1"]})
        assert made.json() == {"Id": "ab"}
        await recorder.close()
    rows = [r for r in read_log(log) if r["kind"] == "request"]
    assert [(r["method"], r["status"]) for r in rows] == [
        ("GET", 200),
        ("HEAD", 200),
        ("POST", 201),
    ]
    assert rows[2]["body"] == {"Env": ["A=1"]}
    assert rows[2]["reply"] == {"Id": "ab"}


def test_capabilities_as_the_docker_client_sends_them():
    """Found by the synthetic matrix (cap_add SYS_PTRACE): Compose's model
    keeps the task's spelling, its Docker client normalises at create."""
    assert dd.capabilities(["net_admin", "CAP_SYS_PTRACE", "NET_ADMIN", "all"]) == [
        "ALL",
        "CAP_NET_ADMIN",
        "CAP_SYS_PTRACE",
    ]
    assert dd.capabilities([]) is None
    assert dd.capabilities(None) is None


# --- Teardown deletes only what is the project's (PERF-E-B-L3, L4) -------------


class _Reply:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


class _FakeClient:
    """Answers list calls with fixed rows (as a proxy that dropped the filter
    would) and records every call."""

    def __init__(self, containers, networks):
        self.containers, self.networks, self.calls = containers, networks, []

    @staticmethod
    def quote(name):
        return name

    async def call(self, method, path, *, query=None, body=None, timeout=None):
        self.calls.append((method, path, query, timeout))
        if path == "/containers/json":
            return _Reply(self.containers)
        if path == "/networks":
            return _Reply(self.networks)
        if path == "/containers/create":
            return _Reply({"Id": "c" * 64})
        return _Reply(None)


OURS = "a" * 64
THEIRS = "b" * 64


def _row(cid, project, state="running", oneoff="False"):
    labels = {dd.COMPOSE_PROJECT: project, dd.COMPOSE_ONEOFF: oneoff}
    return {"Id": cid, "State": state, "Labels": labels}


def _direct(client) -> dd.DirectProject:
    return dd.DirectProject(
        client,
        _taken(),
        image_id="sha256:" + "d" * 64,
        compose_version="2.40.3",
        working_dir="/tmp/perf-e-fixture/project",
    )


@pytest.mark.asyncio
async def test_teardown_removes_only_rows_labelled_with_its_project():
    """Rows the label filter should have dropped (another project's, or no
    well-formed ID) are never killed or removed."""
    client = _FakeClient(
        [_row(OURS, "perf-e-fixture"), _row(THEIRS, "someone-else"), {"Id": "x"}],
        [
            {"Id": "e" * 64, "Labels": {dd.COMPOSE_PROJECT: "perf-e-fixture"}},
            {"Id": "f" * 64, "Labels": {dd.COMPOSE_PROJECT: "someone-else"}},
        ],
    )
    await _direct(client).teardown(volumes=True)
    touched = [(m, p, q) for m, p, q, _ in client.calls if m in ("POST", "DELETE")]
    # One container: its forced removal is the kill and the removal at once.
    assert touched == [
        ("DELETE", f"/containers/{OURS}", {"force": "1", "v": "1"}),
        ("DELETE", "/networks/" + "e" * 64, None),
    ]


@pytest.mark.asyncio
async def test_teardown_kills_every_container_before_removing_any():
    """Several containers (a task's sidecars) are all killed first, as
    Compose's kill does, then each is removed by force."""
    other = "c" * 64
    client = _FakeClient(
        [
            _row(OURS, "perf-e-fixture"),
            _row(other, "perf-e-fixture", state="exited"),
            _row("d" * 64, "perf-e-fixture", oneoff="True"),
        ],
        [],
    )
    await _direct(client).teardown(volumes=False)
    touched = [(m, p, q) for m, p, q, _ in client.calls if m in ("POST", "DELETE")]
    assert touched == [
        ("POST", f"/containers/{OURS}/kill", {"signal": "SIGKILL"}),
        ("POST", f"/containers/{'d' * 64}/kill", {"signal": "SIGKILL"}),
        ("DELETE", f"/containers/{OURS}", {"force": "1"}),
        ("DELETE", f"/containers/{other}", {"force": "1"}),
        ("DELETE", f"/containers/{'d' * 64}", {"force": "1"}),
    ]


@pytest.mark.asyncio
async def test_container_ids_refuse_a_row_of_another_project():
    client = _FakeClient([_row(OURS, "perf-e-fixture"), _row(THEIRS, "else")], [])
    with pytest.raises(RuntimeError, match="Cannot identify Compose containers"):
        await _direct(client).container_ids()
    client.containers = [_row(OURS, "perf-e-fixture")]
    assert await _direct(client).container_ids() == [OURS]


@pytest.mark.asyncio
async def test_create_and_start_wait_as_long_as_compose_up():
    """Compose's up had no deadline; neither do the direct create and start."""
    from benchflow.sandbox._engine_api import UNBOUNDED

    client = _FakeClient([], [])
    direct = _direct(client)
    cid = await direct.create(direct.project["services"]["main"], [], replace=False)
    await direct.start(cid)
    timeouts = {path.split("/")[-1]: t for m, path, _, t in client.calls if m == "POST"}
    assert timeouts == {"create": UNBOUNDED, "start": UNBOUNDED}


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [None, {"message": "odd"}, "x"])
async def test_teardown_fails_when_the_listing_is_not_a_list(answer):
    """B-R3-L4: never 'nothing to remove', so stop() falls back to its
    force-kill by label instead of leaking the project."""
    client = _FakeClient(answer, [])
    with pytest.raises(RuntimeError, match="container list was not a list"):
        await _direct(client).teardown(volumes=True)
    client = _FakeClient([_row(OURS, "perf-e-fixture")], answer)
    with pytest.raises(RuntimeError, match="network list was not a list"):
        await _direct(client).teardown(volumes=True)


@pytest.mark.asyncio
async def test_every_start_time_call_waits_as_long_as_compose_up():
    """B-R3-L3: network create, prototype remove, inspect and listing too."""
    from benchflow.sandbox._engine_api import UNBOUNDED, EngineError

    class _NoNetworkYet(_FakeClient):
        async def call(self, method, path, **kwargs):
            if method == "GET" and path.startswith("/networks/"):
                self.calls.append((method, path, None, kwargs.get("timeout")))
                raise EngineError(method, path, 404, "no such network")
            if (
                path.startswith("/containers/")
                and path.endswith("/json")
                and path != "/containers/json"
            ):
                self.calls.append((method, path, None, kwargs.get("timeout")))
                return _Reply({"Id": OURS})
            return await super().call(method, path, **kwargs)

    client = _NoNetworkYet([_row(OURS, "perf-e-fixture")], [])
    direct = _direct(client)
    await direct.ensure_network()
    await direct.remove(OURS)
    await direct.inspect(OURS)
    await direct.container_ids()
    assert client.calls and all(t == UNBOUNDED for _, _, _, t in client.calls)
