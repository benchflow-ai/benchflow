"""bfx's host side: pins, the verifying locator, the protocol client and the
delivery helpers (src/benchflow/_bfx). The protocol cases run against the
real binary where this machine has a pinned one (Linux, after ``make bfx``)
and against a stand-in otherwise."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import re
import sys
from pathlib import Path

import pytest

from benchflow._bfx import client, delivery, locate, pins


def test_pins_name_both_architectures_by_sha256():
    assert set(pins.SHA256) == {"x86_64", "aarch64"}
    for value in pins.SHA256.values():
        assert re.fullmatch(r"[0-9a-f]{64}", value)
    # The build script's check reads them with this exact layout.
    text = Path(pins.__file__).read_text()
    for arch, value in pins.SHA256.items():
        assert f'    "{arch}": "{value}",\n' in text


@pytest.mark.parametrize(
    ("given", "want"),
    [("x86_64", "x86_64"), ("amd64", "x86_64"), ("linux/amd64", "x86_64"),
     ("aarch64", "aarch64"), ("arm64", "aarch64"), ("linux/arm64/v8", "aarch64")],
)  # fmt: skip
def test_architecture_names(given, want):
    assert locate.normalize_arch(given) == want


def test_other_architectures_are_refused():
    with pytest.raises(locate.BfxUnavailable):
        locate.normalize_arch("riscv64")


def _fake_bin(tmp_path, monkeypatch, content: bytes, *, link: bool = False) -> Path:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    target = bindir / "bfx-x86_64"
    if link:
        real = tmp_path / "real"
        real.write_bytes(content)
        target.symlink_to(real)
    else:
        target.write_bytes(content)
    monkeypatch.setattr(locate, "BIN_DIR", bindir)
    monkeypatch.setattr(locate, "_checked", {})
    return target


def test_the_locator_refuses_a_binary_that_is_not_the_pin(tmp_path, monkeypatch):
    _fake_bin(tmp_path, monkeypatch, b"not bfx")
    with pytest.raises(locate.BfxUnavailable, match="not the pinned"):
        locate.verified_binary("x86_64")
    assert not locate.available("amd64")


def test_the_locator_accepts_the_pinned_bytes_and_refuses_a_link(tmp_path, monkeypatch):
    import hashlib

    data = b"pretend binary"
    monkeypatch.setitem(pins.SHA256, "x86_64", hashlib.sha256(data).hexdigest())
    path = _fake_bin(tmp_path, monkeypatch, data)
    assert locate.verified_binary("x86_64") == path
    # Changed after the check: a new stat key, checked again, refused.
    path.write_bytes(b"pretend binary, changed")
    with pytest.raises(locate.BfxUnavailable):
        locate.verified_binary("x86_64")
    linked = tmp_path / "other"
    linked.mkdir()
    monkeypatch.chdir(linked)
    path.unlink()
    path.symlink_to(tmp_path / "anything")
    with pytest.raises(locate.BfxUnavailable, match="not a regular file"):
        locate.verified_binary("x86_64")


def test_a_missing_binary_says_how_to_build_it(tmp_path, monkeypatch):
    monkeypatch.setattr(locate, "BIN_DIR", tmp_path)
    with pytest.raises(locate.BfxUnavailable, match="make bfx"):
        locate.verified_binary("aarch64")


def test_one_shot_argv_is_absolute_and_can_keep_secrets_out_of_argv():
    argv = client.one_shot_argv("exec", {"argv": ["/bin/true"], "env": {"K": "secret"}})
    assert argv[:3] == ["/.bfx/bfx", "exec", "--json"]
    assert json.loads(argv[3])["env"] == {"K": "secret"}
    quiet = client.one_shot_argv("exec", {"env": {"K": "secret"}}, args_on_stdin=True)
    assert quiet == ["/.bfx/bfx", "exec", "--json", "-"]
    assert "secret" not in " ".join(quiet)
    assert client.stdin_args_line({"env": {"K": "s"}}).endswith(b"\n")
    with pytest.raises(ValueError):
        client.one_shot_argv("version", bfx="bfx")


def test_parse_answer():
    assert client.parse_answer('{"ok": true, "result": {"a": 1}}\n') == {"a": 1}
    with pytest.raises(client.BfxError) as refused:
        client.parse_answer(
            b'{"ok": false, "error": {"kind": "link", "message": "x", "errno": 40}}'
        )
    assert (refused.value.kind, refused.value.errno) == ("link", 40)
    for bad in (
        "",
        "{}",
        "not json",
        '{"ok": true}',
        '{"ok":true,"result":{}}\n{"ok":true,"result":{}}',
    ):
        with pytest.raises(client.BfxProtocolError):
            client.parse_answer(bad)


def test_docker_delivery_choice(monkeypatch):
    assert delivery.docker_mode(local_daemon=True, environ={}) == "bind"
    assert delivery.docker_mode(local_daemon=False, environ={}) == "volume"
    assert (
        delivery.docker_mode(
            local_daemon=True, environ={delivery.DELIVERY_ENV: "volume"}
        )
        == "volume"
    )
    # A remote daemon cannot bind this machine's folders.
    assert (
        delivery.docker_mode(
            local_daemon=False, environ={delivery.DELIVERY_ENV: "bind"}
        )
        == "volume"
    )
    assert (
        delivery.docker_mode(local_daemon=True, environ={delivery.DELIVERY_ENV: "off"})
        is None
    )
    with pytest.raises(ValueError):
        delivery.docker_mode(local_daemon=True, environ={delivery.DELIVERY_ENV: "copy"})
    assert delivery.GUARANTEE == {
        "bind": "any-agent",
        "volume": "any-agent",
        "upload": "non-root-agent",
    }


def test_compose_overlays_mount_read_only_and_never_copy_from_the_image():
    bind = json.loads(delivery.compose_overlay("bind", source="/host/bfx"))
    assert bind == {"services": {"main": {"volumes": [
        {"type": "bind", "source": "/host/bfx", "target": "/.bfx", "read_only": True}]}}}  # fmt: skip
    name = delivery.volume_name("x86_64")
    assert re.fullmatch(r"benchflow-bfx-[0-9a-f]{12}", name)
    vol = json.loads(delivery.compose_overlay("volume", source=name))
    (mount,) = vol["services"]["main"]["volumes"]
    assert mount == {
        "type": "volume",
        "source": name,
        "target": "/.bfx",
        "read_only": True,
        "volume": {"nocopy": True},
    }
    assert vol["volumes"] == {name: {"external": True, "name": name}}
    assert delivery.run_args("volume", source=name)[1].endswith(
        ",readonly,volume-nocopy"
    )


def test_remote_admission_takes_only_the_read_only_bfx_volume():
    from benchflow.sandbox.remote_docker import _host_reach

    name = delivery.volume_name("x86_64")
    ours = json.loads(delivery.compose_overlay("volume", source=name))
    assert _host_reach(ours, bfx_volumes={name}) == []
    # Only the volume this start delivered: not by its name's form alone.
    assert _host_reach(ours)
    older = json.loads(
        delivery.compose_overlay("volume", source="benchflow-bfx-" + "0" * 12)
    )
    assert _host_reach(older, bfx_volumes={name})
    writable = json.loads(json.dumps(ours))
    writable["services"]["main"]["volumes"][0]["read_only"] = False
    assert _host_reach(writable, bfx_volumes={name})
    elsewhere = json.loads(json.dumps(ours))
    elsewhere["services"]["side"] = {
        "volumes": [{"type": "volume", "source": name, "target": "/data"}]
    }
    assert _host_reach(elsewhere, bfx_volumes={name})
    other = {"volumes": {"benchflow-bfx-zz": {"external": True}}}
    assert _host_reach(other, bfx_volumes={name})


def test_snapshot_replay_keeps_the_bfx_volume_read_only_and_uncopied():
    from benchflow.sandbox.docker import _replayed_run_args

    name = delivery.volume_name("x86_64")
    container = {
        "HostConfig": {
            "NetworkMode": "proj_default",
            "Mounts": [{"Type": "volume", "Source": name, "Target": "/.bfx", "ReadOnly": True,
                        "VolumeOptions": {"NoCopy": True}}],
        },
        "Mounts": [{"Type": "volume", "Name": name, "Destination": "/.bfx", "RW": False}],
        "NetworkSettings": {"Networks": {"proj_default": {}}},
    }  # fmt: skip
    args = _replayed_run_args(container, default_network="proj_default")
    assert f"type=volume,src={name},dst=/.bfx,readonly,volume-nocopy" in args


# ------------------------------------------------------------- the protocol


def _real_bfx() -> str | None:
    if not sys.platform.startswith("linux"):
        return None
    try:
        return str(locate.verified_binary(platform.machine()))
    except locate.BfxUnavailable:
        return None


needs_bfx = pytest.mark.skipif(
    _real_bfx() is None, reason="no pinned bfx for this machine"
)


@needs_bfx
@pytest.mark.asyncio
async def test_serve_reports_its_pin_and_protocol_and_acts_beneath_a_root(tmp_path):
    serve = client.BfxServe([_real_bfx(), "serve"])
    version = await serve.start()
    try:
        assert version["protocol"] == client.PROTOCOL
        assert (
            version["sha256"] == pins.SHA256[locate.normalize_arch(platform.machine())]
        )
        made = await serve.call(
            "write", {"root": str(tmp_path), "path": "f", "data": "x", "mode": 0o600}
        )
        assert made["size"] == 1
        (tmp_path / "l").symlink_to("/etc")
        with pytest.raises(client.BfxError) as refused:
            await serve.call("stat", {"root": str(tmp_path / "l"), "paths": ["passwd"]})
        assert refused.value.kind == "link"
    finally:
        await serve.close()


@needs_bfx
def test_serve_journals_each_start_and_refuses_late_and_withdrawn_requests(tmp_path):
    """The wire, as bfx speaks it: a request with no time left and one
    withdrawn by a cancel sent with it never start (no journal entry, no
    started line); the next request starts and is journalled first."""
    import subprocess

    journal = tmp_path / "journal"
    lines = [
        {
            "id": 1,
            "v": 1,
            "cmd": "mkdir",
            "args": {"root": str(tmp_path), "path": "late"},
            "r": 0,
        },
        {
            "id": 2,
            "v": 1,
            "cmd": "mkdir",
            "args": {"root": str(tmp_path), "path": "withdrawn"},
        },
        {"cancel": 2},
        {
            "id": 3,
            "v": 1,
            "cmd": "mkdir",
            "args": {"root": str(tmp_path), "path": "made"},
        },
    ]
    done = subprocess.run(
        [_real_bfx(), "serve", "--journal", str(journal)],
        input="".join(json.dumps(x) + "\n" for x in lines).encode(),
        capture_output=True,
        timeout=30,
    )
    out = [json.loads(x) for x in done.stdout.splitlines()]
    assert (
        out[0]["id"] == 1
        and out[0]["started"] is False
        and out[0]["error"]["kind"] == "deadline"
    )
    assert (
        out[1]["id"] == 2
        and out[1]["started"] is False
        and out[1]["error"]["kind"] == "cancelled"
    )
    assert out[2] == {"id": 3, "started": True}
    assert out[3]["id"] == 3 and out[3]["ok"] is True
    assert journal.read_text() == "started 3\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["journal", "made"]


@pytest.mark.asyncio
async def test_the_client_says_not_started_only_when_bfx_says_so(tmp_path):
    """A stand-in that refuses every request as bfx does after a deadline:
    BfxNotStarted, safe to run another way."""
    script = tmp_path / "standin.py"
    script.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    r = json.loads(line)\n"
        "    if r.get('cmd') == 'version':\n"
        "        a = {'id': r['id'], 'ok': True, 'result': {'protocol': 1}}\n"
        "    else:\n"
        "        a = {'id': r['id'], 'ok': False, 'started': False,\n"
        "             'error': {'kind': 'deadline', 'message': 'late'}}\n"
        "    print(json.dumps(a), flush=True)\n"
    )
    serve = client.BfxServe([sys.executable, str(script)])
    await serve.start()
    try:
        with pytest.raises(client.BfxNotStarted) as late:
            await serve.call("mkdir", {"root": "/", "path": "x"})
        assert late.value.kind == "deadline"
    finally:
        await serve.close()


@needs_bfx
@pytest.mark.asyncio
async def test_a_channel_that_dies_mid_request_is_ambiguous_never_retried(tmp_path):
    serve = client.BfxServe([_real_bfx(), "serve"])
    await serve.start()
    mark = tmp_path / "ran"
    call = asyncio.ensure_future(
        serve.call(
            "exec",
            {"argv": ["/bin/sh", "-c", f": > {mark}; sleep 5"], "uid": os.getuid(), "gid": os.getgid(),
             "groups": sorted(set(os.getgroups())), "caps": "keep", "nnp": True},
        )
    )  # fmt: skip
    for _ in range(100):
        if mark.exists():
            break
        await asyncio.sleep(0.05)
    assert serve._process is not None
    serve._process.kill()
    with pytest.raises(client.BfxAmbiguous) as died:
        await call
    assert died.value.started is True


@needs_bfx
def test_args_on_stdin_never_reach_argv(tmp_path):
    import subprocess

    argv = client.one_shot_argv("write", args_on_stdin=True, bfx=_real_bfx())
    line = client.stdin_args_line(
        {"root": str(tmp_path), "path": "s", "data": "secret-value"}
    )
    done = subprocess.run(argv, input=line, capture_output=True, timeout=30)
    assert client.parse_answer(done.stdout)["size"] == len("secret-value")
    assert (tmp_path / "s").read_text() == "secret-value"


# ------------------------------------------------ tar-in ownership (v2 #12)


def test_agent_derived_tar_in_always_refuses_root_ownership():
    args = client.tar_in_args("/w", uid=1000, gid=1000)
    assert args == {"root": "/w", "uid": 1000, "gid": 1000, "refuse_root_owner": True}
    for uid, gid in ((0, 0), (0, 1000), (1000, 0)):
        with pytest.raises(ValueError, match="never extracted as root"):
            client.tar_in_args("/w", uid=uid, gid=gid)
    with pytest.raises(ValueError):
        client.tar_in_args("/w", uid=1000, gid=1000, refuse_root_owner=False)
    # Only BenchFlow's own bytes may go without the option.
    own = client.tar_in_args("/w", uid=0, gid=0, agent_derived=False, mtime=False)
    assert own == {"root": "/w", "uid": 0, "gid": 0, "mtime": False}


def _setid_archive() -> bytes:
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name, mode, data in (("su", 0o4755, b"x"), ("sg", 0o2755, b"y")):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.uid, info.gid = len(data), mode, 0, 0
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


@needs_bfx
@pytest.mark.asyncio
async def test_bfx_refuses_root_ownership_and_drops_set_id_bits(tmp_path):
    import base64

    if os.getuid() == 0:
        pytest.skip("runs as a non-root account (its own ids are the owner)")
    data = base64.b64encode(_setid_archive()).decode()
    serve = client.BfxServe([_real_bfx(), "serve"])
    await serve.start()
    try:
        for owner in ({"uid": 0, "gid": 0}, {"uid": os.getuid(), "gid": 0}, {}):
            with pytest.raises(client.BfxError) as refused:
                await serve.call(
                    "tar-in",
                    {
                        "root": str(tmp_path),
                        "refuse_root_owner": True,
                        "data_b64": data,
                        **owner,
                    },
                )
            assert refused.value.kind == "root-owner", owner
            assert list(tmp_path.iterdir()) == []
        with pytest.raises(client.BfxError) as refused:
            await serve.call(
                "tar-in", {"root": str(tmp_path), "mode_mask": 0o7777, "data_b64": data}
            )
        assert refused.value.kind == "bad-request"
        args = client.tar_in_args(str(tmp_path), uid=os.getuid(), gid=os.getgid())
        await serve.call("tar-in", {**args, "data_b64": data})
        for name in ("su", "sg"):
            st = (tmp_path / name).lstat()
            assert st.st_mode & 0o7000 == 0, oct(st.st_mode)
            assert (st.st_uid, st.st_gid) == (os.getuid(), os.getgid())
    finally:
        await serve.close()
