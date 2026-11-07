"""bfx delivery on the providers (TRUSTED-RUNTIME.md §6): Modal's read-only
Volume, Daytona DinD's harness volume through the VM's control path, and
the root-only upload on native Daytona and AgentCore.

None of these providers is reachable from the build VMs, so every case here
runs on fakes of the provider's SDK or exec path: they check what BenchFlow
asks the provider for, in which order, and what it accepts back. Until a
provider's delivery passes a live run it is off unless
``BENCHFLOW_BFX_PROVIDERS=on`` (``delivery.PROVEN_LIVE``), and the backend
keeps the kit path.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow._bfx import delivery, locate, pins
from benchflow._bfx.locate import SANDBOX_DIR, SANDBOX_PATH, BfxUnavailable

STAND_IN = {"x86_64": b"bfx stand-in x86_64", "aarch64": b"bfx stand-in aarch64"}


@pytest.fixture
def stand_in(tmp_path, monkeypatch):
    """Pinned stand-in binaries for both architectures, and a private cache."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for arch, data in STAND_IN.items():
        (bindir / f"bfx-{arch}").write_bytes(data)
        monkeypatch.setitem(pins.SHA256, arch, hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(locate, "BIN_DIR", bindir)
    monkeypatch.setattr(locate, "_checked", {})
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv(delivery.DELIVERY_ENV, raising=False)
    monkeypatch.delenv(delivery.PROVIDER_ENV, raising=False)
    monkeypatch.setattr(delivery, "_verified_volumes", set())
    return bindir


@pytest.fixture
def providers_on(monkeypatch):
    monkeypatch.setenv(delivery.PROVIDER_ENV, "on")


def _version_line(arch: str) -> str:
    return json.dumps({"name": "bfx", "sha256": pins.SHA256[arch], "arch": arch})


# ------------------------------------------------------------- the switch


@pytest.mark.parametrize("backend", sorted(delivery.PROVIDER_DELIVERY))
def test_providers_not_proven_live_deliver_only_when_switched_on(backend):
    assert backend not in delivery.PROVEN_LIVE
    assert delivery.provider_enabled(backend, {}) is False
    assert delivery.provider_enabled(backend, {delivery.PROVIDER_ENV: "on"}) is True
    off = {delivery.PROVIDER_ENV: "on", delivery.DELIVERY_ENV: "off"}
    assert delivery.provider_enabled(backend, off) is False


def test_each_provider_has_its_delivery_and_guarantee():
    assert delivery.PROVIDER_DELIVERY == {
        "modal": "volume",
        "daytona-dind": "volume",
        "daytona": "upload",
        "agentcore": "upload",
    }
    assert delivery.GUARANTEE["volume"] == "any-agent"
    assert delivery.GUARANTEE["upload"] == "non-root-agent"
    with pytest.raises(ValueError):
        delivery.provider_enabled("apple", {})


# ------------------------------------------------------------------ Modal


@pytest.mark.parametrize(
    "version,ok",
    [
        ("1.0.5", True),
        ("1.4.2", True),
        ("2.0", True),
        ("1.0.4", False),
        ("0.73.1", False),
        ("1.0.5rc1", True),
        ("", False),
        ("dev", False),
    ],
)
def test_modal_needs_read_only_volumes(version, ok):
    assert delivery.modal_sdk_supported(version) is ok


class FakeVolume:
    def __init__(self, files: dict[str, tuple[bytes, int]], *, corrupt=False):
        self.files = files
        self.corrupt = corrupt
        self.uploads: list[tuple[str, str, int, bool]] = []
        self.read_only_view = object()

    def read_file_into_fileobj(self, path, fileobj):
        if path not in self.files:
            raise FileNotFoundError(path)
        fileobj.write(self.files[path][0])

    def batch_upload(self, force=False):
        volume = self

        class Batch:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def put_file(self, local, remote, mode=None):
                data = Path(local).read_bytes()
                volume.uploads.append((Path(local).name, remote, mode, force))
                if volume.corrupt:
                    data = b"other"
                volume.files[remote.lstrip("/")] = (data, mode)

        return Batch()

    def read_only(self):
        return self.read_only_view


def _fake_modal(volume: FakeVolume, version="1.4.2"):
    names = []

    class Volume:
        @staticmethod
        def from_name(name, create_if_missing=False):
            names.append((name, create_if_missing))
            return volume

    return SimpleNamespace(__version__=version, Volume=Volume), names


def test_modal_fills_an_empty_volume_with_the_pinned_binary(stand_in):
    volume = FakeVolume({})
    modal, names = _fake_modal(volume)
    assert delivery.modal_volume(modal) is volume.read_only_view
    assert names == [(delivery.volume_name("x86_64"), True)]
    assert volume.uploads == [("bfx", "/bfx", 0o555, False)]
    assert volume.files["bfx"][0] == STAND_IN["x86_64"]


def test_modal_reuses_a_volume_that_already_holds_the_pin(stand_in):
    volume = FakeVolume({"bfx": (STAND_IN["x86_64"], 0o555)})
    modal, _ = _fake_modal(volume)
    assert delivery.modal_volume(modal) is volume.read_only_view
    assert volume.uploads == []


def test_modal_refuses_a_volume_that_does_not_read_back_as_the_pin(stand_in):
    volume = FakeVolume({}, corrupt=True)
    modal, _ = _fake_modal(volume)
    with pytest.raises(BfxUnavailable, match="not the pinned"):
        delivery.modal_volume(modal)


def test_modal_refuses_and_never_refills_a_volume_whose_bfx_differs(stand_in):
    """L-7: one rule with Docker's volume."""
    volume = FakeVolume({"bfx": (b"stale", 0o555)})
    modal, _ = _fake_modal(volume)
    with pytest.raises(BfxUnavailable, match="never used or refilled"):
        delivery.modal_volume(modal)
    assert volume.uploads == [] and volume.files["bfx"][0] == b"stale"


def test_modal_below_the_minimum_sdk_delivers_nothing(stand_in):
    volume = FakeVolume({})
    modal, names = _fake_modal(volume, version="1.0.4")
    with pytest.raises(BfxUnavailable, match=r"1\.0\.5"):
        delivery.modal_volume(modal)
    assert names == []


@pytest.mark.asyncio
async def test_modal_uses_the_volume_only_after_bfx_reports_the_pin(stand_in):
    from benchflow.sandbox.modal_impl import ModalSandbox

    calls = []

    def sandbox_with(output: str, code: int = 0):
        class Process:
            stdout = SimpleNamespace(read=SimpleNamespace(aio=lambda: _value(output)))

            class wait:
                @staticmethod
                async def aio():
                    return code

        async def exec_aio(*argv, **kwargs):
            calls.append((argv, kwargs))
            return Process()

        box = object.__new__(ModalSandbox)
        box.logger = logging.getLogger("test")
        box.bfx_delivery = None
        box._sandbox = SimpleNamespace(exec=SimpleNamespace(aio=exec_aio))
        return box

    good = sandbox_with(_version_line("x86_64") + "\n")
    await good._check_bfx()
    assert good.bfx_delivery == "volume"
    assert calls[0][0] == (SANDBOX_PATH, "version")
    assert "secrets" not in calls[0][1]

    wrong = sandbox_with(json.dumps({"name": "bfx", "sha256": "0" * 64}))
    await wrong._check_bfx()
    assert wrong.bfx_delivery is None


async def _value(v):
    return v


# ----------------------------------------------------------------- upload


class UploadTarget:
    """A sandbox's root shell and upload, recorded."""

    def __init__(self, *, arch="x86_64", fail=None, report=None):
        self.arch = arch
        self.fail = fail or set()
        self.report = report
        self.steps: list[str] = []
        self.uploaded: list[tuple[bytes, str]] = []

    async def run_root(self, command):
        if command == "uname -m":
            self.steps.append("arch")
            return (1, "") if "arch" in self.fail else (0, self.arch + "\n")
        if command == delivery.upload_prepare_command():
            self.steps.append("prepare")
            return (1, "exists") if "prepare" in self.fail else (0, "")
        if command == delivery.upload_finish_command():
            self.steps.append("finish")
            if "finish" in self.fail:
                return 1, "not sealed"
            return 0, (self.report or _version_line(self.arch)) + "\n"
        raise AssertionError(command)

    async def upload(self, source, target, **_kwargs):
        self.steps.append("upload")
        if "upload" in self.fail:
            raise OSError("provider refused the upload")
        self.uploaded.append((Path(source).read_bytes(), target))


@pytest.mark.asyncio
@pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
async def test_upload_makes_the_folder_then_uploads_then_seals_and_checks(
    stand_in, arch
):
    target = UploadTarget(arch=arch)
    assert await delivery.deliver_by_upload(target.upload, target.run_root) == arch
    assert target.steps == ["arch", "prepare", "upload", "finish"]
    assert target.uploaded == [(STAND_IN[arch], SANDBOX_PATH)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fail,steps",
    [
        ({"arch"}, ["arch"]),
        ({"prepare"}, ["arch", "prepare"]),
        ({"upload"}, ["arch", "prepare", "upload"]),
        ({"finish"}, ["arch", "prepare", "upload", "finish"]),
    ],
)
async def test_upload_stops_at_the_first_failed_step(stand_in, fail, steps):
    target = UploadTarget(fail=fail)
    with pytest.raises(BfxUnavailable):
        await delivery.deliver_by_upload(target.upload, target.run_root)
    assert target.steps == steps


@pytest.mark.asyncio
async def test_upload_refuses_a_binary_that_reports_another_hash(stand_in):
    target = UploadTarget(report=json.dumps({"name": "bfx", "sha256": "1" * 64}))
    with pytest.raises(BfxUnavailable, match="pinned"):
        await delivery.deliver_by_upload(target.upload, target.run_root)


@pytest.mark.asyncio
async def test_deliver_upload_sets_the_flag_only_when_on_and_done(
    stand_in, monkeypatch
):
    box = SimpleNamespace(bfx_delivery="stale", logger=logging.getLogger("test"))
    target = UploadTarget()
    await delivery.deliver_upload(box, "daytona", target.upload, target.run_root)
    assert box.bfx_delivery is None and target.steps == []

    monkeypatch.setenv(delivery.PROVIDER_ENV, "on")
    await delivery.deliver_upload(box, "daytona", target.upload, target.run_root)
    assert box.bfx_delivery == "upload"

    failing = UploadTarget(fail={"prepare"})
    await delivery.deliver_upload(box, "agentcore", failing.upload, failing.run_root)
    assert box.bfx_delivery is None


def _prepare_in(folder: Path) -> subprocess.CompletedProcess:
    command = delivery.upload_prepare_command().replace(SANDBOX_DIR, str(folder))
    return subprocess.run(["/bin/sh", "-c", command], capture_output=True)


def test_the_prepare_step_makes_a_new_private_folder(tmp_path):
    if Path("/").stat().st_uid != 0:
        pytest.skip("/ is not root's on this machine")
    made = _prepare_in(tmp_path / "bfx")
    assert made.returncode == 0, made.stderr
    assert (tmp_path / "bfx").is_dir()
    assert (tmp_path / "bfx").stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("kind", ["folder", "file", "link", "dangling-link"])
def test_the_prepare_step_refuses_a_path_that_already_exists(tmp_path, kind):
    if Path("/").stat().st_uid != 0:
        pytest.skip("/ is not root's on this machine")
    there = tmp_path / "bfx"
    if kind == "folder":
        there.mkdir()
    elif kind == "file":
        there.write_bytes(b"")
    elif kind == "link":
        (tmp_path / "elsewhere").mkdir()
        there.symlink_to(tmp_path / "elsewhere")
    else:
        there.symlink_to(tmp_path / "missing")
    assert _prepare_in(there).returncode != 0
    if kind == "link":
        assert list((tmp_path / "elsewhere").iterdir()) == []


def test_the_finish_step_checks_the_folder_and_file_before_sealing():
    finish = delivery.upload_finish_command()
    check = finish.index('[ ! -L "$d" ]')
    assert check < finish.index("chown 0:0")
    assert finish.index('[ "$(ls -A "$d")" = bfx ]') < finish.index(
        f"{SANDBOX_PATH} version"
    )


@pytest.mark.asyncio
async def test_native_daytona_uploads_as_root_on_the_system_shell(
    stand_in, providers_on, monkeypatch
):
    from benchflow.sandbox import daytona_strategies

    route = object()

    async def system_route(env):
        return route

    monkeypatch.setattr(daytona_strategies, "system_shell_route", system_route)
    target = UploadTarget()
    seen = []

    async def sandbox_exec(command, user=None, timeout_sec=None, route=None, **kw):
        seen.append((user, route))
        code, out = await target.run_root(command)
        return SimpleNamespace(return_code=code, stdout=out, stderr="")

    env = SimpleNamespace(
        bfx_delivery=None,
        logger=logging.getLogger("test"),
        _sandbox_exec=sandbox_exec,
        _sdk_upload_file=target.upload,
    )
    await daytona_strategies._deliver_bfx(env)
    assert env.bfx_delivery == "upload"
    assert seen and all(s == ("root", route) for s in seen)
    assert target.uploaded == [(STAND_IN["x86_64"], SANDBOX_PATH)]


@pytest.mark.asyncio
async def test_agentcore_uploads_over_its_channel_as_root(stand_in, providers_on):
    from benchflow.sandbox.agentcore import AgentCoreSandbox

    target = UploadTarget(arch="aarch64")
    seen = []

    async def exec_(command, user=None, timeout_sec=None, system_shell=False, **kw):
        seen.append((user, system_shell))
        code, out = await target.run_root(command)
        return SimpleNamespace(return_code=code, stdout=out, stderr="")

    box = object.__new__(AgentCoreSandbox)
    box.logger = logging.getLogger("test")
    box.bfx_delivery = None
    box.exec = exec_
    box.upload_file = target.upload
    await box._deliver_bfx()
    assert box.bfx_delivery == "upload"
    assert box.bfx_guarantee == "non-root-agent"
    assert seen and all(s == ("root", True) for s in seen)
    assert target.uploaded == [(STAND_IN["aarch64"], SANDBOX_PATH)]


# ------------------------------------------------------------------- DinD


class FakeVM:
    """The DinD VM's exec and upload: an inner daemon with volumes and
    created containers, and a VM folder tree."""

    def __init__(self, *, arch="x86_64"):
        self.arch = arch
        self.images: dict[str, tuple[bytes, str]] = {}
        self.created: dict[str, tuple[str, str]] = {}
        self.commands: list[str] = []
        self.uploads: list[tuple[str, bytes]] = []
        self.volumes: dict[str, bytes | None] = {}
        self.labels: dict[str, dict[str, str]] = {}
        self.extra: dict[str, list[str]] = {}
        self.staged: dict[str, bytes] = {}
        self.populated = 0

    async def vm_exec(self, command, cwd=None, env=None, timeout_sec=None, **kw):
        import shlex

        self.commands.append(command)
        ok = SimpleNamespace(return_code=0, stdout="", stderr="")
        if command.startswith("docker info --format '{{.Architecture}}'"):
            return SimpleNamespace(return_code=0, stdout=self.arch + "\n", stderr="")
        if command.startswith("d=$(mktemp -d) && docker cp ") and '"$d/v"' in command:
            name = shlex.split(command.split("docker cp ", 1)[1])[0].split(":")[0]
            volume = self.created[name][0]
            names = (["bfx"] if self.volumes.get(volume) is not None else []) + (
                self.extra.get(volume, [])
            )
            return SimpleNamespace(return_code=0, stdout="\n".join(names), stderr="")
        if command.startswith("d=$(mktemp -d) && docker cp "):
            source = shlex.split(command.split("docker cp ", 1)[1])[0]
            name, _, path = source.partition(":")
            volume, image = self.created[name]
            data = self.images[image][0] if path == "/bfx" else self.volumes.get(volume)
            if data is None:
                return SimpleNamespace(return_code=1, stdout="", stderr="no file")
            return SimpleNamespace(
                return_code=0,
                stdout=f"{hashlib.sha256(data).hexdigest()}  /tmp/x/f\n",
                stderr="",
            )
        argv = shlex.split(command)
        if argv[:2] == ["docker", "import"]:
            with tarfile.open(fileobj=io.BytesIO(self.staged[argv[-1]])) as tar:
                [member] = tar.getmembers()
                assert (member.name, member.mode) == ("bfx", 0o555)
                data = tar.extractfile(member).read()
            label = next(c.split("=", 1)[1] for c in argv if c.startswith("LABEL "))
            image = "sha256:" + hashlib.sha256(label.encode()).hexdigest()
            self.images[image] = (data, label)
            return SimpleNamespace(return_code=0, stdout=image + "\n", stderr="")
        if argv[:3] == ["docker", "image", "ls"]:
            want = argv[-1].split("=", 2)[2]
            found = [i for i, (_, label) in self.images.items() if label == want]
            return SimpleNamespace(return_code=0, stdout="\n".join(found), stderr="")
        if argv[:3] == ["docker", "rmi", "--force"]:
            for image in argv[3:]:
                self.images.pop(image, None)
            return ok
        if argv[:2] == ["mkdir", "-m"] or argv[:2] in (
            ["rm", "-rf"],
            ["chmod", "0555"],
        ):
            return ok
        if argv[:3] == ["docker", "volume", "create"]:
            self.volumes.setdefault(argv[-1], None)
            key, _, value = argv[argv.index("--label") + 1].partition("=")
            self.labels.setdefault(argv[-1], {key: value})
            return ok
        if argv[:3] == ["docker", "volume", "inspect"]:
            info = {
                "Name": argv[-1],
                "Driver": "local",
                "Options": None,
                "Labels": self.labels.get(argv[-1], {}),
            }
            return SimpleNamespace(return_code=0, stdout=json.dumps(info), stderr="")
        if argv[:2] == ["docker", "create"]:
            name = argv[argv.index("--name") + 1]
            mount = argv[argv.index("--mount") + 1]
            volume = mount.split("src=")[1].split(",")[0]
            image = argv[argv.index("--mount") + 2]
            # Only BenchFlow's own helper image, never pulled.
            assert image in self.images and "never" in argv
            self.created[name] = (volume, image)
            return ok
        if argv[:3] == ["docker", "start", "-a"]:
            volume, image = self.created[argv[3]]
            self.volumes[volume] = self.images[image][0]
            self.populated += 1
            return ok
        if argv[:3] == ["docker", "rm", "-f"]:
            return ok
        raise AssertionError(command)

    async def upload(self, source, target, **_kwargs):
        data = Path(source).read_bytes()
        self.uploads.append((target, data))
        self.staged[target] = data


def _dind(vm: FakeVM):
    from benchflow.sandbox.daytona_dind import _DaytonaDinD

    strategy = object.__new__(_DaytonaDinD)
    strategy._vm_exec = vm.vm_exec
    strategy._bfx_compose_path = None
    env = SimpleNamespace(
        bfx_delivery=None,
        session_id="trial-1",
        logger=logging.getLogger("test"),
        _sdk_upload_file=vm.upload,
    )
    strategy._env = env
    return strategy, env


@pytest.mark.asyncio
async def test_dind_delivers_nothing_unless_switched_on(stand_in):
    vm = FakeVM()
    strategy, env = _dind(vm)
    await strategy._prepare_bfx()
    assert env.bfx_delivery is None and strategy._bfx_compose_path is None
    assert vm.commands == []


@pytest.mark.asyncio
async def test_dind_fills_the_inner_volume_through_the_vm_and_mounts_it(
    stand_in, providers_on
):
    vm = FakeVM()
    strategy, env = _dind(vm)
    await strategy._prepare_bfx()
    volume = delivery.volume_name("x86_64")
    assert env.bfx_delivery == "volume"
    assert vm.volumes[volume] == STAND_IN["x86_64"]
    assert vm.populated == 1
    # The binary reached the VM by upload, never as command text.
    assert [t for t, _ in vm.uploads if t.endswith("/bfx.tar")]
    assert all(STAND_IN["x86_64"].decode() not in c for c in vm.commands)
    # The overlay: main only, read-only, never copied from the image.
    overlay_target, overlay = vm.uploads[-1]
    assert overlay_target == strategy._bfx_compose_path
    document = json.loads(overlay)
    [mount] = document["services"]["main"]["volumes"]
    assert mount == {
        "type": "volume",
        "source": volume,
        "target": SANDBOX_DIR,
        "read_only": True,
        "volume": {"nocopy": True},
    }
    assert list(document["services"]) == ["main"]
    # Every staging folder on the VM is removed.
    staged = [c for c in vm.commands if c.startswith("mkdir -m 0700 ")]
    assert staged and all(
        any(c == f"rm -rf {s.split()[-1]}" for c in vm.commands) for s in staged
    )


@pytest.mark.asyncio
async def test_dind_reuses_a_volume_that_already_holds_the_pin(stand_in, providers_on):
    vm = FakeVM()
    vm.volumes[delivery.volume_name("aarch64")] = STAND_IN["aarch64"]
    vm.arch = "aarch64"
    strategy, env = _dind(vm)
    await strategy._prepare_bfx()
    assert env.bfx_delivery == "volume" and vm.populated == 0


@pytest.mark.asyncio
async def test_dind_refuses_a_volume_that_does_not_read_back_as_the_pin(
    stand_in, providers_on
):
    vm = FakeVM()
    strategy, env = _dind(vm)
    original = vm.vm_exec

    async def stale_fill(command, **kw):
        result = await original(command, **kw)
        if command.startswith("docker start -a"):
            vm.volumes[delivery.volume_name("x86_64")] = b"stale"
        return result

    strategy._vm_exec = stale_fill
    await strategy._prepare_bfx()
    assert env.bfx_delivery is None and strategy._bfx_compose_path is None


@pytest.mark.asyncio
async def test_dind_refuses_and_never_refills_a_volume_whose_bfx_differs(
    stand_in, providers_on
):
    vm = FakeVM()
    volume = delivery.volume_name("x86_64")
    vm.volumes[volume] = b"other bytes"
    vm.labels[volume] = {delivery.VOLUME_LABEL: pins.SHA256["x86_64"]}
    strategy, env = _dind(vm)
    await strategy._prepare_bfx()
    assert env.bfx_delivery is None and vm.populated == 0
    assert vm.volumes[volume] == b"other bytes"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "labels",
    [{}, {delivery.VOLUME_LABEL: "0" * 64}, "compose"],
    ids=["unlabelled", "other-pin", "compose-project"],
)
async def test_dind_never_uses_a_volume_of_that_name_it_does_not_own(
    stand_in, providers_on, labels
):
    vm = FakeVM()
    volume = delivery.volume_name("x86_64")
    if labels == "compose":
        labels = {
            delivery.VOLUME_LABEL: pins.SHA256["x86_64"],
            "com.docker.compose.project": "task",
        }
    vm.volumes[volume] = STAND_IN["x86_64"]
    vm.labels[volume] = labels
    strategy, env = _dind(vm)
    await strategy._prepare_bfx()
    assert env.bfx_delivery is None and vm.populated == 0


@pytest.mark.asyncio
async def test_dind_refuses_a_volume_holding_other_files(stand_in, providers_on):
    vm = FakeVM()
    volume = delivery.volume_name("x86_64")
    vm.extra[volume] = ["task-file"]
    strategy, env = _dind(vm)
    await strategy._prepare_bfx()
    assert env.bfx_delivery is None and vm.populated == 0


@pytest.mark.asyncio
async def test_dind_uses_no_task_image_and_removes_its_helper_image(
    stand_in, providers_on
):
    vm = FakeVM()
    strategy, env = _dind(vm)
    await strategy._prepare_bfx()
    assert env.bfx_delivery == "volume" and vm.populated == 1
    # Every container was of the helper image (FakeVM asserts it), and the
    # helper image is gone at the end.
    assert vm.created and vm.images == {}


def test_dind_compose_files_end_with_the_bfx_overlay_before_the_bootstrap():
    from benchflow.sandbox.daytona_dind import _DaytonaDinD

    flags_source = Path(_DaytonaDinD._compose_file_flags.__code__.co_filename)
    text = flags_source.read_text()
    body = text[text.index("    def _compose_file_flags") :]
    body = body[: body.index("        return flags")]
    assert body.index("_bfx_compose_path") < body.index("_bootstrap_override")
