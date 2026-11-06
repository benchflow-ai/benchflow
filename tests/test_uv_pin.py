"""The pinned uv and managed Python BenchFlow installs in sandboxes (round PY8, U1).

Guards the fix for F3-4: the in-sandbox LiteLLM proxy install curled the
latest uv and ran ``uv venv`` with no Python named, so on images without
Python uv 0.13 picked CPython 3.15, for which fastuuid (a LiteLLM 1.91
dependency) has no wheel, and every model trial on TB2's 34 images without
Python failed before the agent started. The install now fetches uv 0.13.4 by
version, checks its published sha256 before uv runs, and builds the venv on
a uv-managed CPython 3.12.

The shell tests run the generated commands for real with ``sh`` against a
fake ``curl`` and a fake ``uv`` inside a temporary folder: no network, and
nothing outside ``tmp_path`` is written.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow import _uv_pin
from benchflow._uv_pin import (
    PYTHON_INSTALL_DIR,
    PYTHON_VERSION,
    UV_DIR,
    UV_SHA256,
    UV_VERSION,
    uv_install_command,
)
from benchflow.agents.registry import AGENTS
from benchflow.providers import litellm_runtime as runtime_mod
from tests._root_stand_in import run_sh

REPO_SRC = Path(__file__).resolve().parents[1] / "src" / "benchflow"
TARGETS = (
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "x86_64-unknown-linux-musl",
    "aarch64-unknown-linux-musl",
)
needs_sh = pytest.mark.skipif(
    shutil.which("sh") is None or shutil.which("tar") is None,
    reason="needs a POSIX sh and tar",
)


class _Result:
    def __init__(self, return_code: int, stdout: str = "", stderr: str = ""):
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr


class _RecordingSandbox:
    """Records every command and answers success, without running anything."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    async def exec(self, command: str, timeout_sec=None, user=None) -> _Result:
        self.commands.append(command)
        return _Result(0)


# --------------------------------------------------------------------------
# The pins themselves
# --------------------------------------------------------------------------


def test_pin_is_uv_0_13_4_with_a_sum_per_linux_target():
    """Install the patched uv release with all Linux checksums."""
    assert UV_VERSION == "0.13.4"
    assert set(UV_SHA256) == set(TARGETS)
    for digest in UV_SHA256.values():
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert len(set(UV_SHA256.values())) == len(TARGETS)
    assert UV_DIR == "/opt/benchflow/uv/0.13.4"
    assert PYTHON_VERSION == "3.12.15"
    assert PYTHON_INSTALL_DIR == "/opt/benchflow/python"
    assert _uv_pin.MANAGED_PYTHON_ENV["UV_PYTHON_PREFERENCE"] == "only-managed"


# --------------------------------------------------------------------------
# The generated proxy install command
# --------------------------------------------------------------------------


async def _proxy_install_command() -> str:
    sandbox = _RecordingSandbox()
    await runtime_mod._ensure_sandbox_litellm(
        sandbox,
        venv_dir="/tmp/benchflow-litellm/abc/venv",
        requirements_path="/tmp/benchflow-litellm/abc/litellm-proxy.txt",
    )
    (command,) = sandbox.commands
    return command


@pytest.mark.asyncio
async def test_proxy_install_pins_uv_checks_sum_before_uv_and_names_python():
    command = await _proxy_install_command()

    # uv 0.13.4 fetched by version, every published sum carried.
    assert f"/releases/download/{UV_VERSION}" in command
    for digest in UV_SHA256.values():
        assert digest in command
    assert "astral.sh/uv/install.sh" not in command
    # The image's own uv is never used.
    assert "command -v uv" not in command
    assert f"UV={UV_DIR}/uv" in command

    # The sum is compared, and the tarball unpacked, before uv first runs.
    check = command.index('if [ "$got" != "$want" ]')
    unpack = command.index('tar -xzf "$tmp/$file"')
    first_uv_run = command.index('"$UV" ')
    assert "sha256sum" in command
    assert check < unpack < first_uv_run

    # The venv is built on a uv-managed CPython 3.12 under /opt/benchflow.
    exports = command.index("export UV_PYTHON_PREFERENCE=only-managed")
    assert f"UV_PYTHON_INSTALL_DIR={PYTHON_INSTALL_DIR}" in command
    assert "UV_PYTHON_DOWNLOADS=automatic" in command
    python_install = command.index(
        f'"$UV" python install -q --no-bin --reinstall {PYTHON_VERSION}'
    )
    venv = command.index(f'"$UV" venv -q --python {PYTHON_VERSION} ')
    pip = command.index('"$UV" pip install --python ')
    assert check < exports < python_install < venv < pip
    # No fallback to the image's own Python.
    assert "-m venv" not in command
    assert "-m pip" not in command


def test_agent_installs_use_the_pinned_uv():
    pinned = uv_install_command()
    path = f'export PATH="{UV_DIR}:/opt/benchflow/bin:/opt/benchflow/agents/bin:$PATH"'
    for name, first_use in (
        ("deepagents", "uv venv -q --clear --python 3.12"),
        ("harvey-lab-harness", "uv venv -q --clear --python 3.12"),
        ("openhands", "uv tool install"),
    ):
        install = AGENTS[name].install_cmd
        assert pinned in install, name
        assert install.index(pinned) < install.index(path) < install.index(first_use)
        assert "astral.sh/uv/install.sh" not in install, name
        assert "command -v uv" not in install, name


def test_toolathlon_images_install_the_pinned_uv():
    from benchflow.adapters._toolathlon import _toolathlon_dockerfile

    ctx = SimpleNamespace(repo="example/toolathlon", resolved_sha="0" * 40)
    for variant in ("official", "gym"):
        dockerfile = _toolathlon_dockerfile(ctx, variant=variant)
        assert uv_install_command() in dockerfile, variant
        assert "astral.sh/uv/install.sh" not in dockerfile, variant
        # The image's own uv (official: one in /root/.local/bin) is never
        # what lands in /usr/local/bin.
        assert "command -v uv" not in dockerfile, variant
        assert f"cp {UV_DIR}/uv /usr/local/bin/uv" in dockerfile, variant
        assert f"cp {UV_DIR}/uvx /usr/local/bin/uvx" in dockerfile, variant


_UNPINNED_UV_INSTALL = re.compile(
    r"astral\.sh/uv/(?:[0-9.]+/)?install\.(?:sh|ps1)"
    r"|astral-sh/uv/releases"
    r"|\bpip3? install (?:-{1,2}[\w-]+ )*['\"]?uv(?:[=<>~!]|['\"]?(?:\s|$))"
)


def test_no_unpinned_uv_install_left_in_the_package():
    """Every uv fetch under src/ goes through benchflow._uv_pin."""
    offenders = []
    for path in sorted(REPO_SRC.rglob("*")):
        if not path.is_file() or path.suffix in {".pyc", ".png", ".jpg", ".woff2"}:
            continue
        if path.name == "_uv_pin.py":
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if _UNPINNED_UV_INSTALL.search(line):
                offenders.append(f"{path.relative_to(REPO_SRC)}:{number}: {line}")
    assert offenders == []


# --------------------------------------------------------------------------
# Running the generated shell (fake curl and fake uv, no network)
# --------------------------------------------------------------------------

_FAKE_UV = """#!/bin/sh
printf '%s|%s|%s|%s\\n' "$*" "${UV_PYTHON_PREFERENCE:-}" \\
  "${UV_PYTHON_INSTALL_DIR:-}" "${UV_PYTHON_DOWNLOADS:-}" >> "$FAKE_LOG/uv.log"
printf '%s\\n' "${TMPDIR:-}" >> "$FAKE_LOG/uv-tmpdir.log"
case "$1" in
  --version) echo "uv 0.13.4 (fake)" ;;
  python)
    if [ -n "${FAKE_PYTHON_FAIL:-}" ]; then
      echo "error: Failed to download cpython-3.12" >&2
      exit 1
    fi ;;
  venv)
    for last in "$@"; do :; done
    mkdir -p "$last/bin"
    printf '#!/bin/sh\\ncat >/dev/null\\necho ok\\n' > "$last/bin/python"
    chmod 755 "$last/bin/python" ;;
esac
exit 0
"""

_FAKE_CURL = """#!/bin/sh
out=""; url=""
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    --retry|--connect-timeout) shift 2 ;;
    -*) shift ;;
    *) url="$1"; shift ;;
  esac
done
echo "$url" >> "$FAKE_LOG/curl.log"
case "$url" in "$FAKE_DOWN_PREFIX"*) exit 7 ;; esac
name="${url##*/}"
[ -f "$FAKE_SERVE/$name" ] || exit 22
cp "$FAKE_SERVE/$name" "$out"
"""


def _tarball(target: str) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in ("uv", "uvx"):
            data = _FAKE_UV.encode()
            info = tarfile.TarInfo(f"uv-{target}/{name}")
            info.size = len(data)
            info.mode = 0o755
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class _Box:
    """A temporary folder with a fake release server, curl and log."""

    def __init__(self, root: Path, *, tools: tuple[str, ...] | None = None):
        self.root = root
        self.serve = root / "serve"
        self.log = root / "log"
        self.bin = root / "bin"
        self.uv_dir = root / "opt" / "uv"
        self.tmp = root / "tmp"
        for folder in (self.serve, self.log, self.bin, self.tmp):
            folder.mkdir(parents=True)
        self.sums = {}
        for target in TARGETS:
            data = _tarball(target)
            (self.serve / f"uv-{target}.tar.gz").write_bytes(data)
            self.sums[target] = hashlib.sha256(data).hexdigest()
        curl = self.bin / "curl"
        curl.write_text(_FAKE_CURL)
        curl.chmod(0o755)
        if tools is None:
            self.path = f"{self.bin}:/usr/bin:/bin"
        else:
            # A PATH with only the named tools: an image that lacks the rest.
            for tool in tools:
                found = shutil.which(tool)
                assert found, tool
                (self.bin / tool).symlink_to(found)
            self.path = str(self.bin)
        self.bases = (
            "https://releases.astral.sh/github/uv/releases/download/0.13.4",
            "https://github.com/astral-sh/uv/releases/download/0.13.4",
        )

    def command(self, *, sums: dict[str, str] | None = None) -> str:
        return uv_install_command(
            uv_dir=str(self.uv_dir), sums=sums or self.sums, bases=self.bases
        )

    def env(self, **extra: str) -> dict[str, str]:
        return {
            "PATH": self.path,
            "HOME": str(self.root),
            "TMPDIR": str(self.tmp),
            "FAKE_LOG": str(self.log),
            "FAKE_SERVE": str(self.serve),
            "FAKE_DOWN_PREFIX": "https://unreachable.invalid/",
            **extra,
        }

    def run(self, command: str, **extra: str) -> subprocess.CompletedProcess[str]:
        return run_sh(
            command,
            env=self.env(**extra),
            capture_output=True,
            text=True,
            timeout=60,
        )

    def uv_calls(self) -> list[str]:
        log = self.log / "uv.log"
        return log.read_text().splitlines() if log.exists() else []

    def downloads(self) -> list[str]:
        log = self.log / "curl.log"
        return log.read_text().splitlines() if log.exists() else []


@needs_sh
def test_matching_sum_installs_uv_and_a_second_run_reuses_it(tmp_path):
    box = _Box(tmp_path)

    first = box.run(box.command())

    assert first.returncode == 0, first.stderr
    assert (box.uv_dir / "uv").is_file()
    assert (box.uv_dir / "uvx").is_file()
    assert len(box.downloads()) == 1
    assert box.downloads()[0].startswith(box.bases[0] + "/uv-")
    # uv never ran during the install itself.
    assert box.uv_calls() == []

    second = box.run(box.command())

    assert second.returncode == 0, second.stderr
    assert len(box.downloads()) == 1
    assert [call.split("|")[0] for call in box.uv_calls()] == ["--version"]
    # Nothing left behind in the temporary download folder.
    assert list(box.tmp.iterdir()) == []


@needs_sh
def test_wrong_sum_stops_before_uv_runs_naming_file_and_expected_sum(tmp_path):
    box = _Box(tmp_path)
    wrong = {target: "0" * 64 for target in TARGETS}

    result = box.run(box.command(sums=wrong))

    assert result.returncode != 0
    match = re.search(
        r"benchflow: sha256 mismatch for (uv-[a-z0-9_-]+\.tar\.gz): "
        r"expected (\w+), got (\w+); uv not installed",
        result.stderr,
    )
    assert match, result.stderr
    file_name, expected, got = match.groups()
    target = file_name.removeprefix("uv-").removesuffix(".tar.gz")
    assert expected == "0" * 64
    assert got == box.sums[target]
    assert not (box.uv_dir / "uv").exists()
    assert box.uv_calls() == []


@needs_sh
def test_second_host_is_used_when_the_first_is_unreachable(tmp_path):
    box = _Box(tmp_path)
    box.bases = ("https://unreachable.invalid/uv/0.13.4", box.bases[1])

    result = box.run(box.command())

    assert result.returncode == 0, result.stderr
    assert [url.split("/uv-")[0] for url in box.downloads()] == list(box.bases)


@needs_sh
def test_no_host_reachable_names_the_hosts(tmp_path):
    box = _Box(tmp_path)
    box.bases = ("https://unreachable.invalid/a", "https://unreachable.invalid/b")

    result = box.run(box.command())

    assert result.returncode != 0
    assert "could not download uv-" in result.stderr
    assert "from unreachable.invalid" in result.stderr
    assert not (box.uv_dir / "uv").exists()


@needs_sh
def test_image_without_a_sha256_tool_refuses_rather_than_skip_the_check(tmp_path):
    box = _Box(
        tmp_path,
        tools=(
            "mktemp",
            "tar",
            "gzip",
            "cp",
            "chmod",
            "mkdir",
            "rm",
            "uname",
            "id",
            "ls",
            # The root-only check of the uv folder (PT's C8 check).
            "find",
            "readlink",
            "head",
        ),
    )

    result = box.run(box.command())

    assert result.returncode != 0
    assert "cannot check the sha256 of uv-" in result.stderr
    assert not (box.uv_dir / "uv").exists()
    assert box.uv_calls() == []


# --------------------------------------------------------------------------
# The whole proxy install script, run locally against the fakes
# --------------------------------------------------------------------------


class _LocalShellSandbox:
    def __init__(self, box: _Box, **extra: str) -> None:
        self.box = box
        self.extra = extra

    async def exec(self, command: str, timeout_sec=None, user=None) -> _Result:
        done = self.box.run(command, **self.extra)
        return _Result(done.returncode, done.stdout, done.stderr)


def _point_proxy_install_at(box: _Box, monkeypatch, *, sums=None) -> Path:
    monkeypatch.setattr(runtime_mod, "UV_BIN", str(box.uv_dir / "uv"))
    # The managed Python's folder and uv's cache, made and checked as root
    # before uv runs, in the box (the exports keep the real paths).
    monkeypatch.setattr(_uv_pin, "PYTHON_INSTALL_DIR", str(box.root / "python"))
    monkeypatch.setattr(_uv_pin, "UV_CACHE_DIR", str(box.root / "uv-cache"))
    monkeypatch.setattr(
        runtime_mod, "uv_install_command", lambda: box.command(sums=sums)
    )
    return box.root / "venv"


@needs_sh
@pytest.mark.asyncio
async def test_proxy_install_builds_the_venv_on_managed_python_3_12(
    tmp_path, monkeypatch
):
    box = _Box(tmp_path)
    venv = _point_proxy_install_at(box, monkeypatch)

    python = await runtime_mod._ensure_sandbox_litellm(
        _LocalShellSandbox(box), venv_dir=str(venv), requirements_path=f"{venv}.txt"
    )

    assert python == f"{venv}/bin/python"
    calls = [call.split("|") for call in box.uv_calls()]
    assert [args.split()[:2] for args, *_ in calls] == [
        ["python", "install"],
        ["venv", "-q"],
        ["pip", "install"],
    ]
    assert calls[0][0] == f"python install -q --no-bin --reinstall {PYTHON_VERSION}"
    assert calls[1][0] == f"venv -q --python {PYTHON_VERSION} {venv}"
    for _args, preference, install_dir, downloads in calls:
        assert preference == "only-managed"
        assert install_dir == PYTHON_INSTALL_DIR
        assert downloads == "automatic"


@needs_sh
@pytest.mark.asyncio
async def test_proxy_install_with_wrong_sum_fails_before_uv_runs(tmp_path, monkeypatch):
    box = _Box(tmp_path)
    wrong = {target: "f" * 64 for target in TARGETS}
    venv = _point_proxy_install_at(box, monkeypatch, sums=wrong)

    with pytest.raises(RuntimeError) as raised:
        await runtime_mod._ensure_sandbox_litellm(
            _LocalShellSandbox(box), venv_dir=str(venv), requirements_path=f"{venv}.txt"
        )

    message = str(raised.value)
    assert "install LiteLLM in sandbox failed" in message
    assert re.search(r"sha256 mismatch for uv-[a-z0-9_-]+\.tar\.gz", message)
    assert f"expected {'f' * 64}" in message
    assert box.uv_calls() == []
    assert not venv.exists()


@needs_sh
@pytest.mark.asyncio
async def test_proxy_install_names_the_python_download_host_when_blocked(
    tmp_path, monkeypatch
):
    """Under an allowlist (or any network) that blocks the managed-Python host,
    the proxy install, which runs before the agent, stops naming the host."""
    box = _Box(tmp_path)
    venv = _point_proxy_install_at(box, monkeypatch)

    with pytest.raises(RuntimeError) as raised:
        await runtime_mod._ensure_sandbox_litellm(
            _LocalShellSandbox(box, FAKE_PYTHON_FAIL="1"),
            venv_dir=str(venv),
            requirements_path=f"{venv}.txt",
        )

    message = str(raised.value)
    assert f"could not install managed CPython {PYTHON_VERSION}" in message
    assert "releases.astral.sh" in message
    assert "github.com" in message
    assert "before the agent starts" in message
    assert [call.split()[0] for call in box.uv_calls()] == ["python"]
    assert not venv.exists()


# --------------------------------------------------------------------------
# Under an allowlist the proxy install runs before the policy is armed
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_allowlist_is_armed_only_after_the_proxy_install(tmp_path, monkeypatch):
    """The managed-Python and uv hosts need no allowlist entry: connect()
    installs the in-sandbox proxy first and arms the agent-uid policy after it,
    so the install reaches releases.astral.sh and the agent's allowlist is not
    widened. When the host is blocked anyway, the install stops before the
    agent naming it (the test above)."""
    from tests.test_hosted_search_policy import _fake_planes, _fake_sandbox, _rollout

    async def clear_firewall_audit(*args, **kwargs):
        return {"defense": "firewall_power", "state": "clear", "why": "fixture"}, None

    monkeypatch.setattr("benchflow.rollout.check_firewall_power", clear_firewall_audit)

    task = _hosted_task_with_allowlist(("pypi.org",))
    env = _fake_sandbox()
    planes = _fake_planes(env)
    # S3's audit fails closed on this fixture's empty exec replies. Model a
    # sound image here, as the hosted-search module's local fixture does;
    # this regression checks proxy-install order, not image privilege data.
    monkeypatch.setattr(
        "benchflow.rollout.check_firewall_power",
        AsyncMock(return_value=({"state": "clear", "why": "test fixture"}, None)),
    )
    order: list[str] = []

    async def ensure_litellm_runtime(**kwargs):
        assert kwargs["force_sandbox_local"] is True
        assert kwargs["sandbox"] is env
        order.append("proxy-install")
        return kwargs["agent_env"], None

    planes.ensure_litellm_runtime.side_effect = ensure_litellm_runtime
    planes.start_egress_denylist.side_effect = lambda *a, **k: order.append("arm")
    planes.connect_acp.side_effect = lambda **k: (
        order.append("agent"),
        (MagicMock(), MagicMock(), MagicMock(), "agent"),
    )[1]
    rollout = _rollout(tmp_path, monkeypatch, task=task, planes=planes)
    await rollout.setup()

    await rollout.connect()

    assert order == ["proxy-install", "arm", "agent"]
    policy = planes.start_egress_denylist.await_args.args[2]
    assert policy.allow_mode
    assert policy.allowed_hosts == ("pypi.org",)
    for host in _uv_pin.PYTHON_DOWNLOAD_HOSTS:
        assert host not in policy.allowed_hosts


def _hosted_task_with_allowlist(hosts: tuple[str, ...]) -> SimpleNamespace:
    sandbox = SimpleNamespace(
        env={},
        setup_commands=[],
        network_mode="allowlist",
        allowed_hosts=list(hosts),
        blocked_urls=[],
        blocked_hosts=[],
        allow_internet=True,
        skills_dir=None,
        docker_image=None,
        workdir=None,
    )
    return SimpleNamespace(
        name="allowlist-task",
        config=SimpleNamespace(
            sandbox=sandbox,
            agent=SimpleNamespace(prompt_prefix=None, timeout_sec=60),
            verifier=SimpleNamespace(env={}),
        ),
    )


# --------------------------------------------------------------------------
# uv's settings are BenchFlow's alone
# --------------------------------------------------------------------------


@needs_sh
def test_managed_exports_drop_every_inherited_uv_setting(tmp_path):
    """The proxy install runs as root in the agent's workspace. A ``uv.toml``
    the agent wrote there, or a ``UV_*`` variable from the image, could swap
    uv's Python download table for one without checksums
    (``python-downloads-json-url``), or point uv at a cache or index of the
    task's choosing. After the exports uv sees BenchFlow's settings only, and
    reads no configuration file."""
    script = f"{_uv_pin.managed_python_exports()}\nenv | grep '^UV_' | sort"
    inherited = {
        "UV_PYTHON_DOWNLOADS_JSON_URL": "/workspace/downloads.json",
        "UV_CONFIG_FILE": "/workspace/uv.toml",
        "UV_CACHE_DIR": "/workspace/.cache/uv",
        "UV_INDEX_URL": "https://mirror.invalid/simple",
        "UV_FIND_LINKS": "/workspace/wheels",
        "UV_PYTHON_PREFERENCE": "system",
    }

    done = run_sh(
        script,
        env={"PATH": "/usr/bin:/bin", **inherited},
        capture_output=True,
        text=True,
        timeout=30,
        cwd=tmp_path,
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == sorted(
        f"{key}={value}" for key, value in _uv_pin.MANAGED_PYTHON_ENV.items()
    )
    assert _uv_pin.MANAGED_PYTHON_ENV["UV_NO_CONFIG"] == "1"
    assert _uv_pin.MANAGED_PYTHON_ENV["UV_CACHE_DIR"] == "/opt/benchflow/cache/uv"


def test_proxy_layer_recipe_takes_the_same_settings():
    from benchflow.providers.proxy_requirements import ProxyInstall

    recipe = runtime_mod.proxy_layer_recipe(ProxyInstall(group="litellm-proxy"))
    command = recipe.install_cmd

    exports = command.index(_uv_pin.managed_python_exports())
    assert exports < command.index('"$UV" python install')


@pytest.mark.asyncio
async def test_proxy_python_is_one_exact_patch_release():
    """``uv python install 3.12`` keeps whatever 3.12 is already in the
    folder (an image's, an older layer's), so the proxy's interpreter would
    depend on the image. The install and the venv name the exact patch of
    the original Python pin."""
    from benchflow.providers.proxy_requirements import ProxyInstall

    assert re.fullmatch(r"3\.12\.\d+", PYTHON_VERSION)
    layer = runtime_mod.proxy_layer_recipe(ProxyInstall(group="litellm-proxy"))
    for command in (await _proxy_install_command(), layer.install_cmd):
        assert (
            f'"$UV" python install -q --no-bin --reinstall {PYTHON_VERSION} ' in command
        )
        assert f'"$UV" venv -q --python {PYTHON_VERSION} ' in command
        assert re.search(r"--python 3\.12(?![.\d])", command) is None
        assert re.search(r"--no-bin 3\.12(?![.\d])", command) is None


# --------------------------------------------------------------------------
# A failed pinned-uv fetch fails the agent install
# --------------------------------------------------------------------------


@needs_sh
def test_openhands_install_fails_when_the_pinned_uv_cannot_be_fetched(tmp_path):
    """An image that already has ``openhands`` on PATH, and no uv host
    reachable: the install used to end ``chmod ...; <settings> && command -v
    openhands``, so the failed fetch fell through to a passing tail and
    install_agent recorded success."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (
        ("curl", "exit 7"),  # every uv download host unreachable
        ("git", "exit 0"),
        ("openhands", "exit 0"),  # already in the image
    ):
        tool = bin_dir / name
        tool.write_text(f"#!/bin/sh\n{body}\n")
        tool.chmod(0o755)
    (tmp_path / "tmp").mkdir()

    done = run_sh(
        # The uv download folder sits beside the pinned uv (FF-install-C12):
        # an unprivileged test cannot create /.benchflow.
        AGENTS["openhands"].install_cmd.replace(
            _uv_pin.BENCHFLOW_ROOT, str(tmp_path / "protected")
        ),
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "HOME": str(tmp_path / "home"),
            "TMPDIR": str(tmp_path / "tmp"),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert done.returncode != 0
    assert "could not download uv-" in done.stderr
    assert not (tmp_path / "home" / ".openhands").exists()


def test_agent_installs_let_only_the_traversal_chmod_fail():
    for name in ("deepagents", "openhands"):
        install = AGENTS[name].install_cmd
        assert "2>/dev/null; " not in install, name
        if name == "openhands":
            # The tool's own folders only, mandatory: nothing under /root.
            assert (
                "chmod -R a+rX /opt/benchflow/python /opt/benchflow/uv-tools && "
                in install
            ), name
            assert "/root/.local" not in install, name
        else:
            # U4 no longer needs root-home traversal for deepagents. Its
            # managed interpreter and venv chmod must remain mandatory.
            assert (
                "chmod -R a+rX /opt/benchflow/agents/deepagents-venv "
                "/opt/benchflow/python && " in install
            )


# FF-install-C12's class: what root downloads, checks and then installs never
# sits under a folder the agent can own (/tmp, or the image's TMPDIR).


@needs_sh
def test_uv_download_stays_beside_the_uv_folder_not_in_tmpdir(tmp_path):
    """Guards a0c81002: the tarball was fetched to mktemp's /tmp, where /tmp's
    owner could swap it between the sum check and the unpack."""
    box = _Box(tmp_path)
    targets = box.log / "download-dirs.log"
    curl = box.bin / "curl"
    curl.write_text(
        _FAKE_CURL.replace(
            'cp "$FAKE_SERVE/$name" "$out"',
            f'dirname "$out" >> {targets}\ncp "$FAKE_SERVE/$name" "$out"',
        )
    )
    result = box.run(box.command())
    assert result.returncode == 0, result.stderr
    folders = targets.read_text().splitlines()
    assert folders
    for folder in folders:
        assert folder.startswith(f"{box.uv_dir.parent}/.download."), folder
        assert not folder.startswith(str(box.tmp))
    assert [p.name for p in box.uv_dir.parent.iterdir()] == [box.uv_dir.name]


@needs_sh
@pytest.mark.asyncio
async def test_proxy_install_uv_temp_files_stay_in_the_private_runtime(
    tmp_path, monkeypatch
):
    """Guards a0c81002: uv's temporary cache and Python downloads used the
    image's TMPDIR (/tmp) while installing the proxy that holds the key."""
    box = _Box(tmp_path)
    venv = _point_proxy_install_at(box, monkeypatch)
    await runtime_mod._ensure_sandbox_litellm(
        _LocalShellSandbox(box), venv_dir=str(venv), requirements_path=f"{venv}.txt"
    )
    used = (box.log / "uv-tmpdir.log").read_text().splitlines()
    assert len(used) == 3  # python install, venv, pip install
    assert set(used) == {f"{venv.parent}/uv-tmp"}
    assert (venv.parent / "uv-tmp").stat().st_mode & 0o777 == 0o700


@needs_sh
@pytest.mark.parametrize("kind", ["group-writable", "symlink", "parent-writable"])
def test_private_tmpdir_refuses_a_changeable_folder_and_stops(tmp_path, kind):
    """Guards a0c81002: a failed check must stop even a ``set -e`` script; uv
    must never go on with the image's TMPDIR."""
    parent = tmp_path / "root-only"
    parent.mkdir(mode=0o755)
    parent.chmod(0o755)
    folder = parent / "tmp"
    if kind == "group-writable":
        folder.mkdir()
        folder.chmod(0o770)
    elif kind == "symlink":
        # A link into a folder others may write (a root-owned link to a
        # root-only folder is root's alone, as PT's check has it).
        (tmp_path / "elsewhere").mkdir()
        (tmp_path / "elsewhere").chmod(0o777)
        folder.symlink_to(tmp_path / "elsewhere")
    else:
        parent.chmod(0o777)
    script = (
        f"set -e\n{_uv_pin.private_tmpdir_exports(str(folder))}\n"
        'echo "reached $TMPDIR"\n'
    )
    result = run_sh(
        script,
        env={"PATH": "/usr/bin:/bin", "TMPDIR": "/tmp"},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "reached" not in result.stdout
    assert "uv temp files" in result.stderr


@needs_sh
def test_private_tmpdir_is_created_private_and_exported(tmp_path):
    folder = tmp_path / "root-only/tmp"
    folder.parent.mkdir()
    folder.parent.chmod(0o755)
    script = (
        f'{_uv_pin.private_tmpdir_exports(str(folder))} && echo "$TMPDIR $TMP $TEMP"'
    )
    result = run_sh(
        script,
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [str(folder)] * 3
    assert folder.stat().st_mode & 0o777 == 0o700


@needs_sh
@pytest.mark.asyncio
async def test_proxy_install_takes_ownership_of_its_upload_only_after_the_folder_check(
    tmp_path, monkeypatch
):
    """Guards a0c81002 (5389b2d0): docker cp leaves the host user's uid on the
    uploaded requirements file, and every fresh proxy install on Docker by a
    non-root host was refused. The file is made root's, but only inside a
    runtime folder already checked root-only."""
    box = _Box(tmp_path)
    venv = _point_proxy_install_at(box, monkeypatch)
    calls = box.log / "chown.log"
    chown = box.bin / "chown"
    chown.write_text(f'#!/bin/sh\necho "$*" >> {calls}\n')
    chown.chmod(0o755)
    requirements = f"{venv}.txt"
    await runtime_mod._ensure_sandbox_litellm(
        _LocalShellSandbox(box), venv_dir=str(venv), requirements_path=requirements
    )
    # Everything BenchFlow uploaded to the checked folder (FF-INSTALL-ON-PT-R1).
    owner = f"{os.getuid()}:{os.getgid()}"  # the installing user: root in a sandbox
    assert calls.read_text().split() == ["-R", "-h", owner, "--", str(venv.parent)]

    # A runtime folder others can write: refused before anything is chowned.
    calls.unlink()
    tmp_path.chmod(0o777)
    try:
        with pytest.raises(RuntimeError, match="the proxy runtime folder"):
            await runtime_mod._ensure_sandbox_litellm(
                _LocalShellSandbox(box),
                venv_dir=str(venv),
                requirements_path=requirements,
            )
    finally:
        tmp_path.chmod(0o700)
    assert not calls.exists()


@needs_sh
@pytest.mark.parametrize("valid", [True, False])
def test_private_uv_bootstrap_checks_hash_without_installing_packages(tmp_path, valid):
    """Guards 01727c09e: no task-system install, even on a bad download."""
    box = _Box(tmp_path)
    command = uv_install_command(
        uv_dir=str(box.uv_dir),
        bases=box.bases,
        sums=box.sums if valid else {target: "0" * 64 for target in TARGETS},
        install_packages=False,
    )
    for forbidden in ("apt-get", "apk add", "dnf ", "yum "):
        assert forbidden not in command
    done = box.run(command)
    assert (done.returncode == 0) == valid, done.stderr
    assert (box.uv_dir / "uv").exists() == valid
    assert box.uv_calls() == []
