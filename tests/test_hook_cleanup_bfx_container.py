"""The hook clean-up on bfx gives today's outcome, in real containers.

Each case starts two containers of the same honest image with the same
preparation, builds BenchFlow's real kit in both (``setup_sandbox_user``,
which takes the pre-agent records), makes the same post-agent changes in
both, then runs BenchFlow's clean-up (``cleanup_verifier_python_hooks``):

- in one container bfx is not delivered, so the clean-up is today's shell
  and protected-Python form;
- in the other the pinned bfx is bound read-only at ``/.bfx``, so the host
  decides and bfx changes (:mod:`benchflow.sandbox._hook_cleanup`).

Both must end with the same tree (every path's type, mode, owner, size,
link text and bytes), the same hooks written again as fresh files, and the
same step outcome. The cases are PT round 16's honest images (uv src and
flat layouts, Debian's system Python, a conda-like env, a ``/usr/local``
workspace, a PYTHONPATH folder), seeded generated trees, mounts, the
V8-MERGE-1 shapes, and the clean-up under failure: a concurrent writer,
agent-made trees deeper than bfx's walk, names that are not text, a recorded
package on a read-only mount, and both scoring-path wirings
(``harden_before_verify`` batched and step by step). Two cases differ by
design and say so: an image without Python now runs the clean-up, and a
folder root cannot list (a task that drops root's DAC capabilities) fails
the step where today's find skips it.
"""

from __future__ import annotations

import base64
import contextlib
import os
import random
import shlex
import shutil
import subprocess
import uuid

import pytest

from tests.test_c8_protected_runtime_container import _docker_ready, _dx, _sandbox
from tests.test_pt_r16_hooks_container import (
    PYTHONPATH_APP,
    SLIM,
    VSITE,
    _build,
)
from tests.test_pt_r16_hooks_container import debian as debian
from tests.test_pt_r16_hooks_container import uv_flat as uv_flat
from tests.test_pt_r16_hooks_container import uv_src as uv_src

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
]

LABEL = os.environ.get("BENCHFLOW_TEST_CONTAINER_LABEL", "target=TR-CORE-hooks")
PREFIX = os.environ.get("BENCHFLOW_TEST_CONTAINER_PREFIX", "trcore-hooks-")
USER_SITE = "/root/.local/lib/python3.12/site-packages"
LOCAL_SITE = "/usr/local/lib/python3.12/site-packages"
HOOK = (
    "import os\n"
    "if os.geteuid() == 0:\n"
    "    os.makedirs('/tmp/marks/{name}', exist_ok=True)\n"
)
LABEL_TEXT = "Verifier setup failed: purging Python injection hooks"

# The snapshot: every path but the kernel's, the helper's, the kit's and the
# verifier's guard and tool configuration (per-container secrets and names),
# with its facts and its bytes.
_SKIP = (
    "/proc",
    "/sys",
    "/dev",
    "/.bfx",
    "/opt/benchflow",
    "*/_benchflow_guard_*",
    "/_benchflow_verifier_*",
    "/etc/.benchflow-*",
)
_VOLATILE = ("/etc/hostname", "/etc/hosts", "/etc/resolv.conf")
_SNAPSHOT = (
    "find / \\( "
    + " -o ".join(f"-path '{p}'" for p in _SKIP)
    + " \\) -prune -o -printf '%p\\t%y\\t%m\\t%U\\t%G\\t%s\\t%l\\t%i:%C@\\n' 2>/dev/null; "
    "echo '#bytes'; find / \\( "
    + " -o ".join(f"-path '{p}'" for p in _SKIP)
    + " \\) -prune -o -type f -exec sha256sum {} + 2>/dev/null"
)


def _bfx_mount() -> list[str]:
    from benchflow._bfx import delivery
    from benchflow._bfx.locate import BfxUnavailable

    try:
        return delivery.run_args(delivery.BIND, source=str(delivery.host_dir("x86_64")))
    except BfxUnavailable as exc:
        pytest.skip(f"no pinned bfx in this checkout: {exc}")


def _start(
    image: str,
    *,
    bfx: bool,
    env: tuple[str, ...],
    tmpfs: tuple[str, ...],
    extra: tuple[str, ...] = (),
):
    name = PREFIX + uuid.uuid4().hex[:8]
    key, _, value = LABEL.partition("=")
    flags = [f for v in env for f in ("-e", v)]
    flags += [f for t in tmpfs for f in ("--tmpfs", t)]
    flags += list(extra)
    run = subprocess.run(
        ["docker", "run", "-d", "--name", name, "--label", f"{key}={value}",
         "--network", "none", *flags, *(_bfx_mount() if bfx else []),
         image, "sleep", "1800"],
        capture_output=True,
        text=True,
    )  # fmt: skip
    if run.returncode != 0:
        pytest.skip(f"cannot start {image}: {run.stderr.strip()}")
    return name


@contextlib.contextmanager
def _pair(image: str, prep: str = "", *, env=(), tmpfs=(), extra=()):
    """Two containers of *image*: (today's form, bfx's form). *extra* are
    more ``docker run`` flags, the same for both."""
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    names: list[str] = []
    try:
        for bfx in (False, True):
            names.append(
                _start(
                    image,
                    bfx=bfx,
                    env=tuple(env),
                    tmpfs=tuple(tmpfs),
                    extra=tuple(extra),
                )
            )
        if prep:
            for name in names:
                made = _dx(name, prep)
                assert made.returncode == 0, made.stderr
        yield names[0], names[1]
    finally:
        for name in names:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _sandboxes(today: str, bfx: str):
    shell = _sandbox(today)
    helper = _sandbox(bfx)
    helper.bfx_delivery = "bind"
    from benchflow.sandbox import lockdown

    assert not lockdown._hook_cleanup_on_bfx(shell)
    assert lockdown._hook_cleanup_on_bfx(helper)
    return shell, helper


async def _setup(sandboxes, workspace: str) -> None:
    from benchflow.sandbox.lockdown import setup_sandbox_user

    for sandbox in sandboxes:
        await setup_sandbox_user(sandbox, "agent", workspace, guard_daemon=False)


def _both(names, script: str, user: str = "0") -> None:
    for name in names:
        done = _dx(name, script, user=user)
        assert done.returncode == 0, (name, script, done.stderr)


def _write(names, path: str, data: bytes, user: str = "0") -> None:
    encoded = base64.b64encode(data).decode()
    _both(names, f"echo {encoded} | base64 -d > {shlex.quote(path)}", user=user)


async def _clean(sandbox, workspace: str | None, hardening=None) -> str:
    from benchflow.sandbox import lockdown

    original = lockdown._read_hardening_config
    if hardening is not None:
        lockdown._read_hardening_config = lambda _task_dir: hardening
    try:
        await lockdown.cleanup_verifier_python_hooks(
            sandbox, None, LABEL_TEXT, workspace=workspace, user="root"
        )
        return "ok"
    except RuntimeError as exc:
        return f"failed: {exc}"
    finally:
        lockdown._read_hardening_config = original


def _snapshot(name: str) -> tuple[dict[str, tuple], dict[str, str]]:
    done = subprocess.run(
        ["docker", "exec", "-u", "0", name, "/bin/sh", "-c", _SNAPSHOT],
        capture_output=True,
    )
    facts: dict[str, tuple] = {}
    inodes: dict[str, str] = {}
    lines, _, sums = done.stdout.partition(b"#bytes\n")
    for line in lines.splitlines():
        parts = line.decode("utf-8", "surrogateescape").split("\t")
        if len(parts) != 8 or parts[0] in _VOLATILE:
            continue
        path, kind, mode, uid, gid, size, link, inode = parts
        # A folder's size is the filesystem's bookkeeping, not its contents.
        facts[path] = (kind, mode, uid, gid, "" if kind == "d" else size, link)
        inodes[path] = inode
    for line in sums.splitlines():
        digest, _, path = line.decode("utf-8", "surrogateescape").partition("  ")
        if path in facts:
            facts[path] = (*facts[path], digest)
    return facts, inodes


def _rewritten(before: dict[str, str], after: dict[str, str]) -> set[str]:
    # A file written again has another inode or a later change time (a freed
    # inode number can be given to the new file).
    return {p for p, ident in after.items() if p in before and before[p] != ident}


async def _equal_outcomes(
    today: str,
    bfx: str,
    shell,
    helper,
    workspace: str | None,
    *,
    hardening=None,
    clean=None,
) -> tuple[str, dict[str, tuple]]:
    """Run both clean-ups (*clean*, default :func:`_clean`); assert the same
    tree, the same rewritten paths and the same outcome. Returns the outcome
    and bfx's tree."""
    clean = clean or (lambda sandbox: _clean(sandbox, workspace, hardening))
    today_facts, today_before = _snapshot(today)
    bfx_facts, bfx_before = _snapshot(bfx)
    shell_outcome = await clean(shell)
    bfx_outcome = await clean(helper)
    today_after, today_inodes = _snapshot(today)
    bfx_after, bfx_inodes = _snapshot(bfx)
    # A path the two kit builds made differently (per-container values) is
    # compared only if a clean-up changed it.
    unequal_before = {
        p
        for p in set(today_facts) | set(bfx_facts)
        if today_facts.get(p) != bfx_facts.get(p)
        and today_facts.get(p) == today_after.get(p)
        and bfx_facts.get(p) == bfx_after.get(p)
    }
    differ = sorted(
        p
        for p in (set(today_after) | set(bfx_after)) - unequal_before
        if today_after.get(p) != bfx_after.get(p)
    )
    assert not differ, [(p, today_after.get(p), bfx_after.get(p)) for p in differ[:20]]
    assert _rewritten(today_before, today_inodes) - unequal_before == (
        _rewritten(bfx_before, bfx_inodes) - unequal_before
    )
    assert shell_outcome.split(":")[0] == bfx_outcome.split(":")[0], (
        shell_outcome,
        bfx_outcome,
    )
    return bfx_outcome, bfx_after


def _gone(tree: dict[str, tuple], *paths: str) -> None:
    for path in paths:
        assert path not in tree, path


def _kept(tree: dict[str, tuple], *paths: str) -> None:
    for path in paths:
        assert path in tree, path


# --- PT round 16's honest images ---------------------------------------------


@pytest.mark.asyncio
async def test_uv_venv_hooks_match_today(uv_src):
    """Every sitecustomize form the agent puts in a uv venv goes; the honest
    editable .pth is kept and written again as a fresh root-owned copy."""
    with _pair(uv_src) as (today, bfx):
        shell, helper = _sandboxes(today, bfx)
        await _setup((shell, helper), "/app")
        names = (today, bfx)
        _both(names, f"chmod o+w {VSITE}")
        _both(names, f"mkdir {VSITE}/sitecustomize", user="agent")
        _write(
            names,
            f"{VSITE}/sitecustomize/__init__.py",
            HOOK.format(name="pkg").encode(),
            "agent",
        )
        _write(
            names,
            f"{VSITE}/sitecustomize.cpython-312-x86_64-linux-gnu.so",
            b"x",
            "agent",
        )
        _write(names, f"{VSITE}/zz-agent.pth", b"import os\n", "agent")
        outcome, tree = await _equal_outcomes(today, bfx, shell, helper, "/app")
        assert outcome == "ok"
        _gone(tree, f"{VSITE}/sitecustomize", f"{VSITE}/zz-agent.pth",
              f"{VSITE}/sitecustomize.cpython-312-x86_64-linux-gnu.so")  # fmt: skip
        honest = [p for p in tree if p.startswith(VSITE) and p.endswith(".pth")]
        assert honest and all(tree[p][2] == "0" for p in honest)


@pytest.mark.asyncio
async def test_flat_layout_model_pth_matches_today(uv_flat):
    with _pair(uv_flat) as (today, bfx):
        shell, helper = _sandboxes(today, bfx)
        await _setup((shell, helper), "/app")
        names = (today, bfx)
        _write(names, "/app/model.pth", b"PK\x03\x04 weights", "agent")
        _both(names, f"chmod o+w {VSITE}")
        _write(names, f"{VSITE}/zz-agent.pth", b"import os\n", "agent")
        outcome, tree = await _equal_outcomes(today, bfx, shell, helper, "/app")
        assert outcome == "ok"
        _kept(tree, "/app/model.pth")
        _gone(tree, f"{VSITE}/zz-agent.pth")


@pytest.mark.asyncio
async def test_debian_system_sitecustomize_matches_today(debian):
    hook = "/usr/lib/python3.11/sitecustomize.py"
    with _pair(debian) as (today, bfx):
        shell, helper = _sandboxes(today, bfx)
        await _setup((shell, helper), "/work")
        _write((today, bfx), "/work/model.pth", b"dummy weights\n", "agent")
        outcome, tree = await _equal_outcomes(today, bfx, shell, helper, "/work")
        assert outcome == "ok"
        _kept(tree, hook, "/work/model.pth")


@pytest.mark.asyncio
async def test_user_site_and_conda_env_match_today():
    prep = (
        "/usr/local/bin/python3 -m venv --without-pip /opt/miniconda3 && "
        "mkdir /opt/miniconda3/conda-meta && "
        "/usr/local/bin/python3 -m venv --without-pip /opt/miniconda3/envs/testbed && "
        "echo /testbed > "
        "/opt/miniconda3/envs/testbed/lib/python3.12/site-packages/testbed.pth && "
        "useradd -m -u 1000 agent && mkdir -p /testbed && chown 1000 /testbed"
    )
    path = "PATH=/opt/miniconda3/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    with _pair(SLIM, prep, env=(path,)) as (today, bfx):
        shell, helper = _sandboxes(today, bfx)
        await _setup((shell, helper), "/testbed")
        names = (today, bfx)
        _both(names, "mkdir /testbed/sitecustomize", user="agent")
        _write(
            names,
            "/testbed/sitecustomize/__init__.py",
            HOOK.format(name="t").encode(),
            "agent",
        )
        _both(names, f"mkdir -p {USER_SITE}/usercustomize && chmod 755 /root")
        _write(
            names,
            f"{USER_SITE}/usercustomize/__init__.py",
            HOOK.format(name="u").encode(),
        )
        outcome, tree = await _equal_outcomes(today, bfx, shell, helper, "/testbed")
        assert outcome == "ok"
        _gone(tree, "/testbed/sitecustomize", f"{USER_SITE}/usercustomize")
        _kept(
            tree,
            "/opt/miniconda3/envs/testbed/lib/python3.12/site-packages/testbed.pth",
        )


@pytest.mark.asyncio
async def test_usr_local_workspace_matches_today():
    prep = (
        "useradd -m -u 1000 agent && mkdir -p /opt/vendor && "
        f"echo /opt/vendor > {LOCAL_SITE}/honest-vendor.pth"
    )
    with _pair(SLIM, prep) as (today, bfx):
        shell, helper = _sandboxes(today, bfx)
        await _setup((shell, helper), "/usr/local")
        names = (today, bfx)
        _both(names, f"chmod o+w {LOCAL_SITE}")
        _both(names, f"mkdir {LOCAL_SITE}/sitecustomize", user="agent")
        _write(
            names,
            f"{LOCAL_SITE}/sitecustomize/__init__.py",
            HOOK.format(name="p").encode(),
            "agent",
        )
        outcome, tree = await _equal_outcomes(today, bfx, shell, helper, "/usr/local")
        assert outcome == "ok"
        _gone(tree, f"{LOCAL_SITE}/sitecustomize")
        _kept(tree, f"{LOCAL_SITE}/honest-vendor.pth")


@pytest.fixture(scope="module")
def pythonpath_app(tmp_path_factory):
    return _build(tmp_path_factory, "pt-r16-pythonpath-app:test", PYTHONPATH_APP)


@pytest.mark.asyncio
async def test_pythonpath_folder_matches_today(pythonpath_app):
    with _pair(pythonpath_app) as (today, bfx):
        shell, helper = _sandboxes(today, bfx)
        await _setup((shell, helper), "/app")
        names = (today, bfx)
        _write(names, "/app/sitecustomize.py", HOOK.format(name="pp").encode(), "agent")
        _write(names, "/app/usercustomize.pyc", b"\x00pyc", "agent")
        _write(names, "/app/data.pth", b"/nowhere\n", "agent")
        outcome, tree = await _equal_outcomes(today, bfx, shell, helper, "/app")
        assert outcome == "ok"
        _gone(tree, "/app/sitecustomize.py", "/app/usercustomize.pyc")
        _kept(tree, "/app/data.pth")


# --- generated trees -------------------------------------------------------------


def _generated(seed: int) -> list[tuple[str, str]]:
    """(user, shell) steps planting a seeded mix of what today's policy
    judges: hooks in every kind of folder, links, caches, nested names,
    .pth path and import lines (some naming folders that hold hooks),
    conftest.py and *.py where they go and where they stay."""
    rng = random.Random(seed)
    steps: list[tuple[str, str]] = [
        ("0", f"mkdir -p {USER_SITE} /srv/extra /srv/named && chmod 755 /root")
    ]

    def put(user: str, path: str, data: bytes) -> None:
        encoded = base64.b64encode(data).decode()
        steps.append(
            (user, f"mkdir -p {shlex.quote(os.path.dirname(path))} && "
                   f"echo {encoded} | base64 -d > {shlex.quote(path)}")
        )  # fmt: skip

    folders = [USER_SITE, LOCAL_SITE, "/app", "/srv/extra", "/srv/named"]
    names = [
        "sitecustomize.py", "usercustomize.py", "sitecustomize.pyc",
        "sitecustomize.cpython-312-x86_64-linux-gnu.so", "zz-agent.pth",
        "a-binary.pth", "plain.py", "conftest.py", "notes.txt",
    ]  # fmt: skip
    for _ in range(rng.randint(6, 12)):
        folder = rng.choice(folders)
        name = rng.choice(names)
        user = "agent" if folder in ("/app",) else "0"
        data = (
            b"\x00\x01binary"
            if name == "a-binary.pth"
            else HOOK.format(name=f"g{seed}").encode()
        )
        put(user, f"{folder}/{name}", data)
    # A .pth naming a folder that holds hooks, and an addsitedir line.
    put(
        "0",
        f"{USER_SITE}/zz-names.pth",
        b"/srv/named\nimport site; site.addsitedir('/srv/extra')\n",
    )
    # Caches: hook bytecode goes, other bytecode stays; a nested cache.
    put("0", f"{USER_SITE}/__pycache__/sitecustomize.cpython-312.pyc", b"pyc")
    put("0", f"{USER_SITE}/__pycache__/other.cpython-312.pyc", b"pyc")
    put(
        "0",
        f"{LOCAL_SITE}/__pycache__/__pycache__/usercustomize.cpython-312.pyc",
        b"pyc",
    )
    # Links: to a file, to a folder, dangling, a cache link.
    if rng.random() < 0.7:
        steps.append(("0", f"ln -sf /etc/hostname {USER_SITE}/usercustomize.py"))
    if rng.random() < 0.5:
        steps.append(("0", f"ln -sfn /srv/extra {USER_SITE}/sitecustomize"))
    if rng.random() < 0.5:
        steps.append(("0", f"ln -sf /missing {LOCAL_SITE}/zz-dangling.pth"))
    if rng.random() < 0.5:
        steps.append(
            (
                "0",
                f"rm -rf {LOCAL_SITE}/__pycache__ && ln -s /tmp {LOCAL_SITE}/__pycache__",
            )
        )
    # conftest.py and *.py where they go and where they stay.
    for path in ("/conftest.py", "/srv/deep/a/b/conftest.py", "/tests/conftest.py",
                 "/verifier/conftest.py", "/tmp/x.py", "/tmp/.hidden.py",
                 "/tmp/dir.py/inner.py", "/var/tmp/y.py", "/tmp/keep.txt",
                 "/app/sub/conftest.py"):  # fmt: skip
        if rng.random() < 0.6:
            put("0", path, b"# planted\n")
    if rng.random() < 0.5:
        steps.append(
            ("0", "mkdir -p /srv/cdir/conftest.py && echo x > /srv/cdir/conftest.py/f")
        )
    return steps


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", range(1, 13))
async def test_generated_trees_match_today(seed):
    # The image's own hooks: path and import .pth files, a module, a package
    # (with a link inside) and a .pth that is a link; all recorded.
    prep = (
        "useradd -m -u 1000 agent && mkdir -p /app /opt/vendor && chown 1000 /app && "
        f"echo /opt/vendor > {LOCAL_SITE}/honest-vendor.pth && "
        f"printf 'import os\\n' > {LOCAL_SITE}/honest-import.pth && "
        f"printf '# image hook\\n' > {LOCAL_SITE}/sitecustomize.py && "
        f"mkdir -p {LOCAL_SITE}/usercustomize/sub && "
        f"printf '# pkg\\n' > {LOCAL_SITE}/usercustomize/__init__.py && "
        f"printf '# sub\\n' > {LOCAL_SITE}/usercustomize/sub/mod.py && "
        f"chmod 750 {LOCAL_SITE}/usercustomize/sub && "
        f"ln -s __init__.py {LOCAL_SITE}/usercustomize/alias.py && "
        f"echo /opt/vendor > /opt/vendor/target.pth && "
        f"ln -s /opt/vendor/target.pth {LOCAL_SITE}/linked.pth"
    )
    with _pair(SLIM, prep) as (today, bfx):
        shell, helper = _sandboxes(today, bfx)
        await _setup((shell, helper), "/app")
        # One recorded hook the agent edits (it goes), one it leaves (kept).
        _both((today, bfx), f"echo '# edited' >> {LOCAL_SITE}/honest-import.pth")
        for user, script in _generated(seed):
            _both((today, bfx), script, user=user)
        outcome, tree = await _equal_outcomes(today, bfx, shell, helper, "/app")
        _gone(tree, f"{LOCAL_SITE}/honest-import.pth")
        _kept(tree, f"{LOCAL_SITE}/honest-vendor.pth")
        if f"{LOCAL_SITE}/usercustomize" in tree:
            # Kept as a fresh root-owned tree: the link inside became a file.
            assert tree[f"{LOCAL_SITE}/usercustomize/alias.py"][0] == "f"
            assert tree[f"{LOCAL_SITE}/usercustomize/sub"][1] == "750"
        if f"{LOCAL_SITE}/linked.pth" in tree:
            assert tree[f"{LOCAL_SITE}/linked.pth"][0] == "f"
        assert outcome == "ok" or "start-up hook" in outcome


@pytest.mark.asyncio
async def test_workspace_in_tmp_keeps_its_py_files():
    prep = "useradd -m -u 1000 agent && mkdir -p /tmp/ws/pkg && chown -R 1000 /tmp/ws"
    with _pair(SLIM, prep) as (today, bfx):
        shell, helper = _sandboxes(today, bfx)
        await _setup((shell, helper), "/tmp/ws")
        names = (today, bfx)
        _write(names, "/tmp/ws/pkg/mod.py", b"x = 1\n", "agent")
        _write(names, "/tmp/other.py", b"x = 1\n")
        _both(names, "ln -s /tmp/ws /tmp/ws-link")
        outcome, tree = await _equal_outcomes(today, bfx, shell, helper, "/tmp/ws-link")
        assert outcome == "ok"
        _kept(tree, "/tmp/ws/pkg/mod.py")
        _gone(tree, "/tmp/other.py")


# --- mounts ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mounted_folders_are_covered_as_today():
    """tmpfs mounts in /tmp and /srv are crossed (their *.py and conftest.py
    go); a hook package that is a mount point loses its contents and the
    step fails, as removing a mount point fails."""
    prep = "useradd -m -u 1000 agent && mkdir -p /app && chown 1000 /app"
    tmpfs = ("/tmp/mnt", "/srv/m", f"{USER_SITE}/sitecustomize")
    with _pair(SLIM, prep, tmpfs=tmpfs) as (today, bfx):
        shell, helper = _sandboxes(today, bfx)
        await _setup((shell, helper), "/app")
        names = (today, bfx)
        for path in ("/tmp/mnt/x.py", "/tmp/mnt/sub/y.py", "/srv/m/conftest.py",
                     f"{USER_SITE}/sitecustomize/__init__.py"):  # fmt: skip
            _both(names, f"mkdir -p {os.path.dirname(path)} && echo '# p' > {path}")
        outcome, tree = await _equal_outcomes(today, bfx, shell, helper, "/app")
        assert outcome.startswith("failed")
        _gone(tree, "/tmp/mnt/x.py", "/tmp/mnt/sub/y.py", "/srv/m/conftest.py",
              f"{USER_SITE}/sitecustomize/__init__.py")  # fmt: skip
        _kept(tree, f"{USER_SITE}/sitecustomize")


# --- outcomes that are not changes ----------------------------------------------


@pytest.mark.asyncio
async def test_a_missing_workspace_fails_as_today():
    prep = "useradd -m -u 1000 agent && mkdir -p /app && chown 1000 /app"
    with _pair(SLIM, prep) as (today, bfx):
        shell, helper = _sandboxes(today, bfx)
        await _setup((shell, helper), "/app")
        _both((today, bfx), "rm -rf /app")
        outcome, _ = await _equal_outcomes(today, bfx, shell, helper, "/app")
        assert outcome.startswith("failed")


@pytest.mark.asyncio
async def test_without_a_workspace_tmp_purge_matches_today():
    """Without a workspace today's purge is ``find /tmp /var/tmp -name '*.py'
    -delete``: an empty folder named *.py goes as well."""
    prep = "useradd -m -u 1000 agent && mkdir -p /app && chown 1000 /app"
    with _pair(SLIM, prep) as (today, bfx):
        shell, helper = _sandboxes(today, bfx)
        await _setup((shell, helper), "/app")
        _both(
            (today, bfx),
            "mkdir -p /tmp/e.py /tmp/full.py /var/tmp/d.py/in.py && "
            "echo x > /tmp/full.py/keep.txt && echo x > /var/tmp/d.py/in.py/a.py && "
            "echo x > /tmp/f.py",
        )
        outcome, tree = await _equal_outcomes(today, bfx, shell, helper, None)
        assert outcome == "ok"
        _gone(tree, "/tmp/e.py", "/tmp/f.py", "/var/tmp/d.py")
        _kept(tree, "/tmp/full.py/keep.txt")


@pytest.mark.asyncio
async def test_conftest_opt_out_matches_today():
    prep = "useradd -m -u 1000 agent && mkdir -p /app && chown 1000 /app"
    with _pair(SLIM, prep) as (today, bfx):
        shell, helper = _sandboxes(today, bfx)
        await _setup((shell, helper), "/app")
        _both((today, bfx), "echo x > /app/conftest.py && echo x > /tmp/z.py")
        outcome, tree = await _equal_outcomes(
            today, bfx, shell, helper, "/app", hardening={"cleanup_conftests": False}
        )
        assert outcome == "ok"
        _kept(tree, "/app/conftest.py")
        _gone(tree, "/tmp/z.py")


# --- by design: what only bfx's form does ---------------------------------------


@pytest.mark.asyncio
async def test_an_image_without_python_now_runs_the_clean_up():
    """debian:12 has no Python: today's form removes conftest.py and /tmp's
    *.py but skips the start-up hooks (no protected Python); bfx's form
    removes them too."""
    prep = "useradd -m -u 1000 agent && mkdir -p /work && chown 1000 /work"
    with _pair("debian:12", prep) as (today, bfx):
        shell, helper = _sandboxes(today, bfx)
        await _setup((shell, helper), "/work")
        names = (today, bfx)
        _both(names, f"mkdir -p {USER_SITE} && chmod 755 /root")
        for path in (f"{USER_SITE}/zz.pth", f"{USER_SITE}/sitecustomize.py",
                     "/work/conftest.py", "/tmp/a.py"):  # fmt: skip
            _both(names, f"echo '# p' > {path}")
        assert await _clean(shell, "/work") == "ok"
        assert await _clean(helper, "/work") == "ok"
        today_tree, _ = _snapshot(today)
        bfx_tree, _ = _snapshot(bfx)
        _gone(today_tree, "/work/conftest.py", "/tmp/a.py")
        _kept(today_tree, f"{USER_SITE}/zz.pth", f"{USER_SITE}/sitecustomize.py")
        _gone(bfx_tree, "/work/conftest.py", "/tmp/a.py", f"{USER_SITE}/zz.pth",
              f"{USER_SITE}/sitecustomize.py")  # fmt: skip


# --- V8-MERGE-1 -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_v8_merge1_uv_only_image_without_python3_alias(uv_src):
    """V8-MERGE-1: a uv-only image (no system python3), where the agent
    deletes the venv's python3 alias and plants an import .pth. Today's form
    finds no protected Python and skips the start-up hooks, so the venv's
    python runs the hook as root; bfx's form removes it, and the honest
    editable .pth is kept as a fresh root-owned copy."""
    with _pair(uv_src, "rm -f /usr/local/bin/python3*") as (today, bfx):
        shell, helper = _sandboxes(today, bfx)
        await _setup((shell, helper), "/app")
        names = (today, bfx)
        _both(names, f"chmod o+w {VSITE} /app/.venv/bin")
        _both(names, "rm -f /app/.venv/bin/python3", user="agent")
        hook = "import os; os.geteuid() == 0 and os.makedirs('/tmp/marks/v8', exist_ok=True)\n"
        _write(names, f"{VSITE}/zz-v8.pth", hook.encode(), "agent")
        assert await _clean(shell, "/app") == "ok"
        assert await _clean(helper, "/app") == "ok"
        today_tree, _ = _snapshot(today)
        bfx_tree, _ = _snapshot(bfx)
        _kept(today_tree, f"{VSITE}/zz-v8.pth")
        _gone(bfx_tree, f"{VSITE}/zz-v8.pth")
        honest = [p for p in bfx_tree if p.startswith(VSITE) and p.endswith(".pth")]
        assert honest and all(bfx_tree[p][2] == "0" for p in honest)
        for name, ran in ((today, True), (bfx, False)):
            _dx(name, "cd /app && /app/.venv/bin/python -c pass")
            assert (_dx(name, "test -e /tmp/marks/v8").returncode == 0) is ran


@pytest.mark.asyncio
async def test_v8_merge1_venv_based_image():
    prep = (
        "/usr/local/bin/python3 -m venv --without-pip /opt/venv && "
        "useradd -m -u 1000 agent && mkdir -p /app && chown 1000 /app"
    )
    path = "PATH=/opt/venv/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    site = "/opt/venv/lib/python3.12/site-packages"
    with _pair(SLIM, prep, env=(path,)) as (today, bfx):
        shell, helper = _sandboxes(today, bfx)
        await _setup((shell, helper), "/app")
        names = (today, bfx)
        _both(names, f"chmod o+w {site}")
        _write(names, f"{site}/zz-v8.pth", b"import os\n", "agent")
        _write(
            names, f"{site}/sitecustomize.py", HOOK.format(name="v").encode(), "agent"
        )
        outcome, tree = await _equal_outcomes(today, bfx, shell, helper, "/app")
        assert outcome == "ok"
        _gone(tree, f"{site}/zz-v8.pth", f"{site}/sitecustomize.py")


@pytest.mark.asyncio
@pytest.mark.parametrize("flow", ["recovery", "regrade"])
async def test_v8_merge1_recovery_and_regrade_on_slim(tmp_path, flow):
    """python:3.12-slim. The agent's workspace (its own venv with an import
    .pth) is restored into fresh sandboxes the way recovery and regrade do,
    the kit is built after it, then the workspace hooks purge and the
    clean-up run: both forms end with the same tree, and the venv's python
    runs no hook as root."""
    from benchflow.eval_regrade import _install_bundle
    from benchflow.review.evidence import (
        capture_workspace,
        install_uploaded_workspace,
        prepare_workspace_upload,
    )
    from benchflow.rollout_planes import DefaultRolloutPlanes
    from benchflow.sandbox import lockdown
    from tests._trust_admission import admitted

    hook = (
        "import os; os.geteuid() == 0 and os.makedirs('/tmp/marks/r', exist_ok=True)\n"
    )
    prep = "useradd -m -u 1000 agent && mkdir -p /app && chown 1000 /app"
    with _pair(SLIM, prep) as (original, _unused):
        env = _sandbox(original)
        await lockdown.setup_sandbox_user(env, "agent", "/app", guard_daemon=False)
        _dx(
            original,
            "/usr/local/bin/python3 -m venv --copies --without-pip /app/.venv",
            "agent",
        )
        _write(
            (original,),
            "/app/.venv/lib/python3.12/site-packages/zz_r.pth",
            hook.encode(),
            "agent",
        )
        bundle = tmp_path / "bundle"
        await capture_workspace(env, "/app", bundle)
    with _pair(SLIM, prep) as (today, bfx):
        shell, helper = _sandboxes(today, bfx)
        for sandbox, name in ((shell, today), (helper, bfx)):
            if flow == "recovery":
                archive = tmp_path / f"workspace-{name}.tar"
                prepare_workspace_upload(bundle, archive)
                remote = "/tmp/benchflow-verifier-recovery-" + uuid.uuid4().hex
                await sandbox.upload_file(
                    archive, remote + ".tar", admission=admitted(sandbox)
                )
                await sandbox.upload_file(
                    bundle / "manifest.json",
                    remote + ".json",
                    admission=admitted(sandbox),
                )
                await install_uploaded_workspace(
                    sandbox,
                    remote + ".tar",
                    remote,
                    remote + ".json",
                    restore_modes=True,
                )
            else:
                staging = tmp_path / f"staging-{name}"
                staging.mkdir()
                remote = await _install_bundle(sandbox, bundle, staging, "workspace")
            await DefaultRolloutPlanes().setup_sandbox_user(
                sandbox, "agent", workspace="/app", guard_daemon=False
            )
            _dx(name, f"rm -rf /app && mv {remote} /app")
            await DefaultRolloutPlanes().setup_sandbox_user(
                sandbox, "agent", workspace="/app", guard_daemon=False
            )
            purged = await sandbox.exec(
                lockdown._purge_workspace_python_hooks_cmd("/app"),
                user="root",
                timeout_sec=120,
            )
            assert purged.return_code == 0, purged.stderr
        outcome, _ = await _equal_outcomes(today, bfx, shell, helper, "/app")
        assert outcome == "ok"
        for name in (today, bfx):
            _dx(name, "cd /app && /app/.venv/bin/python -c pass")
            assert _dx(name, "test -e /tmp/marks/r").returncode == 1


def test_the_route_is_chosen_by_delivery_only():
    from types import SimpleNamespace

    from benchflow.sandbox import lockdown
    from benchflow.sandbox.docker import DockerSandbox

    box = object.__new__(DockerSandbox)
    assert not lockdown._hook_cleanup_on_bfx(box)
    box.bfx_delivery = "bind"
    assert lockdown._hook_cleanup_on_bfx(box)
    assert not lockdown._hook_cleanup_on_bfx(SimpleNamespace(bfx_delivery="bind"))


# --- under failure (TR-P2 r3; review H1, M1, M2, M4) ---------------------------

PREP = "useradd -m -u 1000 agent && mkdir -p /app && chown 1000 /app"
# A root writer in main the kill does not stop (an image service's temp files).
CHURN = (
    "mkdir -p /tmp/churn; while :; do i=0; while [ $i -lt 300 ]; do "
    ": > /tmp/churn/f$i; i=$((i+1)); done; rm -f /tmp/churn/f*; done"
)


@pytest.mark.asyncio
async def test_a_concurrent_writer_never_leaves_a_planted_file(caplog):
    """H1: with a writer creating and deleting files in /tmp throughout, 15
    rounds of plant and clean-up: both forms pass and leave no planted file
    in every round. An entry that vanished while bfx walked is skipped and
    recorded; the walk goes on."""
    import logging

    rounds = 15
    with _pair(SLIM, PREP) as (today, bfx):
        shell, helper = _sandboxes(today, bfx)
        await _setup((shell, helper), "/app")
        names = (today, bfx)
        for name in names:
            subprocess.run(
                ["docker", "exec", "-d", "-u", "0", name, "/bin/sh", "-c", CHURN],
                check=True,
            )
        planted = ("/srv/x/conftest.py", "/tmp/evil.py")
        caplog.set_level(logging.WARNING, logger="benchflow.sandbox.lockdown")
        results: dict[str, list] = {"today": [], "bfx": []}
        for _ in range(rounds):
            _both(
                names,
                "mkdir -p /srv/x && echo x > /srv/x/conftest.py && "
                "echo x > /tmp/evil.py",
            )
            for label, name, box in (("today", today, shell), ("bfx", bfx, helper)):
                outcome = await _clean(box, "/app")
                left = [p for p in planted if _dx(name, f"test -e {p}").returncode == 0]
                results[label].append((outcome, left))
        for label in results:
            assert results[label] == [("ok", [])] * rounds, (label, results[label])
        assert any(
            "vanished while they were walked" in r.message for r in caplog.records
        )


@pytest.mark.asyncio
async def test_an_agent_tree_deeper_than_bfx_walks_matches_today():
    """M1: the agent makes folders 600 levels deep in /tmp and a hook package
    600 levels deep in a site folder it can write. bfx hands back each folder
    past its depth and the host walks on; the deep package is removed from
    the bottom. Both forms pass and end with the same tree."""
    deep = "i=0; while [ $i -lt 600 ]; do mkdir a && cd a; i=$((i+1)); done"
    with _pair(SLIM, PREP) as (today, bfx):
        shell, helper = _sandboxes(today, bfx)
        await _setup((shell, helper), "/app")
        names = (today, bfx)
        _both(names, f"chmod o+w {LOCAL_SITE}")
        _both(
            names,
            f"cd /tmp && {deep} && echo x > conftest.py && echo x > evil.py "
            "&& echo x > keep.txt",
            user="agent",
        )
        _both(
            names,
            f"mkdir {LOCAL_SITE}/sitecustomize && cd {LOCAL_SITE}/sitecustomize "
            f"&& {deep} && echo x > __init__.py",
            user="agent",
        )
        outcome, tree = await _equal_outcomes(today, bfx, shell, helper, "/app")
        assert outcome == "ok"
        bottom = "/tmp/" + "a/" * 600
        _gone(
            tree,
            bottom + "conftest.py",
            bottom + "evil.py",
            f"{LOCAL_SITE}/sitecustomize",
        )
        _kept(tree, bottom + "keep.txt")


@pytest.mark.asyncio
async def test_names_that_are_not_text_match_today():
    """M1: a conftest.py under a folder whose name is not UTF-8, a /tmp *.py
    and a .pth in a site folder with such names: bfx is given their bytes,
    and both forms remove them and pass."""
    with _pair(SLIM, PREP) as (today, bfx):
        shell, helper = _sandboxes(today, bfx)
        await _setup((shell, helper), "/app")
        names = (today, bfx)
        _both(names, f"chmod o+w {LOCAL_SITE}")
        _both(
            names,
            'd=$(printf \'/srv/\\377\'); mkdir -p "$d" && echo x > "$d/conftest.py" && '
            "echo x > \"$(printf '/tmp/\\376.py')\"",
        )
        _both(
            names,
            f"echo 'import os' > \"$(printf '{LOCAL_SITE}/\\377.pth')\"",
            user="agent",
        )
        outcome, tree = await _equal_outcomes(today, bfx, shell, helper, "/app")
        assert outcome == "ok"
        _gone(
            tree,
            "/srv/\udcff/conftest.py",
            "/tmp/\udcfe.py",
            f"{LOCAL_SITE}/\udcff.pth",
        )
        _kept(tree, "/srv/\udcff")


@pytest.mark.asyncio
async def test_a_recorded_package_on_a_read_only_mount_is_kept_as_today(tmp_path):
    """M2: PYTHONPATH names /opt/ro, bound read-only, holding a sitecustomize
    package recorded before the agent. Today's fresh copy fails at mkdir and
    keeps it; bfx's removal is partial with EROFS, and it is kept too."""
    ro = tmp_path / "ro"
    (ro / "sitecustomize").mkdir(parents=True)
    (ro / "sitecustomize" / "__init__.py").write_text("# honest\n")
    for path in (tmp_path, ro, ro / "sitecustomize"):
        path.chmod(0o755)
    extra = ("-v", f"{ro}:/opt/ro:ro")
    with _pair(SLIM, PREP, env=("PYTHONPATH=/opt/ro",), extra=extra) as (today, bfx):
        shell, helper = _sandboxes(today, bfx)
        await _setup((shell, helper), "/app")
        outcome, tree = await _equal_outcomes(today, bfx, shell, helper, "/app")
        assert outcome == "ok"
        _kept(tree, "/opt/ro/sitecustomize/__init__.py")


@pytest.mark.asyncio
async def test_a_folder_root_cannot_list_fails_the_step_by_design():
    """By design: where a task drops root's DAC capabilities, the agent can
    make a folder root cannot list. Today's find skips it silently and
    passes; bfx's walk is not complete, so the step fails and says where."""
    extra = ("--cap-drop", "DAC_OVERRIDE", "--cap-drop", "DAC_READ_SEARCH")
    with _pair(SLIM, PREP, extra=extra) as (today, bfx):
        shell, helper = _sandboxes(today, bfx)
        await _setup((shell, helper), "/app")
        _both(
            (today, bfx),
            "mkdir /tmp/locked && echo x > /tmp/locked/x.py && chmod 000 /tmp/locked",
            user="agent",
        )
        assert await _clean(shell, "/app") == "ok"
        outcome = await _clean(helper, "/app")
        assert outcome.startswith("failed") and "could not walk" in outcome


def _scoring_task(tmp_path):
    from unittest.mock import MagicMock

    from benchflow.task.config import TaskConfig

    task_dir = tmp_path / "task"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    task = MagicMock()
    task.task_dir = task_dir
    task.paths.task_dir = task_dir
    task.paths.tests_dir = task_dir / "tests"
    task.paths.test_path = task_dir / "tests" / "test.sh"
    task.config = TaskConfig.model_validate_toml('version = "1.0"\n')
    return task


@pytest.mark.asyncio
@pytest.mark.parametrize("wiring", ["batched", "step-by-step"])
@pytest.mark.parametrize("seed", [3, 7])
async def test_the_scoring_path_wirings_match_today(tmp_path, wiring, seed):
    """M4: the clean-up as harden_before_verify runs it on the scoring path,
    in the batched prep script (bfx: run from the host right after it) and
    step by step, over a generated tree: the same tree and outcome as today's form in the same wiring."""
    from benchflow.rollout_planes import DefaultRolloutPlanes
    from benchflow.sandbox.lockdown import harden_before_verify

    tasks: dict[str, object] = {}

    async def harden(sandbox):
        sandbox.verifier_prep_script = wiring == "batched"
        try:
            await harden_before_verify(
                sandbox,
                tasks[sandbox._main_container],
                "agent",
                workspace="/app",
            )
            return "ok"
        except RuntimeError as exc:
            return f"failed: {exc}"

    with _pair(SLIM, PREP) as (today, bfx):
        shell, helper = _sandboxes(today, bfx)
        await _setup((shell, helper), "/app")
        # The pre-agent baseline, as a rollout records it before the agent.
        planes = DefaultRolloutPlanes()
        for sandbox in (shell, helper):
            task = _scoring_task(tmp_path / sandbox._main_container)
            await planes.snapshot_build_config(sandbox, workspace="/app")
            await planes.seed_verifier_workspace(
                sandbox, workspace="/app", sandbox_user="agent"
            )
            await planes.prepare_verifier_runtime(
                sandbox, task=task, sandbox_user="agent"
            )
            tasks[sandbox._main_container] = task
        for user, script in _generated(seed):
            _both((today, bfx), script, user=user)
        outcome, tree = await _equal_outcomes(
            today, bfx, shell, helper, "/app", clean=harden
        )
        assert outcome == "ok" or "start-up hook" in outcome, outcome
        _gone(tree, "/conftest.py", "/tmp/x.py", "/var/tmp/y.py")
