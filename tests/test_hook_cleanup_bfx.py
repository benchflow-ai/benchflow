"""The hook clean-up's bfx route: one deadline for the whole step, and the
shell form only where bfx was not delivered (never after a bfx answer)."""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace

import pytest

from benchflow._bfx.client import BfxAmbiguous, BfxError
from benchflow._bfx.locate import BfxUnavailable
from benchflow.sandbox import _hook_cleanup, lockdown

LABEL = "Verifier setup failed: purging Python injection hooks"
INPUTS = "a:b\0\0\0/app\0/usr/bin\0/usr/local/bin:/usr/bin\0"


def test_inputs_are_read_whole():
    inputs = _hook_cleanup.parse_inputs(INPUTS)
    assert inputs.values == ("a:b", "", "")
    assert (inputs.cwd, inputs.path, inputs.image_path) == (
        "/app",
        "/usr/bin",
        "/usr/local/bin:/usr/bin",
    )
    with pytest.raises(_hook_cleanup.HookCleanupError):
        _hook_cleanup.parse_inputs("a\0b\0")


class _Serve:
    def __init__(self, answer):
        self.answer = answer
        self.calls: list[str] = []

    async def call(self, cmd, args, timeout=None):
        self.calls.append(cmd)
        return await self.answer(cmd, args, timeout)


class _Env:
    """A bfx-capable sandbox whose execs and bfx session are scripted."""

    bfx_delivery = "bind"
    verifier_prep_script = False

    def __init__(self, serve=None, start_error=None):
        self.serve = serve
        self.start_error = start_error
        self.commands: list[str] = []

    async def bfx_call(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError

    async def exec(self, command, **kwargs):
        from benchflow.sandbox._exec_receipt import (
            encode_receipt,
            unwrap_receipt_command,
        )

        self.commands.append(command)
        unwrapped = unwrap_receipt_command(command)
        if unwrapped is not None:
            return SimpleNamespace(
                return_code=0,
                stdout=encode_receipt(INPUTS.encode(), 0, unwrapped[1]),
                stderr="",
            )
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    @contextlib.asynccontextmanager
    async def bfx_session(self, *, timeout_sec=15.0):
        if self.start_error is not None:
            raise self.start_error
        yield self.serve


def _no_cleanup_shell(env: _Env) -> None:
    assert not any("_BF_HOOK_" in c and "find /" in c for c in env.commands)


def test_the_route_needs_delivery_and_a_session():
    assert lockdown._hook_cleanup_on_bfx(_Env())
    undelivered = _Env()
    undelivered.bfx_delivery = None
    assert not lockdown._hook_cleanup_on_bfx(undelivered)


@pytest.mark.asyncio
async def test_one_deadline_bounds_the_whole_step():
    async def slow(cmd, args, timeout):
        await asyncio.sleep(0.4)
        return {"entries": [], "mounts": []}

    env = _Env(_Serve(slow))
    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(RuntimeError, match=LABEL):
        await lockdown._cleanup_hooks_on_bfx(
            env, {"cleanup_conftests": True}, "/app", (), LABEL, timeout_sec=1
        )
    # Each request fits its own share; the step as a whole stops at 1 s.
    assert loop.time() - started < 2.0
    _no_cleanup_shell(env)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        BfxUnavailable("bfx serve did not start"),
        BfxAmbiguous("lost", 3, True),
        BfxError("changed", "no longer the entry that was scanned"),
    ],
    ids=["did-not-start", "ambiguous", "refused"],
)
async def test_a_bfx_failure_fails_the_step_and_never_runs_the_shell(
    error, monkeypatch
):
    async def answer(cmd, args, timeout):
        raise error

    env = (
        _Env(start_error=error)
        if isinstance(error, BfxUnavailable)
        else _Env(_Serve(answer))
    )

    async def no_aliases(env, paths):
        return ()

    monkeypatch.setattr(lockdown, "_blocked_before_freeze", no_aliases)
    with pytest.raises(RuntimeError, match=LABEL):
        await lockdown.cleanup_verifier_python_hooks(
            env, None, LABEL, workspace="/app", user="root"
        )
    _no_cleanup_shell(env)


@pytest.mark.asyncio
async def test_a_hook_that_cannot_be_removed_fails_the_step():
    changes = _hook_cleanup._Changes(_hook_cleanup._Facts(None, 0.0))
    changes.failed.append("sitecustomize.py")

    async def call(cmd, args, timeout):  # pragma: no cover
        raise AssertionError

    async def nothing(*args, **kwargs):
        return None

    loop = asyncio.get_running_loop()
    original = (
        _hook_cleanup.remove_tmp_py,
        _hook_cleanup.clean_site_hooks,
        _hook_cleanup._Changes,
    )
    _hook_cleanup.remove_tmp_py = nothing
    _hook_cleanup.clean_site_hooks = nothing
    _hook_cleanup._Changes = lambda facts: changes
    try:
        with pytest.raises(_hook_cleanup.HookCleanupError, match="start-up hook"):
            await _hook_cleanup.run(
                call,
                _hook_cleanup.parse_inputs(INPUTS),
                cleanup_conftests=False,
                workspace=None,
                workspace_aliases=(),
                verifier_dir="/opt/benchflow/verifier",
                deadline=loop.time() + 5,
            )
    finally:
        (
            _hook_cleanup.remove_tmp_py,
            _hook_cleanup.clean_site_hooks,
            _hook_cleanup._Changes,
        ) = original


def test_digest_is_hook_rules():
    """The host's digest is HOOK_RULE's, byte for byte, on every node kind."""
    from benchflow._image_python_paths import HOOK_RULE

    namespace: dict = {}
    exec(compile(HOOK_RULE, "HOOK_RULE", "exec"), namespace)
    nodes = [
        ("f", b"import os\n", 0o100644),
        ("l", "../x.pth", b"data"),
        ("l", "/missing", None),
        ("d", 0o40755, [("a.py", ("f", b"a", 0o100644)), ("l", ("l", "a.py", b"a"))]),
        ("l", "caf\udcc3", None),
    ]
    for node in nodes:
        assert _hook_cleanup.digest(node) == namespace["_bf_digest"](node)
    for name in ("sitecustomize", "usercustomize.py", "sitecustomize.cpython-312.so",
                 "x.pth", "sitecustomizer.py", "a.pthx"):  # fmt: skip
        for site in (True, False):
            assert _hook_cleanup.is_hook(name, site) == namespace["_bf_is_hook"](
                name, site
            )


# --- a scripted bfx over a file tree (TR-P2 r3: H1, M1, M2, L3) ----------------


class _FakeFs:
    """bfx's ``scan``, ``walk``, ``remove``, ``write``, ``mkdir``, ``chmod``
    and ``clear`` over a tree of paths (bytes): ``"d"`` a folder, bytes a
    file. Paths that are not UTF-8 are answered as ``path_b64``, and a path
    given as text must be text, as bfx requires. Failures are scripted."""

    def __init__(self, tree: dict[bytes, object]):
        self.tree = {b"/": "d", **tree}
        for path in list(self.tree):
            while path != b"/":
                path = path.rsplit(b"/", 1)[0] or b"/"
                self.tree.setdefault(path, "d")
        self.inodes = {p: i for i, p in enumerate(sorted(self.tree), 100)}
        self.walk_answers: dict[bytes, list] = {}
        self.unlistable: set[bytes] = set()
        self.remove_errors: dict[bytes, list] = {}
        self.mounts: list[str] = []
        self.calls: list[tuple[str, dict]] = []

    @staticmethod
    def given(args, key) -> bytes | None:
        import base64

        if key in args:
            assert isinstance(args[key], str)
            return args[key].encode("utf-8")
        if f"{key}_b64" in args:
            return base64.b64decode(args[f"{key}_b64"])
        return None

    @staticmethod
    def put(row: dict, key: str, path: bytes) -> dict:
        import base64

        try:
            row[key] = path.decode("utf-8")
        except UnicodeDecodeError:
            row[f"{key}_b64"] = base64.b64encode(path).decode()
        return row

    def facts(self, path: bytes, shown: bytes | None = None) -> dict:
        node = self.tree[path]
        return self.put(
            {
                "type": "dir" if node == "d" else "file",
                "exists": True,
                "mode": 0o755 if node == "d" else 0o644,
                "size": 0 if node == "d" else len(node),
                "dev": 1,
                "ino": self.inodes[path],
            },
            "path",
            path if shown is None else shown,
        )

    def kids(self, path: bytes) -> list[bytes]:
        prefix = path.rstrip(b"/") + b"/"
        return sorted(
            p
            for p in self.tree
            if p.startswith(prefix) and b"/" not in p[len(prefix) :]
        )

    def join(self, root: bytes, rel: bytes) -> bytes:
        return (root.rstrip(b"/") + b"/" + rel) if rel else root

    async def call(self, cmd, args, timeout):
        import base64
        import fnmatch

        self.calls.append((cmd, dict(args)))
        if cmd == "scan":
            out = {
                "entries": [],
                "errors": [],
                "contents": {},
                "mounts": [{"mount_point": m} for m in self.mounts],
                "contents_b64": {},
            }
            paths = [p.encode() for p in args.get("paths", [])] + [
                base64.b64decode(p) for p in args.get("paths_b64", [])
            ]
            for path in paths:
                out["entries"].append(self.facts(b"/"))
                at = b""
                for comp in [c for c in path.split(b"/") if c]:
                    at += b"/" + comp
                    if at not in self.tree:
                        out["entries"].append(self.put({"exists": False}, "path", at))
                        break
                    out["entries"].append(self.facts(at))
            for walk in args.get("walks", []):
                root = self.given(walk, "root")
                if root not in self.tree:
                    out["entries"].append(self.put({"exists": False}, "path", root))
                    continue
                row = self.facts(root)
                if root in self.unlistable:
                    out["errors"].append(
                        self.put({"op": "list", "error": "x"}, "path", root)
                    )
                else:
                    row["listing"] = True
                    out["entries"] += [self.facts(k) for k in self.kids(root)]
                out["entries"].append(row)
            for key in ("contents", "contents_b64"):
                for given in args.get(key, []):
                    path = (
                        given.encode() if key == "contents" else base64.b64decode(given)
                    )
                    node = self.tree.get(path)
                    out[key][given] = (
                        {"b64": base64.b64encode(node).decode()}
                        if isinstance(node, bytes)
                        else {"error": "missing"}
                    )
            return out
        root = self.given(args, "root")
        rel = self.given(args, "path") or b""
        path = self.join(root, rel)
        if cmd == "walk":
            scripted = self.walk_answers.get(root)
            if scripted:
                answer = scripted.pop(0)
                if isinstance(answer, BaseException):
                    raise answer
                return answer
            if root not in self.tree:
                raise BfxError("missing", "ENOENT", errno=2)
            pattern = args["patterns"][0].removeprefix("**/").encode()
            prefix = root.rstrip(b"/") + b"/"
            return {
                "entries": [
                    self.facts(p, p[len(prefix) :])
                    for p in sorted(self.tree)
                    if p.startswith(prefix)
                    and pattern
                    and fnmatch.fnmatchcase(p.rsplit(b"/", 1)[1], pattern)
                ],
                "vanished": [],
                "deeper": [],
            }
        if cmd == "remove":
            errors = self.remove_errors.get(path)
            if errors:
                raise errors.pop(0)
            if path not in self.tree:
                raise BfxError("missing", "ENOENT", errno=2)
            assert args["expect"]["ino"] == self.inodes[path]
            for p in [p for p in self.tree if p == path or p.startswith(path + b"/")]:
                del self.tree[p]
            return {"removed": 1}
        if cmd in ("write", "mkdir"):
            data = base64.b64decode(args["data_b64"]) if cmd == "write" else "d"
            self.tree[path] = data
            self.inodes[path] = max(self.inodes.values()) + 1
            return {}
        return {}


def _run_on(fs: _FakeFs, workspace: str | None = "/app", conftests: bool = True):
    loop = asyncio.get_event_loop()
    return _hook_cleanup.run(
        fs.call,
        _hook_cleanup.parse_inputs(INPUTS.replace("a:b", "")),
        cleanup_conftests=conftests,
        workspace=workspace,
        workspace_aliases=(),
        verifier_dir="/opt/benchflow/verifier",
        deadline=loop.time() + 30,
    )


@pytest.mark.asyncio
async def test_a_walk_that_fails_beneath_a_present_root_fails_the_step():
    """H1: bfx's walk failing with ENOENT beneath ``/`` (a vanished entry, as
    the old walk reported it) is not an absent root."""
    fs = _FakeFs({b"/app": "d", b"/srv/x/conftest.py": b"x"})
    fs.walk_answers[b"/"] = [BfxError("missing", "srv/churn/f1: ENOENT", errno=2)]
    with pytest.raises(_hook_cleanup.HookCleanupError, match="could not walk '/'"):
        await _run_on(fs)


@pytest.mark.asyncio
async def test_an_absent_root_is_skipped_and_the_rest_runs():
    """A mount point the mount table names that is gone when it is walked is
    skipped (and recorded); the other roots are walked."""
    fs = _FakeFs(
        {
            b"/app": "d",
            b"/tmp/evil.py": b"x",
            b"/srv/x/conftest.py": b"x",
            b"/etc/hosts": b"127.0.0.1 localhost\n",
        }
    )
    # A mount point that is gone, and one that is a file (Docker's
    # /etc/hosts): both skipped, only the first recorded as vanished.
    fs.mounts = ["/srv/gone", "/etc/hosts"]
    fs.walk_answers[b"/etc/hosts"] = [BfxError("not-dir", "ENOTDIR", errno=20)]
    report = await _run_on(fs)
    assert b"/tmp/evil.py" not in fs.tree
    assert b"/srv/x/conftest.py" not in fs.tree
    assert report.vanished == ["/srv/gone"]


@pytest.mark.asyncio
async def test_vanished_entries_are_recorded_and_deeper_folders_are_walked_on():
    """H1 and M1: the walk's vanished entries are recorded, the rest of the
    walk counts; a folder handed back past bfx's depth is walked as a root of
    its own and its matches go."""
    deep = b"/tmp/" + b"a/" * 300
    fs = _FakeFs({b"/app": "d", deep + b"conftest.py": b"x", deep + b"evil.py": b"x"})
    first = await fs.call(
        "walk", {"root": "/", "path": "", "patterns": ["**/conftest.py"]}, 1
    )
    fs.calls.clear()
    handed = (b"tmp/" + b"a/" * 255 + b"a").decode()
    fs.walk_answers[b"/"] = [
        {
            "entries": [],
            "vanished": [{"path": "tmp/churn/f7"}],
            "deeper": [{"path": handed, "type": "dir"}],
        }
    ]
    assert first["entries"]
    report = await _run_on(fs)
    assert deep + b"conftest.py" not in fs.tree
    assert deep + b"evil.py" not in fs.tree
    assert "/tmp/churn/f7" in report.vanished
    walked = [_FakeFs.given(a, "root") for c, a in fs.calls if c == "walk"]
    assert b"/" + handed.encode() in walked


@pytest.mark.asyncio
async def test_names_that_are_not_text_are_carried_as_bytes():
    """M1: a conftest.py and a start-up hook under names that are not UTF-8
    are removed through bfx, as today's find and os.listdir remove them; the
    step passes."""
    site = b"/root/.local/lib/python3.12/site-packages"
    fs = _FakeFs(
        {
            b"/app": "d",
            b"/srv/\xff/conftest.py": b"x",
            b"/tmp/\xfe.py": b"x",
            site + b"/\xff.pth": b"import os\n",
        }
    )
    await _run_on(fs)
    assert b"/srv/\xff/conftest.py" not in fs.tree
    assert b"/tmp/\xfe.py" not in fs.tree
    assert site + b"/\xff.pth" not in fs.tree
    assert any("root_b64" in a or "path_b64" in a for c, a in fs.calls if c == "remove")


@pytest.mark.asyncio
async def test_a_hook_tree_deeper_than_one_remove_is_removed_from_the_bottom():
    """M1: a hook package the agent made deeper than bfx's remove goes
    (today's recursive remove takes it); the step passes."""
    site = b"/root/.local/lib/python3.12/site-packages"
    pkg = site + b"/sitecustomize"
    deep = pkg + b"/a" * 300
    fs = _FakeFs({b"/app": "d", deep + b"/x": b"x"})
    fs.remove_errors[pkg] = [BfxError("too-deep", "deeper than 512 levels")]
    handed = b"a/" * 255 + b"a"
    fs.walk_answers[pkg] = [
        {
            "entries": [],
            "vanished": [],
            "deeper": [fs.facts(pkg + b"/" + handed, handed)],
        }
    ]
    await _run_on(fs)
    assert not any(p.startswith(pkg) for p in fs.tree)


@pytest.mark.asyncio
async def test_a_recorded_package_on_a_read_only_mount_is_kept():
    """M2: bfx reports a read-only folder's removal as ``partial`` with the
    errno only in ``detail.left``; the kept package stays, as today's
    ``fresh`` keeps it."""
    changes = _hook_cleanup._Changes(_hook_cleanup._Facts(None, 0.0))

    async def lstat(path):
        return {"type": "dir", "dev": 1, "ino": 2}

    async def remove(real_dir, name, entry):
        raise _hook_cleanup._Refused(
            "/opt/ro/sitecustomize",
            "partial: 1 entries could not be removed",
            None,
            frozenset({30}),
        )

    changes.facts.lstat = lstat
    changes.remove = remove
    with pytest.raises(_hook_cleanup._ReadOnly):
        await changes.write_fresh("/opt/ro", "sitecustomize", ("d", 0o755, []))
    error = BfxError(
        "partial",
        "1 entries could not be removed",
        detail={
            "removed": 0,
            "left": [
                {
                    "path": "sitecustomize/__init__.py",
                    "error": {"kind": "os", "errno": 30},
                }
            ],
        },
    )
    assert _hook_cleanup._left_errnos(error) == {30}


@pytest.mark.asyncio
async def test_a_search_folder_that_cannot_be_listed_fails_the_step():
    """L3: a folder test.sh's Python searches that cannot be listed fails the
    step (today's os.listdir raises); it is never counted as empty."""
    site = b"/root/.local/lib/python3.12/site-packages"
    fs = _FakeFs({b"/app": "d", site + b"/sitecustomize.py": b"x"})
    fs.unlistable.add(site)
    with pytest.raises(_hook_cleanup.HookCleanupError, match="start-up hook"):
        await _run_on(fs)


def test_paths_go_as_text_or_as_their_bytes():
    assert _hook_cleanup._put("root", "/srv/x") == {"root": "/srv/x"}
    raw = "/srv/\udcff".encode("utf-8", "surrogateescape")
    import base64

    assert _hook_cleanup._put("root", "/srv/\udcff") == {
        "root_b64": base64.b64encode(raw).decode()
    }
    with pytest.raises(_hook_cleanup._Unaddressable):
        _hook_cleanup._put("root", "/srv/\0")
