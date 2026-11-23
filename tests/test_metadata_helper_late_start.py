"""A helper start whose ``docker run`` got no answer leaves no container.

VM 3 at load 80-190: 8 starts whose ``docker run --detach`` timed out
left the helper container in ``Created``. The daemon was still creating it
when the CLI was stopped, and the helper's remove by name came first ("No such
container"); the container landed after it. The remove now watches for it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchflow.sandbox import _metadata_helper as helper_mod
from tests.test_metadata_helper import MAIN, FakeDocker, _helper


class LateCreate(FakeDocker):
    """``docker run --detach`` gets no answer within its bound (the runner
    raises, as the sandbox's does). With ``lands``, the daemon creates the
    container right after the helper's first remove by name."""

    def __init__(self, *, lands: bool = True) -> None:
        super().__init__()
        self.lands = lands
        self.pending: str | None = None
        self.created: set[str] = set()

    async def __call__(self, args, *, timeout, stdin=None):
        if args[:1] == ["run"] and "--detach" in args:
            self.calls.append(list(args))
            self.stdin.append(stdin)
            self.pending = args[args.index("--name") + 1]
            raise RuntimeError(f"Command timed out after {int(timeout)} seconds")
        if args[:1] == ["rm"]:
            self.calls.append(list(args))
            self.stdin.append(stdin)
            name = args[-1]
            if name in self.created:
                self.created.discard(name)
                return SimpleNamespace(stdout=f"{name}\n", stderr="", return_code=0)
            if name == self.pending and self.lands:
                self.created.add(name)
            self.pending = None
            return SimpleNamespace(
                stdout="",
                stderr=f"Error response from daemon: No such container: {name}\n",
                return_code=1,
            )
        if args[:1] == ["ps"]:
            self.calls.append(list(args))
            self.stdin.append(stdin)
            wanted = args[args.index("--filter") + 1]
            found = [n for n in self.created if wanted.endswith(f"{n}$")]
            return SimpleNamespace(
                stdout="".join("f00d\n" for _ in found), stderr="", return_code=0
            )
        return await super().__call__(args, timeout=timeout, stdin=stdin)

    def name(self) -> str:
        (run,) = [c for c in self.calls if c[:1] == ["run"] and "--detach" in c]
        return run[run.index("--name") + 1]


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(helper_mod, "_READY", {})
    monkeypatch.setattr(helper_mod, "_FAILED", {})
    monkeypatch.setattr(helper_mod, "_BUILD_LOCKS", {})
    monkeypatch.setattr(helper_mod, "_PREBUILDS", {}, raising=False)
    monkeypatch.setattr(helper_mod, "_HOST_NETNS", {}, raising=False)
    monkeypatch.setattr(helper_mod, "_NETNS_IS_HOST", {}, raising=False)
    monkeypatch.delenv(helper_mod.HELPER_ENV, raising=False)
    monkeypatch.setattr(helper_mod, "_LATE_POLL_SEC", 0.01, raising=False)


@pytest.mark.asyncio
async def test_a_start_with_no_answer_is_removed_when_the_daemon_creates_it_late():
    docker = LateCreate()
    helper = _helper(docker)

    assert await helper.installed(MAIN, "echo rules") is None
    assert "timed out" in helper.unavailable
    await helper.aclose()

    assert docker.created == set(), f"left behind: {docker.created}"
    assert [c for c in docker.of("rm")] == [["rm", "--force", docker.name()]] * 2


@pytest.mark.asyncio
async def test_a_start_with_no_answer_that_never_lands_ends_the_watch(monkeypatch):
    monkeypatch.setattr(helper_mod, "_run_sec", lambda: 0.2)
    docker = LateCreate(lands=False)
    helper = _helper(docker)

    assert await helper.installed(MAIN, "echo rules") is None
    await helper.aclose()

    assert docker.created == set()
    assert docker.of("rm") == [["rm", "--force", docker.name()]]
