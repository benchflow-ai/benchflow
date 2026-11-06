"""The cloud metadata block's rules from a helper container (sandbox._metadata_helper).

On Docker, root's rule script runs in BenchFlow's helper container that joins
main's network namespace and brings its own iptables, instead of an iptables
install in main. The block still refuses the same endpoints in the same
namespace and proves it the same way: the agent user's refused connect,
counted by the rule (counted from right before that connect). The scripts
themselves are tested in test_cloud_metadata_block.py; here, where they run.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from benchflow.sandbox import _metadata_helper as helper_mod
from benchflow.sandbox._exec_receipt import encode_receipt, unwrap_receipt_command
from benchflow.sandbox.cloud_metadata import CloudMetadataPowerError
from benchflow.sandbox.lockdown import _metadata_rules, block_cloud_metadata
from tests.test_cloud_metadata_block import AGENT_REFUSED, ROOT_OK, FakeEnv

MAIN = "4d95a10d423d" + "0" * 52


class HelperEnv(FakeEnv):
    """A sandbox whose rules come from a helper: its answers, in turn."""

    def __init__(self, *outputs: str, helper: list[str | None], why: str = "") -> None:
        super().__init__(*outputs)
        self.helper_answers = list(helper)
        self.helper_calls: list[str] = []
        self.why = why
        self.closed = 0
        self.restarts = ""
        self.helper_users: list[str | None] = []

    def _helper(self, command: str) -> tuple[Any, str]:
        self.helper_calls.append(command)
        out = self.helper_answers.pop(0)
        if out is None:
            return None, "TimeoutError"
        return SimpleNamespace(stdout=out, stderr="", return_code=0), ""

    async def metadata_rules_install(self, command: str) -> tuple[Any, str] | None:
        if self.why:
            return None
        return self._helper(command)

    async def metadata_rules_run(
        self, command: str, *, timeout_sec: int, user: str | None = None
    ):
        self.helper_users.append(user)
        return self._helper(command)

    async def metadata_main_restarts(self) -> str:
        return self.restarts

    @property
    def metadata_rules_unavailable(self) -> str:
        return self.why

    def metadata_rules_close(self) -> None:
        self.closed += 1


@pytest.mark.asyncio
async def test_the_rules_come_from_the_helper_and_main_installs_nothing():
    env = HelperEnv(
        AGENT_REFUSED,
        helper=[ROOT_OK, "BF metadata-hits 0\n", "BF metadata-hits 1\n"],
    )

    record = await block_cloud_metadata(env, "agent")

    assert record["state"] == "on", record
    assert record["rules_from"] == "helper"
    assert "refused and the rule counted 1 packet" in record["why"]
    # Root's rule script and both counter reads ran in the helper ...
    assert "BF metadata status=" in env.helper_calls[0]
    assert all("BF metadata-hits" in c for c in env.helper_calls[1:])
    # ... and main ran only the agent user's check and the delegate audit.
    assert [user for _, user in env.calls] == ["agent", "root"]
    assert not any("BF metadata status=" in command for command, _ in env.calls)
    assert env.closed == 1


@pytest.mark.asyncio
async def test_the_count_starts_right_before_the_agents_test_connect():
    """Traffic between the early install and the test connect (a task
    service's) never stands in for the agent user's refused connect."""
    env = HelperEnv(
        AGENT_REFUSED,
        helper=[ROOT_OK, "BF metadata-hits 5\n", "BF metadata-hits 5\n"],
    )

    record = await block_cloud_metadata(env, "agent")

    assert record["state"] == "unproven", record
    assert "its packet counter did not rise" in record["why"]


@pytest.mark.asyncio
async def test_no_count_before_the_test_connect_is_unproven():
    env = HelperEnv(AGENT_REFUSED, helper=[ROOT_OK, None, "BF metadata-hits 3\n"])

    record = await block_cloud_metadata(env, "agent")

    assert record["state"] == "unproven", record
    assert "the count before the test connect" in record["why"]


@pytest.mark.asyncio
async def test_without_the_helper_root_installs_in_main_as_before():
    why = "building the metadata firewall helper image failed: no route to host"
    env = HelperEnv(ROOT_OK, AGENT_REFUSED, "BF metadata-hits 2\n", helper=[], why=why)

    record = await block_cloud_metadata(env, "agent")

    assert record["state"] == "on", record
    assert record["rules_from"] == "sandbox"
    assert record["helper_unavailable"] == why
    assert [user for _, user in env.calls] == ["root", "agent", "root", "root"]
    assert env.helper_calls == []
    assert env.closed == 1


@pytest.mark.asyncio
async def test_the_helper_goes_even_when_the_agent_could_change_the_firewall():
    env = HelperEnv(
        "BF metadata-agent held=1 power= connect=refused\n",
        helper=[ROOT_OK, "BF metadata-hits 0\n", "BF metadata-hits 1\n"],
    )

    with pytest.raises(CloudMetadataPowerError):
        await block_cloud_metadata(env, "agent")
    assert env.closed == 1


def test_a_mocked_sandbox_never_grows_a_helper():
    assert _metadata_rules(MagicMock()) is None
    assert _metadata_rules(FakeEnv()) is None
    assert _metadata_rules(HelperEnv(helper=[])) is not None


STATUS1 = (
    "BF metadata status=1 fw4=iptables fw6=ip6tables v6=local install=none "
    "install_t=- hits=0\n"
)


@pytest.mark.asyncio
async def test_a_helper_result_that_failed_is_never_used_by_the_block():
    """A helper whose rules did not go in (an nft-only tool on a host where
    only iptables-legacy works): root's script runs in main with the image's
    own tools, as before the helper, and the record says why."""
    env = HelperEnv(
        ROOT_OK,
        AGENT_REFUSED,
        "BF metadata-hits 2\n",
        helper=[STATUS1, "BF metadata-hits 0\n", "BF metadata-hits 1\n"],
    )

    record = await block_cloud_metadata(env, "agent")

    assert record["state"] == "on", record
    assert record["rules_from"] == "sandbox"
    assert "could not add the rules (status 1)" in record["helper_unavailable"]
    assert [user for _, user in env.calls] == ["root", "agent", "root", "root"]
    assert any("BF metadata status=" in c for c, u in env.calls if u == "root")
    assert len(env.helper_calls) == 1
    assert env.closed == 1


# --- the helper container ----------------------------------------------------

IMAGE_ID = "sha256:" + "ab" * 32
OTHER_IMAGE = "sha256:" + "cd" * 32


class FakeDocker:
    """The docker CLI for the helper: records argv; ``build`` prints the
    image ID, ``run`` a new container ID, ``inspect`` that container's state
    and image, ``exec`` the next scripted answer (``exec_outs``, then
    ``exec_out``) as its receipt."""

    def __init__(self, *, build_fails: bool = False, image: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.stdin: list[bytes | None] = []
        self.build_fails = build_fails
        # A tagged, labelled helper image already on the daemon (a task's
        # Compose build can make one): never to be trusted.
        self.image = image
        self.exec_out = ROOT_OK
        self.exec_outs: list[str] = []
        # True: the exec prints its answer without the receipt line.
        self.unframed = False
        self.run_fails = False
        self.run_prints_id = True
        self.run_hangs = False
        self.container_image = IMAGE_ID
        self.containers: list[str] = []
        self.gone: set[str] = set()

    async def __call__(self, args: list[str], *, timeout: float, stdin=None):
        self.calls.append(list(args))
        self.stdin.append(stdin)
        out, code = "", 0
        if args[:2] == ["image", "inspect"]:
            out, code = (helper_mod.HELPER_RECIPE + "\n", 0) if self.image else ("", 1)
        elif args[:1] == ["build"]:
            code = 1 if self.build_fails else 0
            out = "ERROR: no route to host\n" if self.build_fails else f"{IMAGE_ID}\n"
        elif args[:1] == ["run"]:
            if self.run_hangs:
                await asyncio.Event().wait()
            if self.run_fails:
                code, out = 1, "Unable to find image locally\n"
            elif self.run_prints_id:
                self.containers.append(f"{len(self.containers) + 1:02d}" * 32)
                out = self.containers[-1] + "\n"
        elif args[:1] == ["inspect"]:
            if args[-1] in self.gone:
                code, out = 1, "Error: No such object\n"
            else:
                out = f"true {self.container_image}\n"
        elif args[:1] == ["exec"]:
            container = args[3] if args[1] == "--user" else args[1]
            if container in self.gone:
                code, out = 1, "Error response from daemon: No such container\n"
            else:
                out = self.exec_outs.pop(0) if self.exec_outs else self.exec_out
                framed = unwrap_receipt_command(args[-1])
                if framed is not None and not self.unframed:
                    out = encode_receipt(out.encode(), 0, framed[1])
        return SimpleNamespace(stdout=out, stderr="", return_code=code)

    def runs(self) -> list[list[str]]:
        return [c for c in self.calls if c[:1] == ["run"]]

    def of(self, verb: str) -> list[list[str]]:
        return [c for c in self.calls if c[:1] == [verb]]


@pytest.fixture(autouse=True)
def fresh_images(monkeypatch):
    monkeypatch.setattr(helper_mod, "_READY", type(helper_mod._READY)())
    monkeypatch.setattr(helper_mod, "_FAILED", {})
    monkeypatch.setattr(helper_mod, "_BUILD_LOCKS", {})
    monkeypatch.setattr(helper_mod, "_PREBUILDS", {}, raising=False)
    monkeypatch.delenv(helper_mod.HELPER_ENV, raising=False)


def _helper(docker: FakeDocker, daemon: str = "local") -> helper_mod.MetadataHelper:
    return helper_mod.MetadataHelper(
        docker,
        daemon=daemon,
        name="trial-a-bf-metadata",
        labels={"benchflow.owned": "process", "benchflow.process": "host:1:b:2"},
    )


@pytest.mark.asyncio
async def test_the_helper_joins_mains_namespace_with_the_least_power_it_needs():
    docker = FakeDocker()
    helper = _helper(docker)

    helper.install_early(MAIN, "echo rules")
    result, failed = await helper.installed(MAIN, "echo rules")

    assert failed == "" and result.stdout == ROOT_OK
    (run,) = docker.runs()
    assert run[run.index("--network") + 1] == f"container:{MAIN}"
    caps = [run[i + 1] for i, a in enumerate(run) if a == "--cap-add"]
    assert sorted(caps) == ["NET_ADMIN", "NET_RAW"]
    assert run[run.index("--cap-drop") + 1] == "ALL"
    assert run[run.index("--security-opt") + 1] == "no-new-privileges"
    assert "--user" in run and run[run.index("--user") + 1] == "0:0"
    assert "--ipc" in run and run[run.index("--ipc") + 1] == "none"
    tmpfs = [run[i + 1] for i, a in enumerate(run) if a == "--tmpfs"]
    assert [t.split(":", 1)[0] for t in tmpfs] == ["/run", "/tmp"]
    assert all(
        {"noexec", "nosuid", "nodev"} <= set(t.split(":", 1)[1].split(","))
        for t in tmpfs
    )
    assert "--read-only" in run and "--rm" in run
    for never in ("-v", "--volume", "--mount", "--privileged", "--pid", "--userns"):
        assert never not in run
    assert "--label=benchflow.owned=process" in run
    # By the ID this process's build gave, never by tag.
    assert IMAGE_ID in run and helper_mod.HELPER_TAG not in run
    # A name of its own for each start, so no task container can hold it first.
    name = run[run.index("--name") + 1]
    assert name.startswith("trial-a-bf-metadata-") and len(name) > len(
        "trial-a-bf-metadata-"
    )
    # Exec'd and checked by the container ID its run gave.
    (exec_call,) = docker.of("exec")
    assert exec_call[:4] == ["exec", docker.containers[0], "/bin/sh", "-c"]
    framed = unwrap_receipt_command(exec_call[4])
    assert framed is not None and framed[0] == "echo rules"
    assert docker.of("inspect")[-1][-1] == docker.containers[0]


@pytest.mark.asyncio
async def test_the_image_is_this_processs_own_build_never_a_tag_on_the_daemon():
    """A tagged and labelled image on the daemon (a task's Compose file can
    build one) is never taken for the helper: this process builds it, once
    per daemon, and runs the ID the build printed."""
    docker = FakeDocker(image=True)

    await _helper(docker).installed(MAIN, "echo rules")
    await _helper(docker).installed(MAIN, "echo rules")
    await _helper(docker, daemon="remote").installed(MAIN, "echo rules")

    assert docker.of("image") == []
    builds = docker.of("build")
    assert len(builds) == 2
    assert all(IMAGE_ID in run for run in docker.runs())
    dockerfile = next(s for s in docker.stdin if s is not None).decode()
    assert dockerfile.startswith("FROM alpine:3.24@sha256:")
    assert f"{helper_mod.HELPER_LABEL}=metadata" in builds[0]
    assert f"{helper_mod.HELPER_RECIPE_LABEL}={helper_mod.HELPER_RECIPE}" in builds[0]


def test_the_helpers_packages_are_pinned():
    """Each package the recipe adds is pinned to a version (iptables-legacy
    too, the fallback root has in a Debian main)."""
    (install,) = [
        line for line in helper_mod.HELPER_DOCKERFILE.splitlines() if "apk add" in line
    ]
    words = install.split(" && ")[0].split()
    packages = dict(w.split("=", 1) if "=" in w else (w, "") for w in words[4:])
    assert {"iptables", "iptables-legacy", "nftables"} <= set(packages)
    assert all(packages.values()), packages


@pytest.mark.asyncio
async def test_a_failed_build_says_why_and_is_tried_again_later(monkeypatch):
    docker = FakeDocker(build_fails=True)
    helper = _helper(docker)

    assert await helper.installed(MAIN, "echo rules") is None
    assert "no route to host" in helper.unavailable
    assert await _helper(docker).installed(MAIN, "echo rules") is None
    assert len(docker.of("build")) == 1

    now = helper_mod._clock()
    monkeypatch.setattr(
        helper_mod, "_clock", lambda: now + helper_mod._RETRY_BUILD_SEC + 1
    )
    docker.build_fails = False
    assert await _helper(docker).installed(MAIN, "echo rules") is not None


@pytest.mark.asyncio
async def test_a_main_that_changed_gets_a_helper_of_its_own():
    docker = FakeDocker()
    helper = _helper(docker)
    helper.install_early(MAIN, "echo rules")
    await helper.installed(MAIN, "echo rules")

    other = "a146163f04d2" + "1" * 52
    await helper.installed(other, "echo rules")

    assert [r[r.index("--network") + 1] for r in docker.runs()] == [
        f"container:{MAIN}",
        f"container:{other}",
    ]
    assert docker.of("rm") == [["rm", "--force", docker.containers[0]]]


@pytest.mark.asyncio
async def test_close_removes_the_helper_by_its_id_and_teardown_waits_for_it():
    docker = FakeDocker()
    helper = _helper(docker)
    helper.install_early(MAIN, "echo rules")
    await helper.installed(MAIN, "echo rules")

    helper.close()
    await helper.aclose()
    await asyncio.sleep(0)

    assert docker.of("rm") == [["rm", "--force", docker.containers[0]]]


def test_the_switch_turns_the_helper_off(monkeypatch):
    assert helper_mod.helper_enabled()
    monkeypatch.setenv(helper_mod.HELPER_ENV, "0")
    assert not helper_mod.helper_enabled()


@pytest.mark.asyncio
async def test_a_helper_that_could_not_add_the_rules_leaves_the_install_to_main():
    docker = FakeDocker()
    docker.exec_out = STATUS1
    helper = _helper(docker)

    assert await helper.installed(MAIN, "echo rules") is None
    assert "could not add the rules (status 1)" in helper.unavailable


@pytest.mark.asyncio
async def test_a_helper_whose_script_does_not_report_leaves_the_install_to_main():
    """No status line from the helper (it died mid-exec): root's script in
    main installs the rules as before, never a block recorded as missing."""
    docker = FakeDocker()
    docker.exec_out = "sh: can't open: No such file\n"
    helper = _helper(docker)

    assert await helper.installed(MAIN, "echo rules") is None
    assert "did not report" in helper.unavailable


@pytest.mark.asyncio
async def test_a_helper_image_gone_since_is_built_again():
    """A prune removed the image after this process built it: the start
    fails once, and the next helper builds it again."""
    docker = FakeDocker()
    await _helper(docker).installed(MAIN, "echo rules")
    docker.run_fails = True

    assert await _helper(docker).installed(MAIN, "echo rules") is None
    docker.run_fails = False
    builds = len(docker.of("build"))
    assert await _helper(docker).installed(MAIN, "echo rules") is not None
    assert len(docker.of("build")) == builds + 1


@pytest.mark.asyncio
async def test_the_helpers_answers_are_read_from_a_receipt():
    """What the helper prints reaches the block only through a checked
    receipt, as main's reads do: a status line or count outside one is a
    failed read with no output, so root's script in main installs instead
    and a count never comes from it."""
    docker = FakeDocker()
    docker.unframed = True
    helper = _helper(docker)

    assert await helper.installed(MAIN, "echo rules") is None
    assert "did not report" in helper.unavailable
    assert "receipt" in helper.unavailable
    docker.exec_out = "BF metadata-hits 7\n"
    result, failed = await helper.run("echo count", timeout_sec=5)
    assert failed == "" and result.stdout == "" and result.return_code != 0


@pytest.mark.asyncio
async def test_a_helper_gone_before_the_block_is_replaced_and_installs_again():
    docker = FakeDocker()
    helper = _helper(docker)
    helper.install_early(MAIN, "echo rules")
    await asyncio.sleep(0.01)
    first = docker.containers[0]
    docker.gone.add(first)  # its lifetime ended, or the daemon restarted

    result, _ = await helper.installed(MAIN, "echo rules")

    assert result.stdout == ROOT_OK
    assert len(docker.runs()) == 2 and helper.container == docker.containers[1]
    assert len(docker.of("exec")) == 2  # the install, again, in the new one
    assert ["rm", "--force", first] in docker.of("rm")


@pytest.mark.asyncio
async def test_a_helper_near_the_end_of_its_lifetime_is_replaced(monkeypatch):
    docker = FakeDocker()
    helper = _helper(docker)
    helper.install_early(MAIN, "echo rules")
    await asyncio.sleep(0.01)
    now = helper_mod._clock()
    later = helper_mod._HELPER_LIFETIME_SEC - 60  # a minute left: too little
    monkeypatch.setattr(helper_mod, "_clock", lambda: now + later)

    assert await helper.installed(MAIN, "echo rules") is not None
    assert len(docker.runs()) == 2 and helper.container == docker.containers[1]


@pytest.mark.asyncio
async def test_a_helper_container_running_another_image_is_never_used():
    docker = FakeDocker()
    docker.container_image = OTHER_IMAGE
    helper = _helper(docker)

    assert await helper.installed(MAIN, "echo rules") is None
    assert "does not run the image this process built" in helper.unavailable
    assert len(docker.runs()) == 1  # no second try with such a daemon


@pytest.mark.asyncio
async def test_a_run_with_no_container_id_back_runs_nothing_by_name():
    docker = FakeDocker()
    docker.run_prints_id = False
    helper = _helper(docker)

    assert await helper.installed(MAIN, "echo rules") is None
    assert "did not start" in helper.unavailable
    assert docker.of("exec") == [] and docker.of("inspect") == []
    await helper.aclose()
    (rm,) = docker.of("rm")
    (run,) = docker.runs()
    # Only the start's own random name, never the base name a task could hold.
    assert rm == ["rm", "--force", run[run.index("--name") + 1]]
    assert rm[-1] != "trial-a-bf-metadata"


@pytest.mark.asyncio
async def test_a_start_cancelled_midway_is_removed_by_its_own_name():
    docker = FakeDocker()
    docker.run_hangs = True
    helper = _helper(docker)
    helper.install_early(MAIN, "echo rules")
    await asyncio.sleep(0.01)

    await helper.aclose()

    (run,) = docker.runs()
    assert docker.of("rm") == [["rm", "--force", run[run.index("--name") + 1]]]
    assert run[run.index("--name") + 1] != "trial-a-bf-metadata"


class HelperBackedEnv(FakeEnv):
    """A sandbox whose block runs through a real MetadataHelper."""

    def __init__(self, *outputs: str, helper: helper_mod.MetadataHelper) -> None:
        super().__init__(*outputs)
        self.helper = helper

    async def metadata_rules_install(self, command: str) -> tuple[Any, str] | None:
        return await self.helper.installed(MAIN, command)

    async def metadata_rules_run(
        self, command: str, *, timeout_sec: int, user: str | None = None
    ):
        return await self.helper.run(command, timeout_sec=timeout_sec, user=user)

    async def metadata_main_restarts(self) -> str:
        return ""

    @property
    def metadata_rules_unavailable(self) -> str:
        return self.helper.unavailable

    def metadata_rules_close(self) -> None:
        self.helper.close()


@pytest.mark.asyncio
async def test_a_block_long_after_the_start_is_still_proven_from_a_live_helper():
    """The early helper is gone by the block (a prepared sandbox waited
    longer than its lifetime): a new one installs again (adding nothing),
    both counts come from it, and the block is on, as at once."""
    docker = FakeDocker()
    docker.exec_outs = [
        ROOT_OK,
        ROOT_OK,
        "BF metadata-hits 0\n",
        "BF metadata-hits 1\n",
    ]
    helper = _helper(docker)
    helper.install_early(MAIN, "echo rules")
    await asyncio.sleep(0.01)
    docker.gone.add(docker.containers[0])

    record = await block_cloud_metadata(
        HelperBackedEnv(AGENT_REFUSED, helper=helper), "agent"
    )

    assert record["state"] == "on", record
    assert record["rules_from"] == "helper"
    assert len(docker.runs()) == 2
    assert all(c[1] == docker.containers[1] for c in docker.of("exec")[1:])


# --- the Docker sandbox ----------------------------------------------------------


HELPER_CONTAINER = "77" * 32


def _helper_aware_docker():
    """test_docker_direct_exec's docker CLI, also answering the helper's
    build, run, inspect and exec."""
    from tests import test_docker_direct_exec as direct

    class Docker(direct.FakeDocker):
        build_fails = False
        # Set: the helper image's build answers only once this is set.
        build_gate: asyncio.Event | None = None

        async def __call__(self, *argv, **kwargs):
            proc = await super().__call__(*argv, **kwargs)
            words = direct._without_client_options([str(a) for a in argv])
            out = None
            if words[1:2] == ["build"] and self.build_fails:
                proc.returncode = 1
                out = "ERROR: no route to host\n"
            elif words[1:2] == ["build"] and self.build_gate is not None:
                gate = self.build_gate

                async def built(*_: Any, **__: Any) -> tuple[bytes, bytes]:
                    await gate.wait()
                    return f"{IMAGE_ID}\n".encode(), b""

                proc.communicate = built
            elif words[1:2] == ["build"]:
                out = f"{IMAGE_ID}\n"
            elif words[1:2] == ["run"]:
                out = f"{HELPER_CONTAINER}\n"
            elif words[1:3] == ["inspect", "--format"]:
                out = f"true {IMAGE_ID}\n"
            elif words[1:3] == ["exec", HELPER_CONTAINER]:
                framed = unwrap_receipt_command(words[-1])
                assert framed is not None
                out = encode_receipt(ROOT_OK.encode(), 0, framed[1])
            if out is not None:
                proc.communicate.return_value = (out.encode(), b"")
            return proc

    return Docker()


async def _settle() -> None:
    for _ in range(50):
        await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("armed", [True, False])
async def test_an_armed_start_puts_the_helper_in_mains_namespace(
    tmp_path, monkeypatch, armed
):
    from tests import test_docker_direct_exec as direct

    container = HELPER_CONTAINER
    docker = _helper_aware_docker()
    monkeypatch.setattr("asyncio.create_subprocess_exec", docker)
    sandbox = direct._sandbox(tmp_path)
    if armed:
        sandbox.task_env_config.arm_cloud_metadata_block()

    await direct._started(sandbox)
    await _settle()

    runs = [c for c in docker.commands() if c[1:2] == ["run"]]
    if not armed:
        assert runs == [] and sandbox._metadata_helper is None
        return
    (run,) = runs
    assert run[run.index("--network") + 1] == f"container:{direct.MAIN_ID}"
    assert run[run.index("--name") + 1].startswith(
        f"{sandbox.compose_project_name}-bf-metadata-"
    )
    assert IMAGE_ID in run
    # The helper's run never makes the sandbox forget main (direct execs).
    assert sandbox._main_container == direct.MAIN_ID
    # The rules went in from the helper, before any agent command.
    helper_exec = [c for c in docker.commands() if c[1:3] == ["exec", container]]
    assert len(helper_exec) == 1 and "BF metadata status=" in helper_exec[0][-1]

    await sandbox._close_metadata_helper()
    assert [c for c in docker.commands() if c[1:2] == ["rm"]] == [
        ["docker", "rm", "--force", container]
    ]


# --- R2-L1: this process's own build, beside the start -------------------------


@pytest.mark.asyncio
async def test_the_image_is_built_without_the_daemons_shared_cache():
    """The daemon's build cache is shared with task builds, which can import
    cache from outside (build.cache_from): the helper's image is this
    process's own run of its steps, not a cached result."""
    docker = FakeDocker()

    await _helper(docker).installed(MAIN, "echo rules")

    (build,) = docker.of("build")
    assert "--no-cache" in build


@pytest.mark.asyncio
async def test_independent_controllers_do_not_export_to_the_same_helper_tag():
    """Guards 88ac33cde's concurrent containerd image-export failure.

    Each controller has its own cache and build lock. The shared daemon
    refuses overlapping exports to one tag, as containerd does, so both
    controllers must keep independently built image IDs without that race.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "independent_metadata_helper", helper_mod.__file__
    )
    assert spec is not None and spec.loader is not None
    other = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(other)
    docker = FakeDocker()
    tags: list[str] = []
    arrived = asyncio.Event()

    async def shared_daemon(args, *, timeout, stdin=None):
        if args[:1] == ["build"]:
            tags.append(args[args.index("--tag") + 1])
            if len(tags) == 2:
                arrived.set()
            await asyncio.wait_for(arrived.wait(), 5)
            if len(set(tags)) != 2:
                return SimpleNamespace(
                    stdout="", stderr="image already exists", return_code=1
                )
        return await docker(args, timeout=timeout, stdin=stdin)

    built = await asyncio.gather(
        helper_mod.ensure_image(shared_daemon, "same-daemon"),
        other.ensure_image(shared_daemon, "same-daemon"),
    )
    assert built == [IMAGE_ID, IMAGE_ID]
    assert all("--no-cache" in command for command in docker.of("build"))


@pytest.mark.asyncio
async def test_a_prebuild_runs_once_and_the_helper_takes_its_image():
    prebuild = getattr(helper_mod, "prebuild", None)
    assert prebuild is not None, "no build beside the sandbox's start"
    docker = FakeDocker()

    prebuild(docker, "local")
    prebuild(docker, "local")
    await asyncio.sleep(0.01)
    await _helper(docker).installed(MAIN, "echo rules")

    assert len(docker.of("build")) == 1
    assert IMAGE_ID in docker.runs()[0]


@pytest.mark.asyncio
async def test_a_failed_prebuild_is_what_the_helper_reports():
    prebuild = getattr(helper_mod, "prebuild", None)
    assert prebuild is not None, "no build beside the sandbox's start"
    docker = FakeDocker(build_fails=True)

    prebuild(docker, "local")
    await asyncio.sleep(0.01)
    helper = _helper(docker)

    assert await helper.installed(MAIN, "echo rules") is None
    assert "no route to host" in helper.unavailable
    assert len(docker.of("build")) == 1


@pytest.mark.asyncio
async def test_an_armed_start_builds_the_helper_image_before_compose_creates_anything(
    tmp_path, monkeypatch
):
    """The uncached build takes about 1.5 s: it starts with the sandbox's
    start, not once main exists, so the helper is ready by the block."""
    from tests import test_docker_direct_exec as direct

    docker = _helper_aware_docker()
    monkeypatch.setattr("asyncio.create_subprocess_exec", docker)
    sandbox = direct._sandbox(tmp_path)
    sandbox.task_env_config.arm_cloud_metadata_block()

    await direct._started(sandbox)
    await _settle()

    commands = docker.commands()
    helper_builds = [
        i
        for i, c in enumerate(commands)
        if c[1:2] == ["build"] and f"{helper_mod.HELPER_LABEL}=metadata" in c
    ]
    first_build = helper_builds[0]
    first_compose_up = next(
        i
        for i, c in enumerate(commands)
        if c[:1] == ["compose"] and ("up" in c or "create" in c)
    )
    assert first_build < first_compose_up
    assert len(helper_builds) == 1


# --- Main without NET_ADMIN where the helper adds the rules ---------------------


def test_compose_gives_main_net_admin_for_the_block_only_without_the_helper():
    from benchflow.sandbox._compose import compose_needs_net_admin
    from benchflow.task.config import NetworkMode, SandboxConfig

    config = SandboxConfig(network_mode=NetworkMode.PUBLIC, allow_internet=True)
    assert not compose_needs_net_admin(config)
    config.arm_cloud_metadata_block()
    assert compose_needs_net_admin(config)
    serve = getattr(config, "serve_cloud_metadata_from_helper", None)
    assert serve is not None, "no way for the helper to take the block's rules"
    serve()
    assert not compose_needs_net_admin(config)


def _net_admin_overlay(docker: Any) -> list[bool]:
    """For each compose create/up call: whether its -f files had the
    NET_ADMIN overlay."""
    from benchflow.sandbox._compose import COMPOSE_NET_ADMIN_PATH

    return [
        any(str(a).endswith(COMPOSE_NET_ADMIN_PATH.name) for a in argv)
        for argv, _ in docker.calls
        if "compose" in argv and ("create" in argv or "up" in argv)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "net_admin"),
    [
        ("helper ready", False),
        ("build failed", True),
        ("no-web firewall", True),
        ("kept by switch", True),
        ("helper off", True),
    ],
)
async def test_main_starts_without_net_admin_only_where_the_helper_adds_the_rules(
    tmp_path, monkeypatch, case, net_admin
):
    """Only where the block is main's one reason for NET_ADMIN, the helper
    is on and its image is built by the time main starts: then nothing in
    main can change the firewall, and the firewall-power check ends at its
    first read (test_firewall_power.py, test_without_net_admin_the_check_
    ends_at_roots_read_even_with_a_barrier). Otherwise root's script can
    still add the rules in main, as before."""
    from tests import test_docker_direct_exec as direct

    docker = _helper_aware_docker()
    docker.build_fails = case == "build failed"
    monkeypatch.setattr("asyncio.create_subprocess_exec", docker)
    if case == "kept by switch":
        monkeypatch.setenv("BENCHFLOW_METADATA_MAIN_NET_ADMIN", "1")
    if case == "helper off":
        monkeypatch.setenv(helper_mod.HELPER_ENV, "0")
    sandbox = direct._sandbox(tmp_path)
    config = sandbox.task_env_config
    config.arm_cloud_metadata_block()
    if case == "no-web firewall":
        getattr(config, "need_agent_uid_firewall", lambda: None)()

    await direct._started(sandbox)
    await _settle()

    overlays = _net_admin_overlay(docker)
    assert overlays, "no compose create/up"
    assert overlays[-1] is net_admin
    assert getattr(config, "cloud_metadata_by_helper", False) is (not net_admin)


@pytest.mark.asyncio
async def test_a_main_without_net_admin_gives_a_new_helper_one_more_try():
    """With main holding no NET_ADMIN, root's script in main cannot add the
    rules: a helper that failed is followed by a new one before the block
    gives up."""
    import inspect

    assert "retry" in inspect.signature(helper_mod.MetadataHelper.installed).parameters
    docker = FakeDocker()
    docker.exec_outs = [STATUS1, ROOT_OK]
    helper = _helper(docker)

    result, _ = await helper.installed(MAIN, "echo rules", retry=True)

    assert result.stdout == ROOT_OK
    assert len(docker.runs()) == 2
    # Main with NET_ADMIN of its own: no second helper, main installs.
    again = FakeDocker()
    again.exec_outs = [STATUS1, ROOT_OK]
    assert await _helper(again).installed(MAIN, "echo rules") is None
    assert len(again.runs()) == 1


@pytest.mark.asyncio
async def test_the_record_says_whether_main_holds_net_admin():
    from benchflow.task.config import NetworkMode, SandboxConfig

    docker = FakeDocker()
    docker.exec_outs = [ROOT_OK, "BF metadata-hits 0\n", "BF metadata-hits 1\n"]
    env = HelperBackedEnv(AGENT_REFUSED, helper=_helper(docker))
    env.task_env_config = SandboxConfig(  # type: ignore[attr-defined]
        network_mode=NetworkMode.PUBLIC, allow_internet=True
    )
    env.task_env_config.arm_cloud_metadata_block()
    getattr(env.task_env_config, "serve_cloud_metadata_from_helper", lambda: None)()

    record = await block_cloud_metadata(env, "agent")

    assert record["state"] == "on", record
    assert record.get("main_net_admin") is False


@pytest.mark.asyncio
async def test_with_the_image_built_before_the_start_every_freeze_reads_one_set_of_files(
    tmp_path, monkeypatch
):
    """A process's later sandboxes (its image built by an earlier start)
    decide before the first launch freeze: every freeze and every compose
    call reads the files without the NET_ADMIN overlay, so a start never
    resolves its launch a second time for it."""
    from benchflow.sandbox._compose import (
        COMPOSE_NET_ADMIN_PATH,
        compose_needs_net_admin,
    )
    from tests import test_docker_direct_exec as direct

    docker = _helper_aware_docker()
    monkeypatch.setattr("asyncio.create_subprocess_exec", docker)
    sandbox = direct._sandbox(tmp_path)
    config = sandbox.task_env_config
    config.arm_cloud_metadata_block()
    await helper_mod.ensure_image(
        sandbox._helper_docker, " ".join(sandbox._docker_compose_global_args())
    )
    freezes: list[bool] = []
    sandbox.freeze_launch = lambda *_: freezes.append(compose_needs_net_admin(config))

    await direct._started(sandbox)
    await _settle()

    assert freezes and not any(freezes), freezes
    compose = [argv for argv, _ in docker.calls if "compose" in argv]
    assert compose
    assert not any(
        str(a).endswith(COMPOSE_NET_ADMIN_PATH.name) for argv in compose for a in argv
    )


@pytest.mark.asyncio
async def test_a_build_still_running_is_not_waited_for(tmp_path, monkeypatch):
    """A process's first sandbox usually starts before its build is done:
    main then keeps NET_ADMIN, and the start does not wait on the build
    (84a2f2ed waited up to 0.5 s for it, about what dropping it saved)."""
    import time

    from tests import test_docker_direct_exec as direct

    docker = _helper_aware_docker()
    docker.build_gate = asyncio.Event()
    monkeypatch.setattr("asyncio.create_subprocess_exec", docker)
    sandbox = direct._sandbox(tmp_path)
    sandbox.task_env_config.arm_cloud_metadata_block()

    started = time.monotonic()
    try:
        await direct._started(sandbox)
        took = time.monotonic() - started
        overlays = _net_admin_overlay(docker)
        assert overlays and overlays[-1] is True
        assert not getattr(sandbox.task_env_config, "cloud_metadata_by_helper", False)
        assert took < 0.5, took
    finally:
        docker.build_gate.set()
        await _settle()


@pytest.mark.parametrize("block", [True, False])
def test_the_no_web_marker_is_set_only_where_the_block_is_armed(tmp_path, block):
    """The factory marks an agent-layer no-web run's config (main keeps its
    NET_ADMIN for root's uid firewall) only where the block is armed, the
    one place the mark is read. 84a2f2ed copied the task's config to mark
    every no-web run (tests/test_internet_policy.py found it)."""
    from unittest.mock import patch

    from benchflow.sandbox.setup import _create_sandbox_environment
    from benchflow.task import RolloutPaths, Task
    from tests.test_cloud_metadata_block import MAIN_ONLY, OPEN, _task_dir

    task_dir = _task_dir(tmp_path, OPEN, MAIN_ONLY)
    task = Task(task_dir)
    with patch("benchflow.sandbox.setup._freeze"):
        sandbox = _create_sandbox_environment(
            "docker",
            task,
            task_dir,
            "rollout",
            RolloutPaths(rollout_dir=tmp_path / "rollout"),
            preserve_agent_network=True,
            block_cloud_metadata=block,
        )
    config = sandbox.task_env_config
    assert config.agent_uid_firewall is block
    if not block:
        assert config is task.config.sandbox


# --- An agent that runs as root is held too ---------------------------

# Root's capability sets: Docker's defaults with NET_RAW dropped (what main
# starts with for a root agent), with NET_ADMIN added, with NET_RAW kept.
CLEAN = "00000000a80405fb"
WITH_NET_ADMIN = "00000000a80415fb"
WITH_NET_RAW = "00000000a80425fb"
NO_CAPS = "0" * 16


def _root_caps(bnd: str) -> str:
    return (
        f"BF metadata-root CapBnd={bnd} CapPrm={bnd} CapInh={NO_CAPS} "
        f"CapEff={bnd} CapAmb={NO_CAPS}\n"
    )


class RootEnv(HelperBackedEnv):
    """A main whose root holds ``caps``; root's test connect is refused, as
    the rule's REJECT answers it. ``commands`` lists every command run in
    main, by the user it ran as."""

    def __init__(self, caps: str, *outputs: str, helper) -> None:
        super().__init__(*outputs, helper=helper)
        self.caps = caps
        self.commands: list[tuple[str, str | None]] = []

    def answer(self, command: str, user: str | None):
        from tests._receipt_fakes import inner_command

        inner = inner_command(command)
        self.commands.append((inner, user))
        if "BF metadata-root" in inner:
            return SimpleNamespace(stdout=self.caps, stderr="", return_code=0)
        if "BF metadata-connect" in inner:
            return SimpleNamespace(
                stdout="BF metadata-connect refused\n", stderr="", return_code=0
            )
        return super().answer(command, user)

    def ran(self, marker: str) -> list[str | None]:
        return [user for command, user in self.commands if marker in command]


def test_the_rules_refuse_root_too_only_for_a_root_agent():
    from benchflow.sandbox.lockdown import (
        _cloud_metadata_block_cmd,
        _cloud_metadata_rule,
    )

    assert "! --uid-owner 0" in _cloud_metadata_rule("169.254.169.254")
    assert "owner" not in _cloud_metadata_rule("169.254.169.254", cover_root=True)
    assert "--uid-owner" in _cloud_metadata_block_cmd()
    assert "--uid-owner" not in _cloud_metadata_block_cmd(cover_root=True)


@pytest.mark.asyncio
async def test_a_root_agent_is_held_by_rules_the_helper_adds_for_root_too():
    """An agent that runs as root was never held (its plan was
    off). Now root's sets are read first (no NET_ADMIN, no NET_RAW), the
    helper adds rules with no root exemption, and root's own connect is
    the proof; no launch-barrier proof or agent-user check applies."""
    docker = FakeDocker()
    docker.exec_outs = [ROOT_OK, "BF metadata-hits 0\n", "BF metadata-hits 1\n"]
    env = RootEnv(_root_caps(CLEAN), helper=_helper(docker))

    record = await block_cloud_metadata(env, None)

    assert record["state"] == "on", record
    assert record["root_agent"] is True and record["rules_from"] == "helper"
    assert record["root_capabilities"]["CapBnd"] == CLEAN
    assert record["delegate_privilege_audit"] == "not_needed"
    assert "every user, root included" in record["why"]
    assert "root's connect" in record["why"]
    framed = unwrap_receipt_command(docker.of("exec")[0][-1])
    assert framed is not None
    assert "--uid-owner" not in framed[0] and "REJECT" in framed[0]
    assert env.ran("BF metadata-root") == ["root"]
    assert env.ran("BF metadata-connect") == ["root"]
    assert not env.ran("BF metadata-construction")
    assert not env.ran("BF metadata-agent")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bnd", "held"), [(WITH_NET_ADMIN, "NET_ADMIN"), (WITH_NET_RAW, "NET_RAW")]
)
async def test_a_root_agent_whose_root_holds_net_admin_or_net_raw_is_not_held(
    bnd, held
):
    """Root with NET_ADMIN could remove any rule, and with NET_RAW send
    packets around them: nothing is installed, and the record (the trial
    stops on it) names the capability and the helper's reason."""
    docker = FakeDocker()
    helper = _helper(docker)
    helper.unavailable = "building the metadata firewall helper image failed: x"
    env = RootEnv(_root_caps(bnd), helper=helper)

    record = await block_cloud_metadata(env, None)

    assert record["state"] == "not_installed"
    assert f"root in main holds {held}" in record["why"]
    assert "helper is unavailable: building" in record["why"]
    assert not docker.of("run") and not docker.of("exec")
    assert not env.ran("BF metadata status=")


@pytest.mark.asyncio
async def test_a_block_with_another_script_runs_it_in_a_new_helper():
    """The early install ran one script; a block that asks for another (a
    root agent's) never takes the early result for it."""
    docker = FakeDocker()
    helper = _helper(docker)
    helper.install_early(MAIN, "echo rules")
    assert await helper.installed(MAIN, "echo rules") is not None
    assert len(docker.runs()) == 1
    assert await helper.installed(MAIN, "echo rules for root too") is not None
    assert len(docker.runs()) == 2


@pytest.mark.asyncio
async def test_a_root_agents_start_waits_for_the_build_and_drops_net_raw(
    tmp_path, monkeypatch
):
    """For a root agent main must start with neither NET_ADMIN nor NET_RAW:
    the start waits for this process's build, then every compose call
    names the NET_RAW drop and none the NET_ADMIN grant."""
    from benchflow.sandbox._compose import (
        COMPOSE_NET_ADMIN_PATH,
        COMPOSE_NO_NET_RAW_PATH,
    )
    from tests import test_docker_direct_exec as direct

    docker = _helper_aware_docker()
    docker.build_gate = asyncio.Event()
    asyncio.get_running_loop().call_later(0.2, docker.build_gate.set)
    monkeypatch.setattr("asyncio.create_subprocess_exec", docker)
    sandbox = direct._sandbox(tmp_path)
    sandbox.task_env_config.arm_cloud_metadata_block(root_agent=True)

    await direct._started(sandbox)
    await _settle()

    assert sandbox.task_env_config.cloud_metadata_by_helper is True
    compose = [argv for argv, _ in docker.calls if "compose" in argv]
    assert compose
    assert not any(
        str(a).endswith(COMPOSE_NET_ADMIN_PATH.name) for argv in compose for a in argv
    )
    starts = [argv for argv in compose if "create" in argv or "up" in argv]
    assert starts
    assert all(
        any(str(a).endswith(COMPOSE_NO_NET_RAW_PATH.name) for a in argv)
        for argv in starts
    )


@pytest.mark.asyncio
async def test_a_root_agents_failed_build_is_named_for_the_block(tmp_path, monkeypatch):
    """No helper image: main gets no NET_ADMIN for a root agent's block
    either (the block holds root only where root has none); it
    keeps NET_RAW, so the block refuses to hold the root agent (test
    above), and the sandbox names the build's failure for the record."""
    from tests import test_docker_direct_exec as direct

    docker = _helper_aware_docker()
    docker.build_fails = True
    monkeypatch.setattr("asyncio.create_subprocess_exec", docker)
    sandbox = direct._sandbox(tmp_path)
    sandbox.task_env_config.arm_cloud_metadata_block(root_agent=True)

    await direct._started(sandbox)
    await _settle()

    assert sandbox.task_env_config.cloud_metadata_by_helper is False
    assert _net_admin_overlay(docker)[-1] is False
    assert "no route to host" in sandbox.metadata_rules_unavailable


# --- The helper image across processes ------------------------------


class _Daemon:
    """A daemon for ensure_image, with the containerd image store's rule: an
    image no tag names is gone. Its ID, the images it holds (ID -> the
    helper and recipe labels), its tags, and how many builds ran."""

    def __init__(self, daemon_id: str = "DAEMON-ONE-0001") -> None:
        self.id = daemon_id
        self.images: dict[str, tuple[str, str]] = {}
        self.tags: dict[str, str] = {}
        self.builds = 0
        self.next_image = IMAGE_ID

    def _collect(self) -> None:
        named = set(self.tags.values())
        for image in [i for i in self.images if i not in named]:
            del self.images[image]

    async def __call__(self, args: list[str], *, timeout: float, stdin=None):
        out, code = "", 0
        if args[:1] == ["info"]:
            out = self.id + "\n"
        elif args[:2] == ["image", "inspect"]:
            held = self.images.get(args[-1])
            if held is None:
                code, out = 1, "Error: No such image\n"
            else:
                out = f"{args[-1]}\t{held[0]}\t{held[1]}\n"
        elif args[:1] == ["build"]:
            self.builds += 1
            self.images[self.next_image] = ("metadata", helper_mod.HELPER_RECIPE)
            self.tags[args[args.index("--tag") + 1]] = self.next_image
            self._collect()
            out = self.next_image + "\n"
        elif args[:1] == ["tag"]:
            if args[1] in self.images:
                self.tags[args[2]] = args[1]
                self._collect()
            else:
                code = 1
        elif args[:2] == ["image", "rm"]:
            self.tags.pop(args[-1], None)
            self._collect()
        return SimpleNamespace(stdout=out, stderr="", return_code=code)


def _new_process(monkeypatch) -> None:
    """What a new bench process starts with: no image of its own yet."""
    monkeypatch.setattr(helper_mod, "_READY", type(helper_mod._READY)())
    monkeypatch.setattr(helper_mod, "_FAILED", {})
    monkeypatch.setattr(helper_mod, "_BUILD_LOCKS", {})


@pytest.mark.asyncio
async def test_a_second_process_reuses_the_build_its_daemon_recorded(monkeypatch):
    """run-08 starts one bench process per trial, and each built
    the helper image with no cache again (4.05 s of the block). The ID is
    recorded per daemon; a later process uses it once the daemon shows it
    with this recipe's labels, and builds nothing."""
    daemon = _Daemon()
    assert await helper_mod.ensure_image(daemon, "local") == IMAGE_ID
    assert daemon.builds == 1
    path = helper_mod._state_path(daemon.id)
    assert json.loads(path.read_text()) == {
        "daemon": daemon.id,
        "recipe": helper_mod.HELPER_RECIPE,
        "image": IMAGE_ID,
    }
    _new_process(monkeypatch)
    assert await helper_mod.ensure_image(daemon, "local") == IMAGE_ID
    assert daemon.builds == 1
    assert helper_mod.image_ready("local")


@pytest.mark.asyncio
async def test_the_recorded_image_survives_removing_its_temporary_build_tag():
    """Guards 88ac33cde: unique export names leave one retained ID."""
    daemon = _Daemon()
    assert await helper_mod.ensure_image(daemon, "local") == IMAGE_ID
    assert daemon.tags == {helper_mod.kept_tag(IMAGE_ID): IMAGE_ID}
    assert IMAGE_ID in daemon.images


def test_the_record_is_private_to_this_user(tmp_path):
    folder = helper_mod.state_dir()
    helper_mod.record_image("DAEMON-ONE-0001", IMAGE_ID)
    assert stat.S_IMODE(os.lstat(folder).st_mode) == 0o700
    path = helper_mod._state_path("DAEMON-ONE-0001")
    assert stat.S_IMODE(os.lstat(path).st_mode) == 0o600
    assert [p.name for p in folder.iterdir()] == [path.name]  # no temp file left


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spoil",
    [
        "missing",
        "not-on-the-daemon",
        "other-image-id",
        "malformed",
        "another-daemon",
        "another-recipe-label",
        "file-not-private",
        "folder-not-private",
        "a-link",
    ],
)
async def test_a_record_that_cannot_be_trusted_rebuilds(monkeypatch, tmp_path, spoil):
    """Only this user's private record, naming this daemon and
    recipe, of an image the daemon holds with this recipe's labels, is
    reused. Anything else builds with no cache again, as before."""
    daemon = _Daemon()
    await helper_mod.ensure_image(daemon, "local")
    path = helper_mod._state_path(daemon.id)
    folder = helper_mod.state_dir()
    record = json.loads(path.read_text())
    if spoil == "missing":
        path.unlink()
    elif spoil == "not-on-the-daemon":
        daemon.images.clear()  # removed, or a new daemon reused the ID's path
    elif spoil == "other-image-id":
        path.write_text(json.dumps({**record, "image": OTHER_IMAGE}))
    elif spoil == "malformed":
        path.write_text("[not a record")
    elif spoil == "another-daemon":
        daemon.id = "DAEMON-TWO-0002"
    elif spoil == "another-recipe-label":
        daemon.images[IMAGE_ID] = ("metadata", "an-older-recipe")
    elif spoil == "file-not-private":
        path.chmod(0o644)
    elif spoil == "folder-not-private":
        folder.chmod(0o755)
    elif spoil == "a-link":
        elsewhere = tmp_path / "planted.json"
        elsewhere.write_text(path.read_text())
        elsewhere.chmod(0o600)
        path.unlink()
        path.symlink_to(elsewhere)
    daemon.next_image = OTHER_IMAGE
    _new_process(monkeypatch)

    assert await helper_mod.ensure_image(daemon, "local") == OTHER_IMAGE
    assert daemon.builds == 2
    # The new build is what is kept now, privately.
    assert helper_mod.recorded_image(daemon.id) == OTHER_IMAGE
    assert stat.S_IMODE(os.lstat(folder).st_mode) == 0o700
    assert stat.S_IMODE(os.lstat(helper_mod._state_path(daemon.id)).st_mode) == 0o600


@pytest.mark.asyncio
async def test_the_recorded_build_outlives_a_build_that_moves_the_shared_tag(
    monkeypatch,
):
    """On the containerd image store an image no tag names is gone, and
    every build moves the shared tag (another process's, another BenchFlow
    version's). The recorded build carries its own tag, so it stays and is
    reused: on VM 14, an interleaved A/B rebuilt 9 of 12 times without it."""
    daemon = _Daemon()
    await helper_mod.ensure_image(daemon, "local")
    assert daemon.tags[helper_mod.kept_tag(IMAGE_ID)] == IMAGE_ID
    # Another BenchFlow builds and moves the shared tag.
    daemon.next_image = OTHER_IMAGE
    await daemon(["build", "--no-cache", "--tag", helper_mod.HELPER_TAG], timeout=1)
    assert daemon.tags[helper_mod.HELPER_TAG] == OTHER_IMAGE
    _new_process(monkeypatch)
    assert await helper_mod.ensure_image(daemon, "local") == IMAGE_ID
    assert daemon.builds == 2  # the other one's; this process built nothing


@pytest.mark.asyncio
async def test_a_build_that_cannot_be_kept_is_not_recorded(monkeypatch):
    daemon = _Daemon()
    real = daemon.__call__

    async def no_tag(args, *, timeout, stdin=None):
        if args[:1] == ["tag"]:
            return SimpleNamespace(stdout="", stderr="denied", return_code=1)
        return await real(args, timeout=timeout, stdin=stdin)

    assert await helper_mod.ensure_image(no_tag, "local") == IMAGE_ID
    assert helper_mod.recorded_image(daemon.id) is None


@pytest.mark.asyncio
async def test_without_a_daemon_id_nothing_is_recorded_or_reused(monkeypatch):
    daemon = _Daemon(daemon_id="")
    await helper_mod.ensure_image(daemon, "local")
    _new_process(monkeypatch)
    await helper_mod.ensure_image(daemon, "local")
    assert daemon.builds == 2
    assert not helper_mod.state_dir().exists()


# --- Round 4: images without bash, and a main that could restart ----------------

AGENT_SKIPPED = "BF metadata-agent held=0 power= connect=skipped\n"


class NoBashEnv(HelperBackedEnv):
    """A main whose image has no bash and timeout (Alpine, BusyBox): the
    agent user's connect is skipped there, and ``ids`` is what the agent's
    own launch reads from /proc/self/status."""

    def __init__(self, *outputs: str, helper, ids: str = "BF metadata-ids 1000 1000\n"):
        super().__init__(*outputs, helper=helper)
        self.ids = ids

    def answer(self, command: str, user: str | None):
        from tests._receipt_fakes import inner_command

        if "BF metadata-ids" in inner_command(command):
            _, user = self.agent_command(command, user)
            self.calls.append(("ids", user))
            return SimpleNamespace(stdout=self.ids, stderr="", return_code=0)
        return super().answer(command, user)


def _helper_execs(docker: FakeDocker) -> list[list[str]]:
    return [c for c in docker.of("exec") if c[1] == "--user"]


@pytest.mark.asyncio
async def test_an_image_without_bash_is_proven_by_the_helpers_connect_as_the_agents_uid():
    """No bash or timeout in main: the helper, in main's namespace, makes the
    test connect as the uid and gid the agent's launch gives (1000:1000).
    It is refused and counted, so the block is on. On 55c66cf4 the same
    image ended unproven ("no bash and timeout") and the trial stopped."""
    docker = FakeDocker()
    docker.exec_outs = [
        ROOT_OK,
        "BF metadata-hits 0\n",
        "BF metadata-connect refused\n",
        "BF metadata-hits 1\n",
    ]
    env = NoBashEnv(AGENT_SKIPPED, helper=_helper(docker))

    record = await block_cloud_metadata(env, "agent")

    assert record["state"] == "on", record
    assert record["test_connect_from"] == "helper"
    assert record["test_connect_ids"] == "1000:1000"
    assert "made from BenchFlow's helper as 1000:1000" in record["why"]
    (connect,) = _helper_execs(docker)
    assert connect[2] == "1000:1000"
    framed = unwrap_receipt_command(connect[-1])
    assert framed is not None and "ftp://169.254.169.254:80/" in framed[0]
    # The ids came through the agent's own launch, as the agent user.
    assert ("ids", "agent") in env.calls


@pytest.mark.asyncio
async def test_a_root_agent_on_an_image_without_bash_is_proven_by_the_helpers_connect_as_root():
    docker = FakeDocker()
    docker.exec_outs = [
        ROOT_OK,
        "BF metadata-hits 0\n",
        "BF metadata-connect refused\n",
        "BF metadata-hits 1\n",
    ]

    class RootNoBash(RootEnv):
        def answer(self, command, user):
            from tests._receipt_fakes import inner_command

            if "BF metadata-connect" in inner_command(command):
                return SimpleNamespace(
                    stdout="BF metadata-connect skipped\n", stderr="", return_code=0
                )
            return super().answer(command, user)

    env = RootNoBash(_root_caps(CLEAN), helper=_helper(docker))

    record = await block_cloud_metadata(env, None)

    assert record["state"] == "on", record
    assert record["root_agent"] is True
    assert record["test_connect_ids"] == "0:0"
    (connect,) = _helper_execs(docker)
    assert connect[2] == "0:0"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ids",
    [
        "",
        "BF metadata-ids 1000\n",
        "BF metadata-ids x 1000\n",
        "BF metadata-ids 1000 1000\nextra\n",
    ],
)
async def test_unreadable_agent_ids_leave_the_block_unproven(ids):
    """Without the agent's own uid there is nothing to connect as: the
    helper makes no connect and the block stays unproven."""
    docker = FakeDocker()
    docker.exec_outs = [ROOT_OK, "BF metadata-hits 0\n", "BF metadata-hits 0\n"]
    env = NoBashEnv(AGENT_SKIPPED, helper=_helper(docker), ids=ids)

    record = await block_cloud_metadata(env, "agent")

    assert record["state"] == "unproven", record
    assert _helper_execs(docker) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["reached", "timeout", "failed"])
async def test_a_helper_connect_that_was_not_refused_is_unproven(answer):
    docker = FakeDocker()
    docker.exec_outs = [
        ROOT_OK,
        "BF metadata-hits 0\n",
        f"BF metadata-connect {answer}\n",
        "BF metadata-hits 0\n",
    ]
    env = NoBashEnv(AGENT_SKIPPED, helper=_helper(docker))

    record = await block_cloud_metadata(env, "agent")

    assert record["state"] == "unproven", record
    assert f"({answer})" in record["why"]


@pytest.mark.asyncio
async def test_the_helper_connect_counts_only_with_the_rules_counter():
    """A refusal the rule did not count (a host's own block) is not proof,
    from the helper either."""
    docker = FakeDocker()
    docker.exec_outs = [
        ROOT_OK,
        "BF metadata-hits 4\n",
        "BF metadata-connect refused\n",
        "BF metadata-hits 4\n",
    ]
    env = NoBashEnv(AGENT_SKIPPED, helper=_helper(docker))

    record = await block_cloud_metadata(env, "agent")

    assert record["state"] == "unproven", record
    assert "counter did not rise" in record["why"]


@pytest.mark.asyncio
async def test_an_image_without_bash_and_no_helper_stays_unproven_and_says_why():
    """Rules installed in main (no helper served them): no tool of
    BenchFlow's can make the connect, so the refusal stays."""
    env = HelperEnv(
        ROOT_OK, AGENT_SKIPPED, "BF metadata-hits 0\n", helper=[], why="helper off"
    )

    record = await block_cloud_metadata(env, "agent")

    assert record["state"] == "unproven", record
    assert "no BenchFlow metadata helper served the rules" in record["why"]
    assert env.helper_users == []


@pytest.mark.parametrize("user", ["agent", "1000", "1000:x", "0:0 --privileged", ""])
@pytest.mark.asyncio
async def test_the_helper_runs_as_a_numeric_uid_and_gid_only(user):
    docker = FakeDocker()
    helper = _helper(docker)
    helper.install_early(MAIN, "echo rules")
    assert await helper.installed(MAIN, "echo rules") is not None

    result, why = await helper.run("true", timeout_sec=5, user=user)

    assert result is None and "not a numeric user" in why
    assert _helper_execs(docker) == []


@pytest.mark.asyncio
async def test_a_main_that_could_restart_is_never_on():
    """A restarted main gets a new network namespace without the rules, so
    the block is not installed for a main whose restart policy is not no."""
    env = HelperEnv(AGENT_REFUSED, helper=[ROOT_OK, "BF metadata-hits 0\n"])
    env.restarts = "main restarts (restart policy 'always')"

    record = await block_cloud_metadata(env, "agent")

    assert record["state"] == "not_installed", record
    assert "restart policy 'always'" in record["why"]
    assert env.helper_calls == [] and env.calls == []


@pytest.mark.asyncio
async def test_a_helper_sandbox_that_cannot_read_mains_restart_policy_is_never_on():
    class NoPolicy(HelperEnv):
        metadata_main_restarts = None

    env = NoPolicy(AGENT_REFUSED, helper=[ROOT_OK])

    record = await block_cloud_metadata(env, "agent")

    assert record["state"] == "not_installed", record
    assert "restart policy cannot be read" in record["why"]


def test_main_never_restarts_wherever_the_block_applies(tmp_path):
    """The no-restart overlay is stacked after the task's own compose file
    whenever the block applies, and only then; it turns off both of
    Compose's restart settings."""
    import yaml

    from benchflow.sandbox._compose import (
        COMPOSE_NO_RESTART_PATH,
        compose_file_paths,
    )
    from benchflow.task.config import SandboxConfig

    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "docker-compose.yaml").write_text(
        "services:\n  main:\n    restart: always\n"
    )
    config = SandboxConfig()
    assert COMPOSE_NO_RESTART_PATH not in compose_file_paths(environment, config)
    config.arm_cloud_metadata_block()
    paths = compose_file_paths(environment, config)
    assert paths.index(COMPOSE_NO_RESTART_PATH) > paths.index(
        environment / "docker-compose.yaml"
    )
    main = yaml.safe_load(COMPOSE_NO_RESTART_PATH.read_text())["services"]["main"]
    assert main["restart"] == "no"
    assert main["deploy"]["restart_policy"] == {"condition": "none", "max_attempts": 0}


def test_daytona_dind_stacks_and_uploads_the_no_restart_overlay():
    from benchflow.sandbox._compose import COMPOSE_NO_RESTART_PATH
    from benchflow.sandbox.daytona_dind import _DaytonaDinD

    assert COMPOSE_NO_RESTART_PATH in _DaytonaDinD._BENCHFLOW_COMPOSE_FILES


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ps", "inspect", "expect"),
    [
        ((0, "abc123abc123\n"), (0, "no\n"), ""),
        ((0, "abc123abc123\n"), (0, "\n"), ""),
        ((0, "abc123abc123\n"), (0, "always\n"), "restart policy 'always'"),
        ((0, "abc123abc123\n"), (0, "on-failure\n"), "restart policy 'on-failure'"),
        ((0, "abc123abc123\n"), (1, ""), "could not be read"),
        ((0, ""), (0, "no\n"), "could not be read"),
        ((0, "abc123abc123\nfed321fed321\n"), (0, "no\n"), "could not be read"),
        ((0, "not-an-id\n"), (0, "no\n"), "could not be read"),
        ((1, ""), (0, "no\n"), "could not be read"),
    ],
)
async def test_daytona_dind_reads_mains_restart_policy(ps, inspect, expect):
    """Daytona DinD reads main's restart policy from the VM's daemon, as
    Docker does, so the block is never on for a main that could restart
    (reviewer A's r4 L1)."""
    from benchflow.sandbox.daytona import DaytonaSandbox
    from benchflow.sandbox.daytona_dind import _DaytonaDinD

    calls: list[list[str]] = []
    strategy = _DaytonaDinD.__new__(_DaytonaDinD)

    async def compose_exec(subcommand, timeout_sec=None, **_):
        calls.append(subcommand)
        return SimpleNamespace(return_code=ps[0], stdout=ps[1], stderr="")

    async def docker_cli(args, check=True):
        calls.append(args)
        if check and inspect[0]:
            raise RuntimeError("Cannot inspect DinD bind identities; retry startup")
        return SimpleNamespace(return_code=inspect[0], stdout=inspect[1], stderr="")

    strategy._compose_exec = compose_exec
    strategy._docker_cli = docker_cli
    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox._strategy = strategy

    why = await sandbox.metadata_main_restarts()

    assert (why == "") if not expect else (expect in why), why
    assert calls[0] == ["ps", "-q", "main"]  # compose_container_ids, running only


@pytest.mark.asyncio
async def test_a_daytona_sandbox_without_compose_has_no_restart_policy():
    from benchflow.sandbox.daytona import DaytonaSandbox

    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox._strategy = object()
    assert await sandbox.metadata_main_restarts() == ""


@pytest.mark.asyncio
async def test_a_sandbox_without_the_helper_whose_main_could_restart_is_never_on():
    """The policy read applies wherever a sandbox has it (Daytona DinD), not
    only where the helper serves the rules."""

    class RestartingDinD(FakeEnv):
        async def metadata_main_restarts(self) -> str:
            return "main restarts (restart policy 'always')"

    env = RestartingDinD(ROOT_OK, AGENT_REFUSED, "BF metadata-hits 1\n")

    record = await block_cloud_metadata(env, "agent")

    assert record["state"] == "not_installed", record
    assert "restart policy 'always'" in record["why"]
    assert env.calls == []
