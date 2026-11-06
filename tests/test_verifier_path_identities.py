"""Guard V3-CONFIRM142-1/2 against the regressions at 2fcc60b4.

The filesystem and symlink walks are real. Mount tables model separate
container namespaces without requiring mounts or a privileged test runner.
Only the ownership predicate is bypassed: these tests isolate provenance
after the ownership freeze; container replays check actual root ownership.
"""

import dataclasses
import os
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult
from tests._receipt_fakes import receipt_aware
from tests.test_verifier_target_aliases import mount


@pytest.fixture
def inspector_tools(tmp_path):
    folder = tmp_path / "inspector"
    folder.mkdir(mode=0o700)
    for name in (
        "sh",
        "stat",
        "readlink",
        "ls",
        "id",
        "find",
        "mkdir",
        "ln",
        "chmod",
        "cat",
        "od",
        "tr",
        "cksum",
        "getfacl",
    ):
        source = shutil.which(name)
        if source:
            shutil.copyfile(source, folder / name)
            (folder / name).chmod(0o700)
    return str(folder)


@pytest.fixture
def frozen_permissions(monkeypatch, inspector_tools):
    # Each case owns its synthetic mount namespace, including its system route.
    monkeypatch.setattr(lockdown, "_SAFE_VERIFIER_PATH_PARTS", ())
    original = lockdown._trusted_path_extras_cmd
    monkeypatch.setattr(
        lockdown,
        "_trusted_path_extras_cmd",
        lambda path, blocked, agent=None, **kwargs: original(
            path,
            dataclasses.replace(blocked, inspector=inspector_tools),
            agent,
            **kwargs,
        ),
    )
    monkeypatch.setattr(lockdown, "_path_inspector", lambda env: inspector_tools)
    monkeypatch.setattr(
        lockdown,
        "_TRUSTED_PATH_EXTRAS_SCRIPT",
        lockdown._TRUSTED_PATH_EXTRAS_SCRIPT.replace(
            "IFS=$nl\nfor d", "bf_may() { return 1; }\nIFS=$nl\nfor d"
        ),
    )


async def capture(main, target, blocked):
    @receipt_aware
    async def execute(command, **kwargs):
        if "printf 'complete\\n'" in command:
            return ExecResult(stdout="complete\n", return_code=0)
        assert "/proc/self/mountinfo" in command
        info = target if kwargs.get("service") == "target" else main
        return ExecResult(stdout=info + "# end\n\0", return_code=0)

    return await lockdown._service_exclusions_before_freeze(
        SimpleNamespace(exec=execute), blocked, "target"
    )


def judge(folder, blocked):
    # Use a different numeric user so locally created fixture entries are not
    # themselves rejected as owned by the sandbox user.
    agent = "nobody" if os.getuid() != 65534 else "root"
    command = lockdown._trusted_path_extras_cmd(str(folder), blocked, agent)
    result = subprocess.run(
        ["/bin/sh", "-c", command], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
    return lockdown._parse_trusted_path_output(result.stdout, [str(folder)])


def executable(path):
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)


@pytest.mark.parametrize("backend", ["overlay", "ext4", "tmpfs"])
@pytest.mark.parametrize(
    "route",
    ["file-mount", "absolute", "relative", "chain-in", "chain-out", "directory-hop"],
)
async def test_each_executable_route_element_retains_exclusion(
    tmp_path, frozen_permissions, backend, route
):
    tools = tmp_path / "tools"
    excluded = tmp_path / "excluded"
    safe = tmp_path / "safe"
    for path in (tools, excluded, safe):
        path.mkdir()
    executable(safe / "honest")
    executable(excluded / "planted")
    if route == "file-mount":
        executable(tools / "cmp")
        target_alias = tools / "cmp"
        source = "/volume/work/bin/cmp"
    else:
        target_alias = excluded
        source = "/volume/work"
        if route == "absolute":
            (tools / "cmp").symlink_to(excluded / "planted")
        elif route == "relative":
            (tools / "cmp").symlink_to("../excluded/planted")
        elif route == "chain-in":
            (tools / "cmp").symlink_to("../safe/hop")
            (safe / "hop").symlink_to("../excluded/planted")
        elif route == "chain-out":
            (tools / "cmp").symlink_to("../excluded/hop")
            (excluded / "hop").symlink_to("../safe/honest")
        else:
            (tools / "cmp").symlink_to("../excluded/hop/honest")
            (excluded / "hop").symlink_to(safe, target_is_directory=True)
    main = mount("/", "/", "0:1", 1) + mount("/volume/work", "/app")
    target = mount("/", "/", "0:2", 1) + mount(source, str(target_alias))
    main, target = (info.replace("ext4", backend) for info in (main, target))
    _, blocked = await capture(main, target, ("/app",))
    kept, _, dropped = judge(tools, blocked)
    assert not kept, f"agent-controlled {route} accepted: {kept}"
    assert str(tools) in dict(dropped)


@pytest.mark.parametrize("same_device", [False, True])
@pytest.mark.parametrize("source", ["/volume/work", "/volume/work with space"])
async def test_main_alias_name_cannot_exclude_unrelated_target_tool(
    tmp_path, frozen_permissions, same_device, source
):
    alias = tmp_path / "mirror"
    tools = alias / "bin"
    tools.mkdir(parents=True)
    executable(tools / "honest")
    main = (
        mount("/", "/", "0:1", 1)
        + mount(source, "/app")
        + mount(source, str(alias), number=3)
    )
    target = mount("/", "/", "0:2", 1) + mount(
        "/independent", str(alias), "8:1" if same_device else "8:9"
    )
    _, blocked = await capture(main, target, ("/app",))
    kept, _, dropped = judge(tools, blocked)
    assert kept == [str(tools)], dropped
    report = lockdown.TrustedPathReport(
        path=":".join([*kept, "/usr/bin", "/bin"]), dropped=tuple(dropped)
    )
    # The reported regression refused an honest task with no tool contract.
    await lockdown._check_required_path_tools(
        None, report, SimpleNamespace(required_tools=None)
    )


async def test_target_seeds_its_own_exclusions(tmp_path, frozen_permissions):
    local = tmp_path / "local"
    local.mkdir()
    executable(local / "cmp")
    main = mount("/", "/", "0:1", 1)
    target = mount("/", "/", "0:2", 1)
    _, blocked = await capture(main, target, (str(local),))
    assert judge(local, blocked)[0] == []


async def test_honest_symlink_chain_stays_usable(tmp_path, frozen_permissions):
    tools = tmp_path / "tools"
    safe = tmp_path / "safe"
    tools.mkdir()
    safe.mkdir()
    executable(safe / "cmp")
    (tools / "cmp").symlink_to("../safe/hop")
    (safe / "hop").symlink_to("cmp")
    main = mount("/", "/", "0:1", 1) + mount("/volume/work", "/app")
    target = mount("/", "/", "0:2", 1)
    _, blocked = await capture(main, target, ("/app",))
    assert judge(tools, blocked)[0] == [str(tools)]


@pytest.mark.parametrize("overmount", ["file", "directory"])
async def test_unrelated_overmount_does_not_inherit_hidden_coordinates(
    tmp_path, frozen_permissions, overmount
):
    tools = tmp_path / "tools"
    tools.mkdir()
    executable(tools / "cmp")
    source = str(tools / "cmp") if overmount == "file" else str(tools)
    # The same device exposes /volume via a parent mount, but an unrelated
    # mount hides exactly the otherwise excluded path in the target.
    main = mount("/", "/", "0:1", 1) + mount(source, "/app")
    target = mount("/", "/", "8:1", 1) + mount("/honest", source, "8:9")
    _, blocked = await capture(main, target, ("/app",))
    assert judge(tools, blocked)[0] == [str(tools)]


@pytest.mark.parametrize("file_mount", [False, True])
async def test_always_kept_system_route_cannot_restore_an_excluded_tool(
    tmp_path, frozen_permissions, monkeypatch, file_mount
):
    tools = tmp_path / "system"
    agent = tmp_path / "agent"
    tools.mkdir()
    agent.mkdir()
    executable(agent / "cmp")
    if file_mount:
        executable(tools / "cmp")
        alias = tools / "cmp"
        source = "/volume/work/cmp"
    else:
        (tools / "cmp").symlink_to("../agent/cmp")
        alias = agent
        source = "/volume/work"
    main = mount("/", "/", "0:1", 1) + mount("/volume/work", "/app")
    target = mount("/", "/", "0:2", 1) + mount(source, str(alias))
    _, blocked = await capture(main, target, ("/app",))
    monkeypatch.setattr(lockdown, "_SAFE_VERIFIER_PATH_PARTS", (str(tools),))
    result = subprocess.run(
        ["/bin/sh", "-c", lockdown._trusted_path_extras_cmd("", blocked)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert "excluded system PATH" in result.stderr


@pytest.mark.parametrize(
    "spelling",
    [
        "with space",
        "with\ttab",
        "with\nline",
        "with\\slash",
        "with'quote",
        "with*[glob]",
        "with\u2028separator",
    ],
)
async def test_coordinate_encoding_preserves_path_bytes(
    tmp_path, frozen_permissions, spelling
):
    tools = tmp_path / "tools"
    tools.mkdir()
    executable(tools / "cmp")
    main = mount("/", "/", "0:1", 1) + mount("/volume/" + spelling, "/app")
    target = mount("/", "/", "0:2", 1) + mount(
        "/volume/" + spelling + "/cmp", str(tools / "cmp")
    )
    _, blocked = await capture(main, target, ("/app",))
    assert judge(tools, blocked)[0] == []


@pytest.mark.parametrize("file", [False, True])
async def test_capture_resolves_physical_seed_in_its_own_container(
    tmp_path, file, inspector_tools
):
    physical = tmp_path / "physical\n"
    if file:
        executable(physical)
    else:
        physical.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(physical)

    async def execute(command, **kwargs):
        result = subprocess.run(
            ["/bin/sh", "-c", command], capture_output=True, text=True
        )
        return ExecResult(
            stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
        )

    exclusions = await lockdown._path_exclusions_before_freeze(
        SimpleNamespace(
            exec=execute, _benchflow_path_inspectors={"main": inspector_tools}
        ),
        (str(alias),),
    )
    assert str(physical) in exclusions.paths()
    assert str(alias) in exclusions.paths()


@pytest.mark.parametrize("link", ["directory", "dangling", "loop"])
async def test_system_non_tools_do_not_refuse_honest_verification(
    tmp_path, frozen_permissions, monkeypatch, link
):
    """Guard CONFIRM142-2's preservation rule in f48cf6da's system-route scan."""
    tools = tmp_path / "system"
    excluded = tmp_path / "excluded"
    tools.mkdir()
    excluded.mkdir()
    executable(tools / "cmp")
    destination = {
        "directory": excluded,
        "dangling": excluded / "absent",
        "loop": tools / "not-a-tool",
    }[link]
    (tools / "not-a-tool").symlink_to(destination)
    main = mount("/", "/", "0:1", 1) + mount("/volume/work", "/app")
    target = mount("/", "/", "0:2", 1) + mount("/volume/work", str(excluded))
    _, blocked = await capture(main, target, ("/app",))
    monkeypatch.setattr(lockdown, "_SAFE_VERIFIER_PATH_PARTS", (str(tools),))
    result = subprocess.run(
        ["/bin/sh", "-c", lockdown._trusted_path_extras_cmd("", blocked)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("spelling", ["__BF_EXCLUSIONS__", "__BF_SYSTEM_EXCLUSIONS__"])
async def test_alias_names_are_data_even_when_they_match_script_placeholders(
    tmp_path, frozen_permissions, spelling
):
    """Guard the coordinate compiler in 6e171fd4 against template collisions."""
    tools = tmp_path / spelling
    tools.mkdir()
    executable(tools / "cmp")
    main = mount("/", "/", "0:1", 1) + mount("/volume/work", "/app")
    target = mount("/", "/", "0:2", 1) + mount("/volume/work/cmp", str(tools / "cmp"))
    _, blocked = await capture(main, target, ("/app",))
    assert judge(tools, blocked)[0] == []
