"""Guard V3-CONFIRM157-1/2/3 against d63d413c's PATH regressions."""

import dataclasses
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox._mount_aliases import PathExclusions
from tests.test_verifier_path_identities import executable, judge
from tests.test_verifier_path_identities import frozen_permissions as frozen_permissions
from tests.test_verifier_path_identities import inspector_tools as inspector_tools
from tests.test_verifier_target_aliases import mount


class Shell:
    async def exec(self, command, **kwargs):
        result = subprocess.run(
            ["/bin/sh", "-c", command], capture_output=True, text=True, timeout=60
        )
        return ExecResult(
            stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
        )


@pytest.mark.parametrize("link", ["absolute", "relative", "chain"])
@pytest.mark.parametrize("ending", ["\nv1", "\n", "\n\n"])
def test_honest_newline_symlink_is_a_valid_system_tool(
    tmp_path, frozen_permissions, monkeypatch, link, ending
):
    tools = tmp_path / "tools"
    safe = tmp_path / "safe"
    tools.mkdir()
    safe.mkdir()
    destination = safe / ("check" + ending)
    executable(destination)
    target = str(destination) if link == "absolute" else "../safe/" + destination.name
    if link == "chain":
        (safe / "hop\n").symlink_to(destination.name)
        target = "../safe/hop\n"
    (tools / "unused").symlink_to(target)
    monkeypatch.setattr(lockdown, "_SAFE_VERIFIER_PATH_PARTS", (str(tools),))
    exclusions = PathExclusions.capture(mount("/", "/", "0:1", 1), ("/app",))
    assert judge(tools, exclusions)[0] == []  # system entries need no extra record


@pytest.mark.parametrize("route", ["direct", "relative", "chain-out"])
async def test_hardlink_identity_survives_freeze(tmp_path, frozen_permissions, route):
    workspace = tmp_path / "workspace"
    tools = tmp_path / "tools"
    hidden = tmp_path / "hidden"
    for folder in (workspace, tools, hidden):
        folder.mkdir()
    executable(workspace / "cmp")
    os.link(workspace / "cmp", hidden / "cmp")
    if route == "direct":
        os.link(hidden / "cmp", tools / "cmp")
    else:
        (tools / "cmp").symlink_to(
            "../hidden/hop" if route == "chain-out" else "../hidden/cmp"
        )
        if route == "chain-out":
            (hidden / "hop").symlink_to("cmp")
    captured = await lockdown._path_exclusions_before_freeze(Shell(), (str(workspace),))
    # Changing ownership is irrelevant to the persisted device/inode identity.
    assert judge(tools, captured)[0] == [], "hard-linked agent bytes became trusted"


def test_excluded_inspection_utility_never_executes(
    tmp_path, frozen_permissions, monkeypatch
):
    live = tmp_path / "live"
    tools = tmp_path / "tools"
    live.mkdir()
    tools.mkdir()
    marker = tmp_path / "dummy-marker"
    real = shutil.which("readlink")
    assert real
    for name in ("sh", "ls", "id"):
        (live / name).symlink_to(shutil.which(name))
    (live / "readlink").write_text(
        f'#!/bin/sh\nprintf touched > {shlex.quote(str(marker))}\nexec {shlex.quote(real)} "$@"\n'
    )
    (live / "readlink").chmod(0o755)
    (tools / "probe").symlink_to("/usr/bin/true")
    monkeypatch.setattr(lockdown, "_SAFE_VERIFIER_PATH_PARTS", (str(tools), str(live)))
    monkeypatch.setattr(
        lockdown,
        "_system_probe",
        lambda script, *args: (
            f"PATH={shlex.quote(str(live))} /bin/sh -c {shlex.quote(script)} "
            + shlex.join(args)
        ),
    )
    captured = PathExclusions.capture(
        mount("/", "/", "0:1", 1), (str(live / "readlink"),)
    )
    result = subprocess.run(
        ["/bin/sh", "-c", lockdown._trusted_path_extras_cmd("", captured)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert not marker.exists(), (
        "inspection executed the excluded utility before refusal"
    )


@pytest.mark.parametrize("kind", ["file", "symlink-hop", "non-executable"])
@pytest.mark.parametrize("local", [False, True])
async def test_service_projection_carries_inodes_and_target_local_seeds(
    tmp_path, frozen_permissions, kind, local
):
    main = tmp_path / "main"
    target = tmp_path / "target"
    tools = tmp_path / "tools"
    safe = tmp_path / "safe"
    for folder in (main, target, tools, safe):
        folder.mkdir()
    executable(safe / "honest")
    source = (target if local else main) / "cmp"
    if kind == "symlink-hop":
        source.symlink_to(safe / "honest")
        os.link(source, tools / "cmp", follow_symlinks=False)
    else:
        executable(source)
        os.link(source, tools / "cmp")
        if kind == "non-executable":
            source.chmod(0o644)
    _, captured = await lockdown._service_exclusions_before_freeze(
        Shell(), (str(main),), "target", target_seeds=(str(target),)
    )
    if kind == "non-executable":
        source.chmod(0o755)
    assert judge(tools, captured)[0] == []
    (tools / "cmp").unlink()
    executable(tools / "cmp")
    assert judge(tools, captured)[0] == [str(tools)], (
        "unrelated target inode was excluded"
    )


async def test_same_inode_number_on_different_device_is_not_an_alias(
    tmp_path, frozen_permissions
):
    tools = tmp_path / "tools"
    tools.mkdir()
    executable(tools / "cmp")
    st = (tools / "cmp").stat()
    captured = dataclasses.replace(
        PathExclusions.capture(mount("/", "/", "0:1", 1), ("/app",)),
        inodes=frozenset({(st.st_dev + 1, st.st_ino)}),
        inspector=lockdown._path_inspector(Shell()),
    )
    captured = await lockdown._index_exclusions(Shell(), captured)
    assert judge(tools, captured)[0] == [str(tools)]


@pytest.mark.parametrize("missing", [True, False])
async def test_missing_or_partial_capture_fails_closed_before_live_tools(missing):
    from types import SimpleNamespace

    calls = []

    async def execute(command, **kwargs):
        calls.append(command)
        return ExecResult(stdout="", return_code=0)

    env = SimpleNamespace(exec=execute)
    if not missing:
        env._benchflow_path_inspectors = {"main": "/etc/missing-protected-inspector"}
    with pytest.raises(RuntimeError):
        await lockdown._path_exclusions_before_freeze(env, ("/app",))
    if missing:
        assert calls == []


def test_checkpoint_carries_only_the_main_inspector_and_rejects_alias_names():
    from types import SimpleNamespace

    from benchflow.sandbox import _path_inspector_state as state
    from benchflow.sandbox.protocol import SandboxImage

    main = "/etc/.runtime-bash-env-" + "1" * 32
    target = "/etc/.runtime-bash-env-" + "2" * 32
    env = SimpleNamespace(_benchflow_path_inspectors={"main": main, "target": target})
    image = state.checkpoint(env, SandboxImage("docker", "dummy"))
    assert state.read(image) == main
    assert target not in str(image.meta)
    state.restored(env, None)
    assert env._benchflow_path_inspectors == {"target": target}
    state.restored(env, state.read(image))
    assert env._benchflow_path_inspectors == {"main": main, "target": target}
    for bad in ("/app/copied-name", main + "/../alias", main + "\n"):
        with pytest.raises(ValueError):
            state.read(
                SandboxImage("docker", "dummy", {"benchflow.path-inspector.main": bad})
            )


async def test_checkpoint_inode_renumbering_uses_the_protected_reference(
    tmp_path, frozen_permissions
):
    """Guard ac543f61: a snapshot can renumber an unlinked workspace inode."""
    tools = tmp_path / "tools"
    tools.mkdir()
    executable(tools / "cmp")
    st = (tools / "cmp").stat()
    inspector = Path(lockdown._path_inspector(Shell()))
    (inspector / "before-links").mkdir()
    # Snapshot reconstruction preserves hard links but can renumber them.
    # Only the private reference survives after the agent unlinks its name.
    (inspector / "before").write_text(f"{st.st_dev}:{st.st_ino + 100000}\n")
    os.link(tools / "cmp", inspector / "before-links" / "pre-snapshot-name")
    captured = await lockdown._path_exclusions_before_freeze(
        Shell(), (str(tmp_path / "removed-workspace"),)
    )
    assert judge(tools, captured)[0] == []


async def test_large_inode_sets_use_bounded_transfer_and_local_membership(
    tmp_path, frozen_permissions
):
    """Guard CONFIRM157-2's numeric index against exec argument-size limits."""
    tools = tmp_path / "tools"
    tools.mkdir()
    executable(tools / "cmp")
    st = (tools / "cmp").stat()
    captured = dataclasses.replace(
        PathExclusions.capture(mount("/", "/", "0:1", 1), ("/app",)),
        inodes=frozenset((st.st_dev, st.st_ino + i) for i in range(2501)),
        inspector=lockdown._path_inspector(Shell()),
    )
    captured = await lockdown._index_exclusions(Shell(), captured)
    assert len(captured.shell_predicate()) < 4000
    assert judge(tools, captured)[0] == []
