"""Guards the fix at 363d815b before ownership can launder aliases."""

import subprocess
from types import SimpleNamespace

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult
from tests.test_sandbox_hardening import _make_env, _make_task


@pytest.mark.parametrize(
    "sandbox_user,agent_paths",
    [("agent", None), (None, ("/app",))],
    ids=["shared", "separate"],
)
async def test_plugin_policy_retains_workspace_mount_aliases_before_freeze(
    monkeypatch, sandbox_user, agent_paths
):
    events = []
    captured = {}
    mounts = (
        "1 0 0:1 / / rw - overlay overlay rw\n"
        "2 1 0:2 / /app/plugins rw - tmpfs tmpfs rw\n"
        "3 1 0:2 / /opt/plugin-view rw - tmpfs tmpfs rw\n"
    )

    def execute(command, **kwargs):
        if "/proc/self/mountinfo" in command:
            events.append("mountinfo")
            return ExecResult(stdout=mounts + "# end\n\0", stderr="", return_code=0)
        return ExecResult(stdout="", stderr="", return_code=0)

    async def freeze(*args, **kwargs):
        events.append("freeze")

    async def build(*args, **kwargs):
        captured.update(kwargs)
        return {}

    async def nothing(*args, **kwargs):
        return ""

    for name in (
        "_kill_sandbox_user_procs",
        "_reclaim_disk",
        "_trusted_verifier_bash_env",
    ):
        monkeypatch.setattr(lockdown, name, nothing)
    monkeypatch.setattr(lockdown, "_freeze_workspace", freeze)
    monkeypatch.setattr(lockdown, "_build_verifier_env", build)
    await lockdown.harden_before_verify(
        _make_env(execute),
        _make_task(),
        sandbox_user=sandbox_user,
        workspace="/app",
        agent_paths=agent_paths,
    )
    assert "/opt/plugin-view" in captured["blocked"].paths()
    assert events.index("mountinfo") < events.index("freeze")


@pytest.mark.parametrize(
    "source,alias,expected",
    [
        ("/", "/", "/view"),
        ("/project", "/", "/view/project"),
        ("/", "/nested", "/view"),
        ("/project", "/project/subdir", "/view"),
        ("/project", "/project2", None),
        ("/space\\040name", "/space\\040name", "/view"),
        ("/back\\134slash", "/back\\134slash", "/view"),
    ],
)
@pytest.mark.parametrize("backend", ["overlay", "ext4", "tmpfs"])
def test_all_mount_root_intersections_are_blocked(source, alias, expected, backend):
    from benchflow.sandbox._mount_aliases import blocked_mount_aliases

    info = (
        "1 0 0:1 / / rw - overlay overlay rw\n"
        f"2 1 0:2 {source} /app/plugins rw shared:5 - {backend} dummy rw\n"
        f"3 1 0:2 {alias} /view ro - {backend} dummy ro\n"
        "4 1 0:3 / /unrelated rw - tmpfs dummy rw\n"
    )
    blocked = blocked_mount_aliases(info, ("/app",))
    if expected:
        assert expected in blocked
        # Both discovery and the armed runtime guard use this same primitive.
        from benchflow.sandbox._pytest_plugin_guard import untrusted_reason

        assert untrusted_reason(expected + "/plugin.py", blocked)
    else:
        assert not any(p.startswith("/view") for p in blocked)
    assert "/unrelated" not in blocked


@pytest.mark.parametrize(
    "output,code", [("", 0), ("broken\n# end\n\0", 0), ("# end\n", 0), ("", 1)]
)
async def test_unknown_mount_provenance_refuses_hardening(output, code):
    from unittest.mock import AsyncMock

    env = SimpleNamespace(
        exec=AsyncMock(
            return_value=ExecResult(stdout=output, stderr="", return_code=code)
        )
    )
    with pytest.raises(RuntimeError, match="mount"):
        await lockdown._blocked_before_freeze(env, ("/app",))


async def test_shell_capture_includes_physical_workspace_paths(tmp_path):
    physical = tmp_path / "physical"
    physical.mkdir()
    link = tmp_path / "workspace link"
    link.symlink_to(physical)

    async def execute(command, **kwargs):
        result = subprocess.run(["sh", "-c", command], capture_output=True, text=True)
        return ExecResult(
            stdout=result.stdout, stderr=result.stderr, return_code=result.returncode
        )

    found = await lockdown._blocked_before_freeze(
        SimpleNamespace(exec=execute), (str(link),)
    )
    assert str(physical) in found


def test_discovery_and_runtime_refuse_an_alias_plugin(tmp_path, monkeypatch):
    import sys

    from benchflow.sandbox import _pytest_plugin_guard as guard
    from benchflow.sandbox._mount_aliases import blocked_mount_aliases
    from tests.test_verifier_uv_state import install_plugin

    alias = tmp_path / "view"
    alias.mkdir()
    install_plugin(alias, "raise AssertionError('agent plugin must never import')\n")
    mounts = (
        "1 0 0:1 / / rw - overlay overlay rw\n"
        "2 1 0:2 / /app/plugins rw - tmpfs dummy rw\n"
        f"3 1 0:2 / {alias} rw - tmpfs dummy rw\n"
    )
    blocked = blocked_mount_aliases(mounts, ("/app",))
    monkeypatch.setattr(sys, "path", [str(alias), *sys.path])
    monkeypatch.setattr(guard, "_BENCHFLOW_OWNERSHIP", False)
    monkeypatch.setattr(guard, "_BENCHFLOW_BLOCKED", blocked)
    receipt = guard.discover(blocked, ["ctrf"])
    assert "ctrf" in receipt["rejected"]
    with pytest.raises(guard.Rejected, match="ctrf"):
        guard._validate(["ctrf"])
    with pytest.raises(guard.Rejected, match="ctrf_model"):
        guard._validate(["ctrf_model"], entry_points=False)
    assert "ctrf_model" not in sys.modules


def test_stacked_mount_targets_do_not_depend_on_listing_order():
    from benchflow.sandbox._mount_aliases import blocked_mount_aliases

    records = [
        "1 0 0:1 / / rw - overlay overlay rw",
        "2 1 0:2 / /app rw - tmpfs dummy rw",
        "3 1 0:3 / /app rw - tmpfs dummy rw",
        "4 1 0:2 / /view-one rw - tmpfs dummy rw",
        "5 1 0:3 / /view-two rw - tmpfs dummy rw",
    ]
    for ordered in (records, records[::-1]):
        found = blocked_mount_aliases("\n".join(ordered), ("/app/project",))
        assert "/view-one/project" in found
        assert "/view-two/project" in found
