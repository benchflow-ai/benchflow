"""V3-M129-C1: freezing must preserve provenance in the verifier's namespace."""

import posixpath
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow._utils.trusted_path import PATH_QUERY_CMD
from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult
from tests._receipt_fakes import receipt_aware


@pytest.mark.parametrize(
    "source",
    [
        "/volume/work",
        "/volume/work with space",
        "/volume/work\twith\ttabs",
        "/volume/work\nwith\nlines",
        "/volume/work\\backslash",
        "/volume/work\u2028separator",
    ],
)
@pytest.mark.parametrize("mapping", ["same", "parent", "child", "nested", "stacked"])
def test_shared_coordinates_cover_each_target_mount_form(source, mapping):
    from benchflow.sandbox._mount_aliases import blocked_mount_aliases

    main = mount("/", "/", "0:1", 1) + mount(source, "/app")
    target = mount("/", "/", "0:2", 1)
    if mapping == "parent":
        target += mount("/volume", "/target")
        expected = "/target/" + posixpath.basename(source)
    elif mapping == "child":
        target += mount(source + "/bin", "/target")
        expected = "/target"
    elif mapping == "nested":
        main += mount("/nested", "/app/submount", "8:2", 3)
        target += mount("/nested", "/target", "8:2")
        expected = "/target"
    else:
        target += mount(source, "/target")
        if mapping == "stacked":
            main += mount("/unrelated", "/app", "8:3", 3)
            target += mount("/unrelated", "/other", "8:3", 3)
        expected = "/target"
    found = blocked_mount_aliases(main, ("/app",), target_mountinfo=target)
    assert expected in found
    if mapping == "stacked":
        assert "/other" in found
    unrelated = mount("/", "/", "0:3", 1) + mount(source, "/safe", "8:9")
    assert "/safe" not in blocked_mount_aliases(
        main, ("/app",), target_mountinfo=unrelated
    )


@pytest.mark.parametrize(
    "target", ["", "garbage\n", "2 1 8:1 /volume /view rw - ext4 /dev/test rw\n"]
)
def test_target_mount_proof_must_be_complete(target):
    from benchflow.sandbox._mount_aliases import blocked_mount_aliases

    with pytest.raises(ValueError):
        blocked_mount_aliases(
            mount("/", "/", "0:1", 1), ("/app",), target_mountinfo=target
        )


def mount(root, target, device="8:1", number=2):
    def escape(value):
        for before, after in [
            ("\\", "\\134"),
            (" ", "\\040"),
            ("\t", "\\011"),
            ("\n", "\\012"),
        ]:
            value = value.replace(before, after)
        return value

    return (
        f"{number} 1 {device} {escape(root)} {escape(target)} rw - ext4 /dev/test rw\n"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize(
    "alias",
    [
        "/opt/shared",
        "/opt/target-only",
        "/opt/with space",
        "/opt/with\\slash",
        "/opt/with\u2028separator",
    ],
)
@pytest.mark.parametrize("suffix", ["/bin", "/bin/../bin", "//bin/"])
async def test_target_path_excludes_shared_agent_content_before_freeze(
    monkeypatch, alias, suffix, separate
):
    main_info = mount("/", "/", "0:1", 1) + mount("/volume/work", "/app")
    if alias == "/opt/shared":
        main_info += mount("/volume/work", alias, number=3)
    target_info = mount("/", "/", "0:2", 1) + mount("/volume/work", alias)
    events = []
    selected = []
    fresh = []

    @receipt_aware
    async def answer(command, **kwargs):
        service = kwargs.get("service", "main")
        if "printf 'complete\\n'" in command:
            return ExecResult(stdout="complete\n", return_code=0)
        if "/proc/self/mountinfo" in command:
            events.append(("mounts", service))
            return ExecResult(
                stdout=(main_info if service == "main" else target_info) + "# end\n\0",
                return_code=0,
            )
        if command == PATH_QUERY_CMD:
            return ExecResult(
                stdout=(
                    "/usr/bin:/bin"
                    if service == "main"
                    else alias + suffix + ":/usr/bin:/bin"
                )
                + "\n",
                return_code=0,
            )
        return ExecResult(stdout="", return_code=0)

    async def judge(env, raw, blocked, agent, *, fresh_image=False):
        assert isinstance(blocked, lockdown.PathExclusions)
        fresh.append(fresh_image)
        blocked = blocked.paths()
        entries = lockdown._path_entries_to_judge(raw)
        dropped = [
            (entry, "agent content")
            for entry in entries
            if any(
                posixpath.normpath(entry) == p
                or posixpath.normpath(entry).startswith(p.rstrip("/") + "/")
                for p in blocked
            )
        ]
        kept = tuple(entry for entry in entries if entry not in dict(dropped))
        return lockdown.TrustedPathReport(
            path=lockdown._merge_trusted_verifier_path(list(kept)),
            kept=kept,
            dropped=tuple(dropped),
        )

    async def freeze(*args, **kwargs):
        events.append(("freeze", "main"))

    async def check_tools(env, report, verifier):
        selected.append(report.path)

    monkeypatch.setattr(lockdown, "_judge_trusted_entries", judge)
    monkeypatch.setattr(lockdown, "_freeze_workspace", freeze)
    monkeypatch.setattr(lockdown, "_check_required_path_tools", check_tools)
    monkeypatch.setattr(
        lockdown,
        "_install_refusal_stubs",
        AsyncMock(side_effect=lambda env, report: report),
    )
    for name, value in [
        ("_kill_sandbox_user_procs", None),
        ("clear_verifier_output_dir", None),
        ("ensure_legacy_app_dir", None),
        ("_reclaim_disk", None),
        ("_probe_verifier_reads", {}),
        ("_trusted_verifier_bash_env", ""),
        ("_trusted_verifier_pythonpath", ""),
        ("_distro_pip_env", {}),
        ("_isolate_verifier_tool_state", {}),
        ("_discover_pytest_plugin_flags", ""),
        ("_discover_pytest_plugins", ([], [])),
    ]:
        monkeypatch.setattr(lockdown, name, AsyncMock(return_value=value))
    monkeypatch.setattr(
        "benchflow.agents.credentials.muse_login_present", AsyncMock(return_value=False)
    )
    verifier = SimpleNamespace(
        service="target", user="root", env={}, required_tools=["cmp"], optional_tools=[]
    )
    task = SimpleNamespace(
        config=SimpleNamespace(verifier=verifier, keep_services=None), task_dir=None
    )
    monkeypatch.setattr("benchflow.task.keep_services.trial_keep", lambda *args: None)
    await lockdown.harden_before_verify(
        SimpleNamespace(
            exec=answer,
            _benchflow_path_inspectors={
                "main": "/etc/fake-main",
                "target": "/etc/fake-target",
            },
        ),
        task,
        None if separate else "agent",
        "/app",
        agent_paths=("/app",) if separate else None,
    )
    assert alias not in selected[0].split(":"), selected
    assert alias + suffix not in selected[0].split(":"), selected
    # A separate verifier's image is judged as fresh (#69); the alias stays out.
    assert fresh and set(fresh) == {separate}
    assert ("mounts", "target") in events
    assert events.index(("mounts", "target")) < events.index(("freeze", "main"))


@pytest.mark.asyncio
async def test_explicit_main_alias_coordinates_reach_target_judgment(monkeypatch):
    """Guard CONFIRM142-2 at 2fcc60b4: the direct build API projects identities."""
    seen = []

    async def judge(env, user, workspace, **kwargs):
        from benchflow._utils.trusted_path import TrustedPathReport

        seen.append(kwargs.get("blocked", ()))
        return TrustedPathReport(path="/usr/bin:/bin")

    monkeypatch.setattr(lockdown, "_judge_verifier_path", judge)
    for name, value in [
        ("_probe_verifier_reads", {}),
        ("_trusted_verifier_pythonpath", ""),
        ("_distro_pip_env", {}),
        ("_isolate_verifier_tool_state", {}),
        ("_discover_pytest_plugin_flags", ""),
        ("_discover_pytest_plugins", ([], [])),
        ("_check_required_path_tools", None),
    ]:
        monkeypatch.setattr(lockdown, name, AsyncMock(return_value=value))
    monkeypatch.setattr(
        lockdown,
        "_install_refusal_stubs",
        AsyncMock(side_effect=lambda env, report: report),
    )
    task = SimpleNamespace(
        config=SimpleNamespace(
            verifier=SimpleNamespace(service="target", user="root", env={})
        )
    )

    @receipt_aware
    async def answer(command, **kwargs):
        if "printf 'complete\\n'" in command:
            return ExecResult(stdout="complete\n", return_code=0)
        info = mount("/", "/", "0:2" if kwargs.get("service") else "0:1", 1)
        info += mount(
            "/shared", "/target-view" if kwargs.get("service") else "/opt/shared"
        )
        return ExecResult(stdout=info + "# end\n\0", return_code=0)

    await lockdown._build_verifier_env(
        SimpleNamespace(
            exec=answer,
            _benchflow_path_inspectors={
                "main": "/etc/fake-main",
                "target": "/etc/fake-target",
            },
        ),
        task,
        "agent",
        "/app",
        blocked=("/app", "/opt/shared"),
        bash_env="",
    )
    assert isinstance(seen[1], lockdown.PathExclusions)
    assert "/target-view" in seen[1].paths()
    assert "/opt/shared" not in seen[1].paths()
