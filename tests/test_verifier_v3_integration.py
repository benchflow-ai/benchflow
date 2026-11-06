"""Guards V3's integration of 85b87bc9 after ebd28778 (PY8 batch 129)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow._utils.trusted_path import PATH_QUERY_CMD, TrustedPathReport
from benchflow.rollout._setup import _refused_programs_run
from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult
from tests._receipt_fakes import receipt_aware


@pytest.mark.asyncio
async def test_missing_path_receipt_cannot_mean_an_empty_image_path():
    """Guards ebd28778: a lost transport response must refuse verification."""
    env = SimpleNamespace(
        exec=AsyncMock(return_value=ExecResult(stdout="", return_code=0))
    )
    with pytest.raises(RuntimeError, match="reading image PATH"):
        await lockdown._trusted_verifier_path(env, None, "/app")


@pytest.mark.asyncio
async def test_refusal_hits_are_checked_before_transport_text_is_parsed():
    """Guards ebd28778: merged stderr must not corrupt valid refusal evidence."""
    report = TrustedPathReport(
        path="/usr/bin:/bin", refusals="/refused", stubs=(("cargo", "/opt/tools"),)
    )

    @receipt_aware
    async def answer(command, **kwargs):
        return ExecResult(stdout="1 yes\0", return_code=0)

    async def transport(command, **kwargs):
        result = await answer(command, **kwargs)
        result.stdout = "transport notice\n" + result.stdout
        return result

    assert await _refused_programs_run(SimpleNamespace(exec=transport), report) == [
        ("cargo", "/opt/tools")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("separate", [False, True])
async def test_path_judgment_keeps_the_pre_freeze_or_transfer_exclusions(
    monkeypatch, separate
):
    """Guards merging 85b87bc9: freezing cannot make an agent mount trusted."""
    seen = []

    async def judge(env, raw, blocked, agent):
        seen.extend(blocked)
        return TrustedPathReport(path=lockdown._SAFE_VERIFIER_PATH)

    @receipt_aware
    async def answer(command, **kwargs):
        stdout = "/opt/shared/bin\n" if command == PATH_QUERY_CMD else ""
        return ExecResult(stdout=stdout, return_code=0)

    monkeypatch.setattr(lockdown, "_judge_trusted_entries", judge)
    for name, value in [
        ("_isolate_verifier_tool_state", {}),
        ("_discover_pytest_plugin_flags", ""),
        ("_discover_pytest_plugins", ([], [])),
        ("_install_pytest_plugin_guard", ("/guard", "", (), ())),
        ("_verifier_uid", "0"),
        ("_trusted_verifier_pythonpath", ""),
        ("_trusted_verifier_bash_env", ""),
    ]:
        monkeypatch.setattr(
            lockdown, name, AsyncMock(return_value=value), raising=False
        )
    verifier = SimpleNamespace(
        service="main", user="root", env={}, required_tools=[], optional_tools=[]
    )
    task = SimpleNamespace(config=SimpleNamespace(verifier=verifier))
    kwargs = (
        {"agent_paths": ("/opt/shared",)}
        if separate
        else {"blocked": ("/app", "/opt/shared")}
    )
    # The old build helper has no blocked keyword; exercise its actual shared
    # defaults with a host-held alias record through the assembler where present.
    if (
        not separate
        and "blocked"
        not in __import__("inspect").signature(lockdown._build_verifier_env).parameters
    ):
        monkeypatch.setattr(
            lockdown,
            "_blocked_verifier_path_prefixes",
            lambda *args: ("/app", "/opt/shared"),
        )
        kwargs = {}
    await lockdown._build_verifier_env(
        SimpleNamespace(exec=answer), task, "agent", None, **kwargs
    )
    assert "/opt/shared" in seen
    if separate:
        assert "/root" not in seen
