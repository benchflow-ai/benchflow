"""The bridge process and the episode session, against the scripted bridge.

Checks what the tools return (the TRL harness's shapes), what reaches the
BenchFlow side (requests, the allowlisted environment), and that the sandbox is
closed in every way an episode can end: a close request, end of input, SIGTERM
after the bridge stops answering, and the bridge dying.
"""

from __future__ import annotations

import asyncio
import json

import benchflow_taskset.session as session_module
import pytest
from benchflow_taskset.session import BridgeError, SandboxStartError, bridge_environment
from conftest import make_session


@pytest.fixture(autouse=True)
def quick_close(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session_module, "REPLY_SLACK_SEC", 1.0)
    monkeypatch.setattr(session_module, "CLOSE_REPLY_SEC", 5.0)
    monkeypatch.setattr(session_module, "EXIT_WAIT_SEC", 2.0)
    monkeypatch.setattr(session_module, "TERM_WAIT_SEC", 10.0)


async def test_a_full_episode_and_its_requests(world) -> None:
    session = make_session(world)
    await session.start()
    assert session.workspace == "/workdir"
    assert await session.run_bash("ls") == "out\n"
    assert await session.submit("42") == "submission recorded"
    decision = await session.verify()
    assert decision["reward"] == 1.0
    assert await session.verify() is decision, "the verifier runs once"
    await session.close()
    assert world.ops() == ["start", "bash", "write", "verify", "close"]
    requests = [row["request"] for row in world.events("request")]
    assert requests[2] == {"op": "write", "path": "/workdir/answer.txt", "text": "42"}
    assert requests[3]["policy_acted"] is True
    assert world.closed() == ["close"]


async def test_an_untouched_sandbox_is_verified_as_clean(world) -> None:
    # The BenchFlow side drops a verifier crash only when the policy never acted.
    session = make_session(world)
    await session.start()
    await session.verify()
    await session.close()
    verify = next(
        row["request"] for row in world.events("request") if row["op"] == "verify"
    )
    assert verify["policy_acted"] is False


async def test_tool_outputs_match_the_trl_harness(world) -> None:
    world.set(
        bash=[
            {
                "ok": True,
                "return_code": 0,
                "stdout": "x" * 500,
                "stderr": "err",
                "timed_out": False,
            },
            {
                "ok": True,
                "return_code": 2,
                "stdout": "partial",
                "stderr": "",
                "timed_out": False,
            },
            {
                "ok": True,
                "return_code": 124,
                "stdout": "partial",
                "stderr": "",
                "timed_out": True,
            },
            {"ok": False, "error": "sandbox gone", "transient": True},
        ]
    )
    session = make_session(world, max_output_chars=300)
    await session.start()
    long = await session.run_bash("big")
    assert len(long) == 300 and long.endswith("\n[benchflow output truncated]\n")
    assert await session.run_bash("fails") == "partial", (
        "no exit-code suffix, as in TRL"
    )
    assert json.loads(await session.run_bash("slow")) == {
        "error": "Command timed out after 5 seconds"
    }
    assert json.loads(await session.run_bash("lost")) == {"error": "sandbox gone"}
    assert session.stats == {
        "bash_calls": 4,
        "bash_timeouts": 1,
        "bash_nonzero": 1,
        "exec_errors": 1,
        "transient_errors": 1,
    }
    await session.close()


async def test_after_submit_the_tools_refuse(world) -> None:
    session = make_session(world)
    await session.start()
    await session.submit("done")
    assert "already submitted" in json.loads(await session.run_bash("ls"))["error"]
    assert "already submitted" in json.loads(await session.submit("again"))["error"]
    await session.close()
    assert world.ops().count("write") == 1


async def test_a_start_failure_carries_the_drop_decision(world) -> None:
    drop = {
        "reward": None,
        "dropped": True,
        "reason": "sandbox_start",
        "detail": "no image",
    }
    world.set(start={"ok": False, "error": "no image", "decision": drop})
    session = make_session(world)
    with pytest.raises(SandboxStartError) as raised:
        await session.start()
    assert raised.value.decision == drop
    await session.close()
    assert world.closed() == ["close"]


async def test_a_bridge_that_dies_mid_episode_ends_it_as_infrastructure(world) -> None:
    world.set(die_on="bash")
    session = make_session(world)
    await session.start()
    reply = json.loads(await session.run_bash("ls"))
    assert "bridge failed" in reply["error"]
    assert session.infra_error is not None
    assert "bridge failed" in json.loads(await session.submit("x"))["error"]
    with pytest.raises(BridgeError):
        await session.verify()
    await session.close()  # the process is already gone; closing must not hang or raise
    assert world.events("dying")


async def test_a_bridge_that_stops_answering_gets_sigterm_and_closes(world) -> None:
    world.set(hang_on="verify")
    session = make_session(world, verify_timeout_sec=1)
    await session.start()
    await session.run_bash("ls")
    with pytest.raises(BridgeError, match="did not answer"):
        await session.verify()
    # A late reply could be read as the next answer, so the pipe is never reused.
    assert session.bridge.broken is not None
    with pytest.raises(BridgeError, match="unusable"):
        await session.bridge.call({"op": "bash", "command": "ls"}, timeout_sec=1)
    await asyncio.wait_for(session.close(), 30)
    assert world.closed() == ["sigterm"]
    assert "close" not in world.ops(), "an unresponsive bridge is not asked politely"


async def test_end_of_input_closes_the_sandbox(world) -> None:
    # What happens when the env worker dies: its end of the pipe closes.
    session = make_session(world)
    await session.start()
    process = session.bridge.process
    process.stdin.close()
    await asyncio.wait_for(process.wait(), 10)
    assert world.closed() == ["eof"]
    await session.close()


async def test_the_bridge_gets_the_sandbox_key_and_no_model_keys(
    world, monkeypatch
) -> None:
    monkeypatch.setenv("DAYTONA_API_KEY", "dtn-test")
    monkeypatch.setenv("BENCHFLOW_DAYTONA_OWNER", "tests")
    for name in (
        "PRIME_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "HF_TOKEN",
        "WANDB_API_KEY",
    ):
        monkeypatch.setenv(name, "secret")
    monkeypatch.setenv("VIRTUAL_ENV", "/verifiers/venv")
    monkeypatch.setenv("PYTHONPATH", "/verifiers/site")
    session = make_session(world)
    await session.start()
    await session.close()
    env = set(world.events("spawned")[0]["env"])
    assert {"DAYTONA_API_KEY", "BENCHFLOW_DAYTONA_OWNER", "PATH", "HOME"} <= env
    assert not env & {
        "PRIME_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "HF_TOKEN",
        "WANDB_API_KEY",
    }
    assert not env & {"VIRTUAL_ENV", "PYTHONPATH"}


def test_bridge_environment_is_an_allowlist() -> None:
    source = {
        "PATH": "/bin",
        "DAYTONA_API_URL": "u",
        "SECRET_TOKEN": "s",
        "LC_ALL": "C",
        "EXTRA": "1",
    }
    assert bridge_environment(source=source) == {
        "PATH": "/bin",
        "DAYTONA_API_URL": "u",
        "LC_ALL": "C",
    }
    assert bridge_environment(("EXTRA",), source=source)["EXTRA"] == "1"
