"""Standalone headless agent runner — `bench agent run` (run + resume).

Drives one ACP turn against an agent launched as a host subprocess
(``StdioTransport``), claude -p style: first run creates a session and persists
the agent's ACP ``sessionId``; a later invocation resumes it via ACP
``session/load`` (gated on the ``loadSession`` capability the agent advertised
at ``initialize``). No task, no sandbox, no verifier — the eval pipeline is
``bench eval run``; this is the interactive/debugging surface.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchflow.acp.client import ACPClient
from benchflow.agents.session_store import SessionStore


class ResumeUnsupportedError(RuntimeError):
    """The agent does not advertise the ACP ``loadSession`` capability."""


@dataclass
class RunResult:
    text: str
    stop_reason: str
    session_id: str
    acp_session_id: str


async def run_turn(
    *,
    agent: str,
    prompt: str,
    cwd: str,
    store: SessionStore,
    model: str = "",
    resume: str | None = None,
    launch_cmd: str | None = None,
    agent_env: dict[str, str] | None = None,
) -> RunResult:
    if not launch_cmd:
        raise ValueError(f"no launch command for agent {agent!r}")

    prior = store.load(resume) if resume else None

    client = ACPClient.from_config(command=launch_cmd, env=agent_env, cwd=cwd)
    await client.connect()
    try:
        init = await client.initialize()
        caps_model = getattr(init, "agent_capabilities", None)
        caps = (
            caps_model.model_dump(by_alias=True, exclude_none=True)
            if caps_model
            else {}
        )

        if prior:
            if not caps.get("loadSession"):
                raise ResumeUnsupportedError(
                    f"agent {agent!r} does not advertise the ACP loadSession "
                    "capability, so its sessions cannot be resumed across "
                    "processes; start a new session instead (omit --resume)"
                )
            session = await client.session_load(prior.acp_session_id, cwd=cwd)
            rec = store.update(prior.session_id, capabilities=caps)
        else:
            session = await client.session_new(cwd=cwd)
            rec = store.create(agent=agent, model=model, cwd=cwd)
            store.update(
                rec.session_id,
                acp_session_id=session.session_id,
                capabilities=caps,
            )

        prompt_result = await client.prompt(prompt)
        return RunResult(
            text=session.full_message,
            stop_reason=str(prompt_result.stop_reason),
            session_id=rec.session_id,
            acp_session_id=session.session_id,
        )
    finally:
        await client.close()
