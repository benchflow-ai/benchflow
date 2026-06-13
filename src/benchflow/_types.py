"""Canonical Scene / Role / Turn data types for benchflow trials.

These are the *declarative* types — they describe what a trial *will* do.
``benchflow.scenes.compile_scenes_to_steps`` lowers them to rollout
``Step`` objects before execution.

Merged from the duplicate definitions that lived in ``trial.py`` and
``_scene.py`` prior to ENG-47.

.. rubric:: Capability boundary (ENG-50)

BenchFlow provides **sandbox + instruction + observation** infrastructure.
It does **not** orchestrate agent-internal loops, tool protocols, or
agent-as-tool invocations. Those are per-agent capabilities declared via
:pyattr:`Role.capabilities` and implemented by the agent itself.

Concretely:

* BenchFlow sets up the sandbox (Docker / Daytona / Modal), injects
  environment variables and skills, and wires up the ACP transport.
* BenchFlow ensures sandbox networking allows inter-agent communication
  (all roles in a scene share a sandbox, so localhost is reachable).
* The ``capabilities`` field on :class:`Role` is a **declaration** — it
  tells downstream tooling / dashboards what the agent supports, but
  BenchFlow itself does not act on it.
* Agent-as-tool is a per-agent capability.  BenchFlow does not invoke
  agents on behalf of other agents; the calling agent uses its own
  native tool-use to reach a companion agent's endpoint.
* *Agent-internal* loop management is the agent's responsibility.
  BenchFlow scenes define *turns* (prompts), not iteration.  Harness-level
  loop strategies (:mod:`benchflow.loop_strategies`, ``--loop-strategy``)
  are a separate evaluand axis: the harness re-prompts the agent across
  verify-retry rounds and scores only the final hardened verify, but never
  orchestrates loops *inside* the agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Role:
    """One agent participant in a scene.

    The ``capabilities`` field is a declarative list of strings describing
    what the agent natively supports (e.g. ``["tool-use", "loop",
    "agent-as-tool"]``).  BenchFlow records these in trial metadata but
    does **not** act on them — the agent itself is responsible for
    implementing whatever capabilities it advertises.
    """

    name: str
    agent: str
    model: str | None = None
    reasoning_effort: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    timeout_sec: int | None = None  # None = inherit from task config
    idle_timeout_sec: int | None = None
    skills_dir: str | Path | None = None
    capabilities: list[str] | None = None  # e.g. ["tool-use", "agent-as-tool", "loop"]


@dataclass
class Turn:
    """One prompt in a scene. *role* selects which Role acts."""

    role: str
    prompt: str | None = None  # None = expand from the task prompt


@dataclass
class Scene:
    """Authoring sugar for role/skill attribution over a sequence of turns."""

    name: str = "default"
    roles: list[Role] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    skills_dir: str | Path | None = None

    @classmethod
    def single(
        cls,
        *,
        agent: str,
        model: str | None = None,
        reasoning_effort: str | None = None,
        prompts: list[str | None] | None = None,
        role_name: str = "agent",
        skills_dir: str | Path | None = None,
    ) -> Scene:
        """Shortcut for single-agent, single-role scene."""
        prompts = prompts or [None]
        return cls(
            roles=[
                Role(
                    name=role_name,
                    agent=agent,
                    model=model,
                    reasoning_effort=reasoning_effort,
                )
            ],
            turns=[Turn(role=role_name, prompt=p) for p in prompts],
            skills_dir=skills_dir,
        )
