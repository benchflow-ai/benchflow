"""Example: Run benchflow eval with smolclaws environments.

Shows how to use benchflow SDK with smolclaws' mock services
(claw-gmail, claw-gcal, etc.) for agent evaluation.
"""

import asyncio
from benchflow import SDK, detect_services_from_dockerfile, build_service_hooks


async def main():
    sdk = SDK()

    # Task path (from smolclaws repo)
    task = "path/to/smolclaws/tasks/email-extract-contact-info"

    # Auto-detect which claw-* services this task needs from its Dockerfile
    services = detect_services_from_dockerfile(task)
    print(f"Detected services: {[s.name for s in services]}")
    # → ['gmail']

    # Build hooks that start services before agent runs
    hooks = build_service_hooks(services)

    # Run with services, skills, and sandbox user
    result = await sdk.run(
        task_path=task,
        agent="claude-agent-acp",
        model="claude-haiku-4-5-20251001",
        environment="daytona",
        skills_dir="path/to/smolclaws/skills",  # gws, slack skills
        sandbox_user="agent",                    # non-root execution
        context_root="path/to/smolclaws",        # resolve Dockerfile COPY paths
        pre_agent_hooks=hooks,                   # start claw-* services
    )

    print(f"Reward: {result.rewards}")
    print(f"Tool calls: {result.n_tool_calls}")
    print(f"Error: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())
"""Example: Batch eval with smolclaws tasks.

Runs all tasks in a directory with skills and services.
"""


async def batch_eval():
    from benchflow import Job, JobConfig

    result = await Job(
        tasks_dir="path/to/smolclaws/tasks",
        jobs_dir="jobs/smolclaws-eval",
        config=JobConfig(
            agent="claude-agent-acp",
            model="claude-haiku-4-5-20251001",
            environment="daytona",
            concurrency=64,
            skills_dir="path/to/smolclaws/skills",
            sandbox_user="agent",
            context_root="path/to/smolclaws",
        ),
    ).run()

    print(f"Score: {result.passed}/{result.total} ({result.score:.1%})")
"""Example: Create a new smolclaws-compatible environment.

All claw-* environments follow the same protocol:
- CLI: `claw-<name> --db <path> serve --host 0.0.0.0 --port <port> --no-mcp`
- CLI: `claw-<name> --db <path> seed --scenario <name>`
- Admin API: /_admin/reset, /_admin/state, /_admin/diff, /_admin/action_log
- Health: /health
- SQLite backend

To add a new environment to benchflow:
"""


def register_custom_env():
    from benchflow import register_service

    register_service(
        name="discord",
        cli_name="claw-discord",
        port=9006,
        db_path="/data/discord.db",
        description="Mock Discord API",
    )
