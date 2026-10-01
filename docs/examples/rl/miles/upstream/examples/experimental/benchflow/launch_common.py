"""The BenchFlow-side launcher wiring, separate from run.py so it imports without
the trainer's heavy dependencies (the tests load it on CPU-only hosts)."""

import json
from pathlib import Path

import httpx

BENCHFLOW_EXAMPLE_DIR = Path(__file__).resolve().parent


def agentic_pythonpath_dirs() -> list[str]:
    """The directory every BenchFlow launcher puts on the workers' PYTHONPATH."""
    return [str(BENCHFLOW_EXAMPLE_DIR)]


def agentic_train_args(
    *,
    tito_model: str,
    session_server_workers: int,
    session_server_port: int = 30000,
    drop_constant_reward_groups: bool = True,
) -> str:
    """The agentic wiring the BenchFlow launchers pass to train.py.

    Groups whose episodes all got the same reward carry no GRPO signal. With
    ``drop_constant_reward_groups`` Miles drops them and samples replacements
    (counted in ``rollout/dynamic_filter/drop_zero_std_*``). Replacements come a
    batch at a time (Miles requires ``--over-sampling-batch-size`` of at least
    ``--rollout-batch-size``, its default); the groups still running when the
    batch is full are aborted through the agent function's ``abort`` hook and
    discarded. Groups holding a discarded episode are always dropped
    (``rollout/aborted/drop_<exit_status>``).
    """
    filter_args = (
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter "
        if drop_constant_reward_groups
        else ""
    )
    return (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path benchflow_agent_function.run "
        "--custom-rm-path benchflow_agent_function.reward_func "
        "--rollout-function-path benchflow_rollout.RolloutFn "
        f"{filter_args}"
        f"--tito-model {tito_model} "
        "--use-session-server "
        f"--session-server-port {session_server_port} "
        f"--session-server-workers {session_server_workers} "
    )


def benchflow_env_vars(*, env_url: str, token_file: str = "", episode_timeout_s: int | None = None) -> dict[str, str]:
    """The rollout workers' environment: where the BenchFlow server is, and the
    PATH of its token file (never the token: worker env rides ray's runtime_env,
    which is logged in plaintext)."""
    env = {"BENCHFLOW_ENV_URL": env_url.rstrip("/")}
    if token_file:
        env["BENCHFLOW_ENV_TOKEN_FILE"] = token_file
    if episode_timeout_s is not None:
        env["BENCHFLOW_EPISODE_TIMEOUT"] = str(episode_timeout_s)
    return env


def preflight(*, env_url: str, prompt_data: str, token_file: str = "") -> dict:
    """Fail at launch, not per sample: the server must answer, and every task the
    prompt data names must be one it serves. A missing task would otherwise fail
    each of its samples before the first model call, and Miles would keep
    replacing the dropped groups without a word."""
    headers = {}
    if token_file:
        headers["Authorization"] = f"Bearer {Path(token_file).expanduser().read_text().strip()}"
    base = env_url.rstrip("/")
    try:
        health_response = httpx.get(f"{base}/health", timeout=10)
        health_response.raise_for_status()
        health = health_response.json()
        tasks_response = httpx.get(f"{base}/tasks", headers=headers, timeout=30)
        tasks_response.raise_for_status()
        served = set(tasks_response.json()["tasks"])
    except (httpx.HTTPError, KeyError, ValueError) as e:
        raise RuntimeError(
            f"no BenchFlow environment server at {base} ({e!r}); start it first:\n"
            "  python -m benchflow.integrations.miles serve --tasks-dir <tasks> --max-sandboxes <n>"
        ) from e
    wanted = set()
    with open(prompt_data) as f:
        for line in f:
            if line.strip():
                wanted.add(json.loads(line).get("metadata", {}).get("instance_id"))
    missing = sorted(str(task) for task in wanted - served)
    if missing:
        raise RuntimeError(
            f"{len(missing)} task(s) in {prompt_data} are not served by {base}: {missing[:5]}"
            f"{' ...' if len(missing) > 5 else ''}"
        )
    print(
        f"BenchFlow environment at {base}: {health.get('tasks')} tasks on {health.get('sandbox')}, "
        f"at most {health.get('max_sandboxes')} sandboxes; {len(wanted)} tasks in {prompt_data}",
        flush=True,
    )
    return health
