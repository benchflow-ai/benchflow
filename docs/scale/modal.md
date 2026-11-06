# Modal

Modal runs each trial in a serverless cloud sandbox, and can give a task GPUs. Use it for tasks that need a GPU, or for many trials without a machine of your own. It needs a Modal account and the `sandbox-modal` extra.

## Set it up

Install BenchFlow with the extra, and give it your Modal token, either from `modal token new` (which writes `~/.modal.toml`) or as variables:

```bash
uv tool install --python 3.12 'benchflow[sandbox-modal]==0.9.1' \
  --constraints https://raw.githubusercontent.com/benchflow-ai/benchflow/v0.9.1/constraints.txt
export MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...
```

The extra installs Modal's own `modal` command inside BenchFlow's tool environment, not on your `PATH`; run `~/.local/share/uv/tools/benchflow/bin/modal token new` if you have no `modal` of your own. `bench doctor` doesn't check Modal: a missing token shows at the first trial.

Without a token, a run stops at the first trial with:

```text
Modal requires authentication. Run 'modal token new' to set up credentials, or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables.
```

## Run on Modal

```bash
bench run skillsbench/tasks/econ-detrending-correlation -a oracle --sandbox modal
```

`skillsbench/` is a clone of SkillsBench at commit f860ecf0 ([how](../evaluate/run-a-benchmark.md#run-three-tasks-three-times-each)).

```text
[econ-detrending-correlation] Rewards: {'reward': 1.0}
[PASS] econ-detrending-correlation (reward=1.00, tools=0)
Job complete: 1/1 (100.0%), mean_reward=1.00, errors=0, idle_timeouts=0, time=1.3min

✓ Score: 1/1 (100.0%), mean reward 1.00, errors=0
Outcomes: 1 scored (1 passed, 0 failed), 0 unscored, 0 errored
Cost:      none: no model tokens were used
Time:      1m 21s
```

As on Daytona, the model proxy runs inside each sandbox. The verifier runs in the agent's sandbox after the agent's turn (`verifier_isolation: shared` in `result.json`); with `--verifier-sandbox separate`, or a task that asks for it, it runs in a fresh sandbox of its own ([Verifiers](../tasks/verifiers.md#run-the-verifier-in-its-own-sandbox)).

BenchFlow starts every sandbox in one Modal app named `__benchflow__`, which it creates the first time; look for the trial's sandboxes there in Modal's dashboard. `result.json` records no `sandbox_id` for Modal in 0.9.1.

## GPUs

A task asks for GPUs in its `task.md`:

```yaml
sandbox:
  gpus: 1
  gpu_types: [H100]
```

Modal accepts one GPU type per task; leave `gpu_types` out to take any.

## Limits

Modal runs one container per sandbox, so a task with a `docker-compose.yaml` is refused before it starts. Of the network policies, `no-network` works; `allowlist` and `denylist` don't, and tasks that use them are refused.

BenchFlow also refuses to start a sandbox where a system library is writable by anyone but root, since the verifier later runs as root there. On Modal, our half-life task's image hit this:

```text
LoaderCustodyError: sandbox startup refused: BenchFlow could not seal its protected runtime: the system loader loads /usr/local/lib/libpython3.12.so (owner 0:0, mode -rwxrwxr-x), which is not root's alone, so it will not run root steps (the verifier's) after the agent's turn.
next: if the task's workdir holds that path, give the task a workdir of its own; otherwise make that path root's in the task's image (in its Dockerfile, RUN chown -R root:root <path> && chmod -R go-w <path>), or run the agent as root with --sandbox-user none
```

The same task runs on Docker and Daytona. Follow the message's fix in the task's Dockerfile.
