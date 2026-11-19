# Modal

Modal runs each trial in a serverless cloud sandbox, and can give a task GPUs. Use it for tasks that need a GPU, or for many trials without a machine of your own. It needs a Modal account and the `sandbox-modal` extra.

## Set it up

Install BenchFlow with the extra, and give it your Modal token, either from `modal token new` (which writes `~/.modal.toml`) or as variables:

```bash
uv tool install --python 3.12 'benchflow[sandbox-modal]==0.9.3' \
  --constraints https://raw.githubusercontent.com/benchflow-ai/benchflow/v0.9.3/constraints.txt
export MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...
```

The extra installs Modal's own `modal` command inside BenchFlow's tool environment, not on your `PATH`; run `"$(uv tool dir)/benchflow/bin/modal" token new` if you have no `modal` of your own. `bench doctor` doesn't check Modal: a missing token shows at the first trial.

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

BenchFlow starts every sandbox in one Modal app named `__benchflow__`, which it creates the first time; look for the trial's sandboxes there in Modal's dashboard. `result.json` records no `sandbox_id` for Modal in 0.9.2.

## GPUs

A task asks for GPUs in its `task.md`:

```yaml
sandbox:
  gpus: 1
  gpu_types: [H100]
```

Modal accepts one GPU type per task; leave `gpu_types` out to take any.

## How a Modal sandbox compares with Docker

**CPUs and memory.** The task's `cpus` and `memory_mb` are the sandbox's hard limits, as Docker's `cpus` and memory limit are. Modal still lets a capped sandbox use more CPU time than its limit on some hosts: 24 busy processes in a `cpus = 12` sandbox got between 13.9 and 31.2 cores' worth (on Docker, at most 12.3). A process that goes over the memory limit makes Modal end the whole sandbox, where Docker kills the one process ([below](#when-modal-ends-a-sandbox)). `BENCHFLOW_MODAL_HARD_LIMITS=0` gives Modal the two numbers as requests alone: Modal's CPU limit is then 16 CPUs above the request, and memory has no limit.

**The image's environment.** Modal sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `BLIS_NUM_THREADS`, `ORT_INTER_OP_NUM_THREADS` and `ORT_INTRA_OP_NUM_THREADS` to the CPU count, `PYTHONHASHSEED=0` and `PYTHONPATH=/pkg/:/root/` in every sandbox, over the image's own values. BenchFlow reads the image's values from the task's Dockerfile (its final stage and the stages that stage is built from), with the task's `env` over them, and gives them back: the thread counts and `PYTHONHASHSEED` when the sandbox is created, `PYTHONPATH` with every command. `PYTHONHASHSEED` and `PYTHONPATH` the image leaves unset are given empty, which Python reads as unset.

Modal cannot unset a variable, so a thread count the image leaves unset keeps Modal's value; set it in the task's Dockerfile to choose it. Values a base image (`FROM`) sets are not read, and a task that names a prebuilt `docker_image` keeps Modal's values unless the task's `env` sets them. Each sandbox logs the names that keep Modal's.

## Sandbox lifetime

Modal ends a sandbox when its lifetime runs out, whatever runs in it. BenchFlow gives each sandbox 24 hours (86400 s, Modal's longest). Set `BENCHFLOW_MODAL_SANDBOX_TIMEOUT` to give less, in whole seconds from 60 to 86400; a sandbox left running after a crash then bills for that long at most.

Before any sandbox, BenchFlow refuses a task whose budgets do not fit in the lifetime: the agent's `timeout_sec`, the verifier's `timeout_sec` (unless the verifier has a sandbox of its own), and the setup budgets (`setup_timeout_sec` in `[environment]`, 600 s when unset, and each setup command's `timeout_sec`).

## When Modal ends a sandbox

Modal can end a sandbox on its own: when its lifetime runs out, when a process goes over its memory limit (Modal ends the whole sandbox, where Docker kills the one process), and when it evicts or preempts it. The trial then fails as an infrastructure error, with Modal's return code:

```text
Sandbox not found: Modal ended the sandbox before BenchFlow stopped it (returncode 124: its lifetime ran out (BENCHFLOW_MODAL_SANDBOX_TIMEOUT)); an exec failed with it
```

124 is the lifetime; 137 is a kill (the memory limit, an eviction, a preemption, or a terminate from outside BenchFlow).

## Limits

Modal runs one container per sandbox, so a task with a `docker-compose.yaml` is refused before it starts. Of the network policies, `no-network` works; `allowlist` and `denylist` don't, and tasks that use them are refused.

BenchFlow also refuses to start a sandbox where a system library is writable by anyone but root, since the verifier later runs as root there. On Modal, our half-life task's image hit this:

```text
LoaderCustodyError: sandbox startup refused: BenchFlow could not seal its protected runtime: the system loader loads /usr/local/lib/libpython3.12.so (owner 0:0, mode -rwxrwxr-x), which is not root's alone, so it will not run root steps (the verifier's) after the agent's turn.
next: if the task's workdir holds that path, give the task a workdir of its own; otherwise make that path root's in the task's image (in its Dockerfile, RUN chown -R root:root <path> && chmod -R go-w <path>), or run the agent as root with --sandbox-user none
```

The same task runs on Docker and Daytona. Follow the message's fix in the task's Dockerfile.
