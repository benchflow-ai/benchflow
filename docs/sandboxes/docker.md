# Docker

```bash
bench doctor                                         # is the daemon up, and is there disk?
bench run tasks/my-task -a claude-agent-acp          # -s docker is the default
bench run tasks/ -a claude-agent-acp -j 64           # 64 trials at a time on one daemon
```

Docker is the default sandbox. Each trial runs in its own Compose project on your Docker daemon, with its own networks. It needs Docker Engine with the Compose plugin. Point `DOCKER_HOST` or a Docker context at another daemon to use it; for a host you reach over SSH, see [Remote Docker](./remote-docker.md).

Docker has no disk cap beyond your machine's, so it is the sandbox for heavy task images.

## Many trials on one daemon

Each trial gets its own small subnet (a /28) from the daemon's address pools. A stock daemon then holds thousands of trial networks instead of about 30, so `--concurrency` above 30 works without editing `daemon.json`.

- BenchFlow skips subnets that overlap an existing network, a route of the Docker host or one of its nameservers.
- A network the task's compose file configures itself (its own `ipam`, `external`, a fixed `name`) is left to the daemon. BenchFlow reserves those subnets before the first trial.
- On Docker Desktop, Colima or a remote daemon, BenchFlow cannot read the host's routes and leaves the choice to the daemon.
- When the range runs out, the trial fails and says to widen the range or lower `--concurrency`.

Keep your Docker context and client settings unchanged while a job runs.

If the daemon refuses a subnet a task pins, the trial ends at once, naming the subnet. Give the task non-overlapping `ipam`, or remove the conflicting network if you own it.

| `BENCHFLOW_DOCKER_SUBNETS` | Effect |
|---|---|
| Unset | BenchFlow chooses on a local Linux daemon; the daemon chooses elsewhere |
| `172.20.0.0/14,192.168.0.0/16` | Use these IPv4 ranges (they must be free on the Docker host) |
| `daemon` | Always leave the choice to the daemon |

## Warm pool

`--concurrency` is the number of trials running at a time. A Docker job with more tasks than that also prepares a few sandboxes ahead, beyond the running ones: `--warm-pool N` of them (0 to `--concurrency`). The default is 1 when `--concurrency` is 2 or more, and 0 when it is 1. So at most `--concurrency` + N sandboxes are alive, and a `--concurrency 1` job keeps one sandbox. The job's start line says "warm pool: up to N prepared sandboxes".

A separate verifier is a sandbox of its own and is not counted there. On Docker it starts as the agent's turn begins, so with `--verifier-sandbox separate` each running trial has two sandboxes during that turn, and a `--concurrency 1` job has two. See [Separate verifier sandboxes](../tasks/separate-verifier.md).

Each time a trial's sandbox is ready, BenchFlow starts the setup of the next trial still waiting for a slot: the same sandbox start, sandbox user, lockdown and agent install that trial would run itself, in the same order, so it gets the same hardening. Its seconds count against `--max-sandbox-seconds` as the trial's own.

- A prepared sandbox belongs to the trial it was made for and is used once. A retry starts a new one.
- If the task's files or image changed before its turn, the prepared sandbox is removed and the trial starts from scratch.
- Sandboxes made for trials that do not run (a budget or usage stop, Ctrl-C) are removed when the job ends, with the trial folders they made.
- Tasks whose compose file reserves or shares something two sandboxes cannot are not prepared ahead: a container name, published ports, a host path mounted into a service, devices or reserved GPUs, a privileged service, the host's network, pid, ipc, cgroup or user namespace, a fixed or external network or volume, a pinned subnet.

In `timing.json`, `sandbox_ready` is how long the trial waited for its sandbox, `warm_pool_wait` the part of it spent waiting for a prepared one, and `warm_pool_idle` how long a prepared one waited for its trial. `--warm-pool 0`, or `BENCHFLOW_WARM_POOL=off`, turns this off.

## Pre-pulled images

At its start, a Docker job pulls the prebuilt task images (`docker_image`) the daemon does not have, four at a time, in task order, so later trials do not pull inside their slot. A task whose compose file pins a service's `platform` is left to its own start, which pulls that variant. Each pull is recorded as BenchFlow's, as a trial's own pull is, for `bench cache clean --only pulled-images`. A pull that fails is only logged: the trial pulls again, and reports a failure, as before. `BENCHFLOW_PREPULL=off` turns this off.

## The model proxy

API-key runs go through BenchFlow's LiteLLM model proxy, which holds the provider key. On Docker it runs on your machine; there, a trial's proxy starts when its sandbox starts, so the proxy's own start (about 4 s) overlaps the sandbox's instead of following the agent's install. Each trial still gets a proxy of its own (its own key, port and log), stopped with the trial. `BENCHFLOW_HOST_PROXY_AHEAD=0` gives back the old order: the proxy starts when the agent connects, the sandbox's security checks run after it, and BenchFlow imports LiteLLM for the agent's model metadata after the proxy is up. `BENCHFLOW_HOST_PROXY_WARM=N` keeps N more started for the job's next trials (none by default). For Docker tasks with a network policy, and on every other sandbox, it runs inside the sandbox. It is then installed from the exact versions BenchFlow locks, each checked by sha256.

To add packages to the proxy, list them in `BENCHFLOW_LITELLM_EXTRA_REQUIREMENTS`, separated by spaces:

```bash
BENCHFLOW_LITELLM_EXTRA_REQUIREMENTS="google-genai>=1.0" bench run tasks/my-task -a gemini
```

Extras cannot change a locked package. One that would is refused before any sandbox starts, naming both. Checking needs `uv` on your machine. With extras, the proxy on your machine uses its own venv under `~/.cache/benchflow/litellm-proxy/`, built once. Each trial's `config.json` records the proxy's requirement file, its sha256 and the extras under `litellm_proxy`.

## Cached install layers

On Docker and remote Docker, BenchFlow installs the model proxy and the agent once per task image, as image layers (`benchflow-layer:proxy-<key>` and `benchflow-layer:agent-<key>`). Later trials start from them. The proxy layer's key is the requirement file's sha256, the Python and uv versions, the install command, and the task image's content digest, libc and architecture, so two task images never share a layer and a changed lock builds a new one. A run with proxy extras installs its proxy in each trial: extras resolve when they install, so a layer could hold an older release of one than a fresh install would. A layer is reused only when BenchFlow on this machine built it (it records each build's image id under `~/.cache/benchflow/layers/built/`); an image some other build tagged with the same name is not used, and the layer is built again. The layers hold no credentials: the proxy's config, keys and logs are written in each trial.

Without a layer, the proxy needs about 600 MB free in the sandbox. With less, the trial stops before the agent starts and says how much is missing.

| Setting | Effect |
|---|---|
| `--fresh-install` or `BENCHFLOW_FRESH_INSTALL=1` | Install the proxy and agent in every trial; no layers |
| `BENCHFLOW_PROXY_LAYER_KEEP` | How many proxy layers to keep (default 32) |
| `BENCHFLOW_PROXY_LAYER_MAX_GB` | Most space proxy layers may use (default 20 GB) |

Each trial's `config.json` says what happened under `layers`, for `proxy` and `agent`:

| Value | Meaning |
|---|---|
| `hit` | Started from the layer, with its key |
| `miss` | Built the layer, with its key and build time |
| `fresh` | Installed in the trial, with the reason |
| `unsupported` | The sandbox builds no layers (Modal, Apple Container, AgentCore) |

Hooks, setup commands, healthchecks, custom uploads and runtime environment overrides turn the agent layer off. BenchFlow removes only images that carry its own `benchflow.layer` label. See [Cache](../jobs/cache.md) to list and clean them.

## Network policies

For `denylist`, `allowlist`, `no-network` with an agent, and the cloud-metadata block on open networks, BenchFlow adds `NET_ADMIN` to `main` through its own compose overlay. Your `Dockerfile` and `docker-compose.yaml` need nothing extra.

The host's kernel must provide the iptables `nat` table and `owner` match for `allowlist`. When either is missing, the trial fails before the agent starts. See [Sandbox security](./security.md#network-policy-by-stage).

## Next steps

- Run on a Docker host you reach over SSH: [Remote Docker](./remote-docker.md).
- Run in the cloud: [Daytona](./daytona.md).
- What the sandbox guarantees: [Sandbox security](./security.md).
