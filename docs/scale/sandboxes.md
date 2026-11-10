# Sandboxes

Every trial runs in a fresh sandbox: a container or a cloud machine built from the task's environment. Docker on your machine is the default and needs no setup. Use a cloud sandbox when your machine can't run as many trials at once as you need, or a task needs hardware it doesn't have. Choose one with `--sandbox`.

## Choose a sandbox

| `--sandbox` | Runs on | Install | Tasks with services (compose) | No-network tasks | Allowlist and denylist | Separate verifier |
|---|---|---|---|---|---|---|
| `docker` (default) | your machine | nothing | yes | yes | yes | yes |
| `remote-docker` | a Docker host you control, over SSH or TLS | nothing | yes | yes | yes | yes |
| `daytona` | Daytona's cloud | `benchflow[sandbox-daytona]` | yes | yes | yes | yes |
| `modal` | Modal's cloud, with GPUs | `benchflow[sandbox-modal]` | no | yes | no | yes |
| `agentcore` | Amazon Bedrock AgentCore | `benchflow[sandbox-agentcore]` | no | no | no | no |
| `apple-container` | Apple's containers, on a Mac | nothing | no | no | no | no |

A task that asks for something its sandbox can't give is refused before the run starts, with code `R401`; it never runs half-set-up. One combination is refused on Daytona in 0.10.0rc2 though each part works: a task with services and an `allowlist`, since only `docker` and `remote-docker` read the services to check them. A no-network task on AgentCore or Apple Container is refused, for example, because neither can cut the network off.

Setup for the cloud sandboxes: [Daytona](./daytona.md), [Modal](./modal.md).

## Docker

Docker needs only a running daemon. `bench doctor` reports it with the space it has:

```text
Sandbox
  PASS  docker                Docker 29.1.3 via default (16 CPU, 62.8 GiB, 27.8 GiB free in /var/lib/docker)
```

The first run of a task builds its image; later runs reuse it. Each agent's install is cached too, as a layer on top of the task image. Both take disk space: on our build machine, six Claude Code trials starting at once added about 14 GB. On a long run, watch the free space; a full disk ends trials with `disk_full`.

## Remote Docker

`--sandbox remote-docker` runs each trial on a Docker daemon you control, over SSH or TCP with TLS:

```bash
export BENCHFLOW_REMOTE_DOCKER_HOST=ssh://builder@gpu-box.example.com
bench doctor --sandbox remote-docker
bench run tasks/ --sandbox remote-docker
```

It behaves like the local `docker` sandbox: the same compose services, network modes and separate verifier. BenchFlow runs the model proxy inside each sandbox, since a remote container can't reach one on your machine.

## Amazon Bedrock AgentCore

`--sandbox agentcore` runs each trial in an AgentCore microVM on AWS, with the `sandbox-agentcore` extra and your AWS credentials. AgentCore has no isolated network mode, so tasks that need no network, compose services or a separate verifier can't run there.

## What BenchFlow keeps between runs

`bench cache list` shows what BenchFlow keeps to start trials faster: agent install layers, model proxy layers, build records and snapshots.

```bash
bench cache list
```

```text
docker  images        benchflow-layer:agent-147f05f104bb...  claude-agent-acp 0.81.2+claude-code@2.1.280  568.5 MB  created 2026-11-04 18:03
docker  images        benchflow-layer:agent-ea685a3d6f0c...  opencode 1.18.11  399.8 MB  created 2026-11-04 18:12  (in use: ...)
docker  images        benchflow-layer:proxy-161b8ae02041...  litellm-proxy  316.5 MB  created 2026-11-04 09:09
...
```

`bench cache clean` prints what it would remove, and `bench cache clean --remove` removes it; anything a run still uses stays, with the reason. On Docker it also keeps every task image a run pulled, in 0.9.2 and 0.10.0rc2 alike, with the reason `Docker cannot atomically remove a checked reference; kept`; remove those yourself with `docker image rm` once no run needs them. `bench daemon start` keeps BenchFlow and its model proxy loaded between commands, for runs started with `BENCHFLOW_DAEMON=auto` in their environment; `bench daemon status` and `bench daemon stop` check and stop it.

## Clean up

A run deletes its sandboxes when it ends, including when you stop it with Ctrl-C. A run killed hard can leave some behind. `bench sandbox cleanup` removes leftover Docker containers and Daytona sandboxes older than a day (`--max-age`, in minutes); `--dry-run` lists them without deleting.
