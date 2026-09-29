# Sandboxes on your own Docker host (`--sandbox remote-docker`)

`remote-docker` runs each task on a Docker daemon you control, reached over SSH or TCP with TLS. It is the local `docker` provider with the daemon somewhere else: the same compose files, sandbox user, verifier hardening, network modes, compose side services, separate verifier sandboxes and snapshots.

```bash
export BENCHFLOW_REMOTE_DOCKER_HOST=ssh://builder@gpu-box.example.com
bench eval run --tasks-dir tasks/ --agent oracle --sandbox remote-docker
```

```python
import benchflow as bf

evaluation = bf.Evaluation(
    tasks_dir="tasks/",
    jobs_dir="jobs/",
    config=bf.EvaluationConfig(agent="oracle", environment="remote-docker"),
)
evaluation.run_sync()
```

## Choosing the host

The host comes from `BENCHFLOW_REMOTE_DOCKER_HOST`, or from `DOCKER_HOST` when that is unset. BenchFlow sets it on every `docker` and `docker compose` call it makes and removes `DOCKER_CONTEXT`, so your current docker context never changes where tasks run.

| Host | Requirements |
|---|---|
| `ssh://user@host[:port]` | The local `docker` CLI connects with `ssh user@host docker system dial-stdio`. Authentication is your SSH setup (keys, agent, `~/.ssh/config`), exactly as for `docker -H ssh://...`. The remote user needs access to the Docker socket, and the host must be in `known_hosts`. |
| `tcp://host:2376` | TLS only: `DOCKER_TLS_VERIFY=1` and `DOCKER_CERT_PATH` (or `~/.docker`) holding `ca.pem`, `cert.pem` and `key.pem`. Plain TCP is refused, because an unauthenticated Docker API gives root on the host to anyone who can reach the port. |

`unix://` and `npipe://` sockets are refused with a pointer to `--sandbox docker`. The SSH user in the URL is replaced with `***` in every message and log line, since some gateways put an access token there.

## What differs from local Docker

- **Nothing on your machine is mounted.** The local provider bind-mounts the rollout's `agent/`, `verifier/` and `artifacts/` folders; a remote daemon cannot see them and would create empty folders on its own disk. The remote provider creates `/logs/agent`, `/logs/verifier` and `/logs/artifacts` inside the container and copies them back with `docker compose cp`, as on Daytona. Extra host mounts are refused.
- **The model proxy runs in the sandbox**, as on Daytona and Modal, because the remote container cannot reach a proxy on your machine.
- **The host is checked first.** `docker info` runs before each job (`bench eval run`, `bf.Evaluation`, reviewer sandboxes) and before each sandbox starts. An unreachable host fails with `Remote Docker host unreachable: <host>: <ssh or docker reason>`; a task asking for more CPUs or memory than `docker info` reports fails with `Remote Docker host lacks resources: task needs 8 CPUs; <host> has 4`. Neither is retried and no container is created. On a daemon inside a VM or container, `docker info` can report the physical machine rather than a quota, so the check is a lower bound.
- **A full remote disk is named** (`remote Docker host is out of disk space`) and not retried.
- **Teardown removes everything of the rollout.** `docker compose down --volumes --remove-orphans` always runs (with `--rmi all` when the sandbox is deleted), then BenchFlow lists the containers, networks and volumes that still carry the rollout's `com.docker.compose.project` label and removes them. If the host is unreachable at teardown, the warning names the project and the command that removes it later: `docker compose -p <project> down --volumes --remove-orphans`.
- **No verifier recovery** after a lost sandbox (that needs an owned `docker` or `daytona` sandbox), and no start-of-job prune of stopped containers: on a shared host that would remove other users' containers.

Images are built on the remote host (the build context is sent over the connection) and cached there like local builds. `--build-concurrency` applies as for `docker`.
