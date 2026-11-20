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

## Limits

Modal runs one container per sandbox, so a task with a `docker-compose.yaml` is refused before it starts. Of the network policies, `no-network` works. `denylist` works behind an egress gateway you run (below); without one, tasks that use it are refused. `allowlist` doesn't work, and tasks that use it are refused.

BenchFlow also refuses to start a sandbox where a system library is writable by anyone but root, since the verifier later runs as root there. On Modal, our half-life task's image hit this:

```text
LoaderCustodyError: sandbox startup refused: BenchFlow could not seal its protected runtime: the system loader loads /usr/local/lib/libpython3.12.so (owner 0:0, mode -rwxrwxr-x), which is not root's alone, so it will not run root steps (the verifier's) after the agent's turn.
next: if the task's workdir holds that path, give the task a workdir of its own; otherwise make that path root's in the task's image (in its Dockerfile, RUN chown -R root:root <path> && chmod -R go-w <path>), or run the agent as root with --sandbox-user none
```

The same task runs on Docker and Daytona. Follow the message's fix in the task's Dockerfile.

## A task's denylist: the egress gateway

A Modal sandbox runs under gVisor, which has no firewall BenchFlow can add rules to, so the rules that hold the agent to the egress proxy on Docker can't go in. What Modal does enforce, outside the sandbox and for root too, is a list of addresses a sandbox may reach, set when it is created. The egress gateway uses that: each sandbox of a `denylist` task may reach the gateway's address and nothing else, and BenchFlow's egress proxy runs on the gateway instead of in the sandbox.

Run the gateway on a host with a fixed public IPv4 address and one TCP port open to the internet. It needs `python3` 3.10 or newer and `openssl`, not BenchFlow; copy the one file from the package:

```bash
python3 -I egress_gateway_server.py serve --state-dir /srv/egress-gateway \
  --listen 0.0.0.0:30100 --public 203.0.113.7:30100
```

Run it under a service manager, or detached; it keeps each trial's state in `--state-dir`, so a restart brings its trials back where they were. A sandbox may reach every port of the gateway's address, so open nothing on that host but the gateway's port and ssh; the proxies themselves listen on its loopback. For the same reason, BenchFlow refuses a `--public` address that is private, shared, loopback, link-local, multicast or reserved. Then point `bench` at its control socket, `<state dir>/control.sock`. From another machine, forward the socket over ssh:

```bash
ssh -N -L /tmp/egress-gateway.sock:/srv/egress-gateway/control.sock me@203.0.113.7 &
export BENCHFLOW_EGRESS_GATEWAY=/tmp/egress-gateway.sock
bench run tasks/my-denylist-task -a oracle --sandbox modal
```

`python3 egress_gateway_server.py status --state-dir /srv/egress-gateway` lists the trials it holds. With the variable unset, a `denylist` task on Modal is refused before any sandbox, as before.

What a trial gets:

- Each sandbox of a `denylist` task is created able to reach the gateway's address alone. Once it starts, BenchFlow registers it with the gateway and starts a relay as root on `127.0.0.1:18628`, the egress proxy's port, so the proxy variables the agent, the oracle and `test.sh` get are the ones they get on Docker. Before the trial's other steps, root proves that a direct connection fails, that the relay answers, and that a URL the task blocks is refused. If the gateway doesn't answer, or any of these fails, the sandbox is stopped and the trial ends with an error, unscored; nothing runs unconfined.
- BenchFlow's own installs run with no policy. From each agent connection on, every connection out of the sandbox gets the task's denylist: the agent's, the oracle's, and the model proxy's, which runs in the sandbox as on Daytona.
- The verifier gets the task's denylist (a task's verifier inherits it), or no policy when its `network_mode` is `public`. Its refusals count as they do on Docker.
- `trajectory/egress_denylist.jsonl` and `egress_allowed.jsonl` hold the proxy's refusals and the requests it let out during each connection. `modal-egress.json` in the trial folder records the gateway, the allowed address, each change of policy and each proof.

How it differs from Docker:

- Every process in the sandbox is under the policy, root included. A program that ignores `HTTP_PROXY` and `HTTPS_PROXY` (raw sockets, git over ssh, UDP) has no network at all (except DNS lookups, which Modal's resolver answers), where on Docker root keeps it.
- The model proxy is under the policy too: a denylist that names your model provider's host blocks the model, and a model endpoint given as an IP address or on a private network is refused.
- The proxy reads the sandbox's HTTPS traffic on the gateway, model calls and their keys included. Run the gateway on a host you would trust with them. The oracle's `[solution].env` values go to the gateway too, so that its proxy can redact them from its logs; they stay in the trial's folder under `--state-dir` (`deny.policy.json`, `rollout.json`) until the trial is deregistered or its time to live ends.
- Whoever can open the control socket, or its ssh forward, runs code on the gateway host as the gateway's user: a registration carries the proxy script the gateway runs.
- Every byte a sandbox exchanges with the internet crosses the gateway, and its cloud bills that traffic. Put the gateway in the region Modal runs your sandboxes, and size it to the run's concurrency (each trial runs one or two proxy processes there).
- One gateway is a single point of failure for its run. Its public port lets 256 connections wait for the challenge at once, from all sources together, for 10 seconds each: one host that holds that many open makes the gateway refuse every sandbox's new connections until they time out. That fails closed, but the run's trials fail with it; a per-source connection limit in the host's firewall prevents it.

Still refused on Modal, gateway or not: `allowlist` (Modal's resolver answers every DNS name inside the sandbox, so an allowlist's DNS filter can't hold), `no-network` with a model agent, a verifier's own `no-network` or `allowlist` in the agent's sandbox, a `denylist` task with an `[agent] network_mode` of its own, and the no-web policy.
