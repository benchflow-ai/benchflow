# Network and secrets

A task decides what the agent can reach while it works: the open internet, nothing but its model, or the internet minus the sites that would give the answer away. Which of your keys and files reach the agent is under [What enters the sandbox](#what-enters-the-sandbox).

## What the agent can reach

A task sets its policy with `sandbox.network_mode`:

| `network_mode` | The agent can reach | Sandboxes |
|---|---|---|
| `public` (the default) | the internet, except cloud metadata addresses | all |
| `no-network` | only its model, through BenchFlow's model proxy | Docker, remote Docker; Daytona and Modal only for an agent that calls no model (the oracle) |
| `denylist` | the internet, minus `blocked_urls` and `blocked_hosts` | Docker, remote Docker |
| `allowlist` | only `allowed_hosts`, plus its model | Docker, remote Docker |

In 0.9.2, an `allowlist` names a task's own Compose services by their private address range, and one with no hosts is refused when the task loads; 0.10.0rc2 also takes the services' names, and an empty list then admits only the model ([What's in 0.10.0rc2](../versions/0.10.md#changes-that-can-move-a-score-or-stop-a-task)).

Under every policy, no process of the trial reaches the cloud metadata addresses (a cloud VM's instance credentials): not the agent, not root, not `test.sh`, not the task's other services. A task about the metadata service itself asks to lift that block with `sandbox.allow_cloud_metadata: true` (task.toml: `allow_cloud_metadata = true` in `[environment]`; the older `metadata.x-allow-cloud-metadata = true` is the same request). A task can't lift it by itself: the key takes effect only when the run also passes `--allow-cloud-metadata` (the run config's `allow_cloud_metadata: true`, `allow_cloud_metadata=True` in the SDK), and a run without that flag refuses such a task before anything starts, naming the task and the flag. With the flag, the run warns at start and each trial's `result.json` (`isolation_opt_ins`) and reward provenance name the key and the flag. Before 0.10.1 the task's key worked alone.

On Docker and remote Docker a task's Compose services start only once their rules are in. Each one, `main` included, starts in BenchFlow's own waiter (`bfx`, mounted read-only), and nothing of the image runs until BenchFlow has added and proven its rules. The waiter then starts the service's own command with the environment Docker gave it, and its healthcheck counts as Docker's would from that moment. Before 0.10.1, under an open network, a service's own start-up code ran first, and the hold itself ran the image's shell. An image with no shell (distroless, `scratch`) is held the same way. A host without a `bfx` for its architecture (or with `BENCHFLOW_BFX_DELIVERY=off`) refuses such a start, the oracle's too, naming why: deliver `bfx` (a release wheel ships it), or, under an open network, grant the task's own `allow_cloud_metadata` with `--allow-cloud-metadata`, so no rule is waited for and `main` waits for its bind proof alone, without `bfx`, as on 0.10.0. `BENCHFLOW_BFX_DELIVERY=off` refuses every other Docker trial. A service that Docker restarts by its `restart:` policy during the start gets its rules again before it runs. One that restarts later in the trial stays waiting, since its rules can't be put back and proven. The trial is scored as `test.sh` says, and `result.json` names the service (`held_after_restart`: when, its exit status and why) and flags the trial in its `warnings` for you to read. A service whose container exits and that Docker doesn't start again (no `restart:` policy) runs nothing from then on, so its rules have nothing to prove: the trial is scored as `test.sh` says. `result.json` names it (`held_exited`: its exit status and when it exited, at start-up, in the agent's turn or in `test.sh`). An exit after start-up also flags the trial in its `warnings`; a one-shot service done at start-up is the task's own shape and isn't flagged. On 0.10.0 an exit in the agent's turn ended the trial "network policy not proven", unscored.

The policy applies to the agent's own user while it works. The agent's install and start-up have the internet. The verifier's `test.sh` runs as root after the agent stops, under `verifier.network_mode` when the task sets it, else the sandbox's policy. What its network refuses `test.sh` (a packet at BenchFlow's rules, or a request at its egress proxy) is counted in `verifier/verifier-network.json`, and the trial is scored as `test.sh` says, a pass or a failure: BenchFlow can't tell `test.sh`'s own packets from those of code the agent left that `test.sh` runs. `result.json` names it (`verifier_network_refused`: the phase, the count, `test.sh`'s exit and whether it passed) and flags the trial in its `warnings` for you to read. Before 0.10.1 a `test.sh` that failed after a refusal ended "verifier network refused", unscored. A task whose policy its sandbox can't enforce is refused before it starts; it never runs with a weaker one. On Daytona and Modal every restricted policy (`allowlist`, `denylist`, an offline task, `no-network` with a model agent, a shared verifier's own network) is refused before any sandbox (`R401`): there the rules would go in from the agent's own container, where root holds `NET_ADMIN` (Daytona), or gVisor gives no iptables (Modal). Run such a task on Docker or remote Docker, or set `network_mode: open`.

A service the task declares in the `environment.toml` at its root (`[[environment.services]]`, which BenchFlow starts in the main container) can't run beside a model agent under `denylist`, `allowlist` or the agent's own `no-network` (`agent.network_mode`). The policy holds only the agent's user; the service runs as root, outside it, and the agent reaches it on localhost, so such a trial is refused before it starts. Run the service as a Compose service on an internal network that `main` shares instead.

Under `allowlist`, a task's own Compose services may sit only on internal networks: networks the compose file marks `internal: true`. A service on the default network, or on one merely named `internal` without `internal: true`, has a route out of the sandbox, and the run is refused before the agent starts, naming the service and its network. `main` may join such an internal network beside the default one:

```yaml
services:
  main:
    networks: [default, svc]
  db:
    image: postgres:16
    networks: [svc]
networks:
  svc:
    internal: true
```

We ran one probe script as the agent under three policies, on three copies of the half-life task (`t-public/`, `t-no-network/`, `t-denylist/`, each with its `network_mode` set). It tries two sites and lists any environment variable whose name looks like a key:

```bash
cat > net-probe.sh <<'EOF'
#!/bin/bash
# Probe: which sites can the agent reach? And which environment variables can it see (names only)?
python3 - <<'PY'
import urllib.error
import urllib.request
for url in ["https://example.com", "https://pypi.org/simple/"]:
    try:
        code = urllib.request.urlopen(url, timeout=10).status
        print(url, f"reachable (HTTP {code})")
    except urllib.error.HTTPError as e:
        print(url, f"reachable (HTTP {e.code})")
    except Exception as e:
        print(url, "unreachable:", type(e).__name__, str(e)[:80])
PY
echo "variables named like a key: $(env | cut -d= -f1 | grep -E 'KEY|TOKEN|SECRET' | sort | tr '\n' ' ')"
EOF
chmod +x net-probe.sh
bench run t-public/half-life -a script:./net-probe.sh
```

| Policy | Output (`agent/script.txt`) |
|---|---|
| `public` | `https://example.com reachable (HTTP 200)`<br />`https://pypi.org/simple/ reachable (HTTP 200)` |
| `no-network` | `https://example.com unreachable: URLError <urlopen error [Errno -3] Temporary failure in name resolution>`<br />`https://pypi.org/simple/ unreachable: URLError <urlopen error [Errno -3] Temporary failure in name resolution>` |
| `denylist`, blocking `example.com` | `https://example.com unreachable: URLError <urlopen error Tunnel connection failed: 403 Forbidden>`<br />`https://pypi.org/simple/ reachable (HTTP 200)` |

## Keep the agent off the web

For a task whose answer is published somewhere, use `no-network`, or `allowlist` with only the hosts the task needs:

```yaml
sandbox:
  network_mode: no-network
```

Under `no-network`, `denylist` and `allowlist`, BenchFlow also turns off the model provider's own web search, and refuses models that search the web by themselves. A model BenchFlow can't classify is refused too, with what to do:

```text
RestrictedModelRefused: Refused 'xiaomi/mimo-v2.6-flash' (sent to the provider as 'openai/mimo-v2.6-flash') with web access restricted: BenchFlow can't confirm it has no hosted web search or retrieval (its effective provider, endpoint and base model are not in the curated first-party model allowlist). next: if it runs without web access, set BENCHFLOW_RESTRICTED_WEB_MODELS=xiaomi/mimo-v2.6-flash in the controller's environment (comma-separated ids; 'xiaomi/*' allows an operator-run provider). The run records the override in config.json (restricted_web_model).
```

That was Claude Code on `xiaomi/mimo-v2.6-flash`, with the half-life task set to `no-network`; the message names the model by the route BenchFlow resolved it to. If you know the model doesn't search the web, set the variable the message gives in your shell, not with `--agent-env`, and run again. Each trial's `config.json` records how its model was let through. The check runs before any image is built or sandbox started, and again when the agent starts; a refused trial is filed as `restricted_model_refused`, your setup's to fix.

## Block given sites

When the agent needs the internet but not the paper or repository with the answer, block those:

```yaml
sandbox:
  network_mode: denylist
  blocked_urls:
    - https://example.org/papers/lattice-qcd-2026
  blocked_hosts:
    - mirror.example.net
```

A `blocked_urls` entry blocks that path and every path under it; a `blocked_hosts` entry blocks the host and all its subdomains.

## What enters the sandbox

Your shell's variables stay on your machine. In the probe above, the host had `XIAOMI_API_KEY` and `DEEPSEEK_API_KEY` exported, and the sandbox saw no variable named like a key, in all three modes:

```text
variables named like a key:
```

| What | Reaches the sandbox? |
|---|---|
| A model API key you exported | Not to the agent. With an API key, the agent gets a key for BenchFlow's model proxy, never yours. The proxy holds yours: on your machine for local Docker with an open network; inside each sandbox on Daytona and Modal, and on Docker under a restricted network, where the proxy runs as root ([Daytona](../scale/daytona.md), [Modal](../scale/modal.md)). |
| A subscription login (Claude, ChatGPT) | Only that login, for that agent. |
| A gcloud credential file (a Vertex model's application default credentials) | Only with `--allow-cloud-credential-file`: the Gemini CLI's own Vertex route writes it into the agent's home, and a proxy in the sandbox keeps it in its runtime folder. Without the flag such a run is refused before it starts. The proxy on the host (Docker, `public`) keeps it on your machine and needs no flag. |
| `oracle`, `nop` and script agents | No credentials at all. |
| A variable you pass with `--agent-env` | Yes, to the agent. Don't pass provider keys such as `DAYTONA_API_KEY` this way. |
| A variable a task's files ask for | Only if you name it with `--allow-task-env NAME`; otherwise the task is refused before it starts. |

A process in the sandbox can read whatever is staged there: task code the agent runs, and under `public` the task's own processes that run on into the agent's turn as root. Each trial that stages one of your credentials there says so when it starts, and its `result.json` names each one (`operator_credentials`: its kind, route, variable names and where it is staged, never a value; under `would_stage` when the trial was refused before anything went in). With tasks you don't trust, use API keys through the proxy on the host (Docker, `public`), not a subscription login; rotate a login that ran such tasks.

When one of your credentials goes into the sandbox, the agent starts only if main runs nothing but BenchFlow's own processes, checked before anything is staged, and again before each later step or round once the task's tests have run as root (Docker and remote Docker; Daytona and Modal can't list main's processes yet, and the record says `"quiet_main": "not checked on daytona"`). A task that leaves a server running in main, or runs a service there of its own (one its `environment.toml` declares), is refused, unless you pass `--allow-task-processes-with-credentials`, which each trial records in `result.json` and in its reward's provenance. With an API key through the proxy on the host (Docker, `public`) nothing of yours is in the sandbox, so such a task needs no flag.

In a multi-step run whose steps don't all stage a credential, a process the agent starts in a step that stages none counts as the task's at the next step that does, and that step is refused. Give every step the same credentials, or pass `--allow-task-processes-with-credentials`.

A credential that is the run's own, not one of your logins or keys, can be named with `--scoped-credential NAME` (repeatable; run config `scoped_credentials: [NAME]`). It still goes into the sandbox, and each trial records it as `scoped`, but a task process in main is not refused for it. A hosted run's per-run token is one: `--scoped-credential ANTHROPIC_AUTH_TOKEN`, with `ANTHROPIC_BASE_URL` at the service's key proxy. A name is scoped only where it holds the run's own value: a scene role with another value under the same name is your credential. A name the agent's env doesn't set, one that is no credential (a base URL), or one staged as a file (a login file, the gcloud credential file), is refused before the run starts.

`bench eval continue` and `bench branch` take both flags too: on Daytona and Modal a continuation's live model runs through a proxy in the sandbox, which holds that model's key. `bench check` takes `--allow-cloud-credential-file`, so a task you would run with it can be checked.

## What the results keep

Trial folders hold the trajectory, the model calls and the agent's logs, so they can contain whatever the agent printed. `bench train convert` and `bench traj upload` redact secrets on the way out. Read a folder before you share it.
