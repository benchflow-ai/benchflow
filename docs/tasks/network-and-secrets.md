# Network and secrets

A task decides what the agent can reach while it works: the open internet, nothing but its model, or the internet minus the sites that would give the answer away. Which of your keys and files reach the agent is under [What enters the sandbox](#what-enters-the-sandbox).

## What the agent can reach

A task sets its policy with `sandbox.network_mode`:

| `network_mode` | The agent can reach | Sandboxes |
|---|---|---|
| `public` (the default) | the internet, except cloud metadata addresses | all |
| `no-network` | only its model, through BenchFlow's model proxy | all but AgentCore and Apple Container |
| `denylist` | the internet, minus `blocked_urls` and `blocked_hosts` | Docker, remote Docker, Daytona |
| `allowlist` | only `allowed_hosts`, plus its model | Docker, remote Docker, Daytona |

In 0.9.1, an `allowlist` names a task's own Compose services by their private address range, and one with no hosts is refused when the task loads; 0.10.0rc1 also takes the services' names, and an empty list then admits only the model ([What's in 0.10.0rc1](../versions/0.10.md#changes-that-can-move-a-score-or-stop-a-task)).

The policy applies to the agent's own user while it works. The agent's install and start-up have the internet; the verifier, which runs as root after the agent stops, isn't filtered. A task whose policy its sandbox can't enforce is refused before it starts; it never runs with a weaker one.

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
RuntimeError: With web access restricted, BenchFlow runs only models it can identify as having no hosted web search or retrieval. 'openai/mimo-v2.6-flash' is not one: its effective provider, endpoint and base model are not in the curated first-party model allowlist. If this model runs without web access, allow it explicitly in the controller's environment: BENCHFLOW_RESTRICTED_WEB_MODELS=openai/mimo-v2.6-flash (comma-separated ids; 'xiaomi/*' allows an operator-run provider). The run records the override in config.json (restricted_web_model).
```

That was Claude Code on `xiaomi/mimo-v2.6-flash`, with the half-life task set to `no-network`; the message names the model by the route BenchFlow resolved it to. If you know the model doesn't search the web, set the variable the message gives in your shell, not with `--agent-env`, and run again. Each trial's `config.json` records how its model was let through. The check runs when the agent starts, after the sandbox is up (2.4 minutes into this run), so try one trial before a large run.

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
| A model API key you exported | Not to the agent. With an API key, the agent gets a key for BenchFlow's model proxy, never yours. The proxy holds yours: on your machine for local Docker; inside each sandbox on Daytona and Modal, where the proxy runs ([Daytona](../scale/daytona.md), [Modal](../scale/modal.md)). |
| A subscription login (Claude, ChatGPT) | Only that login, for that agent. |
| `oracle`, `nop` and script agents | No credentials at all. |
| A variable you pass with `--agent-env` | Yes, to the agent. Don't pass provider keys such as `DAYTONA_API_KEY` this way. |
| A variable a task's files ask for | Only if you name it with `--allow-task-env NAME`; otherwise the task is refused before it starts. |

## What the results keep

Trial folders hold the trajectory, the model calls and the agent's logs, so they can contain whatever the agent printed. `bench train convert` and `bench traj upload` redact secrets on the way out. Read a folder before you share it.
