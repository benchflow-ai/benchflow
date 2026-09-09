# Sandbox hardening
## Why

Frontier agents will reward-hack any verifier they can. [BenchJack](https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/) (Berkeley RDI, April 2026) showed that ~100% of tasks across eight major AI agent benchmarks are solvable by exploiting the *verifier's runtime* rather than the agent's intelligence: pytest `conftest.py` hooks, planted PATH binaries, leaked answer keys, `.pth` injection, `sitecustomize.py` shadowing. [Meerkat](https://debugml.github.io/cheating-agents/) (Penn, same week) confirmed it from the trace side — frontier models discover and use these shortcuts in the wild.

benchflow's verifier hardening blocks these patterns by default. Tasks opt out individually when their legitimate test setup requires a hook the cleanup would otherwise delete (e.g. qutebrowser's `conftest.py` resolves a real circular import).

## What gets hardened

Between the agent's last action and the verifier's first command, `harden_before_verify()` (in `src/benchflow/sandbox/lockdown.py`) runs this sequence:

1. **Kill sandbox-user processes** — `pkill -u <sandbox_user>`, twice with a sleep, so the agent can't keep writing during teardown.
2. **Wipe `/logs/verifier`** — fresh, root-owned, mode 0777 (so a non-root verifier can write its own results).
3. **Restore build-config files** from a pre-agent snapshot — `setup.py`, `pyproject.toml`, `setup.cfg`, `tox.ini`, `noxfile.py`, `hatch.toml`, `flit.ini`, `MANIFEST.in`, `requirements.txt`, `requirements-dev.txt`, `Makefile`. Closes the editable-install hijack vector.
4. **Optionally restore the entire workspace** from `/testbed_verify` snapshot — opt-in per task because some tasks score workspace edits.
5. **Purge symlinks whose realpath escapes the workspace** — agent can't redirect to `/tmp/evil.py`.
6. **Purge `__pycache__` trees** that didn't exist in the pre-agent baseline.
7. **chown workspace to root** — belt-and-suspenders against zombie sandbox-user processes.
8. **`CLEANUP_CMD`** — delete `conftest.py` outside `/tests/`, `*.py` from `/tmp`/`/var/tmp`, `sitecustomize.py`/`usercustomize.py`/`*.pth` from writable `sys.path` entries.
9. **Set `VERIFIER_ENV`** — pinned `PATH`, `PYTEST_ADDOPTS=-c /dev/null --confcutdir=/tests --rootdir=/app -p no:cacheprovider`, `PYTHONPATH=""`, `PYTHONDONTWRITEBYTECODE=1`, plugin auto-discovery off.
10. **Discover allowed pytest plugins** — `importlib.metadata` scan inside the container, then merge with the task-declared `pytest_plugins` from its config (`task.md` front-matter, or `task.toml` for split-layout tasks). Anything not in the allow-list is blocked.

The verifier then runs against this hardened workspace.

## Per-task opt-outs

Tasks declare opt-outs in their task config (`task.md` front-matter, or `task.toml` for split-layout tasks):

```toml
[verifier.hardening]
cleanup_conftests = false
```

| Flag | Default | Effect when `false` |
|------|---------|---------------------|
| `cleanup_conftests` | `true` | Don't delete `conftest.py` outside `/tests/` before verify |

Other cleanup steps (`sitecustomize.py`, `.pth`, `/tmp` `*.py`) always run — they have no legitimate use in a test artifact and disabling them would broaden the attack surface beyond what real tasks need.

Unknown keys in `[verifier.hardening]` are warned and ignored. String values for boolean flags are rejected.

See [`progressive-disclosure.md`](./progressive-disclosure.md#per-task-hardening-opt-outs) for the qutebrowser case study (legitimate `conftest.py` for circular-import fix).

## Network policy: denylist egress

`network_mode: denylist` keeps the internet reachable and makes a list of URLs and hosts unreachable for the agent. The use case is a task built from a published paper: the agent may search and read freely, but the paper, its mirrors, and its code repository are off limits ([benchflow-ai/FrontierPhysics#365](https://github.com/benchflow-ai/FrontierPhysics/issues/365)).

```yaml
sandbox:
  network_mode: denylist
  blocked_urls:
    - https://example.org/papers/lattice-qcd-2026
    - github.com/example-org/lattice-qcd-code
  blocked_hosts:
    - mirror.example.net
```

See [task authoring](./task-authoring-task-md.md#network-policy) for the field rules. The proxy filters the agent only; oracle runs and the verifier are not filtered.

### Mechanism

1. **Loopback proxy.** Before the agent starts, benchflow uploads a stdlib Python proxy (`src/benchflow/sandbox/_egress_denylist_proxy.py`) and starts it as root on `127.0.0.1:18628`. A request that matches the denylist gets `403 Forbidden` with an `X-BenchFlow-Blocked: 1` header; everything else is tunneled to its destination.
2. **Uid firewall.** The same `iptables` owner rule that backs the no-web mode lets the sandbox user reach loopback only. Every other outbound packet from that uid is rejected, so the proxy is the only way out. `iptables` is installed on first use (apt, dnf, or apk) when the image lacks it.
3. **Proxy and CA environment.** The agent env gets `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`, `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE`, `GIT_SSL_CAINFO`, `NODE_EXTRA_CA_CERTS`, and `NODE_USE_ENV_PROXY`, plus the `BENCHFLOW_EGRESS_DENYLIST=1` marker that arms the firewall. These are added after the sandbox-local LiteLLM gateway starts, so model traffic does not pass through the egress proxy.
4. **Selective TLS interception.** Hosts named in `blocked_urls` need their paths inspected, so the proxy terminates TLS for those hosts with a leaf certificate signed by a per-rollout CA (`BenchFlow egress policy CA`). Certificates are minted on the host; the CA private key never enters the sandbox. Hosts in `blocked_hosts` are refused at `CONNECT` time, and every other host passes through as an opaque tunnel.
5. **Hosted search off.** Provider-side search tools fetch pages from the model provider's servers, outside the sandbox, so the proxy cannot see them. Benchflow disables them per harness:

   | Harness | Switched off | Still on |
   |---|---|---|
   | `claude-agent-acp` | `WebSearch` | `WebFetch` (fetches from inside the sandbox, through the proxy) |
   | `codex-acp` | `tools.web_search` | |
   | `gemini` | `google_web_search`, `web_fetch` (tries a hosted fetch first) | |
   | `opencode`, `mimo` | `websearch` | `webfetch` |
   | other harnesses | nothing | whatever hosted tools they ship |

6. **Block log.** Each refused attempt is appended to a root-owned log that benchflow downloads to `trajectory/egress_denylist.jsonl` in the rollout directory at cleanup: one JSON object per line with `ts`, `action`, `method`, `url`, and `rule` (`host:<host>`, `url:<host><path>`, or `ip-literal`). For a refused `CONNECT`, `url` holds the `host:port` the client asked for.

Matching ignores scheme, port, query string, and case, strips a leading `www.`, and compares a normalized path: percent-encoding is decoded (repeatedly), `.` and `..` segments are resolved, duplicate slashes and backslashes collapse, and `;` path parameters are dropped, so `/abs/../abs/2401.12345` and `/abs/%2e%2e/abs/2401.12345` match the same entry as `/abs/2401.12345`. A `blocked_urls` entry blocks every path under it; a `blocked_hosts` entry blocks the host and its subdomains. Requests to addresses are refused in every notation a resolver accepts (dotted, decimal, hex, octal) and through wildcard DNS names that embed an address (`1-2-3-4.sslip.io`), so a blocked host cannot be reached by its address. A name the agent controls that resolves to the blocked address is not detected; that is the inherent limit of a hostname denylist. Before connecting anywhere, the proxy resolves the destination and refuses names that resolve to loopback, private, link-local, or other non-global addresses (cloud metadata included), so a hostname the agent controls cannot turn the root proxy into a bridge to sandbox-internal or host services. The uid firewall stays for the rest of the sandbox life, as in the no-web mode: a later oracle role in the same sandbox, and a verifier configured with `verifier.user` equal to the sandbox user, run without egress.

### Requirements

- A non-root `sandbox_user`. Setup fails closed before any sandbox is created when it is missing.
- `python3` (or `python`) on `PATH` in the task image. The proxy is a stdlib script and installs nothing.
- An ACP agent. Session-factory agents raise at connect time because the uid firewall only runs in the ACP path.
- `docker`: the agent container needs `NET_ADMIN` for `iptables`. Benchflow adds it through its own compose overlay (`src/benchflow/sandbox/_compose_files/docker-compose-net-admin.yaml`), so the task's `Dockerfile` and `docker-compose.yaml` need nothing extra.
- `daytona`: verified on direct sandboxes with `iptables`.
- `modal`, `apple-container`, and `agentcore` refuse the mode at preflight. `bench tasks check --sandbox <backend>` reports it, and the runtime raises before the sandbox is created.

### Limits

- **A blocklist hides pages, not knowledge.** The model may have read the paper in training. The denylist only stops it from re-reading the page during the run.
- **Mirrors and citations stay reachable.** Only listed prefixes and hosts are blocked. Preprint mirrors, aggregator copies, cached versions, citing papers that restate the result, and the same repository under another host all stay reachable unless listed. `blocked_hosts` covers subdomains, not other domains. List every form of a page you know about (abstract, PDF, HTML, versioned).
- **The block is visible.** A refused request gets a `403` that names the policy, and the agent can tell intercepted hosts from the certificate issuer (`BenchFlow egress policy CA`). Do not expect the agent to be unaware that a page is off limits.
- **Tools that ignore proxy variables fail closed.** A client that does not honor `HTTP_PROXY` and `HTTPS_PROXY` cannot reach the network at all, because the uid firewall rejects non-loopback traffic. It gets a connection error, not the page, and the attempt does not appear in the block log.
- **Hosted search coverage is per harness.** Only the harnesses in the table have a switch. A harness with a hosted fetch that is not listed there can reach blocked pages through the provider.
- **Only HTTP requests and `CONNECT` targets are inspected.** A tunnel to an unblocked host carries any protocol the client chooses, uninspected. A client that does not speak the HTTP proxy protocol at all is rejected by the firewall rather than filtered.

## Threat model and known gaps

Benchflow's hardening assumes:
- The Docker image and pre-installed packages are trusted (they're built before the agent runs).
- The agent runs as a non-root sandbox user (`sandbox_user`) — when set to `None`, hardening still runs but the agent can defeat lockdown by being root. Don't run untrusted agents as root.
- The verifier itself is trusted code shipped with the task; benchflow only hardens *what the agent can do to the verifier's environment*.

Known residual risk:
- An agent with sustained access can poison `__pycache__` for files that exist in the baseline (those caches aren't deleted because some tasks diff workspace against `/testbed_verify`). Mitigated by the workspace chown but not eliminated.
- Tasks that don't ship a build-config snapshot can still be hijacked via `setup.py` edits. Snapshot is automatic for declared filenames — task authors don't need to opt in.

## Related

- [`progressive-disclosure.md`](./progressive-disclosure.md) — soft-verify (the relaxed hardening used between rounds in multi-round trials).
- [`task-authoring.md`](./task-authoring.md) — the task config schema, including the `[verifier.hardening]` opt-outs.
