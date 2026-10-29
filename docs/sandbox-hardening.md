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
9. **Set `VERIFIER_ENV`** — pinned `PATH`, `PYTEST_ADDOPTS=-c /dev/null --confcutdir=/tests --rootdir=/app -p no:cacheprovider`, `PYTHONPATH=""`, `PYTHONDONTWRITEBYTECODE=1`, plugin auto-discovery off. The verifier runs with `HOME=/root`, which is often the agent's workspace, so per-user configuration there is switched off: Python's user site (`PYTHONNOUSERSITE=1`), every pip configuration file (`PIP_CONFIG_FILE=/dev/null`) and git's per-user config (`GIT_CONFIG_GLOBAL=/dev/null`, git 2.32+; `/etc/gitconfig` still applies). A task that needs one of these back sets it in its verifier `env`. Other tools that read `HOME` are not covered.
10. **Discover allowed pytest plugins** — `importlib.metadata` scan of the interpreter's own site-packages inside the container, then merge with the task-declared `pytest_plugins` from its config (`task.md` front-matter, or `task.toml` for split-layout tasks). Anything not in the allow-list is blocked; a plugin installed outside site-packages (a `PYTHONPATH` directory, a user site) must be declared.

The verifier then runs against this hardened workspace.

These steps, and the workspace evidence capture that runs before them, run small Python helpers as root. The sandbox's default working directory is the task's WORKDIR, which is the agent's workspace, and a plain `python3 -c` puts that directory first on `sys.path`. So every helper runs `python3 -I` (no working directory, user site or `PYTHON*` variables on its path): a `json.py` or `glob.py` the agent left in its workspace is never imported as root. The `.pth`/`sitecustomize.py` purge is the one exception: it must see the `PYTHONPATH` and user-site directories a verifier interpreter would load, so it keeps its environment, runs from `/`, and imports only `sys` and `os`.

## Separate verifier sandbox

In the default (shared) mode the verifier runs in the agent's sandbox after the steps above, so its integrity depends on that cleanup being complete. A separate verifier sandbox removes the dependence: the verifier never runs where the agent ran.

Request it per task with `[verifier] sandbox_mode = "separate"` (Harbor: `environment_mode`) or a `[verifier.sandbox]` table, or for every task of a run with `bench eval run --separate-verifier` (the task's files, and so its digest, stay unchanged). Then:

1. After the agent stops, its workspace is frozen into the rollout's `evidence/`, the same hash-checked capture the rubric review uses.
2. A fresh sandbox starts from the verifier image: `[verifier.sandbox].docker_image`, else the task's `[sandbox].docker_image`, else `tests/Dockerfile`, else the task's own `environment/Dockerfile`.
3. Only the frozen workspace is unpacked there, at its original path; a dedicated workspace directory such as `/app` is replaced, a shared one such as `/root` is laid over the image's copy. Nothing else from the agent's sandbox exists in the verifier sandbox: not `/logs/verifier`, not site-packages, not anything a process with root changed outside the workspace.
4. The hardening above (with no sandbox user) and `test.sh` run there; outputs are copied back to the rollout's `verifier/`.

`verifier-sandbox/verifier-sandbox.json` in the rollout records the image source, sandbox id, transfer inventory and status, and `config.json` records `verifier_sandbox_mode`. A failure before the verifier produces a reward — no frozen workspace, a manifest mismatch, an image that does not build, an upload that does not unpack — is a verifier infrastructure error (`verifier_error` starts with `separate verifier`), never a score. Both sandboxes are billed while the verifier runs.

**Outputs outside the workspace.** The workspace is `sandbox.workdir`, else the image's `WORKDIR`. A task whose agent writes graded files elsewhere (for example a `/app` workspace whose instructions ask for `/root/paper.pdf`) needs those paths carried too, or its verifier finds them missing and scores 0: pass `--separate-verifier-include /root` (repeatable; SDK and config `separate_verifier_include`). Each path is captured beside the workspace into `evidence/artifacts/`, hash-checked like it, and laid over the verifier image's copy at the same path; the rubric review sees it as an output outside the workspace. Pseudo filesystems and roots such as `/`, `/usr` or `/home` are refused. So that a missing include does not pass silently, the freeze also lists the regular files the agent phase wrote outside everything that crosses (skipping system, scratch and provider trees, package caches and the agent's own home): the log warns with a count per directory and examples, and `verifier-sandbox.json` records them under `outside_workspace_writes`. The list is advisory and never fails the rollout.

Package-manager caches (`.cache/pip`, `.cache/uv`, npm's `_cacache`, `_npx` and `_logs`) are left out of the capture and recorded as `package_cache` exclusions. With a `/root` workspace the harness's own installs fill them before the agent starts; on a FrontierPhysics run they were 97% of the 745 MB that crossed.

Limits: Docker and Daytona only; test-script verifiers on the `main` service; `soft_verify` is unavailable (it would upload `tests/` into the agent's sandbox); permission bits are normalized to 0644/0755 (the executable bit is kept); declared `artifacts` are still refused at launch, so only the workspace and include paths cross. The workspace capture needs Python 3.10+ in the agent image, provisioned as for rubric-reviewed runs: installed before the agent starts when the image lacks it.

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

This is a shared sandbox policy, independent of the task's domain, harness name,
model identifier, or provider. Every supported ACP harness, including custom
registrations, receives the same proxy, certificates and UID firewall. Primary
connections and later roles use the same enforcement path. Model compatibility
is still governed by each harness/provider adapter. The requirements and hosted
fetch limitations below apply regardless of model; they are not exceptions for
particular model names.

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

1. **Loopback proxy.** Before the agent starts, benchflow uploads a stdlib Python proxy (`src/benchflow/sandbox/_egress_denylist_proxy.py`) and starts it as root on `127.0.0.1:18628`. A request that matches the denylist gets `403 Forbidden` with an `X-BenchFlow-Blocked: 1` header; allowed HTTP requests are forwarded to their checked destination.
2. **Uid firewall.** The same `iptables` owner rule that backs the no-web mode lets the sandbox user reach loopback only. Every other outbound packet from that uid is rejected, so the proxy is the only way out. `iptables` is installed on first use (apt, dnf, or apk) when the image lacks it.
3. **Proxy and CA environment.** The agent env gets `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`, `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE`, `GIT_SSL_CAINFO`, `NODE_EXTRA_CA_CERTS`, and `NODE_USE_ENV_PROXY`, plus the `BENCHFLOW_EGRESS_DENYLIST=1` marker that arms the firewall. These are added after the sandbox-local LiteLLM gateway starts, so the gateway's upstream provider traffic does not pass through the egress proxy.
4. **HTTPS inspection.** The proxy terminates every external TLS connection with a leaf certificate signed by a per-rollout CA (`BenchFlow egress policy CA`). Known policy hosts have pre-generated leaves; OpenSSL mints other leaves on demand. The CA signing key and proxy files are root-only inside `/opt/benchflow-egress` (directory mode `0700`, files `0600`) and are removed at cleanup. `CONNECT`, TLS SNI when supplied, the HTTP Host, and any absolute request URL must agree on the destination; mismatches fail closed even when a client disables certificate verification. This prevents shared-CDN domain fronting. The proxy forwards one framed HTTP request per connection and does not relay later client bytes. Hosts in `blocked_hosts` are refused at `CONNECT` time.
5. **Hosted search off.** Provider-side search tools fetch pages from the model provider's servers, outside the sandbox, so the proxy cannot see them. Benchflow disables them per harness:

   | Harness | Switched off | Still on |
   |---|---|---|
   | `claude-agent-acp` | `WebSearch` | `WebFetch` (fetches from inside the sandbox, through the proxy) |
   | `codex-acp` | `web_search` (`CODEX_CONFIG`) | local shell tools |
   | `gemini` | `google_web_search`, `web_fetch` (tries a hosted fetch first) | |
   | `opencode`, `mimo` | `websearch` | `webfetch` |
   | other harnesses | nothing | whatever hosted tools they ship |

6. **Hosted fetches refused.** Switching off a harness's tool does not stop the agent from writing its own model request: it can call the model gateway (its address and key are in the agent's environment) or, on subscription and direct-provider routes, the provider through the proxy. Both refuse, with `403`, a request that asks the provider to fetch or search the web, wherever it sits in the request body, inline batch requests and other envelopes included: server tools (`web_search*`, `web_fetch*`, `url_context`, `google_search*`, `x_search`, remote `mcp` servers), request switches (`web_search_options`, `search_parameters`, `enable_search`, `plugins`, `mcp_servers`, `data_sources`, and LiteLLM's `extra_body`), and content given by address instead of bytes (any `scheme://` value in a file, image, video, audio or document part, Anthropic `url` sources, Gemini `file_data` outside its Files API). Tool definitions, JSON Schema property maps and the arguments of tool calls already made are data and are not read. The rules live in `src/benchflow/sandbox/_egress_policy.py`, which both enforcement points load.
   - **Gateway.** In no-web and denylist runs a guard wraps LiteLLM (`src/benchflow/providers/litellm_egress_guard.py`). It serves only the model-call routes harnesses use (chat completions, Responses, Messages, embeddings, Gemini GenerateContent, model listings, health), refuses every other route (provider pass-through, files, batches) and WebSocket sessions, and checks each request body. It also turns off LiteLLM's own image-URL downloads, whose traffic would bypass the proxy. benchflow confirms the guard is active before the agent starts and fails the rollout otherwise.
   - **Proxy.** The proxy reads each request body that opens like JSON whole before contacting the origin, whatever the host, the way a lenient server would (gzip and deflate undone, UTF-16 and UTF-32 detected, invalid bytes replaced, anything after the first value ignored). It refuses a body it cannot read (`uninspectable-body`: another content coding, more than 16 gzip members, or JSON nested deeper than its parser) and one over 64 MiB (`oversized-body`); every other body streams through as before. Codex runs with request compression off in these modes, because its zstd bodies cannot be read.
7. **Block log.** Each refused attempt is appended to a root-owned log that benchflow downloads to `trajectory/egress_denylist.jsonl` in the rollout directory at cleanup: one JSON object per line with `ts`, `action`, `method`, `url`, and `rule` (`host:<host>`, `url:<host><path>`, `ip-literal`, `private-address`, `authority-mismatch`, `tls-sni-mismatch`, `hosted-fetch:<feature>`, `uninspectable-body`, `oversized-body`, `gateway-route`, or `gateway-websocket`). A TLS SNI mismatch is rejected during the handshake and logged with `server_name`, rather than returning an HTTP response. For a refused `CONNECT`, `url` holds the requested hostname (with its port for pre-TLS denials). A refusal by the gateway guard names the gateway route in `url` (`http://127.0.0.1:<port>/<path>`); no-web runs have no block log, so their gateway refusals show only in the agent's trajectory.

The controller also registers the exact `127.0.0.1:<port>` endpoint of its local
model gateway with the proxy. Clients such as Gemini's Undici `ProxyAgent`
ignore `NO_PROXY` and tunnel even HTTP model calls through the egress proxy.
Both direct and proxied requests can reach that one endpoint. This exception
comes from the running provider gateway, never task metadata or agent-supplied
environment variables; other IP addresses and private destinations remain
blocked. Each reconnect registers the current gateway port.

Codex runs in its `agent-full-access` session mode when BenchFlow has already
selected a non-root sandbox user, unless the caller explicitly sets
`INITIAL_AGENT_MODE`. This avoids nesting Codex's bubblewrap sandbox inside
Docker or Daytona, where namespace creation can fail before a tool runs.
BenchFlow's user, filesystem restrictions, proxy and UID firewall still apply.
Hosted search is disabled through `CODEX_CONFIG.web_search`, including when a
caller configured live search; codex-acp's CLI does not consume `-c` overrides.

For OAuth runs, use the Claude Code harness (`claude-agent-acp`) with a bare
Claude model and `CLAUDE_CODE_OAUTH_TOKEN`, without `ANTHROPIC_API_KEY`. Its
native authenticated traffic passes through the same egress proxy, while
`WebSearch` stays disabled and local requests still receive policy denials.

Matching ignores scheme, port, query string, and case, strips a leading `www.`, and compares a normalized path: percent-encoding is decoded (repeatedly), `.` and `..` segments are resolved, duplicate slashes and backslashes collapse, and `;` path parameters are dropped, so `/abs/../abs/2401.12345` and `/abs/%2e%2e/abs/2401.12345` match the same entry as `/abs/2401.12345`. A `blocked_urls` entry blocks every path under it; a `blocked_hosts` entry blocks the host and its subdomains. A listed page is also refused when its address appears, decoded, inside another URL's path or query, so a reader service, web archive or redirector cannot fetch it for the agent (`r.jina.ai/https://arxiv.org/abs/2401.12345`, `web.archive.org/web/2026/https://arxiv.org/abs/2401.12345`, `?url=https%3A%2F%2Farxiv.org%2Fabs%2F2401.12345`); the log names the listed rule. Apart from the registered model gateway, requests to addresses are refused in every notation a resolver accepts (dotted, decimal, hex, octal) and through wildcard DNS names that embed an address (`1-2-3-4.sslip.io`), so a blocked host cannot be reached by its address. A name the agent controls that resolves to the blocked address is not detected; that is the inherent limit of a hostname denylist. For every other destination, the proxy resolves its address and refuses names that resolve to loopback, private, link-local, or other non-global addresses (cloud metadata included), so a hostname the agent controls cannot turn the root proxy into a bridge to sandbox-internal or host services. The uid firewall stays for the rest of the sandbox life, as in the no-web mode: a later oracle role in the same sandbox, and a verifier configured with `verifier.user` equal to the sandbox user, run without egress.

### Requirements

- A non-root `sandbox_user`. Setup fails closed before any sandbox is created when it is missing.
- `python3` (or `python`) and `openssl` on `PATH` in the task image. The proxy is a stdlib script; OpenSSL signs dynamic leaf certificates. Setup fails closed if either runtime is absent.
- An ACP agent. Session-factory agents raise at connect time because the uid firewall only runs in the ACP path.
- `docker`: the agent container needs `NET_ADMIN` for `iptables`. Benchflow adds it through its own compose overlay (`src/benchflow/sandbox/_compose_files/docker-compose-net-admin.yaml`), so the task's `Dockerfile` and `docker-compose.yaml` need nothing extra.
- `daytona`: verified on direct sandboxes with `iptables`.
- `modal`, `apple-container`, and `agentcore` refuse the mode at preflight. `bench tasks check --sandbox <backend>` reports it, and the runtime raises before the sandbox is created.

### Limits

- **A blocklist hides pages, not knowledge.** The model may have read the paper in training. The denylist only stops it from re-reading the page during the run.
- **Mirrors and citations stay reachable.** Only listed prefixes and hosts are blocked, wherever their address appears in a URL. Preprint mirrors, aggregator copies, cached versions, citing papers that restate the result, and the same repository under another host all stay reachable unless listed, as does a service handed the address in a request body or another encoding. `blocked_hosts` covers subdomains, not other domains. List every form of a page you know about (abstract, PDF, HTML, versioned).
- **The block is visible.** A refused request gets a `403` that names the policy, and the agent can tell intercepted hosts from the certificate issuer (`BenchFlow egress policy CA`). Do not expect the agent to be unaware that a page is off limits.
- **Tools that ignore proxy variables fail closed.** A client that does not honor `HTTP_PROXY` and `HTTPS_PROXY` cannot reach the network at all, because the uid firewall rejects non-loopback traffic. It gets a connection error, not the page, and the attempt does not appear in the block log.
- **Hosted fetches are refused by feature.** The harness switches above keep the agent's own tools working; the gateway guard and the proxy refuse the provider features listed in step 6 whichever harness or script sends them. Some routes stay open: a provider feature that is not on that list (a provider-hosted shell container given network access, for one), a provider or model that browses by default, a request in a binary encoding the proxy does not read (protobuf, for one), and a batch submitted as an uploaded file (the gateway refuses file and batch routes, but a provider key in the agent's environment reaches them through the proxy).
- **The body check applies to every JSON request.** The proxy cannot tell a model provider from any other host, so any JSON body it forwards is read whole (and refused above 64 MiB), and one that sets a field from step 6 anywhere in the body (a `plugins` list, say) is refused whatever the host.
- **External egress is HTTP/1.x.** External `CONNECT` requests must carry TLS and a checked HTTP/1.x request. Arbitrary TCP tunnels, HTTP upgrades, and clients pinned to an origin certificate are not supported. TLS clients must trust the injected CA; HTTP/2-capable clients must permit HTTP/1.1 fallback. Only the exact controller-registered loopback HTTP model gateway keeps opaque tunneling. A client that ignores the HTTP proxy protocol is rejected by the firewall.

## Threat model and known gaps

Benchflow's hardening assumes:
- The Docker image and pre-installed packages are trusted (they're built before the agent runs).
- The agent runs as a non-root sandbox user (`sandbox_user`) — when set to `None`, hardening still runs but the agent can defeat lockdown by being root. Don't run untrusted agents as root.
- The verifier itself is trusted code shipped with the task; benchflow only hardens *what the agent can do to the verifier's environment*.
- In shared mode, the agent never gains root in its sandbox. A separate verifier sandbox drops this assumption for everything outside the workspace: only the frozen workspace reaches the verifier.

Known residual risk:
- An agent with sustained access can poison `__pycache__` for files that exist in the baseline (those caches aren't deleted because some tasks diff workspace against `/testbed_verify`). Mitigated by the workspace chown but not eliminated.
- Tasks that don't ship a build-config snapshot can still be hijacked via `setup.py` edits. Snapshot is automatic for declared filenames — task authors don't need to opt in.

## Related

- [`progressive-disclosure.md`](./progressive-disclosure.md) — soft-verify (the relaxed hardening used between rounds in multi-round trials).
- [`task-authoring.md`](./task-authoring.md) — the task config schema, including the `[verifier.hardening]` opt-outs.
