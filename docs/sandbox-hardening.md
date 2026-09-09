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

## Threat model and known gaps

Benchflow's hardening assumes:
- The Docker image and pre-installed packages are trusted (they're built before the agent runs).
- The agent runs as a non-root sandbox user (`sandbox_user`) — when set to `None`, hardening still runs but the agent can defeat lockdown by being root. Don't run untrusted agents as root.
- The verifier itself is trusted code shipped with the task; benchflow only hardens *what the agent can do to the verifier's environment*.

Known residual risk:
- An agent with sustained access can poison `__pycache__` for files that exist in the baseline (those caches aren't deleted because some tasks diff workspace against `/testbed_verify`). Mitigated by the workspace chown but not eliminated.
- Tasks that don't ship a build-config snapshot can still be hijacked via `setup.py` edits. Snapshot is automatic for declared filenames — task authors don't need to opt in.

## Egress blocklist (`network_mode = "blocklist"`)

`no-network` and `allowlist` answer "may the agent reach the internet at all".
The blocklist answers a different research question: **the agent may use the
whole web except a list of URLs it must not discover** — e.g. hide the paper
under evaluation (and its mirrors) from a deep-research agent while every
other paper stays readable.

```toml
[sandbox]
network_mode = "blocklist"
blocked_urls = [
  "https://arxiv.org/abs/2401.12345",   # normalized to arxiv.org/abs/2401.12345
  "arxiv.org/pdf/2401.12345",           # list every mirror path you care about
  "openreview.net",                     # a bare host blocks it and its subdomains
]
```

Per run, without editing the task: `bench eval run … --block-url arxiv.org/abs/2401.12345 --block-url-file hidden.txt`
(the run-level list replaces the task's `blocked_urls`; the config is
re-validated, so a `no-network` or `allowlist` task fails loudly, and so does
a task whose `agent` section pins its own `network_mode` — that override would
otherwise shadow the sandbox blocklist).

### How it is enforced

A URL blocklist has to be enforced in three places, because no single layer
sees every route an agent has to the web (`src/benchflow/sandbox/egress.py`):

1. **Sandbox-local filtering proxy.** Before the agent launches, a root-run
   stdlib proxy starts on the sandbox loopback and the agent env gets
   `HTTP_PROXY`/`HTTPS_PROXY` (plus `NODE_USE_ENV_PROXY=1` for Node fetch).
   Plain HTTP exposes the full URL, so host and path rules both apply. HTTPS
   `CONNECT` only exposes the host: a host rule rejects the tunnel; a host that
   carries **path** rules is TLS-inspected — the proxy terminates TLS with a
   leaf certificate signed by a per-run CA (installed into the system trust
   store and exported via `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`,
   `CURL_CA_BUNDLE`, `NODE_EXTRA_CA_CERTS`, `GIT_SSL_CAINFO`), reads the path,
   and forwards allowed requests upstream over a fresh verified TLS connection.
   The CA key, the rule file, and the log are root-only.
2. **Agent-UID firewall.** The same iptables rule the no-web policy uses
   confines the agent UID to loopback, so anything that ignores the proxy env
   (a hand-rolled socket, `curl --noproxy`, DoH) fails closed instead of
   bypassing the filter. Under the blocklist it is applied **before the agent
   process is launched** (the proxy is already listening), so there is no
   pre-handshake window; the no-web policy keeps its post-handshake placement. The LiteLLM model proxy is forced
   sandbox-local for the same reason. Docker stacks a `NET_ADMIN` compose
   overlay for these runs; Daytona allows iptables natively.
3. **Server-side web tools** run at the provider, outside the sandbox. The
   LiteLLM pre-call hook merges the rules into Anthropic `web_search_*` /
   `web_fetch_*` `blocked_domains` (domains and `domain/path` prefixes) and
   strips OpenAI hosted `web_search` tools (allowlist-only filter). Harness
   knobs cover the rest: Codex gets `-c tools.web_search=false`, Gemini
   excludes `google_web_search` and `web_fetch`. Client-side fetchers
   (OpenCode/MiMo `webfetch`, OpenHands browsing, Claude Code `WebFetch`) stay
   enabled because their traffic goes through layer 1.

Blocked requests answer **`404 Not Found`**, so a research agent cannot tell a
hidden page from a missing one. The rule list is deliberately kept out of the
agent process env (`BENCHFLOW_EGRESS_BLOCKED_URLS` reaches only the model
proxy). Before the first prompt a self-check runs **as the sandbox user**: the
first blocked host must answer 404 through the proxy and direct egress must be
rejected; a failed check aborts the rollout rather than running it open.

### Batch safety

`bench eval run` resolves every selected task's network posture under the
run's overlay and backend before the first rollout starts; a `--block-url`
against a task that declares `no-network`/`allowlist`, or a blocklist task on
a backend that cannot enforce it, fails the whole batch up front naming the
offending tasks. Resuming a job whose completed tasks recorded a different
`network_policy` (open vs. blocklisted, or a different list) is refused, the
same way an agent mismatch is — those scores belong to different experiments.

### Auditing a run

- `config.json` carries a `network_policy` block (`mode`, `blocked_urls`,
  `tls_inspection_hosts`, `blocked_status`).
- `agent/egress.jsonl` (downloaded from the root-only log at disconnect) lists
  every `allow` / `block` decision with host, path, and matched rule, plus the
  `probe` self-check record — the evidence that the agent attempted (or never
  attempted) the hidden URLs.
- Server-side search really being off is verified from the request bodies in
  `trajectory/llm_trajectory.jsonl`, not from config files.

### Limits

- The blocklist hides **URLs**, not knowledge: search-result snippets from
  unblocked engines, citations in other papers, and the model's own training
  data can still reveal that a paper exists. Block the mirrors you care about
  (Semantic Scholar, OpenReview, alphaXiv, HF Papers, …) by host.
- Path rules are prefix matches: `arxiv.org/abs/2401.12345` also hides
  `…/abs/2401.123456` (and, usefully, `…/abs/2401.12345v2`). Matching runs on
  the canonical path an upstream server would route — percent-decoded,
  `..`/`.`/`//` collapsed, case-folded — and on both the URL host and the
  `Host` header, so encoding tricks or an IP-literal URL with a spoofed
  `Host` do not slip past. Percent-decoding is repeated until stable, so a
  double-encoded separator cannot reach an upstream that decodes twice.
- Only the agent phase is covered; `verifier.network_mode = "blocklist"` is
  rejected before launch. Modal, Apple Container, and AgentCore cannot run
  the root proxy + UID firewall and refuse blocklist tasks.
- The proxy needs `python3` in the task image (and `openssl` for TLS
  inspection; it is apt/dnf/apk-installed on demand).
- The proxy runs as root, outside the agent-UID firewall, so it vets every
  resolved upstream address and answers `403` for loopback, link-local (cloud
  instance metadata such as `169.254.169.254`), unspecified, multicast,
  reserved, and private ranges. The only private addresses it will reach are
  the container's own directly-connected subnets (from `/proc/net/route`),
  which is where compose side-services live; other RFC1918 space — the host
  LAN behind the bridge, other projects' networks — is refused.
- Session-factory agents run in-process on the host, outside the sandbox
  proxy and firewall; a blocklist run refuses them before connecting.
- TLS inspection speaks HTTP/1.1 only (ALPN advertises just `http/1.1`, so
  HTTP/2-capable clients negotiate down).
- The proxy handles at most 256 connections at once; a burst of concurrent
  fetches queues in the listen backlog rather than exhausting threads or file
  descriptors.
- When the primary agent is the oracle, the oracle itself is exempt from the
  blocklist, but the container is still provisioned for it (docker
  `NET_ADMIN`, sandbox-local model proxy) so role agents connecting later are
  covered.

## Related

- [`progressive-disclosure.md`](./progressive-disclosure.md) — soft-verify (the relaxed hardening used between rounds in multi-round trials).
- [`task-authoring.md`](./task-authoring.md) — the task config schema, including the `[verifier.hardening]` opt-outs.
