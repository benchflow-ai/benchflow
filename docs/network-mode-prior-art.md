# Network access: design and prior art

BenchFlow controls a task's outbound network with `network_mode`
(`no-network` / `public` / `allowlist`) plus `allowed_hosts` and
`allow_model_endpoint`. This page records the **design model**, what is
enforced today, and **credits the prior-art platforms** whose network-access
designs informed it. For the authoring reference see
[Authoring native task.md tasks](./task-authoring-task-md.md#network-policy); for
enforcement internals see [Sandbox hardening](./sandbox-hardening.md).

The enforcing substrate is the owned egress proxy from
PR [#785](https://github.com/benchflow-ai/benchflow/pull/785)
(`feat/network-mode-enforcement`, ENG-219): an `internal: true` Docker network
plus a stdlib CONNECT/forward proxy sidecar. On the `daytona` sandbox the same
policy is enforced through the platform's native IPv4 `network_allow_list`
(ENG-264). Everything below distinguishes **what BenchFlow owns** from
**patterns credited to other platforms**.

## What BenchFlow enforces today

- **`no-network` / `public`** — every sandbox.
- **`allowlist`** — enforced on the `docker` and `daytona` sandboxes. On
  `docker`, the container joins an `internal: true` (no-egress) network and all
  HTTP(S) traffic is forced through the proxy sidecar, which forwards only to
  `allowed_hosts`. A host matches **exactly or as a parent domain** (`example.com`
  matches `api.example.com`). A single **leading-label wildcard** `*.example.com`
  matches subdomains at any depth but **not** the bare apex (Harbor / nginx
  semantics); mid- or trailing-wildcards are rejected at parse time.
  Non-allowlisted hosts, raw-IP connections, and proxy-ignoring tools have no
  route off-box (default deny). On `daytona` the same intent is enforced through
  the platform's native IPv4 allow list (`network_allow_list`): the hostname
  allowlist is resolved to `/32` CIDRs at lockdown and each host is pinned in the
  sandbox `/etc/hosts` (so it resolves without DNS egress — the resolvers are not
  allowlisted — and without IP-rotation drift). Daytona enforces **when the
  policy is faithfully expressible** as IPv4 (exact hosts resolving to ≤10 IPs)
  and otherwise **fails closed with a precise reason**: wildcard allowlists are
  rejected at preflight (an IPv4 list can't express `*.x`) and over-cap
  allowlists (>10 IPs, e.g. CDN-fronted hosts) fail closed at lockdown pointing
  to `docker`. On sandboxes with no per-host egress control (`modal`) an
  `allowlist` task is **rejected at preflight** rather than run unrestricted.
  (ENG-264)
- **Model lane** — under a restrictive policy on `docker`, a single always-allow
  lane to the host-side model proxy stays open, so an agent run reaches the model
  without opening the sandbox to the public internet (a `no-network` run becomes
  model-only egress). Governed by `allow_model_endpoint` (default `true`; set
  `false` for a fully hermetic, no-model run). This replaced an earlier
  blanket lift-to-`public` for web-disabled docker runs.

## Behavioral modes

Four behavioral postures, plus orthogonal modifiers. "own" = BenchFlow's own
design/mechanism; "credit" = pattern adopted from the named platform.

| Mode | How it is implemented | BenchFlow status | Credit / source |
|---|---|---|---|
| **no-network** | Docker `network_mode: none` compose overlay | have | own (universal pattern) |
| **public** (default) | No restriction | have | own (field default everywhere) |
| **hostname allowlist** (exact / subdomain) | docker: internal net + CONNECT egress-proxy sidecar; daytona: resolve to IPv4 `/32`s + `/etc/hosts` pin | have (#785, docker + daytona) | own proxy; taxonomy ≡ Harbor, AISI Inspect |
| **wildcard host** (`*.example.com`, multi-level) | Glob match in proxy + validator accepts leading `*.` | have (#785) | credit: Harbor, Modal |
| **CIDR allowlist** | Daytona native `network_allow_list` (IPv4 `/32`s resolved from the host allowlist) | have (#785, daytona, ENG-264) | own resolver; mechanism credit: Modal, Daytona |
| **port-scoped egress** (`host:port`) | Rule type carries port; proxy already tunnels any CONNECT port | mechanism ok, rule type missing | credit: Modal |
| **offline mirror** (pkg / content) | Pre-baked layered images / pinned mirror | gap | credit: SWE-bench |
| **temporal package pin** | pip-index proxy filtering by release date | gap | credit: SWE-bench-Live |
| **record / replay (+ fault inject)** | MITM proxy + CA injected into sandbox | gap (proxy is non-MITM by design) | credit: WAREX |
| **mocked tool layer** | Tools served from a local DB; no real net | task-authored, not a mode | credit: τ²-bench |
| **ingress / published ports** | Host-publish + PREROUTING, or internal shared net | no inbound axis today | credit: WebArena, Cybench |
| **always-on model lane** | Auto-allow the host model proxy; default-on, settable | have (#785, docker) | own |
| **per-scene / per-turn policy** | network field on Role/Turn → `Step.data` → switch at connect/execute | design (cheap add) | own (Harbor has per-phase/step) |
| **separate-verifier-env** | Distinct policy for the grading env | field exists, enforcement is sandbox-wide | credit: Harbor |

## Enforcement mechanisms

| Mechanism | Enforces | Who uses it | BenchFlow |
|---|---|---|---|
| Docker `network_mode: none` / internal network | no-network; default-deny base for allowlist | everyone (AISI, SWE-bench, …) | have |
| **Owned forward/CONNECT egress proxy sidecar** | hostname allowlist, vendor-neutral | **benchflow #785** | have / own |
| Cloud-platform egress allowlist (domain + CIDR) | hostname & CIDR allowlist | Modal, Daytona, Harbor backends | have (daytona, ENG-264) |
| CoreDNS `allowDomains` (k8s) | hostname allowlist via DNS | UK AISI Inspect (k8s sandbox) | credit |
| MITM proxy + CA injection | record/replay, fault injection, body inspection | WAREX | credit (deliberately not done) |
| Pre-baked layered images / offline mirror | hermetic — no runtime egress needed | SWE-bench, DeepSWE | credit |
| Temporal pip-index proxy | version pinning by date (not connectivity) | SWE-bench-Live | credit |
| Host-published ports + PREROUTING | inbound to self-hosted target apps | WebArena / VisualWebArena | credit (no inbound axis today) |
| Shared internal Docker network | agent ↔ target topology (CTF) | Cybench | credit |
| Browser-SDK `allowed_domains` | app-level (browser) egress scope | browser-use / BenchFlow CUA | have (browser app) |
| Mocked tools over a local DB | no network in the tool layer | τ²-bench | credit (task-authored) |

## Credits

Primary sources for each credited platform. Mechanism descriptions are stated as
verified against these sources; see the caveats inline.

- **Harbor** — agent-evaluation / RL framework (the `harbor-framework` org, not
  the container registry). Per-host `network_mode = "allowlist"` + `allowed_hosts`
  with leading-wildcard subdomain matching (nginx-style, multi-level), per-phase
  overrides (`[environment]`/`[agent]`/`[verifier]`/`[steps.*]`), and a
  shared-by-default vs. separate verifier environment. BenchFlow's `allowlist`
  taxonomy and wildcard semantics follow Harbor's.
  <https://github.com/harbor-framework/harbor> · docs <https://www.harborframework.com/docs/tasks>.
  Introduced in [#1455](https://github.com/harbor-framework/harbor/pull/1455)
  (network mode + optional allowlist) and
  [#1840](https://github.com/harbor-framework/harbor/pull/1840) (wildcard host
  support); [#1854](https://github.com/harbor-framework/harbor/pull/1854) later
  *clarifies* that wildcards match multi-level subdomains.
- **Modal** — Sandbox outbound controls: `block_network`,
  `outbound_cidr_allowlist` (CIDR ranges), and `outbound_domain_allowlist`
  (domains). Source of the port-scoping pattern BenchFlow does not yet implement;
  the CIDR-allowlist pattern is now enforced on the `daytona` sandbox via its
  native `network_allow_list` (see Daytona below, ENG-264).
  <https://modal.com/docs/guide/sandbox-networking>.
- **Daytona** — Sandbox `network_block_all` and `network_allow_list`
  (comma-separated IPv4 CIDRs, max 10) at creation, plus a runtime
  `update_network_settings`. BenchFlow uses these directly to enforce
  `allowlist` / `no-network` on the `daytona` sandbox — resolving the hostname
  allowlist to `/32`s and pinning `/etc/hosts`, since the IPv4 list cannot carry
  hostnames (verified against SDK `daytona==0.184.0`). Allow/block-list format +
  validation: [#4124](https://github.com/daytonaio/daytona/pull/4124); runtime
  `update_network_settings`: [#4604](https://github.com/daytonaio/daytona/pull/4604).
  Docs: <https://github.com/daytonaio/daytona/blob/main/apps/docs/src/content/docs/en/network-limits.mdx>.
- **UK AISI Inspect** (k8s sandbox) — hostname allowlisting via `allowDomains`
  (FQDN list) enforced by a per-Pod CoreDNS sidecar, plus `allowCIDR`.
  <https://github.com/UKGovernmentBEIS/inspect_k8s_sandbox> · docs
  <https://k8s-sandbox.aisi.org.uk/security/network-access/>. *(The "N0/N1/N2"
  network-level shorthand sometimes used in BenchFlow design notes is **our own**
  label, not Inspect nomenclature.)*
- **WAREX** — routes web-agent traffic through mitmproxy with split-TLS (its CA
  is trusted inside the sandbox) to decrypt and **inject faults** (delays,
  4xx/5xx, injected JS/overlays). BenchFlow deliberately keeps its proxy
  **non-MITM**; WAREX is the reference for the record/replay + fault-injection
  axis we do not implement. <https://arxiv.org/abs/2510.03285>.
- **SWE-bench** — per-task hermetic execution via three layered Docker images
  (base → environment → instance), removing runtime network dependence.
  <https://github.com/SWE-bench/SWE-bench> · <https://arxiv.org/abs/2310.06770>.
- **SWE-bench-Live** — "time-machine" pip-index proxy that serves only package
  versions released no later than the instance's base-commit timestamp.
  <https://github.com/microsoft/SWE-bench-Live> · <https://arxiv.org/abs/2505.23419>.
- **WebArena / VisualWebArena** — self-host target web apps as Docker containers
  with host-published ports, resettable to an initial state; VisualWebArena adds
  a self-hosted Wikipedia knowledge base. The reference for an inbound/ingress
  axis BenchFlow does not model today.
  <https://github.com/web-arena-x/webarena> ·
  <https://github.com/web-arena-x/visualwebarena>.
- **Cybench** — CTF benchmark where a Kali agent container reaches one or more
  task-server containers over a shared Docker network.
  <https://github.com/andyzorigin/cybench> · <https://arxiv.org/abs/2408.08926>.
- **τ²-bench** — tool-calling benchmark whose tools operate over a local JSON
  database (`db.json` per domain); grading compares final DB state to a goal
  state, with no real network in the tool layer.
  <https://github.com/sierra-research/tau2-bench> · τ-bench paper
  <https://arxiv.org/abs/2406.12045>.
- **browser-use** — browser-automation SDK whose `BrowserProfile.allowed_domains`
  scopes which domains the browser may visit (supports `*.example.com` patterns).
  BenchFlow's CUA browser-app egress scope reuses this. Introduced in
  [#364](https://github.com/browser-use/browser-use/pull/364).
  <https://github.com/browser-use/browser-use>.

> **Scope note.** Per-tool network scoping and a 3-way
> agent / verifier / environment egress split are *not* standard in the field —
> treat them as BenchFlow-specific design choices, not parity gaps.
