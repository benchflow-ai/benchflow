# SkillsBench-scope compatibility

Status: draft, 2026-06-09. Scope: **SkillsBench only** (the 87 `tasks/` + 13
`tasks-extra/` = 100 packages on `benchflow-ai/skillsbench@main`).

This is a deliberately *minimal*, SkillsBench-only slice of the
[task.md standard](../task-standard.md). It exists to make every SkillsBench
task run under a **fixed, validated vocabulary** instead of the free-form
`[metadata]` + binary `allow_internet` flag the source benchmark ships — and to
replace that binary flag with three executable network modes.

Two artifacts back this doc:

- [`glossary.yaml`](./glossary.yaml) — the fixed controlled vocabulary (enriches Harbor).
- [`network-map.yaml`](./network-map.yaml) — generated per-task network mode + allowlist + keywords for all 100 tasks.

## Why a fixed glossary

SkillsBench tasks are authored as Harbor-style split packages (`task.toml` +
`instruction.md` + `environment/` + `solution/` + `tests/`). Harbor's `[metadata]`
is free-form, so the only thing keeping categories/keywords consistent is
SkillsBench's own `taxonomy.yaml` + `lint_taxonomy.py`. This branch promotes that
taxonomy into the task.md standard as a **closed, lint-enforced vocabulary**, kept
inside `metadata:` so no new root keys are added (per the standard's root-key rule).

The glossary has three tiers:

| Tier | What | Enforcement |
|---|---|---|
| **Controlled** | `category`, `task_type`, `modality`, `interface`, `skill_type`, `difficulty` | closed enums, lint-enforced (mirror `taxonomy.yaml`) |
| **network_class** | the enrichment this branch adds (below) | closed enum, maps to runtime |
| **Open** | 511 distinct domain `tags` + 228 `skills` | advisory; skills trigger via SKILL.md `description` |

### Skill format

Each task ships skills under `environment/skills/<name>/SKILL.md` in the
[Anthropic skill format](https://github.com/mattpocock/skills/blob/main/skills/productivity/grill-me/SKILL.md):
YAML frontmatter with just `name` + `description`, then free-form instructions
(no required headings). The **`description` is the trigger vocabulary** — its
keywords decide when the skill loads — which is why the glossary keeps skills as an
*open* tier: they are matched by description, not validated against a closed set.
Skill mounting is governed by `benchflow.agent_policy.skill_access`.

## Network modes (the core enrichment)

SkillsBench ships a single binary: `[environment] allow_internet = true|false`.
Going through all 100 tasks shows that binary is both **over-permissive and
occasionally wrong**:

- Of the **55** tasks with `allow_internet = true`, **48 contact no runtime host
  at all** — the `true` is only build-time `pip`/`apt`, which a pre-baked image
  removes.
- **Zero** tasks need open web search or arbitrary browsing (the 5 "search"-tagged
  tasks search *local* data or use a named API, not the web).
- Conversely **1** task (`find-topk-similiar-chemicals`) declares
  `allow_internet = false` but actually needs a data API — it would fail offline.

So the real requirement space is two classes, plus a third kept only for lossless
import of the source default:

| Mode | Meaning | Agent egress | Maps to (benchflow) | SkillsBench tasks |
|---|---|---|---|---|
| **`disabled`** | no task-env internet at runtime; data pre-baked | model-only (LLM still allowed) | `network_mode: no-network` + `agent_policy.search: disabled` + forbid `web_search`,`browser_fetch` | **92** |
| **`api`** | small enumerable allowlist; **no** web search | model + allowlist | `network_mode: allowlist` + `allowed_hosts` + forbid `web_search` | **8** |
| **`enabled-blacklist`** | full internet minus a denylist (SkillsBench source default) | open | `network_mode: public` + `blocked_hosts` | **0 required** |

`disabled` is the user's "fully disabled (still allow LLM)": the task environment
has no internet, but the agent harness still reaches its own model. `api` is
"enable api … but no websearch". `enabled-blacklist` is "fully enabled (blacklist,
like the source benchmark)" — retained so a raw SkillsBench import round-trips, but
**no task actually needs it**.

> Enrichment gap: benchflow's `NetworkMode` enum has `no-network | public |
> allowlist` but **no denylist**, so `enabled-blacklist` currently degrades to
> `public` (allow-all). A real denylist (`blocked_hosts`) is the one new runtime
> primitive this scope would add.

### The 8 `api` tasks and their allowlists

| Task | `api_subtype` | `allowed_hosts` |
|---|---|---|
| `citation-check` | data-api | `api.crossref.org`, `api.semanticscholar.org`, `api.datacite.org`, `eutils.ncbi.nlm.nih.gov`, `export.arxiv.org`, `doi.org` |
| `find-topk-similiar-chemicals` | data-api | `pubchem.ncbi.nlm.nih.gov` |
| `shock-analysis-supply` | data-api | `data.ecb.europa.eu` |
| `scheduling-email-assistant` | cloud-api | `www.googleapis.com`, `oauth2.googleapis.com` |
| `pedestrian-traffic-counting` | llm-provider | `generativelanguage.googleapis.com` |
| `video-tutorial-indexer` | llm-provider | `api.openai.com` |
| `diff-transformer_impl` | compute-backend | `modal.com` |
| `mhc-layer-impl` | compute-backend | `modal.com` |

`llm-provider` is a **distinct** external AI API used as a task *tool* (Gemini
vision, OpenAI Whisper) — not the agent's own reasoning model, which `disabled`
already permits. Tasks whose reference solution merely calls the agent's own model
family stay `disabled`.

## Correctness findings (filed against SkillsBench)

Going through every task surfaced source-benchmark defects this glossary catches:

1. **`find-topk-similiar-chemicals`** declares `allow_internet = false` but its
   solution queries `pubchem.ncbi.nlm.nih.gov` → fails offline. Should be `api`.
2. **6 `tasks-extra` packages use off-taxonomy categories** that
   `lint_taxonomy.py` would reject: `cobol-gl-batch-reconcile` (`programming`),
   `diff-transformer_impl` / `mhc-layer-impl` (`ml-implementation`),
   `gpu-cluster-online-scheduling` (`cloud-infrastructure`),
   `scheduling-email-assistant` (`Scheduling`),
   `speaker-diarization-subtitles` (`audio-visual`).
3. **48 tasks over-declare internet** (`allow_internet = true`, build-time only) —
   tightenable to `disabled` for a smaller attack surface and faster, hermetic runs.

## Mapping to the standard

Nothing here is a new root key. The glossary lives in `metadata:`; the network
class compiles to existing fields:

```yaml
# a disabled task (the common case)
metadata: {category: natural-science, task_type: [analysis], skill_type: [domain-procedure]}
environment: {network_mode: no-network}
benchflow:
  agent_policy: {search: disabled, forbidden_tools: [web_search, browser_fetch]}

# an api task (citation-check)
environment:
  network_mode: allowlist
  allowed_hosts: [api.crossref.org, api.semanticscholar.org, doi.org, eutils.ncbi.nlm.nih.gov]
benchflow:
  agent_policy: {forbidden_tools: [web_search]}
  metadata: {network_class: api, api_subtype: data-api}
```

## Reproduction

`network-map.yaml` is generated, not hand-edited. The extraction clones
`benchflow-ai/skillsbench`, parses every `task.toml`, scans `solution/` +
`environment/` (excluding skill-doc markdown) for runtime hosts, maps LLM clients
to provider hosts, and excludes build/CDN/doc hosts. Allowlists reflect what each
reference solution actually contacts at runtime.
