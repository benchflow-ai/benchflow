# The task.md standard

Status: draft, 2026-06-09.

This page is the concise spec for a BenchFlow `task.md` (and the equivalent
split `task.toml`) frontmatter. It is the reference target for the
"root-key rule" and the "Agent And Runtime Policy" section linked from
[`skillsbench-scope/`](./skillsbench-scope/README.md). The shapes here are
grounded in `src/benchflow/task/config.py` (the Pydantic models) and
`src/benchflow/task/verifier.py` / `verifier_document.py` (the verifier).

## Document body

A `task.md` is YAML frontmatter delimited by `---`, immediately followed by the
agent instruction as the Markdown body. No heading precedes the instruction: the
text after the closing `---` *is* the prompt. For multi-turn tasks an optional
`## role:<name>` / `## scene:<name>` / `## user-persona` section may follow the
instruction; a literal `## prompt` / `## role:` line inside the instruction is
escaped on write and restored on read, so it round-trips verbatim. A legacy
`## prompt` heading is still accepted on read for backward compatibility.

## Root keys

A task's frontmatter is a closed set of root keys, modeled by `TaskConfig` in
`src/benchflow/task/config.py`. The allowed root keys are:

| Root key | Meaning |
|---|---|
| `schema_version` | Schema version string, `MAJOR.MINOR` (default `1.3`). Only known majors are accepted — see below. |
| `task` | Package metadata (`PackageInfo`): `name` in `org/name` form, `description`, `authors`, `keywords`. |
| `metadata` | Free-form `dict` for task-level annotations (categories, keywords, etc.). The escape hatch for everything that is not a first-class key. |
| `verifier` | Verifier ($V$) configuration — see "Agent And Runtime Policy" and "Reward files". |
| `agent` | Agent harness ($H$) configuration (timeout, user, network policy). |
| `environment` | Sandbox configuration. Stored internally as `sandbox`; `environment` is the on-disk key. |
| `oracle` | Reference solution config. Stored internally as `solution`; `oracle` is the on-disk key. |
| `source` | Optional provenance string. |
| `reward` | Legacy task-level reward expression metadata (`dict`). |
| `multi_step_reward_strategy` | How to fold per-step verifier rewards into one rollout reward (`mean` or `final`). |
| `steps` | Optional list of `StepConfig` for Harbor-style multi-step tasks. |
| `artifacts` | Paths copied out of the environment after verification. |

### The root-key rule: unknown keys are rejected

`TaskConfig` (and every nested section model) sets
`model_config = ConfigDict(extra="forbid")`. Any root key not in the table
above is a hard validation error, not a silently ignored extra. This is
deliberate: `task.md` / `task.toml` must not become a lossy subset of the
upstream task schema. To attach extra task-level information, nest it under
`metadata:` rather than inventing a new root key.

### Key aliases

A few keys carry a stable on-disk name that differs from the internal model
field, for backward compatibility:

| On-disk key | Internal field | Notes |
|---|---|---|
| `environment` | `sandbox` | `TaskConfig.sandbox` is loaded from the `environment` key; `config.environment` is a read-only alias back to `sandbox`. |
| `oracle` | `solution` | Native tasks use `oracle`; the legacy Harbor/Pier `solution` key is still accepted. Supplying **both** `oracle` and `solution` is an error. |
| `docker_image` | `docker_image` | `SandboxConfig.docker_image` also accepts the legacy `image` key (`AliasChoices("docker_image", "image")`). |

### Schema version

`schema_version` is `MAJOR.MINOR`. Only majors the loader understands
(`_SUPPORTED_SCHEMA_MAJORS = {1}`) are accepted; an unknown major or a
non-numeric version raises rather than being carried through. The minor part is
permissive. The legacy `version` key is migrated to `schema_version` on load.

## Agent And Runtime Policy

### Network policy

Network access is controlled by `network_mode`, an enum (`NetworkMode`) with
three values:

| `network_mode` | Effect |
|---|---|
| `no-network` | No network access. |
| `public` | Unrestricted egress (the `SandboxConfig` default). |
| `allowlist` | Egress only to the hostnames in `allowed_hosts`. |

`allowed_hosts` is a list of bare hostnames (no scheme, port, or path; entries
are normalized to lowercase and validated against a DNS-label pattern). The
rules enforced by `_validate_network_policy_fields`:

- `network_mode='allowlist'` **requires** a non-empty `allowed_hosts`.
- `allowed_hosts` is **only** valid with `network_mode='allowlist'`; supplying
  it under any other mode is an error.

`network_mode` is settable on the sandbox (`environment`), and may be
**overridden** per-component on `agent` and on `verifier` (each carries its own
optional `network_mode` + `allowed_hosts`).

#### Deprecated `allow_internet`

`SandboxConfig.allow_internet` is a deprecated boolean kept for compatibility.
**`network_mode` is authoritative.** Reconciliation
(`handle_deprecated_fields_and_network_policy`):

- If `network_mode` was set explicitly and `allow_internet=False` contradicts
  it (i.e. `network_mode` is not `no-network`), that is a **hard error** — drop
  `allow_internet` and rely on `network_mode` (use `network_mode='no-network'`
  to disable networking).
- If `network_mode` was **not** given and `allow_internet=False`, it downgrades
  to `network_mode='no-network'`.
- `network_mode='no-network'` always forces `allow_internet=False`.

New tasks should set `network_mode` and never write `allow_internet`.

### Verifier strategies

The verifier ($V$) maps a completed rollout to a reward. The strategy is
selected by `verifier.type` in `config.py` (`test-script` | `llm-judge`) and,
for `verifier.md`-driven tasks, by the richer `VerifierStrategyType` in
`verifier_document.py`:

| Strategy | What it does |
|---|---|
| `test-script` (`script`) | **Default.** Uploads the task's `tests/` directory into the sandbox, runs `test.sh`, and parses `reward.json` / `reward.txt`. |
| `llm-judge` | Downloads the agent's deliverables and scores them against a rubric using an LLM judge; writes the aggregate reward to `reward.json`. |
| `reward-kit` | Runs a reward-kit manifest-driven verifier. |
| `agent-judge` | An agentic judge that reads declared input artifacts to produce a reward. |
| `ors-episode` | Scores an Open-Rollout-Spec episode from declared reward evidence. |

A `verifier.md` lists named strategies under `verifier.strategies` and picks one
via `verifier.default_strategy`; selecting a strategy `type` the loader cannot
execute raises `UnsupportedVerifierStrategyError`.

## Reward files

The `test-script` verifier reads two possible reward files written into the
rollout output directory:

- `reward.txt` — a single finite scalar in `[0.0, 1.0]`.
- `reward.json` — a JSON object mapping reward names to numeric values.

**`reward.json` is read in preference to `reward.txt`.** In the verify flow,
if `reward.json` exists it is parsed first (`_parse_reward_json_with_text_compat`);
`reward.txt` is consulted only when `reward.json` is absent.

**Cross-check when both exist.** If both `reward.json` and `reward.txt` are
present, the scalar from `reward.txt` is compared against the JSON's `reward`
(directly, or after applying the aggregate policy when JSON declares one) using
`math.isclose(..., abs_tol=1e-9)`. If the two disagree beyond that tolerance the
verifier **raises** `VerifierOutputParseError` rather than silently preferring
one — the task author must make the two files agree.
