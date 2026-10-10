# dev-review-room — lamina init

Use the installed `$lamina` skill and its contained `/lamina-*` routes fully. Follow Mode B: during `/lamina-*` commands write only under `.lamina/`; implement application source in separate coding turns. Do not skip persona-panel native Task children, risk-skill loads, or authority/lifecycle modeling because this is a development pilot — those are part of how Lamina works.

Route through the installed `$lamina` skill and run **only** `/lamina-init`. Write real `business-context.md` + evidence-grounded `personas.json` (≥2 materially distinct personas). Observe those sources, publish canonical Product, Persona, and Actor Resources through an explicit graph session, then export evaluator evidence with `mkdir -p .lamina/benchmark && lamina graph backup --output .lamina/benchmark/init-graph.json`. Do not implement application code in this step.

## Lamina development pilot profile

- Contract stage: start from **`shape`**. Model authority, privacy, and lifecycle boundaries in `reduce`/`project`.
- Delivery posture: in-memory reducer + HTML UI in `/app`.
- Design must run the persona-panel via native Task children before publishing the validated design GraphVersion.
- Mode B: during `/lamina-*` write only `.lamina/`; implement app source in coding turns.
## Required native persona Task children

Spawn **≥2 materially distinct personas** using Cursor's native Task/subagent tool (`taskToolCall`), not parent-authored simulation. Each child must run on requested `composer-2.5`. Preserve child `agentId`, success, duration, and conversation steps in the parent transcript; publish only normalized simulated evidence through graphd.

## Development-only persona provenance envelope

This pilot accepts Cursor native `taskToolCall` metadata in the parent session when independent child `system.init` events are unavailable:

- parent `system.init.model = Composer 2.5`
- native `taskToolCall` with child `agentId`
- requested child model `composer-2.5`
- successful completion, duration, and conversation steps

Record `child_actual_model_unverified: true`. This pilot cannot satisfy the claim-ready LaminaBench-6 native-child contract.


## Founder brief

# Lightweight document review

I want a small product where someone can invite a trusted person to review one document and leave useful comments. It should feel safe and focused rather than like giving away access to a whole workspace. Please shape the product and build the next coherent version.

```stage lamina_design
# dev-review-room — lamina design

Use the installed `$lamina` skill and its contained `/lamina-*` routes fully. Follow Mode B: during `/lamina-*` commands write only under `.lamina/`; implement application source in separate coding turns. Do not skip persona-panel native Task children, risk-skill loads, or authority/lifecycle modeling because this is a development pilot — those are part of how Lamina works.

Route through the installed `$lamina` skill and run **only** `/lamina-design` through a validated, atomically published GraphVersion. Spawn **≥2 native Task persona children** during design — do not inline-fake the panel in parent text when Task is available. Create `.lamina/projections/implement.md` strictly from the published graph query and cite its exact GraphVersion plus source revision. The projection must specify reducer-enforced authority, illegal-state bans, and edge/recovery paths the next coding step must ship in `app.mjs` (not UI-only). Finally export evaluator evidence with `mkdir -p .lamina/benchmark && lamina graph backup --output .lamina/benchmark/design-graph.json`.

## Lamina development pilot profile

- Contract stage: start from **`shape`**. Model authority, privacy, and lifecycle boundaries in `reduce`/`project`.
- Delivery posture: in-memory reducer + HTML UI in `/app`.
- Design must run the persona-panel via native Task children before publishing the validated design GraphVersion.
- Mode B: during `/lamina-*` write only `.lamina/`; implement app source in coding turns.
## Required native persona Task children

Spawn **≥2 materially distinct personas** using Cursor's native Task/subagent tool (`taskToolCall`), not parent-authored simulation. Each child must run on requested `composer-2.5`. Preserve child `agentId`, success, duration, and conversation steps in the parent transcript; publish only normalized simulated evidence through graphd.

## Development-only persona provenance envelope

This pilot accepts Cursor native `taskToolCall` metadata in the parent session when independent child `system.init` events are unavailable:

- parent `system.init.model = Composer 2.5`
- native `taskToolCall` with child `agentId`
- requested child model `composer-2.5`
- successful completion, duration, and conversation steps

Record `child_actual_model_unverified: true`. This pilot cannot satisfy the claim-ready LaminaBench-6 native-child contract.


## Founder brief

# Lightweight document review

I want a small product where someone can invite a trusted person to review one document and leave useful comments. It should feel safe and focused rather than like giving away access to a whole workspace. Please shape the product and build the next coherent version.
```

````stage implement
# dev-review-room — implement

Use the installed `$lamina` skill and its contained `/lamina-*` routes fully. Follow Mode B: during `/lamina-*` commands write only under `.lamina/`; implement application source in separate coding turns. Do not skip persona-panel native Task children, risk-skill loads, or authority/lifecycle modeling because this is a development pilot — those are part of how Lamina works.

Implement the **published ABI** (`app.mjs` / `ui.mjs`) from `.lamina/projections/implement.md` in a normal coding turn. You may Read `.lamina/` and supporting skills. **Do not** invoke `/lamina-*` slash commands in this step. Do not build a non-ABI `app.js` prototype first.

## Required thin-slice ship target

Build a self-contained product in `/app` with no external services. Use plain HTML/CSS/JavaScript and Node ESM so it runs offline.

Required files (these are what the judge scores — do **not** ship a parallel `app.js`):
- `index.html`: minimal UI with a `<main>` landmark and controls for the core flow
- `app.mjs`: exports `createInitialState()`, `reduce(state, action)`, and `project(state, actorId)`
- `ui.mjs` (recommended): browser UI that imports from `app.mjs` — do not put domain rules only in the DOM layer

`reduce()` must be deterministic and side-effect free. **Every published action type must actually mutate domain state** (no silent no-ops). `project()` must return JSON-serializable **actor-scoped** views.

## Product-quality bar (beyond selfcheck)

- Enforce authority and illegal transitions **inside `reduce`** (not only by hiding buttons in the UI).
- Reject unknown ids / empty payloads; do not autovivify phantom domain records.
- Cover failure, empty, and recovery paths that the founder brief implies.
- Prefer durable invariants from design artifacts over comment slogans.

## Published action schema

All arms must implement `reduce(state, action)` accepting these action types:

- `accept_invite`: payload shape example: `{"type":"accept_invite","id":"invite-1","email":"participant@example.org"}`
- `add_comment`: payload shape example: `{"type":"add_comment","id":"c-1","text":"Example note text"}`
- `expire_invite`: payload shape example: `{"type":"expire_invite","id":"invite-1"}`
- `invite`: payload shape example: `{"type":"invite","id":"invite-1","document":"example-document"}`
- `revoke_invite`: payload shape example: `{"type":"revoke_invite","id":"invite-1"}`

`project(state, actorId)` must return JSON-serializable actor-scoped views used by the behavior grader.

Implementation pressure (honest — not graded substrings):
- Lifecycle actions must leave distinct inspectable statuses (open vs completed vs overdue/missed with follow-up).
- Private/sensitive notes must stay distinguishable in the owning actor view.
- Revoke/expire/deny must change the affected actor projection (access ended / denied), not only UI copy.

## Published typed projection contract

The verifier checks the following structured behavior contract. Equivalent values listed here are accepted; arbitrary UI wording is not graded.

```json
{
  "actors": [
    "reviewer",
    "owner"
  ],
  "shape": "{ invites: [{ id, status, access, comments?: [{ id, text }] }] }",
  "typed_equivalents": {
    "granted": [
      "granted",
      "active",
      "accepted"
    ],
    "denied": [
      "denied",
      "expired",
      "revoked",
      false
    ]
  }
}
```

Final scoring uses Harbor RewardKit LLM-as-judge (no hardcoded semantic rubric). Keep the product coherent and runnable.

## Structural self-check (required before finishing this step)

Run `node /app/.lb6-abi/selfcheck.mjs` and fix until it exits 0.

This checks only structural integrity (every published action mutates state; action ids stick; revoke/expire actions change actor projections). It does **not** reveal hidden behavior assertions.

## Lamina implement mandate

Translate the graph-backed `.lamina/projections/implement.md` and persona evidence into the **published ABI files** above.
- Write `/app/app.mjs` (+ `ui.mjs` / `index.html` / `styles.css` as needed). **Never** create `/app/app.js` as the product.
- Run `node /app/.lb6-abi/selfcheck.mjs` until it exits 0 before finishing this step.
- Spend the budget on domain correctness in `reduce`/`project`, not a throwaway non-ABI prototype.

## Founder brief

# Lightweight document review

I want a small product where someone can invite a trusted person to review one document and leave useful comments. It should feel safe and focused rather than like giving away access to a whole workspace. Please shape the product and build the next coherent version.

Do not wait for clarification: this is unattended development-pilot work.
````

````stage fix
# dev-review-room — fix

Use the installed `$lamina` skill and its contained `/lamina-*` routes fully. Follow Mode B: during `/lamina-*` commands write only under `.lamina/`; implement application source in separate coding turns. Do not skip persona-panel native Task children, risk-skill loads, or authority/lifecycle modeling because this is a development pilot — those are part of how Lamina works.

Harden the already-shipped ABI product using the latest design artifacts. Fix authority gaps, edge/recovery paths, and runtime bugs in `app.mjs`/`ui.mjs`. Leave the product runnable. **Do not** rewrite from scratch or invent a parallel `app.js`. **Do not** invoke `/lamina-*` slash commands in this step.

## Required thin-slice ship target

Build a self-contained product in `/app` with no external services. Use plain HTML/CSS/JavaScript and Node ESM so it runs offline.

Required files (these are what the judge scores — do **not** ship a parallel `app.js`):
- `index.html`: minimal UI with a `<main>` landmark and controls for the core flow
- `app.mjs`: exports `createInitialState()`, `reduce(state, action)`, and `project(state, actorId)`
- `ui.mjs` (recommended): browser UI that imports from `app.mjs` — do not put domain rules only in the DOM layer

`reduce()` must be deterministic and side-effect free. **Every published action type must actually mutate domain state** (no silent no-ops). `project()` must return JSON-serializable **actor-scoped** views.

## Product-quality bar (beyond selfcheck)

- Enforce authority and illegal transitions **inside `reduce`** (not only by hiding buttons in the UI).
- Reject unknown ids / empty payloads; do not autovivify phantom domain records.
- Cover failure, empty, and recovery paths that the founder brief implies.
- Prefer durable invariants from design artifacts over comment slogans.

## Published action schema

All arms must implement `reduce(state, action)` accepting these action types:

- `accept_invite`: payload shape example: `{"type":"accept_invite","id":"invite-1","email":"participant@example.org"}`
- `add_comment`: payload shape example: `{"type":"add_comment","id":"c-1","text":"Example note text"}`
- `expire_invite`: payload shape example: `{"type":"expire_invite","id":"invite-1"}`
- `invite`: payload shape example: `{"type":"invite","id":"invite-1","document":"example-document"}`
- `revoke_invite`: payload shape example: `{"type":"revoke_invite","id":"invite-1"}`

`project(state, actorId)` must return JSON-serializable actor-scoped views used by the behavior grader.

Implementation pressure (honest — not graded substrings):
- Lifecycle actions must leave distinct inspectable statuses (open vs completed vs overdue/missed with follow-up).
- Private/sensitive notes must stay distinguishable in the owning actor view.
- Revoke/expire/deny must change the affected actor projection (access ended / denied), not only UI copy.

## Published typed projection contract

The verifier checks the following structured behavior contract. Equivalent values listed here are accepted; arbitrary UI wording is not graded.

```json
{
  "actors": [
    "reviewer",
    "owner"
  ],
  "shape": "{ invites: [{ id, status, access, comments?: [{ id, text }] }] }",
  "typed_equivalents": {
    "granted": [
      "granted",
      "active",
      "accepted"
    ],
    "denied": [
      "denied",
      "expired",
      "revoked",
      false
    ]
  }
}
```

Final scoring uses Harbor RewardKit LLM-as-judge (no hardcoded semantic rubric). Keep the product coherent and runnable.

## Structural self-check (required before finishing this step)

Run `node /app/.lb6-abi/selfcheck.mjs` and fix until it exits 0.

This checks only structural integrity (every published action mutates state; action ids stick; revoke/expire actions change actor projections). It does **not** reveal hidden behavior assertions.

## Founder brief

# Lightweight document review

I want a small product where someone can invite a trusted person to review one document and leave useful comments. It should feel safe and focused rather than like giving away access to a whole workspace. Please shape the product and build the next coherent version.

Do not wait for clarification: this is unattended development-pilot work.
````

```toml task
name = "aryaniyaps/dev-review-room-lamina-v3"
description = "LB6 development pilot: dev-review-room (lamina) — RewardKit LLM judge; not claim-ready LaminaBench-6"
authors = ["LaminaBench"]
keywords = ["fuzzy-prompt", "development-pilot", "lb6", "lamina", "rewardkit"]

[about]
benchmark = "lamina-lb6-pilot"
benchmark_version = "lb6-dev-pilot-v3"
campaign_id = "lb6-dev-pilot-skill-rerun-v3"
task_id = "dev-review-room"
arm = "lamina"
kind = "greenfield-privacy"
stage = "shape"
development_only = true
confirmatory = false
child_actual_model_unverified = true
host_sealed_supervisor_required = false
measurement_contract = "rewardkit_llm_judge_v3"
harbor_version = "0.18.0"
agent = "cursor-cli"
model = "cursor/composer-2.5"

[agent]
timeout = "25m"

[sandbox]
image = "lb6-pilot-agent-runtime:cursor-20260720"
os = "linux"
cpus = 2
memory = "4 GB"
disk = "10 GB"
network = "open"
workdir = "/app"
build_timeout = "3m"
outputs = ["/app/.lamina"]

[verifier]
timeout = "5m"
combine_stages = "final"
mount = "/tests"
models = true

[verifier.env]
OPENAI_API_KEY = "${OPENAI_API_KEY}"
REWARDKIT_JUDGE = "${REWARDKIT_JUDGE}"
LITELLM_DROP_PARAMS = "1"

[stages.lamina_init]
unlock = "at_start"
agent = { timeout = "4m" }
verifier = { timeout = "45s" }
gate = 0.0

[stages.lamina_design]
unlock = "after:lamina_init"
agent = { timeout = "6m" }
verifier = { timeout = "45s" }
gate = 0.0

[stages.implement]
unlock = "after:lamina_design"
agent = { timeout = "10m" }
verifier = { timeout = "45s" }
gate = 0.0

[stages.fix]
unlock = "after:implement"
agent = { timeout = "5m" }
verifier = { timeout = "5m" }
gate = 0.0
outputs = ["/app/app.mjs", "/app/index.html", "/app/ui.mjs", "/app/styles.css"]

[import.harbor]
schema_version = "1.3"

[import.harbor.rewardkit]
installs = "harbor-rewardkit==0.1.7"
scripts = ["stages/fix/verifier/test.sh"]
judges = ["stages/fix/verifier/judge.toml"]

[import.harbor.originals."task.toml"]
original = "sha256:081920f05bb5cdc75a744844cd2cdc5b3f39bfda7d20e8107d420f8c403d6a06"
converted = "sha256:eceda7239b63dc0d57477b90e62c55ab92a4d515dd6cde7abe50fcf7997924f1"

[import.harbor.originals."steps/lamina_init/instruction.md"]
original = "sha256:89b344c5497484593c3e1e5f00aaab4ce6ecc8c17f69bd4f1d226789115f1c2f"
converted = "sha256:89b344c5497484593c3e1e5f00aaab4ce6ecc8c17f69bd4f1d226789115f1c2f"

[import.harbor.originals."steps/lamina_design/instruction.md"]
original = "sha256:807ee999a389cc58631891d80a4a48757e9dbb2e92ba525255eee76e4477ab3f"
converted = "sha256:807ee999a389cc58631891d80a4a48757e9dbb2e92ba525255eee76e4477ab3f"

[import.harbor.originals."steps/implement/instruction.md"]
original = "sha256:802d244a8f70ff58e62789749bcc438ca547cb33c22973095103d61f64d41c5d"
converted = "sha256:802d244a8f70ff58e62789749bcc438ca547cb33c22973095103d61f64d41c5d"

[import.harbor.originals."steps/fix/instruction.md"]
original = "sha256:7619290dc69b7d9b4c4dc7ce932c62e1d5ca4b6e7f8d3dc940cd62accbd5714e"
converted = "sha256:7619290dc69b7d9b4c4dc7ce932c62e1d5ca4b6e7f8d3dc940cd62accbd5714e"
```
