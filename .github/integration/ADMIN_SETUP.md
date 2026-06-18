# Integration CI — admin setup (protected env, branch protection, secrets)

One-time GitHub configuration the L0–L3 integration workflows depend on. None of
this is enforceable from a workflow file; a repo admin must set it in the GitHub
UI / API. Workflows:

| Level | File | Trigger | Required? |
|------|------|---------|-----------|
| L0 | `.github/workflows/test.yml` | push / PR / merge_group | yes (every PR) |
| L1 | `.github/workflows/integration-light.yml` | PR (opened/sync/reopen/ready) + dispatch | yes (every PR) |
| L2 | `.github/workflows/integration-scope.yml` | PR (labeled `integration:medium`, …) + dispatch | yes (real work only on label) |
| L3 | `.github/workflows/integration-final-review.yml` | `workflow_dispatch` only | gated by protected env approval |

## 1. Branch protection (required status checks)

On the protected `main` branch, mark these checks **required**:

- `test` (L0 unit/lint/type lane)
- L1 job check from `integration-light` — the `detect-scope` + `rollout-smoke`
  jobs. Because `detect-scope` reports a **green no-op** (`should_run=false`)
  when a diff needs no rollout, this check can be required **unconditionally**
  without blocking docs-only PRs.
- L2 job check from `integration-scope` — likewise its `detect-scope` is a cheap
  green no-op when the `integration:medium` label is absent, so it is safe to
  require unconditionally.

> Required checks must **always run** (no `on: paths` filter) or a PR that
> doesn't touch the filtered paths would wait forever on a check that never
> starts. That is why every level's first job is the secret-free `detect-scope`
> no-op rather than a path filter.

L3 is **not** a required status check. It is enforced by the protected
environment approval below, plus the `integration-final-review` check-run that
the L3 review job publishes on the PR head SHA.

## 2. Protected environments

Use the existing **`pypi-internal-preview`** environment for L1/L2/L3 — **no
separate protected environment is created** (per the agreed simplification, L3's
golden truth is the HuggingFace leaderboard `main` runs vs the latest benchflow
main, not a human approval gate).

### `pypi-internal-preview` (L1 + L2 + L3)

- Holds the provider/Daytona/reviewer secrets (see §3).
- No required reviewers — L1/L2 run automatically on internal PRs; L3 is a manual
  `workflow_dispatch`.
- L1 attaches **only low-value** provider keys (DEEPSEEK / GEMINI /
  GITHUB_MODELS). L2/L3 attach **all** keys per the agreed Q6 residual. The L3
  codex reviewer authenticates with the **existing repo `OPENAI_API_KEY`** — no
  separate secret to create.
- *Optional, later:* to add a hard human gate, create a protected environment
  with required reviewers and point the L3 `run-matrix` / `review-pack` jobs at it.

## 3. Secrets

Set on each environment (not as bare repo secrets) so they rotate without
repo-admin Actions access. Most `*_BASE_URL` are optional overrides — but
`DEEPSEEK_BASE_URL` is **required** (see below).

Provider (rollout + judge):
`DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `GLM_API_KEY`, `GLM_BASE_URL`,
`QWEN_API_KEY`, `QWEN_BASE_URL`, `GEMINI_API_KEY`, `OPENAI_API_KEY`,
`LITELLM_API_KEY` (or `BF_TOKEN`), `LITELLM_BASE_URL`, `GITHUB_MODELS_TOKEN`.

Per the agent model policy (see
[`../../docs/integration-tiers.md`](../../docs/integration-tiers.md) §3.3), these
specific keys are load-bearing:

- **`DEEPSEEK_API_KEY` + `DEEPSEEK_BASE_URL` — both required.** The 5 open agents
  (`openhands`, `openclaw`, `opencode`, `pi-acp`, `mimo`) run on
  `deepseek/deepseek-v4-flash` through the LiteLLM usage proxy; the `deepseek`
  provider declares `url_params={"base_url": "DEEPSEEK_BASE_URL"}`
  (`src/benchflow/agents/providers.py`), so `DEEPSEEK_BASE_URL` is **not** an
  optional override — without it the deepseek cells cannot resolve.
- **`GEMINI_API_KEY` — required regardless of which agents run.** It powers both
  the `gemini` / `harvey-lab-harness` rollout agents **and the judge**
  (`gemini-3.1-flash-lite` is the default `judge_model`), so it is needed on any
  lane that judges.
- **`OPENAI_API_KEY`** — `codex-acp` rollout model (`gpt-5.4-nano`) **and** the L3
  Codex equivalence reviewer.
- **`AWS_BEARER_TOKEN_BEDROCK` (+ `AWS_REGION`) — L3 `claude-agent-acp`
  credential, TO BE UPLOADED.** `claude-agent-acp` routes through Bedrock's
  `anthropic-messages` surface
  (`aws-bedrock/us.anthropic.claude-haiku-4-5-20251001`), which requires
  `AWS_BEARER_TOKEN_BEDROCK` and `AWS_REGION`
  (`src/benchflow/providers/litellm_config.py`). **These are not in CI yet.**
  Until they are provisioned, credential-aware emission
  ([`../scripts/filter_credentialed_cells.py`](../scripts/filter_credentialed_cells.py))
  **drops** the `claude` cells as a documented skip (not a red slot), so no
  Daytona sandbox is burned. Upload both to the `pypi-internal-preview`
  environment to enable the full L3 claude lane.

Sandbox: `DAYTONA_API_KEY` (L2/L3 only; L1 is docker-only).

Codex reviewer (L3, required): the **existing repo `OPENAI_API_KEY`** — durable
and revocable, unlike a personal ChatGPT OAuth blob. `codex_review.py` writes it
as an apikey `auth.json` at the codex config path (`$CODEX_HOME/auth.json`,
default `~/.codex/auth.json`) before invoking `codex exec`. Auth precedence:
`OPENAI_API_KEY` / `CODEX_API_KEY`, then an optional `CODEX_AUTH_JSON` blob, then
a pre-existing on-host `auth.json`. If none resolve, L3 fails closed to
**`not mergeable (codex unavailable)`**.

## 4. Labels

- `integration:medium` — opens the L2 lane on a PR (its `detect-scope` only does
  real work when this label is present).
- `integration:deep` — signals an L2/L3-class change; L1's `rollout-smoke` steps
  aside when present so the lanes don't double-spend provider credits.

## 5. Security invariants (do not weaken)

- All workflows are plain `pull_request` / `workflow_dispatch`, never
  `pull_request_target`. Fork PRs receive no secrets.
- `detect-scope`, `plan`, and `review-pack` check out the planner / grader /
  harness / review skill from **trusted `origin/main`** (sparse), never the PR
  head. Only `src/benchflow` (the orchestrator code-under-test) is overlaid from
  the PR head SHA.
- The L3 final verdict is `worst(deterministic, codex)`: codex can only make the
  verdict stricter, never upgrade a deterministic `not mergeable`.
- Actions are SHA-pinned. `actions/download-artifact`
  (`d3f86a106a0bac45b974a628896c90dbdf5c8093`, v4.3.0) and
  `actions/github-script` (`60a0d83039c74a4aee543508d2ffcb1c3799cdea`, v7.0.1)
  are new surface relative to L0/L1's existing pins — re-verify on bump.
