# 4. Breadth-tiered agent roster — representative SUBSET at L2, full roster at L3

- Status: Accepted; implemented in #802
- Date: 2026-06-18

## Context

The two Default-config-rules that mean "this change affects *every* agent" —
**agent runtime infra** (`agent-runtime-infra`: `agents/registry.py`,
`protocol.py`, `install.py`, `credentials.py`, `env.py`, shared `acp/**`) and
**network / LLM-proxy routing** (`network-package`: providers, `usage_tracking.py`,
sandbox lockdown, compose files) — must be probed across the agent roster, not
just the baseline agent, because a registry / ACP / proxy-routing change can break
any agent's launcher or model call.

An earlier iteration (commit `b4d10ea`) put the **FULL 9-agent roster at L2** on
both rules. On the auto-on-push label trigger that meant ~40 Daytona cells per
push, which is too expensive for a per-push lane and overlaps with what L3 already
covers: `nine` / `expanded` scopes already fan the full roster via
`_FULL_ROSTER_SCOPES` in `.github/scripts/integration_matrix.py`.

Separately, putting the full roster on L2 exposed a real bug: agents whose
credentials are not in CI (notably `claude-agent-acp`, which needs
`AWS_BEARER_TOKEN_BEDROCK`) would spin up a Daytona sandbox, fail at the first LLM
call, burn the sandbox, and leave the grader logging a **false-red** slot.

## Decision

Make the roster **breadth-tiered** — a variant of the existing rules, not a new
level:

- **L2 (auto-on-push) emits a representative SUBSET.** The `agent-runtime-infra`
  and `network-package` rules carry an `all-agents-subset` axis tag
  (`.github/integration/scope_map.yml`); the planner expands it to `roster_subset`
  = `openhands` + `codex-acp` + `gemini` + `pi-acp`
  (`.github/integration/scope_defaults.yml`). One representative per agent family:
  deepseek-proxy lane / baseline (`openhands`), openai-native (`codex-acp`),
  gemini-native (`gemini`), acp launcher (`pi-acp`). Every family is in CI keys.
- **L3 (`expanded`) runs the FULL 9-agent roster.** `nine` / `expanded` already
  fan the full roster via `_FULL_ROSTER_SCOPES`, so **no new rule is needed** for
  full-at-L3.

Two companion decisions ride with this:

- **Empirically-grounded per-agent model policy** (`agent_models` in
  `scope_defaults.yml`): the 5 openai-completions-family agents (`openhands`,
  `openclaw`, `opencode`, `pi-acp`, `mimo`) run `deepseek/deepseek-v4-flash`
  through the LiteLLM usage proxy (`mimo`-on-deepseek replaces `xiaomi`, closing
  the XIAOMI gap); `gemini` / `codex-acp` / `harvey-lab-harness` stay on their
  native models (harvey's `_create_adapter` raises for non-claude/gpt/gemini
  models, so it *cannot* use deepseek); `claude-agent-acp` routes through Bedrock's
  `anthropic-messages` surface (`aws-bedrock/...`), not the bare claude-haiku id.
  Branded CLIs are protocol-locked — the `deepseek` provider serves only
  `openai-completions` — which is *why* the model is chosen per family.
- **Credential-aware emission** (`.github/scripts/filter_credentialed_cells.py`):
  the planner stays PURE/deterministic; a separate env-aware step between plan and
  run-matrix drops any cell whose required credential is absent (via the
  `resolve_agent_env` `ValueError` "<KEY> required ... not set") and records it as
  a documented SKIP rather than a red slot. So `claude-agent-acp` is skipped until
  `AWS_BEARER_TOKEN_BEDROCK` is provisioned — no burned sandbox, no false red.

## Consequences

- (+) The per-push L2 lane drops from ~40 full-roster Daytona cells to a 4-agent
  representative subset, while the full 9-agent coverage is preserved at L3.
- (+) Every credential / launcher family is still exercised auto-on-push (the
  subset covers deepseek-proxy / openai-native / gemini-native / acp-launcher).
- (+) Un-keyed agents no longer burn sandboxes or log false-red slots; missing
  credentials are documented skips.
- (−) An agent-family-specific regression that only manifests on a non-subset
  member of the same family (e.g. `openclaw` but not `openhands`) is caught at L3,
  not on the auto-on-push L2 lane.
- (−) The subset roster and the `all-agents-subset` ↔ `roster_subset` mapping must
  be kept in sync between `scope_map.yml` and `scope_defaults.yml`.

## Alternatives considered

- **Keep the full 9-agent roster at L2** (the `b4d10ea` state): maximal per-push
  coverage but ~40 Daytona cells per push and redundant with L3's `expanded`
  fan-out. Rejected as too expensive for an auto-on-push lane.
- **Add a new dedicated breadth level** between L2 and L3: more ladder complexity
  for no benefit, since `_FULL_ROSTER_SCOPES` already gives full coverage at L3.
  Rejected — this is a *variant* of the existing rules, not a new level.
- **Filter un-keyed agents inside the planner**: would make the planner
  environment-dependent and non-deterministic, breaking its MiniYaml-safe,
  pure-data contract. Rejected in favor of a separate env-aware filter step.
