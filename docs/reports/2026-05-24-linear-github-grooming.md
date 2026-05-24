# Linear ↔ GitHub grooming (2026-05-24)

Stress-test swarm + grooming agent pass. GitHub API blocked for PAT lifetime (>366 days); grooming used Linear MCP + public GitHub issue titles from agent where available.

## Link now (bidirectional)

| Linear | GitHub | Action |
|--------|--------|--------|
| ENG-147 | #337 | Daytona startup retry — only clean pair |
| ENG-157 | #366 | Dashboard OPEN-2 / evidence integrity |
| ENG-166 | #368, #371 | str `task_path` / audit PR |
| ENG-164 | #343 | Shared `DEFAULT_MODEL` cross-provider fallback |

## Linear-only (audit cluster ENG-148–153, 154–156)

SkillsBench failure-semantics fixes live on integration branch with tests; no duplicate GitHub issues needed unless release tracking requires it.

## GitHub-only (no Linear) — file or link

| GH | Topic | Suggested Linear parent |
|----|-------|-------------------------|
| #338 | `agent_idle_timeout` CLI exposure | Sibling of ENG-149 (config vs diagnostics) |
| #339 | `--skills-dir` not linked to agent home | ENG-126 / self-gen |
| #341 | Agent install pipefail | ENG-97 |
| #342 | GEMINI vs GOOGLE_API_KEY naming | New small bug |
| #365 | ACP capability over-advertise | Self-gen / instruction path |
| #377 | Scene rollouts undercount timing | ENG-126 |

## New from stress swarm (2026-05-24 afternoon)

| Finding | Linear | GitHub |
|---------|--------|--------|
| Single-task `eval create` does not resume (SDK path) | Extend **ENG-160** | — |
| `summary.json` `idle_timeout` ignores retry rollouts | **ENG-167** | — |
| Unknown agent accepted at CLI | **ENG-168** | — |
| `--tasks-dir` missing → resume noise + traceback | **ENG-169** | — |
| gemini `write_file` JS error on Daytona | Confirm ENG-154 context | — |
| ENG-155 still blocked (empty `_self_gen` export) | Comment on ENG-155 | — |

## Do not duplicate

- ENG-164 ↔ #343 (same fix)
- ENG-166 ↔ #368
- ENG-163 ↔ #361 (different bugs, same subsystem)
- #371 is audit index only, not a product bug

## GitHub issue creation blocked

Org policy rejects fine-grained PAT. Use Linear as source of truth until token renewed; mirror with `gh issue create` locally or GitHub UI.
