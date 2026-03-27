# benchflow — Developer Guide

## Repos

| Repo | Purpose |
|------|---------|
| **benchflow** | SDK, ACP client, agent registry, Job orchestration |
| **smolclaws** | Tasks, mock environments (claw-*), clawbench eval/optimize |

benchflow is a dependency of smolclaws' `clawbench` package (pinned to git commit in `packages/clawbench/pyproject.toml`).

## Cross-Dev Workflow

1. Make changes in benchflow → push branch → create PR
2. After merge, get commit: `gh api repos/benchflow-ai/benchflow/commits/main --jq '.sha'`
3. Update pin in `smolclaws/packages/clawbench/pyproject.toml`
4. Test: `clawbench eval --backend benchflow --agent claude-agent-acp -e daytona`

## Documents

| What | Where |
|------|-------|
| Meeting notes | [Google Doc](https://docs.google.com/document/d/1gej25JHQvVB-oejWS8LThEzxHlb_c2vkKIvQUv50foM/edit) |
| Paper | [Overleaf](https://www.overleaf.com/3586481587dnbbghkcjrfn#4373f6) |
| Discord | main channel, benchflow-beta |

## Setup

```bash
pip install -e .
source .env  # ANTHROPIC_API_KEY, DAYTONA_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY
```

DevContainer has everything pre-installed. Run `gcloud auth login --enable-gdrive-access` for Google Docs access.

## Agents

| Agent | Requires | Status |
|-------|----------|--------|
| `claude-agent-acp` | ANTHROPIC_API_KEY | Tested |
| `pi-acp` | ANTHROPIC_API_KEY | Tested |
| `openclaw` | ANTHROPIC_API_KEY | Tested |
| `openclaw-gemini` | GEMINI_API_KEY | Tested |
| `codex-acp` | OPENAI_API_KEY | Tested |
| `gemini` | GOOGLE_API_KEY | Tested |

Add new agents: `register_agent(name, install_cmd, launch_cmd, requires_env)`.

## Team

xiangyi (lead), kyoung (eval/infra), yimin (gmail), amy (slack), wenbo (gdrive), bingran+jayson (gcal), shenghan+zodi (gdoc)
