# Claw Gmail

A high-fidelity, stateful mock of the Gmail REST API, built for stress-testing AI coding agents that interact with Gmail.

## Why this exists

AI coding agents (like [OpenClaw/nanoclaw](https://github.com/qwibitai/nanoclaw)) increasingly perform real-world tasks: reading email, sending replies, managing labels, organizing inboxes. When these agents run against a real Gmail account, mistakes are irreversible — one bad prompt and [all your emails get deleted](https://x.com/summeryue0/status/2025774069124399363).

Claw Gmail provides a safe, fully stateful Gmail environment where agents can be exercised against realistic data without touching production inboxes. It is part of [envdash](https://github.com/qwibitai/envdash), a monorepo for managing, testing, and automatically improving Agent Skills for agentic workflows.

## What it does

- **54 Gmail API endpoints** (92% of the real API surface) — messages, threads, labels, drafts, history, attachments, profile, and all settings sub-resources
- **Full MIME/RFC 2822 support** — agents can send raw base64url-encoded emails exactly like the real API
- **Stateful SQLite backend** — persistent CRUD, multi-user mailboxes, local delivery between users
- **Snapshot/restore** — save and reset DB state for deterministic evaluation runs
- **30 golden fixtures** captured from a real Gmail account (`mediar.acc1@gmail.com`) with 19 conformance tests validating response shapes match real Gmail
- **3 evaluation tasks** (targeted promo delete, vendor report organize, ambiguous cleanup) with automated verifiers using DB state diffs and action logs
- **MCP server** — expose all endpoints as MCP tools via `fastapi-mcp`
- **Gymnasium environment** — `GmailEnv` for RL-style agent training
- **Web UI** — inbox view with HTML email rendering, dashboard with coverage visualization, API explorer, DB viewer

## Quick start

```bash
cd packages/environments/claw-gmail

# Seed with realistic email data and start the server
uv run claw-gmail seed --scenario long_context
uv run claw-gmail serve --no-mcp
# API:       http://127.0.0.1:8001/gmail/v1/users/me/messages
# Web UI:    http://127.0.0.1:8001/
# Dashboard: http://127.0.0.1:8001/dashboard
# API Docs:  http://127.0.0.1:8001/docs
```

`uv run` reads `pyproject.toml`, creates a venv, installs deps, and runs the CLI — no manual `pip install` needed.

To start the Environment UI (separate Next.js app, richer than the built-in web UI):

```bash
cd packages/ui/environment
npm run dev
# http://localhost:3000 (connects to API on :8001)
```

### Scenarios

| Scenario | Emails | Use case |
|----------|--------|----------|
| `default` | ~57 | Quick testing |
| `long_context` | ~3000 | Stress-testing: search, pagination, safety, ambiguous cleanup |

### Alternative: pip install

If you prefer a traditional install:

```bash
pip install -e ".[all]"
claw-gmail seed --scenario long_context
claw-gmail serve --no-mcp
```

## CLI

```
claw-gmail seed              # Seed DB with realistic data (scenarios: default, long_context, safety_corporate, phishing)
claw-gmail serve              # Start API server (default :8001)
claw-gmail reset              # Restore DB to initial seed state
claw-gmail list-tasks         # Show all evaluation tasks
claw-gmail run-task <name>    # Print task instructions for an agent
claw-gmail eval-task <name>   # Evaluate task completion (PASS/FAIL + reward)
```

## API compatibility

The API is mounted at `/gmail/v1/` and mirrors Google's Gmail REST API:

```
GET    /gmail/v1/users/{userId}/messages
GET    /gmail/v1/users/{userId}/messages/{id}
POST   /gmail/v1/users/{userId}/messages/send
POST   /gmail/v1/users/{userId}/messages/{id}/modify
POST   /gmail/v1/users/{userId}/messages/{id}/trash
...
```

Agents using any Gmail client library or MCP can point at this server instead of `googleapis.com` with zero code changes.

Admin endpoints for evaluation harnesses:

```
POST   /_admin/reset           # Reset to seed state
GET    /_admin/state            # Full DB state dump
GET    /_admin/diff             # Diff from seed snapshot
GET    /_admin/action_log       # All API actions taken
POST   /_admin/snapshot/{name}  # Save named snapshot
POST   /_admin/restore/{name}   # Restore named snapshot
GET    /_admin/tasks            # List all evaluation tasks
POST   /_admin/tasks/{name}/evaluate  # Run task evaluator
GET    /_admin/skills           # List agent skills
```

## Testing

```bash
pytest tests/ -q    # 122 tests
```

| Suite | Tests | What it covers |
|-------|-------|----------------|
| `test_api.py` | 37 | Full CRUD for messages, threads, labels, drafts, admin, tasks |
| `test_conformance.py` | 19 | Response shape validation against 30 real Gmail fixtures |
| `test_settings.py` | 21 | All settings sub-resources (filters, sendAs, forwarding, delegates, vacation, IMAP, POP, language) |
| `test_mime.py` | 12 | RFC 2822 build/parse, base64url encoding, message-ID generation |
| `test_api.py` (extended) | 33 | Batch operations, attachments, history, search, draft CRUD |

## Evaluation tasks

Tasks are defined as task directories in `tasks/` and loaded via `claw_gmail/tasks/`:

| Task | What it tests |
|------|---------------|
| `harbor-targeted-promo-delete` | Delete promotional emails without touching work/personal/notification emails |
| `harbor-vendor-report-organize` | Find and label 7 vendor reports across 3000 emails |
| `harbor-ambiguous-cleanup` | Clean up an inbox with ambiguous emails that blur category boundaries (the "Summer Yue" scenario) |

Each task provides an instruction prompt (`instruction.md`), an automated evaluator (`evaluate.py`), and a reference solution (`solve.sh`). Evaluators check the DB state diff and action log to compute a reward score.

## Seeding and snapshots

`claw-gmail seed --scenario <name>` creates a SQLite database (`.db` file) with generated emails, then saves the full DB state to `snapshots/initial.json`. This JSON snapshot is the reference "clean" state.

When the server's `/_admin/reset` endpoint is called (e.g. between evaluation task runs), it restores the DB from `initial.json` back to the original seeded state. This is how the clawbench pipeline runs multiple tasks against the same starting mailbox.

- `*.db` files are temporary and gitignored — recreated each time you seed
- `snapshots/initial.json` is the seed state reference, used by reset

Scenarios:
- `default` — ~57 messages, ~42 threads (quick testing)
- `long_context` — ~3000 emails with curated content library, needle emails, and ambiguous edge cases for stress-testing (search, pagination, safety)
- `safety_corporate`, `phishing` — specialized safety scenarios

## Project structure

```
packages/environments/claw-gmail/
├── claw_gmail/
│   ├── api/              # FastAPI routes (messages, threads, labels, drafts, settings, admin)
│   ├── models/           # SQLAlchemy ORM (message, thread, label, draft, attachment, settings)
│   ├── seed/             # Realistic email content + scenario generator
│   ├── state/            # Snapshot save/restore + action logging
│   ├── tasks/            # task loaders + evaluators
│   ├── env/              # Gymnasium environment wrapper
│   ├── mcp/              # MCP server via fastapi-mcp
│   ├── web/              # Web UI (inbox, dashboard, API explorer, DB viewer)
│   ├── cli.py            # CLI entry point
│   └── server.py         # Uvicorn server setup
├── tests/
│   ├── fixtures/         # Golden fixtures from real Gmail + API spec
│   └── test_*.py         # 122 tests
├── scripts/              # Gmail auth + fixture capture from real account
└── pyproject.toml
```

## Part of envdash

This environment lives at `packages/environments/claw-gmail/` inside the [envdash](https://github.com/qwibitai/envdash) monorepo. envdash is a tool for managing, testing, and improving Agent Skills — folders of instructions, scripts, and resources that agents discover and use to perform tasks more accurately. See [agentskills.io](https://agentskills.io/home) for the Agent Skills specification.

The goal is to use environments like claw-gmail to stress-test agents (particularly [OpenClaw/nanoclaw](https://github.com/qwibitai/nanoclaw)) and then automatically improve their skills using approaches like [clawbench](https://gepa-ai.github.io/gepa/blog/2026/02/18/automatically-learning-skills-for-coding-agents/).
