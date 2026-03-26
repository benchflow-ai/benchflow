# SmolClaws Environment Protocol

All claw-* mock environments must implement this contract.

## CLI Interface

```bash
# Start server
claw-<name> --db <path> serve --host 0.0.0.0 --port <port> --no-mcp

# Seed database with scenario data
claw-<name> --db <path> seed --scenario <scenario-name>

# Reset to initial state
claw-<name> --db <path> reset
```

## Database

- **Engine**: SQLite (single file, portable, snapshot-friendly)
- **Path convention**: `/data/<name>.db` in container
- **Snapshots**: Save/restore via `/_admin/snapshot/<name>` and `/_admin/restore/<name>`

## Required HTTP Endpoints

### Health
```
GET /health → 200 {"status": "ok"}
```

### Admin (for verifier/evaluator)
```
GET  /_admin/state       → Full DB state as JSON (users, messages, etc.)
GET  /_admin/diff         → Changes since last reset (created, modified, deleted)
GET  /_admin/action_log   → List of API calls made by agent
POST /_admin/reset        → Restore to seeded initial state
POST /_admin/seed?scenario=<name>  → Re-seed with scenario data
POST /_admin/snapshot/<name>       → Save current state
POST /_admin/restore/<name>        → Restore saved state
```

### API Surface
- Must mirror the real API's URL structure and response format
- Agents point at `localhost:<port>` instead of the real API
- Use the gws CLI wrapper for routing: `gws gmail ...` → `CLAW_GMAIL_URL`

## Parity Validation

Each environment must include:
1. **API tests** against the mock that also pass against the real service
2. **Human verification** that responses match real API behavior
3. **Coverage of endpoints** used by tasks

## Seeding

### Scenario Types
- `default` — quick test data (~50 items)
- `long_context` — stress test (~3000 items)
- `task:<task-name>` — task-specific data (loads from `tasks/<task>/data/needles.py`)

### needles.py Format
```python
NEEDLES = [
    {"sender_email": "...", "subject": "...", "body_plain": "...", "labels": ["INBOX"]},
]
FILL_CONFIG = {"target_count": 3000, "distribution": {...}}
```

## Port Registry (from config.toml)

| Service | Port | Env Var |
|---------|------|---------|
| claw-gmail | 9001 | CLAW_GMAIL_URL |
| claw-slack | 9002 | CLAW_SLACK_URL |
| claw-gcal | 9003 | CLAW_CALENDAR_URL |
| claw-gdoc | 9004 | CLAW_GDOC_URL |
| claw-gdrive | 9005 | CLAW_GDRIVE_URL |

## Creating a New Environment

1. Create `packages/environments/claw-<name>/`
2. Implement CLI (seed, serve, reset) following the pattern above
3. Add `[claw-<name>]` section to `config.toml`
4. Write API tests that pass against both mock and real service
5. Register in benchflow: `register_service(name, cli_name, port, db_path)`
