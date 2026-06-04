# BenchFlow v0.5 — status dashboard

A small, dependency-free dashboard for the v0.5 architecture migration —
sidebar navigation, one static `index.html`, Python stdlib only. No build step.

## Run it

```bash
LINEAR_API_KEY=... python dashboard/serve.py  # mirror Roadmap from Linear + serve
LINEAR_API_KEY=... LINEAR_PROJECT_ID=... python dashboard/serve.py  # stable project selector
BENCHFLOW_DASHBOARD_JOBS_ROOT=/path/to/worktree LINEAR_API_KEY=... python dashboard/serve.py  # mirror git-ignored jobs from another worktree
LINEAR_API_KEY=... python dashboard/serve.py --run-tests  # re-run tests first (~70s)
LINEAR_API_KEY=... python dashboard/serve.py --port 9000  # pick a different port
```

Then open <http://localhost:8777>.

## Layout

A left **sidebar** with the sections below; under **Jobs** the sidebar nests the
`jobs/` folder groups so the navigation mirrors the folder on disk. The
**Daytona** section is live-server-only — it is hidden on the static (Vercel)
build, which lacks the `/daytona/available` probe.

| Section | What it shows |
|---|---|
| **Overview** | Headline stats — tests, capabilities, roadmap, artifacts, experiments. |
| **Concept map** | The architecture — kernel, four planes, the tree-native execution model, the eight capabilities. |
| **Architecture** | The full architecture write-up (`architecture.md`) rendered inline, with a section outline. |
| **Tickets** | A read-only, two-column mirror of the Linear v0.5 project: ticket list on the left, selected issue detail on the right. |
| **Tests** | The live pytest suite, suite by suite. |
| **Jobs** | Every rollout under `jobs/` as a tree: **group → run → task → artifacts**. Each artifact expands to show its **actual file contents** (JSON pretty-printed, JSONL line-by-line, CSV as a table). Each group links to the agent advisories that correspond to it. |
| **Daytona** | Live cloud sandboxes — count, state breakdown, and per-sandbox age — polled from the running server (auto-refresh every 15s). |
| **Advisories** | The agents' review advisories — each cross-linked to a capability and a job group (the jobs ↔ agent-advisor correspondence). |
| **Timeline** | The v0.5 experiments — scanned from `experiments/` and `labs/`, newest first; each experiment expands to its files, every file viewable inline. |

## Refresh the data

```bash
LINEAR_API_KEY=... python dashboard/generate.py              # mirror Linear + reuse tests
LINEAR_API_KEY=... LINEAR_PROJECT_ID=... python dashboard/generate.py
BENCHFLOW_DASHBOARD_JOBS_ROOT=/path/to/worktree LINEAR_API_KEY=... python dashboard/generate.py
LINEAR_API_KEY=... python dashboard/generate.py --run-tests  # re-run suite, then mirror
python dashboard/generate.py --allow-missing-linear          # local UI dev only
```

The served page polls `data.json` and rerenders when the feed changes. While
`dashboard/serve.py` is running, `data.json` refreshes on startup, when the git
working tree fingerprint changes, and on the periodic live-data interval.

## Data sources

- **Live** — `junit.xml` (the pytest suite), the `jobs/` folder tree (each
  artifact's file content embedded, capped), and the `experiments/` + `labs/`
  folders (the experiments timeline). `BENCHFLOW_DASHBOARD_JOBS_ROOT` may point
  at another worktree root, or directly at its `jobs/` directory, when the
  active dashboard worktree does not contain the git-ignored rollout artifacts.
- **Live / mirrored** — Linear drives Tickets. Normal generation requires
  `LINEAR_API_KEY`, fetches the project issues directly from Linear, and refuses
  to publish stale/non-live ticket data. `--allow-missing-linear` is only for
  local UI development and renders an explicit unavailable state.
  Project selection prefers `LINEAR_PROJECT_ID`, then `LINEAR_PROJECT_URL`,
  then `LINEAR_PROJECT_SLUG`, then `LINEAR_PROJECT_NAME` / `LINEAR_PROJECT`,
  then the built-in v0.5 project name. Name, URL, and slug selectors resolve the
  project first; issue pagination then filters by the resolved Linear project ID.
- **Repo status** — branch, head, and dirty counts are embedded so the page can
  show when the working tree changed without exposing a file list.
- **Authored** — the concept map and advisories live as
  Python data at the top of `generate.py`. The advisories carry `capability`
  and `group` fields — that is what wires each advisory to its job group.

## Deployment

A static deploy config (`vercel.json`) is included. Redeploy to Vercel with:

```bash
LINEAR_API_KEY=... python dashboard/generate.py && (cd dashboard && vercel deploy --prod --yes)
```

The generated static `data.json` contains internal Linear ticket metadata and
repo status summary. Deploy only behind the intended access controls.

## Files

| File | Role |
|---|---|
| `index.html` | The whole UI — sidebar + views, inline CSS + vanilla JS. |
| `generate.py` | Collects the sources into `data.json`. |
| `roadmap.py` | Fetches and normalizes live Linear data. |
| `jobs_visibility.py` | Run-visibility policy for the `jobs/` tree (which runs surface, and their target/signal labels). |
| `daytona_status.py` | Live Daytona sandbox snapshot for the Daytona panel; `serve.py` imports `snapshot`. |
| `serve.py` | Refreshes `data.json`, then serves `dashboard/`. |
| `data.json` / `junit.xml` | Generated — git-ignored. |
