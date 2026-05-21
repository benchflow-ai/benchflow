# BenchFlow v0.5 — status dashboard

A small, dependency-free dashboard for the v0.5 architecture migration —
sidebar navigation, one static `index.html`, Python stdlib only. No build step.

## Run it

```bash
python dashboard/serve.py              # refresh data + serve at localhost:8777
python dashboard/serve.py --run-tests  # re-run the test suite first (~70s)
python dashboard/serve.py --port 9000  # pick a different port
```

Then open <http://localhost:8777>.

## Layout

A left **sidebar** with seven sections; under **Jobs** the sidebar nests the
`jobs/` folder groups so the navigation mirrors the folder on disk.

| Section | What it shows |
|---|---|
| **Overview** | Headline stats — tests, capabilities, roadmap, artifacts, experiments. |
| **Concept map** | The architecture — kernel, four planes, the tree-native execution model, the eight capabilities. |
| **Roadmap** | The v0.5 milestones M1–M4 and their Linear issues. |
| **Tests** | The live pytest suite, suite by suite. |
| **Jobs** | Every rollout under `jobs/` as a tree: **group → run → task → artifacts**. Each artifact expands to show its **actual file contents** (JSON pretty-printed, JSONL line-by-line, CSV as a table). Each group links to the agent advisories that correspond to it. |
| **Advisories** | The agents' review advisories — each cross-linked to a capability and a job group (the jobs ↔ agent-advisor correspondence). |
| **Timeline** | The v0.5 experiments — scanned from `experiments/` and `labs/`, newest first; each experiment expands to its files, every file viewable inline. |

## Refresh the data

```bash
python dashboard/generate.py              # reuse the last test run
python dashboard/generate.py --run-tests  # re-run the suite, then refresh
```

Reload the page — `data.json` is re-fetched each load.

## Data sources

- **Live** — `junit.xml` (the pytest suite), the `jobs/` folder tree (each
  artifact's file content embedded, capped), and the `experiments/` + `labs/`
  folders (the experiments timeline).
- **Authored** — the concept map, the roadmap, and the advisories live as
  Python data at the top of `generate.py`. The advisories carry `capability`
  and `group` fields — that is what wires each advisory to its job group.

## Deployment

A static deploy config (`vercel.json`) is included. Redeploy to Vercel with:

```bash
python dashboard/generate.py && (cd dashboard && vercel deploy --prod --yes)
```

## Files

| File | Role |
|---|---|
| `index.html` | The whole UI — sidebar + seven views, inline CSS + vanilla JS. |
| `generate.py` | Collects the sources into `data.json`. |
| `serve.py` | Refreshes `data.json`, then serves `dashboard/`. |
| `data.json` / `junit.xml` | Generated — git-ignored. |
