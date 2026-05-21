# Dashboard artifact audit

The shared charter — and the **memory** — for the artifact-audit subagent
suite and the main session.

The audit subagent runs every rule below against the dashboard's produced
artifacts (`dashboard/data.json`, and where relevant the rendered UI), logs
what it finds in **Findings**, and proposes new rules. The main session fixes
the issues and keeps the rules current. **Both sides append to this file** —
it is the iterated memory of what "good artifacts" means here.

## How the audit runs (the suite)

1. `python dashboard/generate.py` — produce a fresh `data.json` from the real
   `jobs/`, `experiments/`, and `labs/` directories.
2. Check **every** rule under "Rules" against `data.json`.
3. Log each violation in "Findings" (newest first); mark fixed ones `fixed`.
4. If a new *class* of problem appears, add a numbered rule for it.
5. Re-run on every change to `generate.py` or the dashboard.

## Rules

- **R1 — No empty artifacts.** A 0-byte file carries nothing to view, so
  `generate.py` must not emit it as an artifact row. The fact that, e.g., a
  trajectory produced 0 events belongs on the task (`trajectory_events`),
  never as a free-standing "0 lines · 0 events · empty file" row.
- **R2 — `content: null` only for real reasons.** A `null` `content` must mean
  a directory or a genuine binary — never a readable text file silently
  dropped or mis-read.
- **R3 — Truthful `info`.** Every `info` / count string must be accurate and
  non-misleading: line, byte, and event counts must match the file; "0 …"
  strings must not reach the UI as if they were content (see R1).
- **R4 — Bounded payload.** `data.json` stays well under 2 MB so the static
  deploy loads fast; trajectory/text content is capped (240 lines / 64 KB)
  and any cap is flagged with `truncated: true`.
- **R5 — Content integrity.** Every embedded `content` round-trips: JSON
  parses, a `.jsonl`'s real line count equals `content_lines`, a CSV's
  column count is consistent across rows.
- **R6 — No dead references.** Every advisory `group` names a real job group;
  every capability number / Linear issue id referenced actually exists.
- **R7 — Every section has data or an honest empty-state.** No view renders a
  blank panel or a JS error when its source list is empty.

## Findings

Newest first. Status ∈ `open` · `fixed` · `wontfix`.

| #  | Date       | Rule | Issue | Status |
|----|------------|------|-------|--------|
| F1 | 2026-05-21 | R1   | Empty `trajectory/acp_trajectory.jsonl` and `agent/gemini.txt` were surfaced as "0 lines · 0 events · empty file" rows in the Jobs view. | fixed — `generate.py` now skips 0-byte files via `_is_empty()` (artifacts + experiment files). |
