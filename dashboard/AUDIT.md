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
- **R8 — Truncated `content` must stay self-describing, not silently
  malformed.** When `truncated: true`, the embedded `content` is a fragment.
  A fragment of a structured format (JSON especially) no longer parses — so
  R5's "every embedded JSON `content` `json.loads()`" must be read as "every
  *un-truncated* JSON does; a truncated one is exempt but MUST carry
  `truncated: true` and a real `content_lines`." `kind`/`lang` must still name
  the true format so the UI degrades gracefully (raw view, not a crash).
- **R9 — Counts are logical, not physical.** Any user-facing row/record count
  (`rows`, "N rows", "N events") must reflect *logical* records, not raw
  newline count. CSV fields may contain embedded newlines inside quotes; a
  `.jsonl` event is one JSON object. `content_lines` may stay physical (it is
  a line count by definition), but derived counts must not be `lines - 1`.
- **R10 — Collection caps must be visible.** When `generate.py` caps a list
  (e.g. `labs/` experiment files at 24, `files[:24]`), the artifact must
  record that a cap was applied (a `files_truncated` / `n_total` field) so the
  UI can say "showing 24 of N" instead of silently dropping the rest.

## Findings

Newest first. Status ∈ `open` · `fixed` · `wontfix`.

| #  | Date       | Rule | Issue | Status |
|----|------------|------|-------|--------|
| F6 | 2026-05-21 | —/R10 | `collect_experiments()` caps each `labs/` experiment at `files[:24]` with no `n_total`/`files_truncated` marker. `labs/benchjack-sandbox-hardening/` currently has exactly 24 non-empty files — at the boundary, nothing dropped *this run* — but a 25th file would vanish from `data.json` with no UI indication. Latent silent-truncation risk; new rule R10 added. | fixed — every experiment now carries `files_truncated` + `n_files_total`; the `labs/` cap was raised to 60. |
| F5 | 2026-05-21 | —    | `labs/benchjack-sandbox-hardening/comparison.ipynb` is embedded as `kind: "data"`, `lang: "text"` — a Jupyter notebook is JSON, but `.ipynb` is absent from `generate.py`'s `_LANG` map. 10,580-byte payload shown as raw text instead of structured/JSON. Misleading `kind`/`lang`. | fixed — `.ipynb` added to `_LANG` (→ `json`) and to the script-kind set. |
| F4 | 2026-05-21 | R8/R5 | `labs/reward-hack-matrix/sweep_0.2.0_vs_0.2.2.json` is `lang: "json"`, `truncated: true` (real `content_lines: 7994`, embedded ~9 KB / 240 lines). Its embedded `content` does **not** `json.loads()` — fails `Expecting ',' delimiter: line 240 column 6`. R5 as literally worded ("every embedded JSON `content` actually `json.loads()`") is violated; new rule R8 scopes R5 to un-truncated JSON. UI's `prettyJSON()` already "leaves malformed JSON untouched" so it degrades to raw view rather than crashing — quality issue, not a render break. | wontfix — accepted per R8: truncating a large JSON is inherent; the fragment carries `truncated: true` + real `content_lines`, `kind`/`lang` name the true format, and the UI degrades to a raw view. Not a defect. |
| F3 | 2026-05-21 | R3/R9 | `experiments/skillsbench-byos-validation.csv` reports `rows: 22` in `data.json`, but the file has only **20 data records** (21 logical CSV rows incl. header). `_exp_file()` computes `rows = content_lines - 1` from *physical* line count (23 lines); two records contain quoted multi-line `error` fields (rows 19-20 and 21-22 are each one logical record). The Experiments view's file-meta will display "22 rows" — overstated by 2. All 6 other experiment CSVs match. New rule R9 added. | fixed — `_csv_rows()` counts logical records via `csv.reader` (quoted multi-line fields handled); skillsbench-byos now reports 20. |
| F2 | 2026-05-21 | R9   | `_artifact()` labels every `.jsonl` trajectory `info` as "`{lines} events`" where `lines` is the physical newline count, not a parsed event count. Currently harmless (all 39 trajectory `.jsonl` files are one JSON object per line, so lines == events), but the count is structurally physical-line-based and would misreport on any file with a blank trailing line or pretty-printed record. Logged against new rule R9 as a latent correctness risk. | fixed — `_count_jsonl_events()` counts non-blank lines; used by `_artifact` trajectory `info` and `_task_row.trajectory_events`. |
| F1 | 2026-05-21 | R1   | Empty `trajectory/acp_trajectory.jsonl` and `agent/gemini.txt` were surfaced as "0 lines · 0 events · empty file" rows in the Jobs view. | fixed — confirmed: re-run of `generate.py` produced 214 job artifacts, **0** with empty content + 0 lines; `_is_empty()` skips 0-byte files in both `_task_artifacts()` and `collect_experiments()`. |
