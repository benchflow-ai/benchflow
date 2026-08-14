# Trace Viewer — TODO
Only for 'bench eval view <rollout|job>' for now
Benchmarked against the [posttrainbench traces viewer](https://posttrainbench.com/traces/); replaces today's `src/benchflow/trajectories/viewer.py`.

**Principle**: every slice ends with a page you can actually open — each one just does more than the last. Build the frontend properly once, and **omit any field the data doesn't have** instead of changing the capture side just to fill a slot.

## Rendering architecture

`rollout dir → viewer.py → payload (JSON) → render.js → HTML`
The payload is inlined as a data island locally, or fetched from HF by the hosted site.

- **Python emits data only, JS emits HTML only**: the static site has no Python, so keeping the rendering in JS is what lets one copy serve both `bench eval view` and the static site instead of drifting into two
- **Pure-function boundary**: `render.js` has no DOM / no fetch / no globals, so it is testable outside a browser; the DOM belongs entirely to `boot.js`
- **The payload is derived data**: never written to disk, never becomes a fourth trace schema; it carries a `schema_version` and fails loudly on a mismatch instead of mis-rendering (page and data will ship from different places)

---

## Data as it stands (decides which fields get built)

Already **present** in a single rollout directory:

| Source | Contents |
|---|---|
| `trajectory/acp_trajectory.jsonl` | event stream: `type` / `kind` / `title` / `status` / `content` / `text` / `tool_call_id` |
| `result.json` | agent, model, rewards, n_tool_calls, started_at/finished_at, error_category, skill_mode, trajectory_summary |
| `result.json` → `agent_result` | `n_input_tokens` / `n_output_tokens` / `n_cache_read_tokens` / `n_cache_creation_tokens` / `total_tokens` |
| `timing.json` | environment_setup / agent_setup / agent_execution / verifier / total |
| `verifier/` | `reward.txt` / `ctrf.json` / `test-stdout.txt` |
| `config.json` `prompts.json` `artifacts/` `agent/` `trainer/` | |
| `turn*.txt` | claude-code-harness runs only (stream-json, carries cost) |


---

## Step 1 — single-run page

**Deliverable**: `bench eval view <rollout_dir>` → a complete page that visually matches ptb.

### 1. Reuse ptb's style, adapt our rendering logic

Vendor their `styles.css` byte for byte (stored as `ptb-styles.css`, never edited); `render.js` emits
**their DOM contract** (`.topbar` / `.layout` / `.rail` / `.summary-card` / `.score-big` / `.event` +
`.event-marker` + `.event-body` / `.tool-call` / `.tool-result-body` / `.diff-add`), or the vendored
CSS has nothing to attach to. Our own layer is always `bf-`-prefixed and lives in `benchflow.css`, so
re-vendoring stays a straight overwrite.

**Their `run.js` is not used**: it dispatches on Claude Code stream-json's `assistant/user/system/result`
records and hardcodes the Bash/Edit/Read/TodoWrite tool names (Codex gets a second branch, `codex_item`).
BenchFlow normalizes 28 harnesses into ACP, so their run.js would render none of reward / verifier /
skill_mode / oracle / timeout.

Fonts don't follow them either: they link Google Fonts, we inline a woff2 data URI (offline + CSP). Their CSS itself has no `url()` / `@import`, so vendoring carries no network dependency.

### 2. Run status reads explicit fields, it doesn't guess

Six states — `passed / failed / errored / verifier-errored / timeout / not-scored` (upstream has only
good/bad, because their score is a percentage). The verdict reads `result.json`'s `error` /
`verifier_error` / `partial_trajectory` / `error_category` rather than inferring it from whether
`agent_timeout` shows up in the trace; timeout classification reuses `_utils/scoring.classify_error`
(the fallback fires only for old rollouts missing `error_category`) instead of matching a `"timed out"`
substring ourselves, which would create a second classification table.

**reward is not a percentage**: upstream's `.score-big::after` `%` is overridden with the raw value plus
a `reward` unit label. It renders only when the scalar `rewards.reward` exists; otherwise `Not scored` —
unscored and 0 are two different things.

### 3. Event model: ACP is the single source of truth

The ACP trajectory is canonical; stream-json / `turn*.txt` are a fallback only when ACP is missing. When
both exist they are neither merged nor rendered twice. Both sources are normalized into one internal
event model first, and the rendering layer only ever sees one shape.

Turn boundaries: a `user_message` opens a turn and everything up to the next `user_message` belongs to
it; setup / oracle events before the first user message form their own group. `agent_timeout` and
`oracle` each get their own rendering branch — a timeout must not look like a clean finish.

**One view only**, no Focus/Full toggle — upstream uses it to filter Claude Code's system records, and
after normalization we have no such noise. An `agent_thought` with an empty body renders as a
placeholder instead of being hidden.

Known gap: neither `_normalize_acp_events` nor `_capture.py` has an else branch, so an unrecognized
event type is silently dropped. A new event type has to be added on both sides.

### 4. Full tool-output rendering (the most important item in this slice)

ACP `tool_call.content` must survive verbatim. Review found three violations; the current behavior is:

- **Block-by-block rendering**: every content block is handled independently, and a block that yields no
  text falls back to its own JSON. Previously only blocks carrying text were rendered, and a single text
  block suppressed the JSON fallback — the `diff` block of edit-type tools vanished wholesale
- **Binary detection per block**: one image no longer marks the text of the same tool call as
  `[binary output omitted]`
- **100k-character cap per output**: head and tail are kept, with an explicit truncation marker pointing
  at `trajectory/acp_trajectory.jsonl`. CSS `max-height` bounds the visuals, not the bytes — a 500 KB
  log still lands in the HTML verbatim

Presentation is a dark terminal block (command line + description + collapsible OUTPUT); `{kind:"diff"}`
renders as colored `- old` / `+ new` lines plus the file path, long ones collapsed, with a global
"expand outputs" switch.

### 5. The trace is untrusted input

Markdown goes through a small subset only (headings / lists / quotes / fenced code / inline code /
bold-italic / links), and it is **escaped first, transformed second**; links are restricted to http(s),
and a rejected one is kept as inert text rather than deleted. ANSI SGR color/bold becomes spans and every
other control sequence is dropped — upstream escapes the output as-is, so `\x1b[31m` is shown to the reader.

### 6. A field the data doesn't have is not rendered

Upstream's right rail (GPU/CPU curves) is dropped as a whole column, collapsing the layout to two columns
via `.layout.bf-no-right-rail`; events carry no timestamps, so the time column is not rendered. The
granularity is **a whole block missing → omit the block, a single value missing → show `—`**: the token
table, the four timing bars and skill invocations are not rendered at all without data; a summary row
like `agent_result.cost_usd` (normally unavailable on agent-native runs) stays and shows `—`, because
"the field exists, it just wasn't measured this time" is itself information. The top has one tab, Run trace.

### 7. BenchFlow-specific signals (upstream has no equivalent)

Four timing bars (environment / agent setup / agent / verifier — upstream has a single total), skill mode
+ `n_skill_invocations`, result-level notices (agent error / verifier error / partial trajectory), the ACP
tool's `kind` + `status` lifted into the tool header (upstream hardcodes tool names), and the
`usage: <source> · price: <source>` provenance label (an unlabeled token count reads as measured). The
token table reads `agent_result` and not `final_metrics`, which is missing cache_creation.

### Tests come in two layers

- `tests/test_trajectory_viewer.py` tests the **payload**: status verdict, tokens taken from
  `agent_result`, block completeness, truncation metadata, turn grouping, `</script>` escaping in the
  data island
- `tests/test_trajectory_viewer_render.py` drives `render.js` **through node** to test rendering: tool
  output / diff / binary placeholder, markdown, ANSI, injection defense, reward 0 vs unscored, schema
  mismatch. Skipped when node is absent
- Why two layers: the data is all inlined in the page, so asserting that a string appears in the HTML is
  a tautology that always holds — it passes even when rendering is broken. Fixtures use
  `_build_rollout_result()` to produce a canonical rollout artifact

---

## Later features

Page capabilities:

- [ ] **Job list page** — `bench eval view <job_dir>` produces a run list, clicking through opens the
      single-run page
- [ ] **Verifier panel** — see why a run was judged a failure, straight from the rollout's own
      `verifier/` (`reward.txt` / `ctrf.json` / `test-stdout.txt`). Frontend-only, the data is already there
- [ ] **Cheating-audit verdicts** — reward hacking / no-skill leakage on the page. Different source:
      `bench review` writes `review-result.json` into `jobs/review-<stamp>/`, not into the rollout, so this
      needs a run↔review join first. Renders as a judgment with provenance (reviewer agent / model /
      rubric), never as a verdict of fact

- [ ] **Cost** — how much each run spent
- [ ] **Per-call overhead** — tokens and latency for each LLM call, from `llm_trajectory.jsonl` (the
      provider HTTP audit log; only runs going through the proxy have it — agent-native talks to the
      provider directly and has none, same root cause as cost)
- [ ] **System metrics** — GPU / CPU, only exists for local deployments

Data and coverage:

- [ ] **Event timestamps** — needed before the time column can be filled
- [ ] **Multi-agent validation** — validate a codex run besides claude, across both access paths (local
      deployment and API)
