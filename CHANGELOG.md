# Changelog

## [Unreleased]

### Added
- **Uploads are confirmed all the way into cloud storage.** After the
  progress bar finishes, `bench traj upload` now polls the contribution
  service's new `GET /v1/uploads/{digest}` capture-status endpoint (the
  validation ledger) until the validator's verdict: `✓ Verified in cloud
  storage` once the capture is promoted to `sources/community/<digest>/`, a
  concise exit-1 error with the fixable detail if the validator rejects it,
  and a `bench traj status sha256:<digest>` handoff line if validation is
  still running when the budget (default 240 s, `BENCHFLOW_TRAJ_WAIT_SECONDS`
  override, `--no-wait` opt-out) runs out. A handshake 409 ("already
  submitted") prints the verified line immediately, and a deployed broker
  that predates the endpoint (404) keeps today's behavior unchanged. The new
  `bench traj status DIGEST` command runs one check on demand. Status polls
  consume a separate, higher rate-limit budget (`TRAJ_STATUS_RATE_LIMIT`,
  default 720/hour/IP) and reveal only the ledger state, the bounded
  rejection detail, and the public promotion prefix — never contributor
  identity or quarantine internals. The broker must be redeployed
  (`deploy-trajectory-upload` workflow or `scripts/deploy.sh`) before the
  endpoint answers in production; the CLI degrades gracefully until then.
- **The `bench traj` family shares one polished terminal design language.**
  A new presentation-only kit (`cli/_traj_tui.py`) gives `traj setup`,
  `traj upload`, and `traj status` a coherent look: a `◆ benchflow · <command>`
  banner, styled `◇` input prompts, an arrow-key recent-session picker (↑/↓,
  1-9 jump, esc to fall back to typing a path) on real terminals, colored
  step kinds in the report preview matching the browser viewer's palette,
  and rounded panels. Every interactive affordance degrades to the exact
  previous prompt-driven flow off-TTY (agents, pipes, CI, Windows), and all
  machine-read lines (`Masked for you:`, `Digest:`, `Repo:`) stay plain.
- **Uploads can carry the session's workspace folder as a zip attachment.**
  `bench traj upload` reads the session's recorded working directory (the
  same Claude `cwd` / Codex `session_meta` provenance as repo tagging) and
  archives it into the capture as `workspace/<name>.zip`, printing
  `Workspace: <path> (from session cwd; use --no-workspace to omit)` and a
  `Workspace attached:` line with size, file count, and exclusion count.
  VCS internals, dependency trees, caches, symlinks, and secret-shaped
  filenames (`.env*`, `*.pem`, `*.key`, `id_rsa*`, `.netrc`, …) never enter
  the archive; everything else is archived as-is without content redaction,
  and the attach line says so. Workspaces over 1 GiB (measured before
  compression, so the zip is never created), over 50,000 files, missing, or
  empty are skipped with a printed reason instead of failing the upload.
  When detection fails on a real terminal, one optional prompt accepts a
  folder or skips on Enter; `--workspace-dir` overrides detection and
  `--no-workspace` opts out. The archive is staged in the upload's
  temporary directory and always deleted afterwards. Server side, the
  contribution service accepts the new `workspace/*.zip` namespace with a
  1 GiB per-archive cap (trajectory JSONL keeps 128 MiB), allows at most
  one archive per capture and never an archive alone, verifies the zip
  container format instead of JSONL strictness, promotes it with an
  `application/zip` content type, and scopes trajectory-report
  cross-checks to trajectory artifacts so an attachment cannot fail
  report equality.

## 0.7.3 — 2026-08-16

### Added
- **`bench traj upload` waits out short rate-limit responses instead of
  failing.** When the contribution service answers 429 with a short
  `Retry-After`, the handshake now sleeps it out with jitter and retries up
  to three times (two-minute cap per wait), so a crowd of simultaneous
  contributors self-heals instead of surfacing errors. Longer waits still
  fail fast with the actionable retry-after message. (#1027)

### Changed
- **The contribution service rate-limits per contributor, not per venue.**
  Upload budgets are token buckets keyed on contributor identity with a
  wide per-IP abuse backstop, refill continuously, and answer 429 with
  seconds-until-next-token instead of the remainder of the clock hour, so
  many contributors behind one NAT no longer starve each other. Contended
  bucket updates back off with jitter rather than shedding simultaneous
  crowds. (#1027, #1028)

## 0.7.2 — 2026-08-16

### Added
- **The upload preview itemizes what redaction masked, by kind.** The
  redactor now categorizes every replacement by the rule that fired — API
  keys, bearer tokens, private key blocks, passwords, URL credentials, and
  credential-bearing field values — and the terminal trajectory report shows
  a `Masked for you: 2 API keys, 1 bearer token — originals never leave this
  machine` breakdown under the masked-count row, plus a reassurance that
  redaction ran locally and the server independently rescans staged
  artifacts (or `No secrets or personal identifiers detected — nothing
  needed masking.` when nothing matched). `bench traj upload --dry-run`
  prints the same breakdown as a plain `Masked for you:` line, and
  `bench eval view --confirm` gains a display-only `--redaction-summary`
  flag that renders it in the confirm bar next to the Approve button; the
  `benchflow-traj-upload` skill stages a dry run first and passes the line
  through. The total `redaction_replacements` count and the manifest
  `trajectory_report` contract are unchanged (the server validates the
  report with a closed schema and exact recompute equality, so per-category
  counts stay display-only).
- **The trajectory viewer can collect the eval-prize confirmation in the
  browser.** `bench eval view PATH --confirm` renders the normal page plus a
  sticky site-styled bottom bar ("Submit this trajectory to the BenchFlow
  eval prize?") with **Approve & submit** / **Not this one** buttons. A click
  POSTs to `/decision`; the server prints a machine-readable
  `DECISION: approved` or `DECISION: rejected` line to stdout, shuts down,
  and the CLI exits `0` on approve and `3` on reject (non-1/2 so rejection
  never collides with error or usage exits). Without the flag, behavior is
  unchanged (no bar, no endpoint, Ctrl+C to stop). The
  `benchflow-traj-upload` skill now prefers the button flow and falls back
  to chat confirmation on CLIs older than 0.7.2.
- **Trajectory uploads are tagged with the session's repository by
  default.** Unless `--source-id` is given, `bench traj upload` reads the
  session's recorded working directory (Claude `cwd` events, Codex
  `session_meta`), resolves its git `origin` remote, and stores
  `repo/<owner>/<name>` as the manifest source id, printing
  `Repo: owner/name (from session cwd /path; use --no-repo to omit)` (the
  local path is terminal output only, never uploaded). `--no-repo` opts out
  — the `benchflow-traj-upload` skill now surfaces the detected tag during
  the confirm step so contributors can decline it for private repos — and
  undetectable repos fall back silently to the path-derived source id.

### Fixed
- **The repo tag derives only from the session's own recorded cwd.** The
  initial repo-tagging implementation (#1015) fell back to the upload
  invocation directory's git remote when the session cwd yielded nothing; a
  collector-side audit showed this mis-attributes provenance — a session
  recorded in a non-repo directory, uploaded from the benchflow checkout,
  was tagged `repo/benchflow-ai/benchflow` (two community-dataset entries
  carry the mis-tag). The fallback is removed: no session cwd, a missing
  directory, or no GitHub remote now mean no repo tag, exactly like
  `--no-repo`.
- **The trajectory viewer header no longer shows `?` badges on real Claude
  Code sessions.** `bench eval view` on a `~/.claude` session JSONL rendered
  `model: ?`, `session: ?...`, `claude code: ?`, and `total cost: $0.0000`
  because the header only read a `type: system` event that real session
  files don't contain. The header now derives its metadata from what the
  file actually carries (first assistant event's `message.model`, per-event
  `version` / `sessionId`, the filename stem as a session fallback) and
  hides any badge whose value is unknown — including the cost badge when no
  event carries cost data. Presentation only.
- **Upload progress no longer claims the broker is waking up.** A warm
  retry printed `Uploading… the first request can take a minute while the
  service wakes up` even when the service was already up. The line is now
  `Uploading… this can take up to a minute; retries are safe.`
- **`bench traj setup --list` no longer wraps session paths mid-token.**
  Each hit prints index/source/time, then the path on its own line, then
  the snippet, via plain `print` so Rich does not split a long JSONL path
  and break copy-paste.

### Changed
- **Trajectory viewer tool calls are color-coded and backgrounds are light.**
  Each tool kind now gets a muted GitHub-label-style accent on its name pill
  and a left border strip on the card: shell/exec → amber, write/edit → blue,
  read → teal, agent/task/skill → purple, web/search/fetch → cyan, everything
  else → neutral gray. Tool arguments and tool outputs render on light
  surfaces (`#f5f5f5` / white with dark ink) instead of near-black blocks;
  the dark `#141414` terminal treatment is reserved for shell-command output
  only, and the ink-black result card stays as the deliberate bento-ink
  accent. Presentation only (CSS classes + a tool-name→accent mapping);
  content strings and behavior are unchanged. Applies to all three viewer
  templates, which share one stylesheet since #1019.
- **The contributor prompt now tells the agent to upgrade BenchFlow first.**
  `CONTRIBUTOR_PROMPT` (kept in sync in `README.md`, `docs/traj-upload.md`,
  and the `benchflow-traj-upload` skill evals) is a three-line block that
  says to run `uv tool install --python 3.12 --upgrade --force benchflow`
  before reading the skill, so agents that skim the skill or hit a stale copy
  still install the latest CLI. The prompt also names OpenCode and Cursor
  sessions and the re:Agent hackathon 72-hour window; README/docs render it
  as a blockquote (soft-wraps on GitHub) behind an explicit "send this to
  your coding agent" framing, and `bench traj setup` prints the same framing.
  The skill's Discover step gains best-effort Cursor
  (`~/.cursor/projects/*/agent-transcripts/`) and OpenCode
  (`~/.local/share/opencode/opencode.db`, `opencode db path`) locations plus
  a prefer-recent-matching-sessions note. Follow-up to the version
  precondition from #1013/#1014.
- **Trajectory viewer restyled to match www.benchflow.ai.** All three
  `bench eval view` pages (stream-json/JSONL, ACP events, multi-turn trial)
  now share one inline stylesheet with the site's design language: light
  monochrome palette (`#fafafa` page, white cards, `#0a0a0a` ink, dark
  `#141414` code blocks), Satoshi/Google Sans Code font stacks with
  system-safe fallbacks, mono pill badges, and a small BenchFlow wordmark
  header with the inline SVG logo. Pages remain fully offline (no external
  font or CDN requests) and content/structure semantics are unchanged;
  follows the contributor paste-line flow from #1013.

## 0.7.1 — 2026-08-16

### Added
- **`bench traj setup` / `bench traj upload` print an upgrade hint when
  outdated.** Both commands start with a lightweight PyPI latest-version
  check (2 s timeout, completely silent on any network or parse failure) and
  print a one-line `uv tool install --python 3.12 --upgrade --force
  benchflow` hint when the installed version is older than the latest
  release; dev/prereleases of a newer-or-equal base are not outdated.
  `BENCHFLOW_SKIP_UPDATE_CHECK=1` disables the check. The
  `benchflow-traj-upload` skill and docs now tell contributors to upgrade to
  the latest BenchFlow (0.7.1+) before using the trajectory-upload flow.

### Changed
- **Trajectory contribution is a copy-paste line, plus optional setup.**
  Contributors paste one line into their agent; the agent reads
  `benchflow-traj-upload`, opens the viewer, then uploads. Optional setup is
  `npx skills add benchflow-ai/benchflow --skill benchflow-traj-upload` or
  `bench traj setup`. `bench eval view` accepts a raw session JSONL file and
  does not write `trajectory.html` next to it. The CLI infers GitHub
  username and email from `gh` / `git` before prompting, prints `Submitted` /
  `Already submitted` plus a digest for public uploads (not a private Azure
  inbox URL), waits up to 90s for broker cold start, and treats Azure
  `403 UnauthorizedBlobOverwrite` as an idempotent skip. (#1008's interactive
  report, local secret masking to `<XXX-benchflow-key-values-XXX>`, schema-1.2
  manifest report binding, and byte progress are included; the PR #1008
  operator manual now lives at `benchflow-traj-upload-ops` so the public skill
  name stays contributor-facing.)

## 0.6.9 — 2026-08-15

### Changed
- **Trajectory uploads require contributor provenance.** The single
  `bench traj upload` command now requires `--github-id` and `--email`; both
  values are validated locally and server-side and stored in `manifest.json`
  under the structured `contributor` field. Existing 0.6.8 manifests remain
  readable by the validator during the client transition.

## 0.6.8 — 2026-08-15

### Added
- **Public trajectory contribution.** `bench traj upload PATH` validates and
  structurally redacts trajectory JSONL, creates a content-addressed manifest,
  and uploads through the built-in public broker. Azure Blob quarantine,
  versioning, short-lived create-only user-delegation SAS grants, and an
  event-driven fail-closed validator keep untrusted captures out of the
  community namespace until hashes, sizes, strict JSONL (including duplicate-key
  and non-finite-number rejection), and artifact/manifest secret scans pass.
  Replaying an ingested digest is a no-op; trusted operators can opt into direct
  Azure upload with the `azure` extra and `--direct`.

### Changed
- **BREAKING (task.md): the `environment:` frontmatter key is renamed to
  `sandbox:`.** The native task-config surface now accepts only `sandbox:`
  (plus `verifier.sandbox:` for the verifier's separate sandbox spec and
  `verifier.sandbox_mode:` for shared/separate selection); `environment:`,
  `verifier.environment:`, and `verifier.environment_mode:` no longer
  validate and fail with an actionable message naming the rename. **The
  one-line fix for existing task.md files is renaming the key.**
  Legacy/Harbor `task.toml` imports are unaffected: the toml loader
  converts `[environment]`, `[verifier.environment]`, and
  `environment_mode` to the `sandbox` spellings (declaring both spellings
  in one file is an error), and `bench tasks export` emits the inverse —
  a stock-Harbor `[environment]`-spelled `task.toml`. All native emitters
  — `model_dump_toml`, `bench tasks migrate`, task scaffolding,
  skill-eval/trace/adapter task generation, rubric-review wrappers — now
  write `sandbox`. Python API: the compat property
  `TaskConfig.environment` is removed (use `TaskConfig.sandbox`),
  `VerifierConfig.environment`/`environment_mode` became
  `sandbox`/`sandbox_mode`, and the `VerifierEnvironmentMode` enum is now
  `VerifierSandboxMode`. The Environment plane
  (`--environment-manifest`, `benchflow.environment.manifest`, the
  eval-config `environment:` docker/daytona selector) is a different
  subsystem and is unchanged.

### Fixed
- **Coherent integration coverage for code-and-fixture changes.** The
  credential-free fixture job now tests pull-request source together with its
  task fixtures, while the secret-bearing smoke job retains trusted fixtures;
  the release gate requires both results. (#969)
- **Retryable Daytona transport failures stay retryable.** Transient SDK
  transport errors are stamped while their vendor type is still available,
  empty exception messages retain useful type information, and permanent
  errors remain outside the retry policy. (#970)
- **Live token counters no longer freeze behind callback-log reads.** Larger,
  bounded ranged reads distinguish EOF from read failure, expose lag, and keep
  terminal phase labels from moving backwards. (#971)
- **Fresh OpenClaw installs and GPT-5.4 calls use compatible limits.** OpenClaw
  is pinned to a Node-compatible runtime, and raw plus ACP-alias GPT-5.4 model
  IDs clamp output tokens to the provider's 128,000-token limit. (#977, #986)
- **Anthropic Vertex routes include their required runtime.** The gateway now
  installs the Google Cloud Vertex AI SDK used by LiteLLM's Anthropic Vertex
  path. (#978)
- **ACP subprocess diagnostics survive normal teardown.** A bounded, redacted
  stderr tail is retained even when stdout remains open; stderr drain failures
  cannot mask the structured transport diagnosis, and repeated process close
  calls are idempotent. (#980)

## 0.6.7 — 2026-08-09

### Added
- **Built-in environment registry.** The committed env-axis pins
  (`env0@prod`, `env0@outage`) moved from `benchmarks/_environments/` into
  the package (`benchflow/environment/_registry/`) and ship inside the
  wheel, so `--environment-manifest env0@prod` resolves on a bare
  `pip install benchflow` with no checkout and no env vars.
  `$BENCHFLOW_ENV_REGISTRY`, when set, still wins entirely; resolution
  stays content-addressed (sha256 logged), and unknown names now error
  listing the available specs. (#961)
- **Console progress heartbeat.** Single-concurrency eval runs print a
  throttled progress line (`… 6.2min, 12 tool calls (last: …)`) about every
  45 seconds while the agent works, so a long prompt is distinguishable from
  a hang. The heartbeat is auto-gated off for multi-concurrency jobs;
  `bench eval run --quiet` suppresses it, and `BENCHFLOW_PROGRESS=on`/`off`
  overrides the auto-gate. (#951)
- **Live per-task activity in the eval dashboard.** Under a TTY the
  running-now table gains an activity column ("38 calls · last:
  file_editor", plus tokens once a usage snapshot exists), polled from the
  ACP session's existing heartbeat counters. The agents-manifest autoload
  also clones quietly — one "Cloning …" summary line instead of raw git
  progress. (#956)
- **Phase labels and per-task failure reasons.** The activity cell is never
  blank while a row exists: sandbox create, agent install, and verify show
  dim phase labels ("creating sandbox…", "installing agent…",
  "verifying…"). The final block prints one dim "✗ task: reason" line per
  failed task — verifier error first, else a compact reward/metric
  breakdown, else the reward — capped at 5 lines. `--quiet` now silences
  the dashboard as well as the heartbeat. (#957)
- **Failure reasons mined from verifier artifacts.** When a displayed
  failure would read as a bare "reward X", the CLI reads a small bounded
  artifact from the rollout's own verifier dir — the CTRF report (first
  failed test plus its assertion line) or a tail of `test-stdout.txt` — and
  a dim "(details: …/verifier)" pointer names the artifact directory.
  Artifacts resolve by the recorded rollout name, never by glob. (#959)
- **Fractional rewards in console summaries.** Per-task lines carry the
  scored reward (`✗ task (reward=0.30, tools=47)`), the Score line renders
  ", mean reward 0.30" alongside the binarized counts, and
  `EvaluationResult` / `summary.json` gain a `mean_reward` field (mean over
  scored rollouts; errors excluded, not zeroed). (#960)
- **Full failure counts on per-task console lines.** A CTRF report with
  more than one failed test rolls up as " (+N more failure(s); P/T checks
  passed)" after the first failure — the count suffix is never truncated
  away — and parametrized test names keep their `[param]` ids whenever the
  report carries them. (#962)
- **Live token usage in the dashboard footer.** The footer sums completed
  tasks' trusted telemetry plus every running rollout's live ACP session
  usage, so spend is visible while the run executes (usage lands per
  completed prompt; cost stays scoring-time from the gateway log).
  Single-tool agents' activity cell drops the redundant "last:" suffix in
  favor of "38 calls · 412.0k tok" once tokens are available. (#963)
- **env0-shaped failure breakdowns.** The failure-reason tiers understand
  metrics nested one level under `metrics` / `details`, pair
  `<name>_found` with `<name>_total` into fractions ("deadlines 1/5",
  lowest-signal first), probe `verifier/reward.json` on disk between the
  CTRF and stdout tiers, and every failure block with on-disk verifier
  artifacts gets one "(details: …)" pointer. (#964)
- **Mid-prompt live token usage via the gateway's live capture.** The #963
  live tokens stepped forward only when an ACP prompt completed, so a
  single-prompt rollout showed "— tokens" for its whole agent phase. The
  proxy runtime's existing live-capture loop (which already tails the
  sandbox gateway's callback log every second) now also accumulates
  provider tokens into an O(1) counter, and the dashboard reconciles the
  two non-decreasing live signals as max(ACP, gateway) — display-only,
  still replaced by the trusted scoring import at completion, and any
  gateway-side failure degrades to the ACP-only behavior. (#965)

- **Compact flow-style arrays in emitted task.md frontmatter.** Short
  scalar-only lists render on one line (`tags: [parsing, nlp]`) instead of
  multiline bullets, so hand-written flow arrays survive `bench tasks
  migrate` / normalize round-trips. The style predicate measures each list
  with the real YAML emitter (so it can never disagree with PyYAML's own
  quoting) and falls back to block style for long, nested, or multiline
  items. Parsing is unchanged — both styles were always accepted. (#967)

### Changed
- Migrated the clawsbench `archive-amazon-shipping` task to the native
  `task.md` package layout (`bench tasks migrate --remove-legacy`), and a
  clawsbench run launched without `--environment-manifest` now fails with an
  actionable verifier message naming the exact flag to pass instead of an
  opaque connection error. (#952)

### Fixed
- Isolated pull-request and post-test integration concurrency groups so a
  credential-free workflow completion cannot cancel an active PR rollout.
- **Host-side hard rollout deadline.** An await wedged below the phase-level
  watchdogs (e.g. a Daytona PTY teardown on a dead websocket) could freeze an
  entire eval job. Every rollout attempt now runs under a host-side hard
  deadline derived from the task's own phase budgets plus a fixed margin; a
  trip returns a retryable infra-error result and abandons the sandbox after
  a bounded cleanup grace. Override with `BENCHFLOW_ROLLOUT_HARD_DEADLINE`
  (seconds; `off`/`none`/`0` disables). Verifier timeouts that produced zero
  output are retried once — that signature is an exec-layer wedge, not a slow
  verifier. Agent-sent reasoning parameters are also forwarded through the
  LLM gateway, and DeepSeek routes natively with `reasoning_effort` passed
  verbatim. (#949)
- Quieter Daytona teardown: successful runs end at the score line — the
  atexit client cleanup no longer prints a cancelled-reader traceback after
  `✓ Score`, and benign engineio PTY-disconnect errors are silenced. (#951)
- Accept the Harbor 1.3 `[task] version` field in task configs. It is
  informational and stored verbatim; the strict schema previously rejected
  it, which made every Harbor-1.3 curated task unloadable. (#953)
- Synced the committed `env0@prod` / `env0@outage` environment pins with the
  upstream `benchflow-ai/env0` manifest (the pins had drifted to stale
  service CLI names, a phantom service, and shifted ports), and a manifest
  environment now fails loud when it declares services but none are startable
  in the image, instead of passing a vacuous readiness gate and dying at the
  verifier. (#954)
- Pointed the built-in `env0@prod` / `env0@outage` pins at the org-owned
  `ghcr.io/benchflow-ai/env0:0.2.0` base image. The wheel-shipped pins named a
  personal Docker Hub image while their own comments and the environment docs
  declared the ghcr image authoritative; `base_image` is recorded as rollout
  provenance, so every env0 result carried the wrong base. Verified on Daytona
  (env0 `auth-least-privilege-summary`, 8/8 services ready, reward 1.0).
- Restored the `env0@outage` perturbation (gmail and slack removed relative
  to `env0@prod`) that the #954 upstream sync had erased by mirroring the
  full service list into both pins, and corrected the pin header's stale
  slack port. (#955)
- **ACP protocol JSON glued to PTY shell noise now decodes.** On PTY
  transports, an agent's initialize response could arrive on the same line
  as the shell prompt (ANSI/OSC-prefixed), fail the strict whole-line JSON
  parse, and get filed as noise — the handshake then "timed out" at any
  window with the answer sitting in the agent log. The PTY-facing transport
  now retries from the first `{` and once more after an ANSI scrub, but
  only until the first successfully decoded protocol message per
  connection; afterwards the strict contract rules, so log-echoed envelopes
  cannot impersonate protocol traffic mid-session. The pre-prompt handshake
  window is also env-configurable via `BENCHFLOW_ACP_HANDSHAKE_TIMEOUT`
  (seconds; default 60). (#958)

## 0.6.6 — 2026-08-04

### Added
- **Apple Container and Amazon Bedrock AgentCore sandboxes.** Apple Silicon
  users can run supported single-container arm64 tasks through Apple's
  Virtualization.framework, while `--sandbox agentcore` builds task-specific
  AWS runtimes with lease-aware cleanup for supported public-network,
  single-container arm64 tasks. (#936, #937)
- **Rubric review (`bench review`).** Detached agentic grading of finished
  rollouts against a `rubric.json` — rubric contract **v0.1**: an object with
  a `criteria` list, each entry carrying a `name` (structured-output field), a
  `description` (author documentation, never shown to the reviewer), and
  `guidance` (the grading contract). The document carries no in-file
  version key. One
  reviewer agent per rollout runs as an ordinary rollout of a throwaway
  wrapper task on a digest-pinned multi-architecture base image and no
  task-authored Dockerfile
  (AgentCore still builds a derived runtime image), reads a read-only evidence
  copy of the rollout and, when admitted from a trusted `--tasks-root` with a
  verified digest, its task, and answers every criterion with `pass` / `fail` /
  `not_applicable` plus an explanation. The wrapper's own reward means only
  "the reviewer produced a structurally valid result"; graded outcomes land
  in `review_report.json`. Reviews never modify a reviewed rollout's rewards
  or `result.json`. Rubric resolution: `-r` > the task's own
  `verifier/rubric.json` > a built-in default (`reward_hacking`,
  `task_specification`). `--passing` / `--failing` filter the rollouts under
  review; a job-level prose summary aggregates multi-rollout runs. The
  default reviewer harness is `opencode` (pinned to `1.18.11`).
  Evidence mounts at `/evidence` outside the agent workdir (root-owned,
  unwritable by the reviewer), symlinks are dropped rather than
  dereferenced, task skills, shipped rubrics, and cumulative provider-history
  trajectories are excluded while the canonical ACP trajectory is retained;
  dropped ACP tool observations and generic tool titles are repaired from
  exact-ID events in the trusted provider capture,
  reviewer egress is gateway-scoped via the sandbox lockdown flag, artifact
  consumption is pinned to each invocation's unique runtime leaf, and the
  job summary is a deterministic aggregation.
- **Sealed AgentCore uploads.** Every AgentCore **upload and staged
  environment** is now encrypted end-to-end: the sandbox generates a
  keypair, only the public key appears in command output, payloads travel
  as AES-256-CTR ciphertext with an HMAC-SHA256 tag over IV and
  ciphertext (verified before decryption), and the decrypted key never
  appears in command text. Fixes provider credentials from
  `launch_config.json` and command environments being recoverable from
  the runtime's CloudWatch command log; the generated wrapper image
  installs `openssl` when missing. Downloads are not sealed: they return
  file contents as base64 through command output, so they must only carry
  non-secret run artifacts.
- **`RolloutConfig.uploads`.** Generic post-start host→sandbox uploads
  (directory or file → absolute sandbox path), used by rubric review to
  deliver evidence into prebuilt-image sandboxes and available to any caller
  whose task data is not baked into the image.
- **Native TRL tool-calling SFT export.** `bench train convert --format
  trl-sft` emits conversational prompt/completion rows with a `tools` column,
  excludes OpenCode title/summary helper calls, and accepts rollout trees,
  canonical `results.jsonl`, or existing TRL JSONL. `bench train validate
  --format trl-sft` can render rows with a pinned tokenizer and fail closed on
  missing assistant masks or overlength samples. Tokenizer-aware
  `--context-policy message-window` keeps the harness/task prefix and the
  longest complete recent assistant/tool suffix when exact rows exceed the
  student context window. (#925)
- **LLM call-purpose provenance.** Captured LLM exchanges retain provider/model
  metadata and classify agent, title, summary, compaction, and helper calls;
  `results.jsonl` carries the metadata on each trajectory step. (#925)
- **Live trajectory streaming and token logprobs.** Redacted LLM trajectory
  snapshots are written during active rollouts, and training runs can opt into
  sampled-token logprobs with `BENCHFLOW_CAPTURE_TOKEN_LOGPROBS=1`. (#922, #926)

### Changed
- OpenHands supports requested reasoning effort through its LiteLLM request
  body and now fails closed if completed provider exchanges are missing from
  trainer trajectories. The built-in OpenCode harness is version-pinned for
  reproducibility. (#921, #931)

### Fixed
- Restored the documented `test` → live `integration-light` → internal-preview
  publication chain for successful `main` pushes, pinned to the exact tested
  commit and fail-closed against untrusted workflow sources.
- Fixed OpenHands startup outside root-owned workdirs, OpenCode gateway setup
  in non-Python task images, Qwen3.5 TRL SFT tokenization, and incomplete agent
  skill-path validation. (#919, #924, #929, #932)
- Moved recursive review evidence work and blocking platform probes off the
  async event loop so concurrent reviews and rollouts retain responsive
  scheduling and timeouts.
- Remove staged Docker and Apple Container credential files when live process
  launch fails before the agent can source and unlink them.
- Corrected release-facing authentication, sandbox, trajectory-artifact,
  branching, and provider documentation to match the shipped interfaces.

## 0.6.5 — 2026-07-10

### Added
- **Reproducible evaluation artifact workflows.** `bench eval run` can emit
  task/run manifests, trajectory-health summaries, canonical one-rollout-per-task
  selections, materialized trainer inputs, repeated model matrices, and
  Hugging Face dataset uploads with source provenance. (#844, #907)
- **Prime-RL supervised fine-tuning integration.** `bench train validate` and
  `bench train run sft --backend prime-rl` add fail-closed trainer-row checks,
  local/Hugging Face dataset staging, publishing hooks, and compatibility
  controls for reproducing the Mobile300 PR828 training recipe. (#842–#865)
- **Native external benchmark adapters.** BenchFlow can materialize MCP Atlas
  and Toolathlon sources, credential-backed task packages, and the service
  sidecars required by their hosted runtimes. (#878, #885, #889)
- **Registry-driven agent extension.** Agent packages can autoload through
  `benchflow.agents` entry points or declarative manifests, including namespace
  shorthand for externally registered agents. (#873, #877)
- **BenchFlow-native GRPO/TRL pipeline.** Adds reusable `TaskRuntime`, the
  optional `BenchFlowSpec` TRL adapter, selective Hugging Face task snapshots,
  paired `bench eval compare-lift` reporting, and an end-to-end GRPO runbook.
  (#901–#907)

### Changed
- BenchFlow CLI and SDK installations now explicitly require Python 3.12 or
  newer, with `uv tool install --python 3.12` as the recommended CLI path.
  (#899)

### Fixed
- Hardened LiteLLM and provider routing across Responses/chat bridges, Gemini
  custom endpoints, Claude OAuth, Harvey LAB, OpenHands Azure reasoning effort,
  diagnostics, and parallel ACP shim startup. (#868, #871, #879–#881, #886,
  #888, #911)
- Stabilized Daytona execution for Toolathlon and ACP agents, including DinD
  retries, PTY handling, orphan-free long-command heartbeats, and Gemini's SSH
  transport fallback. (#890, #892–#896, #910, #912)
- Tightened trajectory and Prime-SFT artifact integrity with secret redaction,
  tool-call validation, repaired results conversion, and reproducible
  compatibility controls. (#849–#865)

### Removed
- **`BENCHFLOW_SKILL_NUDGE` skill prompt nudge.** The optional prompt injection
  that prepended mounted-skill names/descriptions/bodies to the task
  instruction (#207) is gone. Setting the environment variable now has no
  effect: prompt resolution never reads skill directories, and mounted skills
  reach agents only through their native skill paths. This keeps prompts
  identical across skill modes and closes off accidental prompt-level skill
  leakage. (#908)

## 0.6.4 — 2026-06-27

### Added
- **Environment and config as run-time axes on `bench eval run`.** `--state`
  binds the environment (S-axis) per run — inline JSON, a registry
  `name@version` resolved through the environment registry, or a manifest path
  (takes precedence over `--environment-manifest`). `--config-override` overlays
  the task config (C-axis) — inline JSON/YAML/TOML or `@file`, deep-merged into
  each task's resolved config. `--config` also gains a `--run-config` alias.
  (#790)
- **Content-addressed environment binding.** Registry environment resolution is
  content-addressed — `env_hash = sha256(manifest)` — so a `name@version`
  resolves to an exact, pinned environment that is recorded for replay; the
  C-axis `--config-override` is likewise persisted with its content hash and the
  applied patch. Every rollout is attributable to the precise world and config
  it ran against. (#790)
- **MLE-bench adapter.** Adds an MLE-bench benchmark adapter, parity fixture, and
  task plumbing for running and auditing MLE-bench through BenchFlow. (#792)
- **Agent adapter skill.** Adds the canonical adapter skill under `.agents/skills`
  for harness-side adapter work. (#793)
- **Prime-RL SFT export.** Adds `bench train convert prime-sft` support for
  exporting BenchFlow trajectories into Prime SFT-ready JSONL artifacts. (#828)

### Changed
- **`bench continue` is now `bench eval continue`.** The command (and its
  `continue-batch` companion) moved under the `eval` group, where it is now
  discoverable in `bench eval --help` alongside `run`/`adopt`. The original
  top-level `bench continue` / `bench continue-batch` remain as hidden,
  deprecated aliases (they print a deprecation notice) so existing scripts keep
  working. (#800)
- **Routable agents always go through the LiteLLM usage proxy.** OpenCode-family
  and pi-acp model calls now stay on the proxy path so token usage, cost, and
  trajectory capture are preserved consistently. (#797, #803, #820)
- **Agent manifest loading is now the additive decoupling path.** The core agent
  manifest loader and Omnigent/session-factory seam are gated in while preserving
  existing ACP manifests and byte-identical parity coverage. (#825, #836, #837)

### Fixed
- Resolved the sharded and run-config paths so the S-axis environment and C-axis
  config overlay are applied consistently in `bench eval run`. (#804)
- Added `bench eval run --context-root` plumbing and early validation for missing
  paths. (#816)
- Fixed verifier-error resume logging and streaming `claude-agent-acp`
  trajectory emission so failed or streamed runs retain the expected evidence.
  (#819, #839)
- Resolved bare model IDs to their provider, avoided pi-acp context-window retry
  storms, and kept provider failure causes visible while preserving redaction.
  (#805, #831, #834, #835)
- Preserved Codex subscription-auth behavior and auth-file permissions in the
  launcher path. (#825)
- Rejected `.git` and `file://` source paths with clear errors. (#822)
- Hardened experiment-review and integration gates around missing trajectories,
  summaryless roots, file-editor false positives, and L3 review calibration.
  (#802, #806, #807, #808, #809, #810, #811, #812, #814, #817, #821, #823, #824)

## 0.6.3 — 2026-06-16

### Changed
- **`bench eval create` renamed to `bench eval run`.** The verb now matches what
  the command does (it runs an evaluation, single task or batch). `bench eval
  create` stays as a deprecated alias that prints a deprecation notice on use, so
  existing scripts, YAML configs, and downstream repos (e.g.
  `benchflow-ai/skillsbench`) keep working unchanged. Switch to `bench eval run`.
- **`task.md` is now the sole task authoring format.** `bench tasks init`
  scaffolds a native `task.md` package (`task.md` + `environment/` + `oracle/` +
  `verifier/`). `bench tasks init --format legacy` is retired and now exits with
  an error pointing at `bench tasks migrate <dir> --remove-legacy`. Existing
  split-layout packages remain readable, and `bench tasks migrate` /
  `bench tasks export` continue to cover the migration and compatibility paths.
  Authoring docs now lead with `task.md`; the split layout is documented only as
  a migration/export target.

### Fixed
- `bench skills eval` now exits non-zero when any eval case errors (e.g. missing
  credentials), matching `bench eval run`. A 100%-error run printed `0/1`
  but exited `0`, so CI/scripts read a total failure as success.
- The "task.md already exists" migrate error now names both surfaces
  (`--overwrite` for the CLI, `overwrite=True` for the Python API) instead of
  only the API kwarg.
- `bench eval view <job-dir>` no longer shows a blank "No trajectory files
  found" when given a job directory (the natural value from `eval run`'s
  "Artifacts:" line) — it now indexes the rollout subdirectories to drill into.
- `bench hub env list` prints a footer (`Showing N…`) with how to refine
  (`--search`/`--owner`/`--limit`/`--json`), so a small page of a large catalog
  no longer reads as "the provider only has N environments".
- `bench tasks check --level publication-grade` errors for a missing verifier
  package now include a remediation hint (author `verifier/verifier.md`; note
  that `bench tasks migrate` does not generate it), instead of a dead-end.

## 0.6.0 — 2026-06-13

### Added

- **The `task.md` task standard** — a single-file unified task format (parser,
  verifier planes, prompt sidecars, round-trip export with a machine-readable
  loss report) plus the authoring CLI: `bench tasks init / check / migrate /
  export`, with a layered `check --level` ladder up to a leaderboard-grade
  acceptance gate. See [`docs/task-standard.md`](docs/task-standard.md) and the
  [native authoring guide](docs/task-authoring-task-md.md).
- **`bench eval adopt` benchmark-adoption router** — `init` scaffolds a benchmark
  conversion per [`benchmarks/CONVERT.md`](benchmarks/CONVERT.md), `convert` drives
  the host `codex` CLI through the conversion workflow, and `verify` runs the
  parity gate (deterministic per-criterion conversion parity plus the
  agent-scale reward-distribution layer) and emits a confidence verdict, with a
  drafted support issue on divergence. `bench eval adopt verify --rerun`
  independently re-executes the benchmark's `parity_test.py` and scores its fresh
  output (instead of trusting the recorded `parity_experiment.json`), failing
  closed if the output is not scoreable; `bench eval adopt convert -c key=value`
  passes codex config overrides through to the host codex driver (e.g. to work
  around `~/.codex` drift). `bench tasks digest` recognizes native `task.md` tasks
  as well as legacy `task.toml`.
- **ATIF and ADP trajectory artifacts** — every scored rollout now emits
  `trainer/atif.json` and `trainer/adp.jsonl` (alongside the existing
  `verifiers.jsonl`), with job-level ADP aggregation. One canonical raw
  trajectory, multiple ecosystem formats out of the box.
- **OpenReward (ORS) reward-format interop** — export BenchFlow rewards in the
  Open Reward Standard shape (`benchflow.adapters.ors`) and the `ors-episode`
  verifier strategy is recognized. (The hosted-environment episode runner that
  executes ORS environments end-to-end is in progress, not in this release.)
- **Daytona sandbox auto-reap** — orphaned sandboxes are cleaned at eval start
  (TTL-tiered; failure states reaped sooner; an idle-activity guard protects
  live runs), gated by `BENCHFLOW_DAYTONA_AUTO_REAP` (any of `0`/`false`/`no`/
  `off`, case-insensitive, disables it).
- **Registry-pinned dataset runs** — `bench eval create -d name@version`
  (e.g. `-d skillsbench@1.1`) resolves a dataset from a git-backed
  `registry.json` (see skillsbench `docs/dataset-versioning.md`): tasks are
  cloned at their pinned `git_commit_id` into `.cache/datasets` and every
  task directory is verified against its sha256 content digest before
  anything runs; the entry's `bench_version` range is checked against the
  installed benchflow. `--registry` overrides the default (skillsbench)
  registry. `result.json`/`config.json` are stamped with `dataset_name`,
  `dataset_version`, and a per-task `task_digest` (`summary.json` carries
  the name/version); `--tasks-dir` dev runs carry no dataset fields but
  still stamp a live-computed `task_digest`, so every trajectory stays
  attributable to exact task content. `bench tasks digest <dir>` prints
  the digest for task authoring, and `check_results.py` audits the stamps.
  See [`docs/running-benchmarks.md`](docs/running-benchmarks.md). (#689,
  #690, #691; `packaging` promoted to a core dependency for the
  `bench_version` check.)
- **`benchflow continue <run-folder>`** — resume a previous, unfinished
  (timed-out) `openhands` run to completion. A standalone tool (it does not
  touch the normal run path) that reconstructs the run's exact workspace and
  agent memory from the recorded `llm_trajectory.jsonl` via record-replay,
  then continues with the live model — no injected prompt — and writes a new
  HF-compatible folder with `continued_from` provenance. See
  [`docs/continue-runs.md`](docs/continue-runs.md).

### Changed

- `bench metrics` → `bench eval metrics` and `bench view` → `bench eval view`
  (the deprecated hidden top-level forms are gone; use the `eval` subgroup).
- Quickstart and CLI reference now match observed run behavior — the real jobs
  directory layout and artifact map, the `<PROVIDER>_API_KEY` /
  `<PROVIDER>_BASE_URL` convention, and exit-code semantics.
- Document the public vs internal preview install/upgrade command matrix,
  including `uv tool` exact pins, internal preview upgrades, and the
  `--force` path for replacing stale entrypoint scripts.

### Renamed (aliased; old names removed in 0.7)
- Benchmark adoption is now `bench eval adopt {init,convert,verify}`. It lives
  under `eval` because `eval` is the universal benchmark entry point (`eval
  create` runs a benchmark; `eval adopt` makes a foreign one runnable). Two prior
  spellings remain as hidden deprecated aliases, each printing a one-line stderr
  notice pointing at `bench eval adopt`: the original `bench agent
  create|run|verify`, and the 0.6-dev intermediate top-level `bench adopt`.
  `bench agent` now means agent management only (`list` / `show`).
- The overloaded `bench environment` group was split and is now a hidden
  **deprecated alias group** (removed in 0.7): the local sandbox lifecycle moved
  to `bench sandbox {create,list,cleanup}`, and hosted-provider browsing to
  `bench hub env {list,show,inspect}`. The old `bench environment
  create|list|cleanup|show|inspect` (plus `list --provider`/`--hub`) still work,
  each printing a one-line stderr deprecation notice. The hosted *run* path stays
  on `bench eval create --source-env`.

### Removed
- **Removed the unwired `OTelCollector`** (`benchflow.OTelCollector` /
  `benchflow.trajectories.OTelCollector`) and its `trajectories/otel.py` module.
  It was a designed-but-never-wired OTLP receiver from the v2 rewrite — never
  instantiated, never tested, and not part of any run path (BenchFlow captures
  trajectories via ACP session events and the LiteLLM callback path instead).
  This drops it from the public `__all__`; re-add it (with a test + real wiring)
  if OpenTelemetry-based capture is revived.
- Removed two unimplemented stub methods (`read_file`, `write_file`) from the
  `@runtime_checkable` `Sandbox` Protocol. No backend implemented them (backends
  expose the `upload_file`/`download_file` family) and there were no call sites,
  so they were a latent `isinstance` trap on the contract surface.
- Dead-code purge, round 3 (no public-API impact; each symbol re-verified
  zero-reference with class context): removed `TaskMetrics.audit_outcome`,
  `OTelCollector.endpoint`, `ReplayRouter.cursor`, `RuntimeResult.to_run_result`
  (legacy SDK-compat converter, unused), the never-read dataclass fields
  `ToolCall.output` and `JudgeConfig.{reference, prompt_template}`, the write-only
  `ReplayProxy._host`, the inert `AgentProtocolError.code` annotation, and an
  unused `retry_if_exception_type` import + fallback in `sandbox/daytona.py`.
- Dead-code purge, round 2 (no public-API impact; each symbol adversarially
  verified zero-reference with class context): removed seven unused `*_path`
  `@property`s from `TaskPaths`/`RolloutPaths` (`readme_path`, `gitignore_path`,
  `verifier_document_path`, `artifacts_manifest_path`, `result_path`,
  `exception_message_path`, `log_path`), the vestigial `ModalSandbox.supports_gpus`
  / `can_disable_internet` capability properties (not on the Sandbox Protocol),
  an unused module-level `logger` in `cli/continue_cmd.py`, and the orphaned
  `mcp_service_hooks_from_config` helper.
- Dead-code purge (no public-API impact unless noted): removed the unused
  `job_config_from_yaml` helper, the nominal `TASK_REPOS` back-compat dict
  (use `TASK_ALIASES`), the `_looks_like_verifier_dep_install_error` shim
  (use `contains_verifier_dep_install_marker`), the unused `parse_binary_verdict`
  reward helper (use `parse_verdict`), the dead `SandboxBackend` type alias,
  an unused `StdioTransport._read_buffer` field, and 12 redundant `rollout`
  package re-export aliases (submodule definitions unchanged).
- Removed the deprecated, hidden `benchflow skills install` CLI command. The
  SDK function `benchflow.skills.install_skill` is unchanged.
- Retired the deprecated top-level legacy CLI (`cli/legacy.py`). The dead
  0.3-era `job`/`agents`/`eval` commands are removed; `metrics` and `view` are
  promoted to first-class `bench eval metrics` / `bench eval view`; and the
  redundant `cleanup` command is dropped in favor of the existing
  `bench environment cleanup`.
- Removed the `experiments/` research/dev tooling tree (never shipped in the
  wheel) and its 6 dependent test modules, completing the dev-tree cleanup
  alongside the earlier `dashboard/` removal and `labs/` → `docs/labs`
  migration. Benchmark result files were preserved out-of-tree, not deleted.

### Fixed
- **CLI errors now go to stderr.** `print_error` (the single CLI error sink) wrote
  to stdout, so a `bench … --json | jq` pipeline could get a non-JSON error line on
  the JSON channel. All CLI errors (and the dataset bench-version remediation hint)
  now route to stderr; exit codes are unchanged, so failures stay detectable.
- **`bench hub env list --json` now emits valid JSON at any width.** The raw
  payload was printed through Rich's console, which soft-wrapped long strings and
  injected literal newlines mid-value (unparseable JSON when piped). It is now
  written verbatim.
- **No more raw tracebacks on bad input.** Hardened the unguarded front doors a
  stress sweep surfaced: `eval create --source-repo` clone failures and
  `--tasks-dir <file>`; `eval view` on corrupt/partial trajectory artifacts
  (`prompts.json`, a bad `acp_trajectory.jsonl` line, `result.json`, a null
  `session_id`); `sandbox create` with an unknown `--sandbox` backend or a missing
  optional sandbox dependency; `tasks digest` on an unreadable file (single = clean
  error, batch = warn-and-skip); and `hub check` with a malformed/missing
  `--registry` (now a user-meaningful message, not a raw `JSONDecodeError`/`OSError`).
- **Markup-safe output.** User/author-controlled strings that look like Rich markup
  no longer crash or silently garble output: `eval list` job names, `eval metrics`
  title, `skills list` cells, and `tasks init`'s reported path are now escaped.
- **`skills eval` schema errors** no longer leak pydantic internals (private model
  name, `[type=…]` tags, the pydantic.dev URL) — just the actionable per-field text.
- **`bench environment` deprecation notice** now fires exactly once (one line,
  once per process) instead of doubling up with Typer's generic
  `DeprecationWarning`, and its aliased verbs are hidden from `--help`, matching the
  `agent` / `eval adopt` alias families.
- `benchmarks/CONVERT.md` now references the canonical `bench eval adopt verify`
  (was the deprecated `bench agent verify`) in the conversion prompt.
- `bench tasks migrate` emits minimal, canonical (`schema_version`) front
  matter instead of a full defaults dump.
- Verifier `timeout_sec` is validated as a positive, finite budget
  (fail-closed at parse time; omission inherits the documented default).
- Docker `compose up` retries on the daemon network create/attach race.
- Console error messages truncate at word boundaries instead of mid-token.
- Recorded sandbox-setup timeouts and trajectory artifacts are consistent
  across the Docker and Daytona backends.
- The `task.md` init scaffold is agent-neutral, so `--agent oracle` works on a
  freshly scaffolded task.
- `gemini/`-prefixed judge/simulated-user models now resolve to the Google
  backend instead of passing the slashed name through and 404-ing.
- Model-backed judges raise a clear error naming the provider and pointing at
  `pip install benchflow[judge]` when the judge SDK is missing, instead of the
  misleading "Missing OPENAI_API_KEY" fall-through.
- `bench tasks check` recognizes a rubric-backed `llm-judge` verifier as a valid
  entrypoint and no longer demands a `test.sh`.
- Pre-verifier disk reclaim is workspace-aware and symlink-safe: it rejects
  symlinked cache candidates and realpath-guards every deletion against the
  workspace and `/logs`, so an agent-planted `~/.cache` symlink cannot steer the
  reclaim into workspace or output state (#601).
- Bedrock Claude 4.8+ routes fail closed when LiteLLM's adaptive-thinking patch
  is inactive, instead of silently sending a request the proxy cannot satisfy
  (#602).

## 0.5.2 — 2026-06-05

### Changed

- **PyPI project README badge** — replace the dynamic PyPI version badge with
  a stable package badge so the rendered project description cannot show a
  stale external version image after a public release.
- **Release documentation refresh** — update public install snippets,
  release-channel docs, examples, and citation metadata to `0.5.2`.

## 0.5.1 — 2026-06-05

### Added

- **Daytona usage telemetry by default** — Daytona runs now start a sandbox-local provider usage proxy so token/cost telemetry works without an external tunnel; use `--usage-tracking off` to bypass proxying when needed.
- **Azure AI Foundry providers** — new `azure-foundry-openai/` and `azure-foundry-anthropic/` prefixes routing through Foundry's unified resource. Export `AZURE_API_KEY` plus `AZURE_API_ENDPOINT` (e.g. `https://<resource>.openai.azure.com/`); benchflow derives the resource name from the endpoint host, builds the per-surface base URL, and maps the key onto the agent-native auth env automatically. Missing/unrecognized endpoints and unsupported agent/provider protocol pairings fail fast with clear errors instead of falling through to the wrong endpoint.
- **Azure Foundry auth guidance** — agent discovery output and docs now call out that provider-prefixed models can use provider-specific credentials instead of the agent's native/default API key.

### Changed

- **PyPI project documentation refresh** — the public package README, install snippets, release-channel docs, examples, and citation metadata now point at `0.5.1`.

### Fixed

- Inherit `BENCHFLOW_PROVIDER_BASE_URL` / `BENCHFLOW_PROVIDER_API_KEY` from the host environment so self-hosted / OpenAI-compatible endpoints route correctly instead of falling back to `api.openai.com`; empty or whitespace-only host values are skipped so they cannot shadow the resolved provider URL (benchflow-ai/skillsbench#817).

## 0.5.0 — 2026-06-04

### Added

- **Public/internal preview release channels** — tag-driven public releases publish stable PyPI packages and GitHub Releases; merges to `main` publish internal preview `.devN` packages after CI passes.
- **v0.5 integration evidence** — release validation docs now cover urgent blocker closure, SkillsBench infra-fix validation, adapter evidence, trace-to-task evidence, hosted env compatibility, and diagnostic fields.
- **Release automation guardrails** — public release tags must point at commits contained in `main`, version tags must match `pyproject.toml`, and PyPI publishing uses Trusted Publishing/OIDC instead of stored tokens.

### Changed

- `main` now tracks the next public version as `0.5.1.dev0`; the published public SDK is `0.5.0`, and internal previews are emitted as `0.5.1.dev<N>`.
- Documentation now directs downstream users to depend on public PyPI releases by default and use prerelease-enabled internal previews only for validation before the next public cut.

### Fixed

- Closed the v0.5 release blocker set covering structured sandbox/verifier diagnostics, Daytona startup/export retries, verifier dependency classification, CTRF path consistency, and SkillsBench task compatibility evidence.

## 0.3.3 — 2026-05-15

### Added

- **Harvey LAB benchmark** — converter, agent shim, and parity validation for 1,251 legal AI tasks (#239).
- **Harvey LAB Claude Sonnet judge** — switched verifier from Gemini to `claude-sonnet-4-6`, matching the original benchmark default (#264).
- **ProgramBench integration** — new benchmark adapter; TB2 removed; `.ref/` migrated to `benchmarks/` (#237).
- **CLI progress output** — `bench eval create` / `bench run` now show progress messages by default (#264).
- **Skill nudge** — optional prompt injection for skill-enhanced agent runs (#207).
- **Self-generated skill mode** for Codex agent (#233).
- **Integration test suite** for ENG-6 + `OPENAI_BASE_URL` inheritance fix (#255).
- **Modal backend support** — Dockerfile compatibility for Modal environments.
- **CITATION.cff** (#246).
- **`AGENTS.md`** — canonical contributor guide; `CLAUDE.md` deprecated (#258).

### Changed

- **Two-field source pattern** for dataset sourcing (#252).
- **Docs overhaul** — synced from www.benchflow.ai; Mintlify config added then orphaned config removed (#259, #257, #226).
- **`uv sync`** for package management (#232).

### Fixed

- Prevent `TypeError` in `metrics.collect_metrics` when reward is `None` (#243).
- Copy eval `requirements.txt` into Docker build context (#245).
- Resolve agent aliases in `bench agent show` and display aliases in `bench agent list` (#251).
- Guard ACP transports against JSON scalar logs (#236).
- Agent timeout reward fallback for Codex (#234).
- Isolate JS agent runtime installs (#231).
- Route Codex ACP through responses API (#224).
- Deploy skills and forward `solution.env` for oracle runs (#223).
- Honor no-internet tasks for agent runs; disable web tools without prompt mutation (#215).
- Propagate `OPENAI_API_KEY` for vllm provider (#3).
- Preserve arrival order of thought/message within flush windows (#214).
- Record user messages and per-turn agent text in ACP trajectory (#745).
- Chown skill-link parent dirs so sandbox user can write into them.
- Dynamic `--rootdir` in `PYTEST_ADDOPTS` based on task workspace.
- Unique env-file path in `DaytonaPtyProcess` to avoid race conditions (#200).

## 0.2.3 — 2026-04-15

### Added

- `benchmarks/tb2_multiturn-claude-haiku45.yaml` — shipped config for the README's TB2 multi-turn Claude result.
- Daytona resource clamping via `BENCHFLOW_DAYTONA_MAX_CPUS` / `MAX_MEMORY_MB`.

### Changed

- Renamed `skillsbench-claude-glm5.yaml` → `skillsbench-claude-glm51.yaml` to match the model ID.
- `codex --login` correction in `docs/getting-started.md`.
- Restricted sdist build to `src/`, `tests/`, and metadata.

### Fixed

- Verifier sandbox hardening follow-ups across several base-image and tooling edge cases.
- Preserve trusted verifier path entries and workspace answer files.
- Redirect oracle output to container log.
- Align YAML path resolution to config file location.

## 0.2.2 — 2026-04-13

### Added

- **Sandbox hardening tiers 1–3** — layered defense (env scrubbing, path lockdown, workspace
  freeze, wider snapshot, oracle privilege drop) blocking F1–F6 red-team findings.
- **`labs/reward-hack-matrix`** — per-trial timeout support and 0.2.2 sweep handoff scripts.

### Fixed

- Multiple sandbox bypass vectors identified in red-team testing.

## 0.2.1 — 2026-04-12

### Added

- **Sandbox hardening on by default** — `sandbox_user` now defaults to `"agent"` (was `None`/root). Blocks conftest-hook and answer-lookup exploit patterns.
- **Path lockdown** — new `sandbox_locked_paths` parameter makes `/solution` and `/tests` read-only before the verifier runs, blocking `.pth`-injection and similar pre-verify tampering.
- **Verifier failure isolation** — agent errors and verifier errors are now stored separately; a crashing verifier no longer masks the agent result.
- **`labs/benchjack-sandbox-hardening`** — cookbook demonstrating three exploit patterns (P1 conftest-hook, P2 answer-lookup, P7 `.pth`-injection) and their defenses.

### Fixed

- **Oracle runs as `sandbox_user`** — oracle agent now respects path lockdown instead of running as root and bypassing it.
- **Multi-endpoint provider routing** — providers with multiple endpoints now route by the agent's native API protocol.
- **Stale API key shadowing subscription auth** — emits a warning when `ANTHROPIC_API_KEY` env var is present alongside `claude login` credentials.
- **pytest `ini`-injection bypass** — closed a verifier hardening edge case.

### Changed

- Version is now single-sourced via `importlib.metadata`; no more duplicate version string in `__init__.py`.
- **User-facing docs** — new `docs/` directory with getting-started guide, CLI reference, architecture overview, task-authoring guide, and labs index. README trimmed; detailed content moved to `docs/`.

## 0.2.0 — 2026-04-09

**First public release.** A near-complete rearchitecture from the 0.1.x era. API surface has changed — assume breaking changes. Future releases will maintain compatibility within the 0.2.x line. 0.1.x users should treat this as a fresh install; see `.dev-docs/sdk-reference.md` for the new SDK.

### Added

- **Multi-agent, multi-provider, multi-auth matrix** — one YAML config, any supported agent × model × provider × auth combination.
- **Subscription auth support** — use `claude login`, `codex --login`, `gemini` OAuth credentials directly. No API keys required for host-based agent workflows.
- **Vertex AI support** — ADC auth for `google-vertex/`, `anthropic-vertex/`, `vertex-zai/` prefixed models.
- **Provider registry** — add a new LLM endpoint via a dict entry in `providers.py`, no code changes.
- **`benchmarks/` directory** with reusable YAML configs and runner scripts for TB2 and SkillsBench.
- **Auto task download** — YAML configs reference datasets as `org/repo/path` (e.g. `harbor-framework/terminal-bench-2`). Repos are cloned on first use and cached under `.cache/datasets/`.
- **`benchflow tasks init`** — scaffold new tasks.
- **`benchflow tasks check`** — validate task structure.
- **`benchflow cleanup`** — delete old sandboxes with `--max-age` filtering (default 24h).
- **Oracle agent support** — run `solution/solve.sh` directly for task validation.
- **Hello-world-task example** for sanity-testing the agent pipeline.
- **Model generation params** via env vars (`BENCHFLOW_TEMPERATURE`, `BENCHFLOW_TOP_P`, `BENCHFLOW_MAX_TOKENS`).
- **OpenClaw ACP shim** with trajectory parsing and skills support.
- **ACP trajectory capture** — full multi-turn agent trajectories via ACP protocol.

### Changed

- **Skill loading** — agent-targeted with proper precedence; auto-distributed from `task.toml` `skills_dir`.
- **`openclaw-gemini` merged** into `openclaw` — provider mode selected at runtime via `BENCHFLOW_PROVIDER_NAME`.

### Fixed

- **API keys leaking in `ps aux`** — env vars now written inside the container instead of passed via Docker exec `-e`.
- **Subscription auth skipped without `-m`** — `benchflow run` without `--model` now checks correctly.
- **ADC credentials break with `sandbox_user`** (#111) — credentials written to sandbox user's home instead of `/root/`.
- **Daytona sandboxes not cleaned up** (#102) — auto-delete after max age.
- **`benchflow cleanup` ignoring `--max-age`** — was deleting everything regardless of age.
- **readline buffer overflow crashes trial** (#98).
- **OpenClaw ACP shim loses tool command text** (#96).
- **OpenClaw ACP shim hardcodes `anthropic/` prefix** (#95) — now routes correctly for Gemini/GLM models.
- **Oracle agent `PermissionError`** writing `agent/oracle.txt` (#91).
- **Oracle path skips `pre_agent_hooks`** (#92) — services now start before oracle runs.
- **Trial data parity with Harbor** (#90) — richer `result.json`, agent logs, per-phase timing.
- **`SDK.run()` `PermissionError`** — `jobs_dir` subdirectories created as root (#88).
- **Partial trajectory lost on timeout** — saved before timeout raises.
- **Redundant `--version` binary check** removed — was wasting 30s per trial.
- **Trajectory fallback** — scrapes agent-native files when ACP `session/update` is empty (#94).
- **`litellm` upgraded to 1.83.0** for CVE-2026-35030; transitive dep security alerts resolved (13 Dependabot alerts closed).

### Deprecated

- `BaseAgent` re-export — planned removal in 0.3.0
- `Trial` re-export — planned removal in 0.3.0
