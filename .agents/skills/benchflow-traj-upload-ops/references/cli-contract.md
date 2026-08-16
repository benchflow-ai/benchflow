# `bench traj upload` CLI contract

This reference describes the current user-visible and operational behavior of
`bench traj upload`. Use it to operate the command, review changes, diagnose
failures, and distinguish local validation from a production upload.

## Contents

- [Command surface](#command-surface)
- [Prompt and confirmation rules](#prompt-and-confirmation-rules)
- [Destination selection](#destination-selection)
- [Input discovery and validation](#input-discovery-and-validation)
- [Local redaction](#local-redaction)
- [Trajectory report](#trajectory-report)
- [Contributor metadata and manifest](#contributor-metadata-and-manifest)
- [Dry-run behavior](#dry-run-behavior)
- [Upload progress and results](#upload-progress-and-results)
- [Public upload protocol](#public-upload-protocol)
- [Trusted direct upload](#trusted-direct-upload)
- [Failure behavior](#failure-behavior)
- [Verification levels](#verification-levels)
- [Implementation map](#implementation-map)

## Command surface

```text
bench traj upload [PATH]
  [--github-id TEXT]
  [--email TEXT]
  [--source-id TEXT]
  [--repo | --no-repo]
  [--direct]
  [--container-url TEXT]
  [--dry-run]
  [--preview-steps INTEGER]
  [--wait | --no-wait]

bench traj status [DIGEST]
```

| Input | Contract |
| --- | --- |
| `PATH` | One JSONL file, a directory containing JSONL files, or a BenchFlow trial directory containing `trajectory/`. |
| `--github-id` | Contributor GitHub username without `@`; 1-39 characters, GitHub-compatible placement of hyphens, and no consecutive hyphens. |
| `--email` | Bounded ASCII contributor email, stored in `manifest.json`. |
| `--source-id` | Optional stable source identifier. If omitted, derive it from the selected file stem or directory name. Accept 1-128 letters, numbers, dots, underscores, hyphens, or single path separators; reject `.`/`..` segments and repeated separators. |
| `--repo` / `--no-repo` | Repo tagging, on by default. When no `--source-id` is given, read the session's recorded working directory (Claude `cwd` events or the Codex `session_meta` payload), resolve its git `origin` remote (2 s timeout, silent failure), normalize https/ssh URLs to `owner/name`, and use `repo/<owner>/<name>` as the source id, printing `Repo: <owner>/<name> (from session cwd <path>; use --no-repo to omit)` (the local path is terminal output only, never uploaded). The session's own recorded cwd is the only provenance source — there is no fallback to the upload invocation directory. No recorded cwd, a missing directory, or no GitHub remote mean no tag: the upload silently keeps the derived source id, exactly like `--no-repo`. Local-path remotes never produce a tag; an explicit `--source-id` always wins. |
| `--preview-steps` | Number of meaningful redacted steps to show; range 0-20, default 5. Zero suppresses the preview table. |
| `--dry-run` | Validate, redact, report, and finalize a temporary manifest without confirmation or network access. |
| `--direct` | Use trusted local Azure credentials instead of the public contribution service. |
| `--container-url` | Azure container URL for `--direct`; invalid without `--direct`. |
| `--wait` / `--no-wait` | Storage verification, on by default for public uploads: after the byte transfer, poll the broker's capture-status endpoint until the validator's verdict. `BENCHFLOW_TRAJ_WAIT_SECONDS` overrides the polling budget (default 240; `0` disables the wait entirely). Never applies to `--direct` or `--dry-run`. |
| `DIGEST` (`traj status`) | `sha256:<64 hex>` or the bare 64-hex digest printed by an upload; prompted for when omitted. |

Use `uv run bench` in a source checkout and `bench` for an installed release.

## Prompt and confirmation rules

1. Classify the run as interactive if any of `PATH`, `--github-id`, or
   `--email` is missing.
2. Prompt for a missing path first because inspection must finish before
   contributor metadata is requested. Strip matching outer single or double
   quotes and expand `~`; retry until the path exists.
3. Stage, redact, analyze, and render the report.
4. Prompt for only the missing GitHub ID and email. Retry invalid values with a
   concise validation error.
5. Finalize the manifest.
6. Return immediately for `--dry-run`.
7. For an interactive real upload, ask `Upload this trajectory?` with No as the
   default. Declining prints `Upload cancelled.` and exits successfully without
   network access.
8. When all three required inputs were supplied on the command line, render the
   report and upload without a confirmation prompt.

The ordering is intentional: the user sees the report before providing missing
identity fields and before authorizing a network write.

## Destination selection

Resolve the destination before staging:

- Normal mode uses the built-in public contribution destination. The
  `BENCHFLOW_TRAJ_BROKER_URL` environment variable is only an override for
  development or disaster recovery; do not ask normal contributors to set it
  and do not print its value.
- `--container-url` without `--direct` is an error.
- `--direct` requires either `--container-url` or
  `BENCHFLOW_AZURE_CONTAINER_URL`.
- `BENCHFLOW_TRAJ_UPLOADED_BY` is an optional operator label included in the
  manifest. It is not a substitute for GitHub ID and email and should not be
  requested from normal contributors.

## Input discovery and validation

Accept exactly one of these shapes:

1. A direct file whose suffix is `.jsonl`, case-insensitively.
2. A directory whose top-level regular files include one or more `.jsonl`
   files.
3. A trial directory containing a real `trajectory/` directory; select the
   top-level `.jsonl` files inside it and read optional run metadata from the
   trial root.

Apply these boundaries before redaction or upload:

- Reject a symlink used as the selected path, `trajectory/` directory, or JSONL
  artifact.
- Stage only top-level regular `.jsonl` files from the selected payload
  directory. Report non-JSONL sibling files as ignored. Do not recurse.
- Require at least one nonblank JSON object record per JSONL file.
- Require strict UTF-8 JSONL: each nonblank line is one top-level object. Reject
  invalid JSON, duplicate object keys, non-finite numbers, and non-object
  records.
- Allow at most 8 JSONL artifacts, 128 MiB per artifact, 256 MiB total, and
  8 MiB per JSONL record.
- Limit each record to 100 levels of JSON nesting and 100,000 JSON values.
- Limit the generated manifest to 1 MiB. Read at most 1 MiB from each optional
  run-metadata file.
- Copy accepted content to a temporary staging directory. Never mutate the
  source capture.

The staged artifact namespace is `trajectory/<filename>.jsonl`. Artifact names
that escape this namespace or resemble secret material are rejected.

## Local redaction

Redact every artifact before report generation and upload. Replace detected
secret-bearing values with the exact marker:

```text
<XXX-benchflow-key-values-XXX>
```

Redaction covers secret-like object fields and keys, known token patterns,
authorization values, CLI argument values, URLs and query parameters, and
nested strings. Repeat passes until the result converges, preserving an already
safe marker. Count every replacement.

Do not reject otherwise valid JSONL merely because it contains a detected key.
Upload only the locally redacted staging copy. Re-scan and redact the manifest
to convergence too; reject a source ID or contributor field that itself looks
like a secret instead of silently changing identity or storage routing.

## Trajectory report

Build one canonical report from the redacted artifacts:

- Choose `acp_trajectory.jsonl` as primary when present, another JSONL file
  next, and `llm_trajectory.jsonl` last. This avoids double-counting parallel
  ACP and LLM logs from one trial.
- Detect BenchFlow ACP, Claude Code, Codex, LLM exchange, OpenTrace, or generic
  JSONL formats from a bounded initial scan.
- Use the earliest recognized trajectory timestamp. Fall back to the primary
  source file timestamp and label the source.
- Count report file count and size across all staged JSONL artifacts, while
  counting steps only from the primary artifact.
- Count only meaningful human, thinking, and tool-call steps. A genuine user
  message is a human step. Tool results, queue/status events, metadata,
  attachments without a meaningful step, blank content, and synthetic labels
  such as `Assistant response` are not steps.
- Enforce `total_steps = thinking_steps + tool_call_steps + human_steps`.
- Preview the first N meaningful steps in chronological order. Show the first
  100 whitespace-delimited words of each already-redacted step, with a 4,000
  character safety bound and terminal control characters stripped.
- Display artifact redaction count as `API keys / secrets masked` and the safe
  marker separately.

The rendered report includes primary file, format, created time and source,
JSONL file count, trajectory byte size, total/thinking/tool-call/human counts,
masked count, safe marker, and the optional preview.

## Contributor metadata and manifest

GitHub ID and email are self-asserted provenance. Validate and store both under
`contributor`; never accept only one. Do not print the email in the final CLI
result or operator report.

The CLI report causes `manifest.json` schema `1.2.0` to be generated. It stores:

- kind `bronze.trajectory`, creation time, source ID, and the content digest;
- BenchFlow tool name and version;
- optional scalar run metadata (`agent`, `model`, `harness`, `skill_mode`,
  `task_id`, and `reward`) read from valid trial `result.json`/`config.json`;
- each artifact name, SHA-256, and byte size;
- whether redaction ran and the final replacement count;
- contributor GitHub ID and email;
- every report field: primary file, format, file count, size, total/thinking/
  tool-call/human counts, created time and source, masked count, and every
  preview row (`number`, `kind`, `summary`).

Compute the trajectory content digest from sorted artifact-name/SHA-256 pairs.
It deliberately excludes contributor identity, upload time, manifest bytes, and
transport so the same redacted capture is idempotent across contributors and
retries. Append `manifest.json` after all trajectory artifacts and upload it
last.

## Dry-run behavior

`--dry-run` performs the full local pipeline through manifest finalization. It
still needs contributor metadata, but it bypasses confirmation and all network
calls. Print:

- `Dry run — no files uploaded`;
- the `sha256:` trajectory digest;
- each staged trajectory file and `manifest.json`, with sizes;
- ignored sibling names, when present;
- the final redaction count.

A dry run proves only input resolution, strict JSONL validation, local
redaction, report generation, contributor validation, and manifest creation for
that CLI build.

## Upload progress and results

For a real upload, stream byte-level progress over every staged file, including
`manifest.json`. Resynchronize at file boundaries so transport retries cannot
inflate completed bytes. Finish the progress label as `Upload complete`.

If at least one object was created, print the destination plus uploaded and
skipped object counts, total staged bytes including the manifest, and final
redaction count. If no object was created, print `Already uploaded` and treat it
as a successful no-op.

Never print signed PUT URLs, request headers, internal service endpoints,
credentials, detected secrets, or contributor email.

## Storage verification

After a public upload, the CLI polls `GET /v1/uploads/<digest>` on the broker
(the validation-ledger state) until one of these outcomes:

- `ingested` → print `Verified in Azure storage`. The validator records
  `ingested` only after promoting every file to `sources/community/<digest>/`,
  so this line is proof the capture reached durable storage — a stronger claim
  than the transfer finishing.
- `rejected` → exit 1 with the bounded, user-fixable rejection detail. The
  transfer succeeded but the capture is not in the dataset.
- HTTP 404 from the endpoint → the deployed broker predates status reporting;
  return silently with exactly the pre-verification behavior.
- Budget exhausted (default 240 s with backoff from 2 s to 10 s) or repeated
  transport failures → say the upload is safely queued and print the
  `bench traj status sha256:<digest>` line to check later.

A handshake 409 (already ingested) prints the verified line immediately
without polling. `bench traj status DIGEST` runs one poll on demand and maps
states to exit codes: `ingested`/`pending`/`validating` exit 0; `rejected`,
an unknown digest, rate limiting, and a status-less deployment exit 1.

The status endpoint reveals only the ledger state, the bounded rejection
detail, and the public promotion prefix for a digest the caller already
holds — never contributor identity, source ids, or quarantine internals.
Status polls consume a separate, much higher rate-limit budget than upload
grants (`TRAJ_STATUS_RATE_LIMIT`, default 720/hour/IP).

## Public upload protocol

The public client:

1. Sends the schema-1.2 declaration, contributor metadata, artifact
   declarations, content digest, and exact manifest SHA-256 to the contribution
   service.
2. Accepts a canonical object list only when it matches the staged files exactly
   in manifest-last order.
3. Validates that every returned PUT target is authenticated HTTPS and every
   header is a string-to-string mapping.
4. Uploads each object with only the returned headers. Treats PUT 409/412 as an
   idempotent skip.
5. Treats a handshake 409 for an already ingested digest as `Already uploaded`.

The trusted service assigns a random quarantine identity and create-only,
prefix-scoped writes. The validator independently parses strict JSONL, rescans
for secrets, recomputes hashes, digest, report, and manifest binding, then
promotes the accepted capture to `sources/community/<digest>/` with the trusted
manifest last. Client ordering and client-generated metadata are declarations,
not the security boundary.

## Trusted direct upload

Use `--direct` only for an authorized operator writing to a private Azure Blob
container. Install the Azure extra, then authenticate through
`DefaultAzureCredential` (for example, `az login` or managed identity):

```bash
uv tool install --python 3.12 --upgrade 'benchflow[azure]'
bench traj upload <PATH> \
  --github-id <GITHUB_ID> \
  --email <EMAIL> \
  --direct \
  --container-url 'https://<account>.blob.core.windows.net/<container>'
```

Direct mode writes create-only blobs under
`sources/<source-id>/<digest>/`; existing blobs are skipped individually.
It does not use public quarantine or the public validator, so never describe a
direct-upload success as proof that the public contributor route works.

## Failure behavior

User-fixable validation and transport errors become concise CLI errors with
exit code 1. Diagnose from the message without exposing sensitive values:

- missing path, no JSONL files, symlinks, invalid JSONL, complexity or size
  limits, invalid source ID, GitHub ID, or email;
- invalid `--direct`/`--container-url` combinations;
- public service connection or protocol errors;
- HTTP 413 for an oversized public capture;
- HTTP 429 for rate limiting, including `Retry-After` when provided;
- direct-mode missing Azure dependencies, invalid container URL,
  authentication failure, missing container, or forbidden writes.

Treat an interactive cancellation and an idempotent already-uploaded result as
successful no-op outcomes, not upload failures.

## Verification levels

Use precise claims:

| Evidence | Maximum valid claim |
| --- | --- |
| Static code/skill review | Intended behavior only. |
| Unit or CLI-runner tests | Local implementation behavior under mocks. |
| `--dry-run` on a real capture | Real local parsing, masking, report, and manifest behavior; no network proof. |
| Real public command returns success | Client and contribution-service upload path accepted the capture; promotion still needs checking. |
| CLI prints `Verified in Azure storage` (or `traj status` reports `ingested`) against a deployed broker | The validation ledger recorded promotion for that digest; equivalent to promotion unless the ledger itself is suspect. |
| Trusted storage shows the exact digest under `sources/community/`, with validator-recomputed schema-1.2 manifest and no quarantine residue | Production public upload worked end to end for that exact build and capture. |

Record the CLI version or Git SHA, digest, command mode, result, and storage
evidence. Use a safe canary or user-authorized real capture. Do not delete or
overwrite unrelated storage objects while verifying.

## Implementation map

Use the implementation, not this prose, as the final authority when reviewing a
new code change:

- `src/benchflow/cli/traj.py`: options, prompts, confirmation, dry-run, routing,
  result text, the storage-verification wait, and `bench traj status`.
- `src/benchflow/cli/_traj_upload_ui.py`: report and byte-progress rendering.
- `src/benchflow/cli/_traj_tui.py`: the shared `bench traj` design language —
  banner, styled prompts, and the TTY-only session picker (presentation only;
  every interactive affordance falls back to the plain prompt contract).
- `services/trajectory_upload/broker_app.py` + `azure_backend.py`: the
  `GET /v1/uploads/{digest}` capture-status endpoint over the ledger.
- `src/benchflow/publish/traj_capture.py`: input limits, strict JSONL, staging,
  digest, contributor metadata, and manifest.
- `src/benchflow/publish/redact.py`: local secret detection and convergence.
- `src/benchflow/publish/traj_report.py`: format adapters, step accounting,
  preview, and manifest report serialization.
- `src/benchflow/publish/broker.py`: public handshake, response validation,
  upload ordering, idempotency, and public transport errors.
- `src/benchflow/publish/azure_blob.py`: trusted direct Azure transport.
