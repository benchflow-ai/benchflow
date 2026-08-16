# Contribute trajectory captures

**Don't follow these steps yourself — send them to your coding agent.** Copy
the block below and paste it as a message to Claude Code, Codex, OpenCode, or
Cursor. You do not run a BenchFlow command.

> Submit my relevant local Claude Code, Codex, OpenCode, or Cursor session from the re:Agent e2e agentic science / ai4bio hackathon (last 72 hours, on this laptop) to the BenchFlow eval prize.
>
> 1. First make sure the latest benchflow CLI is installed: uv tool install --python 3.12 --upgrade --force benchflow
>
> 2. Then read https://raw.githubusercontent.com/benchflow-ai/benchflow/main/.agents/skills/benchflow-traj-upload/SKILL.md and follow it: find a session, open the viewer, and upload only after I (the human) review it.

The agent reads the skill, finds a local session, opens the viewer, and
uploads after you say it looks good.

The prompt has the agent install the latest BenchFlow first (`uv tool
install --python 3.12 --upgrade --force benchflow`) because `bench traj setup`
and the session-JSONL viewer require 0.7.1+. When a newer release is available
on PyPI, `bench traj setup` and `bench traj upload` print a one-line upgrade
hint.

## Optional: set the skill up once

If you want later chats to know the workflow without pasting the long prompt:

```bash
npx skills add benchflow-ai/benchflow --skill benchflow-traj-upload
```

or, with BenchFlow already installed:

```bash
bench traj setup
```

`npx skills add` is interactive: it asks which agents to install for
(Claude Code, Codex, Cursor, and other [Agent Skills](https://agentskills.io)
hosts). `bench traj setup` copies the skill into
`.agents/skills/benchflow-traj-upload/`, prints the same agent prompt, and can
list sessions or open the viewer. After setup, a short ask is enough:
**submit my relevant session to the eval prize**.

`bench traj setup --prompt` prints only the copy-paste agent prompt.
`bench traj setup --list` lists recent local sessions.

## Optional: interactive upload without an agent

Prefer the terminal? The guided flow inspects and confirms before anything
leaves the machine:

```bash
bench traj upload
```

It prompts for the path first, then validates, redacts, and inspects the local
trajectory; renders the report and preview; fills GitHub ID and email from
`gh` / `git` (or `BENCHFLOW_GITHUB_ID` / `BENCHFLOW_EMAIL`), prompting only for
what inference cannot find; and asks for confirmation before uploading.

The fully specified command is non-interactive for scripts and agents:

```bash
bench traj upload path/to/trial \
  --github-id YOUR_GITHUB_ID \
  --email YOU@example.com
```

`path/to/trial` may be a trial directory containing `trajectory/`, a directory
of JSONL files, or one JSONL file. BenchFlow rejects duplicate object keys and
non-finite numbers, but detected secret-like values do not make
otherwise-valid JSONL ineligible: the local staging pass replaces them with
`<XXX-benchflow-key-values-XXX>`. It applies the same structural redaction to
manifest metadata, computes a content digest, and uploads a manifest last. The
first request can take a minute while the public broker wakes up; retries are
safe. Use `--dry-run` to inspect the staged file list, digest, sizes, ignored
siblings, and redaction count without making a network request.

## Local trajectory report

The report is generated only from the staged, redacted copy. It shows:

- the primary trajectory and detected format;
- the earliest trajectory timestamp, falling back to the source file timestamp;
- JSONL file count and total trajectory size;
- mutually exclusive step counts where total steps always equals thinking steps
  plus tool-call steps plus human steps. Human steps are user-authored messages,
  tool-call steps are agent tool invocations, and thinking steps are reasoning
  or other agent-authored non-tool messages. Tool results and status/metadata
  records are observations rather than separately counted steps. Records with
  no extractable redacted text are skipped instead of producing placeholder
  steps such as `Assistant response`;
- the number of API-key or secret-like values replaced with
  `<XXX-benchflow-key-values-XXX>`; and
- the first five meaningful steps as a preview containing up to the first 100
  words of each step's redacted text.

Use `--preview-steps N` to show 0–20 steps. When a trial contains both
`acp_trajectory.jsonl` and `llm_trajectory.jsonl`, the report uses ACP as the
primary interaction view so the same run is not double-counted; file count,
size, and masked-value totals still cover every uploaded JSONL artifact. Format
classification is exact for BenchFlow ACP, Claude Code, Codex, OpenTrace, and
BenchFlow LLM-exchange files, with a conservative generic JSONL fallback.

The uploaded `manifest.json` persists the complete redacted report under
`trajectory_report`: primary file, detected format, JSONL file and byte totals,
the mutually exclusive step counts, creation time and its source, masked-value
count, and every displayed preview row. The server validates those values
against the declared artifacts and rejects inconsistent report metadata.

Interactive mode shows this report before contributor prompts and requires an
explicit confirmation. A command whose path and identity resolve without
prompting — flags or `gh` / `git` inference — remains non-interactive: it shows
the same report and starts uploading without another prompt. Upload progress is
displayed by processed file bytes.

GitHub ID and email are required inputs for both public and direct uploads, but
may be provided through options, inference, or prompts. They are self-asserted
contributor provenance, not proof of account ownership, and are stored in
`manifest.json` as `{"contributor":{"github_id":"...","email":"..."}}`. An
interactively entered email is visible in the terminal prompt but is not
repeated in the success output. Dataset operators may retain or publish the
manifest; use an address you are comfortable associating with the contribution.

The public broker URL is built into the CLI. `BENCHFLOW_TRAJ_BROKER_URL` can
override it for development or disaster recovery, and
`BENCHFLOW_TRAJ_UPLOADED_BY` can add a non-secret contributor label. Do not put
credentials or personal data in either label.

## Viewer

`bench eval view PATH` serves a localhost page for a trial directory, a job
directory, or a raw Claude Code / Codex / ACP session JSONL file. The
contributor skill opens this before upload. Viewing a JSONL file does not
write `trajectory.html` next to the session.

## What reaches the dataset

Public uploads first enter a private, versioned Azure Blob quarantine prefix.
The broker issues short-lived user-delegation SAS URLs scoped to create
one expected blob at a time; they do not grant list, read, or delete access.
An Event Grid-triggered validator independently checks the manifest contract,
the 8 MiB per-record JSONL bound and structural complexity limits,
allowlisted object names, byte sizes, SHA-256 hashes, strict JSONL syntax, and
final artifact and manifest secret scans. The server recognizes the exact local
replacement marker but still fails closed if any raw secret-like value survives.
Only then does it copy artifacts into
the content-addressed `sources/community/<digest>/` namespace, with
`manifest.json` as the commit marker. Failed captures are removed from the live
quarantine namespace and are never promoted. Blob versioning and lifecycle
policy provide recovery and bound retention for attempted overwrites; the
deployment does not configure an immutable-storage policy.

The digest excludes contributor labels, timestamps, and transport details, so
the same redacted bytes are idempotent across machines. Repeating a submitted
upload prints `Already submitted` and performs no blob writes.

The local replacement pass is designed to make otherwise-valid JSONL containing
detected keys safe to upload. Review sensitive trajectories before contributing
them because automated detection can still have false negatives; once a capture
is promoted, dataset operators may retain it for benchmark provenance.

## Trusted direct upload

Operators with Azure RBAC can bypass the public broker while keeping the same
staging and manifest contract:

```bash
uv tool install 'benchflow[azure]'
az login
bench traj upload path/to/trial --direct \
  --github-id YOUR_GITHUB_ID \
  --email YOU@example.com \
  --container-url https://ACCOUNT.blob.core.windows.net/bronze
```

Direct mode uses `DefaultAzureCredential` and create-only blob calls. The
identity needs a custom role with blob create/write data actions on the target
container. The production deployment creates this as
`TasksMiner Blob Data Creator`; Azure's broader `Storage Blob Data Contributor`
role also works but grants more than direct upload needs. For routine community
contributions, use the default broker mode.

Deployment configuration and verification live beside the service in
[`services/trajectory_upload/`](../services/trajectory_upload/README.md).
