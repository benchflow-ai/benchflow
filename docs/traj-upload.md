# Contribute trajectory captures

Copy this line into your coding agent. You do not run a BenchFlow command.

```
Submit my best local Claude Code or Codex session to the BenchFlow eval prize. Read https://raw.githubusercontent.com/benchflow-ai/benchflow/main/.agents/skills/benchflow-traj-upload/SKILL.md and follow it: find a session, open the viewer, upload only after I review it.
```

The agent reads the skill, finds a local session, opens the viewer, and
uploads after you say it looks good.

## Optional: set the skill up once

If you want later chats to know the workflow without pasting the long line:

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
`.agents/skills/benchflow-traj-upload/`, prints the same paste line, and can
list sessions or open the viewer. After setup, a short ask is enough:
**submit my best session to the eval prize**.

`bench traj setup --prompt` prints only the copy-paste line.
`bench traj setup --list` lists recent local sessions.

The [skill source](../.agents/skills/benchflow-traj-upload/SKILL.md) is the
contributor contract. The agent infers GitHub username and email from `gh` /
`git` (or `BENCHFLOW_GITHUB_ID` / `BENCHFLOW_EMAIL`) and asks in chat if it
cannot. What follows is the transport the agent uses.

The [skill source](../.agents/skills/benchflow-traj-upload/SKILL.md) is the
contributor contract. What follows is the transport the agent uses.

## What the agent runs

After you review the viewer, the agent uploads a JSONL file, a folder of
JSONL files, or a trial directory containing `trajectory/`:

```bash
bench traj upload path/to/your-session.jsonl
```

Both `--github-id` and `--email` are self-asserted contributor provenance,
stored in `manifest.json` as
`{"contributor":{"github_id":"...","email":"..."}}`. The email is not
printed.

BenchFlow rejects duplicate object keys and non-finite numbers, structurally
redacts credential-bearing keys and secret-like values, computes a content
digest, and uploads a manifest last. The first request can take a minute
while the public broker wakes up. The same upload again is safe. Use
`--dry-run` to inspect the staged file list and digest without making a
network request.

The public broker URL is built into the CLI. `BENCHFLOW_TRAJ_BROKER_URL` can
override it for development or disaster recovery, and
`BENCHFLOW_TRAJ_UPLOADED_BY` can add a non-secret contributor label. Do not put
credentials or personal data in either label.

## Viewer

`bench eval view PATH` serves a localhost page for a trial directory, a job
directory, or a raw Claude Code / Codex / ACP session JSONL file. The skill
opens this before upload. Viewing a JSONL file does not write
`trajectory.html` next to the session.

## What reaches the dataset

Public uploads first enter a private, versioned Azure Blob quarantine prefix.
The broker issues short-lived user-delegation SAS URLs scoped to create
one expected blob at a time; they do not grant list, read, or delete access.
An Event Grid-triggered validator independently checks the manifest contract,
the 8 MiB per-record JSONL bound and structural complexity limits,
allowlisted object names, byte sizes, SHA-256 hashes, strict JSONL syntax, and
final artifact and manifest secret scans. Only then does it copy artifacts into
the content-addressed `sources/community/<digest>/` namespace, with
`manifest.json` as the commit marker. Failed captures are removed from the live
quarantine namespace and are never promoted. Blob versioning and lifecycle
policy provide recovery and bound retention for attempted overwrites; the
deployment does not configure an immutable-storage policy.

The digest excludes contributor labels, timestamps, and transport details, so
the same redacted bytes are idempotent across machines. Repeating a submitted
upload prints `Already submitted` and does not fail.

Redaction is a safety net, not a license to upload secrets. Review sensitive
trajectories before contributing them; once a capture is promoted, dataset
operators may retain it for benchmark provenance.

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

Deployment configuration and verification live in
[`infra/trajectory-upload/`](../infra/trajectory-upload/README.md).
