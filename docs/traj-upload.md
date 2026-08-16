# Contribute trajectory captures

Anyone with BenchFlow installed can contribute a completed trajectory through
either an interactive prompt:

```bash
bench traj upload
```

or one fully specified command:

```bash
bench traj upload path/to/trial \
  --github-id YOUR_GITHUB_ID \
  --email YOU@example.com
```

When values are omitted, BenchFlow prompts for the path first. It then validates,
redacts, and inspects the local trajectory; renders the report and preview;
prompts for a missing GitHub ID and email; and asks for confirmation before
uploading. `path/to/trial` may be a trial directory containing `trajectory/`, a
directory of JSONL files, or one JSONL file. BenchFlow rejects duplicate
object keys and non-finite numbers, but detected secret-like values do not make
otherwise-valid JSONL ineligible: the local staging pass replaces them with
`<XXX-benchflow-key-values-XXX>`. It applies the same structural redaction to
manifest metadata, computes a content digest, and uploads a manifest last. Use
`--dry-run` to inspect the
staged file list, digest, sizes, ignored siblings, and redaction count without
making a network request.

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
explicit confirmation. A command containing the path, `--github-id`, and
`--email` remains non-interactive: it shows the same report and starts uploading
without another prompt. Upload progress is displayed by processed file bytes.

GitHub ID and email are required inputs for both public and direct uploads, but
may be provided through options or prompts. They are self-asserted contributor
provenance, not proof of account ownership, and are stored in `manifest.json` as
`{"contributor":{"github_id":"...","email":"..."}}`. An interactively
entered email is visible in the terminal prompt but is not repeated in the
success output. Dataset operators may retain or publish the manifest; use an
address you are comfortable associating with the contribution.

The public broker URL is built into the CLI. `BENCHFLOW_TRAJ_BROKER_URL` can
override it for development or disaster recovery, and
`BENCHFLOW_TRAJ_UPLOADED_BY` can add a non-secret contributor label. Do not put
credentials or personal data in either label.

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
the same redacted bytes are idempotent across machines. Repeating an ingested
upload prints `Already uploaded` and performs no blob writes.

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
