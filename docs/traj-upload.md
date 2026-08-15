# Contribute trajectory captures

Anyone with BenchFlow installed can contribute a completed trajectory with one
command:

```bash
bench traj upload path/to/trial
```

`path/to/trial` may be a trial directory containing `trajectory/`, a directory
of JSONL files, or one JSONL file. BenchFlow validates every line as JSON,
structurally redacts credential-bearing keys and secret-like values, computes a
content digest, and uploads a manifest last. Use `--dry-run` to inspect the
staged file list, digest, sizes, ignored siblings, and redaction count without
making a network request.

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
allowlisted object names, byte sizes, SHA-256 hashes, JSONL syntax, and a final
secret scan. Only then does it copy artifacts into the immutable
`sources/community/<digest>/` namespace, with `manifest.json` as the commit
marker. Failed captures are removed from the live quarantine namespace and are
never promoted. Blob versioning and lifecycle policy bound recovery and
retention for attempted overwrites.

The digest excludes contributor labels, timestamps, and transport details, so
the same redacted bytes are idempotent across machines. Repeating an ingested
upload prints `Already uploaded` and performs no blob writes.

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
  --container-url https://ACCOUNT.blob.core.windows.net/bronze
```

Direct mode uses `DefaultAzureCredential` and create-only blob calls. The
identity needs `Storage Blob Data Creator` on the target container. For routine
community contributions, use the default broker mode.

Deployment configuration and verification live in
[`infra/trajectory-upload/`](../infra/trajectory-upload/README.md).
