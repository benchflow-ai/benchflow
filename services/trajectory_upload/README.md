# Trajectory upload service

This directory is the single ownership boundary for the public trajectory
upload service. It contains the broker, validator, public/storage contracts,
container image, Azure infrastructure, and deployment operations. The installed
CLI remains under `src/benchflow/` because it runs on contributor machines.

```text
services/trajectory_upload/
├── broker_app.py
├── azure_backend.py
├── validator.py
├── validation.py
├── contract.py
├── Dockerfile
├── validator-entrypoint
├── infra/
│   ├── main.bicep
│   ├── production.bicepparam
│   └── lifecycle.json
├── scripts/
│   ├── deploy.sh
│   ├── smoke-test.sh
│   └── rollback.sh
└── README.md
```

`scripts/deploy.sh` provisions the production upload path with Azure CLI:

- one private, versioned storage account with `bronze`, `silver`, and `gold`
  containers, Shared Key and anonymous access disabled;
- a scale-to-zero Container App broker with a user-assigned identity limited to
  create/write blob data actions, delegation-key signing, and the upload ledger;
- Event Grid delivery of inbox object creations to an Entra-authenticated
  Storage Queue, so manifests commit pending captures and terminal replays are
  cleaned;
- an event-driven Container Apps validator Job with separate read, promotion,
  cleanup, queue, and ledger authority;
- two-day inbox/version expiry plus storage read/write/delete diagnostics.

The broker also serves `GET /v1/uploads/{digest}`: the public capture-status
endpoint the CLI polls after an upload (and `bench traj status` on demand). It
reads the validation ledger and reports `pending`, `validating`, `ingested`
(with the `sources/community/<digest>/` promotion prefix), `rejected` (with
the bounded rejection detail), or `unknown`; unknown digests are a 200 so
clients can use 404 to detect a deployment that predates the endpoint. Status
polls consume their own per-IP budget (`TRAJ_STATUS_RATE_LIMIT`, default
720/hour) instead of upload-grant quota.

`infra/main.bicep` owns the stable resource topology: private storage and data
containers, queue/table, diagnostics, registry, identities, log workspace, and
Container Apps environment. `scripts/deploy.sh` owns operations that require
the built image or dynamic identity state: least-privilege RBAC, Event Grid
wiring, broker and validator revisions, and the lifecycle policy.

The script is idempotent for the named production resources. It requires an
Azure subscription Owner or User Access Administrator because it creates a
custom role and managed-identity role assignments.

```bash
./services/trajectory_upload/scripts/deploy.sh
```

Override names and region with the `BENCHFLOW_UPLOAD_*` variables declared at
the top of the script. Use `scripts/smoke-test.sh` after deployment, and use
`scripts/rollback.sh` with a known-good image tag and broker revision if a
deployment must be reverted.

The manual `deploy-trajectory-upload` GitHub workflow contains no deployment
logic; it signs in with Azure OIDC and invokes these scripts. Configure the
`trajectory-upload-production` environment with `AZURE_CLIENT_ID`,
`AZURE_TENANT_ID`, and `AZURE_SUBSCRIPTION_ID`. The federated identity needs
Owner or User Access Administrator plus resource-write authority because the
deployment creates a custom role and scoped managed-identity assignments.

## Verification

Compile and inspect the infrastructure before deployment:

```bash
az bicep build --file services/trajectory_upload/infra/main.bicep
az deployment group what-if \
  --resource-group tasksminer-upload-prod \
  --template-file services/trajectory_upload/infra/main.bicep \
  --parameters services/trajectory_upload/infra/production.bicepparam
```

After deployment, run:

```bash
./services/trajectory_upload/scripts/smoke-test.sh
```

The smoke test verifies compute provisioning, image parity, storage hardening,
private containers, and the live health response without printing the internal
service endpoint.

## What the service does with an upload

This section moved here from the user page `docs/results/share-a-trajectory.md`; it describes the server side of `bench traj upload`.

Public uploads first enter a private, versioned Azure Blob quarantine prefix.
The broker issues short-lived user-delegation SAS URLs scoped to create one
expected blob at a time; they do not grant list, read, or delete access.

An Event Grid-triggered validator independently checks:

- the manifest contract;
- the 8 MiB per-record JSONL bound and structural complexity limits;
- allowlisted object names, byte sizes and SHA-256 hashes;
- strict JSONL syntax;
- final secret scans of the JSONL artifacts and the manifest.

The server recognizes the exact local replacement marker
(`<XXX-benchflow-key-values-XXX>`) but fails closed if any raw secret-like
value survives. A workspace archive (`workspace/*.zip`) is opaque to the
validator: it checks only the hash binding and the zip magic
(`validation.py`), not the contents.

Only then does the validator copy artifacts into the content-addressed
`sources/community/<digest>/` namespace, with `manifest.json` as the commit
marker. Failed captures are removed from the live quarantine namespace and
are never promoted. Blob versioning and the lifecycle policy provide recovery
and bound retention for attempted overwrites; the deployment does not
configure an immutable-storage policy.

The digest excludes contributor labels, timestamps, and transport details, so
the same redacted bytes are idempotent across machines.

The uploaded `manifest.json` keeps the complete redacted report under
`trajectory_report`: primary file, detected format, JSONL file and byte totals,
the step counts, creation time and its source, the masked-value total, and
every preview row. The server validates those values against the declared
artifacts with a closed schema and an exact recompute, and rejects
inconsistent report metadata. The per-kind masked counts the CLI shows are
display-only and are not uploaded.

### CLI environment variables for operators

| Variable | Effect |
|---|---|
| `BENCHFLOW_TRAJ_BROKER_URL` | Overrides the broker URL built into the CLI, for development or disaster recovery. |
| `BENCHFLOW_TRAJ_UPLOADED_BY` | Adds a non-secret contributor label to the manifest. |
| `BENCHFLOW_TRAJ_WAIT_SECONDS` | The post-upload wait budget (default 240; `0` disables waiting). |
| `BENCHFLOW_AZURE_CONTAINER_URL` | The container for `--direct` when `--container-url` is not given. |

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

`BENCHFLOW_AZURE_CONTAINER_URL` can replace `--container-url`. Direct mode uses
`DefaultAzureCredential` and create-only blob calls, and skips the
post-upload status wait. The identity needs a custom role with blob
create/write data actions on the target container, for example a custom
`Blob Data Creator` role. Azure's broader `Storage Blob Data Contributor` role
also works but grants more than direct upload needs. For routine community
contributions, use the default broker mode.
