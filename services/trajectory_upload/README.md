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
