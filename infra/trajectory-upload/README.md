# Azure trajectory upload deployment

`deploy-azure.sh` provisions the production upload path with Azure CLI:

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

The script is idempotent for the named production resources. It requires an
Azure subscription Owner or User Access Administrator because it creates a
custom role and managed-identity role assignments.

```bash
./infra/trajectory-upload/deploy-azure.sh
```

Override names and region with the `BENCHFLOW_UPLOAD_*` variables declared at
the top of the script. The final line is the public broker URL to bake into the
CLI release after a successful end-to-end validation.
