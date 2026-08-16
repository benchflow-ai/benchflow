---
name: benchflow-traj-upload
description: Upload or safely test completed BenchFlow trajectory contributions with local redaction, report inspection, and required contributor metadata. Use whenever a user asks to upload, test, validate, submit, share, or contribute one or more BenchFlow trajectories.
user-invocable: true
allowed-tools:
  - Bash
---

# Upload BenchFlow trajectories

Require a local capture path, GitHub ID, and email. The CLI asks for any missing
values in that order. Use the public default; do not add broker URLs, Azure
credentials, or direct-upload flags for normal contributions.

For a released CLI, install or upgrade BenchFlow:

```bash
uv tool install --python 3.12 --upgrade benchflow
```

When testing an unreleased repository checkout, do not substitute the stable
tool. Run `uv sync --extra dev --locked`, then replace `bench` below with
`uv run bench`.

Run once per capture. Prefer the interactive flow so the user can review and
confirm the report:

```bash
bench traj upload
```

For a no-network rehearsal, use:

```bash
bench traj upload --dry-run
```

Or provide every input in one command. This form shows the report but starts
uploading without a confirmation prompt unless `--dry-run` is present:

```bash
bench traj upload <PATH> --github-id <GITHUB_ID> --email <EMAIL>
```

After the path is entered, verify the report before continuing:

- Preview rows contain the first 100 words of each meaningful redacted step;
  `--preview-steps` accepts 0-20 and defaults to 5.
- Total steps equal thinking steps plus tool-call steps plus human steps. Human
  steps are genuine user messages; tool results, status/metadata events, empty
  records, and invented placeholders such as `Assistant response` do not count.
- Detected secret values are replaced locally with
  `<XXX-benchflow-key-values-XXX>`; the original files remain unchanged.

The uploaded schema-1.2 `manifest.json` stores GitHub ID and email under
`contributor` and every displayed report field under `trajectory_report`. The
server independently validates and rescans the capture before promotion. A dry
run proves only local staging, so do not claim a real end-to-end upload unless a
compatible live service accepts and promotes a canary.

Report whether each capture was uploaded, cancelled, already present, or only
dry-run validated. Never expose contributor email, signed upload URLs, broker
internals, or detected secret values in the final response.
