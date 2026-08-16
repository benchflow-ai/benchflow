---
name: benchflow-traj-upload
description: Upload completed BenchFlow trajectory captures with required contributor metadata. Use whenever a user asks to upload, submit, share, or contribute one or more BenchFlow trajectories.
user-invocable: true
allowed-tools:
  - Bash
---

# Upload BenchFlow trajectories

Require a local capture path, GitHub ID, and email. The CLI asks for any missing
values in that order, so use either its interactive flow or provide every value
up front.
If needed, install the latest stable CLI:

```bash
uv tool install --python 3.12 --upgrade benchflow
```

Run once per capture, interactively:

```bash
bench traj upload
```

Or provide every input in one command:

```bash
bench traj upload <PATH> --github-id <GITHUB_ID> --email <EMAIL>
```

Use the public default: do not add broker URLs, Azure credentials, or extra
flags. The CLI validates JSONL locally and replaces detected secret values with
`<XXX-benchflow-key-values-XXX>` before upload. It stores the GitHub ID and email
under `contributor` in `manifest.json`. Interactive mode renders a redacted
trajectory report immediately after the path, asks for missing contributor
details, and requires confirmation before upload. Review that report with the
user when acting interactively. The complete redacted report is retained under
`trajectory_report` in the uploaded manifest. Report whether each capture was
uploaded, cancelled, or already present.
