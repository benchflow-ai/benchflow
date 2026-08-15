---
name: benchflow-traj-upload
description: Upload completed BenchFlow trajectory captures with required contributor metadata. Use whenever a user asks to upload, submit, share, or contribute one or more BenchFlow trajectories.
user-invocable: true
allowed-tools:
  - Bash
---

# Upload BenchFlow trajectories

Require a local capture path, GitHub ID, and email; ask only for missing values.
If needed, install the latest stable CLI:

```bash
uv tool install --python 3.12 --upgrade benchflow
```

Run once per capture:

```bash
bench traj upload <PATH> --github-id <GITHUB_ID> --email <EMAIL>
```

Use the public default: do not add broker URLs, Azure credentials, or extra
flags. The CLI stores the GitHub ID and email under `contributor` in
`manifest.json`. Report whether each capture was uploaded or already present.
