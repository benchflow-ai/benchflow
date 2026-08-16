---
name: benchflow-traj-upload
description: Operate, test, troubleshoot, and explain `bench traj upload` for public or trusted-direct trajectory contributions, including interactive and fully specified commands, dry runs, input validation, local secret masking, trajectory reports and previews, manifest metadata, upload progress, idempotency, and production promotion checks. Use this skill whenever a user wants to upload, contribute, share, test, locate, inspect, or debug a trajectory upload; validate a trajectory, report, or manifest; or verify the public upload path end to end, even if they do not explicitly name this skill.
---

# Upload BenchFlow trajectories

Use the public contribution route for ordinary users. Treat `--direct` as a
trusted-operator escape hatch, not as an equivalent public-path test.

Read [references/cli-contract.md](references/cli-contract.md) before explaining
non-default modes, troubleshooting a failure, reviewing a generated manifest,
or claiming an end-to-end production result. The reference is the complete
behavior contract for the current CLI.

## Choose the matching CLI build

Use the released CLI when validating the published user experience:

```bash
uv tool install --python 3.12 --upgrade benchflow
bench traj upload
```

Use the repository checkout when validating unreleased PR behavior:

```bash
uv sync --extra dev --locked
uv run bench traj upload
```

Do not substitute the installed `bench` binary for `uv run bench` while testing
unreleased code. Record `bench --version` or the exact Git SHA so the tested
artifact is unambiguous.

## Run the requested flow

Prefer the guided flow when a person wants to inspect and confirm the capture:

```bash
bench traj upload
```

The CLI asks for the path, renders a report from a locally redacted staging
copy, asks for missing GitHub and email metadata, and defaults the upload
confirmation to No.

Use the fully specified form for scripts or an intentional no-prompt upload:

```bash
bench traj upload <PATH> --github-id <GITHUB_ID> --email <EMAIL>
```

Providing all three required inputs makes the command non-interactive: it still
renders the report, then uploads without a confirmation prompt. If any one is
missing, the session is interactive and asks only for missing values before a
final confirmation.

Use a dry run before a real upload when testing new files or CLI changes:

```bash
bench traj upload <PATH> \
  --github-id <GITHUB_ID> \
  --email <EMAIL> \
  --dry-run
```

A dry run validates, redacts, reports, and creates the temporary manifest, but
never prompts for confirmation and never makes a network request.

## Inspect before uploading

Verify these invariants in the rendered report:

- `Total steps = Thinking steps + Tool-call steps + Human steps`.
- Human steps are real user messages. Tool results, status or metadata records,
  empty records, and invented placeholders such as `Assistant response` are not
  trajectory steps.
- Each preview row shows up to the first 100 words of a meaningful, already
  redacted step. `--preview-steps` accepts 0 through 20 and defaults to 5.
- Every detected secret value is replaced locally with
  `<XXX-benchflow-key-values-XXX>`. Valid JSONL containing secrets is accepted
  after masking; the original source files remain unchanged.
- File count, byte size, creation time, primary file, format, step counts,
  masked-value count, and preview are plausible for the selected capture.

## Verify the result at the right boundary

For a dry run, report only local validation. For a real public upload, verify
that the trusted validator promoted the digest to
`sources/community/<digest>/`, with `manifest.json` present last and bound to
the uploaded artifacts. A client success message or quarantine write alone is
not production end-to-end proof.

Report whether the capture was uploaded, cancelled, already present, rejected,
or only dry-run validated. Never expose contributor email, detected secret
values, signed upload URLs, credentials, or internal service endpoints.
