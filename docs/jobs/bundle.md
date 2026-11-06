# Share a run's evidence

`bench bundle` writes one zip that someone without your job folder can read: what each trial did, how it was graded, and what it left.

```bash
bench bundle jobs/<job>                 # writes jobs/<job>.bundle.zip
bench bundle jobs/<job>/<trial> -o one-trial.zip
bench bundle jobs/<job> --no-workspace  # without the saved workspaces
```

## What is in it

For each trial:

- `result.json`, `config.json`, `timing.json`, and `steps.json` for a multi-step task;
- `trajectory/`: the agent's trajectory and its model calls;
- `verifier/`: test.sh's output, the reward files, judge results, and the plugin guard and trusted-PATH records;
- `reviews/` (rubric reviews), `regrade.json` and `regrade/` (regrades), `artifacts/`;
- the records kept beside them: `trust-state.json`, `locked-system-path.json`, `kill-before-grading.json`;
- `evidence/`: the saved workspace, when the run saved one (`--save-outputs`, a rubric review, a separate verifier).

For a job folder it also holds `run-card.json`, `summary.json`, `report.md`, `evaluation.json` and `results.jsonl`. The agent's own logs (`agent/`) are left out; the trajectory holds what it did.

## Redaction

Every text file is redacted the way a trajectory upload is: API keys, bearer tokens, private keys, passwords, credentials in URLs, and the values of credential-named fields and command options. A file whose redaction does not settle is left out. Links and other non-regular files are left out too, and so is any file over 256 MiB. The bundle stops at 4 GiB; leave the workspaces out to go under it.

Redaction finds what looks like a secret. Read the bundle before you share it, and leave out what you would not publish.

## bundle.json

The first file in the zip lists each file with its size, SHA-256 and the number of values masked in it, the masked values by kind, and each file left out with the reason. The same folder always gives the same zip, byte for byte.
