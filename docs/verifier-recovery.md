# Verifier recovery

A lost verifier transport should not run an expensive completed solver again. BenchFlow records every script verifier command's startup receipt. The command writes it under `/run/benchflow` in the service that runs the verifier, never in `/logs/verifier`, which a task may require to hold only what its `test.sh` wrote. If the command does not confirm startup within 30 seconds of being issued, the exec layer lost it: a task with an eligible recovery contract (below) reports `verifier_wedge` at once, as verifier infrastructure failure, and recovers in a fresh sandbox; any other task runs the command once more in place, and reports `verifier_wedge` when the second one does not start either. Before receipts, such a task waited out its whole verifier budget, then a second one. A receipt probe that itself fails leaves startup unknown: BenchFlow keeps waiting and probes again instead of cancelling the verifier. Empty stdout alone is never proof of a transport failure: a verifier with a confirmed startup may run quietly for its full configured budget and then receive a normal verifier timeout.

Tasks without an eligible recovery contract keep the in-place retry from PR #949: a verifier that times out with an empty `test-stdout.txt` is rerun once in the same sandbox; a timeout with output is never rerun.

Tasks can opt into verifier-only recovery:

```toml
[verifier]
workspace_recovery = true

[sandbox]
# Use the verified digest of the task's prebuilt image, including Python 3.10+.
docker_image = "registry.example/task@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
```

The equivalent task.md frontmatter is `verifier: {workspace_recovery: true}`.
This declaration is a task-author contract: all solver changes used in scoring
are in the workspace, while the original task image and setup reproduce all
other dependencies. Do not enable it for tasks relying on solver-installed
system packages, global file changes, live processes, external databases, or
other uncaptured state. A workspace archive is not a VM snapshot.

Recovery is limited to owned Docker/Daytona sandboxes; externally supplied
runtimes and physical systems are excluded. The implementation additionally
rejects custom rollout planes, external build contexts, unrecorded runtime
environment values, external skills, pre-compose hooks, setup commands, symlinked task inputs, and separate services/environment manifests,
custom hooks/uploads, MCP services, external artifacts, compose tasks, and
workspace captures containing exclusions. It supports deterministic script
verifiers. Workspace capture uses the existing validated archive and requires
Python 3.10+ already present in the retained baseline image. Package-manager installation is
disabled on this path, so capture setup cannot change the pinned baseline.

For an eligible task, before post-solver capture or teardown, BenchFlow records `solver-complete.json`. This is a stage checkpoint with unfinished telemetry, not a scoring verdict or terminal `result.json`. Resume recognizes it and cannot rerun the solver if the process dies before finalization. A hard deadline in this stage retains the rollout identity and reports incomplete evidence on the scoring channel.

After original sandbox cleanup, BenchFlow preserves immutable `solver.json` and attempts verification in a fresh sandbox using the same task, runtime configuration, and trusted build baseline. It never starts a solver there. Each attempt appends `verifier-recovery/<id>/recovery.json`; `verification.json` references the admitted revision. A trial that is not eligible for recovery gets no attempt directory and no `verification.json`; without automatic review it also gets no `solver.json`, and evaluation resume leaves its `result.json` untouched, as on `main`. Failed original verifier outputs remain in the attempt directory. Successful recovered outputs occupy the normal verifier folder for viewers and reviewers: they are staged beside `verifier/` and swapped in by rename, and the replaced folder moves to the attempt's `previous-verifier/`. If that publication fails, `verifier/` keeps its original files, the recovered verdict stays admitted, and the receipt records `publication_error`.

Interrupted or failed recovery can be retried with:

```bash
bench eval score jobs/<job>/<trial> --tasks-root ./tasks
```

Manually driven `TaskRuntime.verify()` uses the same finalization and recovery
path. The exact original task digest must match. Normal evaluation resume follows the
same verifier-only path. Successful deterministic verification is reused, and
rubric review runs afterward where the task declares a rubric. Solver trajectory,
usage, and the original solver record are preserved across these retries.

Tasks without an eligible recovery contract (no declaration, or a runtime the contract does not cover) keep the ordinary verifier behaviour: errors keep their original text, and verifier infrastructure failures are retried by the evaluation retry loop and on resume, which reruns the solver. For an eligible task, an unavailable capture or failed recovery explicitly retains a `[solver-preserved]` verification error. It does not fabricate a score or silently repeat the solver. Setup failures before solver execution retain normal retries. Old runs without a recorded recovery contract cannot automatically gain one by changing the task: their digest would differ. Tasks that keep deliverables outside the workspace (for example under /root) need their state contract checked before opting in; BenchFlow does not establish that a captured workspace covers all state.

## Current support boundary

Owned Docker runs can retain the original built image, including a task built from a mutable base tag. Before task uploads or agent installation, recovery records the running container's resolved image ID and a same-daemon lease tag. A fresh verifier validates that lease, task identity and effective configuration, starts the exact image with no build or pull, and checks its running image ID. This captures the pre-agent image, not the solver's modified container. Missing or changed leases fail recovery; the code does not rebuild today's Dockerfile. They are not a cross-host backup or a Daytona template identity.

The rollout releases its lease at teardown when no verifier-only recovery is needed, and a recovery attempt releases it once the attempt finishes, whether it succeeded or failed. Release runs `docker image rm <lease tag>` without `--force`: it only removes the tag while other tags or digests reference the image, deletes the image only when the lease was its last reference, and never deletes an image a container still uses. Releasing an already-removed lease is a no-op. An interrupted attempt keeps its lease for `bench eval score`, as does a process that dies before finalization; those leases do not expire automatically.

An explicit file contract, `submission-files-v1`, recovers a task's submission files without archiving or replacing a whole directory such as `/root`. Declare the files in the task's own `task.toml` (not in an agent or sandbox configuration override):

```toml
[verifier]
submission_files = ["/root/result.csv", "/root/summary.json", "/root/report.md"]
```

The SDK hard-codes no file names: each task's list is its contract. The list is validated when the task loads. It holds 1 to 8 distinct paths, and each path must be absolute, canonical (no `.`, `..`, empty or trailing segments), literal (no glob characters such as `*`, `?`, `[` or `{`), at most 512 characters, a file inside a directory rather than a top-level entry, and outside `/dev`, `/logs`, `/proc`, `/solution`, `/sys` and `/tests`. The order of the list is part of the contract: a bundle captured for one declaration cannot be restored under another.

This contract is mutually exclusive with `workspace_recovery`. The task author asserts that the retained baseline, pinned verifier and these files are sufficient for scoring. Solver-installed dependencies or other mutable state are not restored.

After solver disconnection and quiescence, capture records present and missing
files with a separately pinned manifest digest and per-file hashes. Only regular
files with no symlink components are admitted (128 MiB per file, 256 MiB total).
The fresh sandbox validates all inputs and destinations before replacing the
declared files. Missing outputs remain missing; all other baseline contents
stay intact. Replacement is atomic per file, not a
multi-file transaction. A failed restore prevents verification.

`submission.json`, `submission-evidence/` and `docker-recovery-baseline.json`
retain the local recovery evidence. Each verifier attempt uses its own Docker
project identity, so concurrent retries do not share a Compose project.
Existing runs without trustworthy original image and submission receipts cannot
be retroactively recovered using this contract. Numerical verifier recovery and
any subsequent model-based rubric review are separate completion stages.

Initial recovery and CLI resume use the same nonblocking scoring lock. Successful
verdicts are reread under that lock before new work, and task inputs are staged and
checked against the original digest before the fresh verifier starts.
Terminal lifecycle/finalize publication uses that same lock before writing solver
or result artifacts. An already-admitted completed score is loaded without invoking
the stale result writer. Cancellation or deadline abandonment revokes the attempt's
publication admission before releasing the lock; a cancellation-resistant attempt
may finish its local receipt and cleanup, but cannot replace canonical verifier
files or the latest-attempt pointer afterward.

Fresh recovery explicitly restores declared denylist proxy policy and the
sandbox UID firewall without starting an agent or model gateway. No-web tasks
also regain their UID firewall when the original model bootstrap kept container
networking enabled. Policy setup failures abort recovery. These restrictions
apply to the sandbox UID; root verifier behavior remains unchanged. Provider
network configuration and effective runtime identity are also preserved.

## Released leases and failed publication

Two outcomes of the rules above are deliberate:

- **A released lease cannot serve a later retry.** Every recovery attempt that finishes, successful or not, releases the Docker image lease, as do teardown and finalization when no recovery is needed. Only an interrupted attempt, or a process that dies before finalization, keeps it. Running `bench eval score` again after a finished but failed attempt therefore starts a new attempt that stops at sandbox start with `Recovery image lease is missing or changed`. That attempt's receipt is `unavailable`, and the trial reports `[solver-preserved] verifier recovery unavailable: Recovery image lease is missing or changed`. BenchFlow does not rebuild the image to replace the lease, and it still does not rerun the solver.
- **A failed publication keeps the recovered score, not the recovered files.** When swapping the recovered outputs into `verifier/` fails (for example, the disk is full), `verifier/` keeps the outputs it had before recovery, but `verification.json` still points at the attempt, so its recovered rewards are the admitted score. The recovered outputs stay in `verifier-recovery/<id>/verifier/`, and that attempt's `recovery.json` records the failure as `publication_error`. Viewers and rubric reviewers read `verifier/`, so they see the earlier outputs; check `publication_error` before comparing those files with the score.
