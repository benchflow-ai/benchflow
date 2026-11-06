# Verifier recovery: internals

This page records how [verifier recovery](../tasks/verifier-recovery.md) keeps its promises: the startup receipt, the solver checkpoint, the image lease, file capture and the scoring lock. Read the task page first.

## Startup receipts

Each script verifier command writes a startup receipt under `/run/benchflow` in the service that runs the verifier. It never writes into `/logs/verifier`, which a task may require to hold only what its `test.sh` wrote.

The verifier user can forge or remove that receipt. If it is still missing after 30 seconds, startup is unconfirmed: the command may not have started, or its receipt may have been removed. After the existing in-place retry where allowed, the trial scores 0 with a named warning. A receipt probe that itself fails leaves startup unknown: BenchFlow keeps waiting and probes again, instead of cancelling the verifier.

Empty stdout alone is never proof of a transport failure. A verifier that confirmed startup may run quietly for its whole budget, and then scores 0 with a timeout warning. Neither signal can discard the trial.

## The solver checkpoint

For an eligible task, before post-solver capture or teardown, BenchFlow records `solver-complete.json`. This is a stage checkpoint with unfinished telemetry, not a scoring verdict or terminal `result.json`. Resume recognizes it and cannot rerun the solver if the process dies before finalization. A hard deadline in this stage retains the rollout identity and reports incomplete evidence on the scoring channel.

## Attempts and publication

For an eligible BenchFlow setup or sandbox API failure, after original sandbox cleanup BenchFlow preserves immutable `solver.json` and attempts verification in a fresh sandbox using the same task, runtime configuration, and trusted build baseline. It never starts a solver there. Each attempt appends `verifier-recovery/<id>/recovery.json`; `verification.json` references the admitted revision. A trial that is not eligible for recovery gets no attempt directory and no `verification.json`; without automatic review it also gets no `solver.json`, and evaluation resume leaves its `result.json` untouched. Failed original verifier outputs remain in the attempt directory. Successful recovered outputs occupy the normal verifier folder for viewers and reviewers: they are staged beside `verifier/` and swapped in by rename, and the replaced folder moves to the attempt's `previous-verifier/`. If that publication fails, `verifier/` keeps its original files, the recovered verdict stays admitted, and the receipt records `publication_error`.

## The image lease

Owned Docker runs can retain the original built image, including a task built from a mutable base tag. Before task uploads or agent installation, recovery records the running container's resolved image ID and a same-daemon lease tag. A fresh verifier validates that lease, task identity and effective configuration, starts the exact image with no build or pull, and checks its running image ID. This captures the pre-agent image, not the solver's modified container. Missing or changed leases fail recovery; the code does not rebuild today's Dockerfile. They are not a cross-host backup or a Daytona template identity.

The rollout releases its lease at teardown when no verifier-only recovery is needed, and a recovery attempt releases it once the attempt finishes, whether it succeeded or failed. Release runs `docker image rm <lease tag>` without `--force`: it only removes the tag while other tags or digests reference the image, deletes the image only when the lease was its last reference, and never deletes an image a container still uses. Releasing an already-removed lease is a no-op. An interrupted attempt keeps its lease for `bench eval score`, as does a process that dies before finalization; those leases do not expire automatically.

## Submission-file capture

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

## The scoring lock

Initial recovery and CLI resume use the same nonblocking scoring lock. Successful
verdicts are reread under that lock before new work, and task inputs are staged and
checked against the original digest before the fresh verifier starts.
Terminal lifecycle/finalize publication uses that same lock before writing solver
or result artifacts. An already-admitted completed score is loaded without invoking
the stale result writer. Cancellation or deadline abandonment revokes the attempt's
publication admission before releasing the lock; a cancellation-resistant attempt
may finish its local receipt and cleanup, but cannot replace canonical verifier
files or the latest-attempt pointer afterward.

## Network policy in the fresh sandbox

Fresh recovery explicitly restores declared denylist proxy policy and the
sandbox UID firewall without starting an agent or model gateway. No-web tasks
also regain their UID firewall when the original model bootstrap kept container
networking enabled. Policy setup failures abort recovery. These restrictions
apply to the sandbox UID; root verifier behavior remains unchanged. Provider
network configuration and effective runtime identity are also preserved.

## Next steps

- [Verifier recovery](../tasks/verifier-recovery.md): opting in and retrying.
- [Separate verifier](../tasks/separate-verifier.md): a different way to grade in a fresh sandbox.
