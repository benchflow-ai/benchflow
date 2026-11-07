# Reviewing Changes to Sandbox and Security Code

Use this when a change touches the code that sets up, hardens, grades or tears down a sandbox. That covers lockdown, cloud-metadata, firewall, verifier preparation, exec transport, lifecycle, and image recipes. Most of these steps fail closed. A change that only reorganises how they run, such as batching execs, reusing an answer, or moving work into an image, can still change what they decide.

## Evidence that counts

- **A differential reproduction.** It fails on the branch head and passes on the base, through the real code path, with only provider I/O controlled. A passing normal-path test is not evidence of equivalence.
- **The failure model.** A transport error or a timeout does not mean the command stopped. An accepted exec can still be running when the host gives up on it, so any fallback must assume the original may still write.
- **Run time paired with review.** Run the A/B or live measurement alongside the review, not after it. Keep the PR a draft until the review's findings are fixed.

## Defect classes to check, each against the code

1. **A security answer discarded, then re-run.** A batch fails or times out after a step's answer, possibly positive or refusing, has arrived. The fallback runs that step again, and a stateful probe comes back clean.
2. **A status gate in place of host validation.** The step exits 0, but the host's own receipt check, which ran before the next mutation, would have refused. Mutations now run before the refusal.
3. **A check removed at a boundary.** A check that ran at a specific point is dropped on the argument that nothing changes in between, without proof that nothing can. One example: a custody check before root imports or executes.
4. **Reuse across a window.** A listing, identity, proof, output clear or environment value is carried across a window where something can change it:
   - a kill or a freeze;
   - verifier construction;
   - surviving processes;
   - a root oracle's children;
   - filesystems whose change notifications are incomplete, such as overlay lower layers.
5. **The smallest budget with no working fallback.** Steps that each fit their own budget fail together. Or a transport exception on the merged exec ends in refusal instead of the separate path.
6. **Lost diagnostics.** A step's stderr no longer reaches the reason, record or error text it reached before.
7. **Record or schema drift.** A field is added, removed or reshaped in a record such as `cloud-metadata.json`, `result.json`, receipts or exec counters. Check failure paths as well as success.
8. **Ordering.** A step moves past a host decision, a kill, a restore, a cleanup obligation or an upload.

## Rules that kept changes equivalent

- **Merging execs.**
  - Merge only steps that are read-only, or idempotent with a report that doesn't depend on state the batch changes.
  - Merge nothing whose failure would lead to a mutating fallback while the original exec may still be running.
  - Leave host decisions, uploads and cleanup where the base had them.
  - Keep per-step budgets, stderr and records identical.
  - Prefer making each step cheaper over merging steps. That means the transport, cached image layers, or a persistent command channel where one exists.
- **Lifecycle transitions** (start, stop, restore, delete).
  - Commit the provider work and its bookkeeping together, with no await in between: handle release, inspector and trust state, the ledger.
  - Optional work is retired synchronously.
  - Background work ends when its provider's ownership ends. It never writes records or starts tasks after that, and nothing outlives the event loop.
- **Cached image layers.**
  - A layer install that fails or times out must fail that build, so the plain recipe is used. A half-finished package transaction is never baked into a cached image.
  - Remember only confirmed build failures. A transient gateway or read error is not one.
  - The run-time checks stay authoritative.
- **Content caches and digests.**
  - Anything reused at run time is re-proved there against a pinned value, before it executes.
  - A marker file is never an authority.

## Running the review

- Give each fix round a principle, not a list of symptoms, and re-review the fix itself. Fixes that add awaits, pools or background tasks have repeatedly introduced the next defect.
- On shared build VMs the full suite is red on the base for environment reasons, such as group-writable ancestors or missing browsers and tools. Either run the gate in an isolated container with root-owned ancestors, or replay every failure on the base with the same command on the same VM.
- A gate is not green until every failure is reproduced on the base, or explained.
