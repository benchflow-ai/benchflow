# Composed checkpoints

Start with the [Branching guide](branching.md) for which option to use when; this page is the reference for how branching works.

`Rollout.branch` forks a rollout at its current point into N children, runs each child from the same checkpoint, and returns V, the mean of the children's rewards. The checkpoint can capture the declared environment state, the sandbox filesystem, or both:

```python
await rollout.branch(2, run_child=run_child,
                     snapshot_layers={"environment", "sandbox"})
```

## Branch from the command line

`bench eval branch` runs each task up to a checkpoint, forks it into labelled children and scores each child with the task's verifier:

```bash
bench eval branch --tasks-dir tests/examples/hello-world-task \
  --agent claude-agent-acp --model claude-sonnet-5 --sandbox daytona \
  --prompt "Create draft.txt containing exactly one line: Hello world. Do not create hello.txt." \
  --prompt @instruction --checkpoint-after-prompt 1 \
  --child "label=baseline" \
  --child "label=hint-reuse-draft,prompt=Rename draft.txt to hello.txt without changing it, then stop."
```

- `--prompt` gives the parent's prompts in order (`@instruction` is the task instruction; the default is the task's own prompts). `--checkpoint-after-prompt N` branches after the first N of them; `0` branches before the first prompt, which is the default for `--agent oracle` (every oracle child runs the task's `solve.sh`).
- `--child label=NAME[,parent=LABEL][,prompt=TEXT|,prompt-file=PATH]`, at least two at the top level. `prompt=` takes the rest of the option, so it may contain commas. A child without a prompt runs the parent's remaining prompts, or the task's prompts when the checkpoint is after the last one. `parent=LABEL` makes a nested fork (below). By default every child starts a fresh agent session, so its prompt must stand alone; `--resume-session` makes children resume the parent's conversation instead (below).
- `--concurrency K` runs up to K children of a fork at once, each in its own sandbox (below); `--isolate-children` asks for own sandboxes with K = 1.
- `--snapshot-layers sandbox` (default), `environment`, or `sandbox,environment`.
- `--parent continue` (default) restores the parent world after the last child, sends the remaining prompts and verifies. `--parent discard` skips that restore (`restore_parent=False`, below) and leaves the parent unverified.
- `--retain-snapshots` keeps the checkpoint; `--checkpoints every-prompt|prompt:N,M` keeps automatic checkpoints of the parent after those prompts; `--from-checkpoint <trial folder> [--checkpoint ID|prompt:N]` branches again from a kept snapshot (below).

The job folder has the usual layout: one `<task>__<id>` folder per task with `result.json`, `results.jsonl`, `tree.json` and `branches/<fork>/children/<node>/` (`observation.json`, the child's `mounted/` logs and verifier output), plus a job-level `results.jsonl` and a `summary.json` whose `total`/`passed`/`score` count children. The command prints one row per child and V per task, and exits 0 when every fork completed, 1 otherwise, 2 for an inconsistent request. Open the result with `bench eval view <job folder>`: the Lineage tab lists each fork and opens each child's trajectory.

From Python, `bf.branch(task, agent=..., prompts=[...], checkpoint_after=N, children={label: prompt_or_None, ...})` (or `await bf.abranch(...)`) runs the same driver with the same options as keywords and returns a typed `BranchResult` (V, each child's reward and source, the parent's `RolloutResult`); see [Python API: Branching](./reference/python-api.md#branching).

Each child's `intervention.requested` in `tree.json` says what it ran: `parent's remaining prompts (k)`, `the task's prompts (k)`, `oracle solve.sh`, or `own prompt (n characters, sha256:…)`; the prompt text itself is in the child's trajectory.

### Parallel children

`--concurrency K` (`branch(isolate_children=True, concurrency=K)`) runs each child as its own sub-rollout instead of restoring the parent's sandbox between children: the child gets its own sandbox created from the snapshot (Daytona creates it straight from the snapshot; elsewhere the task sandbox is started and the snapshot restored into it), installs the agent (credential files never enter a snapshot), runs, is verified and is torn down, with at most K children alive at once. Each child is a full trial folder at `branches/<fork>/children/<node>` (`result.json`, `trajectory/`, `verifier/`, `observation.json`); `bench eval metrics` and the other result scanners do not count it as an extra trial because it sits below the parent's `result.json`. The parent's sandbox is not touched by the children, and the parent restore after the fork (or `--parent discard`) behaves as before. One failing child does not stop its siblings. Only the sandbox layer can be isolated: declared environment state lives inside the parent's sandbox. On Daytona each child costs one extra sandbox and an agent install, and a fork of n children with K ≥ n needs n + 1 sandboxes at once. `tree.json` records `children_mode: {isolated, concurrency}`, and `timing_sec.children` times the whole children phase.

### Nested forks

`--child label=a1,parent=a,prompt=…` (at least two per parent) forks again from child `a`'s own state after `a` has run its prompts and before `a` is verified; nested children without a prompt run the task's prompts, and nesting implies isolated children. In Python, a runner calls `child.rollout.branch(...)` on an isolated child. The nested fork is recorded in the same `tree.json`: its `parent_node` is a node below child `a`'s node, `rollout` names the sub-rollout that forked (`a`'s node id), and its children archive under the root trial's `branches/`, so tree depth can exceed one while node ids stay unique within the trial. Nesting multiplies sandboxes: each nested fork may run K children of its own.

### Resuming the parent's conversation

`--resume-session` (`branch(resume_session=True)`) makes each child resume the parent's agent conversation with ACP `session/load`, so a child prompt can say "continue". It works for agents that advertise `loadSession` and keep their session on disk, where the snapshot captures it (Claude Code keeps it under `~/.claude/projects`); other agents are refused rather than silently given a fresh session. The replayed history is the shared prefix, so it is not part of the child's trajectory. `tree.json` records `snapshot.agent_session: "resumed"` (`"fresh"` otherwise). It needs a conversation before the fork (not the oracle, not `--checkpoint-after-prompt 0`) and is refused with `--from-checkpoint`, which does not know the source session's id.

### Automatic checkpoints

`--checkpoints every-prompt|prompt:N,M` on `bench eval run` and `bench eval branch` (`RolloutConfig(checkpoints=parse_checkpoint_policy(...))`) takes a sandbox snapshot after the chosen prompts and keeps it, so any run can be branched later with `bench eval branch --from-checkpoint <trial> --checkpoint prompt:N`. `--checkpoint-keep K` (default 3) keeps at most K per trial and deletes the oldest when a newer one is taken. Each checkpoint is listed in the trial's `checkpoints.json` (`id` `prompt:N`, node id, provider, ref, seconds, status `kept`, `deleted`, `failed` or `unsupported`). A checkpoint never fails the run: an unsupported sandbox or a failed snapshot is recorded and the run continues. The snapshot is taken while the agent is idle between prompts, with credential files scrubbed as for branch snapshots.

Cost and storage: each checkpoint is a full filesystem image held by the provider, taking as long as a branch snapshot (longer for a larger filesystem), so `every-prompt` adds that per prompt. Kept Daytona snapshots count toward the account's snapshot storage until deleted; `bench sandbox cleanup` (and the eval-start auto-reap) deletes an owner's Daytona snapshots once they are older than the reap age (`--max-age`, default 24 h), and also removes Docker `bf-snap-*` images older than `--max-age` that no container uses.

### Branch again from a kept checkpoint

A trial run with `--retain-snapshots` keeps its sandbox snapshot (`snapshot.retention: kept` in `tree.json`), and a run with `--checkpoints` keeps its automatic checkpoints (`checkpoints.json`). `--from-checkpoint <that trial folder>` starts a new trial of the same task (`--tasks-dir` must contain it), replaces its fresh sandbox with one restored from the kept snapshot, installs the agent again and branches there, so new children can be compared against the old ones without re-running the parent. The new trial records `checkpoint_source.json` (source trial, fork id, provider, snapshot ref); the agent and model default to the source trial's. Limits:

- Only sandbox snapshots can be reused, on the provider that made them (a Docker image on the same host, or a Daytona snapshot in the same account). A fork that also captured the environment layer is refused: declared database state is a provider-local handle with no export.
- The agent's on-disk transcript is in the snapshot but its session id is not recorded, so children start fresh sessions (`--resume-session` is refused); credential files are scrubbed at capture, and the new trial installs the agent again.
- `--checkpoint prompt:N` (alias of `--fork`) picks an automatic checkpoint; by default the last kept fork, else the last kept checkpoint. A deleted checkpoint is refused.
- The parent defaults to `--parent discard`, because there is no parent conversation to continue; `--parent continue` sends the task's prompts from the checkpoint.
- The kept snapshot is not deleted by the new trial. `bench sandbox cleanup` (and the eval-start auto-reap) deletes an owner's Daytona snapshots once they are older than the reap age, and Docker `bf-snap-*` images older than `--max-age`.

## Branch a real agent run from Python

Drive the rollout lifecycle yourself, in this order:

`setup` → `start` → `install_agent` → `connect` → `execute` (first prompt) → `branch` → `connect` → `execute` → `verify` → `finalize`

`branch()` disconnects the agent before the checkpoint and leaves it disconnected, so the parent calls `connect()` again afterwards. `finalize()` cleans up and writes `result.json`; `cleanup()` alone writes no result. A plain task (no environment manifest) branches with `snapshot_layers={"sandbox"}`, which checkpoints the container filesystem on Docker and on Daytona in direct mode.

```python
from benchflow.rollout import BranchChild, Rollout, RolloutConfig

rollout = Rollout(RolloutConfig(task_path=task, agent="claude-agent-acp",
                                model="claude-sonnet-5", environment="docker"))
prompts = {"baseline": instruction, "hint": "Rename draft.txt to hello.txt."}

async def run_child(node, *, child: BranchChild) -> float | None:
    rewards = None
    try:
        await rollout.connect()                      # a fresh agent session
        await rollout.execute([prompts[child.label]], node=node)
        rewards = await rollout.verify()             # the task's own verifier
    finally:
        await rollout.disconnect()
    return (rewards or {}).get("reward")             # None marks it unscored

try:
    await rollout.setup()
    await rollout.start()
    await rollout.install_agent()
    await rollout.connect()
    await rollout.execute(["Write draft.txt containing: Hello world"])
    value = await rollout.branch(2, run_child, snapshot_layers={"sandbox"},
                                 child_labels=list(prompts))
    await rollout.connect()                          # branch() left it disconnected
    await rollout.execute([instruction])
    await rollout.verify()
    result = await rollout.finalize()                # result.json and tree.json
except BaseException:
    await rollout.cleanup()
    raise
```

The complete script is [`examples/branch-agent-run.py`](examples/branch-agent-run.py) (`uv run python docs/examples/branch-agent-run.py --sandbox docker`); it branches the bundled hello-world task with Claude, one child per label.

**Children start fresh agent sessions** unless `resume_session=True`. Only the sandbox filesystem and the declared database state are restored, not the agent's conversation. A child prompt such as "continue" or "now finish the task" reaches an agent that never saw the first prompt, so write every child prompt to stand on its own, as above. `tree.json` lists `agent_session` under each fork's `snapshot.excluded`.

### The child runner

- `run_child(node)` receives the pending tree node of the child. Declare a keyword-only `child` parameter (or `**kwargs`) to also receive a `BranchChild` with `index`, `label` (from `child_labels`), `node` and `fork_id`, so per-child prompts need no call counter.
- Pass `node=node` to `execute()`. Inside a child, `execute()` without `node` fills the pending child node the same way.
- Return the child's reward. When the runner calls `verify()`, a returned number equal to the canonical verifier reward is recorded as `reward_source: "verifier"`, any other number as `runner_return`; a verifier that produced no canonical reward makes the child unscored whatever the runner returns. Returning `None` also marks the child unscored. A missing reward is never converted into zero.
- Omit `run_child` to use the default runner: a fresh session, the task's own prompts, and the verifier.

### Skipping the parent restore

`branch(..., restore_parent=False)` (the CLI's `--parent discard`) is for callers that finish after the fork. The parent restore after the last child is skipped, so the fork restores n times instead of n+1. The world then holds the last child's state, so the rollout is marked discarded: `tree.json` records `parent_restore: "skipped"`, and every later `setup`, `connect`, `execute`, `verify` or `branch` raises. `finalize()` and `cleanup()` still work; the parent's `result.json` describes the parent up to the fork, with no reward unless it was verified before branching. The snapshot is still deleted (unless retained), and a child failure still leaves the rollout discarded.

The first child is restored from the snapshot too, although the live world was just captured: processes started before the checkpoint (a server the agent left running, for example) keep running in the live sandbox but not in a restored one, so skipping that restore would let the first child see a different world from the others.

`child_requests=[...]` records, per child, what the caller's runner does differently (at most 200 characters each) as `intervention.requested` with `execution: "runner"`.

### Cost

`branch(n)` captures once and restores n+1 times (each child, then the parent), or n times with `restore_parent=False`. Each fork records `timing_sec` in `tree.json`: `checkpoint`, one `child_restore` per child and `parent_restore` (null when skipped). The first restore after a capture is usually the slowest, because the new snapshot is still becoming available, so `restore_parent=False` saves one restore per fork. Snapshot and restore times grow with the sandbox's filesystem size, from seconds for a small sandbox to minutes for one holding several GB.

## Layers

The default remains `{"environment"}`, which needs an environment manifest; a task without one fails with an error that names `snapshot_layers={"sandbox"}`. Requesting `{"sandbox"}` alone permits a stateless environment to branch without invoking its unsupported environment snapshot methods. Empty or unknown layer sets fail before the agent disconnects. The older `require_sandbox_snapshot=True` checks capability only; it does not add a captured layer. Use `snapshot_layers` to request actual container capture.

Capture runs environment first, then sandbox; restore runs sandbox first, then environment. This preserves the in-container SQLite backup files. Collision-safe `StateSnapshot.files` mappings remain attached to the environment handle. Each child begins at the checkpoint and the parent world is restored after the fork, including child failure. If both the child and parent restoration fail, both errors are retained in an exception group. Runtime restore failures can leave a partially restored world and must not be treated as a successful rollback.

Docker uses a committed image and recreates its main container, preserving the supported host configuration and mounts. It rejects unsupported reconstruction before removing the old container. Daytona direct advertises snapshot support; Daytona DinD and other unsupported backends fail the capability check. The local Docker proof does not establish remote-provider equivalence.

These are filesystem and declared database checkpoints. They do not capture agent-session memory, mounted contents, sibling services, running-process memory, physical robot state or simulator state unless that state has its own validated capture implementation. The engine disconnects each child's session before artifact handoff, including custom runners that return or raise without doing so. Framework-started ManifestEnvironment services restart after container/database restoration and must pass readiness before execution continues. SQLite restoration after container recreation removes captured WAL, shared-memory, and rollback-journal sidecars before installing the standalone online backups; otherwise later service writes in the container image could replay over them. This cleanup requires stopped database processes and is not applied to the environment-only restoration path. Entrypoint-owned service lifecycles are rejected before capture because their restart semantics are not validated. This does not establish cross-service transactional snapshots. Container snapshot images remain provider-local handles with no export/import. The engine deletes the fork's container snapshot when the fork finishes, on every exit path (completed, failed, cancelled), and records the outcome as `snapshot.retention` in `tree.json`: `deleted`, `deferred` (still in use by the restored parent, so the sandbox deletes it when it stops; always the case on Docker), `delete_failed`, or `kept` when the caller passed `retain_snapshots=True`. Daytona snapshot names carry the owner scope (`bf-snap-<owner>-<owner hash>-…` when `BENCHFLOW_DAYTONA_OWNER` is set), and `bench sandbox cleanup` plus the eval-start auto-reap delete that owner's snapshots once they are older than the reap age, including kept ones.

Agent credential files never enter a container snapshot. Before capture, the sandbox reads the files listed in `CREDENTIAL_EVIDENCE_PATHS` (`benchflow/agents/credentials.py`) under `/root` and every `/home/<user>`, for example `~/.codex/auth.json` and `~/.claude/.credentials.json`, into host memory, removes them, checks none remain, captures, and writes them back with their owner and mode: to the live sandbox, and to every sandbox restored from that snapshot. Contents never pass through a command line; on Daytona they move through the file API, because the Daytona daemon keeps session command output on disk inside the sandbox. A credential path that is a symbolic link is refused. Scrubbing was chosen over refusing the sandbox layer because each child starts a fresh agent session that needs those files: refusing would rule out sandbox branching for codex-acp and for Claude host-login users. Native Claude OAuth through `CLAUDE_CODE_OAUTH_TOKEN` writes no file at all. A kept snapshot restored by another process has no credential files; install the agent again there. Credentials a task places elsewhere are not covered.

In-place children hold the parent's mounted log entries aside, keeping the mount directories themselves intact. Each invocation uses a unique directory under `branches/`; children receive their own host trajectory directory, an `observation.json` record, and a `mounted/` archive. Parent entries and mutable result state (including timing, native usage and solver recovery markers) are restored after child or environment restore failure when the child was quiesced. If disconnect or provider cleanup cannot finish within its bounded grace, the parent's mounted entries remain safely held under the fork directory; they are not restored over potentially live child writers. World restoration is also deferred (`parent_restore: deferred`), so a live child cannot write over an apparently restored database or sandbox. The rollout retains an unsafe-world marker and refuses further setup, connection, execution, verification or branching. A failed parent restore also retains that marker. Dispose of and recreate the rollout; a late cleanup completion alone does not establish successful rollback. Partial custody failures also retain evidence under the fork directory and surface the failure. A missing canonical reward is unscored and raises; it is never converted into zero.

Already-active shared provider runtimes are rejected before branching: background callback writers and cumulative usage need an explicit runtime fork contract. The bounded custody path supports native-subscription/provider-free continuations; this is not general proxy-runtime isolation. Nested shared-instance forks are rejected.

`tree.json` is an atomic, versioned observation record. Each invocation retains its own fork ID, snapshot scope, children, outcomes and parent restoration status. The mean uses only that invocation's children. Failed, cancelled and unscored children retain null rewards; remaining arms are explicitly not started. `child_labels=["baseline", "variant"]` adds caller annotations with intervention execution marked `unspecified`; `child_requests` adds a description with execution `runner`. Neither executes a modification: the runner does. Each child entry also records its native token counters (`usage`) and phase seconds (`timing_sec`). `result.json` keeps `final_metrics` and `timing` for the parent's own turns and adds a `branches` block pointing at `tree.json`, with each fork's status, value, child counts, `parent_node`, `parent_restore` and per-child `nodes` (index, node id, label, status, reward, reward source, archive path), plus the children's summed `child_usage` and `child_timing_sec`, so cost dashboards can count branched work. The rollout's `results.jsonl` row carries the same block as `info.branches`, and each child's `observation.json` has a `lineage` block (parent rollout, fork id, parent node, index, label).

The record includes only node/step IDs and allowlisted fork metadata; it excludes arbitrary node state, step payloads, configs and exception messages. Child artifact links appear only after observation writing and custody succeeded. A publication failure preserves the previous valid file and surfaces alongside any primary execution failure. A last record that remains `running` is incomplete. Snapshot refs remain provider-local observations with unknown future restore availability; `tree.json` is not a durable checkpoint import or replay format.

The composed primitives were selectively adapted from JeremyJC67's [PR #1046](https://github.com/benchflow-ai/benchflow/pull/1046), retaining the current engine's parent rollback and disconnect fixes. Stage capture, executable branch deltas, ablation CLI, replay and viewer integration are separate features.
