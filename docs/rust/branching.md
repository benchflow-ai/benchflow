# Checkpoints and rollout branching

`bench branch` runs a task up to a fork point, stores the sandbox's files as a checkpoint, and runs several children from it, each in a sandbox of its own and each graded by the task's verifier. The fork's value is the mean of its children's rewards, and each child's advantage is its reward minus that value. The records are BenchFlow 0.8's (`tree.json`, `checkpoints.json`, `branches/`), so tools that read 0.8's branch jobs read these too.

Forks run at `start` (after setup, before any agent has run), at `prompt:N` (after the parent's N-th prompt, for model agents), or at `stage:<name>` (after a graded stage of a task with stages). Children run in sandboxes of their own. The parent is discarded after its fork, or with `--parent continue` goes on in its own sandbox while the children run, and is graded too. Children that run in place in the parent's sandbox are not built yet.

## Run a fork

```console
$ bench branch tasks/hello --children 3 -s docker -j 3
Branching oracle at start into 3 children · docker · 3 at a time
  - hello   –   6.1s  not graded: the parent was discarded after its fork (its children are graded)
  ✓ hello   1   6.1s
  ✓ hello   1   6.0s
  ✓ hello   1   6.2s
  hello__a4cb98cb start  fork 9a5ee9b8  completed  value 1.000  checkpoint docker (0.9s, deleted)
    n1   c1               scored      1.000
    n2   c2               scored      1.000
    n3   c3               scored      1.000
  job jobs/2026-10-10__19-00-00
```

- `-a <agent>` picks the attempt: `oracle` (the default), `nop`, `script:<file>`, or a model agent with `-m <model>`.
- `--children K` makes K children (two or more, as 0.8 requires: a group of one has no advantage) that each get the task's own instruction, labelled `c1` to `cK`.
- `--child label=<name>[,prompt=<text>|prompt-file=<path>]`, once per child, gives each child a label and, for model agents, a prompt of its own. `@instruction` in a prompt stands for the task's instruction, so `prompt=@instruction Start with the tests.` adds a hint. A prompt runs to the end of the value, commas included.
- `--at prompt:N` forks after the parent's N-th prompt instead (model agents only). Without its own prompt, a child sends the task's instruction again, as 0.8 does after the last prompt. Each child starts a fresh agent session in the parent's files.
- `--resume-session` (with `--at prompt:N`) makes each child continue the parent's conversation instead: the child's agent loads the parent's session with ACP `session/load` and then gets the child's prompt. The agent must be an ACP agent that offers `loadSession` (the manifests of OpenCode, Goose, Claude Code, Codex, and Gemini CLI say theirs do; OpenCode was checked live); the parent is refused right after `initialize`, before its first prompt, when it does not, and a child whose `session/load` fails fails rather than starting a fresh session (it is `failed`, not a scored 0).
- `--at stage:<name>` forks after the named stage of a task with graded stages (`[stages.*]` in task.md, or `[[steps]]`). The stage must be graded and not the last: its handoff has stopped every agent process and checked the sandbox, and its verifier has graded it. Each child runs the stages after it, in order, and its `result.json` carries the parent's earlier stages in `step_results` (each marked `from_parent: true`), so its reward follows the task's rule over all stages. A child's own prompt replaces the next stage's prompt (`@instruction` is that stage's prompt); with `--resume-session`, the child's first stage resumes the parent's session. The oracle and nop work here too (they run each later stage's solution, or nothing), as do ACP agents that say what their login leaves; other model agents are refused. A child's `prompts.json` lists the parent's prompts before its own, as its `step_results` do. The child carries the parent's time-out too: under `on_timeout = "fail"` a stage that ran out of time before the fork zeroes every child, as it would the run. Stages after the fork place their own files and mounts in each child.
- `--parent continue` lets the parent go on after the checkpoint in its own sandbox, at the same time as its children: after `start` it runs its attempt, after `prompt:N` it is graded, after `stage:<name>` it runs the remaining stages. Its reward is in `tree.json` (`parent_reward`) and its own `result.json`, never in the fork's value, which is the children's. If its checkpoint is refused at the fork point (a value of the run in a file, its agent out of time, a mount Docker cannot commit), the fork is recorded as `failed` with the refusal and the parent still goes on and is graded as a plain trial; it stops only when its state folder was changed (R314) or its agent's processes survived the kill (R305), which void its own grade too. A child's own prompt gets the task's `prompt_prefix` first, as 0.8's children do. 0.8's `bench eval branch` continues the parent when `--parent` is absent, and so does its spelling here.
- `-e KEY=VALUE` (0.8's `--agent-env`) reaches the agent in the parent and every child, as `bench run --env` does, and its values are on the secret search's list.
- `-j N` runs N children at once (a continuing parent runs beside them).
- `--retain-checkpoints` keeps the checkpoint after the fork; `bench cache` lists it.

`bench branches <job>` prints each fork again: value, checkpoint, and each child's label, status, reward, and advantage (`--json` for a document). `bench view <job>` shows the same on its `/branches` page (every fork in the jobs on view, with the parent's fate and reward), and each child's label opens the child's trajectory and verifier output; a parent's run page lists its forks too.

A fork after a prompt, on Daytona, with OpenCode on the Vercel AI Gateway:

```console
$ bench branch tasks/hello -a opencode -m vercel/deepseek/deepseek-v4-flash-0731 -s daytona \
    --at prompt:1 --resume-session -j 2 \
    --child "label=keep,prompt=Check that /app/hello.txt still holds exactly Hello, world! and fix it if not." \
    --child "label=bye,prompt=Now append a second line containing bye to /app/hello.txt."
  hello__d37840d0 prompt:1  fork c68b0fb0  completed  value 0.500  checkpoint daytona (37.4s, deleted)
    n1   keep             scored      1.000
    n2   bye              scored      0.000
```

BenchFlow 0.8's spellings still work: `bench eval branch --tasks-dir t -a oracle --checkpoint-after-prompt 0 --child label=a --parent discard --isolate-children --retain-snapshots ...` runs `bench branch t --agent oracle --at prompt:0 --child label=a --parent discard --retain-checkpoints ...`, and `bench eval branches <job>` runs `bench branches <job>`. Where 0.8's default differs, it is filled in (after prompt 1 for model agents; a continuing parent without `--parent`; one child at a time without `--concurrency`; Docker without `--sandbox`) or named as an error: 0.8 runs `claude-agent-acp` without `--agent`, and runs children in place without `--isolate-children` (or `--concurrency` above 1). 0.8 runs a continuing parent after its children; here it runs beside them, so `-j N` uses up to N+1 sandboxes. A 0.8 flag with no counterpart yet (such as `--from-checkpoint`) is an error that names it.

## What a checkpoint holds

A checkpoint is the sandbox's filesystem: on Docker and Podman, an image made with `docker commit`; on Daytona, a snapshot. Processes, memory, and anything mounted are not in it. A child is a new sandbox started from it.

What the parent does at the fork point, in order:

1. After prompts, the agent's ACP connection has already been shut down cleanly and the key forwarder stopped. The parent stops every remaining process of the agent user (the same kill as before grading, which reads `/proc` and needs no `pgrep`) and records their names; the fork is marked `lossy` when anything had to be killed. If any survives, the fork is refused (the files could still be changing).
2. It deletes BenchFlow's own archives from its root-only state folder and, after prompts, the files the agent's login and launch wrote, and its family's credential files (`~/.claude/.credentials.json`, `~/.codex/auth.json`, `~/.gemini/oauth_creds.json`) whatever login was chosen: a login from a variable has no file of its own, but the agent may write or refresh one during its turn, holding a rotated token the value search below would not know. Root never removes a file through a link: when a login, settings, or credential file has a link at the home, above it, or among the folders on its way (the agent's, made during its turn), or cannot be removed, the file is left in place and the fork is refused (R401, the agent's), since such a file may hold a token the agent refreshed during its turn, which the search for secret values cannot know. `login_files_removed` in `tree.json` counts the files that were there and are gone. A child's launch writes the files again as the agent user, as every launch does (root stages them in `/var/lib/benchflow-place` and the agent user copies them into its home), so a link there gains the agent nothing. A parent whose agent ran out of time during its prompts is refused: its state is not "after prompt N".
3. It checks that BenchFlow's state folder (`/var/lib/benchflow`, which holds the baseline the handoff grades against) is still root's alone, as the handoff before grading does. Children never run setup again, so they keep this folder from the checkpoint; after prompts, a change to it is the agent's (R314) and the fork is not taken.
4. It searches the whole filesystem, as root, for the run's secret values: the task's sandbox variables whose names look secret (`*_KEY`, `*_TOKEN`, `*_SECRET`, `*PASSWORD*`, …), every `-e` value, the login's secrets, and the key the forwarder held (each line of a multi-line value separately). Files under 64 MB are searched, and the agent user's files of any size. If any file holds a value, the fork is refused with R401 and the file is named, because the checkpoint would keep it. If the image lacks the tools to search (`find`, `xargs`, `grep`), the fork is refused too. The search finds values as written, not encoded (base64, compressed) copies.
5. It stores the checkpoint. On Daytona it first deletes every command session the sandbox's daemon keeps and waits until none is left (each keeps its commands' output, the agent's included, on the sandbox's disk); if some stay open after 60 seconds, no snapshot is taken. On Docker, every variable the sandbox was started with is blanked in the image (`ENV NAME=`), so the image does not carry the host's values. Each child sets them again. The image carries the label `ai.benchflow.kind=checkpoint` and BenchFlow's owner labels. On Daytona, the snapshot is polled until it is active, and a snapshot that ends in `error`, `build_failed`, or `inactive` fails the fork.

What a child does before its agent starts:

- It confirms the agent user has the parent's uid. The agent user, the folders locked to root, and the baselines BenchFlow took before any agent ran all come from the checkpoint; they are never taken again in the child. It checks that the state folder came through the checkpoint root's alone and with the parent's baseline in it; if not, the child fails (R103, the sandbox's) before its agent starts.
- It sets the memory limit and the out-of-memory order again, and enforces the parent's network plan again (firewall rules, host pins, and Daytona's allow list), including its self-test.
- It runs the task's readiness check, if the task has one, in the verifier's environment.
- Skills are not placed again (they are in the checkpoint).
- Then the attempt runs as in `bench run`: the login files, the settings files BenchFlow writes from templates (so edits the parent's agent made to them do not carry over), and the key forwarder are all set up again at the agent's launch, and the handoff and verifier apply every defense. After prompts, the agent already worked in this sandbox, so a home it left unwritable makes the child's launch fail as the agent's (R205), as in a later stage.

A child that fails before its agent did anything (while its sandbox starts or in setup) because of the sandbox backend or the model provider (but not a provider failure, R214, which repeats) is run once more, never after a usage limit; the first attempt's folder is kept beside it as `n<k>.attempt-1`. A child whose agent ran out of time is `failed` and unscored, as in 0.8, even when its verifier graded it (its own `result.json` keeps that grade), so the fork is then `partial` with no value. A child whose agent crashed or failed its turn (R204, R205, during its turn; an R205 from placing its files in setup, a home the parent's agent broke, is `failed`) is scored: 0, or in a task with stages the task's own rule over its stages (those before, the parent's included, as graded; the failed stage 0; those it never reached per `unreached`), and its row in `results.jsonl` carries that reward; one whose agent never installed, never started, went silent, or never reached its model (R201 to R203, R206) is `failed` and unscored, as are infrastructure failures, and the fork is then `partial` with no value.

On Daytona, a child's sandbox gets the resources of the sandbox the snapshot was taken from. If it gets less than the task asks for, the child fails rather than running smaller.

## What is refused

Before any sandbox starts, `bench branch` names each reason it will not fork:

- the backend cannot take checkpoints;
- `--runtime` is set (no live test yet shows a parent's files reaching a child under another runtime);
- the task has sidecar services (declared in the task or in its Compose file), which a checkpoint does not hold;
- the task's verifier runs in a sandbox of its own;
- `prompt:N` in a task with graded stages (fork after a stage with `stage:<name>`) or with more than one prompt;
- `stage:<name>` naming no stage of the task, an ungraded stage, or the last stage; or after a stage whose files include a `setup.sh`, or that has a readiness check, at or before the fork (what they started dies at the checkpoint, and a child never runs them again), or in a task with a readiness check;
- the agent runs as root, since nothing then keeps it out of BenchFlow's state before the checkpoint (the parent also refuses at the fork point if the agent user's separation from root is not enforced);
- `--keep`, which would leave sandboxes behind;
- the oracle, nop, or a script with a child's own prompt (they take no prompt), or at `prompt:N` (they have no prompts);
- `prompt:N` with an agent that does not say which files and values its login leaves (today, every agent but the ACP ones);
- `prompt:N` past the parent's last prompt, or in a task with a readiness check (the services it waits for die at the checkpoint, and nothing restarts them in a child);
- `--resume-session` at `start` (there is no session yet), or with an agent that cannot resume a session over ACP;
- two children with one label, or no children;
- concurrent children, or a parent that continues after `start` or a stage beside its children, of an agent that would log in with a subscription login whose tokens rotate (Claude, Codex, Gemini); use `-j 1` (or `--parent discard`) or a key.

On Docker and Podman, a container with mounts, or an image with `VOLUME` paths, is refused at the fork point: a commit leaves those paths out.

## Names, cleanup, and `bench cache`

Every checkpoint is named after who made it, on every backend (Daytona snapshots carry no labels):

```text
bf-ckpt-<owner, lowercased>-<sha6 of owner>-<sha6 of machine>-<pid>-<uuid12>
```

The owner is BenchFlow's owner scope (`BENCHFLOW_OWNER`). The machine is a hash of this machine's id (`/etc/machine-id` on Linux, the platform UUID on macOS, where there is one), its host name, and on Linux its PID namespace: two machines with one host name differ by machine id, and two containers from one image that baked a machine id in differ by host name and PID namespace, so neither takes the other's checkpoints for its own dead run's. Cloned machines that share all three still collide, as before. A kept checkpoint carries pid `0`.

- A fork deletes its checkpoint when its children are done, on success and on failure, unless `--retain-checkpoints`. The records say what happened (`retention`: `deleted`, `kept`, `deferred` when the backend accepted the delete but had not finished it, `delete_failed`).
- At the start of each `bench branch`, the sweep deletes this owner's checkpoints from this machine whose process has ended. Checkpoints from other machines (and those an earlier build named after the host name) are deleted after 24 hours, and kept ones after 7 days. Another owner's checkpoints are never touched.
- Checkpoint images count toward `BENCHFLOW_IMAGE_CACHE_MAX_GB`, but the image-cache trim never removes them.
- `bench cache` lists this owner's checkpoints (Daytona's too when `DAYTONA_API_KEY` is set), and `bench cache clean --only checkpoints --remove` deletes them. A plain `bench cache clean` leaves checkpoints alone. Cleaning skips checkpoints whose run is still going, and kept checkpoints or those made on another machine unless `--older-than` is given.

## Records

The trial folder of the parent holds:

| File | What it holds |
|---|---|
| `tree.json` | `kind: "benchflow-branch-tree"`, `schema_version: 1`, the nodes (`root`, `n1`…), and each fork: `fork_point`, `snapshot` (provider, `ref`, `retention`, `agent_session`: `fresh` or `resumed`, `lossy`, the names of processes stopped), `status` (`completed`, `partial`, `failed`, `cancelled`), `value` (only when every child is scored; also `value_stderr`, `value_scored`, and `scored`), `cost`, `timing_sec`, and each child with its `intervention.label`, `intervention.requested` (as 0.8 words it: `own prompt (N characters, sha256:…)`, never the prompt's text), `status`, `reward`, `reward_source`, `usage`, `cost`, and `artifacts.path`; errors carry `type` and `code` only, as 0.8's display record does (the message is in the trial's own `result.json`, and a continuing parent's refusal in `branch-refusal.json`) |
| `checkpoints.json` | `kind: "benchflow-checkpoints"`, `schema_version: 1`, `rollout`, `policy`, and one row per checkpoint: `id` (`prompt:N`; after a stage, N counts the stages run), `after_prompt`, `agent_session_id`, `trajectory_events`, `node_id`, `provider`, `ref`, `seconds`, `status` (`kept`, `deleted`, `delete_deferred`, `delete_failed`, `failed`), `size_bytes`, and `fork_id`, `fork_point`, `depth`, `lossy` |
| `branches/<fork id>/labels.json` | each child's label, folder, status, and reward |
| `branches/<fork id>/children/n<k>/` | the child's full trial folder (as `bench run` writes one) plus `observation.json` (0.8's child archive) with its lineage (`parent_rollout`, `fork_id`, `parent_node`, `index`, `label`), reward, timing, usage, and its own trajectory events, which `bench view` shows when the child is opened from the parent's run page |
| `result.json` | the parent's own record, with 0.8's `branches` block: the forks, each child's status and reward, the children's summed usage and timing, and `parent: "discarded"` (or `"kept"` for a continuing parent, whose record is then graded like any trial's, and for a parent that never reached its fork point; `parent_restore` is `skipped`, `not_needed`, or `not_attempted`, and `parent_reward` is set whenever the parent was kept and scored) |

While a fork runs, its records are written again whenever the parent or a child ends (the fork `running`, with no value; children not yet ended as `not_started`), and the job's `summary.json` and `results.jsonl` after each fork, so a run that is killed leaves readable records of what ended; such a job is not resumed (a second `bench branch` into its folder is refused). The job folder's `results.jsonl` has one row per child, and one per continuing parent, in `bench run`'s shape, with `info.branch` saying where the row sits (`fork_id`, `node_id`, `role`, `label`, `child_status`, `child_reward`, `advantage`); a discarded parent has no row. The job folder's `lock.json` lists the parent and every child; a child's entry has `branched_from: {trial, checkpoint, session_id}` (`session_id` is the parent's session when the children resumed it, else `null`). `summary.json` is 0.8's `benchflow-branch-job` summary, whose counts are over children.

## Costs

A fork costs one sandbox for the parent (setup, its prompts if any, and the checkpoint: about 1 second to commit a small image on Docker; on Daytona about 4 seconds for a small image and 37 seconds for a 1.2 GB one), plus one sandbox per child. On Docker a checkpoint image is as large as what setup added to the task image. On Daytona the snapshot is stored until it is deleted.
