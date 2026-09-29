# Branch view: the JSON contract for branched runs

`benchflow.branch-view/1` is one JSON document per trial that describes its branches: every fork (including nested forks and checkpoint retries) and every child, with what the child was asked to do, its reward and advantage, tokens, USD, sandbox-seconds, timings, what it reused from the snapshot, and where its files are. Viewers and scripts should read this instead of the raw `tree.json` and `result.json`, whose fields grew over several releases; fields an older run lacks are `null`.

Get it from Python, the CLI, or compute it from the files (table below):

```python
import benchflow as bf
view = bf.load_trial("jobs/branch-…/hello-world-task__0000abcd").branch_view
views = bf.load_job("jobs/branch-…").branch_views()      # every branched trial
from benchflow.branch_view import load_branch_view, BRANCH_VIEW_SCHEMA  # JSON Schema
```

```bash
bench eval branches jobs/branch-…            # a table per branched trial
bench eval branches jobs/branch-… --json     # the documents, as a JSON list
```

Privacy rules: snapshot refs (capability handles) never appear; prompts never appear (`requested` is a short description with a sha256 prefix; the prompt text is in the child's trajectory); errors are `{type, code}` only.

## Shape

```json
{
  "kind": "benchflow.branch-view",
  "schema_version": "1.1",
  "schema": "benchflow.branch-view/1",
  "trial": {"name": "hello-world-task__0000abcd", "task": "hello-world-task",
            "reward": 1.0, "retry": null},
  "totals": {"forks": 1, "children": 4,
             "cost": {"tokens": 160000, "usd": null, "usd_known": false,
                      "sandbox_seconds": 450.0}},
  "forks": [{
    "id": "00000000000000000000000000000001", "kind": "fork",
    "reason": null, "checkpoint": null, "parent_node": "n4", "depth": 1,
    "forked_by": {"rollout": "hello-world-task__0000abcd", "child_node": null, "child_label": null},
    "status": "completed", "value": 0.5, "parent_restore": "restored", "error": null,
    "children_mode": {"isolated": true, "concurrency": 2, "prewarm": 2},
    "snapshot": {"layers_requested": ["sandbox"], "layers_captured": ["sandbox"],
                 "agent_session": "fresh", "retention": "deleted", "provider": "daytona"},
    "timing_sec": {"checkpoint": 45.0, "child_restore": [18.5, 18.5, 4.0, 4.0],
                   "children": 108.0, "parent_restore": 1.5},
    "cost": {"tokens": 160000, "usd": null, "usd_known": false, "wall_seconds": 150.0,
             "parent_sandbox_seconds": 150.0, "children_sandbox_seconds": 300.0,
             "sandbox_seconds": 450.0},
    "children": [{
      "node_id": "n7", "index": 2, "label": "hint",
      "requested": "own prompt (153 characters, sha256:000000000abc)", "execution": "runner",
      "status": "scored", "reward": 0.0, "reward_source": "verifier", "advantage": -0.5,
      "error": null,
      "cost": {"tokens": 40000, "usd": null, "sandbox_seconds": 75.0},
      "usage": {"n_input_tokens": 4, "n_output_tokens": 150, "n_cache_read_tokens": 38846,
                "n_cache_creation_tokens": 1000, "total_tokens": 40000},
      "timing_sec": {"agent_setup": 3.5, "agent_execution": 5.5, "verifier": 27.5,
                     "sandbox_from_snapshot": 4.0, "install_agent": 5.5, "finalize": 1.5},
      "snapshot_start": {"agent": "reused", "verifier_baseline": "inherited",
                         "setup_commands": "skipped"},
      "archive": {"path": "branches/0000…/children/n7",
                  "observation": "branches/0000…/children/n7/observation.json",
                  "result": "branches/0000…/children/n7/result.json"},
      "nested_forks": []
    }]
  }]
}
```

(Illustrative output for the bundled hello-world task with synthetic ids and round numbers; one of four children shown.)

## Fields and where they come from

All paths are relative to the trial folder. "Fork" means an entry of `tree.json` `forks[]`; "child" an entry of its `children[]`.

| View field | Source | Meaning; null when |
|---|---|---|
| `trial.name` | `result.json` `rollout_name`, else the folder name | |
| `trial.task`, `trial.reward` | `result.json` `task_name`, `rewards.reward` | the trial's own reward, never a retry's |
| `trial.parent`, `trial.unscored_by_design` (1.1) | `result.json` `branches.parent`, else derived from the trial's own forks' `parent_restore` | `discarded` when a fork of the trial skipped restoring the parent (`--parent discard`), else `kept`; null without forks. `unscored_by_design` is true when the parent was discarded and the trial has no reward: leave such trials out of attempted-and-unscored counts |
| `trial.checkpoint_source` (1.1) | `checkpoint_source.json` (`load_branch_view`; `build_branch_view(..., checkpoint_source=)`) | for a trial started from a kept checkpoint (`--from-checkpoint`): `{trial, checkpoint, provider, prefix_events}`, where `prefix_events` is how many of the source trial's events the branch-tree export prepends as the shared prefix (null for checkpoints taken before this field existed); the snapshot ref is left out. Null otherwise |
| `trial.retry` | `result.json` `retry` | `{status, reason, checkpoint, fork_id, reward, original_reward, path}` for `--retry-from-checkpoint`; null when none |
| `totals.cost` | `result.json` `branches.cost`, else summed from forks | nested forks' parent sandboxes are not counted twice; null for runs before cost accounting |
| `fork.kind` | fork `kind` | `fork` or `retry` |
| `fork.reason`, `fork.checkpoint` | fork `reason`, `checkpoint` | retries only (`failure`/`timeout`, `prompt:N`) |
| `fork.depth` | computed | 1 for the trial's own forks; 1 + the depth of the fork that made the child, for a nested fork |
| `fork.forked_by` | fork `rollout` + the child whose `node_id` equals it | `child_node`/`child_label` null for top-level forks |
| `fork.value` | fork `value` | mean child reward; null when any child is unscored |
| `fork.value_stderr` | fork `value_stderr` | standard error of `value` (sample std of the children's rewards / √n); null below two children or in trees written before this field existed |
| `fork.parent_restore` | fork `parent_restore` | `restored`, `skipped` (`--parent discard`), `deferred`, `failed`, `not_needed` (retry) |
| `fork.children_mode` | fork `children_mode` | `{isolated, concurrency, prewarm}`; null in trees written before this field existed |
| `fork.snapshot` | fork `snapshot` (`requested_layers`, `captured_layers`, `agent_session`, `retention`, `sandbox.provider`, `reused`) | `agent_session` `fresh`/`resumed`; `retention` `deleted`/`deferred`/`kept`/`delete_failed`; `reused` (1.1) true when the fork used an existing image (the kept or automatic checkpoint at the fork point) instead of taking a snapshot, so `timing_sec.checkpoint` is near 0 |
| `fork.timing_sec` | fork `timing_sec` | seconds: `checkpoint`, `child_restore[]` (per child), `children` (whole phase), `parent_restore` |
| `fork.cost` | fork `cost` | tokens, `usd` + `usd_known`, `wall_seconds`, `parent_sandbox_seconds`, `children_sandbox_seconds`, `sandbox_seconds` |
| `child.label`, `child.requested`, `child.execution` | child `intervention.label/requested/execution` | `requested` null in trees written before this field existed |
| `child.status`, `child.reward`, `child.reward_source` | child | `status` `scored`/`unscored`/`failed`/`cancelled`/`not_started`; unscored is null, never 0 |
| `child.advantage` | computed | `reward − fork.value`; null when either is null |
| `child.attempts`, `child.retried_after` | child `attempts`, `retried_after` | runs of the child (2 when a failure before the agent did anything was retried) and that first error's `{type, code}`; null in trees written before these fields existed |
| `child.cost` | child `cost` | `{tokens, usd, sandbox_seconds}`; `usd` null unless the provider reported a price (native-subscription runs never do) |
| `child.usage`, `child.timing_sec` | child `usage`, `timing_sec` | `usage`: `n_input_tokens` (not counting cache), `n_output_tokens`, `n_cache_read_tokens`, `n_cache_creation_tokens`, `total_tokens` (their sum); isolated children's `timing_sec` add `sandbox_from_snapshot`, `install_agent`, `finalize` |
| `child.usd_estimate` (1.1) | computed from `child.usage` and `pricing` | USD at list price: input, output, cache-read and cache-write tokens times their prices; 0 when the child used no tokens (oracle); null without a price for the model or a usage record. An estimate, never mixed into `cost.usd` (what the provider reported) |
| `fork.usage`, `totals.usage` (1.1) | summed from `child.usage` | `{input, output, cache_read, cache_creation, total}`; null when no child has a usage record |
| `fork.usd_estimate`, `totals.usd_estimate` (1.1) | summed `child.usd_estimate` | null when any child's estimate is null |
| `totals.per_scored_child` (1.1) | computed | `{scored, tokens, usd, usd_estimate, sandbox_seconds}`: the totals divided by the number of scored children (the cost of one training sample or one verdict); values null when `scored` is 0 or the total is null |
| `pricing` (1.1) | `load_branch_view`: LiteLLM's bundled price table for `result.json` `model` (read as a file; the provider prefix is dropped when needed); `build_branch_view(..., prices=)`: what the caller passes | `{model, source, usd_per_token: {input, output, cache_read, cache_creation}}`; null for no model or an unknown model. List prices: subscription, batch or negotiated prices differ |
| `child.snapshot_start` | child `snapshot_start` | isolated children: `{agent: reused/installed/none, verifier_baseline: inherited/recaptured, setup_commands: skipped}`; `recaptured` means tampering before the fork is not undone at verification, so show it as a warning |
| `child.archive` | child `artifacts.path` + which files exist | `observation` exists for every published child; `result` only for isolated children (full trial folders) |
| `child.nested_forks` | forks whose `rollout` equals this child's `node_id` | ids of forks this child made |

## Using the view in a viewer

A suggested backend route: `GET /api/runs/<id>/branches` returning this document for the run's trial folder. A viewer may not depend on benchflow and may read files through its own storage layer, so `src/benchflow/branch_view.py` is written to be copied: it imports nothing from benchflow, and `build_branch_view(tree, result, trial_name=..., file_exists=...)` takes the parsed `tree.json` and `result.json` (`{}` when missing) plus a callback that says whether a path relative to the trial exists. `BRANCH_VIEW_SCHEMA` in the same file can back the route's typed response. Fields a viewer typically shows:

- Fork title: `kind` (`Retry (failure) from prompt:1` for retries, which are not experiment arms), `forked_by.child_label` for nested forks ("forked from hint"), `depth`.
- Fork note lines: `children_mode` ("own sandboxes, 2 at once, 2 prepared ahead" or "in place"), `snapshot.agent_session` and `retention`, one cost line (`tokens · USD unknown · sandbox-s · wall s`, "unknown" whenever `usd_known` is false, never 0), one time line (`snapshot · children · parent restore`).
- Child table columns: Requested, Tokens, USD, Sandbox-s, Advantage, Setup (`sandbox_from_snapshot + install_agent + finalize` for isolated children), Reused (`snapshot_start`, with `recaptured` flagged), and a link to each id in `nested_forks`.
- Run header: `totals.cost`; `trial.retry` as its own line next to the reward ("Retry from prompt:1 (failure): 1.0; trial reward 0.0"), never replacing it.

The contract is versioned by `schema` (the family, `benchflow.branch-view/1`, which stays for every 1.x document) and `schema_version` (the minor version); fields are only added within `/1`, so a reader written for 1.0 reads a 1.1 document unchanged.

## Changelog

- 1.1: `kind` (`benchflow.branch-view`) and `schema_version` (`"1.1"`) at the top, the envelope `bench eval inspect`/`compare` JSON already uses. `bench eval branches --json` still prints a JSON list of these documents.
- 1.1: token classes and an estimated USD: `child.usd_estimate`, `fork.usage`, `fork.usd_estimate`, `totals.usage`, `totals.usd_estimate`, `totals.per_scored_child`, top-level `pricing`. `build_branch_view` gains an optional `prices=` argument; `load_branch_view` prices from LiteLLM's bundled table by default (`prices=None` turns estimates off).
- 1.1: `trial.checkpoint_source`; `trial.retry` also carries `tool_calls` and `no_work` for new retries.
- 1.1: `fork.snapshot.reused`.
- 1.1: `trial.parent` and `trial.unscored_by_design`; `result.json` `branches.parent` records the same for new runs.
- 1.0: first version; documents have `schema` only.
