# Training signal: reward vectors and group advantages

`bench train convert --format prime-sft|trl-sft` writes one row per rollout (or per LLM exchange) with the rollout's scalar `reward`. Two options add more signal for RL and reward modelling:

- `--reward-vector` (`reward_vector=True`): the reward's components with names, kinds and weights.
- `--group-advantage grpo|loo` (`group_advantage="grpo"|"loo"`), with `--group-by` (`group_by=`): the reward normalised against the other rollouts of the same task.

The Python equivalents are keyword arguments of `benchflow.trajectories.export_prime_sft.export_prime_sft_jsonl` and `benchflow.trajectories.export_trl_sft.export_trl_sft_jsonl`. `benchflow.trajectories.training_signal.rollout_training_signals(jobs_dir)` computes the same vectors and groups without converting anything, which works on jobs without LLM trajectories (oracle runs, subscription runs). The row fields are described by `docs/reference/schemas/benchflow-training-signal.v1.schema.json`.

## Reward vector

| Trial | `source` | Components |
|---|---|---|
| Rubric-scored (`result.json` has a complete `scoring` block and its revision `scoring/<attempt>.json` is readable) | `rubric` | `tests` (kind `gate`, the verifier reward), then each rubric criterion in rubric order: `scored` = score / 2 with the criterion's weight; `blocker` = 1 pass, 0 fail; `legacy` (v0.1) = 1 pass, 0 fail, null `not_applicable`. `revision` and `rubric_sha256` are recorded. |
| Rubric-scored, revision file missing or without checks | `scoring` | `tests` and `rubric_reward`, with a `note` naming the missing file |
| Anything else with a reward | `verifier` | every finite numeric key of `rewards`, nested dicts flattened with `.` (`metrics.task_success`), `reward` first, weights null |
| Unscored | — | `reward_vector: null` |

A criterion the reviewer gave no score or outcome is null, never 0. `formula` states how the scalar reward was aggregated from the components.

## Group advantages

Rollouts are grouped by `--group-by` (default `task,agent,model,job`, so oracle runs, different agents, different models and different jobs never share a baseline; `task_digest` separates task versions). A group stays inside one job because two jobs may have run different policy checkpoints under one model name, which nothing in the result records; leave `job` out (`--group-by task,agent,model`) to pool jobs you know ran one policy, such as the `trial-NN` jobs of `--matrix --trials`. The `job` key is the job folder's name, or its path below the jobs' common folder when names repeat. A policy that changes during one job cannot be told apart: run one job per checkpoint. Within an Evaluation job, an attempt that a retry or a resume replaced is not a sample: it is listed with `excluded: "retried"` and counted in no baseline (the final attempt is the one `bf.load_job` keeps: scored first, then newest). Within a group, only scored rollouts count (`finite_reward(extract_reward(result))`, the rule the summaries use).

- `grpo`: `advantage = (r − mean) / (std + 1e-4)`, sample std (ddof 1, as TRL's `GRPOTrainer`). When every scored reward is equal the advantages are 0.0.
- `loo`: `advantage = r − mean(rewards of the other scored rollouts)` (RLOO baseline).

Each row gets `advantage` and `group`: `id` (`task=…|agent=…|model=…|job=…`), `by`, `key`, `normalisation`, `formula`, `rollouts` (the group's rollouts, scored or not, retried attempts left out), `scored`, `mean`, `std`, `std_ddof`, `eps`, and `excluded` when the row has no advantage (`unscored`, or `single_scored_rollout` for a group with one scored rollout). With `--reward-vector`, `advantage_vector` applies the same normalisation per component over the members that have it; `vector_names_differ: true` marks a group whose members have different components (a rubric changed between runs).

Groups are computed over every selected rollout (after `--canonical-selection`) before `--min-reward` filters rows, so filtering never moves a baseline. The `--manifest` file's `training_signal` block lists each group with its members, rewards and advantages, including rollouts that produced no row.

## Example

Four rollouts of one task: rewards 0.0 and 1.0, and two that lost their transport (`pipe_closed`, no reward).

```bash
bench train convert jobs/demo --out train.jsonl --manifest m.json --reward-vector --group-advantage grpo
```

The two scored rows get advantages −0.707 and +0.707 (mean 0.5, sample std 0.7071); with `loo`, −1.0 and +1.0. The manifest lists the two unscored rollouts with `reward: null`, `advantage: null`, `excluded: "unscored"`.

## Not covered

`--format branch-tree` already carries `advantage = reward − value` per branch child and refuses these options. Converting an existing trainer JSONL or a `results.jsonl` refuses them too (there are no rollout folders to group).
