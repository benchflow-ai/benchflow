# task.md examples

`task.md` packages: some are runnable tasks, some are fixtures that hold only a `task.md`.

```bash
bench check docs/examples/task-md/real-skillsbench/weighted-gdp-calc --level publication-grade
```

## Fixtures

These folders hold only a `task.md`. Check them at `--level schema`:

```bash
bench check docs/examples/task-md/clean-body-roles-scenes --level schema
bench check docs/examples/task-md/task-toml-parity --level schema
bench check docs/examples/task-md/multi-scene --level schema
bench check docs/examples/task-md/nudgebench-team --level schema
```

They are not runnable tasks. The default check, and `bench run`, need `environment/` and `verifier/` (or `tests/` in a split task).

## Runnable tasks

Check these with the default level:

```bash
bench check docs/examples/task-md/generated-skill-eval/models-as-skills/regex-email-parser
```

The SkillsBench task and the generated skill-eval task pass the publication-grade level:

```bash
bench check docs/examples/task-md/real-skillsbench/weighted-gdp-calc --level publication-grade
bench check docs/examples/task-md/generated-skill-eval/models-as-skills/regex-email-parser --level publication-grade
```

`user-runtime/private-facts-nudges` is a runnable task with a simulated user. It uses scenes, `user.private_facts`, `benchflow.nudges` and a `## user-persona` section:

```bash
bench check docs/examples/task-md/user-runtime/private-facts-nudges --level publication-grade
bench check docs/examples/task-md/user-runtime/private-facts-nudges --level runtime-capability --sandbox docker
```

## Run a verifier outside the sandbox

The generated skill-eval verifiers read their paths from the environment. To smoke-test one locally:

```bash
BENCHFLOW_VERIFIER_DIR=/path/to/verifier \
BENCHFLOW_WORKSPACE=/path/to/workspace \
BENCHFLOW_AGENT_LOG_DIR=/path/to/logs/agent \
BENCHFLOW_REWARD_TEXT=/path/to/logs/verifier/reward.txt \
BENCHFLOW_REWARD_JSON=/path/to/logs/verifier/reward.json \
bash /path/to/verifier/test.sh
```

## Next steps

- Write your own: [task.md authoring](../../task-authoring-task-md.md).
- What each check level checks: [CLI reference: bench check](../../reference/cli.md).
