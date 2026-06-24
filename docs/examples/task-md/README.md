# task.md Examples

This directory contains both runnable task packages and schema-only authoring
fixtures.

Use `--level schema` for fixtures that intentionally contain only `task.md`:

```bash
uv run --extra dev bench tasks check docs/examples/task-md/clean-body-roles-scenes --level schema
uv run --extra dev bench tasks check docs/examples/task-md/harbor-parity --level schema
uv run --extra dev bench tasks check docs/examples/task-md/multi-scene --level schema
uv run --extra dev bench tasks check docs/examples/task-md/nudgebench-team --level schema
```

Those fixtures are not runnable eval tasks. Default structural validation, and
`bench eval run`, require runtime assets such as `environment/` and
`verifier/` or legacy `tests/`.

Use default structural validation for runnable examples:

```bash
uv run --extra dev bench tasks check docs/examples/task-md/generated-skill-eval/models-as-skills/regex-email-parser
```

Real SkillsBench and generated skill-eval examples are native
publication-grade packages:

```bash
uv run --extra dev bench tasks check docs/examples/task-md/real-skillsbench/weighted-gdp-calc --level publication-grade
uv run --extra dev bench tasks check docs/examples/task-md/generated-skill-eval/models-as-skills/regex-email-parser --level publication-grade
```

The `user-runtime/private-facts-nudges` example is a runnable native package
for simulated-user semantics, including scenes, `user.private_facts`,
`benchflow.nudges`, and `## user-persona`:

```bash
uv run --extra dev bench tasks check docs/examples/task-md/user-runtime/private-facts-nudges --level publication-grade
uv run --extra dev bench tasks check docs/examples/task-md/user-runtime/private-facts-nudges --level runtime-capability --sandbox docker
```

Generated skill-eval verifiers also accept env-overridden paths for local
smoke tests outside the sandbox mount layout:

```bash
BENCHFLOW_VERIFIER_DIR=/path/to/verifier \
BENCHFLOW_WORKSPACE=/path/to/workspace \
BENCHFLOW_AGENT_LOG_DIR=/path/to/logs/agent \
BENCHFLOW_REWARD_TEXT=/path/to/logs/verifier/reward.txt \
BENCHFLOW_REWARD_JSON=/path/to/logs/verifier/reward.json \
bash /path/to/verifier/test.sh
```
