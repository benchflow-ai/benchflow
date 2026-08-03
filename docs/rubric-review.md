# Rubric review

Rubric review is a post-verifier, agentic assessment of a solver's plan and
recorded method. The deterministic verifier continues to own `reward`; the
reviewer adds a separate `plan` channel for criteria that are difficult to
express as output tests.

This is distinct from the [`llm-judge` verifier strategy](./llm-judge.md): an
llm-judge grades deliverable text with a chat call, while rubric review runs a
registered agent harness over sanitized workspace, trajectory, and artifact
snapshots.

## Task contract

A task adds exactly one file:

```text
tasks/<task-id>/
  task.md
  environment/
  oracle/
  verifier/
    test.sh
    test_outputs.py
    rubric.json
```

`schema_version` distinguishes a review rubric from an llm-judge
`rubric.json`. Unknown keys are rejected everywhere.

```json
{
  "schema_version": "1.2",
  "reviewer": {
    "harness": "gemini",
    "model": "gemini-2.5-flash",
    "timeout_sec": 600,
    "mode": "per_criterion"
  },
  "pass_threshold": 0.7,
  "criteria": [
    {
      "id": "method-derived",
      "criterion": "The plan derives the requested result from the supplied data rather than a target answer.",
      "criterion_type": "data-handling",
      "weight": 2
    },
    {
      "id": "self-check",
      "criterion": "The plan includes a final validation against the requested output contract.",
      "criterion_type": "failure-check",
      "gating": true
    }
  ]
}
```

Each criterion has three required fields: `id`, `criterion`, and
`criterion_type`. Types are a closed set:

- `physical-model`
- `approximation`
- `numerical-method`
- `uncertainty`
- `data-handling`
- `failure-check`

`weight` defaults to `1`. Positive weights earn credit, negative weights are
penalties, and zero records a metric. `gating: true` makes a criterion
must-pass and forbids `weight`.

The task checker also rejects duplicate ids, non-finite weights, rubrics with
no positive non-gating weight, and numeric answer literals copied from
`test_outputs.py`. Numeric tolerances expressed as tolerances remain valid.

## Running

```bash
bench eval run --tasks-dir tasks \
  --agent codex-acp --model <model-under-test> \
  --review --reviewer-harness gemini \
  --reviewer-model gemini-2.5-flash
```

Reviewer precedence is CLI override, then `rubric.json`, then the harness's
registry default for the model. A reviewer harness has no implicit default: it
must be provided by the rubric or CLI. `--reviewer-timeout-sec` and
`--reviewer-mode per_criterion|batched` override the other reviewer fields.
`--reviewer-reasoning-effort` sets reviewer provider effort independently from
the solver, so heterogeneous model pairs do not leak one model's effort into
the other.

- Omitted `--review`: run only when the task ships a review rubric.
- `--review`: require a valid review rubric.
- `--no-review`: skip review.

## Isolation

The reviewer runs after verification in a separate sandbox. It never
reconnects to the solver container. The runtime provides fixed evidence paths:

| Path | Access |
|---|---|
| `/review/workspace/root` | sanitized read-only solver workspace copy |
| `/review/trajectory` | key-redacted, delimiter-neutralized trajectory copies |
| `/review/artifacts` | sanitized read-only artifact copy |
| `/review/rubric.json` | read-only rubric |
| `/review/control` | read-only injection-control evidence |

The reviewer runs as the non-root `reviewer` user with `network_mode:
no-network`. Original oracle, verifier, hidden tests, solver credentials, and
reward outputs are not present. A hidden integrity criterion detects prompt
injection; failure marks the review `compromised` and discards its scores.

## Verdicts and scoring

The reviewer returns a binary result per criterion:

```json
{
  "verdicts": [
    {
      "id": "method-derived",
      "explanation": "The trajectory shows the parser deriving each value.",
      "evidence": ["/review/trajectory/acp_trajectory.jsonl:27"],
      "criterion_met": true
    }
  ]
}
```

Evidence must correspond to a read/search tool event. Malformed booleans,
missing ids, empty or invented evidence, and other schema failures receive up
to two corrective retries. A remaining non-metric failure makes the plan
unscored rather than scoring the solver zero.

Scoring uses a positive-only denominator:

```text
any unmet gate => plan = 0
otherwise plan = clamp(sum(weight * criterion_met) / sum(positive weights), 0, 1)
```

## Outputs

On a scored review, reward artifacts add a second channel:

```json
{
  "reward": 1.0,
  "plan": 0.5,
  "plan_passed": 0.0,
  "plan/method-derived": 1.0,
  "plan/self-check": 0.0
}
```

`result.json` always records `rubric_review.status`, failure reason, reviewer
harness and model, rubric SHA-256, isolation metadata, and the details path.
Full verdicts are written to `review/review-details.json`; reviewer ACP and
provider trajectories remain separate from the solver under `review/`.
