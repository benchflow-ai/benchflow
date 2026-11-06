# Rubric reference

Every field of an [LLM judge](./llm-judge.md) rubric, how scores combine, which provider runs the judge, and the Python API.

```toml
[judge]
files = ["analysis.md"]

[[criterion]]
name = "key-terms-identified"
description = "All material terms from the contract are identified and listed"
type = "binary"
weight = 2.0

[[criterion]]
name = "risk-assessment"
description = "Each identified risk includes severity rating and mitigation suggestion"
type = "likert"
points = 5
weight = 3.0

[[criterion]]
name = "completeness"
description = "Percentage of contract sections addressed in the analysis"
type = "numeric"
min = 0
max = 100

[scoring]
aggregation = "weighted_mean"
```

## `[judge]`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `model` | string | `"claude-sonnet-4-6"` | the judge model. In a task run, the strategy's `model` (or `[verifier.judge] model`) wins, so this field only applies in [library use](#python-api). |
| `mode` | string | `"individual"` | `"individual"` or `"batched"`. Both score each criterion in its own call. |
| `files` | list of strings | `[]` | files to grade when a criterion names none |
| `timeout` | integer | `120` | parsed, not applied yet: judge calls use the provider SDK's own timeout |

## `[[criterion]]`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `description` | string | required | what the judge grades; this text is sent to the judge |
| `name` | string | the first 40 characters of `description` | the criterion's id in outputs |
| `type` | string | `"binary"` | `"binary"`, `"likert"` or `"numeric"` |
| `weight` | float | `1.0` | importance in aggregation |
| `points` | integer | `5` | scale size for `likert` (1 to N) |
| `min` | float | `0.0` | lowest value for `numeric` |
| `max` | float | `100.0` | highest value for `numeric` |
| `files` | list of strings | `[]` | files this criterion grades; falls back to `[judge] files` |

## `[scoring]`

| Field | Default | Meaning |
|---|---|---|
| `aggregation` | `"weighted_mean"` | how criterion scores combine; `method` is accepted as another name, and `"mean"` means `"weighted_mean"` |
| `threshold` | `0.7` | the pass bar for `"threshold"` |

## Score normalization

| Type | The judge returns | Normalized score |
|---|---|---|
| `binary` | `{"verdict": "pass" or "fail", "reasoning": "..."}` | 1 for pass, 0 for fail |
| `likert` | `{"score": 4, "reasoning": "..."}`, from 1 to `points` | `(raw - 1) / (points - 1)` |
| `numeric` | `{"score": 75.0, "reasoning": "..."}`, from `min` to `max` | `(raw - min) / (max - min)`, clamped to 0..1 |

A Likert 3 on a 5-point scale is `(3 - 1) / (5 - 1) = 0.5`.

## Aggregation

| `aggregation` | Reward |
|---|---|
| `weighted_mean` | `sum(score × weight) / sum(weight)` |
| `all_pass` | 1 if every criterion scores at least 0.5, else 0 |
| `any_pass` | 1 if any criterion scores at least 0.5, else 0 |
| `threshold` | 1 if the weighted mean is at least `threshold`, else 0 |

A criterion passes at 0.5 or more. So for `all_pass` and `any_pass`, a Likert 3 of 5 passes. Use `threshold` for a stricter bar.

## Providers

The judge model's prefix picks the provider:

| Prefix | Provider | Key |
|---|---|---|
| `claude-`, `anthropic/` | Anthropic | `ANTHROPIC_API_KEY` |
| `gpt-`, `o1`, `o3`, `o4`, `openai/` | OpenAI | `OPENAI_API_KEY` (and `OPENAI_BASE_URL` for another endpoint) |
| `gemini`, `google/`, `gemini/` | Google | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |

- A model with one of these prefixes goes only to that provider. A failure there is reported, not passed to another provider.
- A model with any other name is tried on Anthropic, then OpenAI, then Google.
- Each call is tried 3 times, waiting 1 s and then 2 s between tries.

The provider SDKs come with the `judge` extra (`'benchflow[judge]'`). Without the one a model needs, the verifier fails with an error that names it.

## When a judge call fails

| Where | A failed call |
|---|---|
| in a task run | is a verifier error. The trial has no reward; it is not scored 0. |
| in [library use](#python-api) | scores that criterion 0, with `"Judge error: ..."` as its reasoning |

## Files the judge reads

| Extension | Reader | Needs |
|---|---|---|
| `.txt`, `.md`, `.json`, `.csv` | built in | nothing |
| `.docx` | pandoc, or python-docx | `pandoc`, or `pip install python-docx` |
| `.xlsx` | openpyxl | `pip install openpyxl` |
| `.pptx` | markitdown | `pip install markitdown` |
| `.pdf` | pdfplumber | `pip install pdfplumber` |

- Files over 50 MB are skipped.
- Hidden files (names starting with `.`) and `rubric.json` are skipped.
- Each file is cut at 15,000 characters in the judge's prompt.

## Outputs

In a task run, the verifier writes:

| File | Holds |
|---|---|
| `verifier/reward.json` | `{"reward": <score>}` |
| `verifier/reward-details.json` | one entry per criterion (`source: "criterion:<name>"`, `reward`, `step`) and the aggregate |

In library use, `score()` also writes `evaluation_details.json` into the folder it scored:

```json
{
  "score": 0.75,
  "n_passed": 2,
  "n_total": 3,
  "results": [
    {
      "id": "accuracy",
      "description": "The response accurately addresses the question",
      "score": 1.0,
      "weight": 3.0,
      "verdict": {"verdict": "pass", "reasoning": "..."}
    }
  ]
}
```

`score` is the aggregate from `[scoring]`, not `n_passed / n_total`.

## Python API

The judge is also a `RewardFunc` you can call yourself:

```python
import asyncio
from pathlib import Path

from benchflow import LLMJudgeRewardFunc

func = LLMJudgeRewardFunc(rubric_path=Path("rubric.toml"))
score = asyncio.run(func.score(Path("/app")))
print(f"Score: {score:.2f}")
```

Without `rubric_path`, it looks for `rubric.toml` or `rubric.json` in the folder it scores, then in that folder's parent.

### Inline criteria

```python
func = LLMJudgeRewardFunc(
    criteria=[
        {"description": "The response is factually accurate", "type": "binary", "weight": 2.0},
        {"description": "The response addresses all parts of the question", "type": "binary"},
    ],
    judge_model="claude-sonnet-4-6",
)
```

Harvey LAB `match_criteria` keys work too:

```python
func = LLMJudgeRewardFunc(
    criteria=[
        {"match_criteria": "Identifies the key risk factors", "type": "binary"},
        {"match_criteria": "Provides supporting evidence", "type": "binary"},
    ],
)
```

### Per-criterion events

Each criterion emits a `RewardEvent` during `score()`, for observability and training signal:

```python
score = asyncio.run(func.score(Path("/app")))
for event in func.events:
    print(f"{event.source}: {event.reward:.2f} (step {event.step})")
```

```text
criterion:accuracy: 1.00 (step 0)
criterion:clarity: 0.50 (step 1)
```

Events have type `"dense"`, a `reward` from 0 to 1, a `source` of `"criterion:<name>"` and a `step` index. They are cleared on each `score()` call.

### Load a rubric

```python
from pathlib import Path

from benchflow import load_rubric

rubric = load_rubric(Path("rubric.json"))   # .toml or .json
print(rubric.judge.model, len(rubric.criteria))
for c in rubric.criteria:
    print(c.id, c.type, c.weight)
```

`Criterion`, `JudgeConfig`, `RubricConfig`, `ScoringConfig`, `load_rubric_toml` and `load_rubric_json` are importable from `benchflow` too.

## Next steps

- [LLM judge](./llm-judge.md): the quick start.
- [Rubric review](./rubric-review.md): grade a whole run with a reviewer agent.
- [Training signal](../reference/training-signal.md): use per-criterion events as rewards.
