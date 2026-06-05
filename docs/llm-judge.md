# LLM-as-Judge Verifier
Use an LLM to evaluate agent outputs against a rubric instead of deterministic tests.

---

## When to use LLM-as-judge

Use LLM-as-judge when the task output is subjective, open-ended, or hard to verify with unit tests — legal analysis, code review quality, document drafting, research summaries. For tasks with a clear right answer (e.g. "write fizzbuzz"), stick with deterministic `test.sh` verifiers.

BenchFlow's LLM judge supports:
- **First-class `[verifier]` type** — `type = "llm-judge"` in `task.toml`, no `test.sh` needed
- **Multi-criterion rubrics** with binary, likert, and numeric scoring
- **Per-criterion weights** for non-uniform importance
- **Dense reward events** emitted per criterion during evaluation
- **Multi-provider routing** across Anthropic, OpenAI, and Google models
- **Configurable aggregation** (weighted mean, all-pass, any-pass, threshold)

The judge is a **first-class verification method** alongside the deterministic
`test.sh` verifier. A task selects it with one line of config — the framework
handles deliverable collection, prompting, provider routing, retries, and
reward aggregation.

---

## Quick start

### 0. Install the judge provider SDKs

The judge calls the Anthropic, OpenAI, and Google SDKs — these are **not**
installed by default. Install the `judge` extra (you only need at least one
provider's SDK for the model you use, but the extra ships all three):

```bash
# in a checkout
uv sync --extra judge

# or as an installed tool
uv tool install --prerelease allow 'benchflow[judge]==0.5.0'

# or with pip
pip install 'benchflow[judge]'
```

If no provider SDK is installed, the judge cannot run: the verifier raises a
**verifier error** (the rollout is marked errored) rather than silently
recording a reward of `0.0` — a missing dependency is an environment failure,
not a score.

### 1. Select the judge verifier in `task.toml`

```toml
[verifier]
type = "llm-judge"
timeout_sec = 600

[verifier.judge]
model = "claude-sonnet-4-6"          # judge model (provider routed from prefix)
rubric_path = "tests/rubric.toml"    # rubric file, relative to the task dir
input_dir = "/app"                   # sandbox dir holding agent deliverables

# API keys for the judge — resolved from the host environment / .env
[verifier.env]
ANTHROPIC_API_KEY = "${ANTHROPIC_API_KEY}"
```

That's the entire verifier. There is **no `tests/test.sh`** to write — the
`Verifier` downloads the agent's deliverables from `input_dir`, scores them
against the rubric, and writes `reward.json` itself.

### 2. Write a `rubric.toml`

Place it where `rubric_path` points (by convention `tests/rubric.toml`):

```toml
[[criterion]]
name = "accuracy"
description = "The response accurately addresses the question with correct facts"
type = "binary"
weight = 3.0

[[criterion]]
name = "clarity"
description = "The response is well-organized and easy to understand"
type = "likert"
points = 5
weight = 1.0

[scoring]
aggregation = "weighted_mean"
```

A Harvey LAB style `rubric.json` works too — set `rubric_path = "tests/rubric.json"`:

```json
{
  "title": "Task Title",
  "criteria": [
    {"id": "criterion-1", "title": "...", "match_criteria": "What constitutes a pass"}
  ]
}
```

That's it. Run the task as usual — the reward is the proportion of criteria
passed (or the configured aggregation), a partial float in `[0, 1]`.

---

## `[verifier]` reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | string | `"test-script"` | `"test-script"` (run `tests/test.sh`) or `"llm-judge"` |
| `timeout_sec` | float | `600` | Overall verifier timeout |
| `env` | table | `{}` | Env vars for the verifier — judge API keys go here |

### `[verifier.judge]` (used when `type = "llm-judge"`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `"claude-sonnet-4-6"` | Judge model; provider routed from prefix |
| `rubric_path` | string | `"tests/rubric.toml"` | Rubric file relative to the task dir (`.toml` or `.json`) |
| `input_dir` | string | `"/app"` | Sandbox dir whose contents are graded |
| `input_type` | string | `"deliverables"` | Only `"deliverables"` is supported — trajectory judging is not available at verify time |
| `context` | string | `""` | Extra judge context (defaults to the task instruction) |

---

## Library use — `LLMJudgeRewardFunc`

The judge is also a composable `RewardFunc`, usable directly or from a custom
`test.sh` verifier:

```python
import asyncio
from pathlib import Path
from benchflow.rewards import LLMJudgeRewardFunc

func = LLMJudgeRewardFunc(rubric_path=Path("rubric.toml"))
score = asyncio.run(func.score(Path("/app")))
print(f"Score: {score:.2f}")
```

Auto-discovery — if `rubric.toml`/`rubric.json` is in the rollout directory or
its parent, it's found automatically:

```python
func = LLMJudgeRewardFunc()
score = asyncio.run(func.score(Path("/app")))
```

---

## rubric.toml reference

### `[judge]` section

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | `"claude-sonnet-4-6"` | LLM model for judging. Prefix with `anthropic/`, `openai/`, or `google/` to force a provider |
| `mode` | string | `"individual"` | `"individual"` scores each criterion separately; `"batched"` is reserved for future use |
| `files` | string[] | `[]` | Default files to evaluate (fallback when a criterion doesn't specify its own) |
| `timeout` | int | `120` | Timeout in seconds per judge call |
| `reference` | string | `null` | Optional reference answer for comparison |
| `prompt_template` | string | `null` | Custom prompt template (overrides built-in templates) |

### `[[criterion]]` entries

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | — | Criterion identifier (falls back to first 40 chars of description) |
| `description` | string | **required** | What the judge should evaluate |
| `type` | string | `"binary"` | `"binary"`, `"likert"`, or `"numeric"` |
| `weight` | float | `1.0` | Relative importance in aggregation |
| `points` | int | `5` | Scale for likert type (1 to N) |
| `min` | float | `0.0` | Minimum for numeric type |
| `max` | float | `100.0` | Maximum for numeric type |
| `files` | string[] | `[]` | Specific files this criterion should evaluate |

### `[scoring]` section

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `aggregation` | string | `"weighted_mean"` | How to combine criterion scores |
| `threshold` | float | `0.7` | Pass threshold (only used with `"threshold"` aggregation) |

### Score normalization

Each criterion type normalizes its raw score to `[0, 1]`:

| Type | Raw | Normalized |
|------|-----|------------|
| `binary` | pass/fail | `1.0` or `0.0` |
| `likert` | 1–N integer | `(raw - 1) / (points - 1)` |
| `numeric` | min–max float | `(raw - min) / (max - min)`, clamped to `[0, 1]` |

### Aggregation strategies

| Strategy | Behavior |
|----------|----------|
| `weighted_mean` | `sum(score × weight) / sum(weight)` — continuous reward |
| `all_pass` | `1.0` if every criterion scores ≥ 0.5, else `0.0` |
| `any_pass` | `1.0` if any criterion scores ≥ 0.5, else `0.0` |
| `threshold` | `1.0` if weighted mean ≥ threshold, else `0.0` |

---

## Criterion types

### Binary (pass/fail)

The judge decides whether the criterion is satisfied. The LLM returns `{"verdict": "pass", "reasoning": "..."}`.

```toml
[[criterion]]
name = "has-executive-summary"
description = "The document includes an executive summary in the first section"
type = "binary"
```

### Likert (scaled)

The judge rates on a 1-to-N scale. The LLM returns `{"score": 4, "reasoning": "..."}`.

```toml
[[criterion]]
name = "writing-quality"
description = "Overall quality of prose — grammar, flow, and precision"
type = "likert"
points = 5
```

A score of 3 on a 5-point scale normalizes to `(3-1)/(5-1) = 0.5`.

### Numeric (range)

The judge assigns a value within a continuous range. The LLM returns `{"score": 75.0, "reasoning": "..."}`.

```toml
[[criterion]]
name = "coverage-pct"
description = "Percentage of key topics from the source material covered in the summary"
type = "numeric"
min = 0.0
max = 100.0
```

---

## Inline criteria (no TOML file)

For programmatic use or Harvey LAB-style criteria, pass criteria directly:

```python
func = LLMJudgeRewardFunc(
    criteria=[
        {
            "description": "The response is factually accurate",
            "type": "binary",
            "weight": 2.0,
        },
        {
            "description": "The response addresses all parts of the question",
            "type": "binary",
            "weight": 1.0,
        },
    ],
    judge_model="claude-sonnet-4-6",
)
```

Harvey LAB `match_criteria` keys are also supported:

```python
func = LLMJudgeRewardFunc(
    criteria=[
        {"match_criteria": "Identifies the key risk factors", "type": "binary"},
        {"match_criteria": "Provides supporting evidence", "type": "binary"},
    ],
)
```

---

## Dense reward events

Each criterion emits a `RewardEvent` during evaluation, enabling per-criterion observability and training signal:

```python
func = LLMJudgeRewardFunc(rubric_path=Path("rubric.toml"))
score = await func.score(rollout_dir)

for event in func.events:
    print(f"  {event.source}: {event.reward:.2f} (step {event.step})")
```

Output:
```
  criterion:accuracy: 1.00 (step 0)
  criterion:clarity: 0.50 (step 1)
  criterion:completeness: 0.75 (step 2)
```

Events have type `"dense"`, a `reward` in `[0, 1]`, a `source` of `"criterion:{name}"`, and a `step` index. Events are cleared between `score()` calls.

---

## Multi-provider routing

The judge model string determines which provider SDK is used:

| Prefix | Provider | Auth env var |
|--------|----------|--------------|
| `claude-*`, `anthropic/*` | Anthropic | `ANTHROPIC_API_KEY` |
| `gpt-*`, `o1*`, `o3*`, `o4*`, `openai/*` | OpenAI | `OPENAI_API_KEY` |
| `gemini*`, `google/*` | Google | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |

If the primary provider fails, the judge falls back through the other providers with retries and exponential backoff.

The provider SDKs ship in the `judge` extra (`uv sync --extra judge`). If
*none* are installed, the judge raises a verifier error instead of recording
a reward — see [step 0](#0-install-the-judge-provider-sdks).

---

## Evaluation output

After scoring, an `evaluation_details.json` is written to the rollout directory:

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
    },
    {
      "id": "clarity",
      "description": "The response is well-organized",
      "score": 0.5,
      "weight": 1.0,
      "verdict": {"score": 3, "reasoning": "..."}
    }
  ]
}
```

The `score` field is the actual aggregated score from the configured strategy, not `n_passed / n_total`.

---

## File discovery

The judge automatically discovers deliverable files in the rollout directory. Supported formats:

| Extension | Reader | Dependency |
|-----------|--------|------------|
| `.txt`, `.md`, `.json`, `.csv` | Built-in | None |
| `.docx` | pandoc or python-docx | `pandoc` (preferred) or `pip install python-docx` |
| `.xlsx` | openpyxl | `pip install openpyxl` |
| `.pptx` | markitdown | `pip install markitdown` |
| `.pdf` | pdfplumber | `pip install pdfplumber` |

Files larger than 50 MB are skipped. Hidden files (starting with `.`) and internal metadata files (`rubric.json`) are excluded. File content is truncated at 15,000 characters per file when sent to the judge.

To scope a criterion to specific files:

```toml
[[criterion]]
name = "memo-quality"
description = "The legal memo follows IRAC structure"
files = ["memo.docx", "analysis.md"]
```

---

## Python API

All rubric config types are importable from the top level:

```python
from benchflow import (
    Criterion,
    JudgeConfig,
    LLMJudgeRewardFunc,
    RubricConfig,
    ScoringConfig,
    load_rubric,        # dispatches on extension (.toml / .json)
    load_rubric_json,
    load_rubric_toml,
)

# Load and inspect a rubric (TOML or Harvey LAB style JSON)
rubric = load_rubric(Path("rubric.json"))
print(f"Model: {rubric.judge.model}")
print(f"Criteria: {len(rubric.criteria)}")
for c in rubric.criteria:
    print(f"  {c.id} ({c.type}, weight={c.weight})")
```

---

## Worked example — Harvey LAB legal task

A legal document analysis task scored entirely by config — no `test.sh`:

```toml
# task.toml
[verifier]
type = "llm-judge"
timeout_sec = 600

[verifier.judge]
model = "claude-sonnet-4-6"
rubric_path = "tests/rubric.toml"
input_dir = "/app"

[verifier.env]
ANTHROPIC_API_KEY = "${ANTHROPIC_API_KEY}"
```

```toml
# tests/rubric.toml
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
weight = 1.0

[scoring]
aggregation = "weighted_mean"
```

The framework downloads the agent's deliverables from `/app`, grades each
criterion, aggregates, and writes `reward.json` — no scripting required.

---

## Where to go next

- [Concepts](./concepts.md) — the five primitives including Verifier
- [Task authoring](./task-authoring.md) — `task.toml`, `tests/`, verifier contract
- [Running benchmarks](./running-benchmarks.md) — Harvey LAB uses LLM-as-judge
- [Python API reference](./reference/python-api.md) — `LLMJudgeRewardFunc` and friends
