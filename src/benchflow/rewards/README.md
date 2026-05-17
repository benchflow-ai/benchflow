# benchflow.rewards — LLM-as-Judge & Reward Functions

Composable reward functions for scoring agent outputs. The module includes `LLMJudgeRewardFunc`, which uses an LLM to evaluate agent work against a declarative rubric.

## Module layout

```
rewards/
├── __init__.py         # Public API re-exports
├── builtins.py         # TestRewardFunc, LLMJudgeRewardFunc, StringMatchRewardFunc, CodeExecRewardFunc
├── events.py           # RewardEvent dataclass (dense/terminal/process events)
├── file_readers.py     # Document text extraction (.docx, .xlsx, .pptx, .pdf, .txt, .md, .json, .csv)
├── llm.py              # Multi-provider LLM routing (Anthropic/OpenAI/Google) and verdict parsing
├── protocol.py         # RewardFunc protocol + VerifyResult
├── rubric.py           # Rubric class (multi-function reward composition)
└── rubric_config.py    # Criterion, JudgeConfig, ScoringConfig, RubricConfig, load_rubric_toml
```

## Usage

### Rubric mode (recommended)

Write a `rubric.toml`:

```toml
[judge]
model = "claude-sonnet-4-6"

[[criterion]]
name = "accuracy"
description = "Response is factually correct"
type = "binary"
weight = 2.0

[[criterion]]
name = "quality"
description = "Writing quality and clarity"
type = "likert"
points = 5

[scoring]
aggregation = "weighted_mean"
```

Score a rollout:

```python
from pathlib import Path
from benchflow.rewards import LLMJudgeRewardFunc

func = LLMJudgeRewardFunc(rubric_path=Path("rubric.toml"))
score = await func.score(Path("/app"))  # returns float in [0, 1]

# Dense reward events (one per criterion)
for event in func.events:
    print(f"{event.source}: {event.reward}")
```

### Inline criteria

```python
func = LLMJudgeRewardFunc(
    criteria=[
        {"description": "Correct answer", "type": "binary", "weight": 2.0},
        {"description": "Clear explanation", "type": "likert", "points": 5},
    ],
    judge_model="claude-sonnet-4-6",
)
```

### Auto-discovery

If no `rubric_path` or `criteria` is provided, `LLMJudgeRewardFunc` looks for `rubric.toml` in the rollout directory and its parent.

### Legacy mode

When no rubric is found, falls back to reading `llm_judge_score.txt` from the rollout directory (backward compatible with pre-rubric tasks).

## Criterion types

| Type | Raw output | Normalization |
|------|-----------|---------------|
| `binary` | `{"verdict": "pass"}` | 1.0 or 0.0 |
| `likert` | `{"score": 3}` (1–N) | `(raw - 1) / (points - 1)` |
| `numeric` | `{"score": 75}` (min–max) | `(raw - min) / (max - min)`, clamped |

## Aggregation strategies

| Strategy | Behavior |
|----------|----------|
| `weighted_mean` | `Σ(score × weight) / Σ(weight)` |
| `all_pass` | 1.0 if all scores ≥ 0.5, else 0.0 |
| `any_pass` | 1.0 if any score ≥ 0.5, else 0.0 |
| `threshold` | 1.0 if weighted mean ≥ threshold, else 0.0 |

## Provider routing

The model string determines which SDK is used:

- `claude-*` or `anthropic/*` → Anthropic (`ANTHROPIC_API_KEY`)
- `gpt-*`, `o1*`, `o3*`, `o4*`, or `openai/*` → OpenAI (`OPENAI_API_KEY`)
- `gemini*` or `google/*` → Google (`GOOGLE_API_KEY` / `GEMINI_API_KEY`)

Falls back through other providers on failure. All provider calls are async.

## File readers

`find_deliverables()` discovers files in the rollout directory. `read_file_as_text()` extracts plain text. Rich format support requires optional dependencies:

- `.docx` → pandoc (preferred) or python-docx
- `.xlsx` → openpyxl
- `.pptx` → markitdown
- `.pdf` → pdfplumber

Files > 50 MB are skipped. Content is truncated at 15,000 chars when sent to the judge.

## Output

After scoring, `evaluation_details.json` is written to the rollout directory with the aggregated score, per-criterion results, and judge reasoning.

## See also

- [docs/llm-judge.md](../../../docs/llm-judge.md) — full user-facing guide with worked examples
- [docs/concepts.md](../../../docs/concepts.md) — the Verifier primitive
- [docs/task-authoring.md](../../../docs/task-authoring.md) — verifier contract
