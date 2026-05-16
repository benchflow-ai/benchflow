---
name: testing-benchflow-api
description: Test the BenchFlow Python API surface, backward-compat aliases, reward composition, adapter features, and run e2e integration tests. Use when verifying refactors, new modules, or API changes.
---

# Testing BenchFlow API & Integration

This skill validates that BenchFlow's public API, backward-compat layer, adapters, and e2e pipeline work correctly.

## Prerequisites

### Devin Secrets Needed
- `GEMINI_API_KEY` — for integration tests with gemini agent
- `DAYTONA_API_KEY` — for Daytona sandbox backend
- `AWS_BEARER_TOKEN_BEDROCK` — for Bedrock provider tests (optional)

### Environment Setup
```bash
uv sync --extra dev --locked
```

### .env File (Critical)
`resolve_agent_env()` reads credentials from `.env` via `load_dotenv_env()`, **not** from `os.environ`. You must create a `.env` file:
```bash
echo "GEMINI_API_KEY=${GEMINI_API_KEY}" > .env
echo "DAYTONA_API_KEY=${DAYTONA_API_KEY}" >> .env
echo "AWS_BEARER_TOKEN_BEDROCK=${AWS_BEARER_TOKEN_BEDROCK}" >> .env
```
The `.env` file is already in `.gitignore`.

## Test Categories

### 1. Public API Import Verification
Verify all public types import correctly from `benchflow`:
```bash
uv run python -c "
from benchflow import (
    Rollout, RolloutConfig, RolloutResult,
    Evaluation, EvaluationConfig, EvaluationResult,
    Role, Scene, Turn,
    Sandbox, ImageBuilder, ImageConfig, ImageRef,
    Rubric, RewardFunc, RewardEvent, VerifyResult,
    InspectAdapter, ORSAdapter, to_inspect_task, to_ors_reward,
)
print('All imports OK')
"
```

### 2. Backward-Compat Alias Identity
Aliases must be `is`-equal (not copies) so `isinstance` checks work:
```bash
uv run python -c "
from benchflow import Trial, Rollout, TrialConfig, RolloutConfig, RunResult, RolloutResult, Job, Evaluation
assert Trial is Rollout
assert TrialConfig is RolloutConfig
assert RunResult is RolloutResult
assert Job is Evaluation
print('All aliases identity-equal')
"
```

### 3. Rewards Composition
The `Rubric` takes `reward_funcs` (list of `RewardFunc`), not `items`. `RewardEvent` fields are `type`, `reward`, `source`, `step`, `ts`.
```bash
uv run python -c "
from benchflow.rewards import Rubric, RewardEvent, VerifyResult
from pathlib import Path
import asyncio

class AlwaysPassFunc:
    async def score(self, rollout_dir: Path) -> float:
        return 1.0

rubric = Rubric(reward_funcs=[AlwaysPassFunc()], weights=[1.0])
result = asyncio.run(rubric.score(Path('/tmp')))
assert result.reward == 1.0
print(f'Rubric score: {result.reward}, events: {len(result.events)}')
"
```

### 4. InspectAdapter Testing
Exercise `to_inspect_task()` with realistic scenes:
```bash
uv run python -c "
from benchflow._types import Scene, Role, Turn
from benchflow.rewards import Rubric
from benchflow.adapters.inspect_ai import to_inspect_task
from pathlib import Path

# Multi-role scene
scene = Scene(
    name='code-review',
    roles=[Role(name='coder', agent='gemini'), Role(name='reviewer', agent='claude')],
    turns=[Turn(role='coder', prompt='Write fibonacci'), Turn(role='reviewer', prompt='Review it')],
)
result = to_inspect_task(scene)
assert result['name'] == 'code-review'
assert len(result['dataset']) == 2
assert 'scorer' not in result  # no rubric

# With rubric
class F:
    async def score(self, d: Path) -> float: return 0.8
rubric = Rubric(reward_funcs=[F()], weights=[1.0])
result2 = to_inspect_task(scene, rubric=rubric)
assert result2['scorer']['type'] == 'benchflow_rubric'
assert result2['scorer']['reward_funcs'] == 1
print('InspectAdapter tests passed')
"
```

Key behaviors to verify:
- `name` matches `scene.name`
- `dataset` has one entry per turn with `input` (prompt or `""` for None) and `role`
- `scorer` only present when rubric is provided
- Edge: empty scene → `dataset=[]`; None prompt → `input=""`

### 5. ORSAdapter Testing
Exercise `to_ors_reward()` with both success and error cases:
```bash
uv run python -c "
from benchflow.rewards import RewardEvent, VerifyResult
from benchflow.adapters.ors import to_ors_reward, ORSAdapter

# Success case
events = [RewardEvent(type='terminal', reward=0.8, source='TestFunc')]
vr = VerifyResult(reward=0.8, items={'TestFunc': 0.8}, events=events, error=None)
ors = to_ors_reward(vr)
assert ors['is_valid'] is True
assert ors['reward'] == 0.8
assert len(ors['metadata']['events']) == 1
assert ors['metadata']['events'][0]['source'] == 'TestFunc'

# Error case
err_vr = VerifyResult(reward=0.0, items={}, events=[], error='Timeout')
err_ors = to_ors_reward(err_vr)
assert err_ors['is_valid'] is False
assert err_ors['metadata']['error'] == 'Timeout'
print('ORSAdapter tests passed')
"
```

Key behaviors to verify:
- `is_valid` is `True` when `error is None`, `False` otherwise
- Events map: `type`→`type`, `reward`→`reward`, `source`→`source`, `step`→`step`, `ts`→`timestamp`
- `ORSAdapter.reward_event_to_ors()` works for individual events

### 6. Full Test Suite
```bash
uv run python -m pytest tests/ -q --tb=short
```
Expect 960+ tests passing, 0 failures.

### 7. Lint + Format
```bash
uv run ruff check . && uv run ruff format --check .
```

### 8. E2E Integration Test
The canonical integration test path is `bench eval create` (full Job/Evaluation pipeline), **not** `bench run`:
```bash
uv run bench eval create \
  --source-repo benchflow-ai/skillsbench \
  --source-path tasks/jax-computing-basics \
  -a gemini \
  -m gemini-3.1-flash-lite-preview \
  -e daytona \
  -c 1 \
  -o /tmp/test-integration
```
Expect: exit code 0, output directory with trial results and `reward.txt`.

## Architecture Notes

### Module Layout (v0.4)
- `rollout.py` — Single execution path (`Rollout` class, was `Trial`)
- `trial.py` — Backward-compat shim, re-exports from `rollout.py`
- `_types.py` — Canonical types (`Role`, `Scene`, `Turn`)
- `sandbox/` — `Sandbox` protocol + Docker/Daytona adapters
- `rewards/` — `Rubric`, `RewardFunc`, `RewardEvent`, `VerifyResult`
- `adapters/` — External format converters (Inspect AI, ORS)
- `evaluation.py` — Batch orchestration (was `job.py`)
- `_provider_runtime.py` — Bedrock proxy runtime

### Test Patching
When patching in tests, target `benchflow.rollout.<name>` (not `benchflow.trial.<name>`) since `trial.py` is now a shim. The actual implementation lives in `rollout.py`.

### Common Gotchas
- `.env` file is required for agent env resolution — `os.environ` alone is insufficient
- `Rubric` constructor takes `reward_funcs` (list), not `items` (dict)
- `RewardEvent` fields: `type`, `reward`, `source`, `step`, `ts` — no `name`, `reason`, `weight`
- `VerifyResult` fields: `reward`, `items`, `events`, `error` — no `reward_text`
- `InspectAdapter` constructor takes `scene` and optional `rubric` — not a static class
- `ORSAdapter` is all static methods — no constructor needed
- `Sandbox` protocol `isinstance` check may return `False` for incomplete implementations (missing methods)
- `Scene.single()` is a convenience classmethod for single-agent scenes
