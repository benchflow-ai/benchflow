# Self-Generated Skill Eval Mode Design

Date: 2026-05-05

## Purpose

BenchFlow should support a runtime experiment mode for skill evaluation where the agent generates a task-specific skill pack before solving. This creates a directly comparable matrix:

- `baseline`: solve with no skills.
- `with-skill`: solve with the provided static skill.
- `self-generated`: generate a skill pack at runtime, then solve with that generated pack.

The feature must run through BenchFlow commands and runtime surfaces. Users should invoke it with `bench` or `benchflow`; no direct Harbor command should appear in docs, examples, or implementation paths for this mode.

## Source Policy

The default generator policy should follow Anthropic's `skill-creator` defaults from `anthropics/skills`. The relevant standard is that a skill is a directory containing required `SKILL.md` frontmatter and optional bundled resources such as `scripts/`, `references/`, and `assets/`.

BenchFlow should embed or vendor a stable local prompt/template derived from that workflow rather than depending on live GitHub content at runtime. The initial version should not expose multiple creator policies; Anthropic-style skill creation is the default.

References:

- https://github.com/anthropics/skills/tree/main/skills/skill-creator
- https://raw.githubusercontent.com/anthropics/skills/main/skills/skill-creator/SKILL.md

## User Interface

Existing behavior stays compatible:

```bash
bench skills eval ./my-skill/ -a claude-agent-acp
```

continues to run the current `with-skill` plus `baseline` comparison.

New explicit mode selection:

```bash
bench skills eval ./my-skill/ \
  --mode baseline \
  --mode with-skill \
  --mode self-generated
```

Creator configuration:

```bash
bench skills eval ./my-skill/ \
  --mode self-generated \
  --creator-agent claude-agent-acp \
  --creator-model claude-sonnet-4-5
```

Defaults:

- Solver agent and model remain controlled by existing `--agent` and `--model`.
- Creator agent and model default to the solver agent and model unless overridden.
- `--no-baseline` remains a compatibility alias that removes `baseline` from the default mode set.
- `--export-gepa` includes selected mode names in trace filenames and metadata.

## Architecture

Add `self-generated` as a first-class `bench skills eval` mode.

The core runner should use named modes instead of treating the comparison as only `with_skill: bool`.

- `baseline`: materialize a normal eval task with no mounted skill.
- `with-skill`: copy the provided skill directory into the task environment and mount it through existing BenchFlow skill deployment.
- `self-generated`: run a creator phase that writes a complete skill pack, validate the generated pack, then run a fresh solver phase with the generated pack mounted.

The self-generated path should use BenchFlow `Trial` and scene execution surfaces. It should not bypass BenchFlow by shelling out to Harbor or using Harbor-specific APIs directly.

## Components

The implementation should keep changes localized:

- `SkillEvalMode`: enum or literal type for `baseline`, `with-skill`, and `self-generated`.
- `CaseResult.mode`: records the named mode for each result. The existing boolean `with_skill` can be retained temporarily for compatibility if needed.
- `SkillEvaluator.run(..., modes=...)`: drives selected modes per agent and per case.
- `generate_tasks(..., mode=...)`: continues to materialize ordinary BenchFlow task folders for `baseline` and `with-skill`.
- Self-generation runner: creates and runs the creator/solver flow for each case, persists artifacts, and converts outcomes into `CaseResult`.
- CLI mode parsing: maps repeated `--mode` values, default behavior, and `--no-baseline`.

## Self-Generated Data Flow

For each eval case and solver agent:

1. BenchFlow creates isolated per-case task materialization.
2. BenchFlow runs a creator scene through `Trial`.
3. The creator reads the task/eval instruction and Anthropic-style skill-creator prompt.
4. The creator writes a full skill directory to a deterministic sandbox path, such as `/app/generated-skills/<case-id>/<skill-name>/`.
5. BenchFlow validates only the generated skill directory boundary and `SKILL.md` metadata.
6. BenchFlow persists the full generated skill pack under the job output.
7. BenchFlow runs a fresh solver scene/session with the generated skill directory mounted through normal skill discovery paths.
8. BenchFlow records reward, rubric details, errors, and artifact paths under `mode=self-generated`.

The creator must be allowed to generate more than `SKILL.md`. Optional `scripts/`, `references/`, `assets/`, examples, or other support files should be preserved and mounted when present.

## Validation

Validation is intentionally narrow. A generated skill pack is valid when:

- The generated skill directory exists.
- It contains `SKILL.md`.
- `SKILL.md` has valid YAML frontmatter.
- Frontmatter has non-empty `name` and `description`.

BenchFlow should not reject a generated pack because optional files are absent or because the instructions are low quality. Skill quality is measured by eval reward, not by runtime validation.

## Error Handling

Self-generated mode records partial failures as case results:

- Creator failure: result has `mode=self-generated`, `reward=None`, an error beginning with `skill generation failed:`, and no solver run.
- Invalid generated pack: result has `mode=self-generated`, `reward=None`, an explanatory validation error, and retained generated artifacts when available.
- Solver failure: result follows existing eval error behavior, with `mode=self-generated`.

Temporary task materialization may be cleaned up after execution, but generated skill packs and result metadata must persist in `jobs/`.

Recommended artifact path:

```text
jobs/skill-eval/<skill>/<agent>/self-generated/<case>/generated-skills/
```

## Results And Reporting

Summary tables should report `self-generated` as a peer mode, not as a variant hidden under `with-skill`.

Each result should include:

- case id
- agent
- model
- mode
- reward
- error
- tool-call count
- rubric details when available
- generated skill artifact path for `self-generated`

Lift reporting should compare selected modes clearly. For backward compatibility, the existing `LIFT` row can continue to mean `with-skill` minus `baseline`; self-generated can add an explicit delta against baseline when both modes are present.

## Testing

Add unit tests for mode parsing:

- Default modes remain `with-skill` plus `baseline`.
- `--no-baseline` maps the default mode set to `with-skill`.
- `self-generated` can run alone or with other modes.
- Invalid modes fail clearly.

Add unit tests for generated skill validation:

- Accepts valid `SKILL.md` with `name` and `description`.
- Rejects missing `SKILL.md`.
- Rejects invalid frontmatter.
- Rejects empty `name` or `description`.
- Preserves optional generated files such as `scripts/`, `references/`, and `assets/`.

Add runtime wiring tests with mocks:

- `baseline` and `with-skill` continue using BenchFlow `Job`.
- `self-generated` uses BenchFlow `Trial`/scene execution.
- Creator failure records a case error and skips solver.
- Invalid generated skill records a case error and skips solver.
- Generated skill artifacts persist under `jobs/skill-eval/.../self-generated/...`.
- Summary tables report `self-generated` as a distinct mode.

No live benchmark is required for default CI. A documented optional live smoke test may run a small self-generated eval through `bench`.

## Non-Goals

- No GEPA optimization loop in the initial implementation.
- No live GitHub fetch of Anthropic prompts at runtime.
- No custom HTML viewer for this mode.
- No direct Harbor command or user workflow.
- No broad refactor of BenchFlow task, trial, or job abstractions beyond what named modes require.
