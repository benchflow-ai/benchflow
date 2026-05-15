# BenchFlow

BenchFlow evaluates agents, skills, and benchmark workflows through reproducible task runs.

## Language

**Model-as-skill**:
A skill-evaluation benchmark where an agent receives specialist-model delegation guidance as a skill.
_Avoid_: live specialist sidecar, smoke runner, model placeholder cleanup

**Specialist model**:
A model framed as a delegated expert for a narrower class of work than the orchestrating agent.
_Avoid_: judge model, baseline agent

**Baseline run**:
An evaluation run where the same task is attempted without the skill being tested.
_Avoid_: no-op test, unrelated smoke run

## Relationships

- A **Model-as-skill** benchmark compares with-skill runs against **Baseline runs**.
- A **Specialist model** is the delegated expert described by a **Model-as-skill**.

## Example Dialogue

> **Dev:** "Should the PR add the SkillsBench smoke matrix?"
> **Domain expert:** "No. This PR should prove the existing **Model-as-skill** fixture works as a skill evaluation, comparing skill-mounted tasks against **Baseline runs**."

## Flagged Ambiguities

- "test as model as a skill" was used to mean a test for the existing **Model-as-skill** fixture, not a live MCP specialist sidecar or broad smoke runner.
