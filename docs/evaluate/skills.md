# Skills

A skill is a folder of instructions an agent can read while it works: a `SKILL.md`, often with `references/` and `scripts/`. SkillsBench tasks ship skills in `environment/skills/`, and the benchmark asks whether they help. BenchFlow leaves a task's skills out unless you ask for them. To find out, run an agent with them, check that it read them, and measure the difference.

Runs here use MiMo (`-m xiaomi/mimo-v2.6-flash`): export its key and base URL first ([Quick start](../start/quick-start.md#on-another-provider)), or use `-m sonnet` with a Claude login.

## Give the agent the task's skills

`--task-skills` installs the task's skills where the agent looks for its own; for Claude Code, that's its skills folder, with the files under `/skills`. The same run as in [Run a benchmark](./run-a-benchmark.md), with skills:

```bash
bench run skillsbench/tasks \
  -i econ-detrending-correlation \
  -i crystallographic-wyckoff-position-analysis \
  -i offer-letter-generator \
  -a claude -m xiaomi/mimo-v2.6-flash --trials 3 -j 3 \
  --task-skills --jobs-dir jobs/skills
```

```text
Matrix: 3 trial job(s), 7/9 trials passed, errors=0 verifier-errors=0
```

The instruction is the same with and without skills; the agent finds them the way it finds any skill. Each trial's `config.json` records the arm: `"skill_mode": "with-skill"` and `"skills_sandbox_dir": "/skills"`. `--skills-dir PATH` installs a folder of skills of your own instead.

## Confirm the agent used them

An installed skill isn't a used one, and a trajectory that mentions the skills folder hasn't necessarily read it. Search the agent's tool calls, whose commands are in each event's `title`, for the folder:

```bash
find jobs/skills/claude-agent-acp -path '*/trajectory/acp_trajectory.jsonl' -exec \
  grep -l -E '"title": "[^"]*/skills/' {} +
```

```text
jobs/skills/claude-agent-acp/trial-02/2026-11-04__18-56-59/crystallographic-wyckoff-position-analysis__2168c849/trajectory/acp_trajectory.jsonl
```

One trial of nine ran a tool on the skills folder; the other eight have no tool call that names it. Its two calls listed the folder and read one reference file:

```bash
grep -o -E '"title": "[^"]*/skills/[^"]{0,40}' \
  jobs/skills/claude-agent-acp/trial-02/2026-11-04__18-56-59/crystallographic-wyckoff-position-analysis__2168c849/trajectory/acp_trajectory.jsonl
```

```text
"title": "Read /skills/pymatgen/references/wyckoff-positions-fr
"title": "ls /skills/ 2>/dev/null; echo \
```

It was the crystallographic task's second try, the only one of its three that passed, and its final message says how the skill helped:

```text
**Approach** (following the `pymatgen` skill reference at `/skills/pymatgen/references/wyckoff-positions-from-cif.md`):
...
Key pitfall avoided: the skill explicitly warns *not* to use `get_symmetrized_structure().wyckoff_symbols`, which returns symbols like `"4a"` and orbit counts rather than International Tables letters with atom counts.
```

The other two crystallographic tries never named the folder, and timed out as they did without skills. The two other tasks passed every time, skills or not, with no tool call on a skill.

The search sees reads that name the folder's path. An agent that reaches the files another way, such as a relative path after `cd /skills`, is in its full trajectory, so read a few of those before you say a skill went unused.

## Measure whether skills help

Run the same tasks, agent and model with and without `--task-skills`, then compare the two runs task by task:

```bash
bench results jobs/no-skills/claude-agent-acp jobs/skills/claude-agent-acp
```

```text
3 tasks paired (3 scored on both sides): skills higher on 1, lower on 0, same on 2; mean delta (skills - no-skills) 0.111111. 0 only in no-skills, 0 only in skills.

| Task | no-skills | skills | Delta | Status |
|---|---|---|---|---|
| crystallographic-wyckoff-position-analysis | 0 | 0.333333 | 0.333333 | paired |
| econ-detrending-correlation | 1 | 1 | 0 | paired |
| offer-letter-generator | 1 | 1 | 0 | paired |
```

Seven of nine trials passed with skills, six without. That difference is one trial on one task, and three tries on three tasks can't tell it from chance; [Compare runs](../audit/compare-runs.md) shows how many tries would. What the run does show is how rarely the agent opened them: in one trial of nine. A skill the agent never reads can't change its score, so report how often it read one beside the score.

## Check a skill before you use it

A skill runs with the agent's permissions, so read it, or check it, before you give it to an agent. `bench skills validate` runs two static checkers on skill folders, NVIDIA's SkillEvaluator and SkillSpector. Install them first; these are the versions 0.9.2 asks for when they're missing (0.10.0rc2 asks for SkillEvaluator v0.6.0):

```bash
uv tool install --python 3.13 "skillevaluator[all] @ git+https://github.com/NVIDIA/SkillEvaluator.git@v0.5.0"
uv tool install "skillspector @ git+https://github.com/NVIDIA/SkillSpector.git@v2.12.0"
```

`BENCHFLOW_SKILLEVALUATOR_BIN` and `BENCHFLOW_SKILLSPECTOR_BIN` point BenchFlow at copies installed elsewhere. On the crystallographic task's two skills:

```bash
bench skills validate skillsbench/tasks/crystallographic-wyckoff-position-analysis/environment/skills
```

```text
┃ Skill    ┃ Tier 1     ┃ Quality ┃ Security                             ┃ Blocking ┃
│ pymatgen │ incomplete │ B 80.5  │ CRITICAL 86 DO_NOT_INSTALL (partial) │        6 │
│ sympy    │ incomplete │ B 87.2  │ LOW 0 CAUTION (partial)              │        1 │
pymatgen
  HIGH     skillevaluator SCHEMA author_missing: Author not specified in metadata (SKILL.md)
  HIGH     skillspector analysis-evasion AE1: references/io_formats.md (partial) (SKILL.md:168)
...
Incomplete SkillEvaluator scans (scanner missing or partial): gitleaks, semgrep
2 skill(s) checked, 2 blocking (--fail-on high)
```

SkillSpector marks the `pymatgen` skill `DO_NOT_INSTALL`, over the way its `SKILL.md` points into a reference file; read the findings (`--json` lists them all) and the files they name, and decide for yourself. Tier 1 is `incomplete` here because two of SkillEvaluator's scanners, gitleaks and semgrep, weren't installed. `bench skills eval <skill>` runs a skill's own test cases from its `evals/evals.json`, with and without the skill.
