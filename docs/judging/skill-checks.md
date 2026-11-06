# Skill checks

Scan skills for security and quality issues without running an agent, a sandbox or a model.

```bash
bench skills validate ./my-skill
bench skills validate tasks/ --json > skill-checks.json
bench skills validate tasks/citation-check --fail-on critical -o reports/
```

Point the command at skills, or at any folder to search for `SKILL.md`. Every path must exist and hold at least one skill; otherwise it exits 1 before checking anything.

## Install the checkers

`bench skills validate` runs two NVIDIA tools:

| Tool | Checks |
|---|---|
| [SkillEvaluator](https://github.com/NVIDIA/SkillEvaluator) Tier 1 | schema, version, PII, license, code integrity, dependency audit, Unicode smuggling, script lint, and a quality score (0-100 with a letter grade) |
| [SkillSpector](https://github.com/NVIDIA/SkillSpector) | security patterns such as prompt injection, data exfiltration, anti-refusal statements and undeclared capabilities, with a 0-100 risk score and a verdict: `SAFE`, `CAUTION` or `DO_NOT_INSTALL` |

Both are Apache-2.0 and not on PyPI, so BenchFlow does not depend on them. Install them once from their release tags:

```bash
uv tool install --python 3.13 "skillevaluator[all] @ git+https://github.com/NVIDIA/SkillEvaluator.git@v0.5.0"
uv tool install "skillspector @ git+https://github.com/NVIDIA/SkillSpector.git@v2.12.0"
```

BenchFlow runs whichever tool is installed, and warns when its version is older than these. To use an executable that is not on `PATH`, set `BENCHFLOW_SKILLEVALUATOR_BIN` or `BENCHFLOW_SKILLSPECTOR_BIN`.

## Read the output

```text
$ bench skills validate tasks/3d-scan-calc/environment/skills/mesh-analysis \
    tasks/citation-check/environment/skills/citation-management \
    tasks/crystallographic-wyckoff-position-analysis/environment/skills/pymatgen
                   Skill checks (SkillEvaluator 0.5.0 + SkillSpector 2.12.0)
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Skill               ┃ Tier 1     ┃ Quality ┃ Security                             ┃ Blocking ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ mesh-analysis       │ incomplete │ C 79    │ LOW 0 SAFE                           │        1 │
│ citation-management │ incomplete │ C 78.5  │ HIGH 65 DO_NOT_INSTALL (partial)     │        4 │
│ pymatgen            │ incomplete │ B 80.5  │ CRITICAL 86 DO_NOT_INSTALL (partial) │        6 │
└─────────────────────┴────────────┴─────────┴──────────────────────────────────────┴──────────┘
mesh-analysis
  HIGH     skillevaluator SCHEMA author_missing: Author not specified in metadata (SKILL.md)
citation-management
  HIGH     skillevaluator SCHEMA author_missing: Author not specified in metadata (SKILL.md)
  HIGH     skillspector Anti-Refusal AR2: no warnings (assets/citation_checklist.md:203)
  ...
Incomplete SkillEvaluator scans (scanner missing or partial): gitleaks, semgrep
3 skill(s) checked, 3 blocking (--fail-on high)
```

This is three SkillsBench skills, checked with SkillEvaluator 0.5.0 and SkillSpector 2.12.0. `(partial)` means SkillSpector could not inspect every file.

## When a skill blocks

The command exits 1 when any skill blocks. A skill blocks when:

- a schema, security or dependency finding is at or above `--fail-on`; or
- a checker fails to run.

| Finding | Blocks? | From |
|---|---|---|
| schema | yes | SkillEvaluator |
| security | yes | SkillSpector, plus SkillEvaluator's PII, secrets, Unicode-smuggling and code-risk (Bandit, Semgrep) findings |
| dependency | yes | known vulnerabilities pip-audit finds in pinned packages, at SkillEvaluator's severity, and banned packages (high) |
| quality, lint, license, version, dead links | no; advisory | SkillEvaluator |

A failed dependency audit with nothing BenchFlow can parse still blocks. Advisory findings appear in `--json` and in the raw reports that `-o DIR` keeps.

SkillEvaluator's default policy makes a missing `metadata.author` a high schema finding. That is why mesh-analysis blocks above. `author` is nested under `metadata` in the frontmatter:

```yaml
metadata:
  author: Name <email@example.com>
```

`--fail-on critical` relaxes the gate, and all three skills pass at that level. pymatgen's `CRITICAL` is SkillSpector's overall risk rating, not the severity of any one finding.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--fail-on` | `high` | lowest blocking severity: `critical`, `high`, `medium`, `low`, or `never` to only report |
| `--no-security` | off | skip SkillSpector |
| `--checks` | BenchFlow's list | which SkillEvaluator checks to run |
| `--min-score` | | the quality score SkillEvaluator's own verdict needs. Advisory: it changes the Tier 1 status, not the exit code. |
| `-o`, `--output-dir` | | keep each skill's raw JSON reports under `DIR/<skill>/` |
| `-j`, `--workers` | `4` | skills checked in parallel |
| `--max-findings` | `5` | blocking findings listed per skill; all are in `--json` |
| `--json` | off | print machine-readable JSON |

The default checks leave out SkillEvaluator's `security` check. It reruns SkillSpector, which BenchFlow already runs once directly, so adding it with `--checks` lists SkillSpector's findings twice. Bandit needs no extra check: code integrity already runs it.

## Known gaps

- Without Semgrep and Gitleaks installed, those scans are reported incomplete. Bandit always runs.
- The dependency audit runs pip-audit on the skill's top-level `requirements*.txt` files. It checks pinned packages against an online database, so it needs the network.
- When pip-audit gives no result for a file, for example because its pins can't be installed together, SkillEvaluator 0.5.0 passes the audit silently. BenchFlow lists `pip-audit` as incomplete instead.
- SkillEvaluator makes the dependency audit opt-in. BenchFlow runs it by default; a `--checks` list without `dependency` leaves it out.
- Both tools run with `--no-llm`, so SkillEvaluator's LLM rubric and SkillSpector's LLM pass are off.
- SkillEvaluator Tier 2 (deduplication, which needs an embeddings provider) and Tier 3 (live agent runs) are not wired in. Use [skill evals](./skill-evals.md) to measure a skill live.

## Next steps

- [Skill evals](./skill-evals.md): measure how much a skill helps.
- [Which judge?](./index.md): every way to grade.
