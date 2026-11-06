# SkillsBench skill matrix on Daytona

This recipe runs four cells: Opus 4.8 on Bedrock and Gemini 3.5 Flash, each with and without the task's skills, with the `openhands` agent on Daytona.

Each cell writes a trajectory (`trajectory/{acp,llm}_trajectory.jsonl`) and a verifier reward. A cell counts as done only after the audit in [Check the batch](#check-the-batch) passes.

## Setup (once per shell)

```bash
# 1. Run the CLI from a benchflow checkout (inside the repo, use `uv run bench`).
#    BenchFlow starts LiteLLM as the gateway for Bedrock and Gemini.
cd /path/to/benchflow

# 2. A local skillsbench clone, so the tasks' bundled skills are on disk.
git clone https://github.com/benchflow-ai/skillsbench
export SKILLSBENCH=$PWD/skillsbench

# 3. Credentials. Test each key first: a dead key shows up only as
#    "openhands ACP error -32603" on the first model call.
#      Bedrock: a `converse` call with `Authorization: Bearer $AWS_BEARER_TOKEN_BEDROCK` -> 200
#      Gemini:  `.../v1beta/models/<model>:generateContent?key=$GEMINI_API_KEY`           -> 200
export AWS_BEARER_TOKEN_BEDROCK=...
export AWS_REGION=us-west-2 AWS_DEFAULT_REGION=us-west-2
export GEMINI_API_KEY=...

# 4. Adaptive thinking for Opus 4.8. LiteLLM's Bedrock support clamps
#    xhigh and max to high, so ask for high directly.
export BENCHFLOW_BEDROCK_THINKING_EFFORT=high

# 5. Clear old gateway variables; BenchFlow writes its own LiteLLM config.
unset LLM_BASE_URL LLM_API_KEY OPENAI_BASE_URL BENCHFLOW_PROVIDER_BASE_URL \
      BENCHFLOW_PROVIDER_API_KEY LITELLM_BASE_URL LITELLM_API_KEY
```

Pick a light task: Daytona caps each sandbox at 10 GB. `citation-check` is a good default. Opus's adaptive thinking makes each turn slower than the Gemini cells, so keep the first task light.

## Run the four cells

```bash
TASK=citation-check
COMMON="--tasks-dir $SKILLSBENCH/tasks --include $TASK --agent openhands \
  --sandbox daytona --concurrency 1 --usage-tracking required --idle-timeout none"

# (1) Opus 4.8 (adaptive high), with skills
bench run $COMMON --model aws-bedrock/us.anthropic.claude-opus-4-8 \
  --skill-mode with-skill --jobs-dir jobs/opus-skill

# (2) Opus 4.8 (adaptive high), without skills
bench run $COMMON --model aws-bedrock/us.anthropic.claude-opus-4-8 \
  --skill-mode no-skill --jobs-dir jobs/opus-noskill

# (3) Gemini 3.5 Flash, with skills
bench run $COMMON --model gemini-3.5-flash --env LLM_CACHING_PROMPT=false \
  --skill-mode with-skill --jobs-dir jobs/gemini-skill

# (4) Gemini 3.5 Flash, without skills
bench run $COMMON --model gemini-3.5-flash --env LLM_CACHING_PROMPT=false \
  --skill-mode no-skill --jobs-dir jobs/gemini-noskill
```

| Flag | Why |
|---|---|
| `--usage-tracking required` | A trial with no provider-reported token usage fails, instead of passing with no usage data |
| `--idle-timeout none` | Turns off the idle watchdog; the task's wall-clock limit still applies |
| `--env LLM_CACHING_PROMPT=false` | Passed to the agent for the Gemini cells only |

LiteLLM writes each provider call to `trajectory/llm_trajectory.jsonl`. Check that the effective thinking effort there is `high`; requested `xhigh` or `max` are also recorded as `high`.

## Check the batch

A finished command is not a healthy trial. Audit each batch with the `benchflow-experiment-review` skill (`.agents/skills/benchflow-experiment-review`, also reachable as `.claude/skills`). A trial is healthy only when all of these hold:

- the trajectory and its timing, token and tool metadata are complete;
- its pass, fail or timeout status is right;
- the verifier started after the agent exited;
- there is no reward hacking;
- with-skills cells show `task_skills_loading: 1`. Without-skills cells show `task_skills_loading: 0`, with no task skill in the trajectory.

Openhands' built-in `.agents/skills` and `invoke_skill` appear in every run; they are not leakage.

Quick checks before the full audit, per jobs folder:

```bash
J=jobs/opus-skill
find $J -name rewards.jsonl -exec tail -1 {} \;                                  # reward
grep -ho '"usage_source": "[a-z_]*"' $(find $J -name result.json)                # expect provider_response
grep -ho '"effort": "[a-z]*"' $(find $J -name llm_trajectory.jsonl) | sort -u    # Opus -> "high"
python3 .agents/skills/benchflow-experiment-review/scripts/extract_harness_skills.py \
  "$(find $J -name llm_trajectory.jsonl | head -1)" --task-skill <task-skill-name>
```

## Notes

- On Daytona and AgentCore, the transport's read guard follows the idle timeout (or, with it off, the wall-clock limit). A long silent generation is judged by the watchdog, not by a fixed 900 s read timeout. Set `BENCHFLOW_DAYTONA_PTY_READLINE_TIMEOUT` only to force a value.
- Opus 4.8 on Bedrock needs the adaptive-thinking patch that LiteLLM loads into its proxy (`src/benchflow/providers/litellm_bedrock_patch.py`).
- Claude models newer than LiteLLM's model map (Fable 5.1, Opus 5 and 5.5, Sonnet 5) need the same startup patch to keep their effort setting. Bedrock routes stop at startup if it did not apply. The effort grep above shows the effort actually used.
- For heavy tasks, replace `--sandbox daytona` with `--sandbox docker`; the other flags stay.

## Next steps

- More on running SkillsBench: [Benchmarks](../datasets/benchmarks.md#skillsbench).
- Compare the cells: `bench results jobs/opus-skill`, and `bench eval metrics` in the [CLI reference](../reference/cli.md).
