# SOP: recover startup skill catalogs from SkillsBench harness trajectories

The primary source of truth is `trajectory/llm_trajectory.jsonl`; use sibling `trajectory/acp_trajectory.jsonl` as a fallback for the few usage-only LLM logs.

The field paths and sample skill counts below are point-in-time observations from the audited dataset. Verify them against the current dataset before treating any count or marker as authoritative.

## Table of contents

- [Executive summary](#executive-summary)
- [Fast path](#fast-path)
- [General manual procedure](#general-manual-procedure)
- [Harness rules](#harness-rules)
  - [Claude Code](#claude-code)
  - [Codex](#codex)
  - [OpenHands](#openhands)
  - [Gemini CLI](#gemini-cli)
  - [Pi Agent / `pi-acp`](#pi-agent--pi-acp)
- [Old PR caveat](#old-pr-caveat)
- [Audit checklist](#audit-checklist)

## Executive summary

| Harness | Where startup prompt lives | Where task prompt lives | Skill catalog format | Current PR4 result |
| --- | --- | --- | --- | --- |
| `claude-code` | `request.body.system` plus sometimes a `<system-reminder>` inside `request.body.messages[]` | first user message, often after system reminders | Markdown bullets after `The following skills are available for use with the Skill tool:` | Some trajectories expose a Claude Code Skill catalog; some current c50 rows do not. Do not infer a catalog if the marker is absent. |
| `codex` | OpenAI Responses `request.body.instructions` plus developer message in `request.body.input[]` | `request.body.input[]` role `user`; env context then task text | `<skills_instructions>` block; bullets `- name: description (file: path)` | Catalog is in the developer message, not in `instructions`. |
| `openhands` | `request.body.messages[role=system]` or Bedrock `request.body.system` | `request.body.messages[role=user]` | XML-ish `<SKILLS><available_skills><skill><name>...` | Catalog is directly in startup system prompt; use `invoke_skill(name=...)`. |
| `gemini-cli` | `request.body.systemInstruction.parts[].text` | `request.body.contents[]`; session context then task text | `# Available Agent Skills` then XML-ish `<available_skills>` | Catalog is directly in `systemInstruction`; use `activate_skill`. |
| `pi-acp` / pi-agent | `request.body.messages[0].role == system` | `request.body.messages[1].role == user` | No startup skill catalog serialized in current PR4 | System prompt mentions pi docs/skills.md only; tools are `read/bash/edit/write`. |

Important caveat: the visible startup catalogs in current PR4 are mostly harness-global skills. They are not a reliable proof that a task-specific SkillsBench skill was or was not available. For no-skill leakage audits, also scan tool/file traces for access to `SKILL.md`, `.codex/skills`, `.agents/skills`, `activate_skill`, `invoke_skill`, and Claude Code `Skill` calls.

## Fast path

Use the helper script from this skill package:

```bash
scripts/extract_harness_skills.py /path/to/trajectory/llm_trajectory.jsonl
```

It scans the LLM trajectory, falls back to sibling `acp_trajectory.jsonl` when
startup request bodies are absent, and prints JSON:

```json
{
  "harness": "openhands",
  "catalog_format": "...",
  "skill_count": 42,
  "skills": [{"name": "code-simplifier", "description": "..."}]
}
```

If it prints `unknown`, `skill_count: 0`, or `manual_review_required: true`,
continue with the manual procedure below. If it prints `pi-acp` with
`skill_count: 0`, that means no startup skill catalog was serialized, not that
pi has no skill concept at all.

## General manual procedure

1. Open `trajectory/llm_trajectory.jsonl`.
2. Read JSONL rows from the top until you find a row with a real `request.body`.
3. Skip provider preflight rows, for example Claude Bedrock `GET /inference-profiles?type=SYSTEM_DEFINED`.
4. If rows are usage-only records like `{"type": "llm_completion", "usage": ...}`, use sibling `trajectory/acp_trajectory.jsonl` and look for an early `agent_thought` beginning with `System Prompt:`.
5. Identify the harness by request shape, path, and tool names.
6. Extract the catalog using the harness-specific rules below.
7. Record: harness, trajectory path, JSONL line index, request path, catalog source field, skill names, skill count, and SHA-256 of the exact recovered startup prompt/catalog text.

## Harness rules

### Claude Code

Typical raw shape:

```json
{
  "request": {
    "path": "/model/.../invoke-with-response-stream",
    "body": {
      "system": [{"type": "text", "text": "..."}],
      "messages": [{"role": "user", "content": [{"type": "text", "text": "<system-reminder>..."}]}],
      "tools": [{"name": "Agent"}, {"name": "Skill"}, {"name": "ToolSearch"}]
    }
  }
}
```

Extraction:

1. Flatten `body.system[]` text and `body.messages[]` text.
2. Find:

```text
The following skills are available for use with the Skill tool:
```

3. Parse following bullet entries:

```text
- deep-research: Deep research harness ...
- update-config: Use this skill to configure ...
```

Regex sketch:

```regex
^-\s+([^:\n]+):\s+(.*?)(?=\n-\s+|\Z)
```

Activation evidence:

- Request tools include `Skill`.
- Subsequent traces may contain a `Skill` tool use, or tool loading through `ToolSearch`.

Current PR4 note:

- Some Claude Code rows expose 13 skills, e.g. `deep-research`, `update-config`, `keybindings-help`, `verify`, `code-review`.
- Some c50 rows include the `Skill` tool but no bullet catalog in startup text. Mark these as `catalog_absent_in_startup_prompt`; do not fill from another run.

### Codex

Typical raw shape:

```json
{
  "request": {
    "path": "/responses",
    "body": {
      "instructions": "You are Codex...",
      "input": [
        {"role": "developer", "content": [{"type": "input_text", "text": "<permissions instructions>...<skills_instructions>..."}]},
        {"role": "user", "content": [{"type": "input_text", "text": "<environment_context>..."}]},
        {"role": "user", "content": [{"type": "input_text", "text": "TASK..."}]}
      ]
    }
  }
}
```

Extraction:

1. Do not stop at `body.instructions`; that field has base Codex instructions but no skill list in current PR4.
2. Flatten all `body.input[]` messages with `role == "developer"`.
3. Extract:

```xml
<skills_instructions>
...
### Available skills
- imagegen: ... (file: /home/agent/.codex/skills/.system/imagegen/SKILL.md)
...
### How to use skills
...
</skills_instructions>
```

4. Parse only between `### Available skills` and the next `###` heading.

Regex sketch:

```regex
^- ([^:\n]+): (.*?) \(file: ([^)]+)\)$
```

Current PR4 sample catalog contains 5 skills:

```text
imagegen
openai-docs
plugin-creator
skill-creator
skill-installer
```

Activation evidence:

- Codex normally follows the instruction by opening/reading the listed `SKILL.md`; there is no dedicated `invoke_skill` tool in this trajectory shape.
- For leakage audit, scan tool calls for reads of `/home/agent/.codex/skills/.../SKILL.md` or other skill paths.

### OpenHands

Typical raw shape:

```json
{
  "request": {
    "path": "/chat/completions",
    "body": {
      "messages": [
        {"role": "system", "content": "...<SKILLS>..."},
        {"role": "user", "content": "TASK..."}
      ],
      "tools": [{"function": {"name": "invoke_skill"}}]
    }
  }
}
```

Extraction:

1. Flatten `body.messages[]` where `role == "system"`; for Bedrock, flatten `body.system`.
2. Find:

```xml
<SKILLS>
The following skills are available...
To use a skill, call the `invoke_skill(name="<skill-name>")` tool...

<available_skills>
  <skill>
    <name>code-simplifier</name>
    <description>...</description>
  </skill>
...
</available_skills>
</SKILLS>
```

3. Parse every `<skill>` item under `<available_skills>`.

Regex sketch:

```regex
<skill>.*?<name>(.*?)</name>.*?<description>(.*?)</description>.*?</skill>
```

Current PR4 sample catalog has 42 skills, e.g. `code-simplifier`, `iterate`, `pdflatex`, `openhands-automation`, `qa-changes`.

Activation evidence:

- Startup tools include `invoke_skill`.
- ACP trajectory tool calls show `invoke_skill` with JSON such as `{"name": "..."}`.

### Gemini CLI

Typical raw shape:

```json
{
  "request": {
    "path": "/v1beta/models/...:streamGenerateContent?alt=sse",
    "body": {
      "systemInstruction": {"parts": [{"text": "...# Available Agent Skills..."}]},
      "contents": [
        {"role": "user", "parts": [{"text": "<session_context>..."}]},
        {"role": "user", "parts": [{"text": "TASK..."}]}
      ],
      "tools": [{"functionDeclarations": [{"name": "activate_skill"}]}]
    }
  }
}
```

Extraction:

1. Flatten `body.systemInstruction.parts[].text`.
2. Do not confuse `<available_subagents>` with skills.
3. Find:

```text
# Available Agent Skills
You have access to the following specialized skills. To activate a skill ...
```

4. Parse:

```xml
<available_skills>
  <skill>
    <name>skill-creator</name>
    <description>...</description>
    <location>/opt/.../skill-creator/SKILL.md</location>
  </skill>
</available_skills>
```

Current PR4 sample catalog has 1 skill: `skill-creator`.

Activation evidence:

- Tool declarations include `activate_skill`.
- After activation, traces should include returned `<activated_skill>` instructions/resources.

### Pi Agent / `pi-acp`

Typical raw shape:

```json
{
  "request": {
    "path": "/chat/completions",
    "body": {
      "messages": [
        {"role": "system", "content": "You are an expert coding assistant operating inside pi..."},
        {"role": "user", "content": "TASK..."}
      ],
      "tools": [{"function": {"name": "read"}}, {"function": {"name": "bash"}}, {"function": {"name": "edit"}}, {"function": {"name": "write"}}]
    }
  }
}
```

Extraction:

- Current PR4 pi trajectories do not serialize an available-skills catalog in the startup prompt.
- The system prompt references pi documentation paths such as `docs/skills.md`; those are docs references, not the active skill list.
- Treat `skill_count = 0` as `catalog_not_serialized`.

Activation evidence:

- Current sampled PR4 pi requests expose only `read`, `bash`, `edit`, `write`; no `activate_skill`/`invoke_skill` equivalent in startup tools.

## Old PR caveat

PR1/PR2/PR3 v0.1 generally have only `acp_trajectory.jsonl`. For OpenHands, the early ACP `agent_thought` often includes `System Prompt:` and can expose the skill catalog. For Claude Code, Codex, and Gemini CLI v0.1, the startup LLM request is not present, so the full startup system prompt / skill catalog is not reliably recoverable.

## Audit checklist

For each trajectory:

1. `llm_trajectory.jsonl` exists and has real request bodies.
2. Harness detected.
3. Startup prompt field(s) recorded.
4. Skill catalog anchor found or explicitly marked absent.
5. Skill names extracted with count.
6. SHA-256 of extracted catalog/startup prompt saved.
7. For no-skill trials, scan full trajectory for:

```text
SKILL.md
.codex/skills
.agents/skills
invoke_skill
activate_skill
"name": "Skill"
ToolSearch select:
```

8. If any of the above appears in a no-skill trajectory, inspect manually before marking healthy.
