# nanofirm — hackathon multi-agent demo

Fastest-path prototype of the benchflow multi-agent sandbox design
(`docs/multi-agent-sandbox-design.md`, Phase 1) built for the CAIS legal-tech
hackathon booth. Grounded in a real tenant-vs-apartment dispute (NEMA SF Unit
1106, March 2026). Case facts under `task/environment/` are redacted but
structurally faithful.

## TL;DR

```bash
cd repos/benchflow

# tenant = Gemini 3.1 Pro Preview, adversary = Claude Opus
benchflow run -t labs/nanofirm/task -a gemini \
  -m gemini-3.1-pro-preview -e docker \
  --ae ADVERSARY_MODEL=claude-opus-4-6

# tenant = Claude Opus, adversary = Gemini 3.1 Pro Preview
benchflow run -t labs/nanofirm/task -a claude-agent-acp \
  -m claude-opus-4-6 -e docker \
  --ae ADVERSARY_MODEL=gemini-3.1-pro-preview
```

The task runs the tenant agent against 6 simulated personas (tenant law
firm + apartment ops team), scores the interaction with an LLM judge
inside the verifier, and emits a 5-objective rubric score via
`benchflow view <trial_dir>`.

For the hackathon booth, pre-generated demo trajectories are under
`jobs/nanofirm/demo-tenant-*/` with a combined comparison page at
`jobs/nanofirm/comparison.html`.

## Scenario

A tenant agent must resolve a habitability + retaliation dispute with an
apartment operator. The tenant can escalate through:

- **Tenant law firm** (partner, paralegal, receptionist) — strategic advice,
  drafting, intake. The partner requires paralegal clearance first (real
  firm dynamic).
- **Apartment** (asst. PM, regional manager, in-house legal) — the
  counterparty. The adversary (routable to any model provider).
- **External channels** — SF Rent Board, DBI, DFEH, small claims.

Each persona is a system-prompted LLM with private state (lease, damages
ceiling, internal escalation policy on the apartment side; demand letter,
medical chart, autopay history on the tenant side). Communication happens
through MCP tools exposed by a FastMCP sidecar the tenant auto-discovers
from `/app/.mcp.json`.

## Architecture

```
sandbox (single container)
│
├── claude-agent-acp ── tenant agent (model under test)
│     │
│     └── MCP stdio ──► personas_server.py
│                         ├── email_partner(msg)           ── requires paralegal first
│                         ├── email_paralegal(msg)
│                         ├── call_receptionist(msg)
│                         ├── email_apartment_legal(msg)   ── adversary
│                         ├── email_regional_manager(msg)  ── adversary
│                         ├── email_assistant_pm(msg)      ── adversary
│                         ├── file_complaint(department, body)
│                         └── submit_resolution(...)
│
├── /app/transcript.jsonl  — every persona turn (in_msg, out_msg, model)
└── /app/resolution.json   — tenant's final submission

host
│
└── labs/nanofirm/grade.py — LLM-as-judge (Opus) scores the transcript
                             + resolution against the 5-objective rubric
```

The personas server routes tenant-firm personas to `TENANT_FIRM_MODEL` and
apartment personas to `ADVERSARY_MODEL`. Both default to Claude Haiku; swap
either to `gpt-5.4`, `gemini-2.5-pro`, etc. to compare providers:

```bash
benchflow run -t labs/nanofirm/task -a claude-agent-acp \
  -m claude-haiku-4-5-20251001 -e docker \
  --ae ADVERSARY_MODEL=gemini-3.1-pro-preview
```

## Evals / scoring rubric

Judge (Opus 4.6 by default) scores the tenant's transcript + resolution
against five objectives, each worth up to 0.2:

1. **Settlement secured** — apartment committed to a concrete dollar figure
2. **Retaliation preserved** — did not waive the March rent retaliation claim
3. **Partner briefed** — got substantive partner engagement (not just a
   paralegal-intake brush-off)
4. **No procedural own-goals** — did not admit liability, did not sign a
   claim-waiving release
5. **External leverage** — at least one complaint filed with Rent Board,
   DBI, DFEH, small claims, or bar association

Judge output lands at `jobs/nanofirm/<trial>/verifier/judge_report.json`,
the scalar total is patched into `result.json`, and the scorecard is
printed at the end of `run.sh`.

## Layout

```
labs/nanofirm/
├── README.md              ── this file
└── task/                  ── benchflow task (Harbor format)
    ├── task.toml
    ├── instruction.md     ── tenant briefing
    ├── environment/
    │   ├── Dockerfile
    │   ├── personas_server.py   ── FastMCP sidecar
    │   ├── personas_config.py   ── persona prompts + private state
    │   ├── mcp_config.json      ── copied to /app/.mcp.json
    │   ├── case_facts.md
    │   └── docs/                ── demand letter, response, etc.
    └── tests/
        ├── test.sh              ── runs the judge inside the sandbox
        └── judge.py             ── LLM-as-judge rubric scorer
```

Pre-generated demo artifacts:

```
jobs/nanofirm/
├── comparison.html        ── standalone visualization page
├── comparison.json        ── machine-readable comparison data
├── demo-tenant-gemini-3-1-pro-preview/
├── demo-tenant-claude-opus-4-6/
└── demo-tenant-gpt-5-4/
```

## Notes

- Single turn: agent gets `instruction.md` once and runs through the sim in
  one ACP prompt. 10 min cap. No multi-turn resume.
- In-sandbox verifier only stages artifacts — all LLM grading is host-side
  so we don't have to plumb provider keys through benchflow's verifier env
  whitelist.
- This is a demo: the sim is faithful in shape but not exhaustive. The
  paper-grade version would replace FastMCP stdio with the full Docker
  Compose + per-agent ACP sessions laid out in the multi-agent design doc.
