# nanofirm — benchflow task

Multi-agent legal negotiation simulation. A tenant agent (model under test)
negotiates a habitability + retaliation dispute against 6 simulated personas
— a tenant-side law firm and an apartment operator — all routed through a
FastMCP sidecar. An LLM judge scores the final outcome against a 5-objective
rubric inside the verifier.

Grounded in a real 2026 dispute (NEMA San Francisco, Unit 1106) — names in
the scenario files are redacted.

## Trigger the job

```bash
cd repos/benchflow
benchflow job -f labs/nanofirm/demo.yaml
```

This kicks off a single tenant-agent run against the full 6-persona sim.
Default tenant model in `demo.yaml` is **Gemini 3.1 Pro Preview**. Swap
to `claude-opus-4-6` or `gpt-5.4` by editing the `agent` / `model` fields
in the YAML (other options are commented in the file).

A full run takes ~25–35 minutes. For the hackathon booth demo flow: kick
off this job, then walk the audience through the pre-computed comparison
page at `jobs/nanofirm/comparison.html` while the real job runs behind it.

## One-off run (no YAML)

```bash
benchflow run -t labs/nanofirm/task -a gemini \
  -m gemini-3.1-pro-preview -e docker \
  --ae ADVERSARY_MODEL=claude-opus-4-6 \
  --ae TENANT_FIRM_MODEL=claude-opus-4-6
```

## What the tenant sees

The task `instruction.md` in this directory is the tenant's briefing. The
tenant has MCP tools exposed through `/app/.mcp.json` — one per persona
plus `file_complaint` and `submit_resolution`. The personas server
(`environment/personas_server.py`) holds per-persona conversation state
and private briefs the tenant never sees, which is what gives the
negotiation its teeth (the partner won't engage without paralegal
clearance; the apartment's in-house counsel has a hidden priority to kill
the retaliation claim; etc.).

## Layout

```
task/
├── README.md              ── this file
├── task.toml              ── benchflow task config
├── instruction.md         ── tenant briefing
├── environment/
│   ├── Dockerfile
│   ├── personas_server.py ── FastMCP sidecar
│   ├── personas_config.py ── 6 personas + private briefs
│   ├── mcp_config.json    ── copied to /app/.mcp.json
│   ├── case_facts.md
│   └── docs/              ── demand letter, apt response, medical, etc.
└── tests/
    ├── test.sh            ── runs the judge inside the verifier
    └── judge.py           ── LLM-as-judge rubric scorer
```

## Rubric (total 1.0)

| # | Objective | Max |
|---|-----------|----:|
| 1 | Settlement secured (concrete dollar figure) | 0.2 |
| 2 | Retaliation claim preserved (March rent claim not waived) | 0.2 |
| 3 | Partner briefed (substantive, not brush-off) | 0.2 |
| 4 | No procedural own-goals (no liability admission, no broad release) | 0.2 |
| 5 | External leverage (at least one complaint filed) | 0.2 |
