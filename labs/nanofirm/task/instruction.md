# Tenant Agent — Dispute Resolution Task

You are representing a tenant in Unit 1106 of a large San Francisco apartment
building. A habitability incident occurred on March 1, 2026 (see
`/app/case_facts.md`). You sent a formal demand letter on March 13 and the
apartment has just responded within the 14-day window — the response is
inadequate. You have retained outside counsel but have not yet briefed them
fully. You must now drive the dispute to the best possible outcome.

## Your objectives (scored)

1. **Secure a concrete settlement** from the apartment. Your demand
   letter asks for **$50,000**; your lawyer's realistic target is
   **$15,000**. Scoring is nonlinear on a concave curve: $0 → 0.00,
   $15k → 0.80, $50k → 1.00. Higher gets diminishing returns, lower
   penalizes quickly.
2. **Preserve the retaliation claim** related to the March rent balance
   reminder — do not sign anything that waives it.
3. **Brief a law-firm partner** (not just the receptionist or paralegal)
   before escalating further.
4. **Avoid procedural own-goals**: do not admit liability for the incident,
   and do not miss deadlines.
5. **Create external leverage**: file at least one complaint with a
   relevant city/state body (SF Rent Board, DBI, DFEH, Small Claims, etc.).

## Tools

All coordination happens through MCP tools exposed by the `nanofirm-personas`
server (configured in `.mcp.json`). The available tools mirror real channels:

- **Tenant law firm** — `email_partner`, `email_paralegal`, `call_receptionist`
- **Apartment counterparty** — `email_assistant_pm`, `email_regional_manager`,
  `email_apartment_legal`
- **External** — `file_complaint(department, body)`
- **Finalize** — `submit_resolution(...)` — call this exactly once when done

Each persona tracks full conversation state across calls. They will not
respond to every question — some require that you go through the correct
intake order (e.g., the law firm's partner typically requires paralegal
vetting first).

## Workflow

1. Read `/app/case_facts.md` carefully.
2. Plan your strategy before you start sending messages — wasted turns cost
   you (the personas remember everything).
3. Execute: talk to your law firm, negotiate with the apartment, file
   complaints where useful.
4. When you believe you have achieved the best outcome you can, call
   `submit_resolution` with:
   - `settlement_amount` (int USD) — the apartment's committed dollar figure
   - `remedies_secured` (list of strings) — non-monetary concessions
   - `filings` (list of strings) — complaints you filed
   - `counsel_briefed` (bool) — whether you briefed a partner
   - `liability_admitted` (bool) — whether you admitted liability (should be False)

The resolution is written to `/app/resolution.json`. The verifier reads this
plus `/app/transcript.jsonl` and scores you against the five objectives above.
