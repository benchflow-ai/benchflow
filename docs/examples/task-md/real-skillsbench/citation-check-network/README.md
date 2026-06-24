# citation-check (network-allowlist variant)

This is the **L2/L3 network-lane vehicle** for the Benchflow integration ladder.
It is a minimal overlay of the base
[`../citation-check/`](../citation-check/) task whose **only** behavioural
difference is the sandbox network policy: instead of the base task's deprecated
`allow_internet: true` (which reconciles to `network_mode: public`), this variant
declares a hardened **`network_mode: allowlist`** scoped to exactly the hosts the
`citation-management` skill scripts actually reach.

## What it is

- `task.md` — same prompt / metadata as the base task, but its `environment:`
  block sets:

  ```yaml
  network_mode: allowlist
  allowed_hosts:
  - eutils.ncbi.nlm.nih.gov   # search_pubmed.py / extract_metadata.py  (NCBI E-utilities)
  - scholar.google.com        # search_google_scholar.py               (scholarly library)
  - doi.org                   # doi_to_bibtex.py / validate_citations.py (DOI resolution)
  - api.crossref.org          # extract_metadata.py / validate_citations.py (CrossRef metadata)
  ```

- `environment/`, `oracle/`, `verifier/` — **symlinks** to the base
  `../citation-check/` directories. Nothing about the environment build, the
  oracle, or the verifier changes; this is a pure network-policy overlay, so the
  supporting bytes are shared (not duplicated).

## Why exactly these four hosts

These are the **real egress targets** of the bundled citation-management skill
(`citation-check/environment/skills/citation-management/scripts/`):

| host | reached by | purpose |
|---|---|---|
| `eutils.ncbi.nlm.nih.gov` | `search_pubmed.py`, `extract_metadata.py` | NCBI E-utilities (`/entrez/eutils/…`) for PubMed search + PMID metadata |
| `scholar.google.com` | `search_google_scholar.py` (`scholarly`) | Google Scholar paper search |
| `doi.org` | `doi_to_bibtex.py`, `validate_citations.py` | DOI resolution + CrossRef content negotiation |
| `api.crossref.org` | `extract_metadata.py`, `validate_citations.py` | CrossRef works metadata (`/works/{doi}`) |

The allowlist is deliberately **minimal**: it covers the four hosts the skill's
primary code paths hit and nothing more. `extract_metadata.py` has a secondary
arXiv path (`export.arxiv.org`) that the canonical citation flow does not require;
it is intentionally **excluded** to keep the surface tight. Add it only if a
future citation flow makes arXiv resolution load-bearing.

## How the ladder uses it (CONTRACT Q3)

Network is **not** a `bench eval run` flag — there is no `--network` flag, and
`network_mode` is **never serialized into a rollout artifact**. It is a per-task
config field only. So this variant is a **STATIC hardening vehicle**, not a live
network-egress test:

- The deterministic planner (`.github/scripts/integration_matrix.py`) emits the
  network-lane **allowlist cell** carrying `network_mode: allowlist` as the
  **EXPECTED** value (derived from this task config) — it is **not** passed to
  `bench`.
- The deterministic grader (`.github/scripts/build_integration_review_pack.py`)
  runs the **`V-NETWORK`** static assertion
  (`tests/integration/rubric_checks.py:network_hardening`) over the declared
  policy: the safe default is `no-network`; `allowlist` **passes only with a
  non-empty `allowed_hosts`**; a bare `public` is flagged (and is a hard `fail`
  on a verifier/sandbox PR).
- Validity is enforced upstream by
  `_validate_network_policy_fields` (`src/benchflow/task/config.py`), which
  rejects `allowlist` with an empty `allowed_hosts`.

**Fail-closed by construction.** Because allowlist enforcement is *parsed but not
runtime-enforced per sandbox*
(`src/benchflow/task/runtime_capabilities.py:_append_network_issue`), actually
launching this variant through a sandbox would raise
`UnsupportedTaskFeatureError`. That is intended: the variant proves the *hardened
declaration* statically, it does not open real egress on the secret-bearing host
runner.

## Deferred follow-ups

This variant is the documented placeholder for two **deferred** (documented, not
built) pieces of work, tracked in `docs/integration-tiers.md` → "Deferred
follow-ups":

1. A real `--network-mode` CLI passthrough on `bench eval run`.
2. `persist_sandbox_info` network-mode serialization into the rollout artifact,
   which would let the lane move from a STATIC declaration check to an observed
   egress-policy check.

See `docs/integration-tiers.md` (network lane, Q3) and
`docs/integration-review-rubric.md` (`V-NETWORK` gate, Reward-hacking checklist).
