# citation-check (network-allowlist variant)

This is the **L2/L3 network-lane vehicle** for the Benchflow integration ladder.
It is a minimal overlay of the base
[`../citation-check/`](https://github.com/benchflow-ai/benchflow/blob/main/docs/examples/task-md/real-skillsbench/citation-check) task whose
behavioural difference is the sandbox network policy: instead of the base task's deprecated
`allow_internet: true` (which reconciles to `network_mode: public`), this variant
declares a hardened **`network_mode: allowlist`** scoped to exactly the hosts the
`citation-management` skill scripts actually reach. The verifier's network is the
same allowlist, so its tools come with the image rather than from the internet at
verify time.

## What it is

- `task.md` — same prompt / metadata as the base task, but its `sandbox:`
  block sets:

  ```yaml
  network_mode: allowlist
  allowed_hosts:
  - eutils.ncbi.nlm.nih.gov   # search_pubmed.py / extract_metadata.py  (NCBI E-utilities)
  - scholar.google.com        # search_google_scholar.py               (scholarly library)
  - doi.org                   # doi_to_bibtex.py / validate_citations.py (DOI resolution)
  - api.crossref.org          # extract_metadata.py / validate_citations.py (CrossRef metadata)
  ```

- `environment/`, `oracle/`, `verifier/` — copies of the base
  `../citation-check/` directories. In both tasks `environment/Dockerfile`
  installs the verifier's pytest and pytest-json-ctrf with the image and
  `verifier/test.sh` only runs them, so the verifier needs no network under
  either task's policy. They are copies, not symlinks, because sandbox uploads
  skip symlinks and `bench check` refuses a task that relies on them.

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

Network is **not** a `bench run` flag — there is no `--network` flag;
`network_mode` is a per-task config field. The ladder does not run this
variant:

- The deterministic planner (`.github/scripts/integration_matrix.py`) emits a
  network-lane `citation-check` cell **labelled** `allowlist`. The label is not
  passed to `bench`, so the cell runs skillsbench's `citation-check` under the
  policy that task declares (`public`).
- The deterministic grader (`.github/scripts/build_integration_review_pack.py`)
  runs **`V-NETWORK`** (`tests/integration/rubric_checks.py:network_hardening`)
  on the policy each trial **ran under**, read from its records
  (`integrity/manifest.json`, else the job's `run-card.json`, else
  `verifier/verifier-network.json`), never on the label. `allowlist` **passes
  only with recorded, non-empty `allowed_hosts`**; a `public` run is flagged
  (and is a hard `fail` on a verifier/sandbox PR).
- Validity is enforced upstream by
  `_validate_network_policy_fields` (`src/benchflow/task/config.py`), which
  rejects `allowlist` with an empty `allowed_hosts`.

**Run directly, the allowlist is enforced.** `bench run` holds the agent and
the verifier to the four hosts, and the trial's `verifier/verifier-network.json`
records the verifier's network and any packets it refused. The oracle needs no
network: on Docker it scores 1 with nothing refused, and V-NETWORK passes on its
records.

## Deferred follow-ups

This variant is the documented placeholder for two **deferred** (documented, not
built) pieces of work, tracked in
[Integration CI → Not built yet](https://github.com/benchflow-ai/benchflow/blob/main/.github/integration/README.md#not-built-yet):

1. A real `--network-mode` CLI passthrough on `bench run`, so the network lane
   runs cells under `no-network` and `allowlist`.
2. The allowed hosts recorded in `run-card.json`, so an allowlist run without
   `integrity/manifest.json` (`--integrity off`) is graded on them.

See [Integration CI → The network lane](https://github.com/benchflow-ai/benchflow/blob/main/.github/integration/README.md#the-network-lane)
and the [review rubric's `V-NETWORK` gate](https://github.com/benchflow-ai/benchflow/blob/main/.github/integration/REVIEW_RUBRIC.md#v-network).
