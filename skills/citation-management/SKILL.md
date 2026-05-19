---
name: citation-management
description: Verify academic citations, detect hallucinated BibTeX entries, repair DOI metadata, and produce normalized bibliography outputs without inventing sources.
metadata:
  source: SkillsBench citation-check
  skill-author: BenchFlow
---

# Citation Management

Use this skill when a task asks you to validate, repair, deduplicate, or format
academic citations. Treat citation work as evidence work: never invent metadata,
venues, DOIs, authors, or publication years.

## Workflow

1. Parse every citation entry first. Record the key, title, authors, venue, year,
   DOI/URL/PMID/arXiv ID, and any suspicious fields.
2. Verify identifiers before prose. Prefer DOI resolution, Crossref, PubMed,
   arXiv, ACL Anthology, DBLP, publisher pages, or other primary metadata
   sources over search-result snippets.
3. Check whether title, authors, venue, year, and identifier all refer to the
   same work. A real DOI attached to the wrong title is still a bad citation.
4. Flag hallucinated or fake citations when identifiers do not resolve, the
   claimed venue appears nonexistent, or no credible source supports the title.
5. Keep uncertain cases separate from confirmed fake cases unless the requested
   schema has no uncertainty bucket.
6. When producing BibTeX, normalize braces and escaping, preserve meaningful
   capitalization, and include only fields supported by evidence.
7. Return exactly the requested output format. If JSON is requested, return
   valid JSON only.

## Evidence Standards

- A valid DOI is strong evidence only if it resolves to the same title and
  authors in the citation.
- Missing DOI alone does not make a citation fake; books, older papers, and
  proceedings may lack DOIs.
- Generic titles, generic author names, unverified venues, impossible DOI
  prefixes, and absent source coverage are warning signs, not final proof by
  themselves.
- If the task provides a source-of-truth excerpt, use it as evidence and state
  any mismatch against it.
