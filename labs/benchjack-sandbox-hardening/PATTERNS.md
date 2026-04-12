# BenchJack vulnerability patterns

Canonical registry for the `labs/benchjack-sandbox-hardening/` demo.

Sourced from: https://moogician.github.io/blog/2026/trustworthy-benchmarks-cont/
(7 recurring patterns across 8 major benchmarks)

| id | name | description | 0.2.1 blocks? | defense layer | status |
|---|---|---|---|---|---|
| P1 | conftest-hook | Plant `conftest.py` hook that rewrites all test bodies to no-ops | yes | path lockdown (`/tests` chmod 700) + pre-verify cleanup | shipped |
| P2 | answer-lookup | Read locked answer key from `/solution/` | yes | path lockdown (`/solution` chmod 700) | shipped |
| P3 | eval-injection | Agent output passed to `eval()` in verifier | no — verifier code quality | n/a | out of scope |
| P4 | llm-judge-injection | Prompt injection into LLM judge via unsanitized agent output | no — API boundary | n/a | out of scope |
| P5 | weak-string-match | Substring match lets wrong answers pass | no — verifier logic | n/a | out of scope |
| P6 | trivial-verifier | Verifier always returns 1.0, never checks actual output | no — design | n/a | out of scope |
| P7 | pth-injection | Plant `.pth` file that hijacks Python imports before verifier runs | yes | `CLEANUP_CMD` removes writable `.pth` files | shipped |

## Notes

**P3–P6** are benchflow-agnostic verifier code quality issues. They produce "both
versions fail" in a 0.2.0 vs 0.2.1 comparison and belong in a future
`benchjack-scan/` auditor, not here.

**P7 name history:** Originally called "path-trojan" and described as injecting a
fake `pytest` binary via PATH manipulation. That approach is broken because Harbor
invokes verifiers as `bash -c` (non-login, non-interactive) — shell startup files
(`/etc/profile.d/`, `.bashrc`, `/etc/environment`) are never sourced, so PATH
changes made by the agent have no effect on the verifier process. The correct
mechanism is Python `.pth` file injection: a `.pth` file in a writable
`sys.path` entry executes at Python startup, before any tests run. `CLEANUP_CMD`
in 0.2.1 explicitly removes `.pth` files from writable `sys.path` entries before
the verifier starts.

## Defense layer reference

| defense | what it does | patterns covered |
|---|---|---|
| path lockdown | `chown root:root && chmod 700` on `/tests` and `/solution` before agent/oracle runs | P1, P2 |
| oracle sandbox_user | oracle runs `solve.sh` as `sandbox_user` (non-root), so locked paths deny access | P1, P2 |
| pre-verify cleanup | removes stray `conftest.py` files outside `/tests`, removes writable `.pth` files | P1, P7 |
| verifier env reset | resets `PATH`, `PYTHONSAFEPATH` before verifier subprocess | P7 (belt-and-suspenders) |
