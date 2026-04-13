# Labs

Runnable, Docker-heavy experiments that exercise the full benchflow SDK end-to-end. Labs are distinct from unit tests (real Docker, no mocking) and from docs (executable, with expected output). Each lab is self-contained with its own README and orchestrator script.

Labs live under [`labs/`](../labs/). The contract for every lab is defined in [`labs/README.md`](../labs/README.md).

---

## benchjack-sandbox-hardening

**Question:** Does sandbox hardening in benchflow 0.2.1 block BenchJack-style exploits that succeed under 0.2.0?

**Location:** [`labs/benchjack-sandbox-hardening/`](../labs/benchjack-sandbox-hardening/)

**Prerequisites:** Docker daemon, Python 3.10+, `uv` on PATH, network access to PyPI. No API keys required (uses the `oracle` agent).

**Run:**

```sh
python3 labs/benchjack-sandbox-hardening/run_comparison.py
```

Pass `--clean` to delete `.venvs/` and `.jobs/` before running. First run is ~5 min (Docker builds + pip installs); subsequent runs use cached `.venvs/` (~1 min).

**Key takeaways:**

- Three exploit patterns (P1 conftest-hook, P2 answer-lookup, P7 pth-injection) flip reward from 0.0 → 1.0 against benchflow 0.2.0 and are blocked under HEAD (reward stays 0.0).
- Defenses are layered: `chmod 700` on `/tests` and `/solution`, non-root `sandbox_user`, and pre-verify conftest cleanup.
- The `oracle` agent executes `solution/solve.sh` directly — deterministic and free of API costs. Swap `agent="oracle"` for `agent="claude-agent-acp"` in `_attack_runner.py` to test with a real LLM.

**Related:** [`.dev-docs/harden-sandbox.md`](../.dev-docs/harden-sandbox.md) covers the full seven-pattern BenchJack threat model. `comparison.ipynb` is a narrative deep-dive into P1 — run `run_comparison.py` first, then:

```sh
uv run --with jupyter jupyter notebook labs/benchjack-sandbox-hardening/comparison.ipynb
```
