# Labs

Runnable, Docker-heavy experiments that exercise the full benchflow SDK end-to-end. Labs are distinct from unit tests (real Docker, no mocking) and from docs (executable, with expected output). Each lab is self-contained with its own README and orchestrator script.

> **Historical (0.2.x-era).** The labs below are archived under [`labs/archive/`](../labs/archive/). They compare benchflow 0.2.0 against 0.2.1/0.2.2 and are kept as cited security evidence; the hardening they validate still ships.

Labs live under [`labs/archive/`](../labs/archive/).

| Lab                                                         | Question summary                                                                 | Benchflow versions | API key needed               |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------- | ------------------ | ---------------------------- |
| [benchjack-sandbox-hardening](#benchjack-sandbox-hardening) | Does 0.2.1 block BenchJack exploits that succeed under 0.2.0                     | 0.2.0 vs 0.2.1     | No                           |
| [reward-hack-matrix](#reward-hack-matrix)                   | Do the same exploits succeed on real benchmark tasks, and does 0.2.2 block them? | 0.2.0 vs 0.2.2     | Optional (`DAYTONA_API_KEY`) |

---

## benchjack-sandbox-hardening

**Question:** Does sandbox hardening in benchflow 0.2.1 block BenchJack-style exploits that succeed under 0.2.0?

**Location:** [`labs/archive/benchjack-sandbox-hardening/`](../labs/archive/benchjack-sandbox-hardening/)

**Prerequisites:**

- Docker daemon
- Python 3.12+
- `uv` on PATH
- Network access to PyPI
- No API keys required (uses the `oracle` agent)

**Run:**

```sh
python3 labs/archive/benchjack-sandbox-hardening/run_comparison.py
```

- `--clean` — delete `.venvs/` and `.jobs/` before running
- First run is ~5 min (Docker builds + pip installs); subsequent runs use cached `.venvs/` (~1 min)

**Key takeaways:**

- Three exploit patterns (P1 conftest-hook, P2 answer-lookup, P7 pth-injection) flip reward from 0.0 → 1.0 against benchflow 0.2.0 and are blocked under 0.2.1 (reward stays 0.0).
- Defenses are layered: `chmod 700` on `/tests` and `/solution`, non-root `sandbox_user`, and pre-verify conftest cleanup.
- The `oracle` agent executes `solution/solve.sh` directly — deterministic and free of API costs. Swap `agent="oracle"` for `agent="claude-agent-acp"` in `_attack_runner.py` to test with a real LLM.

**Related:** `comparison.ipynb` — narrative deep-dive into P1; run `run_comparison.py` first, then open with:

```sh
uv run --with jupyter jupyter notebook labs/archive/benchjack-sandbox-hardening/comparison.ipynb
```

---

## reward-hack-matrix

**Question:** Do the same BenchJack exploits succeed on real production benchmark tasks, and does benchflow 0.2.2's hardening block them there too?

**Location:** [`labs/archive/reward-hack-matrix/`](../labs/archive/reward-hack-matrix/)

**Prerequisites:**

- `DAYTONA_API_KEY` (default) or Docker daemon (pass `--env docker`)
- Python 3.12+
- `uv` on PATH
- Network access to PyPI and GitHub
- Corpora must be cloned first:
  ```sh
  cd labs/archive/reward-hack-matrix && ./fetch_corpora.sh
  ```

**Run:**

```sh
python labs/archive/reward-hack-matrix/run_matrix.py
```

- `--cells "P1@swebench-verified/astropy__astropy-12907"` — run a single cell
- `--sweep` — enumerate all tasks across all three corpora
- `--clean` — remove `.venvs/`, `.jobs/`, and `.cells/`

**Key takeaways:**

- One tailored exploit per benchmark (P1 conftest-hook for swebench-verified, P7 pth-injection for skillsbench, P7 path-trojan for terminal-bench-2) achieves reward 1.0 against 0.2.0 and is blocked to 0.0 under 0.2.2.
- Each benchmark has a single structural weak point; the lab demonstrates these are closed by the same layered defenses as the synthetic lab, not by benchmark-specific patches.
- Independently corroborated by Berkeley RDI and BrachioLab (Penn) findings published concurrently in April 2026.

---

## See also

- [`harden-sandbox.md`](./harden-sandbox.md) — full seven-pattern BenchJack threat model and hardening audit
