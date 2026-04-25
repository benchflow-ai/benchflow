# Sandbox hardening

## Why

Frontier agents will reward-hack any verifier they can. [BenchJack](https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/) (Berkeley RDI, April 2026) showed that ~100% of tasks across eight major AI agent benchmarks are solvable by exploiting the *verifier's runtime* rather than the agent's intelligence: pytest `conftest.py` hooks, planted PATH binaries, leaked answer keys, `.pth` injection, `sitecustomize.py` shadowing. [Meerkat](https://debugml.github.io/cheating-agents/) (Penn, same week) confirmed it from the trace side — frontier models discover and use these shortcuts in the wild.

benchflow's verifier hardening blocks these patterns by default. Tasks opt out individually when their legitimate test setup requires a hook the cleanup would otherwise delete (e.g. qutebrowser's `conftest.py` resolves a real circular import).

## What gets hardened

Between the agent's last action and the verifier's first command, `harden_before_verify()` (in `src/benchflow/_sandbox.py`) runs this sequence:

1. **Kill sandbox-user processes** — `pkill -u <sandbox_user>`, twice with a sleep, so the agent can't keep writing during teardown.
2. **Wipe `/logs/verifier`** — fresh, root-owned, mode 0777 (so a non-root verifier can write its own results).
3. **Restore build-config files** from a pre-agent snapshot — `setup.py`, `pyproject.toml`, `setup.cfg`, `tox.ini`, `noxfile.py`, `hatch.toml`, `flit.ini`, `MANIFEST.in`, `requirements.txt`, `requirements-dev.txt`, `Makefile`. Closes the editable-install hijack vector.
4. **Optionally restore the entire workspace** from `/testbed_verify` snapshot — opt-in per task because some tasks score workspace edits.
5. **Purge symlinks whose realpath escapes the workspace** — agent can't redirect to `/tmp/evil.py`.
6. **Purge `__pycache__` trees** that didn't exist in the pre-agent baseline.
7. **chown workspace to root** — belt-and-suspenders against zombie sandbox-user processes.
8. **`CLEANUP_CMD`** — delete `conftest.py` outside `/tests/`, `*.py` from `/tmp`/`/var/tmp`, `sitecustomize.py`/`usercustomize.py`/`*.pth` from writable `sys.path` entries.
9. **Set `VERIFIER_ENV`** — pinned `PATH`, `PYTEST_ADDOPTS=-c /dev/null --confcutdir=/tests --rootdir=/app -p no:cacheprovider`, `PYTHONPATH=""`, `PYTHONDONTWRITEBYTECODE=1`, plugin auto-discovery off.
10. **Discover allowed pytest plugins** — `importlib.metadata` scan inside the container, then merge with task-declared `pytest_plugins` from `task.toml`. Anything not in the allow-list is blocked.

The verifier then runs against this hardened workspace.

## Per-task opt-outs

Tasks declare opt-outs in `task.toml`:

```toml
[verifier.hardening]
cleanup_conftests = false
```

| Flag | Default | Effect when `false` |
|------|---------|---------------------|
| `cleanup_conftests` | `true` | Don't delete `conftest.py` outside `/tests/` before verify |

Other cleanup steps (`sitecustomize.py`, `.pth`, `/tmp` `*.py`) always run — they have no legitimate use in a test artifact and disabling them would broaden the attack surface beyond what real tasks need.

Unknown keys in `[verifier.hardening]` are warned and ignored. String values for boolean flags are rejected.

See [`progressive-disclosure.md`](./progressive-disclosure.md#per-task-hardening-opt-outs) for the qutebrowser case study (legitimate `conftest.py` for circular-import fix).

## Validation: the labs

Two runnable research artifacts demonstrate the hardening's effectiveness:

### `labs/benchjack-sandbox-hardening/`

End-to-end demo that benchflow 0.2.1+ blocks three BenchJack-style exploits that work against 0.2.0. Each exploit flips the reward from 0.0 to 1.0 under 0.2.0 — under hardened benchflow, the same exploits fail and reward stays 0.0.

Patterns covered:
- `pattern1_conftest_hook/` — agent plants `conftest.py` with a pytest_runtest_makereport hook that overwrites the report
- `pattern2_answer_lookup/` — agent reads `/solution/` directly
- `pattern7_pth_injection/` — agent installs a `.pth` file that imports a sitecustomize-style shadow

One-command repro: `cd labs/benchjack-sandbox-hardening && ./run_comparison.py`. Output is `comparison.ipynb`-style diff: 0.2.0 reward=1.0 / hardened reward=0.0 per pattern.

### `labs/reward-hack-matrix/`

Full reward-hack sweep across real benchmark tasks comparing 0.2.0 vs 0.2.2. Scope: 8 major benchmarks, ~100 tasks, multiple exploit families. Output: a sweep JSON (`sweep_0.2.0_vs_0.2.2.json`) showing the hardening rate by exploit class.

Run with `cd labs/reward-hack-matrix && python run_matrix.py`. The lab README documents the methodology and per-task results.

## Threat model and known gaps

Benchflow's hardening assumes:
- The Docker image and pre-installed packages are trusted (they're built before the agent runs).
- The agent runs as a non-root sandbox user (`sandbox_user`) — when set to `None`, hardening still runs but the agent can defeat lockdown by being root. Don't run untrusted agents as root.
- The verifier itself is trusted code shipped with the task; benchflow only hardens *what the agent can do to the verifier's environment*.

Known residual risk:
- An agent with sustained access can poison `__pycache__` for files that exist in the baseline (those caches aren't deleted because some tasks diff workspace against `/testbed_verify`). Mitigated by the workspace chown but not eliminated.
- Tasks that don't ship a build-config snapshot can still be hijacked via `setup.py` edits. Snapshot is automatic for declared filenames — task authors don't need to opt in.

## Related

- [`labs/benchjack-sandbox-hardening/README.md`](../labs/benchjack-sandbox-hardening/README.md) — full BenchJack pattern catalog and repro instructions.
- [`labs/reward-hack-matrix/README.md`](../labs/reward-hack-matrix/README.md) — methodology, exploit taxonomy, sweep results.
- [`progressive-disclosure.md`](./progressive-disclosure.md) — soft-verify (the relaxed hardening used between rounds in multi-round trials).
- [`task-authoring.md`](./task-authoring.md) — the `task.toml` schema including `[verifier.hardening]` opt-outs.
