# Sandbox hardening

> Part of the **Environment plane** — see [Concepts: the three planes](./concepts.md#the-three-planes).

The Environment plane answers *how is the world built, reset, and secured?* Sandbox hardening is the **secured** half: the anti-reward-hacking security model that runs between the agent and the verifier. (For *built* and *reset* — image, network policy, service catalog — see the [Environment plane](./environment-plane.md).)

## Why

Frontier agents will reward-hack any verifier they can. [BenchJack](https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/) (Berkeley RDI, April 2026) showed that ~100% of tasks across eight major AI agent benchmarks are solvable by exploiting the *verifier's runtime* rather than the agent's intelligence: pytest `conftest.py` hooks, planted PATH binaries, leaked answer keys, `.pth` injection, `sitecustomize.py` shadowing. [Meerkat](https://debugml.github.io/cheating-agents/) (Penn, same week) confirmed it from the trace side — frontier models discover and use these shortcuts in the wild.

The core insight: the agent and the verifier share a filesystem. Anything the agent can write before it stops — a `conftest.py`, a `.pth` file, a rewritten `setup.py`, a planted `__pycache__` — is still on disk when the verifier starts. BenchFlow's job is to *neutralize the agent's writes that target the verifier's runtime* without disturbing the legitimate answer the agent produced.

BenchFlow's verifier hardening blocks these patterns by default. Tasks opt out individually when their legitimate test setup requires a hook the cleanup would otherwise delete (e.g. qutebrowser's `conftest.py` resolves a real circular import).

## The trust boundary

Hardening only makes sense relative to what BenchFlow trusts. The model is:

- **Trusted** — the Docker image and its pre-installed packages (built before the agent ever runs), and the verifier itself (`verifier/test.sh` or `tests/test.sh`, shipped with the task). Hardening does **not** defend against a malicious image or a buggy verifier; it defends the verifier's runtime *against the agent*.
- **Untrusted** — everything the agent did. The agent runs as a non-root `sandbox_user`, so the kernel's VFS layer already denies writes to root-owned paths (system site-packages, `/tests`, `/solution`). Hardening then erases the writes the agent *was* allowed to make in its own workspace.

Because the agent is non-root, the single most important precondition for every step below is **the agent is not root**. When `sandbox_user` is `None` the hardening sequence still runs, but a root agent can undo most of it — so do not run untrusted agents as root.

## What gets hardened

Between the agent's last action and the verifier's first command, `harden_before_verify()` (in [`src/benchflow/sandbox/lockdown.py`](https://github.com/benchflow-ai/benchflow/blob/main/src/benchflow/sandbox/lockdown.py)) runs this sequence against the **agent (`main`) container only**:

1. **Kill sandbox-user processes** — `pkill -u <sandbox_user>`, then `sleep 1` and a `pkill -9`, then a second pass that catches stragglers (e.g. `cron`/`at` jobs). This stops the agent from writing concurrently *during* teardown.
2. **Wipe `/logs/verifier/`** — recreated fresh, root-owned, mode `0777` (so a non-root verifier can write its own results), while preserving any remote bind mounts. The parent `/logs` is root-owned `0755` so the sandbox user can't rename the directory out from under the verifier.
3. **Reclaim re-downloadable cache space** — best-effort, result-neutral deletion of `uv`/`pip`/`apt` download caches so the verifier's own dependency install doesn't hit `ENOSPC` on disk-constrained sandboxes (e.g. Daytona's 10 GB cap). Never touches the workspace, agent outputs, or installed tools; any per-user cache overlapping the active workspace is skipped in both directions.
4. **Restore build-config files** from a pre-agent snapshot (snapshotted before the agent launched, into root-only `/tmp/.benchflow_build_snapshot`) — `setup.py`, `pyproject.toml`, `setup.cfg`, `tox.ini`, `noxfile.py`, `hatch.toml`, `flit.ini`, `MANIFEST.in`, `requirements.txt`, `requirements-dev.txt`, `Makefile`. Closes the editable-install / install-time-RCE hijack vector (agent rewrites `setup.py` so `pip install -e .` runs its code). A presence/absence manifest — not a sentinel in the file — records which files existed, so the agent can't forge "this file was absent." *Runs only when the workspace is also being restored (see step 5).*
5. **Optionally restore the entire workspace** from the `/testbed_verify` snapshot — **opt-in** per task via `restore_workspace`, because some benchmarks (SkillsBench, terminal-bench-style) legitimately score *workspace edits* and a full restore would erase the answer. When on, it resets every source file (not just build-config) to its pre-agent state, closing the source-modification vector (agent rewrites `src/pkg/x.py` to make a failing test pass).
6. **Purge external symlinks** — any symlink whose `realpath` escapes the workspace is deleted (agent can't redirect `utils.py → /tmp/evil.py`). In-tree symlinks are preserved, since some real source trees ship them as fixtures.
7. **Purge `__pycache__` trees** that didn't exist in the pre-agent baseline, so agent-planted `.pyc` bytecode can't execute even if the cache-prefix guard is bypassed. Baseline-present caches are kept so tasks that diff the workspace against `/testbed_verify` don't break.
8. **chown workspace to root** — belt-and-suspenders against any zombie sandbox-user process that survived the `pkill`.
9. **Cleanup injected Python hooks** (`CLEANUP_CMD`) — delete `conftest.py` outside `/verifier/` and `/tests/`; delete `*.py` from `/tmp` and `/var/tmp`; delete `sitecustomize.py`/`usercustomize.py`/`*.pth` from writable `sys.path` entries (skipping `/usr/lib` and `/usr/local/lib`, which are root-owned and trusted). Honors the per-task `cleanup_conftests` opt-out (below).
10. **Pin `VERIFIER_ENV`** — a trusted env dict overrides any pollution the agent left in the verifier's environment. Highlights: a fixed `PATH` (`/usr/local/sbin:…:/bin`); `PYTHONPATH=""`; `PYTHONDONTWRITEBYTECODE=1`; `PYTHONPYCACHEPREFIX=/nonexistent` (so agent-precompiled `__pycache__` is never read); empty `LD_PRELOAD`/`LD_LIBRARY_PATH`; `PYTHONSTARTUP=""`; `PYTHONBREAKPOINT=0` and empty `COVERAGE_PROCESS_START`/`DJANGO_SETTINGS_MODULE`/`CELERY_CONFIG_MODULE` (all of which would otherwise import an agent-controlled module at interpreter startup); and `PYTEST_ADDOPTS="-c /dev/null --confcutdir=<verifier-dir> --rootdir=<workspace> -p no:cacheprovider"` — `-c /dev/null` blocks `pyproject.toml`/`pytest.ini`/`tox.ini`/`setup.cfg` config discovery, `--confcutdir` blocks the `conftest.py` walk-up, and `--rootdir` is computed dynamically from the task's real workspace (so it works for both `/app` and `/root` conventions). Task-supplied `[verifier].env` is merged in *first*, then the hard security invariants (`PATH`, `PYTHONPATH`, plugin-autoload, the startup-hook vars) are **re-pinned on top** so a task can't weaken them.
11. **Disable + allow-list pytest plugins** — `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` turns off entry-point (`pytest11`) plugin loading entirely, because `-c /dev/null` does *not* block entry-point registration and the agent can edit a pre-installed package's plugin source to forge a reward. Allowed plugins are then re-enabled explicitly with `-p` flags: a container-side `importlib.metadata` scan enumerates `pytest11` entry points from **root-owned** dist-info only, merged with task-declared `[verifier] pytest_plugins` and the known `pytest-ctrf` reporter inferred from `test.sh`. Anything not on that list stays off.

The verifier then runs against this hardened workspace.

## Per-task opt-outs

Defaults are secure (every flag is `true`). A task opts out only when its *legitimate* test setup conflicts with a default cleanup step. Opt-outs live under `[verifier.hardening]` in the task config — either `task.toml` or the frontmatter of `task.md`:

```toml
[verifier.hardening]
cleanup_conftests = false
```

| Flag | Default | Effect when `false` |
|------|---------|---------------------|
| `cleanup_conftests` | `true` | Don't delete `conftest.py` outside `/verifier/` and `/tests/` before verify |

`cleanup_conftests` is the only opt-out. The other cleanup steps (`sitecustomize.py`, `usercustomize.py`, `.pth`, `/tmp` `*.py`) always run — they have no legitimate use in a test artifact, and exposing them as opt-outs would broaden the attack surface beyond what real tasks need. The full workspace restore (step 5) is governed separately by `restore_workspace`, which is *off* by default for the opposite reason: most tasks score the workspace the agent left behind, so restoring it would erase the answer.

Unknown keys under `[verifier.hardening]`, and non-boolean values for a known flag, are logged as a warning and ignored — the secure default stays in force rather than silently mis-configuring the task.

See [`progressive-disclosure.md`](./progressive-disclosure.md#per-task-hardening-opt-outs) for the qutebrowser case study (legitimate `conftest.py` for a circular-import fix).

## Worked example: a planted conftest hook, blocked

This is the canonical BenchJack **P1** exploit and exactly what step 9 neutralizes. The task is an ordinary "make the tests pass" task; the agent — instead of fixing the code — plants a `conftest.py` at the workspace root that rewrites every collected test into a no-op:

```python
# /testbed/conftest.py  — planted by the agent, NOT shipped by the task
def pytest_collection_modifyitems(items):
    for item in items:
        item.obj = lambda *a, **k: None   # every test now trivially "passes"
```

Without hardening, pytest discovers this `conftest.py` during collection (it walks up from the test files to the rootdir), applies the hook, and the verifier reports `reward = 1.0` for code that was never fixed.

With hardening, two independent layers stop it before the verifier's pytest ever starts:

1. **Step 9 deletes the file.** `conftest.py` lives at `/testbed/`, which is neither `/verifier/` nor `/tests/`, so `CLEANUP_CMD` removes it.
2. **Step 10 would have contained it anyway.** Even if a copy survived somewhere up-tree, `PYTEST_ADDOPTS` pins `--confcutdir=<verifier-dir>`, so pytest never walks up past the verifier directory to pick up a root-level `conftest.py`.

The verifier runs against the real, unfixed code and correctly reports `reward = 0.0`. The labs below run this exact comparison end-to-end (`P1 conftest-hook: 1.00 EXPLOITED under 0.2.0 → 0.00 BLOCKED under hardened`).

If — and only if — a task legitimately ships a root-level `conftest.py` (the qutebrowser circular-import case), it sets `cleanup_conftests = false`, which removes layer 1 for that task; layer 2's `--confcutdir` is unaffected because the task's own `conftest.py` sits *inside* the confcutdir boundary.

## Validation: the labs

Two runnable research artifacts demonstrate the hardening's effectiveness:

### `labs/benchjack-sandbox-hardening/`

End-to-end demo that benchflow 0.2.1+ blocks three BenchJack-style exploits that work against 0.2.0. Each exploit flips the reward from 0.0 to 1.0 under 0.2.0 — under hardened benchflow, the same exploits fail and reward stays 0.0.

Patterns covered:
- `pattern1_conftest_hook/` — agent plants `conftest.py` with a `pytest_collection_modifyitems` hook that rewrites every test body to a no-op
- `pattern2_answer_lookup/` — agent reads the locked answer key from `/solution/` directly
- `pattern7_pth_injection/` — agent plants a `.pth` file that runs `atexit.register(lambda: os._exit(0))` at interpreter startup, forcing exit code 0 even when tests fail

One-command repro: `cd labs/benchjack-sandbox-hardening && ./run_comparison.py`. Output is `comparison.ipynb`-style diff: 0.2.0 reward=1.0 / hardened reward=0.0 per pattern.

### `labs/reward-hack-matrix/`

Full reward-hack sweep across **real** benchmark tasks comparing 0.2.0 (no hardening) vs 0.2.2 (hardened). Scope: 666 tasks across three production benchmarks (skillsbench, swebench-verified, terminal-bench-2), one BenchJack-style exploit per benchmark, run end-to-end through `SDK().run()` on Daytona — 1332 trials. Headline result: exploit rate falls from **217/666 (32.6%)** under 0.2.0 to **0 true bypasses** under 0.2.2 (terminal-bench-2 drops from 92.1% to 0%). Output is a per-cell sweep JSON (`sweep_0.2.0_vs_0.2.2.json`).

Run with `cd labs/reward-hack-matrix && ./fetch_corpora.sh && python run_matrix.py --sweep`. The lab README documents the methodology, the exploit taxonomy, and independent concurrent validation from the Berkeley RDI and Penn groups.

## Known gaps

The trust boundary above states what hardening assumes (non-root agent, trusted image, trusted verifier). Within those assumptions, the known residual risks are:

- **Baseline `__pycache__` poisoning.** Step 7 only deletes `__pycache__` trees that are *new* since the baseline; caches present pre-agent are kept so workspace-diff tasks don't break. An agent could in principle poison a baseline-present cache. This is mitigated — not eliminated — by the step-8 chown and by `PYTHONPYCACHEPREFIX=/nonexistent` in step 10, which redirects the verifier's `.pyc` reads away from the workspace cache entirely.
- **In-tree symlinks.** Step 6 only purges symlinks whose target escapes the workspace; an agent can still plant `utils.py → ./evil.py` *inside* the tree. Covered indirectly by the chown and by the source restore when `restore_workspace = true`.
- **Build-config coverage is by filename.** The snapshot/restore (step 4) covers the declared `_BUILD_CONFIG_FILES` list only. A build-config file outside that list is not snapshotted. The snapshot is automatic for declared filenames — task authors don't opt in — but it only runs when `restore_workspace` is on.

## Related

- [`labs/benchjack-sandbox-hardening/README.md`](https://github.com/benchflow-ai/benchflow/tree/main/labs/benchjack-sandbox-hardening) — full BenchJack pattern catalog and repro instructions.
- [`labs/reward-hack-matrix/README.md`](https://github.com/benchflow-ai/benchflow/tree/main/labs/reward-hack-matrix) — methodology, exploit taxonomy, sweep results.
- [`progressive-disclosure.md`](./progressive-disclosure.md) — soft-verify (the relaxed hardening used between rounds in multi-round trials).
- [`task-authoring.md`](./task-authoring.md) — the `task.toml` schema including `[verifier.hardening]` opt-outs.
