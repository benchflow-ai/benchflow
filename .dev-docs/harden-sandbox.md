# Sandbox Hardening

benchflow's sandbox was audited against the seven vulnerability patterns
documented in
[Trustworthy AI Agent Benchmarks](https://moogician.github.io/blog/2026/trustworthy-benchmarks-cont/).
This doc summarizes the gaps that were closed and how.

## The seven patterns

| # | Pattern | What it means | Status in benchflow |
|---|---------|---------------|---------------------|
| 1 | No isolation between agent and evaluator | Agent code runs in the same environment the evaluator inspects, so the agent can tamper with evaluation. | **Addressed** — [Default non-root sandbox + path lockdown](#default-non-root-sandbox--path-lockdown) |
| 2 | Answers shipped with the test | Reference answers reachable from the agent (task configs, repos, metadata) turn evaluation into lookup. | **Addressed** — [Default non-root sandbox + path lockdown](#default-non-root-sandbox--path-lockdown) |
| 3 | `eval()` on untrusted input | Evaluator executes agent-controlled strings without sandboxing → arbitrary code execution. | Not present — benchflow's verifier does not `eval` agent output. |
| 4 | LLM judges without input sanitization | Agent output is interpolated into judge prompts, enabling prompt injection to bias scoring. | Not applicable — benchflow has no LLM judge in-tree (benchmark author responsibility if used). |
| 5 | Weak string matching | Overly permissive comparison (substring, aggressive normalization) lets wrong answers pass. | Not applicable — scoring is reward-based, not string-match. |
| 6 | Evaluation logic that doesn't evaluate | Scoring functions skip checks, contain dead code, or fail to compare against ground truth. | **Addressed** — [Verifier failure isolation](#verifier-failure-isolation) |
| 7 | Trusting the output of untrusted code | Test results and artifacts produced inside the agent's environment are treated as reliable. | **Addressed** — [Verifier hardening](#verifier-hardening) |

---

## Verifier failure isolation

### Problem

`SDK.run()`'s broad `except Exception` caught verifier crashes and stuffed
them into the agent `error` field. Three distinct outcomes were conflated:

- Verifier crashed (infrastructure bug — should be investigated)
- Verifier timed out (task too slow — should be retried)
- Agent failed and verifier returned zero (legitimate zero)

Result JSON could not distinguish them, retry/resume logic mis-handled them,
and aggregation hid systemic verifier bugs as "agent errors".

### Fix

A new `verifier_error: str | None` field on `RunResult`, mutually exclusive
with `error`. `_verify()` now owns its own try/except with an explicit
`asyncio.wait_for` timeout and returns `(rewards, verifier_error)`.

- **Classification** — `_scoring.classify_verifier_error()` maps the prefix
  to `verifier_failure`, `verifier_timeout`, or `verifier_other`. Prefixes
  in `_verify()` are kept in sync via comments.
- **Retry** — verifier errors are terminal in `job._run_task()`; broken
  verifiers do not self-heal.
- **Resume** — `_get_completed_tasks()` treats verifier-errored tasks as
  complete (with INFO log) so resume does not loop forever on a systemic bug.
- **Aggregation** — `JobResult.verifier_errored` is a separate bucket from
  `errored`. An assertion enforces
  `passed + failed + errored + verifier_errored == total`. A WARNING fires on
  any verifier error and an ERROR fires above 20%.
- **Metrics** — `TaskMetrics.verifier_error`, `BenchmarkMetrics.verifier_errored`,
  `verifier_error_breakdown`, plus `avg_tool_calls` / `avg_duration` filters
  exclude verifier-errored tasks so they cannot skew averages.
- **Scoring** — `score = passed / total` (conservative, counts verifier
  errors against the agent). `score_excl_errors = passed / (passed + failed)`
  (clean, excludes both error types).

---

## Verifier hardening

### Problems

1. **Forged verifier output** — if the agent ran as root or
   `/logs/verifier` was world-writable, the agent could pre-write
   `reward.txt` and the verifier might read its own forged prior output.
2. **Forged scraped trajectory** — `_scrape_agent_trajectory()` reads from
   the agent's own home directory. Tool-call counts derived from it were
   trusted.
3. **conftest.py injection** — pytest's `rootdir` is the common ancestor of
   collected paths and CWD. An agent that drops `/conftest.py` with a
   `pytest_runtest_makereport` hook can force every test to pass.
4. **PATH / environment poisoning** — agent could shadow verifier tools, or
   inject `sitecustomize.py` / `usercustomize.py` / `.pth` files into
   `sys.path`.
5. **Pytest ini-file injection via agent-writable cwd** — extends #3. Real
   tasks `cd /app` (agent-writable scratch dir) before invoking pytest, and
   pytest walks up from **cwd** looking for `pyproject.toml`, `pytest.ini`,
   `tox.ini`, `setup.cfg`. An agent-written `/app/pyproject.toml` with
   `[tool.pytest.ini_options] addopts = "-p evil"` plus `/app/evil.py`
   (importable because `''` is on `sys.path` for `python -m pytest`) loads
   attacker code as root. `--rootdir=/tests` alone does **not** block this:
   it controls test discovery, not ini-file discovery.

### Fix

A new `SDK._harden_before_verify()` runs before every verifier invocation
and is composed of two class constants plus one method:

- `SDK._VERIFIER_ENV` — canonical env merged into the verifier. Task-level
  env from `task.toml` is merged last so authors can override.

  | Var | Value | Layer / purpose |
  |---|---|---|
  | `PATH` | `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin` | locked PATH (Pattern 7) |
  | `PYTEST_ADDOPTS` | `-c /dev/null --confcutdir=/tests --rootdir=/tests -p no:cacheprovider` | L1 (`-c /dev/null`) blocks `pyproject.toml`/`pytest.ini`/`tox.ini`/`setup.cfg` walk-up; L2 (`--confcutdir=/tests`) blocks `conftest.py` walk-up; rootdir pin + cache disable from original Pattern 7 |
  | `PYTHONSAFEPATH` | `1` | L4 — Python 3.11+ drops implicit `''` (cwd) from `sys.path`, blocking module-shadow via `import helpers` finding `/app/helpers.py` |
  | `PYTHONPATH` | `""` | block env-var path injection (empty list = same as unset, safe) |
  | `PYTHONSTARTUP`, `LD_PRELOAD`, `LD_LIBRARY_PATH` | `""` | L5 — clear image-`ENV` carryover (zero-downside insurance against malicious base images) |
  | `PYTHONDONTWRITEBYTECODE` | `1` | no `.pyc` artifacts left behind |

  **Intentionally omitted**, with reasons documented inline at the
  declaration:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` — would break ~94 SkillsBench tasks
    that rely on `pytest-json-ctrf`'s `--ctrf` flag. Entry-point plugin
    injection is structurally blocked anyway: verifier runs as root, system
    site-packages is root-owned, agent's user-site is not on root's
    `sys.path`, and `_CLEANUP_CMD` wipes `.pth` files in any writable
    `sys.path` entry.
  - `PYTHONNOUSERSITE=1` — root verifier means `/root/.local` is the only
    user-site on `sys.path`, and `sandbox_user` cannot write there.
  - `PYTHONHOME=""` — setting it to empty string is NOT equivalent to
    leaving it unset. CPython reads the empty prefix, fails to find
    `lib/python3.X/encodings`, and aborts during `Py_Initialize` with
    `ModuleNotFoundError: No module named 'encodings'`. This broke any
    verifier `test.sh` that spawned a fresh Python interpreter (seen
    deterministically on swebench astropy__7166/7336/7606/7671 which run
    `python -m pip install -e .[test]` before pytest). The PYTHONHOME
    attack surface is already covered structurally: `sandbox_user` cannot
    set env vars that persist across `docker exec` boundaries, and nothing
    in benchflow base images sets `PYTHONHOME`. Removed in commit against
    `main` — see `test_pythonhome_not_set` for the negative guard.
- `SDK._CLEANUP_CMD` — defense-in-depth shell command:
  `find / -maxdepth 5 -name conftest.py -not -path '/tests/*' -delete`
  plus `python3 -c "import sys..."` to enumerate real `sys.path` and remove
  writable `sitecustomize.py`, `usercustomize.py`, and `.pth` files
  (stdlib copies under `/usr/lib` and `/usr/local/lib` are preserved).
- `pkill -u {sandbox_user}` (SIGTERM, sleep 1, SIGKILL) — kills any agent
  background daemons before verification, so they cannot race the verifier.
  Only runs when `sandbox_user` is set.

**Untrusted trajectory accounting** — `RunResult` gained
`trajectory_source: str | None` (`acp` / `scraped` / `partial_acp` / `None`)
and a proper `partial_trajectory: bool` field. When falling back to scraped
or partial trajectories, `n_tool_calls` is **not** overwritten — only the
ACP-tracked count (`session.tool_calls`) is trusted. A WARNING with the
`UNTRUSTED` label is emitted on the scraped path.

`/logs/verifier` filesystem lockdown is **not** done here — it is a host
bind mount and `chmod 700` would block Harbor's SDK from writing.

---

## Default non-root sandbox + path lockdown

### Problems

- **Agent ran as root by default** — full filesystem access made every
  other mitigation moot.
- **Solution / test files were readable** — the agent could simply `cat
  /solution/answer.py` or `cat /tests/test_*.py` to game the verifier.

### Fix

**Default `sandbox_user="agent"`** in `SDK.run()`, `JobConfig`, and the CLI
(`--sandbox-user`). Passing `sandbox_user=None` falls back to root and
emits a WARNING about cheating risk.

**Privilege drop without gosu.** `SDK._build_priv_drop_cmd()` detects
`setpriv` (util-linux) and uses
`setpriv --reuid=USER --regid=USER --init-groups -- bash -c CMD`, falling
back to `su -l USER -c CMD` on Alpine/BusyBox. No Dockerfile mutation, no
external dependency. `DockerProcess` already wraps in `bash -c`, so there
is no outer `sh -c` wrapper to mis-quote.

**`_setup_sandbox_user`** creates the user, then `chown`s the container's
working directory (detected via `pwd`) and the agent home so the agent can
write where the task expects output. `shlex.quote()` is used on the
attacker-controlled `pwd` output and on skill paths to prevent command
injection.

**Path lockdown** — new `sandbox_locked_paths: list[str] | None` parameter:

- Default when `sandbox_user` is set: `["/solution", "/tests"]`
- Default when `sandbox_user` is `None`: `[]` (no-op + warning)
- Caller-supplied paths are unioned with defaults (deduped, order-preserving)
- `[]` opts out explicitly; non-empty paths without `sandbox_user` raises
  `ValueError`
- Shell globs (`/app-*`) supported; `**` is not

`_validate_locked_path()` rejects shell metacharacters, `..` traversal,
paths that change under `os.path.normpath`, and trailing slashes.

`_lockdown_paths()` runs **after** install/setup/skills but **before**
ACP connect:

```shell
for d in /solution; do
  [ -L "$d" ] && echo "WARN: skipping symlink $d" >&2 && continue;
  [ -e "$d" ] || continue;
  chown root:root "$d" && chmod 700 "$d";
done
```

Security properties:
1. `chown` before `chmod` — agent loses access at `chown`
2. Symlink rejection prevents `ln -s /usr/bin /solution` attacks
3. Non-recursive by design — `chmod 700` removes the traverse bit, so
   contents are unreachable regardless of individual file modes

**`/logs/verifier` is not locked** — bind mount; verifier integrity is
covered by `_harden_before_verify` instead.

**Idempotent process close** — `LiveProcess.close()` checks `returncode`
before `terminate()` and catches `OSError` on stdin close, so the finally
block does not crash after `pkill` has already killed the ACP transport.

---

## SDK flow after hardening

```
1.  _start_env_and_upload   — start container, upload task files
2.  pre_agent_hooks         — caller-provided hooks
3.  _install_agent          — install agent binary (as root)
4.  _write_credential_files — write API keys
5.  _setup_sandbox_user     — create user, chown workspace + home
6.  _deploy_skills          — copy skills to agent paths
7.  _lockdown_paths         — chown root + chmod 700 on locked paths
8.  _connect_acp            — setpriv/su privilege drop, ACP handshake
9.  _execute_prompts        — multi-turn agent execution
10. _harden_before_verify   — pkill agent, cleanup, env reset
11. _verify                 — run verifier as root, isolated errors
```

## Threat model coverage

| Threat | Mitigation |
|--------|------------|
| Verifier crash misclassified as agent error | `verifier_error` field, separate bucket, terminal retry |
| Verifier hangs and inflates duration | Explicit `asyncio.wait_for(verifier.timeout_sec)` |
| Systemic verifier bug masquerades as agent failure | `>20%` ERROR threshold, separate aggregation |
| Agent forges `reward.txt` via background process | `pkill -u sandbox_user` before verify |
| Agent forges scraped tool-call count | `n_tool_calls` only sourced from ACP session |
| `conftest.py` injection forces tests to pass | `--confcutdir=/tests` + `find -delete` |
| `pyproject.toml` / `pytest.ini` injection from agent-writable cwd | `-c /dev/null` in `PYTEST_ADDOPTS` |
| Module-shadow via `import x` finding `/app/x.py` | `PYTHONSAFEPATH=1` drops implicit `''` from `sys.path` |
| Image-`ENV` carryover (`LD_PRELOAD`, `PYTHONSTARTUP`) | Cleared in `_VERIFIER_ENV` |
| `sitecustomize.py` / `.pth` injection | sys.path-aware cleanup |
| PATH poisoning to shadow verifier tools | Canonical `_VERIFIER_ENV` PATH |
| Agent reads `/solution` or `/tests` to cheat | `chown root + chmod 700` lockdown |
| Agent runs as root with full FS access | Default `sandbox_user="agent"` + setpriv/su |
| Symlink attack on locked paths | Symlink guard in `_lockdown_paths` |
| Command injection via `pwd` / skill paths | `shlex.quote()` |

**Out of scope** (documented, not addressed):
- Network-based cheating — handled by `allow_internet=false`
- Agent re-escalating to root — mitigated by setpriv/su capability drop;
  SUID binary stripping is future work
- Service-mediated exfiltration — benchmark author responsibility
- Harbor private-attribute coupling in `process.py` — upstream-blocked
- **Custom task layouts that `cd` into deeply nested agent-writable dirs
  other than `/app`** — `--confcutdir=/tests` blocks `conftest.py` walk-up
  to any depth, and `-c /dev/null` blocks ini-file walk-up entirely, so
  the residual surface is small. Tasks that intentionally operate in
  agent-writable dirs deeper than `find -maxdepth 5 -name conftest.py`
  reaches still benefit from L1/L2 (which are walk-depth-independent).

## Future directions

Items deferred but on the roadmap. Each entry lists the trigger that
should prompt revisiting it.

- **Verifier as a dedicated non-root user, distinct from `sandbox_user`.**
  Today the verifier inherits Harbor's default of running as root, which is
  the load-bearing assumption behind several layers (e.g. "agent's user-site
  isn't on the verifier's `sys.path`" only holds because the verifier's user
  is root, not the agent). Inverting this — verifier as its own UID, neither
  root nor `sandbox_user` — would make most of the layered defenses
  *redundant in a good way*: the agent-writable cwd, `/root/.local`, and
  PATH-shadowing vectors all become structurally unreachable instead of
  defended in depth. **Trigger:** Harbor exposes a `verifier.user` setting,
  or benchflow adds its own privilege-drop wrapper around the verifier
  invocation (mirroring `_setup_sandbox_user`).

- **Per-task pytest plugin allowlist.** Would let us safely enable
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` (Layer 3 from the original plan,
  currently dropped). Today ~94 SkillsBench tasks rely on
  `pytest-json-ctrf`'s `--ctrf` flag, so global autoload-disable would
  break them. A per-task `pytest_plugins = [...]` declaration in
  `test.sh` (or `task.toml`) would let those tasks opt back in
  explicitly. **Trigger:** a real entry-point plugin injection attack is
  found, OR a one-time refactor pass over the SkillsBench tasks becomes
  worthwhile (e.g. as part of a broader test-infra cleanup).

- **`pytest --import-mode=importlib`.** Eliminates pytest's default
  `prepend` import mode, which mutates `sys.path` to add each test file's
  rootdir. `PYTHONSAFEPATH=1` (Layer 4) handles the *Python interpreter*'s
  cwd-on-`sys.path` behavior, but pytest's own `--import-mode=prepend`
  injection is independent. The audit confirmed no current task relies on
  cwd-on-`sys.path` for sibling imports, so a migration is feasible but has
  non-trivial breakage risk. **Trigger:** a `sys.path`-based bypass is
  discovered that PYTHONSAFEPATH alone doesn't cover.

- **End-to-end pytest-injection smoke test in CI.** The unit-level
  subprocess test in `tests/test_verify.py` invokes `pytest -c /dev/null`
  directly to bind the static `_VERIFIER_ENV` assertions to real pytest
  behavior. A complementary CI job would build a minimal task fixture with
  a hostile `pyproject.toml`/`conftest.py`/`*.pth` baked in, run the full
  benchflow verifier path against it, and assert the verifier rejects all
  three. Currently this is a manual smoke test step (see commit message of
  this hardening round). **Trigger:** any new layer added to `_VERIFIER_ENV`,
  or any change to `_harden_before_verify` / `_CLEANUP_CMD`.

- **SUID binary stripping in agent base images.** Closes the agent →
  root re-escalation vector that `setpriv` / `su` privilege-drop alone
  does not fully cover. Already in Out-of-scope as "future work"; would
  pair naturally with the verifier-user change above. **Trigger:** an
  audit finds a SUID binary the agent can exploit, OR the verifier-user
  change lands and we want to close the remaining horizontal-escalation
  path symmetrically.
