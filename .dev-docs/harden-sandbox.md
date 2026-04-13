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
| 7 | Trusting the output of untrusted code | Test results and artifacts produced inside the agent's environment are treated as reliable. | **Addressed** — [Verifier hardening](#verifier-hardening) (Tiers 1–3: reward-file wipe, build-config snapshot, dedicated verifier user) |

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

1. **Forged verifier output** — agent could pre-write `reward.txt` (directly
   during the agent phase, or via `setup.py` executing during `pip install -e .`
   in `test.sh`) and lock it `chmod 444` so the verifier reads a forged value.
2. **Build-config hijack** — agent modifies `setup.py`, `pyproject.toml`, or
   other build-backend files; `pip install` in `test.sh` then runs agent code
   as the verifier user, enabling PATH shims, pytest entry-point plugins,
   `sitecustomize.py` plants in system site-packages, and `conftest.py` plants
   in `/tests` after Harbor uploads the canonical copy.
3. **Forged scraped trajectory** — `_scrape_agent_trajectory()` reads from
   the agent's own home directory. Tool-call counts derived from it were
   trusted.
4. **conftest.py injection** — agent drops `/conftest.py` with a
   `pytest_runtest_makereport` hook that forces every test to pass.
5. **PATH / environment poisoning** — agent shadows verifier tools or injects
   `sitecustomize.py` / `.pth` files into `sys.path`.
6. **Pytest ini-file injection via agent-writable cwd** — extends #4. Tasks
   `cd /app` before invoking pytest; pytest walks up from cwd looking for
   `pyproject.toml`, `pytest.ini`, `tox.ini`, `setup.cfg`. An agent-written
   `/app/pyproject.toml` with `addopts = "-p evil"` plus `/app/evil.py`
   (importable because `''` is on `sys.path`) loads attacker code. `--rootdir`
   controls test discovery, not ini-file discovery.

### Fix

`harden_before_verify` runs before every verifier invocation. It applies four
layers in sequence:

**Tier 1 — wipe `/logs/verifier/`.**  
`rm -rf /logs/verifier && mkdir -p /logs/verifier && chmod 777 /logs/verifier`
runs as root, severing hardlinks, removing symlink replacements, and
eliminating any pre-staged reward files — including `chmod 444` ones. Followed
by a second `pkill` pass to catch agent processes that slipped through the
initial SIGTERM/SIGKILL window.

**Tier 2 — build-config snapshot and restore.**  
`_snapshot_build_config` (called before the agent launches) captures
`setup.py`, `pyproject.toml`, `setup.cfg`, `tox.ini`, `noxfile.py`,
`hatch.toml`, `flit.ini`, `MANIFEST.in`, `requirements.txt`,
`requirements-dev.txt`, and `Makefile` into a `chmod 700` root-only
directory. `requirements.txt` / `requirements-dev.txt` prevent agent-injected
deps from loading via `pip install -r`; `Makefile` prevents a hollowed-out
`make test` target. Presence/absence is tracked in a JSON manifest rather than a
sentinel string, so the agent cannot forge "this file was absent" by embedding
a magic string in `setup.py`. `_restore_build_config` (called in
`harden_before_verify`) runs `rm -f {dst} && cp …` for each present file —
the `rm -f` severs any symlink the agent planted at the destination before `cp`
runs, preventing a redirect to an arbitrary target (e.g. `/logs/verifier/reward.txt`).
Files absent pre-agent are simply removed. `_refresh_verifier_workspace` syncs
the restored files into `/testbed_verify` (see Tier 3) with the same pattern.

**Tier 3 — dedicated verifier OS user.**
`_setup_verifier_user` (called after `setup_sandbox_user`, before agent launch)
creates a system group and user `verifier` (no login shell, no supplementary
groups), wipes any pre-staged `/home/verifier`, locks the `/logs/` parent
against rename (`chmod 755`), owns `/logs/verifier/` to `verifier:verifier`,
and seeds a root-owned read-only copy of the workspace at `/testbed_verify`.
The group is created with `groupadd -r` before `useradd --gid verifier` —
`useradd --gid` exits 6 if the group doesn't pre-exist, silently breaking
all Tier 3 tasks. `harden_before_verify` sets
`task.config.verifier.user = "verifier"` unless the task opts out via
`[verifier] user = "root"` in `task.toml` (emits a WARNING). Harbor translates
this to `docker compose exec -u verifier`.

Running as `verifier` instead of root means `pip install -e .` in `test.sh`
cannot write to system site-packages (`/usr/local/lib/python3.12/dist-packages`)
or `/root/.local`, structurally closing the entry-point plugin and
`sitecustomize.py` vectors.

**Tier 4 — workspace freeze.**  
After build-config restore and before `CLEANUP_CMD`, `harden_before_verify`
runs (as root):

```shell
chown -R root:root {workspace} && chmod -R a-w {workspace}
```

Closes the **editable-install source modification** vector: with `pip install -e .`,
imports resolve to live source files (e.g. `/testbed/src/pkg/utils.py`), which the
agent can rewrite during its phase. Freezing the workspace read-only before the
verifier runs ensures canonical source is executed. Runs after Tier 2 restore
so canonical files are written before being locked. Skipped when `workspace=None`.

**Env hardening — `VERIFIER_ENV`.**  
Canonical env applied to every verifier invocation. Task-level env from
`task.toml` is merged in; security-critical keys are then re-pinned so task
env cannot override them.

| Var | Value | Purpose |
|---|---|---|
| `PATH` | `/usr/local/sbin:…:/bin` | Locked PATH |
| `PYTEST_ADDOPTS` | `-c /dev/null --confcutdir=/tests --rootdir=/tests -p no:cacheprovider` | `-c /dev/null` blocks ini-file walk-up; `--confcutdir=/tests` blocks conftest walk-up |
| `PYTEST_DISABLE_PLUGIN_AUTOLOAD` | `1` | Blocks all `pytest11` entry-point plugins (see below) |
| `PYTHONSAFEPATH` | `1` | Drops implicit `''` (cwd) from `sys.path` |
| `PYTHONPATH` | `""` | Blocks env-var path injection |
| `PYTHONSTARTUP`, `LD_PRELOAD`, `LD_LIBRARY_PATH` | `""` | Clears image-`ENV` carryover |
| `PYTHONDONTWRITEBYTECODE` | `1` | No `.pyc` artifacts |
| `PYTHONNOUSERSITE` | `1` | Prevents pip writing to user site-packages |
| `PIP_USER` | `0` | Same — pip flag |
| `PIP_NO_USER_CONFIG` | `1` | Blocks pre-staged `pip.conf` in verifier home |
| `HOME` | `/nonexistent` | Prevents pip reading `pip.conf` even if home wipe failed |

**`PYTEST_DISABLE_PLUGIN_AUTOLOAD` and the per-task plugin allowlist.**  
`-c /dev/null` blocks ini-file discovery but not `importlib.metadata`-based
entry-point loading — an agent can modify a pre-installed plugin source
(`mypkg/plugin.py`) without touching `pyproject.toml` and have it execute.
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is the only flag that blocks this path.

Tasks that need specific plugins declare them in `task.toml`:

```toml
[verifier]
pytest_plugins = ["pytest-json-ctrf"]
```

`harden_before_verify` appends `-p <name>` to `PYTEST_ADDOPTS` for each
entry and re-pins the key to `"1"` after the task-env merge.

**`PYTEST_ADDOPTS` re-pin.**  
`PYTEST_ADDOPTS` is always rebuilt from `VERIFIER_ENV["PYTEST_ADDOPTS"]` after
the task-env merge — never inherited from the merged result. Without this, a
task that sets `PYTEST_ADDOPTS` in `[verifier] env` would strip `-c /dev/null`
and `--confcutdir=/tests`, re-enabling ini-file and conftest walk-up. Plugins
are appended to the hardened base, not to any task-supplied value.

**Intentionally omitted:**
- `PYTHONHOME=""` — empty prefix aborts `Py_Initialize`; omitted, not set to
  empty.

**`CLEANUP_CMD`** — defense-in-depth (runs as root):
`find / -maxdepth 10 -name conftest.py -not -path '/tests/*' -delete` plus
`python3 -c "import sys..."` to remove writable `sitecustomize.py`,
`usercustomize.py`, and `.pth` files (stdlib paths under `/usr/lib` and
`/usr/local/lib` are preserved). Depth raised from 5 → 10 to catch conftest
files deep in nested workspace layouts; root user ensures traversal is not
blocked by agent-altered directory modes.

**Untrusted trajectory accounting** — `RunResult` gained
`trajectory_source: str | None` (`acp` / `scraped` / `partial_acp` / `None`)
and a proper `partial_trajectory: bool` field. `n_tool_calls` is only
overwritten from the ACP-tracked count (`session.tool_calls`); scraped paths
emit a WARNING with the `UNTRUSTED` label.

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

**`/logs/verifier` lockdown** — owned `verifier:verifier 700` at setup time
(Tier 3) so `sandbox_user` cannot pre-write reward files; wiped and recreated
`777` by `harden_before_verify` (Tier 1) immediately before the verifier runs.

**Idempotent process close** — `LiveProcess.close()` checks `returncode`
before `terminate()` and catches `OSError` on stdin close, so the finally
block does not crash after `pkill` has already killed the ACP transport.

---

## SDK flow after hardening

```
1.  _start_env_and_upload      — start container, upload task files
2.  pre_agent_hooks            — caller-provided hooks
3.  _install_agent             — install agent binary (as root)
4.  _write_credential_files    — write API keys
5.  _setup_sandbox_user        — create agent user, chown workspace + home
5b. _setup_verifier_user       — create verifier user, own /logs/verifier/, seed /testbed_verify
6.  _snapshot_build_config     — snapshot setup.py / pyproject.toml / requirements.txt / Makefile / etc. (root-only, pre-agent)
7.  _deploy_skills             — copy skills to agent paths
8.  _lockdown_paths            — chown root + chmod 700 on locked paths
9.  _connect_acp               — setpriv/su privilege drop, ACP handshake
10. _execute_prompts           — multi-turn agent execution
11. _harden_before_verify      — pkill agent, wipe /logs/verifier/, restore build-config, freeze workspace, env reset, set verifier user
12. _verify                    — run verifier as 'verifier' user, isolated errors
```

**Oracle branch parity.** When `agent == "oracle"`, `SDK.run()` previously
skipped steps 5, 5b, and 6. The oracle path now detects `agent_cwd` via `pwd`
(tasks vary in WORKDIR: `/testbed` vs `/app`), then calls
`setup_sandbox_user(workspace=agent_cwd)`, `_snapshot_build_config(workspace=agent_cwd)`,
and `_setup_verifier_user`. Previously, oracle runs had no verifier OS user, no
build-config snapshot, and `_verify` received `workspace=None` — silently
skipping Tier 2 restore and Tier 4 freeze.

## Threat model coverage

| Threat | Mitigation |
|--------|------------|
| Verifier crash misclassified as agent error | `verifier_error` field, separate bucket, terminal retry |
| Verifier hangs and inflates duration | Explicit `asyncio.wait_for(verifier.timeout_sec)` |
| Systemic verifier bug masquerades as agent failure | `>20%` ERROR threshold, separate aggregation |
| Agent forges `reward.txt` during agent phase | `/logs/verifier/` wiped (`rm -rf` + recreate) in `harden_before_verify` |
| `setup.py` re-creates `reward.txt` via `pip install` | Build-config snapshot restores canonical `setup.py` before verify |
| PEP 517 build-backend hijack via `pyproject.toml` | Build-config snapshot restores canonical `pyproject.toml` |
| `uv` PATH shim written by `setup.py` | Build-config snapshot |
| pytest entry-point plugin via `pip install` as root (new install) | Verifier runs as `verifier` user; cannot write system site-packages |
| pytest entry-point plugin via pre-installed dist-info (canonical `pytest11`) | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; per-task allowlist for legitimate plugins |
| Symlink at restore dst redirects `cp` write to `reward.txt` | `rm -f {dst}` severs any symlink before `cp` in `_restore_build_config` |
| `sitecustomize.py` / `.pth` in `/usr/local/lib` | Verifier runs as `verifier` user; cannot write system site-packages |
| `conftest.py` plant in `/tests` after Harbor upload | Build-config snapshot (canonical `setup.py` cannot plant it) |
| Agent forges `reward.txt` via background process | `pkill -u sandbox_user` before verify |
| Agent forges scraped tool-call count | `n_tool_calls` only sourced from ACP session |
| `conftest.py` injection forces tests to pass | `--confcutdir=/tests` + `find -maxdepth 10 -delete` (as root) |
| `pyproject.toml` / `pytest.ini` injection from agent-writable cwd | `-c /dev/null` in `PYTEST_ADDOPTS` |
| Task strips `-c /dev/null` / `--confcutdir` via `verifier.env.PYTEST_ADDOPTS` | `PYTEST_ADDOPTS` re-pinned from hardened base after task-env merge |
| Module-shadow via `import x` finding `/app/x.py` | `PYTHONSAFEPATH=1` drops implicit `''` from `sys.path` |
| Image-`ENV` carryover (`LD_PRELOAD`, `PYTHONSTARTUP`) | Cleared in `VERIFIER_ENV` |
| `sitecustomize.py` / `.pth` injection in writable paths | sys.path-aware `CLEANUP_CMD` (runs as root) |
| PATH poisoning to shadow verifier tools | Canonical `VERIFIER_ENV` PATH |
| Agent rewrites editable-install source (`/testbed/src/pkg/utils.py`) | Workspace frozen `chown root + chmod a-w` (Tier 4) after build-config restore |
| Agent injects malicious deps via `requirements.txt` rewrite | `requirements.txt` / `requirements-dev.txt` snapshotted and restored (Tier 2) |
| Agent replaces `Makefile` test target with no-op or pass | `Makefile` snapshotted and restored (Tier 2) |
| `conftest.py` nested > 5 dirs deep escaping `find` sweep | `find -maxdepth 10` (raised from 5) |
| Oracle path missing verifier setup (no verifier user / no snapshot) | Oracle branch now calls `pwd`, `_setup_verifier_user`, `_snapshot_build_config` |
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
  agent-writable dirs deeper than `find -maxdepth 10 -name conftest.py`
  reaches still benefit from L1/L2 (which are walk-depth-independent).

## Future directions

Items deferred but on the roadmap. Each entry lists the trigger that
should prompt revisiting it.

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
