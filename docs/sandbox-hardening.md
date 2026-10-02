# Sandbox hardening
## Why

Frontier agents will reward-hack any verifier they can. [BenchJack](https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/) (Berkeley RDI, April 2026) showed that ~100% of tasks across eight major AI agent benchmarks are solvable by exploiting the *verifier's runtime* rather than the agent's intelligence: pytest `conftest.py` hooks, planted PATH binaries, leaked answer keys, `.pth` injection, `sitecustomize.py` shadowing. [Meerkat](https://debugml.github.io/cheating-agents/) (Penn, same week) confirmed it from the trace side — frontier models discover and use these shortcuts in the wild.

benchflow's verifier hardening blocks these patterns by default. Tasks opt out individually when their legitimate test setup requires a hook the cleanup would otherwise delete (e.g. qutebrowser's `conftest.py` resolves a real circular import).

Tasks that declare `[verifier] environment_mode = "separate"` go further: the verifier runs in a fresh sandbox that receives only the frozen workspace, declared artifacts and `/logs/artifacts`, so nothing the agent planted elsewhere exists there. The hardening below still runs in that sandbox. See [Separate verifier sandboxes](./separate-verifier.md). To find out whether an agent tried these paths, not only to block them, run with `--integrity audit` or `strict` (see [Reward integrity](./integrity.md)).

## What gets hardened

Between the agent's last action and the verifier's first command, `harden_before_verify()` (in `src/benchflow/sandbox/lockdown.py`) runs this sequence on the `main` container:

1. **Stop sandbox-user processes and prove it** — `pkill -u <sandbox_user>` (TERM, then KILL), then a `/proc` scan that needs neither procps nor Python must find no live process with the sandbox UID. A surviving writer or an unreadable process entry fails hardening, so the verifier never starts next to a live agent process.
2. **Wipe `/logs/verifier`** — its contents are deleted (a bind mount is kept) and the directory is left mode 0777, so a non-root verifier can write its results. `/app` is created as the legacy `--rootdir` fallback.
3. **Reclaim download caches** — best effort, only re-downloadable uv/pip/apt caches, never through symlinks or into the workspace.
4. **Optionally restore the workspace** — with `restore_workspace=True`, build-config files (`setup.py`, `pyproject.toml`, `setup.cfg`, `tox.ini`, `noxfile.py`, `hatch.toml`, `flit.ini`, `MANIFEST.in`, `requirements.txt`, `requirements-dev.txt`, `Makefile`) are restored from the pre-agent snapshot and the workspace is reset from `/testbed_verify`. This is off by default because many tasks score workspace edits, and no task field turns it on today.
5. **Freeze the workspace** — delete symlinks whose realpath escapes the workspace, delete `__pycache__` trees that the pre-agent baseline did not have, then `chown -R root:root`.
6. **Remove Python hooks** — delete `conftest.py` outside `/tests/` and `/verifier/`, `*.py` under `/tmp` and `/var/tmp`, and `sitecustomize.py`, `usercustomize.py` and `*.pth` from `sys.path` entries outside `/usr/lib` and `/usr/local/lib`.
7. **Build the verifier environment** — start from `VERIFIER_ENV` (plugin autoload off, no bytecode writes, `PYTHONPYCACHEPREFIX=/nonexistent`, no user site-packages, `PYTHONBREAKPOINT=0`, empty coverage/Django/Celery startup hooks, `HOME=/root`), merge the task's `[verifier] env`, then re-pin `PATH`, `PYTHONPATH`, autoload and the startup hooks. `PATH` is the safe system path plus image `PATH` entries that are root-owned directories, not group- or world-writable, and outside `/tmp`, `/var/tmp`, `/logs`, `/testbed`, the workspace and `/home/<sandbox_user>`. `PYTHONPATH` keeps image entries that pass the same checks (the workspace is allowed there), after the plugin guard's directory when the verifier needs it (next section). `PYTEST_ADDOPTS` is `-c /dev/null --confcutdir=<verifier dir> -p no:cacheprovider --rootdir=<workspace> -p <guard> -p <trusted plugins>`, where the verifier dir is `/tests` (or `/verifier` for native verifier dirs).
8. **Move the verifier's uv and pip state out of the agent's reach** — uv and pip caches, uv tool environments and uv-managed Pythons move to a new root-owned directory the plugin guard trusts by path, and uv and pip stop reading configuration the agent could have written; see [The verifier's uv and pip state](#the-verifiers-uv-and-pip-state).
9. **Choose trusted pytest plugins and install the plugin guard** — see the next section.

The verifier then runs against this hardened workspace. Its command (test.sh, a script strategy or a reward-kit runner) starts with `umask 022`, so what it creates is mode 0644 or 0755 on every backend, whatever mask the runtime gives an exec: `docker exec` on Docker's own daemon uses 0022, but on Docker-in-Docker (Docker 29.8.1, runc 1.5.1) it uses 0000, and the uv cache test.sh filled came out group- and world-writable. The separate verifier's unpack runs under the same mask.

## Pytest plugin guard

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` stops entry-point autoloading, but pytest still imports plugins by name from `-p` arguments, `PYTEST_PLUGINS` and a conftest's `pytest_plugins`. The name is looked up on the `sys.path` of whichever Python runs pytest, and that path can include the agent-writable workspace (the working directory, `--rootdir`, a test.sh `PYTHONPATH=/app`). A plugin loaded that way runs inside pytest and can rewrite test results. BenchFlow therefore loads a plugin only when its registration and code are root-owned and outside every agent-writable tree, and checks this twice.

**Discovery, before the verifier.** From `/`, with the verifier's trusted `PYTHONPATH` and no user site-packages, `python3` lists the image's `pytest11` entry points. A plugin is proposed for `-p` only when its name is registered once, and its `*.dist-info`, `entry_points.txt` and every component of its module path resolve (through `PathFinder`, without importing anything) to files that are root-owned and not group- or world-writable, as is every parent directory, outside `/tmp`, `/var/tmp`, `/logs`, `/testbed`, the workspace and `/home/<sandbox_user>`. Registrations inside those trees are ignored, so planted metadata can neither add a name nor drop a protected one. Plugins declared in the task's `[verifier] pytest_plugins` or inferred from test.sh (for example `--ctrf`) are added. A declared plugin that discovery rejects fails hardening; one that does not exist yet (test.sh installs it later, for example with `uvx`) is left to the runtime guard.

**The guard, where pytest runs.** After the agent is stopped, hardening writes the guard as `/_benchflow_guard_<32 hex>/_benchflow_guard_<32 hex>.py` (root, mode 0444; the name is random for each verification and the directory must not already exist) and puts `-p _benchflow_guard_<hex>` first among the `-p` plugins. It writes a copy into the first site directory on the isolated `sys.path` of every `python`, `python3` and `python3.N` on the verifier `PATH`, and of the shebang interpreter of each `pytest` or `py.test` script there, so `python -I` and a test.sh that replaces `PYTHONPATH` still import it. Candidates that are not Python 3 are skipped; a Python 3 that cannot take or resolve its copy fails hardening.

Only `PYTHONPATH` reaches a Python that has no copy: one that test.sh creates or installs after hardening, or one it runs by a path off the verifier `PATH`. Hardening therefore puts the guard's directory first on the verifier's `PYTHONPATH` only when no Python took a copy, or when a script in the verifier directory (shell or Python, comments ignored) names such a Python: `uvx`; `uv run`, `uv tool`, `uv venv`, `uv sync` or `uv python`; `python -m venv` or `virtualenv`; a `run`, `exec`, `create`, `install`, `shell`, `activate` or `env` subcommand of conda, mamba, micromamba, pixi, pipx, poetry, pdm, hatch, rye or pyenv; tox or nox; a `bin/activate` script; a distro package install of Python; a `python*`, `pytest` or `py.test` path in a `bin` directory outside the verifier `PATH`; or a `PATH=` assignment that adds such a directory. Otherwise `PYTHONPATH` holds only the image's trusted entries, empty when the image sets none, as before the guard existed. Task anti-tamper checks treat a set `PYTHONPATH` as a startup-injection hook (for example `os.environ.get("PYTHONPATH")` in a `python3 -I -S` preflight or inside pytest), so an entry that loads nothing would score a correct solution 0. In the verifier's own pytest, the guard's `pytest_addhooks` checks every `-p` name (from `PYTEST_ADDOPTS`, the command line and the discovered list) and every `PYTEST_PLUGINS` module against the interpreter pytest actually runs, before later plugins import. Its `pytest_plugin_registered` checks each plugin's `pytest_plugins` before pytest imports them. A duplicate registration or a module shadowing a trusted plugin is refused wherever this pytest can see it.

The guard is installed only when `[verifier].service` is `main`. Hardening never touches other services (#248), so a target-service verifier gets the discovered `-p` flags without the guard.

**In a separate verifier sandbox** (`[verifier] environment_mode = "separate"`) no agent process ever ran, so owners and modes say nothing about the agent there. Discovery and the guard block only the roots the transfer wrote (the frozen workspace, each declared artifact's bundle root, which for a declared file is its directory, and `/logs/artifacts`) plus `/logs`, where the trajectory is published, and trust everything else in that sandbox as the verifier image's own, without the owner and mode rule. `/tmp`, `/var/tmp` and `/testbed` are not blocked there unless the transfer wrote them. A plugin the agent left in its workspace still crosses with the workspace and is still refused.

### The verifier's uv and pip state

The guard trusts plugin code only outside agent-writable trees, so it can accept a plugin that test.sh installs after hardening only if the installer puts it, and reads what it installs, somewhere the agent never could write. With `HOME=/root` and an image that sets `WORKDIR /root`, uv's default cache `$HOME/.cache/uv` is in the workspace. Before this step existed, the environment that `uvx --with pytest-json-ctrf pytest --ctrf ...` built there was refused (`Verifier plugin trust rejected: ctrf`), and citation-check, the README quickstart task, and other SkillsBench tasks with that pattern scored 0 for a correct solution. That refusal was not paranoid: the agent owned that cache, its uv tool directory, its uv-managed Pythons and its `uv.toml` while it ran, and uv reuses a cached environment or a managed Python it finds there. With any other `WORKDIR`, `/root/.cache/uv` was judged by its owner and mode bits, which depend on the runtime: on Docker-in-Docker, whose `docker exec` mask is 0000, uv created it world-writable and the guard refused Terminal-Bench 2's own `ctrf`, so every `--ctrf` task there ended unscored.

After the agent stops, hardening runs a probe as root (`src/benchflow/sandbox/_verifier_tool_state.py`) that moves every location where the verifier's uv and pip keep state into one new directory, `/_benchflow_verifier_<32 hex>` (root, mode 0755, created fresh for each verification; the name must not exist). The guard trusts code below that directory by its path: only the directory and its parents must be root-owned and not group- or world-writable, so what uv writes inside loads whatever modes the runtime's mask gave it. A link inside it is judged where it points, and blocked prefixes still win. The probe also resolves the user's configuration files the way uv and pip do (`$XDG_CONFIG_HOME`, else `$HOME/.config`); such a file is safe when it is outside every agent-writable prefix, written and resolved, and it and every existing parent are root-owned and not group- or world-writable.

| Variable | Default it replaces | New value |
|---|---|---|
| `UV_CACHE_DIR` | `$HOME/.cache/uv` | `<dir>/uv-cache` |
| `UV_TOOL_DIR` | `$HOME/.local/share/uv/tools` | `<dir>/uv-tools` |
| `UV_PYTHON_INSTALL_DIR` | `$HOME/.local/share/uv/python` | `<dir>/uv-python` |
| `PIP_CACHE_DIR` | `$HOME/.cache/pip` | `<dir>/pip-cache` |
| `UV_CONFIG_FILE` | project `uv.toml`/`pyproject.toml`, `$HOME/.config/uv/uv.toml`, system file | a safe `UV_CONFIG_FILE` the verifier environment already sets; else the safe user file, else the image's first safe system `uv.toml`, else an empty `<dir>/uv.toml` (left unset with `UV_NO_CONFIG`) |
| `PIP_CONFIG_FILE` | `$HOME/.pip/pip.conf`, `$HOME/.config/pip/pip.conf` | an empty `<dir>/pip.conf` when a user file is unsafe |

uv's configuration is always pinned because uv reads index URLs from it and what it installs lands in the trusted directory: without `UV_CONFIG_FILE`, uv also reads the project's `uv.toml` or `[tool.uv]` in the working directory, the workspace, where an agent-written index redirect would choose trusted code. uv reads only `UV_CONFIG_FILE` when it is set. pip installs into site-packages or a venv, not into the new directory, and reads no workspace configuration, so its configuration is replaced only when a user file is unsafe: pip skips its user files when `PIP_CONFIG_FILE` names an existing file, and an empty one keeps the image's global `/etc/pip.conf`, which `/dev/null` would also drop.

A plugin test.sh installs with `uvx`, `uv tool` or `uv run --with` then lives under `<dir>` and passes the guard. A cache, tool environment, managed Python or index redirect the agent planted in `$HOME` is never read. A warm cache the image ships in `$HOME/.cache/uv` is not used either: uv downloads into the new directory, as it does in a fresh container. Only a root verifier in `main` is moved: hardening runs there alone, and a non-root verifier could not write the root-owned directory. Without `python3` at hardening, the decision uses the paths alone and the directory is made in `sh`. A probe that fails leaves everything where it was; the guard then judges it by owner and mode, which `umask 022` keeps correct unless test.sh changes its own mask.

Not moved, on purpose:

- `UV_TOOL_BIN_DIR` (and the uv installer's own `$HOME/.local/bin`). test.sh adds `$HOME/.local/bin` to `PATH` by sourcing the installer's `env` script; moving the bin directory would take `uv tool install` executables off that `PATH`. The guard judges plugin code, which lives in `UV_TOOL_DIR`, not the entry-point scripts.
- Project environments (`uv run`, `uv sync`, a `.venv` test.sh creates in the workspace). They are built from the workspace's `pyproject.toml`, which the agent controls; moving them into a trusted directory would let the agent's project metadata choose trusted plugin code. A plugin installed there is still refused, and the refusal is reported as a verifier error (next section). powerlifting-coef-calc (`uv init`, `uv add`, `uv run` in `/root`) and Terminal-Bench 2's mailman (`uv venv .tb`, `uv pip install pytest-json-ctrf`) are such tasks; `bench tasks check` warns about the pattern and names the fix.
- `HOME` itself. Tests read agent output through `~` and `$HOME`, and the uv installer writes where `$HOME` points.

### How guard failures are scored

The guard reports to the verifier through pytest's output and through marker files named `<guard>.<pid>-<random>.<kind>`. A marker turns a run the solution may have failed into an unscored one, so a solution must not be able to write one. Hardening creates the guard's marker directory, `/_benchflow_guard_<hex>/.markers` (mode 0700), and its key file, `/_benchflow_guard_<hex>/.key` (mode 0400), after the agent stopped, both owned by the user test.sh runs as (`id -u` as that user; root unless `[verifier] user` or the image says otherwise). Each marker is one line: an HMAC-SHA256 of its name and body under that key, and the body. The key derives from a secret that never leaves the BenchFlow process, so the verifier recomputes it from the guard's name, and it is never in the environment test.sh and its children inherit (`PYTEST_ADDOPTS` names the guard, not the key). After test.sh, the verifier reads the directory back with one command (regular files only, at most 64 of 64 KiB each), keeps the markers whose signature checks out, and copies them to the trial's `verifier/`. Unsigned files, including any marker-named file in `/logs/verifier`, are ignored. The guard writes its `loading` marker only when pytest's plugin loader imports it, so importing the guard by name leaves nothing behind, and a `registered` marker once a pytest registered it.

| What happened | How the verifier knows | Result |
|---|---|---|
| The guard refuses a plugin and a file behind the refusal predates the guard, or it cannot read plugin files or metadata | pytest stops with `Verifier plugin trust rejected: ...` | Scored as test.sh reports it (normally 0): this is the agent-tampering case |
| The guard refuses a plugin the verifier installed itself: every refused plugin has an untrusted file behind it, and none of those predates the guard | an `<guard>.<pid>-<random>.installed` marker listing each untrusted file and why; pytest stops with `Verifier plugin trust rejected: ... (...); the verifier installed it after the agent stopped, so the run is not scored` | Verifier error (`PluginGuardLoadError`, category `verifier_failure`) naming the first file and why it is untrusted, not scored |
| A Python on the verifier `PATH` cannot take a guard copy | the install command fails during hardening | Verifier error before test.sh runs |
| pytest cannot import the guard (a Python test.sh created itself, run with `-I` or its own `PYTHONPATH`, or one the `PYTHONPATH` scan above did not recognise) | `Error importing plugin "<guard>"` or `No module named '<guard>'` in a top-level file of `/logs/verifier`, when no signed `registered` marker shows that a pytest of this run registered the guard | Verifier error (`verifier crashed: ...`, category `verifier_failure`), not scored |
| pytest imports the guard but pluggy refuses its hooks | `Plugin '<guard>'` in a log (with no `registered` marker), or a leftover signed `<guard>.<pid>-<random>.loading` marker (written when pytest's plugin loader imports the guard, removed once pytest registered it) | Verifier error, not scored |
| A guard hook raises for another reason (a guard bug, a pytest API it did not expect) | a `<guard>.<pid>-<random>.crashed` marker holding the traceback | Verifier error, not scored |

Every refusal names each refused plugin and what the guard could not trust about it: a path and why (it is under an agent-writable prefix such as the workspace, it is owned by a uid other than root, or it or a parent is group- or world-writable, with its mode), a name more than one distribution registers, or a plugin installed nowhere pytest looks.

Which refusals are the verifier's own is decided by change time (`st_ctime`). Hardening writes the guard after the agent's processes are gone and the workspace is frozen (`chown -R`, which also sets the change time of every workspace file), and only the kernel sets a change time, to the current time, so a file changed after the guard was written was changed by root code of the verification. The files behind a refusal are the plugin's registrations (`*.dist-info` and `entry_points.txt`) and the files and package directories its module resolves to, not their parents; only the untrusted ones count. A refused plugin with no untrusted file behind it stays scored: one installed nowhere (pytest could not load it either), or a name that two trusted distributions register.

Errors while reading plugin files and metadata count as refusals because the agent can plant those files. A malformed `entry_points.txt` line alone makes CPython 3.12 raise `TypeError`, so the rule cannot depend on the exception type. A guard bug inside that inspection is therefore still scored.

Limits:

- A pytest that cannot import the guard is recognised only by its message. If test.sh sends pytest's output to `/dev/null`, that run is still scored as test.sh's 0. No sound check exists without the output. The Python involved was created after hardening, so BenchFlow installed nothing in it, and `-I` ignores every `PYTHON*` hook. Before importing `-p` plugins, pytest reads only its `-c` file, whose access time is unreliable and which as a FIFO could hang pytest. A verification with no guard marker cannot count as a failure either, because test.sh need not run pytest: a solution that made test.sh stop before pytest would turn its own 0 into an unscored error.
- The `PYTHONPATH` decision reads the verifier scripts as text. A verifier that makes a Python in a way the scan does not recognise, for example from a script it downloads, cannot import the guard and gets a verifier error, not a score. A verifier that needs the `PYTHONPATH` entry and also rejects a set `PYTHONPATH` still fails its own check; outside `PYTHONPATH`, the only location a venv made after hardening shares with its base Python is that Python's standard library, which integrity checks also inspect (for example by flagging standard-library files written after the image build), and a Python downloaded during verification shares nothing.
- When the marker directory cannot be read back after test.sh, the run is scored as test.sh reports it, as before markers existed. A pytest that runs as a user other than the one test.sh runs as (a test.sh that `su`s to another user) cannot read the key or write markers, so its guard failures are scored the same way.
- Code the verifier runs as test.sh's user, for example a solution module that a test imports, can read the key and sign a marker, as it could rewrite the reward or make pytest exit 0; the signature stops everything else, including processes of other users and a solution that only writes files where markers used to go. pytest's load-failure message, which code under test can print, counts only when no pytest of the run registered the guard, so a test.sh that runs one pytest with the guard and a later one that cannot import it scores that later failure as test.sh reports it.
- The `installed` classification errs toward an unscored error, never toward a pass. A file the agent planted counts as new if test.sh itself changes it after hardening (a `chmod -R`, `chown -R` or hard link over the planted file), and a verifier that installs its plugin from files the agent controls (the workspace's `pyproject.toml` or `requirements.txt`) gets an error, not a 0, for what the agent put there. A system clock stepped backwards between the freeze and verification can make a verifier install look old, which leaves it scored as before.

## Per-task opt-outs

Tasks declare opt-outs in their task config (`task.md` front-matter, or `task.toml` for split-layout tasks):

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

## Network policy: denylist egress

`network_mode: denylist` keeps the internet reachable and makes a list of URLs and hosts unreachable for the agent. The use case is a task built from a published paper: the agent may search and read freely, but the paper, its mirrors, and its code repository are off limits ([benchflow-ai/FrontierPhysics#365](https://github.com/benchflow-ai/FrontierPhysics/issues/365)).

This is a shared sandbox policy, independent of the task's domain, harness name,
model identifier, or provider. Every supported ACP harness, including custom
registrations, receives the same proxy, certificates and UID firewall. Primary
connections and later roles use the same enforcement path. Model compatibility
is still governed by each harness/provider adapter. The requirements and hosted
fetch limitations below apply regardless of model; they are not exceptions for
particular model names.

```yaml
sandbox:
  network_mode: denylist
  blocked_urls:
    - https://example.org/papers/lattice-qcd-2026
    - github.com/example-org/lattice-qcd-code
  blocked_hosts:
    - mirror.example.net
```

See [task authoring](./task-authoring-task-md.md#network-policy) for the field rules. The proxy filters the agent only; oracle runs and the verifier are not filtered.

### Mechanism

1. **Loopback proxy.** Before the agent starts, benchflow uploads a stdlib Python proxy (`src/benchflow/sandbox/_egress_denylist_proxy.py`) and starts it as root on `127.0.0.1:18628`. A request that matches the denylist gets `403 Forbidden` with an `X-BenchFlow-Blocked: 1` header; allowed HTTP requests are forwarded to their checked destination. A relayed connection closes after 15 minutes with no byte in either direction; a response that keeps streaming, such as a long model turn, is relayed whole however long it runs.
2. **Uid firewall.** The same `iptables` owner rule that backs the no-web mode lets the sandbox user reach loopback only. Every other outbound packet from that uid is rejected, so the proxy is the only way out. `iptables` is installed on first use (apt, dnf, or apk) when the image lacks it.
3. **Proxy and CA environment.** The agent env gets `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`, `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE`, `GIT_SSL_CAINFO`, `NODE_EXTRA_CA_CERTS`, and `NODE_USE_ENV_PROXY`, plus the `BENCHFLOW_EGRESS_DENYLIST=1` marker that arms the firewall. These are added after the sandbox-local LiteLLM gateway starts, so the gateway's upstream provider traffic does not pass through the egress proxy.
4. **HTTPS inspection.** The proxy terminates every external TLS connection with a leaf certificate signed by a per-rollout CA (`BenchFlow egress policy CA`). Known policy hosts have pre-generated leaves; OpenSSL mints other leaves on demand. The CA signing key and proxy files are root-only inside `/opt/benchflow-egress` (directory mode `0700`, files `0600`) and are removed at cleanup. `CONNECT`, TLS SNI when supplied, the HTTP Host, and any absolute request URL must agree on the destination; mismatches fail closed even when a client disables certificate verification. This prevents shared-CDN domain fronting. The proxy forwards one framed HTTP request per connection and does not relay later client bytes. Hosts in `blocked_hosts` are refused at `CONNECT` time.
5. **Hosted search off.** Provider-side search tools fetch pages from the model provider's servers, outside the sandbox, so the proxy cannot see them. Benchflow disables them per harness:

   | Harness | Switched off | Still on |
   |---|---|---|
   | `claude-agent-acp` | `WebSearch` | `WebFetch` (fetches from inside the sandbox, through the proxy) |
   | `codex-acp` | `web_search` (`CODEX_CONFIG`) | local shell tools |
   | `gemini` | `google_web_search`, `web_fetch` (tries a hosted fetch first) | |
   | `opencode`, `mimo` | `websearch` | `webfetch` |
   | other harnesses | nothing | whatever hosted tools they ship |

6. **Block log.** Each refused attempt is appended to a root-owned log that benchflow downloads to `trajectory/egress_denylist.jsonl` in the rollout directory at cleanup: one JSON object per line with `ts`, `action`, `method`, `url`, and `rule` (`host:<host>`, `url:<host><path>`, `ip-literal`, `private-address`, `authority-mismatch`, or `tls-sni-mismatch`). A TLS SNI mismatch is rejected during the handshake and logged with `server_name`, rather than returning an HTTP response. For a refused `CONNECT`, `url` holds the requested hostname (with its port for pre-TLS denials).

The controller also registers the exact `127.0.0.1:<port>` endpoint of its local
model gateway with the proxy. Clients such as Gemini's Undici `ProxyAgent`
ignore `NO_PROXY` and tunnel even HTTP model calls through the egress proxy.
Both direct and proxied requests can reach that one endpoint. This exception
comes from the running provider gateway, never task metadata or agent-supplied
environment variables; other IP addresses and private destinations remain
blocked. Each reconnect registers the current gateway port.

Codex runs in its `agent-full-access` session mode when BenchFlow has already
selected a non-root sandbox user, unless the caller explicitly sets
`INITIAL_AGENT_MODE`. This avoids nesting Codex's bubblewrap sandbox inside
Docker or Daytona, where namespace creation can fail before a tool runs.
BenchFlow's user, filesystem restrictions, proxy and UID firewall still apply.
Hosted search is disabled through `CODEX_CONFIG.web_search`, including when a
caller configured live search; codex-acp's CLI does not consume `-c` overrides.

Threads that codex-acp starts without the session's `CODEX_CONFIG` would use Codex's built-in `openai` provider at api.openai.com. Its title generator (codex-acp 1.13.1 through 2.0.1) starts one after the first turn, on the hard-wired `gpt-5.6-luna`, with the task prompt as input. When BenchFlow routes Codex to a provider, the launcher writes that provider into Codex's user `config.toml` (under `$CODEX_HOME`, default `~/.codex`) as the default for every thread, and points the built-in `openai` provider at the same endpoint; a later launch with no provider removes the file BenchFlow wrote. The model gateway serves only the run's model: a request for any other model, such as the title thread's, gets HTTP 400 `model_not_served`, never reaches the provider, and is counted in a warning when the rollout ends.

For OAuth runs, use the Claude Code harness (`claude-agent-acp`) with a bare
Claude model and `CLAUDE_CODE_OAUTH_TOKEN`, without `ANTHROPIC_API_KEY`. Its
native authenticated traffic passes through the same egress proxy, while
`WebSearch` stays disabled and local requests still receive policy denials.

Matching ignores scheme, port, query string, and case, strips a leading `www.`, and compares a normalized path: percent-encoding is decoded (repeatedly), `.` and `..` segments are resolved, duplicate slashes and backslashes collapse, and `;` path parameters are dropped, so `/abs/../abs/2401.12345` and `/abs/%2e%2e/abs/2401.12345` match the same entry as `/abs/2401.12345`. A `blocked_urls` entry blocks every path under it; a `blocked_hosts` entry blocks the host and its subdomains. Apart from the registered model gateway, requests to addresses are refused in every notation a resolver accepts (dotted, decimal, hex, octal) and through wildcard DNS names that embed an address (`1-2-3-4.sslip.io`), so a blocked host cannot be reached by its address. A name the agent controls that resolves to the blocked address is not detected; that is the inherent limit of a hostname denylist. For every other destination, the proxy resolves its address and refuses names that resolve to loopback, private, link-local, or other non-global addresses (cloud metadata included), so a hostname the agent controls cannot turn the root proxy into a bridge to sandbox-internal or host services. The uid firewall stays for the rest of the sandbox life, as in the no-web mode: a later oracle role in the same sandbox, and a verifier configured with `verifier.user` equal to the sandbox user, run without egress.

### Requirements

- A non-root `sandbox_user`. Setup fails closed before any sandbox is created when it is missing.
- `python3` (or `python`) and `openssl` on `PATH` in the task image. The proxy is a stdlib script; OpenSSL signs dynamic leaf certificates. Setup fails closed if either runtime is absent.
- An ACP agent. Session-factory agents raise at connect time because the uid firewall only runs in the ACP path.
- `docker`: the agent container needs `NET_ADMIN` for `iptables`. Benchflow adds it through its own compose overlay (`src/benchflow/sandbox/_compose_files/docker-compose-net-admin.yaml`), so the task's `Dockerfile` and `docker-compose.yaml` need nothing extra.
- `daytona`: verified on direct sandboxes with `iptables`. The DinD strategy (tasks with a `docker-compose.yaml`) stacks the same `NET_ADMIN` overlay on `main`.
- `modal`, `apple-container`, and `agentcore` refuse the mode at preflight. `bench tasks check --sandbox <backend>` reports it, and the runtime raises before the sandbox is created.

### Limits

- **A blocklist hides pages, not knowledge.** The model may have read the paper in training. The denylist only stops it from re-reading the page during the run.
- **Mirrors and citations stay reachable.** Only listed prefixes and hosts are blocked. Preprint mirrors, aggregator copies, cached versions, citing papers that restate the result, and the same repository under another host all stay reachable unless listed. `blocked_hosts` covers subdomains, not other domains. List every form of a page you know about (abstract, PDF, HTML, versioned).
- **The block is visible.** A refused request gets a `403` that names the policy, and the agent can tell intercepted hosts from the certificate issuer (`BenchFlow egress policy CA`). Do not expect the agent to be unaware that a page is off limits.
- **Tools that ignore proxy variables fail closed.** A client that does not honor `HTTP_PROXY` and `HTTPS_PROXY` cannot reach the network at all, because the uid firewall rejects non-loopback traffic. It gets a connection error, not the page, and the attempt does not appear in the block log.
- **Hosted search coverage is per harness.** Only the harnesses in the table have a switch. A harness with a hosted fetch that is not listed there can reach blocked pages through the provider.
- **External egress is HTTP/1.x.** External `CONNECT` requests must carry TLS and a checked HTTP/1.x request. Arbitrary TCP tunnels, HTTP upgrades, and clients pinned to an origin certificate are not supported. TLS clients must trust the injected CA; HTTP/2-capable clients must permit HTTP/1.1 fallback. Only the exact controller-registered loopback HTTP model gateway keeps opaque tunneling. A client that ignores the HTTP proxy protocol is rejected by the firewall.

## Network policy: allowlist egress

`network_mode: allowlist` makes only the listed destinations reachable for the agent. It uses the same proxy, CA, uid firewall and block log as the denylist, in an allow mode. Declare it on `sandbox`, or on `agent` over a public sandbox (Harbor's recommended form, [harbor#2146](https://github.com/harbor-framework/harbor/issues/2146)); the two mean the same thing here because BenchFlow filters the agent's uid only. Entry syntax is in [task authoring](./task-authoring-task-md.md#network-policy).

```yaml
agent:
  network_mode: allowlist
  allowed_hosts: [pypi.org, "*.pythonhosted.org", 10.0.0.0/8]
```

### What is enforced where

| Traffic | Enforced by | Rule |
|---|---|---|
| Agent HTTP(S) to a hostname | proxy (`127.0.0.1:18628`) | allowed if the name matches an exact or `*.` entry; otherwise `403`, rule `not-allowlisted`, and the name is never resolved |
| Agent HTTP(S) to an IP literal | proxy | allowed only inside a listed IP/CIDR entry, relayed without TLS interception; other literals get `not-allowlisted`, and non-canonical notations (`3232235777`, `0x…`, `127.1`) get `ip-literal` |
| Address a listed name resolves to | proxy, at connect time | public addresses, or non-public ones inside a listed range; anything else (metadata `169.254.169.254`, RFC 1918 outside the list, IPv6 ULA/link-local) is `private-address`. Loopback, unspecified and multicast are never reached through the proxy |
| Redirects | the client | the proxy does not follow them; the client's next request is judged on its own, so a redirect to an unlisted host gets `403` |
| HTTPS inner authority | proxy (TLS interception) | CONNECT host, SNI and `Host` must agree, as in the denylist, so an allowed CDN name cannot front an unlisted site |
| Agent DNS (UDP/TCP 53, any server) | uid firewall `nat` redirect to the proxy's DNS filter (`127.0.0.1:18653`) | listed hostnames are forwarded to the resolvers in `/etc/resolv.conf`; every other question is answered `REFUSED` without any upstream query, and logged once per burst as `dns-not-allowlisted`. Docker's embedded resolver (`127.0.0.11`) is refused directly |
| Agent traffic to listed IP/CIDR entries | uid firewall (`iptables`/`ip6tables` ACCEPT) | reachable directly, for any protocol (e.g. `psql` to a compose sibling) |
| All other agent traffic | uid firewall | rejected (IPv4 and IPv6) |
| Model API | controller | the loopback LiteLLM gateway is always reachable (its port is registered with the proxy). Native Claude subscription auth (`claude-agent-acp` with `CLAUDE_CODE_OAUTH_TOKEN`, no gateway) gets `api.anthropic.com:443` for `POST /v1/messages` only, with the URL-source body check; unless the task lists `api.anthropic.com` itself. Other native subscription clients (e.g. Codex ChatGPT login) are refused at connect time with an error, instead of failing silently on their first model call |
| Agent install, setup, oracle | not filtered | the firewall is installed after the agent is installed and its ACP session is bootstrapped; oracle runs get no policy ([harbor#583](https://github.com/harbor-framework/harbor/issues/583)) |
| Verifier | not filtered (root) | the verifier keeps its own rule: it runs as root outside the uid firewall. `verifier.network_mode: allowlist` is refused at preflight; a verifier configured with `verifier.user` equal to the sandbox user inherits the firewall |
| Separate verifier sandbox (`verifier.sandbox_mode: separate`) | not filtered | the agent's allowlist is not carried into it; `network_mode: allowlist` or `denylist` on `verifier.sandbox` is refused at preflight, because there is no agent uid or proxy there |

Hosted search is switched off per harness, as for the denylist (provider-side fetches bypass the sandbox).

### Validated backends

- `daytona`, direct sandbox: live canary `tests/test_network_allowlist_integration.py` (`single` lane).
- `daytona`, DinD compose (Docker engine inside the sandbox, Docker's embedded DNS, the `NET_ADMIN` overlay): the same canary's `compose` lane, including a sibling service reached by name through the proxy and directly.
- `docker` on the host: same code path and compose overlay as the DinD lane; the host lane of the canary has not been run. It needs the `nat` table and the `owner` match in the Docker VM's kernel; when either is missing the firewall setup exits 86 and the rollout fails before the agent starts ([harbor#2527](https://github.com/harbor-framework/harbor/issues/2527) was the opposite: a missing kernel feature went unnoticed).
- `modal`, `apple-container`, `agentcore`: refused at preflight.

### Requirements

The denylist requirements apply: a non-root `sandbox_user`, `python3` and `openssl` in the image, an ACP agent, and `NET_ADMIN` (added by BenchFlow's compose overlay). The kernel must also provide the `iptables` `nat` table.

### Limits

- **Non-HTTP protocols to hostnames are not supported.** A hostname entry is reachable through the HTTP proxy only. Tools that ignore `HTTP_PROXY` (git over ssh, raw TCP clients) reach only IP/CIDR entries.
- **CIDR entries admit addresses, not names.** A name that is not listed does not resolve for the agent even if its address is inside a listed range. List the name too.
- **Wildcard DNS services.** Listing `*.sslip.io` or similar admits names that embed any address; the address check still refuses loopback, metadata and unlisted private ranges, but any public address becomes reachable.
- **Shared addresses.** A CIDR entry admits every site hosted in that range; use hostnames for CDN-hosted services.
- **Empty lists are rejected.** Harbor treats an empty allowlist as "deny all"; BenchFlow requires at least one entry (use `no-network` for none) and has no `--allow-agent-host` flag.
- The denylist limits on HTTP/1.x, certificate pinning and client proxy support apply unchanged.

## Threat model and known gaps

Benchflow's hardening assumes:
- The Docker image and pre-installed packages are trusted (they're built before the agent runs).
- The agent runs as a non-root sandbox user (`sandbox_user`) — when set to `None`, hardening still runs but the agent can defeat lockdown by being root. Don't run untrusted agents as root.
- The verifier itself is trusted code shipped with the task; benchflow only hardens *what the agent can do to the verifier's environment*.

Known residual risk:
- An agent with sustained access can poison `__pycache__` for files that exist in the baseline (those caches aren't deleted because some tasks diff workspace against `/testbed_verify`). Mitigated by the workspace chown but not eliminated.
- Build-config files are snapshotted before the agent runs but restored only with `restore_workspace=True`, which is off by default. A verifier that installs the workspace (`pip install -e .`) therefore runs the agent's `setup.py` or build backend as root.
- When the workspace is the verifier's `HOME` (`WORKDIR /root`), test.sh runs what the agent left there. The uv installer (`curl -LsSf https://astral.sh/uv/<version>/install.sh | sh`) does not overwrite an existing `$HOME/.local/bin/env`, which test.sh then sources as root, and that script puts `$HOME/.local/bin` first on `PATH`, where a planted `mkdir` or `python3` shadows the system one. An agent can make such a test.sh report a pass without any plugin (a shell function named `uvx` is enough); the guard cannot see this, and earlier releases behave the same. The fix needs `$HOME/.local/bin` reset to its pre-agent state before verification, which changes the workspace the verifier scores; that choice is left to the task author.
- For a workspace that is not `HOME`, `uv pip install` and `uv run` still read the workspace's `uv.toml` and `[tool.uv]` settings, as before: replacing them for every task would break task projects that declare their own index. A task can set `UV_NO_CONFIG=1` in `[verifier] env`.

## Related

- [`progressive-disclosure.md`](./progressive-disclosure.md) — soft-verify (the relaxed hardening used between rounds in multi-round trials).
- [`task-authoring.md`](./task-authoring.md) — the task config schema, including the `[verifier.hardening]` opt-outs.
