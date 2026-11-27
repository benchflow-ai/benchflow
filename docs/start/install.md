# Install

BenchFlow is the `benchflow` package on PyPI. It installs one command under two names, `bench` and `benchflow`, and a Python library. Install it with [uv](https://docs.astral.sh/uv/getting-started/installation/), on Linux or macOS, with Docker running for the default sandbox. On Windows, run it inside WSL2 with a Linux Docker daemon; native Windows Python isn't supported, because BenchFlow needs Unix file locking.

## Install

Install a fixed version, with the dependency versions its release was tested with:

```bash
uv tool install --python 3.12 'benchflow==0.10.4' \
  --constraints https://raw.githubusercontent.com/benchflow-ai/benchflow/v0.10.4/constraints.txt
bench --version
```

```text
benchflow 0.10.4
```

uv fetches Python 3.12 if you don't have it. Keep `--python 3.12`: with an older Python, uv picks an old release that has no `bench` command and stops with `No executables are provided by package benchflow`. If your shell then says `bench: command not found`, run `uv tool update-shell` and open a new terminal.

Without `--constraints`, uv installs the newest version of every dependency that BenchFlow allows today, which can be newer than the versions the release was tested with. The tested versions can have flaws found later: on 7 November 2026, pip-audit 2.9.0 found no known vulnerability among 0.9.2's. Each release's known issues are in the [Changelog](../versions/changelog.md).

## Docker on Linux

BenchFlow runs tasks in Docker, with Compose and buildx. On Ubuntu 24.04, Ubuntu's own packages are the ones BenchFlow's release tests run on:

```bash
sudo apt-get update                # a fresh image has no package lists yet
sudo apt-get install -y docker.io docker-compose-v2 docker-buildx
sudo usermod -aG docker "$USER"   # then log out and back in: the group applies to new logins
docker version && docker compose version && docker buildx version
```

Docker CE from [Docker's install guide](https://docs.docker.com/engine/install/) works too. BenchFlow 0.10.4 fully supports Docker Engine 29 and Docker Compose 2.40 or newer: on older versions a trial under a network policy (allowlist, denylist or no-network) is refused before its turn, naming the versions, and trials with an open network run. `bench doctor` checks Docker, Compose, buildx and the docker group, and prints the fix for each one missing.

## Check your machine

`bench doctor` checks Python, uv, Docker, the agent logins it can find and the network. A fresh machine with Docker and no logins looks like this:

```bash
bench doctor
```

```text
BenchFlow doctor - benchflow 0.10.4, Linux x86_64, sandbox docker

Runtime
  PASS  python                Python 3.12.15, benchflow 0.10.4 (...)
  PASS  uv                    uv 0.13.1 (x86_64-unknown-linux-gnu)
  PASS  bench on PATH         .../bin/bench, this install
  PASS  bfx                   bfx x86_64, aarch64 (pinned)
Sandbox
  PASS  docker                Docker 29.1.3 via default (16 CPU, 62.8 GiB, 58.7 GiB free in /var/lib/docker)
  SKIP  daytona               not configured (set DAYTONA_API_KEY to use --sandbox daytona)
Agent credentials
  SKIP  claude-agent-acp      no credential found
                              fix: Run `claude setup-token` and export CLAUDE_CODE_OAUTH_TOKEN=<token> (or export ANTHROPIC_API_KEY)
  SKIP  codex-acp             no credential found
                              fix: Run `codex login` (ChatGPT) or export OPENAI_API_KEY
  ...
  WARN  agent credentials     no model agent can run: no credential found for any agent (the oracle and nop controls still can)
...
12 passed, 1 warning, 0 failed, 6 skipped
```

`...` marks lines cut from the real output. `bfx` is a small Linux program BenchFlow copies into each sandbox to run its own steps there. A `WARN` about agent credentials is expected until you log in to an agent; [Agents and models](../evaluate/agents.md) shows how. You can still run the [Quick start](./quick-start.md)'s first step, which needs no model.

## Add a cloud sandbox

Docker needs nothing extra. Daytona, Modal and Amazon Bedrock AgentCore each need an extra, named in the install line:

```bash
uv tool install --python 3.12 'benchflow[sandbox-daytona]==0.10.4' \
  --constraints https://raw.githubusercontent.com/benchflow-ai/benchflow/v0.10.4/constraints.txt
```

The other extras are `sandbox-modal` and `sandbox-agentcore`; list several as `benchflow[sandbox-daytona,sandbox-modal]`. Each sandbox also needs its own login: see [Sandboxes](../scale/sandboxes.md).

## Use it from Python

To call BenchFlow from your own code, add it to that project's environment:

```bash
uv add 'benchflow==0.10.4'
```

`pip install 'benchflow==0.10.4'` works too, on Python 3.12 or newer.

## Go back to 0.9.3

0.9.3 is the last release before 0.10, and 0.10.3 the one before 0.10.4. A page or section new in 0.10 says so in its first line or its heading, and the nav marks it (0.10). A score from 0.10.4 and one from 0.9.3 aren't comparable on the tasks [the changes that can move a score](../versions/0.10.md#changes-that-can-move-a-score-or-stop-a-task) touch. To go back to 0.9.3, run the install line with 0.9.3 in both places (0.10.3 the same way):

```bash
uv tool install --python 3.12 'benchflow==0.9.3' \
  --constraints https://raw.githubusercontent.com/benchflow-ai/benchflow/v0.9.3/constraints.txt
```

PyPI keeps the 0.10 release candidates too, the last of them 0.10.0rc5: install one by its version the same way.

## Upgrade, go back or uninstall

A pinned install stays on its version. To move it, up or back, run the install line again with the other version in both places and the extras you use; uv replaces the installed version.

```bash
uv tool uninstall benchflow
```

Uninstalling removes BenchFlow's environment and both commands. It leaves your job folders, `~/.cache/benchflow` and the task images in Docker.

Coming from 0.7? Read [Upgrade from 0.7](../versions/upgrade-from-0.7.md) first.

## Which bench runs

BenchFlow's engine rewritten in Rust is a separate prerelease, [bench 0.8.1](https://github.com/benchflow-ai/benchflow/releases/tag/bench-v0.8.1), and its command is also called `bench`. With both installed, `bench` runs whichever comes first on `PATH`; `benchflow` always runs this package.

| You type | Runs |
|---|---|
| `bench` | the first one on `PATH`: `bench --version` prints `benchflow 0.10.4` for this package, `bench 0.8.1 (ba5af6289 2026-10-13)` for the Rust engine |
| `benchflow` | this package |
| `~/.bench/bin/bench` | the Rust engine, installed as below |

To keep both, give the Rust `bench` a folder of its own. Download its release and check it against the published sha256 before unpacking it (`T=aarch64-apple-darwin` on an Apple silicon Mac):

```bash
V=0.8.1; T=x86_64-unknown-linux-musl
F=bench-$V-$T.tar.gz; R=https://github.com/benchflow-ai/benchflow/releases/download/bench-v$V
curl -fsSLO "$R/$F" && curl -fsSLO "$R/$F.sha256" &&
  shasum -a 256 -c "$F.sha256" &&
  mkdir -p ~/.bench/bin && tar -xzf "$F" -C ~/.bench/bin --strip-components 1 "bench-$V-$T/bench"
```

```text
bench-0.8.1-x86_64-unknown-linux-musl.tar.gz: OK
```

A damaged download prints `FAILED` instead, and nothing is unpacked. `bench doctor` warns about the one that's hidden. With this package first on `PATH`:

```text
  WARN  bench on PATH         .../bin/bench, this install; it hides ~/.bench/bin/bench (Rust bench)
```

With the Rust engine first:

```text
  WARN  bench on PATH         bench runs ~/.bench/bin/bench (Rust bench), not this install (.../bin/benchflow)
                              fix: benchflow always runs this install; for bench to run it too, put .../bin before ~/.bench/bin on PATH: ...
```

To choose, put that folder first on `PATH` in your shell's startup file; `export PATH="$HOME/.bench/bin:$PATH"` makes `bench` the Rust engine. Open a new terminal, or run `hash -r` in this one. `benchflow` always runs this package, whichever comes first.
