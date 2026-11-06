# Install

Install BenchFlow with uv.

```bash
uv tool install --python 3.12 benchflow
bench --version    # benchflow <version>
bench doctor       # is this machine ready to run tasks?
```

BenchFlow is the `benchflow` package on PyPI. It installs two names for the same command, `bench` and `benchflow`. You need [uv](https://docs.astral.sh/uv/getting-started/installation/), which fetches Python 3.12 if you don't have it.

`uv tool install` gives BenchFlow its own environment under `~/.local/share/uv/tools/benchflow`. It links `bench` and `benchflow` into `~/.local/bin`. If your shell then says `bench: command not found`, run `uv tool update-shell` and open a new terminal.

Keep `--python 3.12`. With an older Python, uv picks an old `benchflow` release that has no `bench` command. It then stops with `No executables are provided by package benchflow`.

`pip install benchflow` also works on Python 3.12 or newer. uv is the tested path.

## Host platforms

The CLI and Python SDK require a POSIX host, such as Linux or macOS. Native Windows Python is unsupported because BenchFlow requires Unix file locking. On Windows, run BenchFlow inside a Linux environment such as WSL2 and configure access to a Linux Docker daemon. WSL2 has not been validated separately for this release.

The packaged `bfx` helpers execute inside Linux sandboxes on x86_64 or aarch64.

## Configure a sandbox

BenchFlow runs tasks in Docker by default, so you need [Docker](https://docs.docker.com/get-started/get-docker/) running on this machine.

Daytona, Modal and AgentCore sandboxes need an extra. Name it in the install line:

```bash
uv tool install --python 3.12 'benchflow[sandbox-daytona]'
uv tool install --python 3.12 'benchflow[sandbox-modal]'
uv tool install --python 3.12 'benchflow[sandbox-agentcore]'
```

Each cloud sandbox also needs its own login. [Auth](../agents/auth.md#sandbox-credentials) lists them.

## Pin release dependency versions

`uv tool install` ignores BenchFlow's lock file. It installs the newest version of each dependency that BenchFlow allows today. To get the dependency versions recorded for a release, use the constraints file from the same stable release tag:

```bash
uv tool install --python 3.12 benchflow==<version> --constraints https://raw.githubusercontent.com/benchflow-ai/benchflow/v<version>/constraints.txt
```

Use the same version in both places, for example `benchflow==0.9.2` with `v0.9.2/constraints.txt`. uv keeps the constraints with the tool, so `uv tool upgrade` stays inside them.

With an extra, put it before the version: `'benchflow[sandbox-daytona]==<version>'`.

## Use BenchFlow from Python

To call BenchFlow from your own program, add it to that program's environment:

```bash
uv add benchflow
```

See [Run from Python](./quick-start.md#run-from-python).

## Stable and preview releases

BenchFlow publishes two kinds of release to PyPI, both named `benchflow`:

- **Stable** releases, published when maintainers tag one.
- **Preview** releases: a build of `main` after every merge whose tests and integration check pass. Preview versions have a `.devN` suffix.

Use stable releases unless you need something merged to `main` since the last one.

| | Stable | Preview |
|---|---|---|
| CLI | `uv tool install --python 3.12 --upgrade benchflow` | `uv tool install --python 3.12 --prerelease allow --upgrade benchflow` |
| Python package, in a Python 3.12+ environment | `python3.12 -m pip install --upgrade benchflow` | `python3.12 -m pip install --pre --upgrade benchflow` |
| A project that depends on BenchFlow | `uv add benchflow`, then `uv lock --upgrade-package benchflow` | `uv add --prerelease allow benchflow`, then `uv lock --upgrade-package benchflow --prerelease allow` |


### Try a pre-release

To try a release candidate, name it exactly. That also keeps every other package on a release:

```bash
uv tool install --python 3.12 --upgrade 'benchflow==<version>'
```

`--prerelease allow` applies to every dependency, not only BenchFlow, and can pull in a dependency's own release candidate. `--prerelease if-necessary-or-explicit` takes a pre-release only of the package you name. To try a build from a checkout, build `bfx` first, then build its wheel and install that file:

```bash
BFX_REQUIRE_PIN=1 scripts/build_bfx.sh   # or: make bfx
uv build --no-sources --wheel -o dist
uv tool install --python 3.12 --upgrade ./dist/benchflow-*.whl
```

`scripts/build_bfx.sh` builds `bfx`, BenchFlow's helper for root's steps on Docker, for Linux x86_64 and aarch64 into `src/benchflow/_bfx/bin/`, where the wheel picks it up. It runs on Linux and needs `rustup`; the toolchain and its musl targets come from `bfx/rust-toolchain.toml`. `BFX_REQUIRE_PIN=1` installs only binaries whose SHA-256 is the pin in `src/benchflow/_bfx/pins.py`. A wheel built without this step has no `bfx`: Docker trials then run without it, as with `BENCHFLOW_BFX_DELIVERY=off`, and `bench doctor` and the start of each `bench run` on Docker say so.

`bench --version` shows which one you have. To move back, run the install line with a release version.

Publishing is described in [Publishing a release](../contributing/release.md).

## On an Intel Mac

On an Intel Mac, the install builds `cryptography` from source. uv installs the newest `cryptography` BenchFlow allows (50.0.2 in the tested constraints), which has no prebuilt wheel for Intel Macs. Install a Rust toolchain (`rustup`) and OpenSSL (`brew install openssl@3`) first.

## Upgrade

```bash
uv tool upgrade benchflow
```

This moves to the newest release your install allows. A pinned install (`benchflow==<version>`) stays on its version, and a constraints file keeps every dependency at its pin. To move a pinned install, run the install line again with the new version in both places:

```bash
uv tool install --python 3.12 --upgrade 'benchflow==0.9.2' \
  --constraints https://raw.githubusercontent.com/benchflow-ai/benchflow/v0.9.2/constraints.txt
```

Keep any extras your installation needs, for example `'benchflow[sandbox-daytona]==0.9.2'`. See [What's new in 0.9](../whats-new-0.9.md) before resuming an older job.

If `uv tool upgrade` says `Nothing to upgrade` and suggests reinstalling with `benchflow@latest`, run `uv tool install --python 3.12 benchflow@latest`. uv's hint leaves out `--python 3.12`, and without it uv can pick an old `benchflow` release that has no `bench` command.

If uv stops with `Executable already exists: bench (use --force to overwrite)`, another program already sits at `~/.local/bin/bench`. It is an older BenchFlow installed with pip, or the Rust `bench`. `--force` replaces it. If it is the Rust `bench` and you want to keep it, read [Which bench runs](#which-bench-runs) first.

## Uninstall

```bash
uv tool uninstall benchflow
```

This removes BenchFlow's environment and its `bench` and `benchflow`. It leaves three things behind:

- your job folders;
- `~/.cache/benchflow`, with crash logs, registered task formats and the model proxy's virtual environments (`litellm-proxy/`) (the Rust `bench` uses it too);
- the task images in Docker.

For the switch from Rust 0.8.2, including job folders and feature differences, see [the upgrade guide](../whats-new-0.9.md#coming-from-the-rust-engine-082).

## Which bench runs

The Rust engine also installs a `bench`, by default into the same `~/.local/bin`. Neither installer replaces the other's without `--force`. To keep both, put the Rust `bench` in a folder of its own. Download its release for your machine and check it against the release's published sha256 before unpacking it (`T=aarch64-apple-darwin` on an Apple silicon Mac, `x86_64-unknown-linux-musl` on any x86_64 Linux):

```bash
V=<version>; T=x86_64-unknown-linux-musl
F=bench-$V-$T.tar.gz; R=https://github.com/benchflow-ai/benchflow/releases/download/bench-v$V
curl -fsSLO "$R/$F" && curl -fsSLO "$R/$F.sha256" &&
  shasum -a 256 -c "$F.sha256" &&
  mkdir -p ~/.bench/bin && tar -xzf "$F" -C ~/.bench/bin --strip-components 1 "bench-$V-$T/bench"
```

The `shasum` line prints `OK` for a good download; on a mismatch it says `FAILED` and nothing is unpacked. [Installing the Rust bench](https://github.com/benchflow-ai/benchflow/blob/bench-rs/docs/rust/install.md) has the rest. Then:

| You type | Runs |
|---|---|
| `bench` | whichever of `~/.local/bin` and `~/.bench/bin` comes first on `PATH` |
| `benchflow` | this package, always |
| `~/.bench/bin/bench` | the Rust engine, always |
| `uv run bench`, inside a project with BenchFlow in its environment | that project's copy |

`benchflow` always runs this package, wherever the Rust `bench` is. `bench --version` tells you which one `bench` is: this package prints `benchflow <version>`, the Rust engine `bench <version> (<commit> <date>)`.

To choose what `bench` runs, put its folder first on `PATH` in your shell's startup file. For example, `export PATH="$HOME/.bench/bin:$PATH"` makes it the Rust engine. Then open a new terminal, or run `hash -r` in this one. A shell alias works too: `alias bench="$HOME/.bench/bin/bench"`.

`bench doctor` says which `bench` your shell runs. It warns about every other `bench` on `PATH` that it hides:

```text
Runtime
  WARN  bench on PATH  ~/.local/bin/bench, this install; it hides ~/.bench/bin/bench (Rust bench)
```

If the Rust installer overwrote this package's `bench` (with its `--force`), then `uv tool uninstall benchflow` deletes the Rust `bench`, and so does reinstalling with `--force`. To get this package's `bench` back, run the install command you installed with again, with `--force` added and nothing else changed: the same version, extras and constraints file. A bare `uv tool install --python 3.12 --force benchflow` installs the newest release without your extras or the tested versions. Keep the Rust `bench` in its own folder to avoid this.

## Next steps

- Run your first task: [Quick start](./quick-start.md).
- Log in to an agent: [Auth](../agents/auth.md).
