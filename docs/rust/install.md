# Installing bench

`bench` is one binary with no runtime dependencies: no Python, no Node, no virtual environment. To run tasks it also needs a sandbox backend (Docker or Podman, or Daytona with `DAYTONA_API_KEY` set); `bench doctor` checks for one.

## The install script

```bash
curl -fsSL https://raw.githubusercontent.com/benchflow-ai/benchflow/bench-rs/install.sh | sh
```

The script picks the build for your machine, downloads `bench-<version>-<target>.tar.gz` and its `.sha256` from the GitHub release, checks the checksum, runs the binary once to make sure it works on this machine, and puts one file, `bench`, in `~/.local/bin`. It needs no sudo and no Docker, and it never edits your shell profile: when the folder is not on your `PATH`, it prints the line to add for your shell. It ends with the next steps, `bench doctor` and `bench new my-task`.

| Setting or flag | What it does |
|---|---|
| `BENCH_VERSION=0.8.1` | Install that version, a pre-release too, instead of the newest stable release (its release tag, `bench-v0.8.1`, works too) |
| `BENCH_INSTALL_DIR=~/bin` | Install into another folder |
| `BENCH_TARGET=x86_64-unknown-linux-gnu` | Download another build than the one picked for this machine |
| `BENCH_RELEASES_URL=<url>` | Download from a mirror laid out like `https://github.com/benchflow-ai/benchflow/releases` |
| `BENCH_RELEASES_API=<url>` | Where the latest bench release is looked up: a JSON list of releases, newest first, as GitHub's API gives it (the default). A mirror without one uses `<releases>/latest` |
| `GITHUB_TOKEN=<token>` | Sent with that lookup, only over https to GitHub's API: `api.github.com` when `GITHUB_API_URL` is unset or names it, else the https host `GITHUB_API_URL` names (a GitHub Enterprise server, as GitHub Actions sets it); never to a mirror. GitHub allows 60 calls an hour per address without it, which shared CI runners run out of. A token GitHub refuses (stale, or another server's) is dropped and the lookup made again without it. When the list cannot be read, the repository's latest release is used if it is bench's |
| `--from <tarball>` | Install a local `bench-<version>-<target>.tar.gz` instead of downloading; a `.sha256` next to it is checked |
| `--force` | Replace a `bench` that the installer did not put there |

Settings go before `sh`, and flags after `sh -s --`:

```bash
curl -fsSL https://raw.githubusercontent.com/benchflow-ai/benchflow/bench-rs/install.sh | BENCH_VERSION=0.8.1 sh
curl -fsSL https://raw.githubusercontent.com/benchflow-ai/benchflow/bench-rs/install.sh | sh -s -- --force
sh install.sh --from bench-0.8.1-x86_64-unknown-linux-musl.tar.gz   # from a clone
```

**Releases.** bench's releases are tagged `bench-v<version>`; tags of the form `v<version>` are the Python package's. Each has a build for each of the three [supported targets](#supported-targets), each a `bench-<version>-<target>.tar.gz` with its `.sha256`; the command at the top of this page installs the newest.

`BENCH_VERSION` installs the release it names, a pre-release too, by its version or its tag. Without it, the script installs the newest `bench-v*` release that is not a pre-release, read from GitHub's list of releases, since the repository's own latest release is the Python package's. When there is none, it stops, says it found no bench release, and suggests `BENCH_VERSION` or building from source (below).

**If you have BenchFlow 0.8.** `uv tool install benchflow` (and pipx, and pip) put the Python CLI's `bench` in `~/.local/bin` too. The script does not replace it unless you pass `--force`; it says what it found and how to install next to it instead, with `BENCH_INSTALL_DIR=~/.bench/bin`. `uv tool install --force benchflow` brings 0.8's `bench` back.

## Build from source

The workspace pins Rust 1.99 in `rust-toolchain.toml`, and rustup fetches it on the first build. You also need a C compiler, because rustls's crypto library (aws-lc) is written in C: the Xcode Command Line Tools on macOS, gcc or clang on Linux.

```bash
git clone https://github.com/benchflow-ai/benchflow && cd benchflow && git checkout bench-rs
cargo install --locked --path crates/benchflow-cli   # puts bench in ~/.cargo/bin
```

Without a clone: `cargo install --locked --git https://github.com/benchflow-ai/benchflow --branch bench-rs benchflow-cli`.

Built from git, `bench --version` names the commit and its date, like `bench 0.8.1-dev.0 (7a88169aa 2026-10-10)`. Built from a source archive (`git archive`, or GitHub's source tarball of a tag or commit), it names the commit the same way, since git writes it into the archive's `crates/benchflow-cli/archive-commit.txt`; this holds only for commits that have that file, so a bench built from the source tarball of `bench-v0.8.1-rc.1` prints just the version. Built from a plain copy, one that is neither a source archive nor a git checkout of BenchFlow (even inside some other repository), it prints just the version.

The release binaries use the `dist` profile, which is the release profile plus whole-program LTO (6.5% smaller on macOS, for about 20 seconds more build time): `cargo build --locked --profile dist -p benchflow-cli` writes `target/dist/bench`. A clean build takes one to two minutes on a laptop or a 16-core VM. For the static Linux build, install `musl-tools` (it provides the musl C compiler aws-lc needs), then run `rustup target add x86_64-unknown-linux-musl` and `cargo build --locked --profile dist --target x86_64-unknown-linux-musl -p benchflow-cli`.

## Supported targets

| Target | Runs on | Binary | Download |
|---|---|---|---|
| `aarch64-apple-darwin` | macOS 11 or later on Apple silicon | 18.1 MB | 8.6 MB |
| `x86_64-unknown-linux-musl` | Any x86_64 Linux: static, needs no glibc. The install script picks this one on Linux | 22.5 MB | 9.7 MB |
| `x86_64-unknown-linux-gnu` | x86_64 Linux with glibc 2.34 or later: Ubuntu 22.04, Debian 12, RHEL 9, and newer | 22.3 MB | 9.6 MB |

Sizes are for the build of 2026-10-10 at `9a19876f`, built with the `dist` profile; the download is the `.tar.gz`, packed as the `rust` workflow packs it (MB is a million bytes). Intel Macs, Linux on ARM, and Windows have no prebuilt binary yet; build from source there (on Windows, inside WSL).

The `rust` workflow (`.github/workflows/rust.yml`) builds all three on every `bench-v*` tag, once formatting, clippy, the tests, the Docker smoke tests and the install script's lint pass, and fails when the tag is not `bench-v` plus the version in `Cargo.toml`. It checks that each binary runs and names its commit, installs it with `install.sh`, and uploads `bench-<version>-<target>.tar.gz` and its `.sha256` as workflow artifacts; it publishes no release itself. A maintainer makes the release from those artifacts: `gh run download <run-id>` fetches them, one folder per target, and `gh release create <tag> --verify-tag --latest=false` with the six files inside (each target's `.tar.gz` and `.sha256`) publishes them: a release candidate with `--prerelease` added, a stable release without it. Neither is marked latest, so the repository's latest release stays the Python package's. A run started by hand (Actions, then rust, then Run workflow) builds the same artifacts from any branch; to try one, download it with `gh run download <run-id> -n bench-<version>-<target>` and run `sh install.sh --from bench-<version>-<target>.tar.gz`. Tags of the form `v<version>` start the Python package's release (`public-release.yml`, which publishes to PyPI), not this workflow; `install.sh` reads `BENCH_VERSION=0.8.1` as the tag `bench-v0.8.1`.

## Upgrading

Install the new version over the old one; job folders from either read the same. What a newer bench may change on its first runs:

- **Daytona builds each image once more** when the new version changes the layers bench adds to every image (see [Images between runs](./README.md#images-between-runs)); each trial's `image.log` says whether its image was built now or kept.
- **Reward files are read as BenchFlow 0.8 reads them.** A `reward.json` with no `reward` key (and no `reward.txt`) is unscored (`R302`) unless `BENCHFLOW_REWARD_LENIENT=1` is set, and a task's `[verifier] reward_range` is honored.
- **A lost sandbox is rerun once when the backend cannot say what stopped it.** If the same unexplained loss comes back in the next attempt, the trial is kept as it is (`rerun_capped` in its problem) and counted as not passed; `bench run --rerun errored` runs it again.

## Uninstall

```bash
rm ~/.local/bin/bench           # or $BENCH_INSTALL_DIR/bench, if you set it
cargo uninstall benchflow-cli   # instead, if you installed with cargo install
```

That removes the program. `bench` also keeps caches and crash logs in `~/.cache/benchflow` (or `$BENCHFLOW_CACHE_DIR`), which BenchFlow 0.8 uses too; delete it if you use neither. Job folders (`jobs/`) hold your results and stay where they are.

The images `bench` built also stay in Docker or Podman: BenchFlow keeps each task's image, and each agent's install on top of it, so the next run starts without building (the image cache; see [Images between runs](./README.md#images-between-runs)). While `bench` is installed it bounds that cache itself; once it is gone, nothing does. Before you uninstall, `bench cache clean --only images --remove` removes those of this `BENCHFLOW_OWNER` scope, except ones a container uses or a run used in the last 30 minutes. Every one carries the label `ai.benchflow.owner`, so you can also list them and remove them all (use `podman` the same way):

```bash
docker image ls --filter label=ai.benchflow.owner
docker image rm $(docker image ls -q --filter label=ai.benchflow.owner)
```

Daytona builds its images in your Daytona account, not on your machine.
