# Release Channels

BenchFlow uses two PyPI release channels with the same package name:

- **Public** releases are stable versions such as `0.6.0`. They are created by
  pushing a matching `v<version>` tag.
- **Internal preview** releases are development versions such as
  `0.6.1.dev123`. They are created automatically after the `test` workflow
  passes for a push to `main`.

Current release state:

- `0.6.0` is in **release-candidate** testing. It is **not on PyPI yet** — the
  newest PyPI build is still on the `0.5.x` line. The release candidates are
  published as GitHub prereleases (`0.6.0-rc.*`) only.
- The public PyPI `0.6.0` / tag `v0.6.0` step has **not happened yet**; the
  commands below that pin `benchflow==0.6.0` from PyPI start working only after
  that tag is cut.
- After the public tag is cut, `main` should move to `0.6.1.dev0`, which
  publishes internal preview builds as `0.6.1.dev<N>` after CI passes.

## Release-Candidate Install (current path)

While `0.6.0` is RC, install the newest `0.6.0-rc.*` wheel from the
[GitHub releases page](https://github.com/benchflow-ai/benchflow/releases) —
open it, pick the newest `0.6.0-rc.*` prerelease, and install its `.whl` asset:

```bash
uv tool install --prerelease allow \
  'benchflow @ https://github.com/benchflow-ai/benchflow/releases/download/0.6.0-rc.6/benchflow-0.6.0rc6-py3-none-any.whl'
```

The URL pins `0.6.0-rc.6` (newest at time of writing); use a later
`0.6.0-rc.*` tag and filename if one exists. Confirm with `bench --version`.

## Install and Upgrade Commands (once 0.6.0 ships to PyPI)

Use the public channel by default. Opt into internal preview only when you want
the newest build from `main` before the next public tag. The commands in this
section that pin `benchflow==0.6.0` from PyPI only resolve **after** the public
`v0.6.0` tag is cut — until then, use the Release-Candidate install above.

Public Python package users:

```bash
pip install --upgrade benchflow
```

Public `uv`-managed CLI users:

```bash
uv tool install --prerelease allow --upgrade 'benchflow==0.6.0'
```

The exact `benchflow==0.6.0` pin keeps `uv` on the public release while
`--prerelease allow` permits the release-candidate LiteLLM dependency used by
this package line.

Internal preview Python package users:

```bash
pip install --pre --upgrade benchflow
```

Internal preview `uv`-managed CLI users:

```bash
uv tool install --prerelease allow --upgrade benchflow
```

The preview CLI command intentionally omits the exact public pin, so `uv`
selects the latest `0.6.1.dev<N>` package. If a machine was previously
installed with `pip install --user` or another non-`uv tool` method, the command
can fail with `Executables already exist: bench, benchflow`. In that case,
rerun the same command with `--force`:

```bash
uv tool install --prerelease allow --upgrade --force benchflow
```

For downstream projects that use `uv`, keep public dependencies pinned unless
the project intentionally tracks preview builds:

```bash
uv add --prerelease allow 'benchflow==0.6.0'
uv lock --upgrade-package benchflow --prerelease allow
```

To lock the latest internal preview instead:

```bash
uv add --prerelease allow benchflow
uv lock --upgrade-package benchflow --prerelease allow
```

## Version Model

`pyproject.toml` on `main` should track the next public version as `.dev0`.
For example, after publishing `0.6.0`, bump `main` to:

```toml
version = "0.6.1.dev0"
```

The internal preview workflow rewrites that version only inside the CI build,
using the successful `test` workflow run number:

```text
0.6.1.dev0 in git -> 0.6.1.dev123 on PyPI
```

This keeps public and internal preview ordering correct: `0.6.1.dev123` is a
preview of the future `0.6.1`, while `0.6.0` remains the public release users
get by default.

If `main` temporarily contains a final public version during the release flow,
the internal preview workflow skips publishing and lets the tag-driven public
workflow handle that commit.

## Publishing Flow

Internal preview:

1. Merge a PR to `main`.
2. `.github/workflows/test.yml` runs.
3. `.github/workflows/internal-preview-release.yml` publishes to PyPI only if
   the tested `main` commit passed.

Public release:

1. Update `pyproject.toml` from the next `.dev0` version to the final public
   version, for example `0.6.1.dev0 -> 0.6.1`.
2. Merge the release PR to `main`.
3. Push a matching tag, for example `v0.6.1`.
4. `.github/workflows/public-release.yml` validates the tag, publishes to PyPI,
   and creates a GitHub Release. The workflow refuses tags whose commits are
   not contained in `origin/main`.
5. Bump `main` to the next `.dev0`, for example `0.6.2.dev0`.

## One-Time PyPI Setup

Configure PyPI Trusted Publishing for the `benchflow` project. No PyPI token is
stored in GitHub.

Create these PyPI trusted publishers:

| Channel | Repository | Workflow filename | Environment |
| --- | --- | --- | --- |
| Internal preview | `benchflow-ai/benchflow` | `internal-preview-release.yml` | `pypi-internal-preview` |
| Public | `benchflow-ai/benchflow` | `public-release.yml` | `pypi-public` |

Create matching GitHub environments:

- `pypi-internal-preview`: used for automatic preview publishing from `main`.
- `pypi-public`: used for tag-driven public releases.

The workflows build with `uv build --no-sources`, check distributions with
`twine check`, and publish with `uv publish`.
