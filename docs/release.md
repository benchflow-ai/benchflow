# Release Channels

BenchFlow uses two PyPI release channels with the same package name:

- **Public** releases are stable versions such as `0.5.0`. They are created by
  pushing a matching `v<version>` tag.
- **Internal preview** releases are development versions such as
  `0.5.1.dev123`. They are created automatically after the `test` workflow
  passes for a push to `main`.

Regular users install public releases with:

```bash
pip install --upgrade benchflow
```

Internal users install the latest preview with:

```bash
pip install --pre --upgrade benchflow
```

## Version Model

`pyproject.toml` on `main` should track the next public version as `.dev0`.
For example, after publishing `0.5.0`, bump `main` to:

```toml
version = "0.5.1.dev0"
```

The internal preview workflow rewrites that version only inside the CI build,
using the successful `test` workflow run number:

```text
0.5.1.dev0 in git -> 0.5.1.dev123 on PyPI
```

This keeps public and internal preview ordering correct: `0.5.1.dev123` is a
preview of the future `0.5.1`, while `0.5.1` remains the public release users
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
   version, for example `0.5.1.dev0 -> 0.5.1`.
2. Merge the release PR to `main`.
3. Push a matching tag, for example `v0.5.1`.
4. `.github/workflows/public-release.yml` validates the tag, publishes to PyPI,
   and creates a GitHub Release.
5. Bump `main` to the next `.dev0`, for example `0.5.2.dev0`.

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
