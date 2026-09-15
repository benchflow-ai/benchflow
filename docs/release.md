# Release Channels

BenchFlow uses two PyPI release channels with the same package name:

- **Public** releases are stable builds. They are created by pushing a matching
  release tag.
- **Internal preview** releases are development builds from `main`. They are
  created automatically after the `test` and integration workflows pass for a
  push to `main`, as long as `main` is on the next `.dev0` line.

## Install and Upgrade Commands

Use the public channel by default. Opt into internal preview only when you want
the newest build from `main` before the next public tag.

BenchFlow CLI and SDK releases require Python 3.12 or newer. For CLI installs,
prefer `uv tool install --python 3.12 ...` so `uv` provisions a compatible
interpreter. If `uv` uses Python 3.10 or 3.11, it can resolve an old
Python-compatible `benchflow` release that does not provide the `bench` /
`benchflow` executables and fail with `No executables are provided by package
benchflow`.

Public Python package users, inside a Python 3.12+ environment:

```bash
python3.12 -m pip install --upgrade benchflow
```

Public `uv`-managed CLI users:

```bash
uv tool install --python 3.12 --upgrade benchflow
```

If the command reports `Executables already exist: bench, benchflow`, rerun it
with `uv tool install --python 3.12 --upgrade --force benchflow` to replace stale entrypoints
from an older install.

Internal preview Python package users, inside a Python 3.12+ environment:

```bash
python3.12 -m pip install --pre --upgrade benchflow
```

Internal preview `uv`-managed CLI users:

```bash
uv tool install --python 3.12 --prerelease allow --upgrade benchflow
```

The preview CLI command intentionally omits an exact package pin, so `uv`
selects the latest available preview package once the next preview line is open.
If a machine was previously installed with `pip install --user` or another
non-`uv tool` method, the command can fail with `Executables already exist:
bench, benchflow`. In that case, rerun with the forced preview install:

```bash
uv tool install --python 3.12 --prerelease allow --upgrade --force benchflow
```

For downstream projects that use `uv`, keep public dependencies on the default
stable channel unless the project intentionally tracks preview builds:

```bash
uv add benchflow
uv lock --upgrade-package benchflow
```

To lock the latest internal preview instead:

```bash
uv add --prerelease allow benchflow
uv lock --upgrade-package benchflow --prerelease allow
```

## Version Model

`pyproject.toml` on `main` should track the next public version as `.dev0`.
For example, after publishing a public release, bump `main` to the next
development line:

```toml
version = "<next-public-version>.dev0"
```

The internal preview workflow rewrites that version only inside the CI build,
using the successful `test` workflow run number:

```text
<next-public-version>.dev0 in git -> <next-public-version>.dev<run-number> on PyPI
```

This keeps public and internal preview ordering correct: preview builds sort
before their matching future public release, while ordinary users keep getting
the latest stable release by default.

If `main` temporarily contains a final public version during the release flow,
the internal preview workflow skips publishing and lets the tag-driven public
workflow handle that commit.

## Publishing Flow

Internal preview:

1. Merge a PR to `main`.
2. `.github/workflows/test.yml` runs.
3. `.github/workflows/integration-light.yml` runs a real rollout after the
   tested `main` commit passes.
4. The terminal integration gate uploads the exact tested commit and source
   run number as provenance. `.github/workflows/internal-preview-release.yml`
   downloads and validates that artifact from the successful integration run,
   verifies the commit is on `main`, then publishes it.

The integration gate selects the first exposed live LLM provider that can
answer a small probe request, then uses that same provider for the smoke rollout
and agent judge. The workflow maps DeepSeek and GLM credentials from the
`pypi-internal-preview` environment and falls back to GitHub Models through the
workflow token. Keep at least one of those routes working; the L1 job does not
receive Daytona or reviewer credentials.

The GitHub Deployments page for `pypi-internal-preview` can show integration
gate statuses because the integration workflow uses that same GitHub environment
for secrets. Check the workflow name before treating a failed deployment row as
a failed PyPI publish; only `.github/workflows/internal-preview-release.yml`
runs `uv publish`.

Public release:

1. Update `pyproject.toml` from the next `.dev0` version to the final public
   version, and set `CITATION.cff` `version` and `date-released` to that same
   release.
2. Merge the release PR to `main`.
3. Push a matching release tag.
4. `.github/workflows/public-release.yml` validates the tag against
   `pyproject.toml` and `CITATION.cff`, publishes to PyPI, and creates a GitHub
   Release. The workflow refuses tags whose commits are not contained in
   `origin/main`, and tags whose `CITATION.cff` still names an older release.
5. Bump `main` to the next `.dev0`. Leave `CITATION.cff` on the version just
   released: it names the last published release, not the line under
   development.

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

## Zenodo Archiving

Each published GitHub Release is archived on Zenodo through the official
GitHub integration, which snapshots the tag's source tree and mints a version
DOI for it. Zenodo also maintains a concept DOI that always resolves to the
newest archived version; cite the concept DOI, not a version DOI, so the
citation never goes stale.

`.zenodo.json` at the repository root supplies the archive metadata. Without it
Zenodo derives authorship from GitHub contributor statistics, which is not the
project's author list. Its `version` and `publication_date` are deliberately
absent: Zenodo takes both from the tag and the GitHub Release, so pinning them
would go stale on every release.

`CITATION.cff` is the human-facing citation record and names the **last
published** release. On `main` it therefore sits one release behind the `.devN`
version in `pyproject.toml`, which is correct, not drift. The tag-driven
`public-release.yml` validation is what keeps it honest: a tag whose
`CITATION.cff` names a different version fails before anything is built or
published.

One-time setup, which requires **admin** rights on the repository:

1. Sign in to <https://zenodo.org> with the GitHub account, and grant Zenodo the
   GitHub authorization it asks for.
2. Open Zenodo's GitHub page (<https://zenodo.org/account/settings/github/>) and
   flip the switch for `benchflow-ai/benchflow` on. Zenodo installs a release
   webhook; it archives releases published *after* the switch is flipped, so the
   first archived version is the next release, not the current one.
3. After that release, confirm on Zenodo that the record's authors, title, and
   license came from `.zenodo.json` rather than from contributor statistics.
4. Add the concept DOI to `README.md` and to `CITATION.cff` once it exists.

Flip the switch only after a `.zenodo.json` carrying the real author list has
merged to `main`. Zenodo archives the tagged commit, and a published DOI cannot
be withdrawn -- an incomplete author list becomes permanent public metadata.
