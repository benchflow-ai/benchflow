# Publishing a release

This page is for maintainers. To install a stable or preview release, see [Install](../start/install.md).

BenchFlow publishes two kinds of release to PyPI, both under the name `benchflow`:

| Channel | Published | Version | Workflow |
| --- | --- | --- | --- |
| Stable | when a maintainer pushes a `vX.Y.Z` tag | `X.Y.Z` | `public-release.yml` |
| Pre-release | when a maintainer pushes a `vX.Y.ZrcN` (or `aN`, `bN`) tag | `X.Y.ZrcN` | `public-release.yml` |
| Preview | after every push to `main` whose tests and integration check pass | `X.Y.Z.devN` | `internal-preview-release.yml` |

## Version model

`main` always carries the next stable version with `.dev0`, for example `0.9.0.dev0`. After a stable release, bump it to the next one:

```toml
version = "<next-stable-version>.dev0"
```

The preview build replaces `.dev0` with `.devN`, where N is the run number of the `test` workflow that passed. This happens only in CI; git keeps `.dev0`.

```text
<next-stable-version>.dev0 in git -> <next-stable-version>.dev<run-number> on PyPI
```

Previews sort before the stable release they lead to, so `pip install benchflow` still gets the latest stable.

If `main` carries a final or a/b/rc version during a release, the preview workflow skips publishing. The tag-driven stable workflow handles that commit.

A pre-release such as `0.10.0rc2` goes through the same stable steps below, with its own acceptance record. pip installs it only when asked for it (`pip install --pre benchflow` or `benchflow==0.10.0rc2`), and its GitHub release is marked as a pre-release. `.dev` and local versions never publish through `public-release.yml`.

## Preview releases

1. Merge a PR to `main`.
2. `test.yml` runs.
3. When it passes, `integration-light.yml` runs a real rollout on that commit. It saves the commit SHA and the `test` run number as provenance.
4. `internal-preview-release.yml` validates that provenance, checks that the commit is on `main`, builds it and publishes it.

### The integration check's model provider

The integration check probes model providers in order and uses the first that answers a small request. It uses that provider for the smoke rollout and the agent judge.

| Provider | Credentials in the job |
| --- | --- |
| DeepSeek | `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL` |
| GLM | `GLM_API_KEY`, `GLM_BASE_URL`, `GLM_MODEL` |
| GitHub Models | `GITHUB_MODELS_TOKEN`, or the workflow token when that secret is unset |

The keys come from the `pypi-internal-preview` environment (`.github/scripts/select_integration_provider.py` holds the order). Keep at least one working. The job also receives `GEMINI_API_KEY`, but the selector does not probe Gemini. The job gets no Daytona or reviewer keys.

The Deployments page for `pypi-internal-preview` also lists integration runs, because they use the same environment for secrets. A failed row there is not a failed publish unless it comes from `internal-preview-release.yml`. Only that workflow runs `uv publish`.

## Stable releases

1. Change `pyproject.toml` from `X.Y.Z.dev0` to `X.Y.Z`, and merge the PR.
2. Check that the `fresh-resolve` workflow is green on that commit ([below](#the-constraints-file)).
3. Freeze the source commit, build both distributions with the pinned bfx binaries, and complete the release gate ([below](#the-release-gate-on-a-vm)) and installed-artifact checks. Retain the exact wheel and source archive, their SHA-256 hashes, and an acceptance receipt that identifies the tested source, tools, checks and reviews.
4. Commit a reviewed `.github/release-acceptance/X.Y.Z.json` record using the format below. Between its `source_commit` and the release tag, only this acceptance record may change. A product, test, dependency or documentation change requires a new frozen source and applicable acceptance checks. The record is outside both distribution file allowlists, so adding it does not change their contents.
5. Push the tag `vX.Y.Z` on the reviewed acceptance-record commit. `public-release.yml` first runs every job of `test.yml` on the tagged commit, and publishes only when all of them pass. There is no switch that publishes without them: re-run a flaky job from the run's page (the run keeps every attempt), and fix a real failure on `main` and release a new version. It then checks that the tag matches the version and is on `main`, builds both distributions, and requires their exact hashes to match the record before publishing. A missing, pending or stale record stops publication. If the build is not reproducible, retain the failure and resolve it; do not replace accepted hashes with untested ones.
6. Bump `main` to the next `.dev0`.

The acceptance record has this shape; placeholders are deliberately invalid and must be replaced with actual accepted evidence:

```json
{
  "schema_version": 1,
  "version": "X.Y.Z",
  "status": "accepted",
  "source_commit": "<full tested source commit>",
  "acceptance_receipt_sha256": "<SHA-256 of the reviewed acceptance receipt>",
  "artifacts": {
    "benchflow-X.Y.Z-py3-none-any.whl": "<accepted wheel SHA-256>",
    "benchflow-X.Y.Z.tar.gz": "<accepted source archive SHA-256>"
  }
}
```

Keep the receipt and artifact location in the release review. `tools/check_release_acceptance.py` verifies the committed source/artifact binding; it does not execute the gate, validate the receipt's conclusions or grant approval. Run it against the final build before tagging. Both `dist` and the record must contain exactly one wheel and one source archive for this version. The checker also permits the exact regular `.gitignore` marker generated by pinned uv in `dist`; other extra files remain refused. The existing helper-content and distribution checks remain required.

## Which build wrote a record

Every `result.json`, `summary.json` and `run-card.json` names the BenchFlow build that wrote it, as `benchflow_build`: `version`, `commit` (the source commit) and `dirty` (whether that tree had changes git had not committed). A source checkout answers from git. The wheel and the sdist carry the commit their build saw in `benchflow/_build.json`, which `hatch_build.py` writes, because an installed wheel has no `.git`. An install from a git URL falls back to the commit in its `direct_url.json`. Anything else is `null`, never a guess.

The commit leaves out `.github/release-acceptance/`: the tag's commit adds only the acceptance record, so the wheel built at the tag names the accepted source commit and keeps its accepted hash. Between releases, each line carries a `.dev0` version of the next release candidate (`0.10.0rc4.dev0` after `0.10.0rc3`), so `bench --version` also tells a build from the line apart from the published candidate.

## The release gate on a VM

A release candidate's tests run on per-run VMs, and that gate is the release gate, with its receipts in the internal lab repository; the GitHub Actions CI that runs on pull requests into the release lines is in addition to it, never instead.

The gate runs on the rc branch's head after every merge, never on a pull request alone. Two pull requests can each pass their own suite and fail together once both are merged: git merged two cleanly, and the merged tree failed 81 tests because one removed an import the other used. So every leg runs the whole suite on the combined head, with `--rc-branch <rc branch>` naming the branch; the tested commit must be that branch's head (`origin/<rc branch>` as fetched into the checkout, and on GitHub when it answers). The receipt records the branch, its head and every pull request merged since the last rc (the highest version among the accepted records in `.github/release-acceptance/` whose source commit is an ancestor). A run on a pull request branch or a scratch merge may run, but its receipt says it is not a release gate and is never a PASS; `--combine`, which runs where GitHub answers, passes only when every leg gated the same head and GitHub's branch is still there. The static stage also checks the pages generated from the code on that head: the reference pages and the changelog page (`tools/docs_reference.py`), the error index (`tools/docs_errors.py`) and the CLI help golden (`tools/cli_help_golden.py`). A stale page fails the gate, and the receipt names the command that regenerates it. The gate has three legs (`tools/release_gate/legs.json`), each on its own VM, run in parallel:

| Leg | Python | Docker | Stages |
| --- | --- | --- | --- |
| `py3.12` | 3.12 | Ubuntu's | static checks, unit suite, pip-audit, Docker tier, hill-climb test, cost matrix |
| `py3.13` | 3.13 | Ubuntu's | unit suite |
| `docker-ce` | 3.12 | Docker CE | unit suite |

Create each VM as a full Ubuntu 24.04 VM (some tests need `setpriv`, which an unprivileged container refuses), with a `pd-balanced` or faster boot disk of at least 300 GB for `py3.12` (the Docker tier builds about 74 GB of images) and 100 GB for the others, and a service account that may only write to the logs bucket (`--scopes=storage-rw`). On each, from a clean checkout of the candidate whose `origin` names the GitHub repository (tests infer task provenance from it), run `tools/release_gate/bootstrap_vm.sh` (`--docker ce` for the `docker-ce` leg). It installs pinned Docker, Compose and buildx (on the Ubuntu legs the Compose build the direct Docker path is proven on), `sqlite3`, a C compiler and the pinned uv, and adds you to the `docker` group. Log in again: the group applies to new logins, and the gate refuses to run through `sg docker` or `newgrp`, which make `docker` its real group and break the agent entry's tests. Then run the leg, detached, with its one command (`--base-commit` is optional: it runs the unit suite on the base too, so the receipt says which failures are new; every failure fails the gate either way):

```bash
# py3.12: static checks, unit suite, pip-audit, Docker tier, hill-climb test, cost matrix
setsid nohup python3 tools/release_gate/run_gate.py --leg py3.12 --rc-branch <rc branch> \
  --records ~/records/ops/releases/benchflow --secrets ~/cm-secrets \
  --gcs-prefix gs://<bucket>/<path>/py3.12 > ~/gate-leg.out 2>&1 < /dev/null &
# py3.13: unit suite
setsid nohup python3 tools/release_gate/run_gate.py --leg py3.13 --rc-branch <rc branch> \
  --records ~/records/ops/releases/benchflow \
  --gcs-prefix gs://<bucket>/<path>/py3.13 > ~/gate-leg.out 2>&1 < /dev/null &
# docker-ce (after bootstrap_vm.sh --docker ce): unit suite on Docker CE
setsid nohup python3 tools/release_gate/run_gate.py --leg docker-ce --rc-branch <rc branch> \
  --records ~/records/ops/releases/benchflow \
  --gcs-prefix gs://<bucket>/<path>/docker-ce > ~/gate-leg.out 2>&1 < /dev/null &
```

`~/records` is a clone of the internal lab repository whose origin is lab on GitHub (on a VM without GitHub access, clone a bundle of lab, then point `origin` at lab's GitHub URL with `git remote set-url`), holding the waivers and the failure table as committed. The gate refuses any other folder or repository (a `git init` folder could hold a waiver the lead never wrote), and the receipt records lab's origin and the commit it read. `~/gate-leg.out` shows each step; `<work>/FINISHED` holds the leg's exit code (0 PASS, 1 FAIL, 2 refused, 3 INCOMPLETE). Without `--secrets`, give `--skip matrix`: the py3.12 leg then reports INCOMPLETE. A full run takes about 70 minutes per leg for the unit suite on an `e2-standard-8`, and the py3.12 leg's Docker tier and hill-climb test about an hour more.

Release records never go in this repository: its tree goes public. `--records` is the internal lab repository's `ops/releases/benchflow` folder. Each leg writes its receipt there (`receipts/<version>-<commit>-<leg>.md` and `.json`); commit them in lab. When every leg's receipt is in, `run_gate.py --combine --records ...` writes the release receipt. It runs in a lab clone where GitHub answers, with lab's main fetched, and checks each leg's records against lab: the leg read them from a lab commit that is in this clone's history and on GitHub, waivers.json there has the MD5 the leg recorded, and every waiver the leg applied is on lab's main unchanged (the lead merges lab, so a waiver on a branch alone counts for nothing). The release receipt passes only when every leg passed and those checks hold. Only a passing release gate gets the one-line public summary (version, commit, unit pass and fail counts per leg, gate date) for the public release notes; any other verdict gets none.

Each leg runs as GitHub's runners do (uv's own Python, umask 022, the tests' folders under `/tmp`):

1. the unit suite, after pulling the images its tests need (`ubuntu:24.04`: without it `tests/test_direct_docker_container.py` skips whole), and the base commit's (or `--base-failures FILE`) so the receipt names which failures are new. On the Ubuntu legs every test in a must-run file (`legs.json`) must run: a skip fails the leg. Every failure fails the gate: the VM is a fixed environment, so a test that cannot run there says so with a skip reason in the test itself, or the VM is fixed. The one way past a failing test is a waiver the lead writes by name in the records folder's `waivers.json`, for this version (and one leg, when it names one), with a reason and an issue. The gate reads waivers and the failure table only as committed in lab, records that commit, and never writes a waiver. The failure table there (`<version>-failures.json`) records why tests failed on machines that were not prepared; it decides nothing;
2. pip-audit on the lock with every extra (`test.yml` audits three) and on each agent pin file in `src/benchflow/agents/requirements/`, which the lock does not hold, once with PyPI's advisories (`-s pypi`, pip-audit's default) and once with OSV's (`-s osv`): PyPI's missed CVE-2026-118667 on langgraph 1.2.4, which OSV has, and an advisory from either fails the stage. Every file and service is audited, also after one fails, and the receipt lists each with when it was audited. The advisories are read live, so the receipt records which pip-audit ran and when it asked: a pass is dated by that run on the tested head, and the release receipt carries the date (an advisory published later is the next gate's to find);
3. the Docker tier: every test marked `deterministic`, on Docker, with no failure (every skip is listed with its reason), then `tests/test_hillclimb_example_docker.py` alone with `--timeout 2400`, which must pass, not skip;
4. the cost matrix, `tools/cost_matrix/run_matrix.py`, on both sandboxes. Every run must pass. A cell marked `blocked` in `matrix.json` is listed with its reason, as neither a pass nor a failure, beside the matrix's offline evidence and a release-note line saying which costs are unverified live;
5. it scrubs the matrix's secret values from every log and from the receipt, uploads every log to the prefix and checks each copy's MD5. The receipt holds the source commit, the exact commands, exit codes, pass, fail and skip counts, and each log's MD5 and location.

A leg that leaves a stage out (`--skip`), runs a subset (including one sandbox of the matrix) or keeps its logs local reports INCOMPLETE, never PASS. No credential in the VM's environment variables reaches the tests; files in its home folder and its metadata server do, hence the narrow service account.

Before a release reaches the public repository, `public-release.yml` runs `tools/public_tree_guard.py` on the tagged tree: it refuses a `release/` path, a linear.app URL, a key of one of our tracker's teams (`EN`, `ENG`, `BEN3`, `BD`, `CEA` or `OSS`, a hyphen and a number) anywhere in the tree, and in `CHANGELOG.md` and `README.md` any token shaped like a tracker key that is not a public name (`CVE`, `SHA`, `UTF`, ...). `tests/test_public_tree_guard.py` fails on a committed `release/` path or linear.app URL. Run the guard on the candidate before its gate, and clean what it finds (`tools/strip_internal_refs.py --write` rewrites comments and docstrings; it reports the rest), so the gate runs on the tree that is tagged.

## The constraints file

`constraints.txt` at the repository root pins the lock's closure as `name==version` (every extra but `dev`). Users install from it to get exactly what CI tested; each release tag carries it, so no release asset is needed.

After any change to `uv.lock`, regenerate it. `tests/test_constraints_file.py` fails until you do.

```bash
uv run python tools/export_constraints.py
```

The `fresh-resolve` workflow checks the opposite case. It builds the wheel, installs it in a fresh environment with no lock and no constraints, and runs the unit suite against it on Python 3.12 and 3.13.

It runs daily, on every push to `main`, and on pull requests that touch `pyproject.toml`, `uv.lock`, `constraints.txt` or `tools/ci/`. A new upstream release can turn it red with no change here. To run the same check by hand, use `tools/ci/fresh_resolve.sh`.

## One-time PyPI setup

Configure PyPI Trusted Publishing for the `benchflow` project. No PyPI token is stored in GitHub.

Create these PyPI trusted publishers:

| Channel | Repository | Workflow filename | Environment |
| --- | --- | --- | --- |
| Preview | `benchflow-ai/benchflow` | `internal-preview-release.yml` | `pypi-internal-preview` |
| Stable | `benchflow-ai/benchflow` | `public-release.yml` | `pypi-public` |

Create matching GitHub environments:

- `pypi-internal-preview`: automatic preview publishing from `main`.
- `pypi-public`: tag-driven stable releases.

The workflows build with `uv build --no-sources`, check distributions with `twine check`, and publish with `uv publish`.

## Next steps

- [Install](../start/install.md): the stable and preview install commands.
- [What's new in 0.9](../versions/upgrade-from-0.7.md): the changes in this release line.
