# Release checklist

This page is for the release owner: the one maintainer who takes a release candidate from its cut to PyPI. [Publishing a release](./release.md) says how the workflows work; this page is the order to do things in, and what to keep. Every item names its evidence: a run link, a hash or a command's output. A release with an item left open does not ship.

## The owner and the cut

1. Name the owner of this release in its tracking issue. The owner pushes nothing until every item below is done.
2. Cut from the newest trunk commit whose nightly run (`nightly.yml`) passed every job, not from the trunk's head. Write the commit and its tree hash down.
3. From the cut, only fixes go in, for 48 hours. Each fix comes with a test that failed before it.

## The version commit

1. Set the version in `pyproject.toml` (`X.Y.ZrcN`, or `X.Y.Z` for a final) and run `uv lock`.
2. Regenerate what is generated from the code, and read each diff:

   ```bash
   uv run python tools/docs_reference.py --write
   uv run python tools/docs_errors.py --write
   uv run python tools/cli_help_golden.py --write
   uv run python tools/export_constraints.py
   ```

3. Update the docs' version lines: the install pins and the constraints URL in `docs/start/install.md`, `prerelease` in `docs/nav.json`, and `docs/start/why-benchflow.md`. `tests/test_docs_dogfood_gaps.py` fails until they all name this version.
4. For a final whose source is an rc's, check that the commit changes the version alone: `uv run python tools/release_promote_check.py <rc commit> HEAD`. Then the rc's gate runs stand for the final.

## Gate links

The release gate is CI on the exact commit. Keep one link per run, each on the version commit (or on a commit `tools/release_promote_check.py` shows differs from it in the version alone):

- `test` on the commit's pull request: every job passed, including `deterministic-integration` (three legs), `grading-matrix`, `grading-matrix-run-defaults`, `window-check`, `oracle-parity`, `public-tree-guard` and `repro-build`. A job GitHub's runners couldn't run is linked from a `gate-dispatch` run on the same commit instead, with the reason;
- the last `nightly` run on the commit, with `g1-public`, `window-full` and, once the gate runners exist, `g1-full` and `matrix-floods`. In the oracle-parity tables, every row is MATCH, FLAKE, ACCEPTED, ACCEPTED_UNSCORED or REFERENCE_NO_SCORE, and each FLAKE and ACCEPTED row is in the release notes with its reason;
- `secret-sweep` on the commit's pull request;
- `release-dryrun` on the acceptance record's pull request;
- `leak-gate-preflight` on the commit, with nothing open;
- `fresh-resolve` and `docs` on the commit.

The acceptance record (schema 2) names the runs `public-release.yml` checks itself: `secret-sweep` and `release-dryrun`, each with its repository, run id and commit. Until the publish can read them (`GATE_RECEIPTS_TOKEN`), its log says the receipts were not checked, and these links are the check.

## The metadata helper image (once per recipe)

On Docker, BenchFlow's metadata firewall helper runs from the image `HELPER_IMAGE` names in `src/benchflow/sandbox/_metadata_helper.py`: a public image pulled by its index digest, checked (the digest among its RepoDigests, the recipe's two labels, the daemon's platform) and run by its image ID. A host that can't pull it builds `HELPER_DOCKERFILE` itself, as before. A recipe whose prebuilt image isn't pinned yet (`HELPER_IMAGE_RECIPE` is not `HELPER_RECIPE`) always builds. Before the version commit, when `HELPER_RECIPE` changed since the last release (or no image is pinned yet), do these steps on fresh VMs with no cloud credentials. The code reaches them as a git bundle.

1. Write the Dockerfile and its recipe from the module, never retyped, and check one against the other. Each command must succeed before the next:
   `python -c 'from benchflow.sandbox._metadata_helper import HELPER_DOCKERFILE as D, HELPER_RECIPE as R; open("helper.Dockerfile", "w").write(D); print(R)' > recipe.txt && R=$(cat recipe.txt) && test "$(sha256sum helper.Dockerfile | cut -c1-16)" = "$R"`
2. Check that the recipe builds: `docker build --no-cache --label benchflow.helper=metadata --label benchflow.helper.recipe=$R - < helper.Dockerfile`, exit 0.
3. Build it for both architectures and push it to `ghcr.io/benchflow-ai/benchflow-metadata-helper`.
   - The arm64 `RUN` needs emulation on an amd64 VM. Install `qemu-user-static` and `binfmt-support` (`/proc/sys/fs/binfmt_misc` then lists `qemu-aarch64`), and use a docker-container builder: `docker buildx create --name multi --driver docker-container --use`.
   - The push takes an organization owner's token with `write:packages`. It reaches the VM over ssh stdin into a 0600 file and goes in by `docker login ghcr.io --password-stdin < <file>; shred -u <file>`, never on a command line. Run `docker logout ghcr.io` after the push whatever it did, then delete the VM.
   - `docker buildx build --builder multi --platform linux/amd64,linux/arm64 --no-cache --provenance=false --sbom=false --label benchflow.helper=metadata --label benchflow.helper.recipe=$R --metadata-file meta.json --tag ghcr.io/benchflow-ai/benchflow-metadata-helper:$R --push - < helper.Dockerfile`
   - The tag is only a label for people; BenchFlow never reads it.
   - The index digest is `containerimage.digest` in `meta.json`. `docker buildx imagetools inspect` must list exactly linux/amd64 and linux/arm64.
4. In the package's settings, set its visibility to public. Publishing a new public image is part of the release's publish approval: name it there.
5. From another fresh VM, not logged in to any registry, pull `ghcr.io/benchflow-ai/benchflow-metadata-helper@<index digest>` once on linux/amd64 and once with `--platform linux/arm64`.
   - Each image's labels must be `benchflow.helper=metadata` and `benchflow.helper.recipe=$R` (`docker image inspect`).
   - Each must run its tools: `docker run --rm --network none [--platform linux/arm64] <ref> sh -c 'iptables --version && iptables-legacy --version && nft --version && python3 --version && openssl version && readlink /bin/readlink'`. The arm64 run needs the emulation from step 3.
6. Commit `HELPER_IMAGE = "ghcr.io/benchflow-ai/benchflow-metadata-helper@<index digest>"` and `HELPER_IMAGE_RECIPE = "<recipe>"` as a commit of their own: the last one before the version commit.
   - Any later change to `HELPER_BASE`, `HELPER_PACKAGES` or `HELPER_DOCKERFILE` starts again at step 1. `tests/test_metadata_helper.py` fails while the two recipes differ.
7. Never delete a pushed helper image: each released version pins its own digest.

## The exact artifacts

1. Build twice and keep the bytes: `repro-build` shows the two builds are equal, and `release-dryrun` prints the wheel's and the sdist's SHA-256.
2. The wheel's `benchflow/_build.json` names the version commit, with `"dirty": false`, and both bfx binaries are the pins in `src/benchflow/_bfx/pins.py` (`tools/check_dist_bfx.py`).
3. Commit `.github/release-acceptance/<version>.json` with those hashes, the receipt's SHA-256 and the gate runs, as the only change on top of the version commit. `release-dryrun` runs on that pull request; `tools/check_release_acceptance.py dist` must pass there.

## Release notes

Write the CHANGELOG section with no tracker keys and no partner names (`tools/public_tree_guard.py` refuses them). It has:

- **migration notes:** each change that can break a user's run, with what to do instead;
- **capability boundaries:** what each sandbox backend can't do in this release, and what is experimental, with the measured number behind it;
- **known failures:** the list below, each with its evidence;
- the costs that were not verified with a live run.

## Docs

1. Every new outcome, flag and error has its `bench explain` page and its row in the regenerated pages.
2. `docs` (journeys and links) passed on the commit.
3. The changelog page is regenerated from the CHANGELOG.

## The push

1. Read public `main`'s head: `git ls-remote https://github.com/benchflow-ai/benchflow.git refs/heads/main`.
2. Build the public-main merge: first parent public `main`, second the acceptance-record commit.
3. Run `secret-sweep` by hand on the exact range (Actions, secret-sweep, "Run workflow", base = public `main`'s head, tip = the merge). Nothing open. GitHub push protection reads every commit of the push, tests included, and a provider-shaped value there refuses the push; build such a value at runtime instead (`tools/keyshape_fix.py`, proven by `tools/foldproof.py`).
4. Check the merge is a fast-forward of public `main`: `git merge-base --is-ancestor <public main> <merge>`.
5. A person pushes `main`, then the tag (lightweight, on the acceptance-record commit). Never force.

## Publish

1. Watch `public-release`. It runs every job of `test.yml` on the tag, then checks the gate receipts, then publishes.
2. If the leak gate stops it, run `leak-gate-propose` on the tag, review each drafted exception (rule, member, value and file hashes), add the ones that hold, with a reason, an approval and an expiry, to the allowlist and its secret, and re-run the failed job. Never tag again for an allowlist change.

## PyPI checks

`post-publish-verify` runs after each publish. Check its run passed:

- PyPI serves the accepted wheel and sdist, byte for byte;
- the tag's `constraints.txt` URL answers;
- a new user's `uv tool install` from PyPI with those constraints gives the version, and its `_build.json` names the accepted commit;
- `bench doctor` passes and the README's oracle step scores 1/1;
- the GitHub release is marked a pre-release exactly when the version is one.

For a pre-release, also check that `uv tool install benchflow`, with no version, still installs the latest final.

## Known failures

Each known failure ships with its sentence in the release notes:

- AgentCore: BenchFlow's integrity check can't run there by default, so a model agent's trial is refused before its turn; an opt-in route exists for agents that run as their own user and has not run live.
- Apple container: `--sandbox apple-container` can't run BenchFlow's integrity check in 0.10.0, so in `audit` and `off` a model agent's trial there is refused before its turn, unscored, with that reason; the oracle still runs. A route for it (a read-only bind of the pinned `bfx`, started by `container exec`) is designed but waits for a Mac with Apple silicon to be tested on.
- Docker below Engine 29 or Compose 2.40: BenchFlow 0.10.0 fully supports Docker Engine 29 and Docker Compose 2.40 or newer. On older versions, such as Docker CE 28.0.4 with Compose 2.38.2 on GitHub Actions' default ubuntu-24.04 image, `bench doctor` warns; every trial under a network policy (allowlist, denylist or no-network) is refused before its turn, naming the versions, since there the task image's ENTRYPOINT, including one that runs as root, starts before the policy holds, and a connection it opens in that moment stays open for the whole trial; tasks with build-only Compose services or a `models:` list fail to resolve, with the minimum named; trials with an open network run. Upgrade with Docker's apt repository (`docker-ce`, `docker-compose-plugin`) before running BenchFlow.

## If it goes wrong

1. A broken release is yanked on PyPI (the project's web page; trusted publishing can't yank), and the fix ships as the next version. Never replace a file under a published version.
2. A failed publish leaves the tag: fix the cause on the trunk and release the next version, or re-run the failed job if the failure was the runner's.
3. Write down what went wrong and the test that would have caught it, before the next cut.
