# Cache

BenchFlow keeps a few things on your machine, and on your Daytona account if you ask, so later trials start faster. `bench cache list` shows them. `bench cache clean` prints what it would remove, and removes it only with `--remove`.

```bash
bench cache list                                     # images and build records (the default)
bench cache list --json                              # machine-readable; notes go to stderr
bench cache clean                                    # what would be removed; nothing is removed
bench cache clean --only images --remove             # remove idle cached layers
bench cache clean --only build-cache --remove        # remove their BuildKit records only
bench cache clean --only pulled-images --remove      # forget pull records of images that are gone
bench cache clean --older-than 30d --remove          # only what was created 30+ days ago
bench cache clean -s daytona --only images --remove  # this owner's Daytona snapshots
```

## What is kept

| Kind (`--only`) | Backend (`-s`) | What |
|---|---|---|
| `images` | `docker` | Cached layers on a task image: `benchflow-layer:agent-<key>` (an agent's install), `benchflow-layer:proxy-<key>` (the model proxy) and `benchflow-layer:verifier-<key>` (a [separate verifier's image](#verifier-image-reuse)). Also pins (`benchflow-layer-base/…`) left by a build that was killed |
| `images` | `daytona` | Content snapshots (`bf-img2-<owner hash>-<content hash>`) of this `BENCHFLOW_DAYTONA_OWNER`, made with `BENCHFLOW_DAYTONA_SNAPSHOTS=content` |
| `build-cache` | `docker` | The BuildKit records of each cached layer's build |
| `pulled-images` | `docker` | Task images BenchFlow pulled itself |

Without `-s`, `bench cache` looks at Docker when it is installed, and at Daytona when `DAYTONA_API_KEY` is set.

`bench cache` does not touch sandboxes or branch snapshots (`bf-snap-*` images on Docker, `bf-snap2-<owner hash>-*` on Daytona). Use `bench sandbox list` and `bench sandbox cleanup` for those; `--only checkpoints` and `--only sandboxes` exit 2.

Daytona snapshots made before 0.8 (`bf-snap-<owner>-*`, `bf-img-*`) carry no owner hash, so no command removes them. Delete them in Daytona when you no longer need them.

For a remote Docker host, point `DOCKER_HOST` or a Docker context at it and use `-s docker`; `-s remote-docker` exits 2.

## What stays when you clean

`clean` removes only what nothing uses, and looks again right before each removal. These stay:

- a layer that a running trial, a container or another tag uses;
- a layer a trial is building or tagging;
- a leftover pin, unless its process on this machine is gone. Older pins and pins of other machines are listed and kept;
- a pulled image, always: another client could re-tag it while it is removed. `--only pulled-images --remove` only forgets records of images already gone; remove images yourself with `docker rmi`;
- a Daytona snapshot that is being built or removed. Only names with this owner's prefix and a content hash are ever selected.

Images are removed by their own tag, without `--force`, so an image built on a layer keeps its bytes. Records are removed with `docker builder prune` and the layer's exact filter. `--only images` never removes records, and `--only build-cache` never removes images.

`--older-than` counts from creation, or from the pull for pulled images. Anything whose creation time is unknown is kept when you give an age.

## One build per layer

Trials that need the same missing layer build it once. The first takes a per-user lock and builds; the others wait, then reuse it.

`config.json` records how, under `layers.<kind>`: `lock: host` and `lock_wait_sec`.

The lock covers one user on one machine. Processes on other machines that share a remote Docker daemon can build the same layer twice.

A trial that cannot lock records `lock: process`. One that waited longer than the build's budget plus 60 seconds records `lock: timeout`. Both install inside the trial and leave the cached layer alone.

The lock folder is `$BENCHFLOW_LAYER_LOCK_DIR`, by default `benchflow-layer-locks-<uid>` in the temp folder. BenchFlow refuses a lock folder other users can write.

## Verifier image reuse

```bash
BENCHFLOW_VERIFIER_LAYER=1 bench run tasks/ --verifier-sandbox separate
```

`BENCHFLOW_VERIFIER_LAYER=1` caches a separate verifier's image on Docker and remote Docker. Only Docker labels change, so grading runs exactly as without it. `--fresh-install` bypasses it. Daytona is not supported.

Only simple Dockerfiles qualify: digest-pinned `FROM` images, local `COPY` sources and plain image settings. Any other Dockerfile builds as usual, and the trial record says why. It works only on the default Linux amd64 Docker builder. The [admission rules](#reference-when-the-verifier-image-cache-applies) are below.

`verifier-sandbox.json` records `verifier_layer`: `cache: image-only`, hit or miss, the key, the time and the lock wait.

| Setting | Default |
|---|---|
| `BENCHFLOW_VERIFIER_LAYER_KEEP` | 64 images |
| `BENCHFLOW_VERIFIER_LAYER_MAX_GB` | 50, task images included |

The cache only speeds up eligible image builds. It never installs tools or copies package state into grading.

## Daytona content snapshots

```bash
export BENCHFLOW_DAYTONA_SNAPSHOTS=content BENCHFLOW_DAYTONA_OWNER=my-team
bench run tasks/ --sandbox daytona
```

BenchFlow saves each task's built image (the task image plus the proxy and agent layers) as a Daytona snapshot. The name is a hash of what the build reads. Later trials with the same inputs start from it, on any runner and from any machine.

The hash covers the recipe, every file the Daytona SDK uploads (path, mode, bytes), the layers, every `FROM` (pinned by digest), CPU, memory, disk, and the Daytona API URL and target.

`sandbox.json`'s `install_layers` records `snapshot` (`created`, `reused`, `waited`, `fallback` or `off`), the reason and `snapshot_name`.

No snapshot is made, and the image builds as usual, for:

- an unpinned `FROM`, an `ADD` or a `COPY` glob;
- a heredoc or line continuation;
- a `RUN --mount` from outside the recipe;
- a symlink or special file in the context;
- `--fresh-install`, a forced build, a branch checkpoint or an explicit snapshot template.

A snapshot that fails to build falls back to the recipe.

BenchFlow copies the build context to a private folder, then hashes and uploads that same copy. Later edits to the task never change what a snapshot name describes.

The hash pins the recipe, not the network. Packages a `RUN` installs are frozen as of the day the snapshot was built.

Snapshots stay on the account until `bench cache clean -s daytona --only images --remove` removes them.

## Reference: when the verifier image cache applies

The cache is used only when every row holds. Otherwise the verifier builds as usual and the trial record says why.

| Check | Required |
|---|---|
| Dockerfile | Digest-pinned external `FROM` images without inherited `ONBUILD`, local `COPY` sources and declarative settings. `RUN`, `ARG`, `ADD`, external `COPY`, build mounts, deferred instructions, mutable base tags and unknown syntax build as usual |
| Platform | The default Linux amd64 builder, with locally resolved Linux amd64 bases. Any nonempty `DOCKER_DEFAULT_PLATFORM`, custom builder or foreign image variant falls back |
| `DOCKER_BUILDKIT` | Unset, empty or true (`1`, `t`, `T`, `true`, `True`, `TRUE`). Context reuse needs it explicitly true |
| `COMPOSE_BAKE` | Unset or false (`0`, `f`, `F`, `false`, `False`, `FALSE`). Context reuse needs it explicitly false |
| `EXPERIMENTAL_BUILDKIT_SOURCE_POLICY` | Unset in both your environment and Compose's |
| Compose dotenv | No `.env` in the working or project folder, and no `COMPOSE_ENV_FILES` |
| Toolchain | Docker 29.1.3 (static release), Compose 2.35.1 and Buildx 0.30.1, Linux amd64, checked by executable SHA-256. Wrappers and other versions disable context caching |
| Docker context | A local Unix endpoint without TLS. Global options limited to `--host`/`-H` and `--context`/`-c` |
| Compose topology | No task override of generated inputs and no custom Compose topology |

The cache key covers the image's content and platform, every verifier input file with its mode, the verifier config, the cache format (format 9 voids earlier entries), and a digest of the resolved Compose invocation. Environment values are never recorded, and the cache never reads Docker configuration files, which can hold credentials.

## Next steps

- Install everything in each trial instead: `--fresh-install`, in [Run a job](../jobs/run.md#configure-a-job).
- Where the model proxy and agent layers come from: [Docker](../sandboxes/docker.md#cached-install-layers).
- Clean up sandboxes and branch snapshots: `bench sandbox cleanup`, in the [CLI reference](../reference/cli.md#bench-sandbox-cleanup).
