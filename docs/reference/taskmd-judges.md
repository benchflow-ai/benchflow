# task.md draft 2: how BenchFlow runs it

This page is the reference behind [task.md draft 2](../tasks/taskmd-draft-2.md). It covers parsing, the converted package, grading, and the model judges' sandbox and records. Read the task page first.

## Parsing

BenchFlow parses with task-md's own reference parser, vendored in `src/benchflow/taskmd/_vendor/` at the commit `VENDOR.json` pins. Local changes use neutral diagnostic wording and `import.toml` metadata; the manifest records both the source and installed hashes. Any error the reference parser reports refuses the package. `tests/test_taskmd_differential.py` compares BenchFlow's parse with the tool's recorded output on the fixtures, and with a task-md checkout at hand (`TASKMD_REPO`), with the live tool on every example.

## The converted package

The package becomes a native one under `~/.cache/benchflow/task-formats/taskmd/tasks/<key>/<name>/` (`BENCHFLOW_TASK_FORMAT_CACHE` moves it): `task.md` with native frontmatter and the agent's instruction, `environment/` from `sandbox/` (or a Dockerfile `FROM [sandbox] image`), `verifier/` or `tests/` where `[verifier] mount` places it (`/verifier` or `/tests`), `oracle/` or `solution/` (`/oracle` or `/solution`), and `metadata.taskmd` recording the source, its tree hash, the reference commit, and the draft 2 config. `<name>` is the source folder's name, so trial names and `--include` match it. `<key>` hashes the source tree and the writer, so a changed package gets a new folder.

## Grading

The native package's `verifier/verifier.md` selects the `taskmd` strategy, which follows the spec's order: the saved outputs are copied for the judges before any script runs; `test.sh` runs in the task's working folder, bounded by `[verifier] timeout`; each test-judged criterion is decided from `ctrf.json` by the spec's matching rule; the model judges run; and the rubric is scored (gates, points, levels, continuous scores, `method`, `headline`, `pass_threshold`). A package with no rubric keeps the script-reward contract: `test.sh` writes the reward. When the verifier's network is `"none"`, a shared verifier's container is taken offline before `test.sh` runs if the agent's run left it online, and the trial is not scored if it cannot be.

## Judges (`llm` and `agent` roles)

### Prompt and loop

BenchFlow compiles each session's prompt with the vendored `judge-prompt@1` compiler (`tests/test_taskmd_judge_vectors.py` holds it to the spec's four vectors, byte for byte) and runs `judge-loop@1` over the Anthropic Messages API: the `llm` role gets its evidence in the prompt and `submit_review`; the `agent` role gets `read`, `run`, and `submit_review`, each result in the spec's format, with the tool-call, token, and time budgets and the runtime's budget line. `submit_review` is checked against the assignment (three rejections end a session), each citation is checked against what the judge was shown and labeled `solver`, `solver-executed`, `judge`, or `environment`, `cite = "independent"` turns a pass without an independent citation into a fail, samples combine by `aggregate`, and a failed sample is judged once more.


### Lazy grading

Lazy grading skips model-judged criteria once a test gate fails, except those that carry penalties, and a criterion whose every file the solver never saved fails without a session.


### The runner sandbox

An agent judge's `run` executes in the separate verifier sandbox, a fresh container of the task's image that holds only the saved outputs: BenchFlow removes `/verifier`, `/tests`, and any oracle from it first, hides `/logs/verifier` and `/logs/agent` from the runner's uid (they are mounted from the trial's folders, so they are never deleted), and before every call kills whatever the runner's uid left running, restores the saved outputs (a restore that fails stops the judging and leaves the trial unscored) and a fresh working folder, and writes the command to a file only root can change. It runs that file with `bash --noprofile --norc` as uid 65534, with no network, stdin at `/dev/null`, under the call's timeout, then kills the uid's processes until none is left.


### Dropping root

Root is dropped with the image's own `setpriv` (one that takes `--reuid`), `runuser` or `su`, each only when root owns it in `/usr/bin`, `/bin`, `/usr/sbin` or `/sbin` and nobody else can write it or its folder; an image with none of them is not judged (the trial is unscored, and the fix is to add util-linux), since a judge's command never runs as root. Every shell the runner, and the copy of the saved outputs made for the judges, run as root there is `/bin/sh`, started by its absolute path with the image's loader variables (`LD_PRELOAD`, `LD_LIBRARY_PATH`, `LD_AUDIT` and the like) set empty, and it looks up its tools only in those four folders, never on the image's `PATH`; the Docker and Daytona sandboxes start it that way (on Daytona's direct sandbox the session's own shell starts first, with the sandbox's environment), and an agent judge on another sandbox backend is not judged.


### test.sh in the verifier sandbox

`test.sh` is the task's own: it runs as `[verifier] user` (root unless the task sets another) with the image's environment, so what it runs as root, the submission's code included, can change anything in the verifier sandbox, the runner's tools too, as it can change the tests' results; a task whose `test.sh` runs the submission should set `[verifier] user` to another user.


### uid 65534

The runner needs uid 65534 to itself, so a verifier sandbox where processes of that uid already run (an image whose `USER` is 65534, or `nobody`, runs its main process as that uid) is not judged either; give the image another `USER`. When `sandbox/Dockerfile` ends with `USER 65534` or `USER nobody`, the plan and `bench check` refuse the task before any trial runs; for an image named in `[sandbox] image` this is found only at grading.


### The judges' copy of the outputs

The judges read the saved outputs from a copy BenchFlow makes on the host before `test.sh` runs, outside the folder the verifier sandbox mounts, so nothing `test.sh` does can change what they read, and every `run` call restores the saved outputs from that same copy, so `read` and `run` see the same files (an output BenchFlow does not save, a link, a special file or one over the caps, is in neither); the copy is published to `verifier/taskmd-judge/fs/` when grading ends.


### Redaction

What the judges are sent has secrets taken out first: the judge's own credential and the secret-looking values of BenchFlow's environment are replaced wherever they appear, and so are provider keys with a prefix of their own (`sk-ant-`, `ghp_`, `AKIA` and the like). A scripted seat's output and the tests view also go through the patterns BenchFlow redacts trajectories with; the saved outputs `read` shows, `run`'s output and an `llm` judge's evidence do not, since a program that prints `password=...` must not look like one that masks it. A `read` or `run` result says how many values it replaced, and a file citation resolves against the text `read` showed.


### Records

`verifier/judgments.jsonl` has one line per judge session (each model call, the tools it called with their input, and each result it got back) and one per model-judged criterion (its samples and verdict).

## Differences from the spec

Each verdict records the setup that produced it (`judge.model`, `judge.harness`, `judge.prompt_sha256`, `judge.judge_setup_sha256`, `judge.evidence_sha256`), and each session's record under `verifier/taskmd-judge/` shows what it sent and received. These rules of the spec's `docs/runtime/judging.md` are not yet honored:

- **The proxy.** Judges call the Anthropic Messages API directly with BenchFlow's own credentials, not through `proxy@1` with a seat token, so the proxy's request policy and its token records do not apply.
- **The system message.** With a Claude Code OAuth token, the system message is Claude Code's identity followed by `SYSTEM_PROMPT_1`, since Anthropic accepts such tokens only so; with an API key it is `SYSTEM_PROMPT_1` alone. Each session records which (`system_note`).
- **Wire APIs and roles.** Only the Anthropic Messages API is spoken: a judge model is Anthropic's, or one another provider serves at an Anthropic-compatible endpoint that `BENCHFLOW_TASKMD_JUDGE_BASE_URL` names (the verdict then records `anthropic-compatible <host>` as its provider). The `vlm` and `panel` roles, `effort`, a hosted harness, an agent judge's own `resources` and `services`, and images in an `llm` prompt are refused.
- **Seeds.** The Messages API takes no seed, so each session's `judge-seed@1` seed is computed and recorded, not sent.
- **Runners.** An agent judge's `run` uses one fresh container per session, reset before every call (the uid's processes killed, the saved outputs restored, a fresh working folder and HOME), not a fresh container per call: state a command leaves outside the saved outputs, such as in `/tmp`, can survive to the next call. Runners get a minimal environment (the image's `PATH` without the saved outputs, `HOME`, `LANG`, `PYTHONNOUSERSITE`, and `PYTHONSAFEPATH` only when the task's `[verifier] env` sets it, since it keeps `python3 /app/main.py` from importing a module beside it), not the image's whole `Env`. No access watch runs, so every `run` result is labeled `solver-executed`, which the spec allows. The runner needs `[verifier] isolation = "separate"` and a verifier with no network; otherwise the agent role is refused.
- **Attribution.** A criterion left with fewer than `min_samples` valid samples, or a judge that cannot be reached, leaves the trial unscored (a verifier error) instead of being attributed by judging the reference with the same setup. A missing or malformed `ctrf.json` fails the test criteria without checking whether the script also fails on the reference.
- **Records.** `review.json` is the verifier-only record; the published form (`record = "published"`, rationales and quotes of protected verdicts withheld, HMAC'd prompt hashes) and `behavior-tags.json` are not written. PDF text extraction, notebook cells, and citations of the task's own inputs (outside the saved outputs) are not resolved; such citations are recorded as unverified with the reason. Files the verifier writes for the judges (`/logs/verifier/judge/`) are not served: a verifier that writes them leaves the trial unscored. Injection controls are not built or run.
- **Kept copy.** The saved outputs are copied through the sandbox's own download, after a probe that runs inside the container, rather than read from outside it with no symbolic link followed; links and special files are refused after the copy, and the size caps are checked on the copy.

## Format equivalence

An imported task.toml task should score what the original scores. To check that, 56 task.toml tasks were each run twice on one host, on Docker, with the oracle and with a do-nothing agent: once on BenchFlow's native path over the `task.toml` folder, and once on the `taskmd` format over the draft-2 package `task-md/tools/convert.py import` writes from it. The tasks were Terminal-Bench 2's terminal-mini set and BenchFlow's own task.toml examples and network, verifier-mode and sidecar matrices.

Twenty-six of them ran and scored on both paths, and every score is identical; twenty-two of those separate the oracle from doing nothing (1 and 0 on both paths), and four score 0 for both agents on both paths. Eleven more reach the same partial or failed outcome on both paths, such as a task that ships no oracle. Sixteen are declined by both paths — BenchFlow's native path raises `UnsupportedTaskFeatureError` for task.toml multi-step `steps` and per-step networks, and the `taskmd` format refuses the same task naming the field.

Three disagree, all in the network-policy matrix, and all because the `taskmd` format is deliberately stricter than the native path:

- `dynamic/e-a-diff` and `dynamic/e-a-diff-v-match` give the agent's phase *more* network than the sandbox. The agent works inside the sandbox's container, so this cannot be honored without rebuilding the container mid-run; the native path ignores the override and scores 0, and the `taskmd` format refuses naming `[agent] network`.
- `dynamic/e-v-diff` asks for an offline verifier in an image with no `iptables`. The native path ignores the setting and scores 0; the `taskmd` format reports that it could not take the verifier offline and does not score the trial.

Refusing by name rather than scoring a task whose settings were not honored is the point of the format hook, so these three stand as they are.

## Families

BenchFlow runs the generator for each seed in a fresh container of the task's own image. The container is offline, runs as the host's unprivileged uid, and sees the package read-only at `/package`. BenchFlow fills the instruction's `{{placeholders}}`, bakes the instance's `agent/` files into the image, and puts its `verifier/` files at `verifier/instance/`. The oracle and the verifier get `TASKMD_SEED` and `TASKMD_PARAMS`.

## The reference tools in BenchFlow

`src/benchflow/taskmd/_vendor/` holds `tools/taskmd.py` and `tools/judgeprompt.py`, from task-md at the commit `VENDOR.json` pins, with the manifest's explicit local naming changes; lint and type checks skip them. `tools/rubrics.py` is not vendored, since importing it adds its folder to `sys.path`; its `score()` and `matching_tests()` are ported in `benchflow.taskmd.grading` and compared with the upstream file on a random battery when `TASKMD_REPO` is set. To follow a new task-md commit, copy the two files again, review the local naming changes and update `VENDOR.json`'s source and installed hashes, refresh `tests/fixtures/taskmd/` (its `golden/` from the new `tools/taskmd.py json`), and run the tests with `TASKMD_REPO` pointing at a checkout of that commit.

## Next steps

- [task.md draft 2](../tasks/taskmd-draft-2.md): write, check and run a package.
- [Separate verifier](../tasks/separate-verifier.md): the sandbox agent judges run in.
