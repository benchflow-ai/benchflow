# Error messages

Every message BenchFlow can print that has a fixed text in its source, and what to do about it. Search this page for the words of your message: `…` marks a part filled in at run time. To find out why a trial failed, start with [When a trial fails](../evaluate/failed-trials.md).

These messages come from BenchFlow's source on the 0.10 line. A message 0.9.2 doesn't print has `(0.10)` after it when 0.10.0rc3 prints it, or `(unreleased)` when no release does yet.

<!-- This page is generated from the code by tools/docs_errors.py, so it lists what this version prints. To change it, change the message in the code, then run `uv run python tools/docs_errors.py --write`. -->

| Section | Messages |
|---|---|
| [Causes a run names](#causes-a-run-names) | 48 |
| [`bench doctor` warnings and failures](#bench-doctor-warnings-and-failures) | 22 |
| [Errors that say what to do](#errors-that-say-what-to-do) | 468 |
| [Command errors](#command-errors) | 234 |
| [Problems in a task or config file](#problems-in-a-task-or-config-file) | 82 |
| [Tasks a sandbox can't run](#tasks-a-sandbox-cant-run) | 46 |
| [What the verifier step reports](#what-the-verifier-step-reports) | 104 |
| [Warnings a run logs](#warnings-a-run-logs) | 395 |

## Causes a run names

The end of a run names a cause for each trial that failed without a usable reward, and `bench explain <code>` prints its entry. Whose problem it is says where the fix belongs.

A test script that exits with an error, or runs past `verifier.timeout_sec`, without writing a reward is not filed under `verifier_failure` or `verifier_timeout` in 0.9.1 and 0.10.0rc1, whatever their entries say: the trial is scored 0, a failure, with a `Verifier scored 0:` warning in its `result.json` ([When the test itself fails](../evaluate/failed-trials.md#when-the-test-itself-fails)). Those two causes name a verifier step that ended with no verdict, such as one the pytest plugin guard refused.

| Cause | Code | What happened | Whose problem | What to do |
|---|---|---|---|---|
| `verifier_failure` |  | the verifier crashed. The agent finished, but the verifier step failed before a reward existed, and the trial's error starts 'verifier crashed:'. BenchFlow names a finer cause when it can (its own infrastructure, a dependency install, a full disk); this one is left when it cannot, and usually comes from the task's verifier setup. This unscored final trial counts as a failure in primary metrics (mean reward and pass@k), remaining in their denominator; only scored-only metrics exclude it. A test.sh that itself exits with an error before writing a reward is not this cause either: that is verifier_no_usable_reward. | the task | fix the verifier from its output (the trial's verifier/test-stdout.txt, whose last lines the summary shows), then run the task again |
| `verifier_timeout` | R303 | the verifier timed out. The verifier ran longer than the task's verifier timeout and was stopped before it wrote a reward. Installing test dependencies while grading is the usual cause: install them in the image instead, and raise the timeout only if the tests themselves are slow. | the task | raise the task's verifier timeout (task.md verifier.timeout_sec) |
| `verifier_dep_install` |  | the verifier's dependency install failed. The verifier installed its dependencies while grading, and the install failed: a package index it could not reach, or no version that matched. Grading that depends on the network at grading time fails whenever the network does. | the task | pin or bake the verifier's dependencies into the image; `bench check <task>` |
| `reward_out_of_range` |  | the verifier wrote a reward outside the task's range. The verifier ran and wrote a reward, but the number is outside the range the task declares (0 to 1 unless task.toml's [verifier] reward_range says otherwise), so it is not used: the trial scores 0, a failure in the primary metrics (mean reward and pass@k) and in the scored-only ones alike, and stays in every denominator, with a 'Verifier scored 0:' warning. reward_out_of_range_info in result.json keeps the raw value. Every run of the task fails the same way until the verifier or the declared range is fixed. | the task | fix the verifier, or declare [verifier] reward_range in task.toml if the task means a wider range; reward_out_of_range_info in result.json keeps the raw value |
| `verifier_infra` |  | the verifier's sandbox failed. The sandbox the verifier ran in failed, or a separate verifier sandbox never received the agent's outputs, so nothing was assessed. It says nothing about the task or the agent. | the sandbox | `bench eval resume <job dir>` scores it again |
| `infra_failure` |  | the review or its sandbox failed. The sandbox, or BenchFlow's connection to it, failed during the trial: a lost container, or a trial past the host's hard deadline whose transport or teardown was stuck. It is retried within the run. | the sandbox | run the review again: `bench eval score <the trial's folder> --tasks-root <the tasks folder> --force` |
| `disk_full` |  | the disk that holds the job folder is full, so the reward was lost. The disk that holds the job folder ran out of space, so BenchFlow could not write the trial's files (result.json, summary.json), or the verifier's reward, which the sandbox writes into that folder, came out empty. It says nothing about the task or the agent, and is not retried within the run; once space is freed, a resume runs the trial again. | your setup | free space on that disk, or pass another --jobs-dir; then `bench eval resume <job dir>` |
| `verifier_other` |  | the verifier failed. The verifier failed in a way BenchFlow has no finer cause for. The trial's verifier/ folder holds what it printed. | the task | the trial's verifier/ folder has the output |
| `verifier_plugin_trust` |  | verifier plugin trust: test.sh installs a pytest plugin where the agent could write. The task's test.sh installs a pytest plugin (usually pytest-json-ctrf) into a Python environment the agent could write: the workspace, the agent's home, /tmp, /var/tmp, /logs or /testbed. The pytest plugin guard refuses to load it, since the agent could have planted it, so every run of the task scores 0, with a warning that starts 'Verifier scored 0: pytest plugin guard not confirmed'. Bake the plugin into the image, or pin its release in [verifier] pytest_plugin_releases and install it as root into a Python environment's site-packages, where the guard trusts it by the bytes PyPI publishes. | the task | `bench check <task>` |
| `verifier_plugin_unrecorded` |  | verifier plugin trust: the plugin guard refused a pytest plugin in a place the agent could write, and with no record of the agent's files it cannot tell who put it there. The pytest plugin guard refused a plugin in a place the agent could write, and the record of the agent's files was missing or incomplete, so it cannot tell whether the agent put the plugin there. The trial is a failure in the scores. When test.sh installs the plugin, pin its release in [verifier] pytest_plugin_releases or bake it into the image. | the task | the trial's verifier/ folder names the files; when test.sh installs the plugin, install it into the image: `bench tasks check <task>` |
| `verifier_plugin_unproven` |  | verifier plugin trust: test.sh installs a pytest plugin BenchFlow cannot check, and the agent changed files that could have chosen it. The task's test.sh installed a pytest plugin after the agent stopped (with uv, uvx or pip, into the workspace or the verifier's uv state), and the agent had changed files in its trees, any of which could have chosen what was installed. The pytest plugin guard trusts such an install only as a release it can check byte for byte, so it refused the plugin. The trial is a failure in the scores, so a planted input gains nothing, but it is the task's problem: pin the plugin's release in [verifier] pytest_plugin_releases, or bake it into the image. | the task | pin the plugin's release in [verifier] pytest_plugin_releases or install it into the image: `bench tasks check <task>` |
| `verifier_plugin_unlisted` |  | verifier plugin trust: test.sh installed a pytest-json-ctrf release missing from the table of published releases. The task's test.sh installed a release of pytest-json-ctrf, the plugin BenchFlow vouches for, that was missing from the table of published releases hardening built: this machine could not read the release list from PyPI, or PyPI published the release after hardening read it. The pytest plugin guard refused it, and the trial is a failure in the scores, since the agent can also leave inputs that install a release of any name. Check that this machine reaches pypi.org and run the task again. | the sandbox | check that this machine reaches pypi.org, then run the task again; the trial's verifier/ folder names the release |
| `verifier_plugin_refused` |  | verifier plugin trust: the plugin guard refused a pytest plugin in a place the agent could write. The pytest plugin guard refused a plugin in a place the agent could write that the agent left there, or changed, after it started. The trial counts as a failure. The trial's verifier/ folder names the plugin and why. | the agent | the trial's verifier/ folder names the plugin and why; when test.sh installs it, `bench tasks check <task>` |
| `verifier_no_usable_reward` |  | the verifier wrote no usable reward (a failure in every score) (0.10). The final verify left no reward BenchFlow could use: no reward file, an empty one, one it could not read, or one without a number. That is the verifier's error, not a wrong answer, and it is never retried. The trial still counts as a failure in every score, as a plugin guard refusal does, because the reward files are evidence the agent can reach. The trial's verifier/ folder has test.sh's output. | the task | the trial's verifier/ folder has test.sh's output; fix test.sh so it always writes one number to /logs/verifier/reward.txt, then run the task again |
| `no_verdict` |  | no verdict was recorded (an assessment still pending, or an interrupted trial). The trial has neither a reward nor an error: a reviewer's assessment is still pending, or the run was interrupted before the trial recorded an outcome. | the sandbox | `bench eval score <job dir>` finishes pending reviews |
| `kept_service_freezer` |  | the sandbox cannot freeze the task's kept services. The task declares services that must survive the agent turn and be frozen until verification, and this kind of sandbox cannot freeze them for a non-root agent (native Daytona without Compose, for one). The trial has no score; where the task alone decides it, the refusal comes before any sandbox is made. | your setup | run with --sandbox docker or --sandbox remote-docker; Daytona also supports a Compose environment |
| `runtime_sealing` |  | the protected runtime could not be sealed. BenchFlow refused to run root's steps using runtime files whose ownership or paths could not be protected from the agent. The refusal preserves isolation; changing the task image is required. | the task | fix the image's runtime paths and ownership named in result.json; keep them root-owned and outside the agent's writable workspace |
| `guard_timeout` |  | the sandbox guard timed out or did not report. The setup guard did not finish or return its report. The trial has no score. Check the setup evidence before retrying; this is not an agent task failure. | the sandbox | the error in result.json and the sandbox setup log say where the guard stopped; `bench eval resume <job dir>` runs the trial again |
| `invalid_provider_response` |  | the provider returned an invalid response. The proxy observed an upstream HTTP 5xx with an invalid JSON body. Lifecycle records preserve the actual status, byte counts and elapsed upstream seconds without copying prompt content. | the model provider | check trajectory/provider_requests.jsonl for upstream status and timing; retry or use another provider endpoint |
| `provider_auth` | R212 | the model provider refused the credential. The model provider answered with an authentication error (HTTP 401, or a key it called invalid): the key or login is wrong, revoked, expired, or for another provider. The trial is not retried. | your setup | `bench doctor` checks each login and key |
| `provider_rate_limit` |  | the model provider rate-limited the run. The model provider refused requests for going over its rate limit (HTTP 429). The trial is not retried within the run. | the model provider | lower --concurrency, then `bench eval resume <job dir>` |
| `provider_rejected` |  | the model provider rejected the request (often a context too long). The model provider rejected a request as invalid (HTTP 400), most often a conversation longer than the model's context window. The trial is not retried: the same request would be rejected again. | the agent | read the error in the trial's result.json; a model with a larger context, or a shorter prompt |
| `api_error` |  | the model provider's API failed. The model provider's API failed: a server error, or a service overloaded or unavailable. api_error_info in result.json has the status and whether it was transient; transient ones are retried within the run. | the model provider | `bench eval resume <job dir>` reruns it; api_error_info in result.json has the status |
| `suspected_api_error` |  | the agent did nothing (no tokens, no tool calls). The agent finished without using a token or calling a tool: it most likely never reached its model (a wrong model id, an expired login, a provider that refused it without saying so). | your setup | check the model id and the login with `bench doctor`, then `bench eval resume <job dir>` |
| `install_failure` |  | the agent's install failed. Installing the agent in the sandbox failed, usually on the network to npm, PyPI or the agent's download site, or on a base image that lacks a tool the install needs. It is retried within the run. | the sandbox | `bench doctor` checks the network to npm and Node.js; the trial's agent/install-stdout.txt has the log |
| `pipe_closed` |  | the connection to the agent closed. The connection between BenchFlow and the agent process closed: the Daytona terminal dropped, its websocket closed, or the local or SSH transport ended. It is retried within the run. | the sandbox | `bench eval resume <job dir>` reruns it; transport_error_info in result.json says why |
| `idle_timeout` | R203 | the agent went silent and the idle watchdog stopped it. The agent sent no tool call, message or thought for longer than the idle timeout, and the idle watchdog stopped it. A hung tool call or a stalled model connection are the usual causes. When the verifier can still score the workspace, the trial is scored instead. | the agent | raise --agent-idle-timeout if the agent thinks for long; idle_timeout_info in result.json has the details |
| `timeout` |  | the agent ran out of time. The agent ran past its time budget (the task's agent timeout times --timeout-multiplier, or the host's hard deadline) and was stopped. When the verifier can still score the workspace, the trial is scored instead; this cause is a timeout it could not score. | the agent | raise the task's agent timeout (task.md agent.timeout_sec) or --timeout-multiplier |
| `trial_budget` |  | the trial spent its token or turn budget and the agent was stopped (0.10). The trial spent its token or turn budget (a turn is one model call): BenchFlow's model proxy refused the agent's next call and BenchFlow stopped the agent. Like a timeout, the verifier then grades what the agent left, so the trial is usually scored, 0 unless its tests passed, and it is never retried; this cause is a budget stop it could not score. result.json's trial_budget says which measure stopped it, what the trial used and its subagents' share. | the agent | raise --token-budget or --turn-budget (or the task's [agent] budget) if the task needs more; trial_budget in result.json has what it used |
| `sandbox_setup` |  | the sandbox did not start. The sandbox did not start: the network or the registry failed while the image was pulled or built, the backend refused it (quota, capacity, a daemon that is down), or the task's Dockerfile names a path that is missing. The run summary prints the cause; error in result.json has the whole message (sandbox_startup_info adds the sandbox's id and state when the backend gave them). | the sandbox | `bench doctor`; error in result.json has the whole message |
| `image_build` |  | the task's image did not build. The task's image did not build, for a reason another attempt would repeat: a RUN step exited nonzero with no network failure in its output, the base image in FROM is not found or not public, or the Dockerfile does not parse. The summary names the Dockerfile line and BuildKit's reason; the trial is not retried. A network or registry failure during the build, a RUN step's included, is sandbox_setup instead and is retried; a registry that refused the login is registry_auth. | the task | fix that line in the task's environment/Dockerfile; `docker build <task>/environment` prints the whole build log |
| `acp_error` | R205 | the agent reported an error. The agent answered a request with an error of its own over the Agent Client Protocol; the message is the agent's. For Claude Code, 'The connection to Claude was lost' means its own process or its API stream ended, not BenchFlow's connection to the sandbox. | the agent | the agent's log (agent/&lt;agent&gt;.txt in the trial folder) has its side |
| `registry_auth` |  | the registry refused this machine's login for the base image. Docker could not pull the task's base image because the registry refused this machine's login: an expired `docker login`, a login without access to that repository, or the registry's token service saying no. It is your setup, not the task's Dockerfile, and it is not retried. Docker Hub gives the same answer for a repository that does not exist, so check the FROM line's name too. | your setup | check `docker login <registry>` on this machine, and that the FROM line names an image that login can read |
| `task_config` |  | a task file does not load. A file of the task does not load: a typo or an unknown key in task.toml, a task.md whose front matter does not parse, or a verifier document or rubric that is not valid. The message names the file and, for a typo, the key it meant. Every run of the task fails the same way until the file is fixed, so it is not retried; `bench check <task>` reports the same problem without running it. | the task | `bench check <task>` names the problem; fix it, then run the task again |
| `unsupported_task_feature` | R401 | the task asks for something this sandbox backend cannot do. The task asks for something the chosen sandbox backend cannot provide, so the trial is refused before it starts. The message names each feature and the backend. | your setup | the error names what; run the task with `--sandbox docker`, or change what it asks for |
| `agent_integration:agent_auth` |  | the agent integration broke: its login or billing failed. The agent started, then did nothing, because its own login or billing failed (an expired subscription, no credit left). | your setup | `bench doctor` |
| `agent_integration:agent_model` |  | the agent integration broke: it does not offer the model. The agent does not offer the model the run asked for. The error lists the models it does offer. | your setup | pick a model the agent lists in the error |
| `agent_integration:agent_effort` |  | the agent integration broke: it does not offer the model at that reasoning effort (0.10). The agent offers the model, but not at the reasoning effort the run asked for (codex-acp lists one model id per effort a model takes). The run is refused before the agent starts rather than run at another effort; the error lists the efforts it does offer. | your setup | pick an effort the error lists for the model, or a model that takes the effort |
| `agent_integration:agent_budget` |  | the agent integration broke: its model calls go around BenchFlow's model proxy, which enforces the trial's token and turn budget (0.10). The trial has a token or turn budget (--token-budget, --turn-budget or the task's [agent] token_budget and turn_budget), but the agent calls its model around BenchFlow's model proxy: a subscription login, an endpoint of its own, or an agent that routes its own model. The proxy is what counts and enforces a trial's budget, so the trial is refused before the agent starts rather than run without one; the error says which route it took. | your setup | run it with an API key, so its calls go through the proxy, or drop --token-budget/--turn-budget (or the task's [agent] budget) |
| `agent_integration:agent_model_catalog` |  | the agent integration broke: its model catalog has no entry for the model, so it ran on fallback tools (0.10). The agent's engine takes each model's tools from its model catalog, and the catalog has no entry for this model, so the agent ran on fallback tools (for Codex: no apply_patch, no code mode). That is another agent than the model is used with, so the trial is not scored. Codex's warning ('Model metadata for ... not found') or the tools its first request offered show it; result.json's tool_surface lists them. | your setup | pick a model the agent's pinned engine catalogues, or an agent release whose engine lists the model |
| `agent_integration:agent_install` |  | the agent integration broke: its install or runtime is broken. The agent started, but its install or runtime in the sandbox is broken: a missing program, or a dependency that does not load. | the sandbox | `bench doctor --agent-start <agent>` |
| `agent_integration:unknown` |  | the agent integration broke: it did nothing useful. The agent did nothing useful, and BenchFlow has no finer cause for it; integration_failure_info in result.json has what it saw. | the agent | integration_failure_info in result.json |
| `usage_limit` | R213 | usage limit. The login's subscription usage limit was reached (its 5-hour or 7-day window). The job starts no more trials on it; the summary lists the trials it did not start, the login, the window and the reset. | your setup | switch to another login or wait for the reset, then `bench eval resume <job dir>`; `bench doctor` shows each login's headroom |
| `pipe_closed:exit` | R204 | the agent process exited. The agent process exited during its turn; the message has its exit code, and says whether the sandbox's cgroup counted an OOM kill (an exit is called a memory kill only with that count, never from its code alone). Its log has its last words. | the agent | the agent's log (agent/&lt;agent&gt;.txt) has its last words; `bench eval resume <job dir>` reruns it |
| `provider_unavailable` |  | the model provider failed with a server error (0.10). The agent stopped with an error after the model provider answered a call with a server error (HTTP 500, 502, 503, 504 or 529): the provider's failure, not the agent's or the model's. It is retried within the run; agent_outcome in result.json counts the failed calls. | the sandbox | `bench eval resume <job dir>` reruns it; agent_outcome in result.json counts the calls that failed |
| `kept_service_freezer:engine` |  | the Docker Engine cannot freeze the task's kept services. The task's kept services are frozen in a cgroup between the agent's turn and verification, and this Docker Engine cannot make one for a non-root agent: it is older than 28, rootless or remaps user namespaces. No agent code ran; the trial has no score. | your setup | use Docker Engine 28 or later with cgroup v2, not rootless and without user-namespace remapping (`docker info`) |
| `kept_service_freezer:probe` |  | the sandbox's cgroups cannot freeze the task's kept services. The task's kept services are frozen in a cgroup between the agent's turn and verification, and the freezer check inside this sandbox failed (no cgroup v2, or cgroups it cannot write). No agent code ran; the trial has no score. The check's own reason is in the error. | your setup | run it where Docker uses cgroup v2 with writable cgroups (`docker info` shows Cgroup Version: 2); the check's reason is in result.json |
| `other` |  | other error. An error none of the other pages covers. If it repeats on other tasks, it may be a bug in BenchFlow: report it at `https://github.com/benchflow-ai/benchflow/issues` with the trial folder. | the sandbox | the trial's result.json and the run's log have the details |

## `bench doctor` warnings and failures

`bench doctor` checks this machine and each agent's login. A `warn` row lets runs go ahead; a `fail` row stops the runs it names. Each row prints a `fix:` line.

| Check | Status | Message | Fix |
|---|---|---|---|
| agents: agent credentials | warn | `no model agent can run: … (the oracle and nop controls still can)` | Log in with `codex login` or `claude setup-token`, or export an API key; see `https://github.com/benchflow-ai/benchflow/blob/main/docs/agents/auth.md` |
| agents: claude usage | warn | `… accepted: no window reported (one 8-token … request)` | The … window is nearly spent; a long run may stop on the usage limit |
| agents: claude usage | warn | `… is out of usage: its … window is spent until … (…)` | Claude runs on this login stop at once with a usage-limit error until …: use another login (CLAUDE_CODE_OAUTH_TOKEN from `claude setup-token` on another account) or wait |
| agents: claude usage | warn | `… was refused (HTTP …): the login is invalid or expired` | Run `claude setup-token` and export `CLAUDE_CODE_OAUTH_TOKEN=<token>`, or export `ANTHROPIC_API_KEY`. |
| agents: claude usage | warn | `…: HTTP … and no usage windows (one 8-token … request)` | Run `claude setup-token` and export `CLAUDE_CODE_OAUTH_TOKEN=<token>`, or export `ANTHROPIC_API_KEY`. |
| agents: claude usage | warn | `…: could not check its usage (…)` | Check that … is reachable; HTTPS_PROXY is honored if set |
| network: … | fail | `unreachable — …: …` | Check VPN/proxy/firewall; HTTPS_PROXY is honored if set |
| proxy: litellm | warn | `LiteLLM CLI not found, so API-key and provider-routed runs on … cannot start their model proxy (subscription logins do not need it)…` | Reinstall BenchFlow so its pinned … is installed next to it (`uv tool install --reinstall benchflow`) |
| runtime: bench on PATH | warn | `bench runs … (…), not this install (…)` | benchflow always runs this install; for bench to run it too, run the install command you installed it with again, with --force (benchflow==… and the same extras and constraints), which replaces …: `https://www.benchflow.ai/docs/benchflow/getting-started/install#which-bench-runs` |
| runtime: bench on PATH | warn | `no bench on PATH; this one is …` | Add … to PATH (for a uv tool install: uv tool update-shell) |
| runtime: bench on PATH | warn | `…; it hides …` | Run a hidden one by its full path, or put its folder first on PATH: `https://www.benchflow.ai/docs/benchflow/getting-started/install#which-bench-runs` |
| runtime: bfx | warn | `bfx is not installed: this BenchFlow was built without its pinned bfx binaries, so Docker trials run root's steps after the agent without it, as with BENCHFLOW_BFX_DELIVERY=off (those steps then need a Python in the image)` | install a release wheel, or build bfx with `make bfx` before building the wheel (docs/getting-started/install.md) |
| runtime: python | fail | `Python … is older than 3.12` | `uv tool install --python 3.12 --upgrade benchflow` |
| runtime: uv | warn | `uv not found on PATH (needed to install, upgrade or add extras)` | Install uv: `https://docs.astral.sh/uv/getting-started/installation/` |
| sandbox: colima | warn | ``docker context points at Colima profile …, but `colima list` does not show it`` | `colima start …` |
| sandbox: colima | warn | `profile … is running with only …` | Restart the profile with more memory: `colima stop <profile> && colima start <profile> --memory 8 --cpu 4`. |
| sandbox: daytona | warn | `SDK …, DAYTONA_API_KEY set (…); the key check failed without rejecting the key (…); sandbox creation reports the error` | Let the run go on; if sandbox creation then fails, check the network path to the Daytona API and `DAYTONA_API_URL`, then run `bench doctor --sandbox daytona`. |
| sandbox: docker buildx | warn | `buildx plugin not found, so Compose prints "Docker Compose requires buildx plugin to be installed" and uses the legacy builder, which cannot build Dockerfiles that use BuildKit features (RUN --mount, heredocs)` | Install the buildx plugin: `sudo apt-get install docker-buildx-plugin` on Debian or Ubuntu; with Colima, `brew install docker-buildx` and add its folder to `cliPluginsExtraDirs` in `~/.docker/config.json`. |
| sandbox: docker disk | warn | `only … free in …; task images and agent installs need at least …` | `docker system df` shows what uses it; remove images you no longer need with `docker image rm` |
| sandbox: docker memory | warn | `Docker VM has …; image builds plus agent installs need at least …` | Give the Docker VM at least 8 GiB: raise it in Docker Desktop's Resources settings, or run `colima stop <profile> && colima start <profile> --memory 8 --cpu 4`. |
| sandbox: docker sandboxes | warn | `… container… left by … ended run…` | `docker rm -f …` removes them; the next `bench run` on this daemon removes only the stopped ones |
| versions: … | warn | `registry installs …, not the built-in …; …` | Remove the agent manifest override, or re-run the policy conformance fixtures for the new version |

## Errors that say what to do

These errors stop a command or a run. Each row is one message an error class is raised with, and what to do: the step it prints as its `next:` line, or, where it prints none, the fix written here. What each class covers is below the table.

| Error | Message | What to do |
|---|---|---|
| `AgentManifestError` | `cannot read manifest …: …` | Fix the TOML syntax error the message quotes in `manifest.toml`, or make that path exist and be readable. |
| `AgentManifestError` | `contract_version must look like MAJOR.MINOR[.PATCH]; got …` | Set `contract_version` to a quoted string of the form MAJOR.MINOR or MAJOR.MINOR.PATCH, such as `'1.1'`. |
| `AgentManifestError` | `contract_version … declares major …; this loader speaks contract ….x` | Declare a 1.x `contract_version`, such as `'1.1'`, and match that contract's fields; this BenchFlow loads only contract major 1. |
| `AgentManifestError` | `duplicate agent name … (second declaration at …)` | Give each `manifest.toml` under the agents folder its own `name`, or remove one of the two manifests. |
| `AgentManifestError` | `manifest … is missing required …` | Add the named key; every `manifest.toml` needs `contract_version`, `name`, `install_cmd` and `launch_cmd`. |
| `AgentManifestError` | `manifest …: aliases must be an array of strings` | Write `aliases` as a TOML array of strings, such as `aliases = ['my-agent']`, or remove it. |
| `AgentManifestError` | `manifest …: model_route must be one of …; got …` (0.10) | Set `model_route` to `proxy` or `agent`, or remove it to get `proxy`. |
| `AgentManifestError` | `manifest …: name must be a string` | Set `name` to a quoted string, such as `name = 'my-agent'`. |
| `AgentManifestError` | `manifest …: protocol … is not supported; the contract is acp-only (…). Non-ACP platforms must ship an ACP-over-stdio shim and declare protocol='acp'.` | Set `protocol = 'acp'` or remove it, and make `launch_cmd` start an ACP-over-stdio shim for your agent. |
| `AgentManifestError` | `manifest …: … needs contract_version ….… or later; it declares …` (0.10) | Raise `contract_version` to `'1.1'` or later, or remove the key it names (`model_route` needs 1.1). |
| `AssessmentError` | (built at run time) | Follow the `next:` line the refusal prints; nothing was appended, and `bench assess show <sidecar>` prints the sidecar's state. |
| `BranchPlanError` | `--checkpoint …: no such checkpoint in …` | Pass `--checkpoint prompt:N` for an N the trial's `checkpoints.json` lists, or drop it to use the last kept one. |
| `BranchPlanError` | `--checkpoint …: that snapshot is …, not kept` | Pick a `prompt:N` checkpoint whose status in `checkpoints.json` is `kept`, or rerun the trial with `--checkpoints` to keep a new one. |
| `BranchPlanError` | `--checkpoint-after-prompt must be 0 or more` | Pass `--checkpoint-after-prompt` 0 (branch before the first prompt) or a positive number of prompts. |
| `BranchPlanError` | `--checkpoint-after-prompt … is after the last of … --prompt` | Lower `--checkpoint-after-prompt` to at most the number of `--prompt` options, or pass more `--prompt` options. |
| `BranchPlanError` | `--child …: cannot read …: …` | Fix the `prompt-file=` path in that `--child`; a relative path is read from the current folder. |
| `BranchPlanError` | `--child …: expected label=NAME[,parent=LABEL][,prompt=TEXT\|,prompt-file=PATH]` | Write each `--child` as `label=NAME`, then optional `parent=LABEL`, then `prompt=TEXT` or `prompt-file=PATH`, joined by commas; no other keys. |
| `BranchPlanError` | `--child …: give prompt or prompt-file, not both` | Keep only one of `prompt=` and `prompt-file=` in that `--child`. |
| `BranchPlanError` | `--child …: label=NAME is required (1-… characters)` | Start the `--child` with `label=NAME`, a name of 1 to 200 characters. |
| `BranchPlanError` | `--child …: the prompt is empty` | Give the child a non-blank `prompt=` or a non-empty `prompt-file=`, or drop both to run the parent's remaining prompts. |
| `BranchPlanError` | `--child …: … given twice` | Give each key once per `--child`; for another child, add another `--child` option. |
| `BranchPlanError` | `--child-retries must be 0 or more` | Pass `--child-retries` 0 or more (`child_retries=` in Python). |
| `BranchPlanError` | `--child: unknown parent …` | Set `parent=` to the label of another `--child` in the same command, or drop it to fork from the checkpoint. |
| `BranchPlanError` | `--concurrency must be at least 1` | Pass `--concurrency` 1 or more (`concurrency=` in Python). |
| `BranchPlanError` | `--concurrency, --isolate-children and nested children use the sandbox layer only (--snapshot-layers sandbox)` | Use `--snapshot-layers sandbox`, or drop `--concurrency`, `--isolate-children` and `parent=` children to also snapshot the environment. |
| `BranchPlanError` | `--fork needs --from-checkpoint` | Add `--from-checkpoint <trial folder>`, or drop `--fork`/`--checkpoint`. |
| `BranchPlanError` | `--fork …: no such fork in …` | Pass a fork id from the trial's `tree.json` (or `prompt:N` for an automatic checkpoint), or drop `--fork` to use the last kept one. |
| `BranchPlanError` | `--from-checkpoint branches at the kept checkpoint; --checkpoint-after-prompt does not apply` | Drop `--checkpoint-after-prompt` (or pass 0) with `--from-checkpoint`; it branches at the kept checkpoint. |
| `BranchPlanError` | `--from-checkpoint branches exactly one task` | Point `--tasks-dir` at the checkpoint's own task folder, so only that task is selected. |
| `BranchPlanError` | `--from-checkpoint reuses a sandbox snapshot: use --snapshot-layers sandbox` | Pass `--snapshot-layers sandbox` (the default) with `--from-checkpoint`. |
| `BranchPlanError` | `--from-checkpoint …: fork … also captured declared environment state, which cannot be re-imported; only sandbox-only checkpoints can be branched again.` | Pass `--fork` with a fork that kept only the sandbox layer, or rerun with `--snapshot-layers sandbox --retain-snapshots` to keep a reusable snapshot. |
| `BranchPlanError` | `--from-checkpoint …: no kept sandbox snapshot. A trial can be branched again when it was made with --retain-snapshots (bench branch) or --checkpoints (bench run or bench branch), and its snapshot has not been deleted since.` | Branch from a trial run with `--retain-snapshots` (`bench branch`) or `--checkpoints` (`bench run`, `bench branch`) whose snapshot still exists. |
| `BranchPlanError` | `--from-checkpoint …: not a trial folder (needs config.json): …` | Pass `--from-checkpoint` a trial folder, the one inside the job folder that holds `config.json`, not the job folder itself. |
| `BranchPlanError` | `--parent is continue or discard` | Pass `--parent continue` or `--parent discard`. |
| `BranchPlanError` | `--resume-session needs an agent conversation before the fork: an agent (not the oracle) and --checkpoint-after-prompt 1 or more` | Drop `--resume-session`, or run a model agent (not the oracle) with `--checkpoint-after-prompt 1` or more. |
| `BranchPlanError` | `--resume-session with --from-checkpoint needs a checkpoint that recorded its agent session id (automatic --checkpoints do; --retain-snapshots forks and oracle runs do not); this one did not` | Branch from an automatic checkpoint (`--checkpoint prompt:N` of a run made with `--checkpoints`), which records the session id, or drop `--resume-session`. |
| `BranchPlanError` | `--snapshot-layers …: use sandbox, environment or both (comma-separated)` | Pass `--snapshot-layers sandbox`, `environment` or `sandbox,environment`. |
| `BranchPlanError` | `--tasks-dir is required, e.g. bench branch --tasks-dir tasks/my-task --agent claude-agent-acp --model claude-sonnet-5 --sandbox daytona --prompt "Write draft.txt containing: Hello world" --prompt @instruction --checkpoint-after-prompt 1 --child "label=baseline" --child "label=hint,prompt=Rename draft.txt to hello.txt."` | Pass `--tasks-dir` with a task folder or a folder of tasks, as in the example the error prints. |
| `BranchPlanError` | `--tasks-dir … has no task named … (the checkpoint's task)` | Point `--tasks-dir` at the checkpoint's task folder, or at a folder holding it under the same name. |
| `BranchPlanError` | `Not a directory: …` | Pass `--tasks-dir` an existing folder: a task folder or a folder of tasks. |
| `BranchPlanError` | `a nested fork needs at least two children under …` | Add a second `--child` with `parent=<label>` under that child, or drop `parent=` from the one you have. |
| `BranchPlanError` | `agent= is required` | Pass `agent=` to `branch()` or `abranch()`, or a `from_checkpoint=` trial whose `config.json` records its agent. |
| `BranchPlanError` | `checkpoint= (or fork=) needs from_checkpoint=` | Pass `from_checkpoint=` with the trial folder too, or drop `checkpoint=`/`fork=`. |
| `BranchPlanError` | `child labels must be unique, got …` | Give every `--child` a different `label=`. |
| `BranchPlanError` | `child …: the prompt is empty (use None for the parent's remaining prompts)` | Give that child a non-blank prompt, or `None` to run the parent's remaining prompts. |
| `BranchPlanError` | `give at least two top-level --child options` | Pass two or more `--child` options without `parent=` (two or more `children` in Python). |
| `BranchPlanError` | `give checkpoint= or fork=, not both` | Pass either `checkpoint=` or `fork=`; both name the same kept snapshot. |
| `BranchPlanError` | `no task selected` | Point `--tasks-dir` at a task folder or a folder of tasks, and check `--include` against the task folder names. |
| `BranchPlanError` | `task_path is required (it defaults to the task of from_checkpoint= when that trial recorded it)` | Pass `task_path` (the task folder) to `branch()`; the `from_checkpoint=` trial recorded none. |
| `BranchPlanError` | `task_path is required: the checkpoint's task … is not a directory here` | Pass `task_path` with the task folder's path on this machine; the recorded path does not exist here. |
| `BranchPlanError` | `the kept checkpoint is a … snapshot; run with --sandbox …` | Pass `--sandbox` with the provider the message names, or leave `--sandbox` out to use the checkpoint's. |
| `BranchPlanError` | `the oracle has one step (solve.sh): --checkpoint-after-prompt must be 0 or 1` | Set `--checkpoint-after-prompt` to 0 or 1 for the oracle. |
| `BranchPlanError` | `the oracle takes no --prompt` | Drop `--prompt` for the oracle, or pass `--agent` with a model agent. |
| `BranchPlanError` | `the oracle takes no prompts: every oracle child runs the task's solve.sh, so drop prompt= from --child` | Remove `prompt=` and `prompt-file=` from every `--child`, or branch a model agent instead of the oracle. |
| `ConfigKeyError` | (built at run time) | change or drop each setting named; what a YAML job config runs is in `https://github.com/benchflow-ai/benchflow/blob/main/docs/scale/configs.md` |
| `ConfigKeyError` | (built at run time) | fix the key; a run config's keys are listed in `https://github.com/benchflow-ai/benchflow/blob/main/docs/scale/configs.md` |
| `Conflict` | `consumer … already recorded using revision …` (0.10) | Don't record it again: this consumer's use of that revision is already logged; `bench assess show <sidecar> --events` lists the `label.consumed` event. |
| `Conflict` | `idempotency key … is already used by event …` (0.10) | Don't edit the sidecar log by hand; restore `assessment-events.v1.jsonl` from your copy, or start a new assessment in a new sidecar. |
| `Conflict` | `idempotency key … was used by event … (…) with a different payload; nothing was appended` (0.10) | Rerun with the original inputs to get the first event back; for a genuinely different write, pass a new `--idempotency-key` (for a changed vote, use `bench assess correct`). |
| `Conflict` | `revision … was already withdrawn (…); a later withdrawal may only move queued_compensation to compensated` (0.10) | Leave the revision withdrawn; only a `queued_compensation` withdrawal may be followed by `bench assess withdraw --disposition compensated`. |
| `Conflict` | `the assessment already has collector assignment …` (0.10) | a restarted collector must return that same assignment |
| `Conflict` | `the sidecar already holds assessment …` (0.10) | create a new assessment in a new sidecar directory |
| `Conflict` | `this assignment already has an answer; replay it with its own idempotency key, or change it with review.corrected` (0.10) | Change the answer with `bench assess correct <sidecar> --supersedes <event id>` and the new choice; `bench assess show <sidecar> --events` lists the answer's id. |
| `Conflict` | `… already holds assessment …` (0.10) | use a new sidecar directory for another assessment |
| `Conflict` | `… vote … (rater …, …): its vote_id … (the file's, or one made from the pair or item, the aspect, the rater and rated_at) is already recorded with other content` (0.10) | to change an answer, import it with a later rated_at and without that vote_id (or with a new one): a rater's latest vote on a pair or item is the one counted; nothing was added |
| `CorruptSidecar` | `the last line (from byte …) has no newline: a torn append; the log is left exactly as found` (0.10) | move the sidecar aside and keep it as evidence; nothing reads past a torn line |
| `CorruptSidecar` | `the log does not contain event …: it was truncated or replaced` (0.10) | Check that `--expect-event` is the event SHA-256 your receipt holds; if it is, restore the sidecar from a kept copy, as this log was cut or replaced. |
| `CorruptSidecar` | `… (…) is not allowed: …` (0.10) | Don't edit the sidecar log by hand; restore `assessment-events.v1.jsonl` from your copy, or start a new assessment in a new sidecar. |
| `CorruptSidecar` | `… is a symbolic link` (0.10) | Replace the symbolic link `assessment-events.v1.jsonl` in the sidecar with the real log file; sidecars refuse links. |
| `CorruptSidecar` | `… is not in RFC 8785 canonical form` (0.10) | Don't edit the sidecar log by hand; restore `assessment-events.v1.jsonl` from your copy, or start a new assessment in a new sidecar. |
| `CorruptSidecar` | `…: event_seq is …, expected …` (0.10) | Don't edit the sidecar log by hand; restore `assessment-events.v1.jsonl` from your copy, or start a new assessment in a new sidecar. |
| `CorruptSidecar` | `…: event_sha256 does not match the event` (0.10) | Don't edit the sidecar log by hand; restore `assessment-events.v1.jsonl` from your copy, or start a new assessment in a new sidecar. |
| `CorruptSidecar` | `…: expected_previous_event_sha256 differs from the event it followed` (0.10) | Don't edit the sidecar log by hand; restore `assessment-events.v1.jsonl` from your copy, or start a new assessment in a new sidecar. |
| `CorruptSidecar` | `…: previous_event_sha256 breaks the hash chain` (0.10) | Don't edit the sidecar log by hand; restore `assessment-events.v1.jsonl` from your copy, or start a new assessment in a new sidecar. |
| `DatasetDigestMismatchError` | `Dataset … content does not match its registry digests: …` | If you edited the cached tasks under `.cache/datasets-v2/`, undo the edits or remove that repository's folders there and rerun; otherwise tell the registry's maintainer. |
| `DatasetResolutionError` | `Could not read …: …` | Check that the `--registry` URL answers without an HTTP error or the file is readable; for BenchFlow's registry, check this machine's network and rerun. |
| `DatasetResolutionError` | `Dataset … could not be looked up in BenchFlow's registry: …` | check this machine's network, or pick a dataset that BenchFlow's registry lists (bench hub list --provider benchflow), or name a registry that has it with --registry &lt;URL or file&gt; |
| `DatasetResolutionError` | `Dataset … could not be pulled from the Hub at …: …` (0.10) | the Hub's stored copy is damaged, so tell its admin; `bench hub show … --hub …` shows what the Hub holds |
| `DatasetResolutionError` | `Dataset … could not be read from the Hub at …: …` (0.10) | check the name@version with `bench hub show` |
| `DatasetResolutionError` | `Dataset … has a malformed digest for …: …` | Write that task's `digest` as the string `bench tasks digest <task>` prints, or remove it to run the task unchecked. |
| `DatasetResolutionError` | `Dataset … has a malformed task entry (expected an object with …): …` | Make each entry under the dataset's `tasks` an object with at least `git_url` and `path`. |
| `DatasetResolutionError` | `Dataset … has no tasks in the registry` | Pick a version that has tasks (`bench hub list --provider benchflow`), or add a `tasks` list to the entry in your `--registry` file. |
| `DatasetResolutionError` | `Dataset … has two tasks named …` | Give every task in the entry a `path` whose last folder name differs; tasks are named by it. |
| `DatasetResolutionError` | `Dataset … not found in BenchFlow's registry…` | pick a dataset that BenchFlow's registry lists (bench hub list --provider benchflow), or name a registry that has it with --registry &lt;URL or file&gt; |
| `DatasetResolutionError` | `Dataset … not found in registry (available: …)` | check this machine's network, or pick a dataset that BenchFlow's registry lists (bench hub list --provider benchflow), or name a registry that has it with --registry &lt;URL or file&gt; |
| `DatasetResolutionError` | `Dataset … pins git_commit_id …, which is neither a full commit id nor HEAD` | Set `git_commit_id` to a full 40- or 64-character commit id, or to `HEAD` to follow the default branch. |
| `DatasetResolutionError` | `Dataset … spans … git snapshots — only single-snapshot datasets are supported` | Give every task in the entry the same `git_url` and `git_commit_id`, or split the dataset into one entry per snapshot. |
| `DatasetResolutionError` | `Dataset … tasks live under … parent directories — only a single tasks directory is supported` | Move the tasks under one folder so every `path` has the same parent, or split the dataset into one entry per folder. |
| `DatasetResolutionError` | `Invalid dataset spec … — expected <name>@<version> (e.g. skillsbench@1.1)` | Pass the dataset as `<name>@<version>`, such as `-d skillsbench@1.1`; `bench hub list --provider benchflow` lists the datasets. |
| `DatasetResolutionError` | `Snapshot for … resolved to … but the registry pins …` | Remove that repository's folders under `.cache/datasets-v2/<org>/` and rerun; if it recurs, report it at `https://github.com/benchflow-ai/benchflow/issues`. |
| `DatasetResolutionError` | `Unsupported git_url … — expected a github.com repository` | Set the task's `git_url` to a github.com repository, as `https://github.com/<org>/<repo>` or `git@github.com:<org>/<repo>.git`. |
| `DatasetResolutionError` | `… is not a list of dataset entries` | Make the registry file's top level a JSON array of dataset entries (objects with `name`, `version` and `tasks`). |
| `DatasetResolutionError` | `… is not valid JSON: …` | Fix the JSON error the message points to in your `--registry` file; if BenchFlow's own registry fails, report it at `https://github.com/benchflow-ai/benchflow/issues`. |
| `DiskFullError` | (built at run time) | free space on that disk, or pass another --jobs-dir |
| `EmptyTaskSelectionError` | (built at run time) | Run `bench check <folder>` to see the tasks it finds and why any is refused, then check `--include` and `--exclude` against their names. |
| `EnvironmentRegistryError` | `--state JSON must be a mapping with a "name", e.g. {"name": "env0", "tools": ["gmail", "gcal"]}` | Give `--state` a JSON object with a `name` key (and an optional `tools` list), or a `name@version` spec such as `env0@prod`. |
| `EnvironmentRegistryError` | `--state tools … not in environment … (available: …)` | List in `tools` only service names the environment declares (the message lists them), or drop `tools` to start every service. |
| `EnvironmentRegistryError` | `environment registry … (from $BENCHFLOW_ENV_REGISTRY) is not a directory` | Set `BENCHFLOW_ENV_REGISTRY` to an existing folder of environment manifests, or unset it to use the built-in registry. |
| `EnvironmentRegistryError` | `environment … not found in … (looked for …@…{.toml,.yaml,.yml}; available: …)` | Use a spec from the available list (built in: `env0@prod`, `env0@outage`), or add `<name>@<version>.toml` to the registry folder. |
| `EnvironmentRegistryError` | `invalid --state JSON: …` | Fix the JSON in `--state` and quote it for your shell, or pass a `name@version` spec or a manifest path instead. |
| `EnvironmentRegistryError` | `invalid environment spec … — expected <name> or <name>@<version>` | Pass a spec like `env0` or `env0@prod` (letters, digits, dots, hyphens, underscores), or a manifest file path. |
| `EnvironmentRegistryError` | `no environment registry available: the built-in registry shipped with benchflow is missing and $BENCHFLOW_ENV_REGISTRY is unset; set $BENCHFLOW_ENV_REGISTRY or pass --environment-manifest a manifest file path instead of a name@version spec` | Set `BENCHFLOW_ENV_REGISTRY` to a folder of environment manifests, or pass `--environment-manifest` a manifest file path instead of a `name@version` spec. |
| `EnvironmentRegistryError` | `no versions of environment … found in … (available: …)` | Use a name from the available list, or add `<name>.toml` or `<name>@<version>.toml` to the registry folder. |
| `EvalPlanError` | `--base-image-override must be a non-empty image reference` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--build-concurrency must be >= 1 (got …)` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--canonical-jobs-dir requires --canonical-selection-out` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--canonical-selection-out requires --canonicalize` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--canonicalize must be 'none' or 'one-healthy-per-task'` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--codex-apps-policy is not supported for hosted source environments` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--codex-apps-policy must be disabled or inherit` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--concurrency must be >= 1 (got …)` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--config not found: …` | pass the path of a run config YAML file, relative to this folder or absolute |
| `EvalPlanError` | `--context-root not found: …` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--eval-results-model requires --eval-results-dataset and --eval-results-task` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--expected-tasks must be >= 1` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--fail-on …: choose from …` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--fail-under is a pass rate between 0 and 1` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--fresh and --job-name are not supported with --worker-concurrency` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--harness is for BenchFlow agent rollouts; --source-env runs its own harness` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--hf-prefix requires --publish-hf or --publish-bucket` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--ignore-bench-version requires --dataset` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--loop-strategy drives the prompt loop and conflicts with multiple --prompt values` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--loop-strategy is not supported with --skill-mode self-gen` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--matrix needs a task folder, a task named by repository or --dataset` (0.10) | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--matrix not found: …` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--max-cost-usd/--max-sandbox-seconds/--max-tokens/--max-rollouts are not supported with --source-env (vf-eval runs the rollouts)` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--max-cost-usd/--max-sandbox-seconds/--max-tokens/--max-rollouts cap one job and are not supported with --worker-concurrency (each worker would get the whole budget)` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--n-tasks must be at least 1` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--registry requires --dataset` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--restart-changed restarts a named job's earlier trials; … starts new jobs, which have none` (0.10) | drop --restart-changed for a new job, or drop --fresh and name the job to restart with --job-name |
| `EvalPlanError` | `--restart-changed restarts a named job's earlier trials; … starts new jobs, which have none` (0.10) | drop --restart-changed: each shard runs a job of its own |
| `EvalPlanError` | `--retry-attempts must be >= 0` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--retry-concurrency must be >= 1` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--retry-policy must be 'default' or 'unscored-only'` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--reviewer-* options require local tasks; --source-env owns its scoring` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--seeds cannot be combined with --matrix or --trials` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--seeds cannot be combined with --worker-concurrency` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--seeds requires --tasks-dir or --source-repo` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--tasks-dir not found: …` | pass a local task folder; a task on GitHub goes in --source-repo and --source-path, or `bench run org/repo/path` |
| `EvalPlanError` | `--timeout-multiplier must be positive` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--token-budget and --turn-budget are not supported with --source-env: vf-eval runs the rollouts, and BenchFlow's model proxy, which enforces a trial budget, does not see their calls` (0.10) | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--token-budget and --turn-budget cannot be enforced on …: …, and BenchFlow's model proxy is what counts a trial's tokens and turns. No job was created.` (0.10) | run it with an API key, so its calls go through the proxy, or drop the budget |
| `EvalPlanError` | `--trials > 1 needs a task folder, a task named by repository or --dataset; a --config file repeats tasks with a job config's n_attempts, and a --source-env run with --source-env-rollouts-per-example` (0.10) | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--trials must be >= 1` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--verifier-sandbox: …` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--warm-pool must be from 0 to --concurrency (…) (got …)` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--worker-concurrency is supported for --tasks-dir and --source-repo batch runs` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--worker-retries must be >= 0` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `--worker-start-stagger-sec must be >= 0` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `Choose only one source: --config, --tasks-dir, --source-repo, --source-env, or --dataset` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `Could not load --environment-manifest …: …` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `Invalid --agent-idle-timeout …: …` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `Invalid --config-override: …` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `Invalid --harness …: …` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `Invalid --loop-strategy …: k must be between 1 and 10 (got …)` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `Invalid --loop-strategy …: …` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `Invalid --reasoning-effort …: …` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `Invalid --sandbox …: choose …` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `Invalid --seeds …: …` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `Invalid --skill-mode …: choose no-skill, with-skill, or self-gen` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `Invalid --source-repo …; expected 'org/repo' (e.g. benchflow-ai/skillsbench)` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `Invalid --state: …` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `Invalid usage tracking config: …` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `Missing optional dependency for 'modal' sandbox. Install it with ….` | run that install command, then `bench doctor` checks this machine |
| `EvalPlanError` | `Missing optional dependency for … sandbox. Install it with ….` | run that install command, then `bench doctor` checks this machine; a Daytona run also needs DAYTONA_API_KEY |
| `EvalPlanError` | `Unsupported eval agent protocol: …` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `give --fresh or --job-name, not both` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `script: does not use a model harness or agent options; drop --harness and --agent-env, or choose a model agent` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `script: runs only on …: its runner needs a trusted root transport the … backend does not have` (0.10) | use --sandbox docker, or run the logic as an ACP agent (docs/agents/custom-agent.md) |
| `EvalPlanError` | ``… (`bench agents` shows the registered agents; to run a command as the agent, give it with its arguments, e.g. 'myagent --acp')`` | pass -a with a name `bench agents` lists |
| `EvalPlanError` | `… No job was created.` | pass -a with a name `bench agents` lists |
| `EvalPlanError` | `… … not supported with --source-env: the hosted environment scores the run with its own verifier, which BenchFlow cannot move into a separate verifier sandbox. Run it without …, or run the tasks with --tasks-dir or --dataset.` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvalPlanError` | `… … not supported with --source-env: vf-eval scores the run as one mean reward, with no trials to count. The run exits 1 when it errors; its reward is in <jobs-dir>/hosted-env/<run>/result.json.` | correct the flag the message names; `bench run --help-all` lists every flag |
| `EvidenceRejected` | `an imported vote is written by its collector (actor role collector)` (0.10) | Import votes only with `bench assess import-vote`; if the sidecar log was edited, restore it from your copy or start a new assessment. |
| `EvidenceRejected` | `answer names assignment …; this assessment's is …` (0.10) | Record answers only with `bench assess import-vote`, `correct` or `retract`; if the sidecar log was edited, restore it from your copy. |
| `EvidenceRejected` | `answer names reviewer …; the assignment's is …` (0.10) | Check who voted in the collector: only the assignment's own reviewer binds, so have that reviewer vote, or start a new assessment with a fresh assignment. |
| `EvidenceRejected` | `assignment … has no vote in the collector yet` (0.10) | Wait until the reviewer votes in the collector, then rerun `bench assess import-vote` with a database copy made after the vote. |
| `EvidenceRejected` | `both references name the same attempt` (0.10) | Pass two different attempts: two `--attempt` reference files made from different rollouts or outputs. |
| `EvidenceRejected` | `cannot read the collector database: …` (0.10) | Pass `--collector-db` an intact SQLite copy of the collector's `workspace.sqlite3`, made with its `cli.py backup`. |
| `EvidenceRejected` | `collector artifact … (…) lists … files; the attempt's output has …` (0.10) | Bind an assignment whose artifact holds exactly the attempt's `--output` files, or hash the output the collector shows with `bench assess attempt-ref` and start a new assessment. |
| `EvidenceRejected` | `collector artifact … (…): its files are not the attempt's output files (path or hash differs)` (0.10) | Bind an assignment whose artifact holds exactly the attempt's `--output` files, or hash the output the collector shows with `bench assess attempt-ref` and start a new assessment. |
| `EvidenceRejected` | `collector artifact … (…): its output hash does not match its own preview and file hashes` (0.10) | Check that `--collector-db` is an intact `cli.py backup` copy; if it is, the artifact record is inconsistent, so have the collector issue a new assignment. |
| `EvidenceRejected` | `collector assignment … is marked invalid (its evidence changed)` (0.10) | Have the collector issue a new assignment for this pair, then run `bench assess assign` with its `--assignment` id. |
| `EvidenceRejected` | `collector database … does not exist` (0.10) | Pass `--collector-db` the path of the collector's `workspace.sqlite3`, preferably a `cli.py backup` copy. |
| `EvidenceRejected` | `collector source … is missing under …` (0.10) | Point `--collector-code` at the collector's code folder (`design-bench/prototype`) checked out at the commit the judge protocol pins. |
| `EvidenceRejected` | `collector source … is not a regular file` (0.10) | Point `--collector-code` at a checkout where that source is a regular file, not a link or folder, at the pinned commit. |
| `EvidenceRejected` | `collector … … is not a JSON object` (0.10) | Use an intact `cli.py backup` copy as `--collector-db`; if the record is still malformed, report it to the collector's maintainers. |
| `EvidenceRejected` | `collector … … is not readable JSON` (0.10) | Use an intact `cli.py backup` copy as `--collector-db`; if the record is still unreadable, report it to the collector's maintainers. |
| `EvidenceRejected` | `imported_from must name the collector and its vote id` (0.10) | Import votes only with `bench assess import-vote`; if the sidecar log was edited, restore it from your copy or start a new assessment. |
| `EvidenceRejected` | `no release observation for …; every execution resource must be observed released before a human waits` (0.10) | finished_at or a missing log line is not a release; observe each runtime |
| `EvidenceRejected` | `output directory … holds no files` (0.10) | Pass `--output` a folder that holds the attempt's output files, or the output file itself. |
| `EvidenceRejected` | `output member … is a symbolic link` (0.10) | Replace the link inside the `--output` folder with the file it points to; links are refused. |
| `EvidenceRejected` | `output member … is not a regular file` (0.10) | Remove the socket, pipe or device file from the `--output` folder; only regular files and folders are hashed. |
| `EvidenceRejected` | `output … is missing` (0.10) | Pass `--output` an existing file or folder, or move the output back to the path the attempt reference records. |
| `EvidenceRejected` | `output … is neither a file nor a directory` (0.10) | Pass `--output` the real file or folder, not a symbolic link or special file. |
| `EvidenceRejected` | `payload.collector does not name the pinned collector build` (0.10) | Bind assignments only with `bench assess assign`; if the sidecar log was edited, restore it from your copy or start a new assessment. |
| `EvidenceRejected` | `payload.collector.assignment_id is not the exported assignment` (0.10) | Bind assignments only with `bench assess assign`; if the sidecar log was edited, restore it from your copy or start a new assessment. |
| `EvidenceRejected` | `presentation must be A = the collector's left, B = its right` (0.10) | Bind assignments only with `bench assess assign`; if the sidecar log was edited, restore it from your copy or start a new assessment. |
| `EvidenceRejected` | `released_at is earlier than an observation it relies on` (0.10) | Correct the `observed_at` times in the `--proof` file; none may be later than the moment `bench assess release` records it. |
| `EvidenceRejected` | `released_at is later than the event that records it` (0.10) | Record releases only with `bench assess release`; if the sidecar log was edited, restore it from your copy or start a new assessment. |
| `EvidenceRejected` | `reviewer_subject must be the collector's reviewer, exactly` (0.10) | Bind assignments only with `bench assess assign`; if the sidecar log was edited, restore it from your copy or start a new assessment. |
| `EvidenceRejected` | `the assignment's … artifact id differs from the export's` (0.10) | Use an intact `cli.py backup` copy as `--collector-db`; if it persists, have the collector issue a new assignment and bind that one. |
| `EvidenceRejected` | `the assignment's … hash is not its artifact's` (0.10) | Have the collector issue a new assignment (this one's artifact changed after it was made), then bind it with `bench assess assign`. |
| `EvidenceRejected` | `the attempts come from different tasks (… and …); an assessment compares two outputs of one task` (0.10) | Pass two `--attempt` references from trials of the same task, with the same `task_digest`. |
| `EvidenceRejected` | `the collector assignment does not show exactly this assessment's two review artifacts (a changed, missing or foreign artifact)` (0.10) | Bind an assignment whose two sides show this assessment's two `--review-artifact` files, byte for byte; check the `--assignment` id. |
| `EvidenceRejected` | `the collector assignment has no task version binding the frozen rubric; unbound prototype records stay research history` (0.10) | issue a new assignment for a task version carrying the frozen rubric digest |
| `EvidenceRejected` | `the collector assignment predates the recorded release; an assignment made before this assessment stays research history and is not bound` (0.10) | Have the collector issue a new assignment now, after the release, and bind it with `bench assess assign --assignment <id>`. |
| `EvidenceRejected` | `the collector build differs from the one the frozen judge protocol pins (implementation, commit or a source file hash)` (0.10) | run the collector at the pinned commit, or freeze a new protocol |
| `EvidenceRejected` | `the collector has no artifact …` (0.10) | Pass a complete, current `cli.py backup` copy as `--collector-db`; this one lacks the assignment's artifact record. |
| `EvidenceRejected` | `the collector has no assignment …` (0.10) | Check the `--assignment` id, and pass a `--collector-db` copy made after the collector issued that assignment. |
| `EvidenceRejected` | `the collector has no task …` (0.10) | Pass a complete, current `cli.py backup` copy as `--collector-db`; this one lacks the assignment's task record. |
| `EvidenceRejected` | `the collector marked this assignment invalid; its vote is not bound` (0.10) | Start a new assessment in a new sidecar with a fresh collector assignment; an invalidated assignment's vote cannot be imported. |
| `EvidenceRejected` | `the collector record does not fit the export schema: …` (0.10) | Use an intact `cli.py backup` copy from the pinned collector build; if the named field still fails, report it to the collector's maintainers. |
| `EvidenceRejected` | `the collector task's frozen rubric digest is not the assessment's rubric` (0.10) | Have the collector issue a new assignment for a task version carrying this assessment's `--rubric` digest, then bind it with `bench assess assign`. |
| `EvidenceRejected` | `the collector vote's tags are not JSON` (0.10) | Use an intact `cli.py backup` copy as `--collector-db`; if the vote's tags are still not JSON, report it to the collector's maintainers. |
| `EvidenceRejected` | `the export's task is not the assignment's task version` (0.10) | Use an intact `cli.py backup` copy as `--collector-db`; if it persists, have the collector issue a new assignment and bind that one. |
| `EvidenceRejected` | `the frozen … … changed` (0.10) | Restore the rubric or judge protocol file to the bytes frozen at `bench assess create`, or start a new assessment with the changed file. |
| `EvidenceRejected` | `the release must list exactly the runtimes the two attempts recorded (…), in (attempt_id, provider, runtime_id) order` (0.10) | Record releases only with `bench assess release`; if the sidecar log was edited, restore it from your copy or start a new assessment. |
| `EvidenceRejected` | `the release proof observes …:…, which neither attempt recorded` (0.10) | Remove that observation from the `--proof` file, or correct its provider and runtime id to one the attempts recorded. |
| `EvidenceRejected` | `the vote is not this assignment's reviewer's vote` (0.10) | Import votes only with `bench assess import-vote`; if the sidecar log was edited, restore it from your copy or start a new assessment. |
| `EvidenceRejected` | `the vote predates its assignment` (0.10) | Check the collector's clock and records; this vote cannot be bound, so start a new assessment with a fresh assignment and vote. |
| `EvidenceRejected` | `the vote's collector … differs from the assignment's export` (0.10) | Rerun `bench assess import-vote` with the same `--collector-code` and a database whose assignment, artifacts and task are unchanged since `bench assess assign`. |
| `EvidenceRejected` | `the … artifact belongs to another collector case` (0.10) | Have the collector issue an assignment whose two artifacts belong to its own case, then bind it with `bench assess assign`. |
| `EvidenceRejected` | `the … output belongs to another task version` (0.10) | Have the collector issue an assignment whose artifacts belong to its task version, then bind it with `bench assess assign`. |
| `EvidenceRejected` | `… has no finished_at: not a completed rollout` (0.10) | Wait for the trial to finish, then run `bench assess attempt-ref` again; a finished trial's `result.json` has `finished_at`. |
| `EvidenceRejected` | `… has no task_digest, so the attempt cannot be bound to a task` (0.10) | Use a trial run with `bench run` or `bench eval run`, whose `result.json` records `task_digest`. |
| `EvidenceRejected` | `… is not a JSON object` (0.10) | Point at a rollout folder whose `result.json` is the JSON object BenchFlow wrote; restore it if it was replaced. |
| `EvidenceRejected` | `… is not readable JSON` (0.10) | Restore the rollout's `result.json` as BenchFlow wrote it; this one is not valid JSON (truncated or edited). |
| `EvidenceRejected` | `… must be the collector vote's (…)` (0.10) | Import votes only with `bench assess import-vote`; if the sidecar log was edited, restore it from your copy or start a new assessment. |
| `EvidenceRejected` | `… names no rollout (rollout_name)` (0.10) | Use a rollout folder BenchFlow wrote, whose `result.json` records `rollout_name`. |
| `EvidenceRejected` | `… records no sandbox and no runtime was declared; name every execution allocation so its release can be proven` (0.10) | Pass `--runtime PROVIDER:RUNTIME_ID[:ROLE]` for each sandbox the attempt ran in, so its release can be proven. |
| `EvidenceRejected` | `… repeats the member …` (0.10) | Restore `result.json` as BenchFlow wrote it; each JSON key may appear only once. |
| `EvidenceRejected` | `… … cannot be read: …` (0.10) | Make the named file readable by your user (check its permissions), then rerun. |
| `EvidenceRejected` | `… … changed while it was hashed` (0.10) | Wait until nothing writes the named file (a running trial or a sync), then rerun. |
| `EvidenceRejected` | `… … changed while it was read` (0.10) | Wait until nothing writes the named file (a running trial or a sync), then rerun. |
| `EvidenceRejected` | `… … is not a regular file (links are refused)` (0.10) | Pass the real file instead of a link or folder, or copy the file over the link. |
| `EvidenceRejected` | `…: a file name is not valid UTF-8` (0.10) | Rename the file in the `--output` folder to a valid UTF-8 name, then hash the attempt again. |
| `EvidenceRejected` | `…: output changed (…)` (0.10) | Restore the output files to what `bench assess attempt-ref` hashed, or hash the attempt again and start a new assessment. |
| `EvidenceRejected` | `…: output manifest digest differs` (0.10) | Don't edit the attempt reference; make a new one for this rollout with `bench assess attempt-ref` and start a new assessment. |
| `EvidenceRejected` | `…: result.json changed (…)` (0.10) | Restore the rollout's `result.json` to the bytes `bench assess attempt-ref` hashed, or hash the attempt again and start a new assessment. |
| `EvidenceRejected` | `…: result.json names another rollout or task` (0.10) | Don't edit the attempt reference; make a new one for this rollout with `bench assess attempt-ref` and start a new assessment. |
| `EvidenceRejected` | `…: review artifact changed (…)` (0.10) | Restore the review artifact to the bytes `bench assess attempt-ref` hashed, or hash the attempt again and start a new assessment. |
| `EvidenceRejected` | `…: runtime_refs do not retain exactly the result.sandbox_id recorded in result.json; declared runtimes cannot replace it` (0.10) | Don't edit the reference's `runtime_refs`; make a new one with `bench assess attempt-ref`, adding other runtimes with `--runtime`. |
| `EvidenceRejected` | `…: task_digest is not 'sha256:<64 hex>'` (0.10) | Restore the rollout's `result.json` as BenchFlow wrote it; `task_digest` must be `sha256:` followed by 64 hex digits. |
| `FileNotFoundError` | `Cannot migrate task without legacy files: …` | check the path, and that you can read it |
| `FileNotFoundError` | `Could not find MCP server config for …` | check the path, and that you can read it |
| `FileNotFoundError` | `Could not find skill-creator. Pass --skill-creator-dir or set BENCHFLOW_SKILL_CREATOR_DIR. Checked: …` | check the path, and that you can read it |
| `FileNotFoundError` | `Could not identify skill-creator under resolved skills root: …` | check the path, and that you can read it |
| `FileNotFoundError` | `Missing task.md: …` | check the path, and that you can read it |
| `FileNotFoundError` | `No evals/evals.json found in …. Create one with test cases for your skill.` | check the path, and that you can read it |
| `FileNotFoundError` | `No readable result.json in …; pass a trial directory (jobs/<job>/<task>__<id>) or use bf.load_job for a job directory` | check the path, and that you can read it |
| `FileNotFoundError` | `No result.json at …; pass a rollout directory (jobs/<job>/<task>__<id>) or its result.json` | check the path, and that you can read it |
| `FileNotFoundError` | `No trials under …` | check the path, and that you can read it |
| `FileNotFoundError` | `Not a job or trial folder: …` | check the path, and that you can read it |
| `FileNotFoundError` | `Oracle requires oracle/solve.sh or legacy solution/solve.sh: …` | check the path, and that you can read it |
| `FileNotFoundError` | `Path does not exist: …` (0.10) | check the path, and that you can read it |
| `FileNotFoundError` | `Path … not found in HF dataset …` | check the path, and that you can read it |
| `FileNotFoundError` | `Scene skills_dir not found: …` | check the path, and that you can read it |
| `FileNotFoundError` | `Session file not found: …` | check the path, and that you can read it |
| `FileNotFoundError` | `Source directory … does not exist` | check the path, and that you can read it |
| `FileNotFoundError` | `Task directory not found: … (resolved against …); pass the folder that holds task.md or task.toml.` | check the path, and that you can read it |
| `FileNotFoundError` | `Task is missing instruction.md: …` | check the path, and that you can read it |
| `FileNotFoundError` | `Task is missing task.toml: …` | check the path, and that you can read it |
| `FileNotFoundError` | `Tasks directory not found: … (resolved against …); pass a folder of task folders, or one task folder (each holds task.md or task.toml)` | check the path, and that you can read it |
| `FileNotFoundError` | `Tasks not found under … in HF dataset …: …` | check the path, and that you can read it |
| `FileNotFoundError` | `configured upload source must be an existing regular file or directory, not a symlink: …` | check the path, and that you can read it |
| `FileNotFoundError` | `load_job needs at least one path` | check the path, and that you can read it |
| `FileNotFoundError` | `no task document in …: expected task.md (native), or legacy task.toml + instruction.md` | check the path, and that you can read it |
| `FileNotFoundError` | `no task.toml in …` | check the path, and that you can read it |
| `FileNotFoundError` | `no trial (a folder with result.json) matches …` | check the path, and that you can read it |
| `FileNotFoundError` | `no trial (a folder with result.json) under …; pass a job directory such as jobs/<job_name>` | check the path, and that you can read it |
| `FileNotFoundError` | `skills_dir not found: …` | check the path, and that you can read it |
| `FileNotFoundError` | `snapshot not found: …` | check the path, and that you can read it |
| `FileNotFoundError` | `task has no bundled skills: …` | check the path, and that you can read it |
| `FileNotFoundError` | `transfer source … does not exist` | check the path, and that you can read it |
| `FileNotFoundError` | `… and … not found. Please ensure at least one of these files exist.` | check the path, and that you can read it |
| `FileNotFoundError` | `… not found. Please ensure the file exists.` | check the path, and that you can read it |
| `FileNotFoundError` | `…: no oracle/solve.sh` | check the path, and that you can read it |
| `GitRemoteCloneError` | (built at run time) | Check this machine's network or proxy and rerun; if git says the branch is missing, set `--source-ref` (or `source.ref`) to one the remote has. |
| `GitRemoteCommandError` | (built at run time) | Check this machine's network or proxy and rerun; if git says the ref is missing, check the branch, tag or commit exists in the remote. |
| `HubError` | (built at run time) | pass --hub &lt;url&gt; or set BENCHFLOW_HUB_URL |
| `HubError` | (built at run time) | set BENCHFLOW_HUB_TOKEN to an API key of this Hub with the scope this needs |
| `HubError` | (built at run time) | use the Hub's HTTPS address; BenchFlow never sends a key over a downgrade |
| `HubError` | `a Hub dataset is named org/name, got …` (0.10) | Name the dataset as `org/name`, with `@version` on `bench run -d`, as `bench hub list` shows it. |
| `HubError` | `could not reach the Hub at …: …` (0.10) | check the address and this machine's network |
| `HubError` | `no task folders in …` (0.10) | Give `bench hub publish` a folder that holds task folders, each with its `task.md` or `task.toml`. |
| `HubError` | `not a Hub address: …` (0.10) | name the Hub as http(s)://host[:port], e.g. --hub `https://hub.example.org` |
| `HubError` | `refusing plain http for a Hub that is not on this machine: …` (0.10) | use https://, or an SSH tunnel to a local port |
| `HubError` | `refusing to cache …@… from …: a name part is not a folder name` (0.10) | The Hub named a dataset or version that can't be a folder name (a slash, `..` or an empty part); pull another version, and tell the Hub's operator. |
| `HubError` | `… answered a dataset document this bench does not read (schema …)` (0.10) | Upgrade BenchFlow to a release that reads this Hub's dataset format, or ask the Hub's operator which release does. |
| `HubError` | `… answered run … with something other than a run` (0.10) | check that --hub names a BenchFlow Hub, not a proxy or another server |
| `HubError` | `… did not answer JSON: …` (0.10) | Check that `--hub` or `BENCHFLOW_HUB_URL` is a BenchFlow Hub's address and that the Hub is up; a proxy or a login page answers with HTML. |
| `HubError` | `… has no dataset registry: it answered … to … …, and its route list has no /api/v1/registry` (0.10) | this Hub can't publish, show or serve datasets by name; run the tasks from a folder (bench run &lt;tasks-dir&gt;), or name a Hub that has the dataset registry |
| `HubError` | `… holds more than … entries` (0.10) | The task archive holds more than 100,000 files, so it isn't unpacked; publish the task without generated or vendored files, or run it from its source. |
| `HubError` | `… is not a task folder (no task.md or task.toml)` (0.10) | Add the task's `task.md` (or `task.toml`), or move that folder out of the folder you publish. |
| `HubError` | `… sent more than the … bytes the manifest allows` (0.10) | The download was larger than the dataset's manifest says, so it was stopped; pull again, and if it repeats, tell the Hub's operator. |
| `HubError` | `… unpacks to more than … bytes` (0.10) | The task unpacks to more than 2 GiB, so it isn't unpacked; publish it without large generated files, or run it from its source. |
| `HubError` | `…: refusing entry … (only plain files and folders inside the task)` (0.10) | The archive holds a link, a device or a path outside the task, which BenchFlow never unpacks; republish the task with plain files and folders only. |
| `HubError` | `…: refusing entry … (…)` (0.10) | The entry broke BenchFlow's unpacking rules, for the reason in parentheses; republish the task with plain files and folders only. |
| `HubError` | `…: … is a link; a published task holds plain files only` (0.10) | Replace the link with the file or folder it points to, then publish again. |
| `HubError` | `…: … is not a regular file` (0.10) | Remove the device, socket or pipe from the task folder, then publish again. |
| `HubError` | `…@… has a malformed or repeated task entry: …` (0.10) | The Hub's document for this version is broken, so nothing is pulled from it; pull another version, and tell its publisher or the Hub's operator. |
| `HubError` | `…@… on … lists no tasks` (0.10) | This version holds no tasks; pull another version, or ask its publisher to publish it again. |
| `HubError` | `…@…: task … has a malformed archive entry` (0.10) | The Hub's document for this version is broken, so nothing is pulled from it; pull another version, and tell its publisher or the Hub's operator. |
| `HubError` | `…@…: task … has no digest or archive` (0.10) | The Hub's document for this version is broken, so nothing is pulled from it; pull another version, and tell its publisher or the Hub's operator. |
| `HubError` | `…@…: task … unpacked to …, the manifest pins …` (0.10) | The task's files don't match the digest the dataset pins, so the task isn't used; pull again, and if it repeats, tell the Hub's operator. |
| `HubError` | `…@…: task …'s archive does not match the manifest's sha256 and size` (0.10) | The download doesn't match the dataset's checksum, so it isn't used; pull again, and if it repeats, tell the Hub's operator. |
| `HubError` | `…@…: task …'s archive is larger than … bytes` (0.10) | The task's archive is over 512 MiB, the most BenchFlow pulls; run the task from its source, or ask its publisher to make it smaller. |
| `HubError` | `…@…: the Hub's dataset digest … does not match its tasks (…)` (0.10) | The Hub's digest for this version doesn't match the tasks it lists, so nothing was pulled; tell the Hub's operator, and don't compare scores with runs of that version. |
| `ImageBuildError` | (built at run time) | fix that line in the task's environment/Dockerfile; `docker build <task>/environment` prints the whole build log |
| `InterruptedJobError` | `… is a job still running in process …: … … running, none finished yet` | read it again once a trial has finished |
| `InterruptedJobError` | `… is a job that was stopped before any trial finished: … … started, none wrote result.json` | `bench eval resume …` runs them |
| `InvalidRecord` | `an assessment compares exactly two attempts` (0.10) | Pass `--attempt` exactly twice to `bench assess create`. |
| `InvalidRecord` | `payload is not I-JSON: …` (0.10) | Remove the value the message names from your input: lone surrogates, non-finite numbers and integers beyond `2**53 - 1` are not I-JSON. |
| `InvalidRecord` | `runtime … must be PROVIDER:RUNTIME_ID[:ROLE]` (0.10) | Write each `--runtime` as `PROVIDER:RUNTIME_ID` or `PROVIDER:RUNTIME_ID:ROLE`, with role `solver`, `verifier` or `shared`. |
| `InvalidRecord` | `… records sandbox …; name its provider (daytona, docker, ...)` (0.10) | Pass `--sandbox-provider` with the provider of that sandbox, such as `--sandbox-provider docker` or `daytona`. |
| `IsADirectoryError` | (built at run time) | check the path, and that you can read it |
| `JobMatrixError` | `…this YAML job config runs … job(s) (…), and an Evaluation is one of them` | run them all with bench run --config …; from Python, pass job_cell=&lt;n&gt; for one of job_plan(path).cells |
| `KeptServiceFreezerUnavailable` | ``This task keeps services (x-keep-services), which BenchFlow holds in a frozen cgroup v2 of root's from the kill after the agent's turn until the verifier starts, and this sandbox cannot (the check's reason is below). No agent code ran. Run it where Docker uses cgroup v2 and knows writable-cgroups (`docker info` shows Cgroup Version: 2), or run the task without x-keep-services.…`` | Run the task where Docker uses cgroup v2 with writable cgroups (`docker info` shows Cgroup Version: 2), or run it without `x-keep-services`. |
| `KeptServiceFreezerUnavailable` | `… declares kept services on port … (x-keep-services, from …); native Daytona cannot freeze them for an agent run as non-root user …. This trial is not scored: unsupported backend feature. next: run with --sandbox docker or --sandbox remote-docker, or supply a Compose environment on Daytona` | Run with `--sandbox docker` or `--sandbox remote-docker`, or give the task a Compose file (`environment/docker-compose.yaml`) to run it on Daytona. |
| `MalformedTaskError` | `… task(s) still hold [REPLACE: ...] placeholders, so they would build and then score 0: … (first: …: …)` | fill in every placeholder; `bench check …` lists them |
| `MissingCredentialError` | `A model is required with BENCHFLOW_PROVIDER_API_KEY.` | pass --model/model so the supplied provider key can be routed |
| `MissingCredentialError` | `AWS_BEARER_TOKEN_BEDROCK required for Bedrock model … but not set. Export it in your shell or put it in a .env file.` | `export AWS_BEARER_TOKEN_BEDROCK=... AWS_REGION=us-west-2` |
| `MissingCredentialError` | `AWS_REGION or AWS_DEFAULT_REGION required for Bedrock model … but not set. Export one of them in your shell or put it in a .env file.` | `export AWS_REGION=us-west-2` |
| `MissingCredentialError` | `GOOGLE_CLOUD_PROJECT required for Vertex AI model ….` | `export GOOGLE_CLOUD_PROJECT=<project>, or pass --agent-env GOOGLE_CLOUD_PROJECT=<project>` |
| `MissingCredentialError` | `Vertex AI model … requires ADC credentials.` | gcloud auth application-default login |
| `MissingCredentialError` | ``muse-code runs on a Muse Code subscription login: set MUSE_AUTH_JSON to the contents of the auth.json that `TBH_CREDENTIAL_BACKEND=file muse login` writes. A META_API_KEY (pay-as-you-go) run has no BenchFlow proxy route, and BenchFlow never sends API-key traffic around the proxy.`` | run `TBH_CREDENTIAL_BACKEND=file muse login`, then `export MUSE_AUTH_JSON="$(cat ~/.config/muse/auth.json)"` (or put MUSE_AUTH_JSON in .env) |
| `MissingCredentialError` | `…Export it in your shell or put it in a .env file (a key typed into a command stays in your shell history).` | `bench doctor` lists the logins and keys it finds |
| `NoEvaluationRecordError` | `… is not a job folder: it has no evaluation.json and no trials.` | pass a job folder, &lt;jobs-dir&gt;/&lt;job name&gt;; `bench results <jobs-dir>` lists them |
| `NoEvaluationRecordError` | `… …: it has no evaluation.json, so its run cannot be rebuilt from the folder. In Python, pass the run again: Evaluation(tasks_dir=..., jobs_dir=…, config=..., job_name=…).` | re-run the command that started it, which resumes this job and reuses its finished trials: `…` plus any other flags it had (--agent-env, --sandbox-user, a source, skills); a setting that differs refuses the run, and --restart-changed sets those trials aside and runs them again |
| `NotADirectoryError` | `Not a directory: …` | check the path, and that you can read it |
| `OwnModelRouteError` | (built at run time) | Run the task with an agent whose model calls go through BenchFlow's proxy; for the `--usage-tracking required` refusal, use `--usage-tracking auto` or `off`. |
| `PermissionError` | (built at run time) | check the path, and that you can read it |
| `Pre08CopyError` | `Could not copy the BenchFlow 0.7 files of job … to … before resuming it (…); nothing in the job was changed.` | free space in that folder's file system and run the same command again, or pass --fresh to start a new job and leave this one as it was |
| `ProxyExtraRequirementError` | `BENCHFLOW_LITELLM_EXTRA_REQUIREMENTS is set, and checking extras against the proxy's locked set needs uv on this machine: install uv (https://docs.astral.sh/uv/) or unset the variable` | Install uv (`https://docs.astral.sh/uv/`) on this machine or point `UV` at its path, or unset `BENCHFLOW_LITELLM_EXTRA_REQUIREMENTS`. |
| `ProxyExtraRequirementError` | `BENCHFLOW_LITELLM_EXTRA_REQUIREMENTS is set, and the host proxy's venv is built with uv: install uv (https://docs.astral.sh/uv/)` | Install uv (`https://docs.astral.sh/uv/`) or point `UV` at it; or unset `BENCHFLOW_LITELLM_EXTRA_REQUIREMENTS` so the host proxy runs BenchFlow's own LiteLLM. |
| `ProxyExtraRequirementError` | `BENCHFLOW_LITELLM_EXTRA_REQUIREMENTS: item … (not repeated here: it may hold credentials) is not a PEP 508 requirement; give package names with optional versions, separated by spaces, for example 'google-genai>=1.0'` | Set `BENCHFLOW_LITELLM_EXTRA_REQUIREMENTS` to PEP 508 package names with optional versions, separated by spaces, such as `google-genai>=1.0`; the item number marks the bad entry. |
| `ProxyExtraRequirementError` | `BENCHFLOW_LITELLM_EXTRA_REQUIREMENTS: … installs from a URL (not repeated here, it may hold credentials); proxy extras come from the package index only, so the run records what it installed by name and version` | Replace that URL in `BENCHFLOW_LITELLM_EXTRA_REQUIREMENTS` with the package's name and version from the package index, such as `google-genai>=1.0`. |
| `ProxyExtraRequirementError` | `could not check BENCHFLOW_LITELLM_EXTRA_REQUIREMENTS against the proxy's locked set with …: …` | Check that `uv` runs here and can reach the package index (the error ends with its output), then rerun; or unset `BENCHFLOW_LITELLM_EXTRA_REQUIREMENTS`. |
| `ProxyExtraRequirementError` | `proxy extra requirement … conflicts with the locked … in BenchFlow's LiteLLM proxy set (…); extras cannot change a locked package. Drop the extra or ask for … in the proxy's lock.` | Remove that extra from `BENCHFLOW_LITELLM_EXTRA_REQUIREMENTS`, or widen its version range to include the locked version the message names. |
| `ProxyExtraRequirementError` | `proxy extra requirement… … conflict… with … in BenchFlow's LiteLLM proxy set (…); extras cannot change a locked package. Resolver: …` | Remove the named extra from `BENCHFLOW_LITELLM_EXTRA_REQUIREMENTS`, or choose a version of it whose dependencies accept the locked packages the resolver output names. |
| `RegistryAuthError` | (built at run time) | check `docker login <registry>` on this machine, and that the FROM line names an image that login can read |
| `RemoteDockerConfigError` | `remote-docker cannot use host mounts: the paths are on this machine, not on the remote Docker host` | Drop `mounts_json` when you construct the remote-docker sandbox in Python, or use `--sandbox docker`, whose local daemon can bind-mount this machine's paths. |
| `RemoteDockerConfigError` | `remote-docker needs BENCHFLOW_REMOTE_DOCKER_HOST or DOCKER_HOST: ssh://user@host[:port] or tcp://host:2376 with DOCKER_TLS_VERIFY=1` | Set `BENCHFLOW_REMOTE_DOCKER_HOST` (or `DOCKER_HOST`) to `ssh://user@host[:port]`, or to `tcp://host:2376` with `DOCKER_TLS_VERIFY=1`. |
| `RemoteDockerConfigError` | `remote-docker runs on Linux and macOS: it passes the host to docker through a shell shim, never through DOCKER_HOST` | Run BenchFlow on Linux or macOS to use `--sandbox remote-docker`. |
| `RemoteDockerConfigError` | `the Docker host in … holds a space, tab, line break or other control character, which docker would quote with the address in its error: remove it, or percent-encode its ssh user (/ as %2F, ? as %3F, # as %23, @ as %40, a space as %20)` | Remove spaces, tabs and line breaks from `BENCHFLOW_REMOTE_DOCKER_HOST` (or `DOCKER_HOST`); inside the ssh user, write a space as `%20`. |
| `RemoteDockerConfigError` | `the Docker host in … is not a URL docker can read (an IPv6 address goes in [ ], a name never does): use ssh://user@host[:port] or tcp://host:port with TLS` | Write the host as `ssh://user@host[:port]` or `tcp://host:port`, with an IPv6 address in brackets and a host name without them. |
| `RemoteDockerConfigError` | `the Docker host … is a local socket; use --sandbox docker for a local daemon, or set an ssh:// or tcp:// host for remote-docker` | Use `--sandbox docker` for a local daemon, or set `BENCHFLOW_REMOTE_DOCKER_HOST` to an `ssh://` or `tcp://` address for remote-docker. |
| `RemoteDockerConfigError` | `the remote Docker TCP host must use an ASCII hostname (use its IDNA spelling), IPv4 or unscoped bracketed IPv6; zone identifiers and percent-encoded hostnames are not supported` | Write the tcp:// host as an ASCII name (its `xn--` IDNA spelling), an IPv4 address, or a bracketed IPv6 address without a `%` zone. |
| `RemoteDockerConfigError` | `the ssh user in … has a character docker cannot read in a URL: percent-encode its ssh user (/ as %2F, ? as %3F, # as %23, @ as %40, a space as %20)` | Percent-encode the ssh user in the host address: `/` as `%2F`, `?` as `%3F`, `#` as `%23`, `@` as `%40`, a space as `%20`, a lone `%` as `%25`. |
| `RemoteDockerConfigError` | `the ssh:// host in … holds a password, which docker does not accept: log in with an ssh key or agent instead` | Remove the password from the ssh:// address and log in with an ssh key or `ssh-agent` instead. |
| `RemoteDockerConfigError` | `the ssh:// host … takes no path, query or fragment` | Write the ssh:// address as `ssh://user@host[:port]` with nothing after the port: no trailing `/`, path, query or fragment. |
| `RemoteDockerConfigError` | `the tcp:// host … needs TLS: set DOCKER_TLS_VERIFY=1 and DOCKER_CERT_PATH (ca.pem, cert.pem, key.pem); a plain tcp daemon gives root on the host to anyone who can reach the port, so it is refused` | Set `DOCKER_TLS_VERIFY=1` and `DOCKER_CERT_PATH` to a folder with `ca.pem`, `cert.pem` and `key.pem`, or use an `ssh://` host instead. |
| `RemoteDockerConfigError` | `the tcp:// host … with TLS: … not found in … (DOCKER_CERT_PATH)` | Put `ca.pem`, `cert.pem` and `key.pem` in the folder `DOCKER_CERT_PATH` names (default `~/.docker`), or point `DOCKER_CERT_PATH` at the folder holding them. |
| `RemoteDockerConfigError` | `the …:// host in … has an @ after a /, ? or #: percent-encode its ssh user (/ as %2F, ? as %3F, # as %23, @ as %40, a space as %20)` | Percent-encode the ssh user (`/` as `%2F`, `?` as `%3F`, `#` as `%23`, `@` as `%40`) so the only raw `@` is the one before the host. |
| `RemoteDockerConfigError` | `unsupported remote Docker host …: use ssh://user@host[:port] or tcp://host:port with TLS` | Set the host to `ssh://user@host[:port]`, or `tcp://host:port` with TLS, with a host name and, if given, a port from 1 to 65535. |
| `RepositoryNotFoundError` | `repository …/… was not found on GitHub, or it is private` | check the name (`https://github.com/…/…`); a private repository needs a git login on this machine (`gh auth login`) |
| `ResumeMismatchError` | `refusing to resume …: its trials ran …, and setting them aside in … failed (…). … item(s) were moved there; the rest are where they were. Start a new job instead: bench run --fresh (or another --job-name); in Python, a new job_name.` (0.10) | Start a new job with `bench run --fresh` or another `--job-name` (in Python, a new `job_name`); items already moved are in the job's `.set-aside/` folder. |
| `ResumeMismatchError` | `… is a folder BenchFlow did not make; nothing in it moves` | pick another --job-name or another --jobs-dir |
| `ReviewRubricError` | `Review rubric must be a regular task file: …` | Replace the symlinked `rubric.json` with a regular file inside the task (in `verifier/`, `tests/` or the task root). |
| `ReviewRubricError` | `unsupported rubric format …: rubrics are JSON files` | Give `bench review --rubric` a JSON rubric file whose name ends in `.json`; convert other formats to JSON first. |
| `ReviewRubricError` | `… is not a valid rubric: …` | Fix the field the error names: a non-empty `criteria` list, each with `name`, `description` and `guidance`, plus both `blocker` and `weight` or neither. |
| `ReviewRubricError` | `… is not valid JSON: …` | Fix the JSON syntax at the line and column the error gives, and save the rubric as UTF-8. |
| `RunCardError` | (built at run time) | fix …; docs/evaluate/before-you-run.md#the-run-card-010 lists its keys |
| `RunCardError` | `The trial budget cannot be enforced on …: …, and BenchFlow's model proxy is what counts a trial's tokens and turns` (0.10) | run it with an API key, so its calls go through the proxy, or drop --token-budget/--turn-budget (and the tasks' [agent] budget) |
| `RunCardError` | `This run differs from the setup in … (…), which asks every run to match it or say why: …` (0.10) | match the reference, or pass --accept-deviations '&lt;why&gt;' (accept_deviations: in a run config); the note is kept in run-card.json |
| `RunCardError` | `…: [reference] … must be …, not …` (0.10) | fix …; docs/evaluate/before-you-run.md#the-run-card-010 lists its keys |
| `RunCardError` | `…: put every setting in one [reference] table` (0.10) | fix …; docs/evaluate/before-you-run.md#the-run-card-010 lists its keys |
| `RunCardError` | `…: unknown key… … in [reference]; known keys: …` (0.10) | fix …; docs/evaluate/before-you-run.md#the-run-card-010 lists its keys |
| `RunFolderError` | `--tasks-dir given but … does not exist; expected the task source for … there.` | Pass `--tasks-dir` the folder holding a subfolder named after the task, so `<tasks-dir>/<task name>` exists. |
| `RunFolderError` | `Live continuation requires …; set it before continuing` | Export the variable the message names (the live model's provider key), or add it to `.env` in the current folder, then rerun `bench eval continue`. |
| `RunFolderError` | `benchflow continue currently supports the 'openhands' agent only; this run used ….` | `bench eval continue` takes only a run made with the `openhands` agent; for another agent, run the task again with `bench run`. |
| `RunFolderError` | `cannot continue with a different reward parsing policy; set BENCHFLOW_REWARD_LENIENT=… to match the recorded run` | Set `BENCHFLOW_REWARD_LENIENT` to the value the message gives (`1` or `0`) when you run `bench eval continue`, matching the recorded run. |
| `RunFolderError` | `cannot locate task source for …. Pass --tasks-dir pointing at the directory that contains the task (its instruction and verifier are needed to re-run and re-verify).` | Pass `--tasks-dir` with the folder that contains the task's folder; the `task_path` recorded in `config.json` no longer exists on this machine. |
| `RunFolderError` | `cannot reconstruct config_override: missing patch` | Restore the run folder's original `config.json` (its `config_override` has no `patch`), or start a new `bench run` with the original configuration. |
| `RunFolderError` | `cannot reconstruct recorded environment, loop, skills or harness; start a new evaluation with the original configuration` | Start a new `bench run` with the original configuration: `bench eval continue` can't rebuild runs with an environment manifest, self-gen skills, a non-ACP harness or a loop strategy. |
| `RunFolderError` | `cannot reconstruct recorded scene settings; start a new evaluation with the original configuration` | Start a new `bench run` with the original configuration: `bench eval continue` handles one scene with one OpenHands role and no per-role skills, timeouts or env. |
| `RunFolderError` | `cannot reconstruct review: missing reviewer settings` | Restore the run folder's original `config.json` (its `review` lacks `reviewer` settings), or start a new `bench run` with the original configuration. |
| `RunFolderError` | `cannot reconstruct reward_lenient: expected a boolean` | Restore the run folder's original `config.json`, where `reward_lenient` must be `true` or `false`, or start a new `bench run` with the original configuration. |
| `RunFolderError` | `cannot reconstruct skills: missing recorded directory` | Start a new `bench run` with the original `--skills-dir`: this run's `config.json` records custom runtime skills without their folder. |
| `RunFolderError` | `cannot reconstruct …: restore its recorded directory before continuing` | Restore the folder `config.json` records for `context_root` or `skills_dir` at that same path on this machine, then rerun `bench eval continue`. |
| `RunFolderError` | `could not parse …: …` | Restore the named file in the run folder from an intact copy of the run; it is unreadable or not valid JSON. |
| `RunFolderError` | `expected a JSON list in …` | Restore the run's `prompts.json` from an intact copy; it must hold a JSON list of prompt strings, or an object with a `prompts` list. |
| `RunFolderError` | `expected a JSON object in …, got …` | Restore the named file (`config.json` or `result.json`) from an intact copy of the run; it must hold a JSON object. |
| `RunFolderError` | `invalid proxy mode …; expected one of ….` | Set `--proxy-mode` to `auto`, `host` or `sandbox`. |
| `RunFolderError` | `missing required artifact: …` | Pass the trial's own folder, which holds `config.json`, not its job folder; if `config.json` was removed, restore it from the original run. |
| `RunFolderError` | `missing required artifact: … — record-replay needs the LLM trajectory. Was this run captured with usage tracking enabled?` | Continue only a run whose folder has `trajectory/llm_trajectory.jsonl`, recorded through BenchFlow's model proxy; this one can't be replayed, so rerun the task instead. |
| `RunFolderError` | `no live-continuation model: the run recorded no model and --model was not given.` | Pass `--model` with the model to continue with, or `--replay-only` (host proxy mode) to rebuild the run without a model. |
| `RunFolderError` | `not a directory: …` | Pass the path of an existing trial folder, the one holding `config.json` and `trajectory/llm_trajectory.jsonl`. |
| `RunFolderError` | `replay-only mode is only supported with host proxy mode.` | Rerun with `--proxy-mode host` (auto already does that for a Docker run); for a run on another sandbox, drop `--replay-only` and continue live with `--model`. |
| `RunFolderError` | `run … has error_category=…, not a timeout/idle_timeout — continuing it may not be meaningful.` | Continue only a run that timed out (`error_category` `timeout` or `idle_timeout`), or drop `--require-timeout` to continue this one anyway. |
| `RunFolderError` | `… contained no usable LLM exchanges — nothing to replay.` | Continue a run whose `trajectory/llm_trajectory.jsonl` holds recorded exchanges; this one is empty or every line is malformed, so rerun the task instead. |
| `SeparateVerifierRefusedError` | (built at run time) | Drop the listed tasks with `--exclude <task>`, or run without `--verifier-sandbox separate` (use `--integrity audit` instead of `--integrity strict`). |
| `StaleExpectation` | `the log ends at event_seq … (…); the writer expected … (…); nothing was appended` (0.10) | reload the sidecar, re-check its state, then retry |
| `TaskConfigKeyError` | (built at run time) | Fix or remove the `task.toml` key the message names (a typo comes with a suggested spelling), or set `BENCHFLOW_TASK_TOML_ALLOW_UNKNOWN_KEYS=1` to load unknown keys anyway. |
| `TaskDocumentParseError` | `Expected integer value, got bool` | Write the role's `timeout_sec` or `idle_timeout_sec` under `agents.roles` in `task.md` as a whole number of seconds, not `true` or `false`. |
| `TaskDocumentParseError` | `Expected integer value, got …` | Write the role's `timeout_sec` or `idle_timeout_sec` under `agents.roles` in `task.md` as a whole number of seconds, without quotes or a decimal point. |
| `TaskDocumentParseError` | `Expected list of strings` | Write the role's `capabilities` under `agents.roles` in `task.md` as a YAML list of strings. |
| `TaskDocumentParseError` | `Expected mapping of strings` | Write the role's `env` under `agents.roles` in `task.md` as a mapping of variable names to values. |
| `TaskDocumentParseError` | `Expected string value, got …` | Write the field as a string in `task.md`, quoting numbers and booleans: a role's `model`, `reasoning_effort` or `skills_dir`, a scene's `name` or `skills_dir`, a turn's `prompt`. |
| `TaskDocumentParseError` | `agents.roles keys must be role names` | Name each role under `agents.roles` in `task.md` with a text key; quote names YAML reads as numbers or booleans, such as `'1'` or `'yes'`. |
| `TaskDocumentParseError` | `agents.roles[…].agent is required` | Add `agent: <name>` to that role under `agents.roles` in `task.md`, naming an agent `bench agents` lists. |
| `TaskDocumentParseError` | `image must be a non-empty string` | Set the `image` shorthand in `task.md` frontmatter to a Docker image name, or remove it and set `sandbox.docker_image`. |
| `TaskDocumentParseError` | `name must be a non-empty string` | Set the `name` shorthand in `task.md` frontmatter to a non-empty task name, or remove it and set `task.name`. |
| `TaskDocumentParseError` | `scene roles must be a list of role names` | Write the scene's `roles` in `task.md` as a YAML list of role names defined under `agents.roles`. |
| `TaskDocumentParseError` | `scene turns must be a list` | Write the scene's `turns` in `task.md` as a YAML list; each turn is a role name or a mapping with `role` and an optional `prompt`. |
| `TaskDocumentParseError` | `scenes must be a list` | Write `scenes` in `task.md` frontmatter as a YAML list of scenes (each a mapping with `name`, `roles`, `turns`), not a single mapping. |
| `TaskDocumentParseError` | `task.md frontmatter is missing closing ---` | Add a line holding only `---` after the frontmatter in `task.md`, before the prompt text. |
| `TaskDocumentParseError` | `task.md frontmatter must be a mapping` | Write the `task.md` frontmatter as YAML `key: value` pairs between the two `---` lines, not a list or a bare value. |
| `TaskDocumentParseError` | `task.md has duplicate section ## …` | Merge the two sections with that heading in `task.md` into one, or write a heading meant as prompt text with a leading backslash, like `\## prompt`. |
| `TaskDocumentParseError` | `task.md must start with YAML frontmatter` | Start `task.md` with a line holding only `---`, then the YAML frontmatter and a closing `---` line, then the prompt. |
| `TaskDocumentParseError` | `turns[…] references role … outside the scene` | Add that role to the scene's `roles` in `task.md`, or change the turn's `role` to a role the scene lists. |
| `TaskDocumentParseError` | `turns[…].role is required` | Give that turn a `role` naming one of the scene's roles in `task.md`, or write the turn as the role name alone. |
| `TaskDocumentParseError` | `unknown task.md profile …; known profiles: …` | Set `profile` in `task.md` to a known profile: `code-change`, `task-toml-compatible`, `criteria-runner`, `acceptance-live`, `multi-agent` or `leaderboard-local`; or remove it. |
| `TaskDocumentParseError` | `… entries must be profile names` | Write `profile` or `profiles` in `task.md` as profile names, for example `profiles: [code-change]`. |
| `TaskDocumentParseError` | `… must be a mapping` | Write the key the message names in `task.md` (such as `agents`, `agents.roles`, `user` or `benchflow`) as a YAML mapping of `key: value` pairs. |
| `TaskDocumentParseError` | `… must be a mapping after normalization` | Write the table the message names (such as `task`, `sandbox` or `benchflow`) as a YAML mapping in `task.md`, so the `name`, `image`, `verifier` or `oracle` shorthand can fill it. |
| `TaskDocumentParseError` | `… must be a mapping or non-empty path` | Set `verifier` or `oracle` in `task.md` frontmatter to a mapping or a relative folder path, not an empty value or a list. |
| `TaskDocumentParseError` | `… path must be safe and relative` | Write the `verifier` or `oracle` path in `task.md` relative to the task folder, without a leading `/` or any `..` part. |
| `TaskDocumentParseError` | `… references unknown role …` | Define that role under `agents.roles` in `task.md`, or correct the name in the scene's `roles` or the turn's `role`. |
| `TaskDocumentParseError` | `…: the frontmatter is not valid YAML: … …` | Fix the YAML syntax at the `task.md` line the message names, then rerun `bench check <task>`. |
| `TaskEnvNotAllowedError` | (built at run time) | Pass `--allow-task-env NAME` for each variable you trust the task with, or export `BENCHFLOW_ALLOW_TASK_ENV` (not from `.env`); for a path or hook outside the task, follow the printed remedy. |
| `TransitionRefused` | `a consumption is recorded by its consumer (actor.subject = consumer_id)` (0.10) | Record a consumption with `bench assess consume <sidecar> --consumer <id>`, which sets the actor to the consumer; don't append `label.consumed` events by hand. |
| `TransitionRefused` | `a … must name the current effective answer …; … is superseded, retracted or unknown` (0.10) | Pass the current effective answer's event id to `--supersedes` (correct) or `--target` (retract); `bench assess show <sidecar> --events` lists the events. |
| `TransitionRefused` | `an assessment starts with assessment.created` (0.10) | Start the sidecar with `bench assess create <sidecar>` before any other `bench assess` command; don't write its event log by hand. |
| `TransitionRefused` | `an unconsumed revision has no consumer to compensate: use not_applicable` (0.10) | Rerun `bench assess withdraw` with `--disposition not_applicable`: nobody consumed this revision. |
| `TransitionRefused` | `assignment.issued needs compute.released first: every execution resource is released before an assignment can wait for a person` (0.10) | record the release with `bench assess release <sidecar> --proof <file>` |
| `TransitionRefused` | `compute was already released by event …` (0.10) | Skip this step: the release is already recorded, so continue with `bench assess assign <sidecar>`; a second proof can't replace the first. |
| `TransitionRefused` | `consumption_event_ids must list every consumption of the revision, in order` (0.10) | Withdraw with `bench assess withdraw`, which lists the revision's consumptions itself; don't append `label.withdrawn` events by hand. |
| `TransitionRefused` | `event names assessment …, the sidecar holds …` (0.10) | Write to each sidecar only with `bench assess` commands; an event from another assessment belongs in that assessment's own sidecar directory. |
| `TransitionRefused` | `label revision … does not exist` (0.10) | Pass `--revision` a revision this label has; `bench assess show <sidecar>` prints the label's current revision. |
| `TransitionRefused` | `label revision … is current; only a revision whose answer was corrected or retracted can be withdrawn` (0.10) | Withdraw only a superseded revision: first change the answer with `bench assess correct` or `bench assess retract`, then withdraw the old revision. |
| `TransitionRefused` | `label revision … may not be consumed: …` (0.10) | Consume only the current, ready revision: run `bench assess reduce <sidecar>` if a correction awaits, then check with `bench assess show <sidecar>`; a withheld label can't be consumed. |
| `TransitionRefused` | `label … is not this assessment's label` (0.10) | Use `bench assess consume` or `bench assess withdraw` on this sidecar, which fill in its own label id; don't append label events by hand. |
| `TransitionRefused` | `label.reduced does not match the deterministic reduction of the log (expected revision … … from event …)` (0.10) | Run `bench assess reduce <sidecar>`, which appends the reduction the log calls for; don't write `label.reduced` events by hand. |
| `TransitionRefused` | `nothing to reduce yet (state …): the label needs an answer` (0.10) | Import the collector's vote first with `bench assess import-vote <sidecar>`, then run `bench assess reduce <sidecar>` again. |
| `TransitionRefused` | `nothing to reduce: no answer yet` (0.10) | Import the collector's vote with `bench assess import-vote <sidecar>`, then reduce with `bench assess reduce <sidecar>`; don't append `label.reduced` events by hand. |
| `TransitionRefused` | `only the reviewer … or an operator may … this answer` (0.10) | Rerun with `--actor` set to the answer's reviewer, or with `--actor-role operator` if you act as an operator. |
| `TransitionRefused` | `the assessment has no collector assignment yet` (0.10) | Bind the assignment first: `bench assess release <sidecar> --proof <file>` if not recorded yet, then `bench assess assign <sidecar>` with the collector's assignment id. |
| `TransitionRefused` | `the assessment has no label yet` (0.10) | Record the label first with `bench assess reduce <sidecar>` (after a vote is imported), then consume or withdraw a revision. |
| `TransitionRefused` | `the label is withheld; only a separately issued replacement assignment (a new assessment) can follow` (0.10) | Start a new assessment in a new sidecar directory with `bench assess create`; a withheld label's answer can't be corrected or retracted. |
| `TransitionRefused` | `the sidecar holds no assessment` (0.10) | Point the command at the sidecar that holds the assessment, or start one there with `bench assess create <sidecar>`. |
| `TransitionRefused` | `the sidecar holds no assessment yet` (0.10) | Point the command at the sidecar that holds the assessment, or start one there with `bench assess create <sidecar>`. |
| `TransitionRefused` | `there is no effective answer to … (it was retracted)` (0.10) | Run `bench assess reduce <sidecar>` to record the retraction; for a new answer, start a new assessment with `bench assess create` in a new sidecar directory. |
| `UsageLimitError` | (built at run time) | switch to another login or wait for the reset, then resume the job (`bench eval resume <job_dir>`); `bench doctor` shows each login's headroom |
| `VerifierDocumentParseError` | `expected a non-empty string` | Write a non-empty string for the `verifier.md` field that is empty or not text, such as `name`, `default_strategy`, an `outputs` path or a strategy field. |
| `VerifierDocumentParseError` | `verifier.default_strategy must name one of verifier.strategies` | Set `verifier.default_strategy` in `verifier.md` to one of the names under `verifier.strategies`, or remove it to use the first strategy. |
| `VerifierDocumentParseError` | `verifier.md frontmatter is missing closing ---` | Add a line holding only `---` after the frontmatter in `verifier/verifier.md`. |
| `VerifierDocumentParseError` | `verifier.md frontmatter is not valid YAML: …` | Fix the YAML syntax in the frontmatter of `verifier/verifier.md` where the error points, then rerun `bench check <task>`. |
| `VerifierDocumentParseError` | `verifier.md frontmatter must be a mapping` | Write the frontmatter of `verifier/verifier.md` as YAML `key: value` pairs, starting with a `verifier:` mapping, not a list or a bare value. |
| `VerifierDocumentParseError` | `verifier.md must start with YAML frontmatter` | Start `verifier/verifier.md` with a `---` line, then YAML frontmatter holding a `verifier:` mapping with `strategies`, then a closing `---` line. |
| `VerifierDocumentParseError` | `verifier.outputs.aggregate_policy must be a mapping` | Write `verifier.outputs.aggregate_policy` in `verifier.md` as a YAML mapping, or remove it. |
| `VerifierDocumentParseError` | `verifier.strategies keys must be non-empty strings` | Name each strategy under `verifier.strategies` in `verifier.md` with a non-empty text key; quote names YAML would read as numbers or booleans. |
| `VerifierDocumentParseError` | `verifier.strategies must contain at least one strategy` | Add at least one strategy under `verifier.strategies` in `verifier.md`, for example a `script` strategy with a `command`. |
| `VerifierDocumentParseError` | `verifier.strategies.… must declare either context or context_file, not both` | Keep only one of `context` and `context_file` in that llm-judge strategy in `verifier.md`. |
| `VerifierDocumentParseError` | `verifier.strategies.….command is required when grading is 'script'` | Add a `command` (a relative path in the verifier folder) to that taskmd strategy in `verifier.md`, or set its `grading` to `rubric`. |
| `VerifierDocumentParseError` | `verifier.strategies.….format must be json, jsonl, or auto` | Set that ors-episode strategy's `format` in `verifier.md` to `json`, `jsonl` or `auto`, or remove it. |
| `VerifierDocumentParseError` | `verifier.strategies.….grading must be 'rubric' or 'script'` | Set that taskmd strategy's `grading` in `verifier.md` to `rubric` or `script`. |
| `VerifierDocumentParseError` | `verifier.strategies.….inputs must be a non-empty list of strings` | Write that strategy's `inputs` in `verifier.md` as a non-empty YAML list of non-empty strings. |
| `VerifierDocumentParseError` | `verifier.strategies.….isolation must be 'shared' or 'separate'` | Set that taskmd strategy's `isolation` in `verifier.md` to `shared` or `separate`, or remove it to get `shared`. |
| `VerifierDocumentParseError` | `verifier.strategies.….isolation must be 'verifier-only'` | Set that agent-judge strategy's `isolation` in `verifier.md` to `verifier-only`. |
| `VerifierDocumentParseError` | `verifier.strategies.….offline must be a boolean` | Set that taskmd strategy's `offline` in `verifier.md` to `true` or `false`, without quotes. |
| `VerifierDocumentParseError` | `verifier.strategies.….role references missing ## role:… section` | Add a `## role:<name>` section for that role to the body of `verifier.md`, or set the strategy's `role` to an existing section's name. |
| `VerifierDocumentParseError` | `verifier.strategies.….type … is an unknown value; choose one of: …` | Set the strategy's `type` in `verifier.md` to `script`, `criteria-runner`, `llm-judge`, `agent-judge`, `ors-episode` or `taskmd`. |
| `VerifierDocumentParseError` | `verifier.strategies.….workdir must be absolute` | Set that taskmd strategy's `workdir` in `verifier.md` to an absolute path starting with `/`, or remove it. |
| `VerifierDocumentParseError` | `… must be a mapping` | Write the key the message names in `verifier.md` (`verifier`, `verifier.strategies`, a strategy, `verifier.rubric` or `verifier.outputs`) as a YAML mapping. |
| `VerifierDocumentParseError` | `… must be a non-empty string` | Add the key the message names to that strategy in `verifier.md` with a non-empty value: `type`, plus `command` (script), `root` (criteria-runner), `rubric` (llm-judge) or `role` and `isolation` (agent-judge). |
| `VerifierDocumentParseError` | `… must be a safe relative path` | Write that path in `verifier.md` relative to the verifier folder, without a leading `/` or any `..` part. |

| Error class | Where | What it covers |
|---|---|---|
| `AgentManifestError` | `benchflow.agents.manifest` | A manifest.toml is unreadable, missing a required field, declares an unsupported contract major version, or collides with an existing agent. |
| `AssessmentError` | `benchflow.assessment.errors` | Base class: an assessment operation was refused and nothing was appended. |
| `BranchPlanError` | `benchflow.branch_run` | The branch request is inconsistent; nothing was started. |
| `ConfigKeyError` | `benchflow._utils.config_keys` | A run config names a key BenchFlow does not read. |
| `Conflict` | `benchflow.assessment.errors` | The write conflicts with what the log already holds (HTTP 409). |
| `CorruptSidecar` | `benchflow.assessment.errors` | The stored event log does not verify; nothing is materialized from it. |
| `DatasetDigestMismatchError` | `benchflow._utils.dataset_registry` | Snapshot content does not match the registry's pinned digests. |
| `DatasetResolutionError` | `benchflow._utils.dataset_registry` | A dataset spec could not be resolved against the registry. |
| `DiskFullError` | `benchflow._utils.disk_space` | BenchFlow could not write a job file: the host's disk is full. |
| `EmptyTaskSelectionError` | `benchflow.evaluation` | Raised when task discovery + include/exclude filters resolve to zero tasks. |
| `EnvironmentRegistryError` | `benchflow._utils.env_registry` | Raised when an environment spec cannot be resolved. |
| `EvalPlanError` | `benchflow.eval_plan` | A ``bench eval run`` validation failure. |
| `EvidenceRejected` | `benchflow.assessment.errors` | A saved result, output, review artifact, rubric, protocol or collector record is missing, changed, or belongs to another task or assessment. |
| `FileNotFoundError` | `builtins` | File not found. |
| `GitRemoteCloneError` | `benchflow._utils.benchmark_repos` | A connection failure, or a requested branch missing from the remote. |
| `GitRemoteCommandError` | `benchflow._utils.benchmark_repos` | A connection failure, or a requested ref missing from the remote. |
| `HubError` | `benchflow.hub.client` | A Hub could not be reached, refused a request, or answered something unreadable. |
| `ImageBuildError` | `benchflow.sandbox.image_build` | The task's image did not build, for a reason another attempt repeats. |
| `InterruptedJobError` | `benchflow.jobs` | A job folder whose trials were all stopped before any finished. |
| `InvalidRecord` | `benchflow.assessment.errors` | A record, payload or input document breaks its schema. |
| `IsADirectoryError` | `builtins` | Operation doesn't work on directories. |
| `JobMatrixError` | `benchflow.evaluation` | A YAML job config of several jobs (agents x datasets x n_attempts) loaded as one Evaluation: the first job alone would be a fraction of the run the file asks for. |
| `KeptServiceFreezerUnavailable` | `benchflow.sandbox.protocol` | The sandbox cannot hold the services a task keeps: raised by the freezer probe before agent code runs, and, once identities are measured, for a non-root agent that keeps services where the sandbox or its Engine cannot give the freezer. |
| `MalformedTaskError` | `benchflow.evaluation` | A single-task input whose ``task.md`` (or legacy ``task.toml``) exists but fails to parse (#3). |
| `MissingCredentialError` | `benchflow.errors` | The agent's model needs a login or an API key that is not set. |
| `NoEvaluationRecordError` | `benchflow.evaluation` | ``Evaluation.resume`` was given a folder with no evaluation.json. |
| `NotADirectoryError` | `builtins` | Operation only works on directories. |
| `OwnModelRouteError` | `benchflow.task.plan_validation` | A run an agent that routes its own model calls cannot honor. |
| `PermissionError` | `builtins` | Not enough permissions. |
| `Pre08CopyError` | `benchflow.evaluation` | A 0.7 job's own files could not be kept before resuming it rewrites them. |
| `ProxyExtraRequirementError` | `benchflow.providers.proxy_requirements.__init__` | An extra requirement the proxy cannot take on top of its locked set. |
| `RegistryAuthError` | `benchflow.sandbox.image_build` | The registry refused this machine's login for the task's base image. |
| `RemoteDockerConfigError` | `benchflow.sandbox.remote_docker` | The remote Docker host is not configured, or configured unsafely. |
| `RepositoryNotFoundError` | `benchflow._utils.benchmark_repos` | A --source-repo GitHub does not show: misspelled, or private. |
| `ResumeMismatchError` | `benchflow.evaluation` | Raised when a named job's earlier trials ran with other settings, or when they could not be set aside. |
| `ReviewRubricError` | `benchflow.review.config` | Raised when a rubric file cannot be loaded or is not a valid rubric. |
| `RunCardError` | `benchflow.run_card` | A run-reference.toml that cannot be read, or a strict reference the run differs from without saying why. |
| `RunFolderError` | `benchflow.continue_run.run_folder` | Raised when a run folder is missing required artifacts or malformed. |
| `SeparateVerifierRefusedError` | `benchflow.integrity.trial` | The run moves every verifier into the separate verifier sandbox, and some selected tasks cannot use it there. Raised before any job or shard folder exists and before any sandbox starts. The message ends with the next step. |
| `StaleExpectation` | `benchflow.assessment.errors` | The writer's ``expected_event_seq``/``expected_previous_event_sha256`` no longer names the log's last event: another write came first. |
| `TaskConfigKeyError` | `benchflow.task.imports` | A task.toml carries an unknown key the run refuses (typo/decision table). |
| `TaskDocumentParseError` | `benchflow.task._document_normalize` | Raised when a ``task.md`` document cannot be parsed. |
| `TaskEnvNotAllowedError` | `benchflow.task.env` | A task asks for variables of the operator's environment nobody named. |
| `TransitionRefused` | `benchflow.assessment.errors` | The event is not allowed in the assessment's current state. |
| `UsageLimitError` | `benchflow.agents.errors` | The login the agent runs on has used up its subscription usage. |
| `VerifierDocumentParseError` | `benchflow.task.verifier_document` | Raised when ``verifier/verifier.md`` cannot be parsed. |

## Command errors

Errors a `bench` command prints before it exits, with the `next:` line it prints after them. The command module is the file under `benchflow/cli/` that prints it.

| Command module | Message | What to do |
|---|---|---|
| `_errors` | (built at run time) | check the path, and that you can read it |
| `_hosted_env` | (built at run time) | name the dataset as name@version; `bench hub list` lists them |
| `_hosted_env` | (built at run time) | references are provider:owner/name; the PrimeIntellect hub needs the `prime` CLI installed and signed in |
| `_hosted_env` | `--json shows a dataset of BenchFlow's registry; a hosted environment's metadata is text` | drop --json, or pass a dataset as name@version |
| `_hosted_env` | `--owner is not supported for the benchflow hub` | drop --owner, or narrow the list with --search |
| `_hosted_env` | `--provider … is an unknown value; choose one of: …` | pass one of those to --provider |
| `_hosted_env` | `Dataset … is not in BenchFlow's registry…` | `bench hub list` lists the datasets and their versions |
| `_hosted_env` | `could not load BenchFlow's registry: …` | check this machine's network, then run the command again |
| `_hub_client` | (built at run time) | check the Hub address and your key (scope upload) |
| `_hub_client` | (built at run time) | check the name and version |
| `_hub_client` | (built at run time) | name the dataset as org/name@version |
| `_hub_client` | (built at run time) | name the version as org/name@version, e.g. robouse/metaworld-sample@0.6.1 |
| `_hub_client` | (built at run time) | point at a folder that holds one folder per task |
| `_hub_client` | `… answered the upload without a run: …` (0.10) | nothing on the Hub names this job; check the Hub's log, then push again |
| `_hub_client` | `… has no dataset registry, so it can't link this run to …; nothing was uploaded` (0.10) | push it to a Hub that has the dataset registry: bench hub push &lt;job&gt; --hub &lt;url&gt; |
| `_hub_client` | `… holds … runs: … and more` (0.10) | push one run: bench hub push … |
| `_hub_client` | `… is not a finished job folder (no summary.json)` (0.10) | pass a job folder under ./jobs (or --jobs-dir) that bench run wrote |
| `_shared` | (built at run time) | `bench explain R213` says more |
| `_shared` | `--allow-task-env takes one variable name (letters, digits and _), not NAME=VALUE or a list; the value stays in your environment` | Pass one variable name after --allow-task-env; repeat the option for additional names. |
| `_shared` | `Invalid env var: …` | pass it as KEY=VALUE |
| `_shared` | `No such file or directory: …` | Pass a path that exists; the `next:` line under the error names the file or folder the command wants there. |
| `adopt` | (built at run time) | name the benchmark with --name &lt;slug&gt; |
| `adopt` | `--verify and --scaffold-only are mutually exclusive` | pass one of them |
| `adopt` | `missing target` | pass a SOURCE (convert) or a benchmark SLUG (--verify / --scaffold-only) |
| `agent` | (built at run time) | `bench agents` lists the agents |
| `agent` | `Agent catalog incomplete (…): …` | the agents listed above work; BENCHFLOW_AGENTS_SOURCE names the catalog (off skips it) |
| `assessment` | `… already exists; a reference is never overwritten` (0.10) | pass a new --out path |
| `assessment` | `… has no label yet (state …)` (0.10) | `bench assess reduce <sidecar>` records one once there is an answer |
| `assessment` | `…'s label file differs from the label its log rebuilds` (0.10) | `bench assess label <sidecar> --write` rewrites it from the log |
| `branch` | (built at run time) | `bench branch --help` shows the --child form and the flags |
| `branch` | (built at run time) | pass a job folder or a folder of jobs (`bench run` writes them under ./jobs, or --jobs-dir) |
| `cache` | (built at run time) | `bench cache --help` lists the kinds, backends and ages it takes |
| `continue_cmd` | (built at run time) | pass a timed-out run's folder (it holds config.json and trajectory/llm_trajectory.jsonl) |
| `continue_cmd` | `No such file or directory: …` | pass a timed-out run's folder (it holds config.json and trajectory/llm_trajectory.jsonl) |
| `continue_cmd` | `not a directory: …` | pass a timed-out run's folder (it holds config.json and trajectory/llm_trajectory.jsonl), or a folder of them |
| `continue_cmd` | `path does not exist: …` | pass a timed-out run's folder (it holds config.json and trajectory/llm_trajectory.jsonl), or a folder of them |
| `daemon` | (built at run time) | `bench daemon stop` (with the installation that started it) or wait for it to stop, then start again |
| `daemon` | `Local runtime is busy: … active calls.` | wait for them to finish, then `bench daemon stop` again |
| `daemon` | `Local runtime is not running.` | `bench daemon start` starts it |
| `doctor` | (built at run time) | `bench eval smoke --help` describes --agent |
| `doctor` | `Daytona is not ready for --sandbox daytona; no job was created.` | run `bench doctor --sandbox daytona` for the full report, or set BENCHFLOW_SKIP_PREFLIGHT=1 to skip this check |
| `doctor` | `Docker is not ready for --sandbox docker; no job was created.` | run `bench doctor` for the full report, or set BENCHFLOW_SKIP_PREFLIGHT=1 to skip this check |
| `doctor` | `No agent to smoke: no working credential found.` | the oracle needs no login: `…`; to smoke a model agent, log in (`bench doctor` shows how) or pass --agent NAME[=MODEL] |
| `doctor` | `No job was created. …` | Pass `--agent` the name the message suggests, or one `bench agents` lists. |
| `doctor` | `The remote Docker host is not ready for --sandbox remote-docker; no job was created. …` | check BENCHFLOW_REMOTE_DOCKER_HOST (or DOCKER_HOST) and that `docker info` works against that host, or set BENCHFLOW_SKIP_PREFLIGHT=1 to skip this check |
| `doctor` | `This machine failed checks the smoke needs; no smoke ran.` | fix the failed checks above (`bench doctor` shows the full report), then run the smoke again |
| `doctor` | `Unknown sandbox …` | pass --sandbox … |
| `doctor` | `… No job was created.` | Set the login or key the message names (export it or add it to `.env`), then rerun; `bench doctor` lists the logins and keys it finds. |
| `doctor` | `… No job was created.` | pass --model with a model this agent supports |
| `doctor` | `… No job was created.` | pass a --model whose provider serves this agent's protocol, or an --agent that speaks one the provider supports |
| `doctor` | `… of … smoke runs did not pass.` | `bench view …` shows what each agent did |
| `doctor` | `… refused the key in … (HTTP …); no job was created.` | check the key with `bench doctor`, or set …=0 to skip this check |
| `embodied` | (built at run time) | pass the job folder of a seeded embodied run (`bench run --seeds` writes one) |
| `embodied` | `invalid embodiment spec: …` | fix the spec; `robo info --json` prints a valid one |
| `embodied` | `no embodied episodes in …` | pass the job folder of an embodied run |
| `embodied` | `no embodied trials in …` (0.10) | pass the job folder of an embodied run |
| `embodied` | `no job directory …` | pass a job folder or a folder of jobs (`bench run` writes them under ./jobs, or --jobs-dir) |
| `eval_artifacts` | (built at run time) | fix the --matrix file; docs/audit/compare-runs.md#choose-the-metric shows one |
| `eval_artifacts` | `selected task count … != expected …` | Compare the tasks listed in the `--task-manifest-out` file with what you meant, then fix `--include`/`--exclude` or `--expected-tasks`. |
| `eval_bundle` | (built at run time) | pass a job folder or a trial folder |
| `eval_bundle` | `No such file or directory: …` | pass a job folder or a trial folder |
| `eval_lift` | (built at run time) | `bench eval compare-lift --help` describes the two job folders it pairs |
| `eval_regrade` | `No such file or directory: …` | pass a job or trial folder of a run with --freeze-workspace |
| `eval_regrade` | `Regrade failed: …` | pass a job or trial folder of a run with --freeze-workspace |
| `eval_regrade` | `the reviewer options (--reviewer-model and the rest) apply only with --review` (0.10) | add --review to rerun each trial's rubric review |
| `explain` | `no explain page for …` | `bench explain` lists every code |
| `explain` | `no explain page for …` | its pages are … |
| `hub` | (built at run time) | check this machine's network (or its proxy), then run the command again |
| `hub` | (built at run time) | pass a registry JSON file or URL to --registry |
| `hub` | `No such file or directory: …` | pass a folder that holds one folder per task |
| `hub` | `No such file or directory: …` | pass a job folder under ./jobs (or --jobs-dir) that bench run wrote |
| `hub` | `No such file or directory: …` | pass an existing registry JSON file or URL to --registry |
| `hub` | `Registry check failed: …` | check --registry and this machine's network, then run the check again |
| `hub` | `… is not valid JSON` | pass a registry JSON file or URL to --registry |
| `main` | (built at run time) | `bench check <folder>` lists the tasks it finds; check --include and --exclude against their names |
| `main` | (built at run time) | `bench check <task folder>` validates the tasks; `bench doctor` checks this machine |
| `main` | (built at run time) | `bench eval metrics --help` describes --k and --solve-threshold |
| `main` | (built at run time) | `bench hub show …` lists the dataset's tasks; check --include and --exclude against their names |
| `main` | (built at run time) | `bench run --help` describes --checkpoints and --retry-from-checkpoint |
| `main` | (built at run time) | `bench run --help` describes --source-env and its flags |
| `main` | (built at run time) | `bench run --help` lists the --integrity modes |
| `main` | (built at run time) | `bench run --help` lists the flags |
| `main` | (built at run time) | check the --dataset name@version and --registry |
| `main` | (built at run time) | check the script path given to --agent |
| `main` | (built at run time) | fix the --matrix file; docs/audit/compare-runs.md#choose-the-metric shows one |
| `main` | (built at run time) | fix the --matrix file; docs/reference/pass-at-k.md shows one |
| `main` | (built at run time) | fix the --separate-verifier-include path |
| `main` | (built at run time) | fix the task folder |
| `main` | (built at run time) | pick a dataset version validated for this version of BenchFlow, or re-run with --ignore-bench-version to proceed anyway |
| `main` | (built at run time) | the run's output and logs are in … |
| `main` | `--source-path/--source-ref require --source-repo` | add --source-repo, or drop --source-path and --source-ref |
| `main` | `Could not open a Daytona client: …` | `bench doctor --sandbox daytona` checks the SDK and the key |
| `main` | `Could not resolve --source-repo …: …` | Check the repository name, `--source-path` and `--source-ref`; for a private repository run `gh auth login`; if offline, fix the network and rerun. |
| `main` | `Could not resolve --source-repo …: …` | check the repository name, --source-path and --source-ref, and that this machine can clone it |
| `main` | `Failed to fetch source repo for …: …` | Check this machine's network or proxy and rerun; if the ref is missing, set `source.ref` in the config to a branch, tag or commit the repository has. |
| `main` | `Failed to fetch source repo for …: …` | check source.repo and its ref in the config, and that this machine can clone it |
| `main` | `Invalid eval config …: …: …` | fix the config; its keys are in `https://github.com/benchflow-ai/benchflow/blob/main/docs/scale/configs.md` |
| `main` | `No such file or directory: …` | pass an existing registry JSON file or URL to --registry |
| `main` | `No such jobs directory: …` | pass a job folder or a folder of jobs (`bench run` writes them under ./jobs, or --jobs-dir) |
| `main` | `No tasks to run: no --config, --tasks-dir, --source-repo, --source-env or --dataset` | name a task, for example `bench run benchflow-ai/skillsbench/tasks/econ-detrending-correlation`; `bench run --help` describes each source |
| `main` | `Not a directory: …` | pass a job folder or a folder of jobs (`bench run` writes them under ./jobs, or --jobs-dir) |
| `main` | `Not a directory: …` | pass a task folder or a folder of tasks to --tasks-dir |
| `main` | `Skipping branch snapshot cleanup: …` | `bench doctor --sandbox daytona` checks the Daytona key |
| `main` | `Unsupported eval agent protocol: …` | name an ACP agent; `bench agents` lists them |
| `main` | `` `bench …` was BenchFlow 0.8's; name one of its commands `` | the detail maps each 0.8 command to its spelling here |
| `main` | `daytona SDK import failed: …` | `bench doctor --sandbox daytona` checks the SDK and the key |
| `main` | `daytona SDK not installed` | install it with `…` |
| `main` | `…: its agents: list names … agents, and … would run one agent in place of all of them` | drop the flag, or edit the agents: list |
| `main` | `…: its tasks: list names the tasks to run (…); --include and --exclude would change that list` | drop --include/--exclude, or edit the tasks: list |
| `main` | `…: … of … cells select no task (…), so they would run nothing` | check --include/--exclude and each dataset's task_names against its tasks (bench check &lt;folder&gt; lists them) |
| `new_verbs` | `--agent must not be empty or whitespace-only` | pass -a &lt;agent&gt; (`bench agents` lists them); to run the oracle, omit -a and model-only flags |
| `new_verbs` | `--harness … is not a harness` | pass acp or native |
| `new_verbs` | `--hub and --registry both name where --dataset comes from` (0.10) | keep one: --hub &lt;url&gt; is --registry hub:&lt;url&gt; |
| `new_verbs` | `--hub reads a --dataset from the Hub, and none was given` (0.10) | bench run -d &lt;org/name@version&gt; --hub … |
| `new_verbs` | `--idle-timeout … is not a duration` | pass seconds (600) or a number with s, m or h (90s, 10m, 1h); 0, none or null turns it off |
| `new_verbs` | `--json, --attempts, --include-controls and --by read one job or compare two` | name the job: bench results &lt;job&gt; --json |
| `new_verbs` | `--task-skills and --skill-mode both set the skill mode` | keep one: --task-skills is --skill-mode with-skill |
| `new_verbs` | `bench results reads one job or compares two; … were given` | bench results &lt;job&gt; &lt;job&gt; |
| `new_verbs` | `bench run needs a target: a task folder, a folder of tasks, or a repository` | bench run tasks/ (or --dataset &lt;name&gt;@&lt;version&gt;, or --config run.yaml) |
| `new_verbs` | `bench run takes a target or --dataset, not both` | drop the target to run the dataset, or drop --dataset |
| `new_verbs` | `bench run takes a target or …, not both` | drop … (the target names the tasks) or drop the target |
| `new_verbs` | `bench run takes one target; … were given` | run each target on its own, or put the tasks in one folder |
| `new_verbs` | `no such task folder: …` | pass a task folder, a folder of tasks, or a repository as org/repo[/path][@ref] |
| `new_verbs` | `… …, but no agent runs it: without -a, bench run runs the oracle (the task's reference solution)` | add -a &lt;agent&gt; (`bench agents` lists them), or drop … to run the oracle |
| `preferences` | (built at run time) | `bench preferences --help` lists each command's inputs |
| `rescore` | `Scoring failed: …` | pass a saved trial folder and the task it ran with --tasks-root |
| `rescore` | `Scoring incomplete: …` | fix that, then score again with `bench eval score --force` |
| `results` | (built at run time) | `bench check <task folder>` validates the tasks; `bench doctor` checks this machine |
| `results` | (built at run time) | `bench eval compare --help` describes --vary, --by and --on-mismatch |
| `results` | (built at run time) | pass --by with the keys the message lists |
| `results` | (built at run time) | pass a job folder or a folder of jobs (`bench run` writes them under ./jobs, or --jobs-dir) |
| `results` | (built at run time) | pass the job folder of a run that stopped before it finished |
| `results` | `--attempts … is not a choice` | pass --attempts best or --attempts all |
| `results` | `--on-mismatch is warn, raise or ignore; --attempts is best or all` | pass one of those values |
| `review` | (built at run time) | `bench review --help` describes the rollout path and --rubric |
| `review` | (built at run time) | pass -a with a name `bench agents` lists |
| `review` | (built at run time) | pass a finished rollout directory or a job directory |
| `review` | `--passing and --failing cannot be used together` | pass one of them, or neither to review every rollout |
| `review` | `No such file or directory: …` | pass an existing path to … |
| `review` | `Unknown sandbox: …` | pass --sandbox with one of: … |
| `reviewer_options` | (built at run time) | `bench run --help-all` says what each flag takes |
| `reviewer_options` | `expected whole seconds, 0 or 'none'` | `bench run --help-all` says what each flag takes |
| `sandbox` | (built at run time) | `bench doctor --sandbox <name>` checks a sandbox |
| `sandbox` | `--all deletes every sandbox of one owner, and no owner is set` | set BENCHFLOW_DAYTONA_OWNER (cleanup never touches sandboxes without your owner label) |
| `sandbox` | `Could not attach to the kept sandbox: …` (0.10) | `bench sandbox release <trial>` deletes it if it is broken |
| `sandbox` | `Could not delete Daytona sandbox …: …` (0.10) | `bench sandbox cleanup --id <sandbox id>` deletes it |
| `sandbox` | `Daytona cleanup failed: …` | `bench doctor --sandbox daytona` checks the Daytona key |
| `sandbox` | `Docker did not remove the … of …: …; the sandbox is still marked kept` (0.10) | `docker … ls --filter label=com.docker.compose.project=…` shows what is left; run `bench sandbox release` again once it is free |
| `sandbox` | `Docker still lists containers of …` (0.10) | `docker ps -a --filter label=com.docker.compose.project=…` shows them |
| `sandbox` | `No kept sandbox at …: …` (0.10) | pass the trial folder of a run with --keep-sandbox (it holds kept-sandbox.json) |
| `sandbox` | `Not a directory: …` | pass a task folder (it holds task.md or task.toml) |
| `sandbox` | `Not a valid task directory …: …` | pass a task folder (it holds task.md or task.toml) |
| `sandbox` | `Skipping AgentCore cleanup: …` | check this machine's AWS credentials and BENCHFLOW_AGENTCORE_REGION |
| `sandbox` | `The kept sandbox … has no main container any more` (0.10) | `bench sandbox release <trial>` clears the record |
| `sandbox` | `The sandbox in … was released at …` (0.10) | run the task again with --keep-sandbox to get a new one |
| `sandbox` | `… is not a kept-sandbox record` (0.10) | pass the trial folder of a run with --keep-sandbox (it holds kept-sandbox.json) |
| `skills` | (built at run time) | `bench doctor` checks this machine and its logins |
| `skills` | (built at run time) | fix …; `bench skills eval --help` shows its form |
| `skills` | (built at run time) | pass a skill folder (it holds SKILL.md) or a folder of skills |
| `skills` | `Invalid --fail-on …` | pass --fail-on … |
| `skills` | `Invalid --sandbox …` | pass --sandbox … |
| `skills` | `No evals/evals.json found in …` | create one with test cases; `bench skills eval --help` shows its form |
| `skills` | `No such file or directory: …` | pass a skill folder (it holds SKILL.md) |
| `skills` | `Not a directory: …` | pass an existing skills directory to --dir |
| `skills` | `got … models for … agents` | --model may be provided once for all agents or once per --agent |
| `skills` | `… is not installed` | Install it with the `uv tool install` command the `next:` line prints, or set `BENCHFLOW_SKILLEVALUATOR_BIN` or `BENCHFLOW_SKILLSPECTOR_BIN` to its executable. |
| `skills` | `…/… eval case(s) errored — the eval did not complete cleanly.` | the cases' trials are under …; `bench doctor` checks the logins they need |
| `tasks` | (built at run time) | check --agent, --model and --sandbox-user |
| `tasks` | (built at run time) | check the dataset id, --revision and --path; --overwrite replaces an existing output folder |
| `tasks` | (built at run time) | fix … and check it again |
| `tasks` | (built at run time) | pass a task folder and a new output folder; --overwrite replaces an existing export |
| `tasks` | (built at run time) | pass a task.md task folder, and an --output file path (not a folder) |
| `tasks` | (built at run time) | pass a task.toml task folder; --overwrite replaces an existing task.md |
| `tasks` | (built at run time) | pass one of those to --target |
| `tasks` | (built at run time) | pass two task manifests (`bench run --task-manifest-out` writes one) |
| `tasks` | (built at run time) | pick a --sandbox that runs these features, or change the task |
| `tasks` | (built at run time) | pick another name, or another parent with --dir |
| `tasks` | `--verifier-sandbox: …` | pass task or separate |
| `tasks` | `Cannot read task files under …: …` | make the task's files readable by this user |
| `tasks` | `Missing output_dir` | pass one, or use --report-only |
| `tasks` | `No such file or directory: …` | pass a task folder to export |
| `tasks` | `No such file or directory: …` | pass a task.md task folder |
| `tasks` | `No such file or directory: …` | pass a task.toml task folder |
| `tasks` | `No tasks under …` | pass a folder with task.toml or task.md in it or in its immediate subfolders |
| `tasks` | `Not a directory: …` | pass a task folder or a folder of tasks |
| `tasks` | `Not a directory: …` | pass a task folder or a folder of tasks; `bench new <name>` creates a task |
| `tasks` | `Unknown sandbox: …` | pass --sandbox with one of: … |
| `tasks` | `bench new no longer scaffolds the legacy split layout.` | `bench new <name>` scaffolds a task.md package; `bench import <dir> --remove-legacy` converts an existing split task |
| `trace_import` | (built at run time) | pass a JSONL trace file to --from-file |
| `trace_import` | `--output … is not a directory` | pass a folder (new or existing) to --output |
| `trace_import` | `--task-format … is not a format` | pass --task-format task-md or legacy |
| `trace_import` | `File not found: …` | pass a JSONL trace file to --from-file |
| `trace_import` | `Invalid --outcome …` | pass --outcome success, failure or unknown |
| `trace_import` | `No tasks generated — no matching traces.` | loosen --min-steps or --outcome |
| `trace_import` | `No traces found.` | check the source; `bench tasks list-sources` lists the known --from-hf datasets |
| `trace_import` | `Not a directory: …` | pass an existing projects directory to --projects-dir |
| `trace_import` | `Only one source allowed at a time` | pass one of --from-local, --from-file and --from-hf |
| `trace_import` | `Specify a source` | pass --from-local, --from-file or --from-hf |
| `trace_import` | `Unknown format: …` | pass --format auto, claude-code, opentraces or claude-messages |
| `train` | (built at run time) | `bench train convert --help` describes --group-by and --group-advantage |
| `train` | (built at run time) | `bench train convert --help` describes each flag |
| `train` | (built at run time) | `bench train run sft --help` describes the flags |
| `train` | (built at run time) | `bench train validate --help` describes the files it reads |
| `train` | (built at run time) | pass a job folder or a folder of jobs (`bench run` writes them under ./jobs, or --jobs-dir) |
| `train` | (built at run time) | pass the job folder of a branched run (`bench branch` writes one) |
| `train` | `--format … is not a training format` | pass --format … or … |
| `train` | `No such file or directory: …` | pass a canonical-selection.json file to --canonical-selection |
| `train` | `No such file or directory: …` | pass a rollout or jobs directory to convert |
| `train` | `No such file or directory: …` | pass the trainer JSONL file to validate |
| `train` | `Prime-RL SFT failed with exit code …` | its manifest and logs are at … |
| `train` | `bench train stream: --format must be 'jsonl'` | pass --format jsonl |
| `train` | `bench train stream: --group-by: …` | `bench train stream --help` describes --group-by |
| `train` | `bench train stream: --group-size must be at least 1` | pass --group-size 1 or more |
| `train` | `bench train stream: …` | Follow the `next:` line printed with it; `bench train stream --help` describes `--format`, `--group-size`, `--group-by` and `--timeout`. |
| `train` | `bench train stream: …` | pass a job folder or a folder of jobs (`bench run` writes them under ./jobs, or --jobs-dir) |
| `train` | `bench train stream: …` | raise --timeout, or check that the job is still running |
| `train` | `bench train stream: …; … rollouts streamed` | `bench eval resume …` runs the trials it did not finish |
| `train` | `bench train validate does not check --format …` (0.10) | load the file with the library it was written for (see bench train convert in docs/reference/commands.md) |
| `train` | `this is a branch run: runs that branch record the ACP trajectory, not an LLM-proxy llm_trajectory.jsonl (…)` | use --format branch-tree |
| `train` | `… does not apply to --format branch-tree (one row per branch child from the ACP events, with advantage = reward - value)` | drop … |
| `train` | `… needs --format branch-tree` | add --format branch-tree, or drop … |
| `traj` | (built at run time) | `bench traj upload --dry-run` checks a trajectory without sending it |
| `traj` | (built at run time) | pass the sha256:… digest `bench traj upload` printed |
| `traj` | `No such file or directory: …` | pass a trajectory .jsonl file or a trial folder |
| `traj` | `bench traj status needs the digest when stdin is not a terminal` | pass it: bench traj status sha256:&lt;digest&gt; (bench traj upload printed it) |
| `traj` | `bench traj upload needs the trajectory when stdin is not a terminal` | pass it: bench traj upload &lt;file.jsonl or trial folder&gt; |
| `traj` | `npx skills add failed.` | use the copy-paste prompt below |
| `traj` | `path not found: …` | Type the path of an existing trajectory `.jsonl` file or trial folder at the prompt; it asks again until one exists. |
| `verbs` | click's error for a command bench doesn't have | `bench --help` lists the commands |
| `verbs` | click's error for a flag the command doesn't have | `bench run --help-all` lists its flags |
| `verbs` | click's error for a value a flag can't take | `bench run --help-all` says what each flag takes |
| `verbs` | click's error for a command bench doesn't have | `… --help` lists its commands |
| `verbs` | click's error for a flag the command doesn't have | `… --help` lists its flags |
| `verbs` | click's error for a value a flag can't take | `… --help` says what each flag takes |

## Problems in a task or config file

A value in a `task.md` header or a run config that BenchFlow refuses gets one line: the file, the line and the key, then the problem, as in `run.yaml:2: concurrenzy: unknown key (did you mean 'concurrency'?)`. Fix the value the line names; each key's type and default are in [Files BenchFlow reads and writes](./files.md).

| File | Problem | What to do |
|---|---|---|
| run config | `agent_setup_timeout_multiplier: …: BenchFlow installs agents with its own time budget, so it cannot scale the agent setup timeout; drop it or set it to 1` | Drop `agent_setup_timeout_multiplier`, or set it to 1. |
| run config | `agents.….n_concurrent: …: agents[…] is the same agent with n_concurrent …, and The same agent shares one pool with one cap; use the same value, or a concurrency_group each` | Give every entry of the same agent the same `n_concurrent`, or give each entry its own `concurrency_group`. |
| run config | `extra_instruction_paths.…: …: file not found (read from …)` | Fix the path in `extra_instruction_paths`; a relative path is read from the folder you run `bench` in, so an absolute path is safest. |
| run config | `extra_instruction_paths: must be a list of files` | Write `extra_instruction_paths` as a YAML list of file paths, one `- path` item per line. |
| run config | `must be a mapping, got …` | Write the named entry as a mapping of `key: value` lines (an `agents` item as `- name: <agent>`), not a list or plain value. |
| run config | `n_attempts: …: must be a whole number >= 1` | Set `n_attempts` to a whole number of 1 or more, or drop it to run once. |
| run config | `orchestrator.kwargs: BenchFlow takes none; drop it` | Drop `orchestrator.kwargs`. |
| run config | `orchestrator.n_concurrent_trials: …: must be a whole number >= 1` | Set `orchestrator.n_concurrent_trials` to a whole number of 1 or more, or drop it (the default is 4). |
| run config | `orchestrator.retry: …: BenchFlow cannot honour this job setting; …` | Drop `orchestrator.retry`; to retry failed trials, run the folder with `bench run --retry-attempts N` instead. |
| run config | `orchestrator.type: …: BenchFlow runs trials on this machine; drop it` | Drop `orchestrator.type`, or set it to `local`. |
| run config | `quiet: …: must be true or false` | Set `quiet` to `true` or `false`. |
| run config | `timeout_multiplier: …: BenchFlow scales the agent, verifier and environment build timeouts with it, but not the agent setup timeout (BenchFlow installs agents with its own time budget); set agent_setup_timeout_multiplier: 1 to run with the rest scaled` | Add `agent_setup_timeout_multiplier: 1` to run with the other timeouts scaled, or set `timeout_multiplier` back to 1. |
| run config | `unknown key (did you mean …?)` | Rename the key to the suggested one, or remove it; `docs/scale/configs.md` lists what a YAML job config reads. |
| run config | `….concurrency_group: …: an agent in a concurrency group must set n_concurrent` | Add `n_concurrent` to that agent entry, or remove its `concurrency_group`. |
| run config | `….env: must be a mapping of names to values` | Write the agent's `env` as a mapping of variable names to values (`NAME: value` lines), not a list. |
| run config | `….n_concurrent: …: agents in concurrency_group … must use the same n_concurrent (…)` | Use the same `n_concurrent` on every agent in that `concurrency_group` (the value in parentheses), or move this agent to another group. |
| run config | `….n_concurrent: …: cannot exceed n_concurrent_trials (…); lower it or raise n_concurrent_trials` | Lower this agent's `n_concurrent` to at most `n_concurrent_trials` (default 4), or raise `n_concurrent_trials`. |
| run config | `….n_concurrent: …: must be a whole number >= 1` | Set the agent's `n_concurrent` to a whole number of 1 or more, or drop it. |
| run config | `….n_tasks: …: must be a whole number >= 1` | Set the dataset's `n_tasks` to a whole number of 1 or more, or drop it to run every task. |
| run config | `….…: must be a list of task names or globs` | Write `task_names` or `exclude_task_names` as a YAML list of strings: task names or globs. |
| run config | `….…: …: BenchFlow cannot honour this job setting; drop it` | Drop the named key from the task entry, or set it back to its default. |
| run config | `….…: …: BenchFlow cannot honour this job setting; …` | Drop the named key, or do what the text after the semicolon says; a key left at its default is accepted. |
| run config | `…: BenchFlow scores a job by its mean reward (and pass@k over n_attempts) only; drop the entry, or compute it from the trials' result.json files` | Drop the `metrics` entry or make it `type: mean` without `kwargs`; compute other metrics from the trials' `result.json` files. |
| run config | `…: …: BenchFlow cannot honour this job setting; …` | Drop the named top-level key; the text after the semicolon says what to use instead, such as `bench run --retry-attempts N` for `retry`. |
| run config | `…: …: must be a finite number above 0` | Set the multiplier to a finite number above 0; 1 keeps the task's own timeouts. |
| run config | `…: …: must be a whole number >= 1` | Set `n_concurrent_trials` to a whole number of 1 or more, or drop it (the default is 4). |
| task.md | `'command' is required for transport 'stdio'` | Give the stdio MCP server a `command` (and `args`), or set its `transport` to `sse` or `streamable-http` with a `url`. |
| task.md | `'url' is required for transport '…'` | Give the MCP server a `url`, or set `transport: stdio` and a `command`; the default transport is `sse`. |
| task.md | `Choose either workspace_recovery or explicit submission_files` | Set either `verifier.workspace_recovery: true` or a `verifier.submission_files` list, not both. |
| task.md | `Conflicting values for solution env var …` | Set the variable once: keep it under `solution.env` and drop the shorthand under `solution`, or give both the same value. |
| task.md | `Invalid TPU topology …; expected e.g. '2x4' or '2x2x1'` | Set `sandbox.tpu.topology` to positive whole numbers joined by `x`, such as `2x4` or `2x2x1`. |
| task.md | `Invalid size format: …. Expected format like '1G', '512M', etc.` | Write `memory` or `storage` as a number ending in G, M or K (`2G`, `512M`), or use `memory_mb`/`storage_mb` instead. |
| task.md | `MCP tool filters must contain at least one entry` | Give the MCP server's `tools`, `include_tags` or `exclude_tags` at least one non-blank entry, or remove the empty list. |
| task.md | `Package name must be in 'org/name' format with alphanumeric characters, hyphens, underscores, and dots. Cannot start with a dot or contain '..'. Got: …` | Set `task.name` to `org/name`, such as `benchflow/hello-world`, using letters, digits, dots, hyphens and underscores; no leading dot or `..`. |
| task.md | `Task config cannot contain both 'oracle' and legacy 'solution'; use 'oracle' for native tasks or 'solution' only when importing legacy task.toml tasks.` | Keep `oracle` in a native `task.md` and remove the legacy `solution` section. |
| task.md | `Unsupported schema_version major … (from …); supported major version(s): …` | Set `schema_version` to a 1.x version, such as the default `1.3`; this BenchFlow reads only major version 1. |
| task.md | `[verifier].sandbox_mode='shared' is incompatible with [verifier.sandbox]` | Remove `verifier.sandbox`, or set `verifier.sandbox_mode: separate` to run the verifier in that sandbox. |
| task.md | `allow_internet=False contradicts the explicit network_mode=…; drop the deprecated allow_internet field and rely on network_mode (use network_mode='no-network' to disable networking).` | Remove `allow_internet` and keep `network_mode`; set `network_mode: no-network` to turn networking off. |
| task.md | `allowed_hosts entry … is not a valid CIDR range` | Write the entry as a strict CIDR range with zero host bits (`10.0.0.0/8`), a single IP, or a hostname without a path. |
| task.md | `allowed_hosts is only valid for network_mode='allowlist'` | Add `network_mode: allowlist` beside `allowed_hosts`, or remove `allowed_hosts`. |
| task.md | `allowed_hosts wildcards must be a single leading '*.' label` | Use a single leading `*.` wildcard (`*.example.com`) and list other hosts in full; no other `*` is allowed. |
| task.md | `artifact … from service … must be an absolute path` (0.10) | Give the artifact's path in that service from the root, such as `/app/out.json`. |
| task.md | `blocked_hosts entries must name a hostname, not an IP` | Replace the IP address in `blocked_hosts` with the hostname to block. |
| task.md | `blocked_urls and blocked_hosts are only valid for network_mode='denylist'` | Add `network_mode: denylist` beside `blocked_urls`/`blocked_hosts`, or remove those lists. |
| task.md | `blocked_urls entries must be non-empty URLs` | Remove the empty entry from `blocked_urls`. |
| task.md | `blocked_urls entries must name a hostname, not an IP` | Use a hostname, not an IP address, in each `blocked_urls` entry. |
| task.md | `blocked_urls entries must not contain a port` | Remove the `:port` from the `blocked_urls` entry. |
| task.md | `blocked_urls entries must not contain userinfo` | Remove the `user@` part (userinfo) from the `blocked_urls` entry. |
| task.md | `blocked_urls entries must use the http or https scheme` | Use `http://` or `https://` in `blocked_urls` entries, or leave the scheme out; `https://` is assumed. |
| task.md | `metadata.x-allow-cloud-metadata must be true or false, not …` | Set `metadata.x-allow-cloud-metadata` to `true` or `false`, or use `sandbox.allow_cloud_metadata: true`, the current spelling. |
| task.md | `network_mode='denylist' requires blocked_urls or blocked_hosts` | Add a `blocked_urls` or `blocked_hosts` list beside `network_mode: denylist`, or choose another `network_mode`. |
| task.md | `optional_tools must contain nonempty program basenames` | List bare program names in `verifier.optional_tools` (`curl`, not `/usr/bin/curl`), with no empty entries. |
| task.md | `preference.min_agreement must be a number from 0 to 1 (or a table whose value is one)` (0.10) | Set `preference.min_agreement` to a number from 0 to 1, such as `0.6`. |
| task.md | `preference.protocol must be pairwise or rubric (best-of-n is not supported yet), not …` (0.10) | Set `preference.protocol` to `pairwise` or `rubric`. |
| task.md | `preference.randomize_sides must be true or false` (0.10) | Set `preference.randomize_sides` to `true` or `false`, unquoted. |
| task.md | `preference.raters_per_pair must be a whole number from 1` (0.10) | Set `preference.raters_per_pair` to a whole number of 1 or more. |
| task.md | `prompt_prefix must contain non-whitespace text` | Give `agent.prompt_prefix` some text, or remove it. |
| task.md | `pytest_plugin_releases entry … is not a pinned PyPI release (name==version, e.g. pytest-asyncio==0.24.0)` | Pin each `verifier.pytest_plugin_releases` entry as `name==version`, such as `pytest-asyncio==0.24.0`. |
| task.md | `required_tools must contain nonempty program basenames` | List bare program names in `verifier.required_tools` (`python3`, not `/usr/bin/python3`), with no empty entries; `[]` declares none. |
| task.md | `schema_version … is not a valid MAJOR.MINOR version` | Set `schema_version` to `MAJOR.MINOR` with a numeric major, such as `1.3`. |
| task.md | `setup command capture_dir and capture_dir_b64_env must be provided together` | Set both `capture_dir` and `capture_dir_b64_env` on the setup command, or neither. |
| task.md | `setup command capture_dir must be an absolute path` | Make the setup command's `capture_dir` an absolute sandbox path, starting with `/`. |
| task.md | `setup command capture_dir must be non-empty when provided` | Give `capture_dir` an absolute path, or remove the key. |
| task.md | `setup command capture_dir_b64_env must be a valid environment variable name` | Name `capture_dir_b64_env` with capital letters, digits and underscores, not starting with a digit, such as `SEED_ARCHIVE`. |
| task.md | `setup command capture_dir_b64_env_file_var must be a valid environment variable name` | Name `capture_dir_b64_env_file_var` with capital letters, digits and underscores, not starting with a digit. |
| task.md | `setup command capture_dir_b64_env_file_var requires capture_dir_b64_env` | Add `capture_dir_b64_env` (and `capture_dir`) to the setup command, or remove `capture_dir_b64_env_file_var`. |
| task.md | `setup command cwd must be an absolute path` | Make the setup command's `cwd` an absolute path, starting with `/`. |
| task.md | `setup command cwd must be non-empty when provided` | Give `cwd` an absolute path, or remove it. |
| task.md | `setup command host_lock must be non-empty when provided` | Give `host_lock` a name, or remove it. |
| task.md | `setup command must be non-empty` | Give each `sandbox.setup_commands` entry a non-blank `command`, or remove the entry. |
| task.md | `setup command service must be non-empty` | Set `service` to a Compose service name, or remove it to use `main`. |
| task.md | `task config declares both '…' and '…' — they are the same section (…); keep only '…'` | Move the settings from the old spelling into the section the message says to keep, then delete the old section. |
| task.md | `the 'environment' key was renamed to 'sandbox' — rename 'environment:' to 'sandbox:' (legacy task.toml imports convert '[environment]' automatically)` | Rename `environment:` to `sandbox:` in the `task.md` front matter. |
| task.md | `the 'verifier.environment' key was renamed to 'verifier.sandbox' — rename the key (legacy task.toml imports convert '[verifier.environment]' automatically)` | Rename `verifier.environment` to `verifier.sandbox` in `task.md`. |
| task.md | `the 'verifier.environment_mode' key was renamed to 'verifier.sandbox_mode' — rename the key (legacy task.toml imports convert 'environment_mode' automatically)` | Rename `verifier.environment_mode` to `verifier.sandbox_mode` in `task.md`. |
| task.md | `… entries must be hostnames, not URLs, ports, or paths` | Write each entry as a bare hostname, such as `api.example.com`, with no scheme, port or path. |
| task.md | `… entries must be non-empty hostnames` | Remove the empty entry from the list the message names. |
| task.md | `… entries must be valid hostnames containing only letters, digits, hyphens, and dots` | Fix the hostname: labels of letters, digits and hyphens, not starting or ending with a hyphen, joined by dots. |
| task.md | `… … is not a compose service name` (0.10) | Use the name of a service in the task's compose file. |
| task.md | `….network_mode='denylist' is not supported; declare the denylist on sandbox.network_mode` | Set `network_mode: denylist` with `blocked_urls` or `blocked_hosts` under `sandbox`, and remove it from `agent` or `verifier`. |
| task.md or run config | `unknown key (did you mean …?)` | Rename the key on that line to the suggested one, or delete it if BenchFlow has no such setting. |
| task.md or run config | `…: …: this file calls it …` | Rename the key to the name the message gives. |

## Tasks a sandbox can't run

`bench check` and `bench run` refuse a task feature the chosen sandbox can't provide, before any sandbox starts, with code `R401`; the line names the task's key and the reason. Run the task on a sandbox that has the feature (usually `--sandbox docker`), or change what the task asks for.

| Task key | Reason | What to do |
|---|---|---|
| `sandbox.gpus` | `GPU task environments require the Modal sandbox backend` | Run the task with `--sandbox modal`, the only backend that runs GPU tasks, or remove `sandbox.gpus` and `sandbox.gpu_types` if it needs no GPU. |
| `sandbox.tpu` | `TPU scheduling is parsed but not implemented` | Remove `sandbox.tpu` from the task: no BenchFlow sandbox backend schedules TPUs yet. |
| `sandbox.os` | `Windows task environments are parsed but not executable` | Set `sandbox.os` to `linux` or remove it: no BenchFlow sandbox backend runs Windows task environments. |
| `steps` | `a multi-step task's steps are verified in the agent's sandbox, so it cannot use a separate verifier sandbox` (0.10) | Drop `--verifier-sandbox separate`, `--separate-verifier` and `--integrity strict` for this task, and set its `verifier.sandbox_mode` to `shared` without a `[verifier.sandbox]` table. |
| `steps[…].verifier.collect` | `a step's collect hooks are parsed but not executed (only the task's [[verifier.collect]] hooks run)` (0.10) | Move the hooks to the task's `[[verifier.collect]]`; a step's own collect hooks don't run. |
| `verifier.service` | `a verifier in a compose service other than main needs a backend that runs it in that service: …` (0.10) | Run the task with `--sandbox docker` or `--sandbox remote-docker`, or set `verifier.service` to `main`. |
| `agent.network_mode` | `agent.network_mode='allowlist' cannot combine with sandbox.network_mode=…` | With `agent.network_mode = 'allowlist'`, set `sandbox.network_mode` to `public` or `allowlist`, or choose another agent network mode. |
| `verifier.sandbox.transfer` | `artifacts[…] … overlaps verifier-owned …; the separate verifier would receive it there` | Declare the artifact at a path that neither lies under nor contains `/tests`, `/verifier`, `/logs/verifier`, `/logs/agent` or `/opt/runtime`, or run the verifier in the agent's sandbox. |
| `benchflow.verifier` | `benchflow.verifier must be a mapping` | Write `benchflow.verifier` in the `task.md` frontmatter as a mapping (keys such as `path`, `spec`, `entrypoint`), or use the `verifier: <folder>` shorthand. |
| `verifier/verifier.md` | `cannot parse verifier package document: …` | Fix `verifier.md` as the parse error says, then rerun `bench check <task>`. |
| `task.md` | `cannot prove native and legacy definitions equivalent: …` | Fix the file the error says fails to load, or remove the legacy `task.toml` and `instruction.md` beside `task.md` to keep one definition. |
| `sandbox.workdir` | `configured workdir must be a non-empty absolute path` | Set `sandbox.workdir` to an absolute path such as `/app`, or remove it. |
| `sandbox.workdir` | `configured workdir must be an absolute non-root path` | Set `sandbox.workdir` to an absolute path below `/`, such as `/app`; a relative path or `/` itself is refused. |
| `verifier.strategies.…` | `criteria-runner criteria are invalid: …` | Fix the criteria file as the error says: criteria with unique non-empty `id`s, `type` binary, likert or numeric, non-negative weights, and `scoring.threshold` from 0 to 1. |
| `verifier.strategies.…` | `criteria-runner criteria not found at …` | Add the criteria file at that path in the verifier folder, or correct the strategy's `criteria` path in `verifier.md`. |
| `verifier.strategies.…` | `criteria-runner runner not found at …` | Add the runner at that path (`root` plus `entrypoint`, default `reward.py`) in the verifier folder, or correct `root` or `entrypoint` in `verifier.md`. |
| `verifier.strategies.…` | `criteria-runner strategy must declare root` | Add `root`, the runner's folder relative to the verifier folder, to the criteria-runner strategy in `verifier.md`. |
| `## user-persona` | `document user runtime is unsupported` | Add `user.private_facts` with string values and fix any other problem the line names, or remove the `## user-persona` section to run without a simulated user. |
| `benchflow.nudges` | `document user runtime is unsupported` | Add `user.private_facts` with string values and fix any other problem the line names, or remove `benchflow.nudges` to run without a simulated user. |
| `user` | `document user runtime is unsupported` | Add `user.private_facts` with string values and fix any other problem the line names, or remove the `user` table to run without a simulated user. |
| `verifier.strategies.…` | `llm-judge context file not found at …` | Add the context file at that path in the verifier folder, or correct the strategy's `context_file` in `verifier.md` (a relative path without `..`). |
| `verifier.strategies.…` | `llm-judge rubric not found at …` | Add the rubric file at that path in the verifier folder, or correct the strategy's `rubric` in `verifier.md` (a relative path without `..`). |
| `environment/…` | `multi-service compose topologies are not supported by …; use the docker or daytona sandbox` | Run the task with `--sandbox docker`, `remote-docker` or `daytona`, which run Compose services; `modal`, `apple-container` and `agentcore` run one container. |
| `task.md` | `native task.md coexists with an incomplete legacy split definition` | Remove the stray `task.toml` or `instruction.md` beside `task.md`, or complete the pair so it matches `task.md`. |
| `task.md` | `native task.md config and legacy task.toml config differ` | Make `task.toml` match the `task.md` frontmatter, or remove `task.toml` and `instruction.md` and keep `task.md` alone. |
| `task.md` | `native task.md prompt and legacy instruction.md prompt differ` | Make `instruction.md` match the prompt in `task.md` (trailing spaces don't count), or remove `instruction.md` and `task.toml` and keep `task.md` alone. |
|  | `network_mode='allowlist' is not enforced by …` | Run the task with `--sandbox docker`, `remote-docker` or `daytona`, which enforce allowlists, or change the task's `network_mode`. |
| `sandbox.network_mode` | `network_mode='allowlist' with compose services needs a sandbox that checks they have no route out (…); … does not` (0.10) | Run the task with `--sandbox docker` or `--sandbox remote-docker`, the backends that check an allowlist task's Compose services, or drop the allowlist. |
|  | `network_mode='denylist' is not enforced by …` | Run the task with `--sandbox docker`, `remote-docker` or `daytona`, which enforce denylists, or change the task's `network_mode`. |
|  | `network_mode='no-network' is not enforced by …` | Run the task with `--sandbox docker`, `remote-docker`, `daytona` or `modal`, which enforce `no-network`, or change the task's `network_mode`. |
| `artifacts` | `root artifact collection is implemented for docker and daytona` | Run the task with `--sandbox docker`, `remote-docker` or `daytona`, which collect root `artifacts`, or remove `artifacts` from the task. |
| `verifier.sandbox.transfer` | `sandbox.workdir … overlaps verifier-owned …; the separate verifier would receive the workspace there` | Set `sandbox.workdir` to a path that neither lies under nor contains `/tests`, `/verifier`, `/logs/verifier`, `/logs/agent` or `/opt/runtime`, or run the verifier in the agent's sandbox. |
| `verifier.strategies.…` | `script artifact not found at …` | Add the file the script strategy's `command` names at that path in the verifier folder, or correct the path in `command`. |
| `verifier.strategies.…` | `script command does not reference a packaged verifier artifact` | Make the script strategy's `command` in `verifier.md` run a file from the verifier folder by relative path, such as `bash test.sh`. |
| `steps[…].artifacts` | `step artifact transfer is parsed but not implemented` | Remove `artifacts` from each entry in `steps`: no sandbox transfers step artifacts yet. |
| `steps[…].healthcheck` | `step healthchecks are parsed but not implemented` | Remove `healthcheck` from each entry in `steps`: no sandbox runs step healthchecks yet. |
| `steps` | `step names must be unique (each is a folder under steps/)` (0.10) | Give each entry in `steps` a unique `name`, since each step is a folder under `steps/`. |
| `metadata.x-keep-services` | `the tests talk to the services the agent leaves running on port … (from …), and a separate verifier sandbox has none of them, so every trial would score 0; run the verifier in the agent's sandbox (no --verifier-sandbox separate, --separate-verifier or --integrity strict, and no separate verifier sandbox_mode in the task), or remove …` | Run the verifier in the agent's sandbox: drop `--verifier-sandbox separate`, `--separate-verifier` and `--integrity strict`, and set the task's `verifier.sandbox_mode` to `shared`; or remove `metadata.x-keep-services`. |
| `verifier.strategies.….command` | `the verifier package has no …` | Add the file the taskmd strategy's `command` names to the verifier folder, or correct `command` in `verifier.md`. |
| `sandbox` | `unknown sandbox backend; use …` | Set `--sandbox` to one of `docker`, `remote-docker`, `daytona`, `modal`, `apple-container` or `agentcore`. |
| `benchflow.verifier.implementation` | `verifier implementation metadata must be a mapping` | Write `benchflow.verifier.implementation` in `task.md` as a mapping with a `type` such as `test-script`, or remove it. |
| `benchflow.verifier.implementation.strategies` | `verifier implementation strategies must be a mapping` | Write `benchflow.verifier.implementation.strategies` in `task.md` as a mapping keyed by strategy name, such as `script` or `llm-judge`, or remove it. |
| `benchflow.verifier.implementation.type` | `verifier implementation type is parsed but not executable; use test-script/script or declare concrete verifier strategies` | Set `benchflow.verifier.implementation.type` to `test-script`, `script`, `deterministic`, or `hybrid` with known `strategies`; or remove it. |
| `verifier.network_mode` | `verifier network allowlists are not enforced (the verifier runs outside the agent uid firewall)` | Set `verifier.network_mode` to `public`, `no-network` or `denylist`, or remove it: the allowlist firewall binds only the agent's uid, not the verifier. |
| `benchflow.verifier.implementation.strategies.…` | `verifier strategy … is an unknown value; choose one of: script, deterministic, test-script, llm-judge, agent-judge, ors-episode, criteria-runner` | Rename that key under `benchflow.verifier.implementation.strategies` to one of `script`, `deterministic`, `test-script`, `llm-judge`, `agent-judge`, `ors-episode` or `criteria-runner`. |
| `verifier.strategies.…` | `… verifier strategies are parsed but not executable; select script, llm-judge, criteria-runner, agent-judge, or ors-episode` | Set `default_strategy` in `verifier.md` to a `script`, `llm-judge`, `criteria-runner`, `agent-judge` or `ors-episode` strategy. |

## What the verifier step reports

Messages about the verifier and the reward, as a trial's `error`, its `warnings` or a run's log line shows them; a reward of 0 the verifier did not write starts with `Verifier scored 0:`. The verifier's own output is in the trial's `verifier/test-stdout.txt`.

| Module | Message | What to do |
|---|---|---|
| `benchflow.rollout._separate_verifier` | `--separate-verifier-include …: … path(s) under it do not reach the verifier (…)` | If the tests read the listed paths, run the verifier in the agent's sandbox instead; credentials, package caches, special files and links leading outside never reach a separate verifier. |
| `benchflow.rollout._separate_verifier` | `Could not measure the workspace before the agent ran (…); a separate verifier leaves any capture failure unscored` | Read the error in parentheses; if the probe timed out, raise `--sandbox-setup-timeout`, which also bounds this probe. The trial still runs. |
| `benchflow.rollout._separate_verifier` | `Could not move the agent's verifier/ aside: …` | Treat that trial's `verifier/` folder as left by the agent's run, not the verifier; fix the file error shown, then rerun the trial with `bench eval resume <job dir>`. |
| `benchflow.rollout._separate_verifier` | `Separate verifier sandbox failed` | Read the traceback in the run's log, fix the sandbox problem it shows, then rerun the trial with `bench eval resume <job dir>`. |
| `benchflow.rollout._separate_verifier` | `Verifier sandbox cleanup failed` | Remove the leftover verifier sandbox, for example with `bench sandbox cleanup` (lower `--max-age` for a recent one); the trial's score is not affected. |
| `benchflow.rollout._separate_verifier` | `Verifier scored 0: …` | Make the task's `test.sh` leave only regular, readable files in `/logs/verifier`; the error after the colon names what the separate verifier could not publish. |
| `benchflow.rollout._separate_verifier` | `image from … would start compose services (…); a separate verifier sandbox has none` | Give the verifier its own image (a `tests/Dockerfile` or `[verifier.sandbox] docker_image`) instead of the agent's Compose environment, or run the verifier in the agent's sandbox. |
| `benchflow.rollout._separate_verifier` | `the solution's outputs cannot reach the verifier: …` | Expect a 0 when an agent's outputs do this; if the oracle does, fix the solution so the workspace and declared artifacts stay at their declared paths within capture limits. |
| `benchflow.rollout._separate_verifier` | `verifier output publication failed: …` | On a disk or connection error, free space or restore the connection, then run `bench eval resume <job dir>`; otherwise make `test.sh` leave only regular files in `/logs/verifier`. |
| `benchflow.rollout._separate_verifier` | `… in the verifier image is a link that leads to no folder, so … cannot land where it was packed` | In the verifier image, make that path a real folder or a link to an existing folder, then run the task again. |
| `benchflow.rollout._separate_verifier` | `… in the verifier image is a link to …, so … would land at …, in …` | In the verifier image, replace that link with a real folder, or point it outside verifier-owned paths such as `/tests` and system folders, then run the task again. |
| `benchflow.rollout._separate_verifier` | `… resolves to …, which overlaps verifier-owned …; before the agent ran it resolved to …` | Expect a 0 when an agent redirects the workspace or an artifact into a verifier-owned path; if the oracle does, make the solution keep those paths as real folders. |
| `benchflow.rollout._setup` | `; the verifier's PATH left out these folders of the image's PATH: …` | Install the program the test could not find into a root-owned `PATH` folder of the image, such as `/usr/local/bin`; `verifier/trusted-path.json` says why each folder was left out. |
| `benchflow.rollout._setup` | `Invalid verifier rewards: …` | Make the verifier write a finite numeric `reward` (0 to 1 unless `[verifier] reward_range` widens it) with numeric metrics, or set `BENCHFLOW_REWARD_LENIENT=1` to drop non-numeric metrics. |
| `benchflow.rollout._setup` | `Keeping the timed-out verifier's output failed: …` (0.10) | Fix the verifier timeout itself (install test dependencies in the image, or raise `verifier.timeout_sec`); only the partial output kept in `verifier/` is affected. |
| `benchflow.rollout._setup` | `Killing the verifier's processes failed: …` (0.10) | Check the sandbox backend with `bench doctor` if this recurs: the kill command got no answer, so test processes may run on until teardown. |
| `benchflow.rollout._setup` | `Publishing for the verifier: … exited …: …` | Check that root can create folders under `/logs` in the sandbox and the agent's user can write `/logs/agent` (the stderr shown says which); BenchFlow retries once. |
| `benchflow.rollout._setup` | `Publishing for the verifier: … failed: …` | Check the sandbox backend with `bench doctor`: the command could not run there, and BenchFlow retries once before failing the trial's verification. |
| `benchflow.rollout._setup` | `Running verifier...` | Ignore it: this log line marks the start of grading; the verifier's output goes to the trial's `verifier/test-stdout.txt`. |
| `benchflow.rollout._setup` | `Verifier scored 0: …` | Fix the cause after the colon, usually in the task's `test.sh` (its output is in the trial's `verifier/test-stdout.txt`), then check with the oracle: `bench run <task> --fresh`. |
| `benchflow.rollout._setup` | `Verifier start-receipt probe failed; retrying: …` | Wait: BenchFlow keeps the verifier running and probes again; if the trial then ends with a verifier error, check the backend with `bench doctor`. |
| `benchflow.rollout._setup` | `Verifier test command ended (exit status …) but its exec did not return within the budget; scoring its outputs` | Ignore it unless it recurs; the outputs were scored. If it repeats, check the sandbox backend with `bench doctor`, whose exec lost the command's return. |
| `benchflow.rollout._setup` | `Verifier test command start is unconfirmed (…); retrying it once` | Ignore it if the retry succeeds; if it recurs, keep `test.sh` from deleting files in `/run/runtime` and check the backend with `bench doctor`. |
| `benchflow.rollout._setup` | `could not read the verifier's refusal stubs: …` | Run the task again (`bench run --fresh`); unreadable refusal markers count as a refused program, which scores 0. If it repeats, check the backend with `bench doctor`. |
| `benchflow.rollout._setup` | `could not write verifier/trusted-path.json` | Check that the disk holding the job folder has space and the trial's `verifier/` folder is writable; only this PATH record is missing, the score stands. |
| `benchflow.rollout._setup` | `pytest plugin guard not confirmed: it may have failed to load, stopped or refused a plugin; code run by test.sh can also print the diagnostic or change the markers. See verifier/test-stdout.txt and verifier/_runtime_guard_*…` (0.10) | Check `verifier/test-stdout.txt` and `verifier/_runtime_guard_*`; run pytest in `test.sh` without `python -I` or a replaced `PYTHONPATH`, and let it run to the end. |
| `benchflow.rollout._setup` | `pytest plugin guard not confirmed: …; test.sh's reward is not used. See verifier/test-stdout.txt` (0.10) | Check `verifier/test-stdout.txt` and `verifier/_runtime_guard_*`; run pytest in `test.sh` without `python -I` or a replaced `PYTHONPATH`, and let it run to the end. |
| `benchflow.rollout._setup` | `test.sh ran without the image's …: the root verifier clears these names, so a test that runs the image's own tools with them fails; set the value the test needs in test.sh or [verifier.env]` | Set each listed variable the tests need in `test.sh` or the task's `[verifier.env]`, since the root verifier clears the image's values. |
| `benchflow.rollout._setup` | `the verifier receipt response is incomplete or invalid` | Leave `/run/runtime` alone in `test.sh` (BenchFlow records the test's exit status there), and raise `verifier.timeout_sec` if the tests run near their limit. |
| `benchflow.rollout._setup` | `verifier PATH refusal: a refusal marker names …, which only folders the verifier's PATH leaves out have: …` | Install that program into a root-owned `PATH` folder of the image, such as `/usr/local/bin`, or list it in `[verifier] optional_tools` if grading works without it. |
| `benchflow.rollout._setup` | `verifier crashed: …` | Read the cause after the colon: rerun a sandbox or connection failure with `bench eval resume <job dir>`; otherwise fix the task's verifier setup it names. |
| `benchflow.rollout._setup` | `verifier evidence publication failed after 2 attempts` | Check that the sandbox answers commands promptly (`bench doctor`), then run the task again in a new job (`bench run --fresh`); a resume keeps this unscored trial. |
| `benchflow.rollout._setup` | `verifier evidence publication failed after 2 attempts: …` | Fix the sandbox error after the colon (`bench doctor` checks the backend), then run the task again in a new job (`bench run --fresh`); a resume keeps this unscored trial. |
| `benchflow.rollout._setup` | `verifier evidence publication timed out` | Check that the sandbox answers commands promptly (`bench doctor`), then run the task again in a new job (`bench run --fresh`); a resume keeps this unscored trial. |
| `benchflow.rollout._setup` | `verifier evidence publication upload failed: …` | Fix the upload error after the colon (`bench doctor` checks the connection), then run the task again in a new job (`bench run --fresh`); a resume keeps this unscored trial. |
| `benchflow.rollout._setup` | `verifier image-env names not read` | Ignore it: BenchFlow couldn't list the image's environment names, so it skipped the warning about variables the verifier clears; grading is unaffected. |
| `benchflow.rollout._setup` | `verifier start receipt is missing after two attempts: the command may not have started, or test.sh's user may have removed the receipt` | Keep `test.sh` from deleting or replacing files in `/run/runtime`; if it doesn't, the sandbox never started the command, so check it with `bench doctor` and run the task again. |
| `benchflow.rollout._setup` | `verifier start receipt is missing: the command may not have started, or test.sh's user may have removed the receipt` | Keep `test.sh` from deleting or replacing files in `/run/runtime`; if it doesn't, the sandbox never started the command, so check it with `bench doctor` and run the task again. |
| `benchflow.rollout._setup` | `verifier timed out after …s` | Install test dependencies in the image instead of in `test.sh`, or raise the task's `verifier.timeout_sec` if the tests are just slow. |
| `benchflow.rollout._verifier_recovery` | `Invalid verifier recovery reference` | Don't edit the trial's `verification.json` by hand; restore the trial folder from your copy, or run the task again in a new job. |
| `benchflow.rollout._verifier_recovery` | `Missing original solver identity before verifier recovery` | Don't edit the trial's `solver.json` by hand; restore the trial folder from your copy, or run the task again in a new job. |
| `benchflow.rollout._verifier_recovery` | `Task digest mismatch before verifier recovery` | Point `--tasks-root` at the exact task version the trial ran (its `task_digest` in `solver.json`); `bench tasks digest <task>` prints a folder's digest to compare. |
| `benchflow.rollout._verifier_recovery` | `Verifier recovery task digest differs from solver` | Don't mix trial folders: restore this trial's `verifier-recovery/` and `solver.json` from the same run, or run the task again in a new job. |
| `benchflow.rollout._verifier_recovery` | `Verifier recovery unavailable for …: …` | Keep the trial's original verifier error; to let the task's trials be re-verified later, meet the requirement the reason names, such as `[verifier] workspace_recovery = true`. |
| `benchflow.rollout._verifier_recovery` | `Verifier scored 0: …` | Make the task's `test.sh` leave only regular, readable files in `/logs/verifier`; the error after the colon names what recovery could not copy. |
| `benchflow.rollout._verifier_recovery` | `[solver-preserved] verifier pending after completed solver` | Run the task again in a new job (`bench run --fresh`): the run stopped after the agent finished and before grading, and a resume keeps this unscored trial. |
| `benchflow.rollout._verifier_recovery` | `[solver-preserved] verifier recovery …: …` | For `exceeded` or `interrupted`, finish it with `bench eval score <trial folder> --tasks-root <tasks>`; for `failed` or `unavailable`, fix what the text after the colon names. |
| `benchflow.rollout._verifier_recovery` | `task inputs changed while staging verifier recovery` | Leave the task folder unchanged while trials are re-verified, then retry with `bench eval score <trial folder> --tasks-root <tasks>`, using the task version the trial ran. |
| `benchflow.rollout._verifier_recovery` | `verifier output capture failed: …` | On a disk or connection error, fix it and retry with `bench eval score <trial folder> --tasks-root <tasks>`; otherwise make `test.sh` leave only regular files in `/logs/verifier`. |
| `benchflow.task.verifier_core` | `(listed in verifier/…)` | Read the named file in the trial's `verifier/` folder for the refused plugin, then apply the fix the surrounding plugin guard message gives. |
| `benchflow.task.verifier_core` | `Cannot create … for the verifier start receipt; startup of this verifier cannot be confirmed` | Let root create `/run/runtime` in the image (no read-only `/run`, no file at that path); grading goes on, but the verifier's start can't be confirmed. |
| `benchflow.task.verifier_core` | `Cannot read verifier/reward.txt or verifier/reward.json: … (invalid contents or unsafe file)` | Write `reward.txt` or `reward.json` from `test.sh` as a regular, readable UTF-8 file, not a link or FIFO, and without deeply nested JSON. |
| `benchflow.task.verifier_core` | `Could not hand the mounted verifier outputs to the host user: …` | Make `test.sh` leave its reward files readable by others (for example `chmod 644 /logs/verifier/reward.txt`); BenchFlow couldn't hand root-only files on the mount to the host user. |
| `benchflow.task.verifier_core` | `Could not recover verifier output …: …` | Expect that file to be missing from the trial's `verifier/` folder; the reward was recovered. If it repeats, check the sandbox connection with `bench doctor`. |
| `benchflow.task.verifier_core` | `Criteria runner exited with rc=… after writing its reward; the reward is scored (reward files were cleared before this run)` | Ignore it: the reward the criteria runner wrote is scored; exit 0 from the runner if its nonzero exit is not meant as an error. |
| `benchflow.task.verifier_core` | `Criteria-runner verifier directory download failed; recovered canonical verifier output files individually: …` | Ignore it if the trial scored: the reward files were recovered one by one. If it repeats, check the sandbox connection with `bench doctor`. |
| `benchflow.task.verifier_core` | `Failed to add verifier directory to sandbox.` | Check the cause chained to the error: make the task's `tests/` or `verifier/` folder readable here and the sandbox reachable (`bench doctor`), then run the task again. |
| `benchflow.task.verifier_core` | `Failed to capture reward entry …: …` | Check the sandbox connection with `bench doctor` and the cause after the colon, then run the task again; both the folder download and the per-file copy of the reward failed. |
| `benchflow.task.verifier_core` | `Failed to download verifier directory from sandbox: …` | Check the sandbox connection (`bench doctor`), then run `bench eval resume <job dir>`, which reruns trials whose verifier output was lost this way. |
| `benchflow.task.verifier_core` | `Failed to parse rewards from JSON file verifier/reward.json` (0.10) | Make `test.sh` write `/logs/verifier/reward.json` as valid JSON: one object with a numeric `reward` key. |
| `benchflow.task.verifier_core` | `Failed to parse rewards from text file verifier/reward.txt` (0.10) | Make `test.sh` write only a number, such as `1` or `0.5`, to `/logs/verifier/reward.txt`. |
| `benchflow.task.verifier_core` | `No marker for the pytest plugin guard … was read; test.sh's reward is scored as it wrote it (guard loading is not confirmed)` | Ignore it if `test.sh` runs no pytest; otherwise check it reached pytest and ran it in the sandbox itself, not over ssh, in another container or as another user. |
| `benchflow.task.verifier_core` | `No reward file found at verifier/reward.txt or verifier/reward.json` (0.10) | Make `test.sh` write its score to `/logs/verifier/reward.txt` (a number) or `reward.json` before it exits; `verifier/test-stdout.txt` shows where it stopped. |
| `benchflow.task.verifier_core` | `Reward JSON file verifier/reward.json aggregates to reward=…, which does not match scalar reward.txt value …` (0.10) | Write the same score to both files: make `reward.txt` equal what `reward.json`'s metrics aggregate to, or drop `reward.txt` and let `reward.json` decide. |
| `benchflow.task.verifier_core` | `Reward JSON file verifier/reward.json has reward=…, which does not match scalar reward.txt value …` (0.10) | Write the same score to `reward.txt` and to `reward.json`'s `reward` key, or write only one of the two files. |
| `benchflow.task.verifier_core` | `Reward JSON file verifier/reward.json must contain an object with numeric rewards` (0.10) | Write `reward.json` as one JSON object with a numeric `reward` key, not a list, number or string. |
| `benchflow.task.verifier_core` | `Reward JSON file verifier/reward.json …` (0.10) | Fix `reward.json` as the rest of the message says: a finite numeric `reward` in the task's range and numeric `metrics`; `BENCHFLOW_REWARD_LENIENT=1` drops non-numeric metrics instead. |
| `benchflow.task.verifier_core` | `Reward text file verifier/reward.txt must contain a finite numeric reward …` (0.10) | Make `test.sh` write a finite number to `reward.txt`, not `nan` or `inf`. |
| `benchflow.task.verifier_core` | `The verifier.env contains an API key (often the case for LLM-based verifiers). You will incur costs associated with the API calls.` | Ignore it: `[verifier.env]` holds an API key, so the verifier's model calls are billed to it; remove the key if the verifier needs no model. |
| `benchflow.task.verifier_core` | `Unsafe or unreadable reward entry …: …` | Make `test.sh` leave `reward.txt` and `reward.json` as regular, readable files in `/logs/verifier`, not links, folders or unreadable files. |
| `benchflow.task.verifier_core` | `Unsafe verifier output capture: …` | Make `test.sh` (or the criteria runner) leave only regular files and folders in `/logs/verifier`; links and special files there make the capture unsafe. |
| `benchflow.task.verifier_core` | `Verifier directory download failed; recovered canonical verifier output files individually: …` | Ignore it if the trial scored: the reward files were recovered one by one. If it repeats, check the sandbox connection with `bench doctor`. |
| `benchflow.task.verifier_core` | `Verifier exited with rc=… after writing its reward; the reward is scored (reward files were cleared before this run)` | Ignore it: a test script that exits nonzero after writing its reward, such as pytest with failing tests, is scored normally. |
| `benchflow.task.verifier_core` | `Verifier setup failed: kept services cannot run beside a verifier in service …; run it in main` | Set the task's `verifier.service` to `main`: services the task keeps run in the agent's container, so the verifier must run there too. |
| `benchflow.task.verifier_core` | `Verifier setup failed: the verifier runs as uid … while agent code is alive, and … can pass its environment only on a command line the agent can read. Run the verifier as root ([verifier] user), or use a sandbox whose exec passes the environment on stdin (Docker, Apple container).` | Set the task's `[verifier] user` to `root`, or use `--sandbox docker`, `remote-docker`, `modal` or `apple-container`, whose exec passes the environment privately. |
| `benchflow.task.verifier_core` | `Verifier setup failed: the verifier user's uid could not be read (rc=…)` | Make the task's `[verifier] user` an account that exists in the image, or set it to `root`, then run the task again. |
| `benchflow.task.verifier_core` | `Verifier setup failed: … did not run` | Check the sandbox connection with `bench doctor`, then run `bench eval resume <job dir>`; BenchFlow tried the named setup step three times and the sandbox never ran it. |
| `benchflow.task.verifier_core` | `Verifier setup failed: … exited with rc=…` | Check that root can write the folders the named step touches in the image (the tests folder, `/logs/verifier`), then run the trial again with `bench eval resume <job dir>`. |
| `benchflow.task.verifier_core` | `clearing verifier output directory` | In a `Verifier setup failed` error, check that root can empty and recreate `/logs/verifier` in the sandbox, then run `bench eval resume <job dir>`. |
| `benchflow.task.verifier_core` | `criteria-runner exited with rc=…; no reward file found at verifier/reward.txt or verifier/reward.json` (0.10) | Fix the criteria runner so it writes `$BENCHFLOW_REWARD_JSON` (or `$BENCHFLOW_REWARD_TEXT`) before exiting; its output and traceback are in the trial's `verifier/test-stdout.txt`. |
| `benchflow.task.verifier_core` | `criteria-runner strategy with declared criteria must write reward.json metrics` | Have the criteria runner write per-criterion `metrics` to `$BENCHFLOW_REWARD_JSON`, since the strategy declares `criteria`; a `reward.txt` alone is refused. |
| `benchflow.task.verifier_core` | `legacy verifier-mount symlink (… -> …) returned nonzero; verifier scripts hardcoding … may fail` | Make the test scripts find their files through `$TEST_DIR` instead of a hardcoded `/tests`; BenchFlow could not link `/tests` to the verifier folder in this image. |
| `benchflow.task.verifier_core` | `legacy verifier-mount symlink skipped: …` | Make the test scripts find their files through `$TEST_DIR` instead of a hardcoded `/tests`; BenchFlow could not link `/tests` to the verifier folder in this image. |
| `benchflow.task.verifier_core` | `llm-judge rubric not found at …. Set verifier.strategies.<name>.rubric in verifier.md or [verifier.judge].rubric_path in task.toml.` | Put the rubric file at that path, or point `verifier.strategies.<name>.rubric` in `verifier.md` (or `[verifier.judge] rubric_path` in `task.toml`) at it. |
| `benchflow.task.verifier_core` | `llm-judge verifier: … criteria → reward …` | Ignore it: this log line reports the judge's score; the per-criterion verdicts are in the trial's `verifier/reward-details.json`. |
| `benchflow.task.verifier_core` | `ors-episode strategy … produced invalid ORS reward evidence: …` | Make the episode files the strategy's `inputs` name report a finite reward from 0 to 1 with no `metadata.error`; the text after the colon names the problem. |
| `benchflow.task.verifier_core` | ``pytest could not load the plugin guard …: the verifier ran pytest in a Python that cannot import or register it (for example `python -I` or a replaced PYTHONPATH)…; its reward is not scored`` | Run pytest in `test.sh` without `python -I` and without replacing `PYTHONPATH` (append to it instead), so the plugin guard loads. |
| `benchflow.task.verifier_core` | `pytest stopped while the plugin guard … was still checking pytest plugins, so it never decided whether to trust them; listed in verifier/…; this run is not scored` | If the oracle hits this, keep `test.sh` from killing or timing out pytest before the plugin check ends; in an agent's run, it counts as a failure. |
| `benchflow.task.verifier_core` | `target verifier dir` | In a `Verifier setup failed` error, check that root can create `/logs/verifier` in the service `verifier.service` names, then run `bench eval resume <job dir>`. |
| `benchflow.task.verifier_core` | `the pytest plugin guard … crashed inside pytest (…); traceback in verifier/…; its reward is not scored` | Read the traceback in the named `verifier/` file; fix the plugin or test setup it points to, or report a crash inside the guard at `https://github.com/benchflow-ai/benchflow/issues`. |
| `benchflow.task.verifier_core` | `the pytest plugin guard … refused a pytest plugin it could not trust, a release of a plugin BenchFlow vouches for that its table of published releases lacks, so pytest stopped and this run is not scored (…); … was installed by the verifier after the agent stopped, and this machine could not read that release from PyPI before the verifier ran, or PyPI published it since` | Check that this machine reaches pypi.org, then run the task again; to avoid it, install the plugin into the image instead of from `test.sh`. |
| `benchflow.task.verifier_core` | `the pytest plugin guard … refused a pytest plugin it could not trust, so pytest stopped and this run is not scored (…); listed in verifier/…` | If `test.sh` installs the plugin the listed `verifier/` file names, install it into the image instead (`bench check <task>` flags it); if the agent left it, the failure stands. |
| `benchflow.task.verifier_core` | ``the pytest plugin guard … refused a pytest plugin it could not trust, which the verifier installed after the agent stopped from inputs the agent could have changed, so pytest stopped and this run is not scored (…); …. A task problem: BenchFlow trusts such an install only as a release it can check byte for byte. Pin it in [verifier] pytest_plugin_releases (for example `pytest_plugin_releases = ["…"]`) or install it into the image`` | Pin the plugin's release in `[verifier] pytest_plugin_releases` (as `<name>==<version>`), or install it into the task's image instead of from `test.sh`. |
| `benchflow.task.verifier_core` | `the pytest plugin guard … refused a pytest plugin it could not trust, with no complete record of the agent's files (…) to tell whether the verifier installed it after the agent stopped or the agent left it, so pytest stopped and this run is not scored (…); files listed in verifier/…. When test.sh installs pytest plugins, pin their releases in [verifier] pytest_plugin_releases or install them into the image (published pytest-json-ctrf releases load wherever installed)` | If `test.sh` installs pytest plugins, pin their releases in `[verifier] pytest_plugin_releases` or install them into the image; the listed `verifier/` file names the refused files. |
| `benchflow.task.verifier_core` | `the verifier environment BenchFlow computes` | Ignore it: it is the reason BenchFlow records for uploading the verifier's environment file as trusted, not an error. |
| `benchflow.task.verifier_core` | `verifier exited with rc=…; no reward file found at verifier/reward.txt or verifier/reward.json` (0.10) | Read `verifier/test-stdout.txt` for why `test.sh` failed, fix it, and make it write `/logs/verifier/reward.txt` (0 when tests fail) before exiting. |
| `benchflow.task.verifier_core` | `verifier start receipt dir failed: …` | Check the sandbox connection with `bench doctor` if it repeats; without `/run/runtime` the verifier's start can't be confirmed, but grading goes on. |
| `benchflow.task.verifier_core` | `verifier strategy … has type …, which is parsed but not executable yet` | Set `default_strategy` in `verifier.md` to a `script`, `criteria-runner`, `llm-judge`, `agent-judge`, `ors-episode` or `taskmd` strategy. |
| `benchflow.task.verifier_core` | `verifier/reward.json is empty` | Make `test.sh` write a JSON object with a numeric `reward` to `reward.json`, and check the disk holding the job folder isn't full. |
| `benchflow.task.verifier_core` | `verifier/reward.txt is empty` | Make `test.sh` write a number to `reward.txt` (0 when tests fail), and check the disk holding the job folder isn't full. |
| `benchflow.task.verifier_core` | `verifier/… exceeds the 16 MiB reward file limit` | Keep `reward.txt` and `reward.json` under 16 MiB; write bulky details to another file in `/logs/verifier`, such as `reward-details.json`. |
| `benchflow.task.verifier_core` | `writing the verifier env` | In a `Verifier setup failed` error, keep `/`, `/var` and `/var/lib` root-owned real folders not writable by group or others, so root can create `/var/lib/benchflow/verifier-env`. |
| `benchflow.task.verifier_core` | `… is empty: … (… free), so … could not write its reward` | Free space on the disk that holds the job folder, or pass another `--jobs-dir`, then run the task again with `bench run --fresh`; this trial's 0 is not the agent's. |
| `benchflow.task.verifier_core` | ``… was installed after the agent stopped, by the verifier…, where the agent could write; the pytest plugin guard … refused it, so this run is not scored, and neither is any run of this task, the reference solution's included. This is a task problem: `…` flags it. Install the plugin into the image, or pin its release in [verifier] pytest_plugin_releases (for example `["…"]`), which BenchFlow installs into its protected verifier runtime before the agent's turn. Refused: … (listed in verifier/…)`` | Install the plugin into the task's image, or pin its release in `[verifier] pytest_plugin_releases` so BenchFlow installs it into its protected runtime; `bench check <task>` confirms. |

## Warnings a run logs

A run prints these as it works, prefixed with the trial's task. A warning doesn't stop the run; the trial's `warnings` in `result.json` keeps the ones about it.

| Module | Message |
|---|---|
| `benchflow.__init__` | `benchflow.… is deprecated; use benchflow.… (it snapshots a workspace directory, not a sandbox).` |
| `benchflow._paths` | `copytree: skipping symlinked entries under …: …` |
| `benchflow._paths` | `…: refusing to descend into symlinked root …` |
| `benchflow._paths` | `…: skipping non-regular path … (symlink or special file)` |
| `benchflow._paths` | `…: skipping symlink … (refusing to follow)` |
| `benchflow._paths` | `…: skipping symlinked directory … (refusing to follow)` |
| `benchflow._utils.benchmark_repos` | `No source provenance recorded for …: …` |
| `benchflow._utils.result_paths` | `Malformed rewards field in … for task …: expected dict or null, got … … — treating as no reward (task will count as errored)` |
| `benchflow._warm_pool` | `Warm pool could not prepare a trial` |
| `benchflow.acp.client` | `ACP protocol version negotiated to … (client implements …)` |
| `benchflow.acp.client` | `ACPClient received unsupported request … — replying method-not-found` |
| `benchflow.acp.client` | `Agent reported: …` |
| `benchflow.acp.client` | `Error handling agent request …: …` |
| `benchflow.acp.client` | `Error handling notification …: …` |
| `benchflow.acp.client` | `Ignoring ACP update for session … (current session …)` |
| `benchflow.acp.client` | `on_ask_user handler raised …; falling back to auto-approve for session/request_permission` |
| `benchflow.acp.runtime` | `ACP connect failed (attempt …): …` |
| `benchflow.acp.runtime` | `Cannot infer models.dev provider for … — defaulting to anthropic/` |
| `benchflow.acp.selection` | `Invalid …=…; using PTY transport` |
| `benchflow.acp.selection` | `Invalid …=…; using SSH transport for the reviewer` (0.10) |
| `benchflow.acp.selection` | `No ssh client on this machine: Daytona reviewers keep the PTY transport (install OpenSSH, or set …=pty to say so)` (0.10) |
| `benchflow.acp.session` | `Ignoring conflicting ACP child-tool parent attribution` |
| `benchflow.acp.session` | `Unknown tool call status: …` |
| `benchflow.acp.timeout_cleanup` | `ACP prompt task did not finish within …s after cancellation` |
| `benchflow.acp.timeout_cleanup` | `Failed to request ACP prompt cancellation` |
| `benchflow.acp.transport` | `Skipped oversized line: …` |
| `benchflow.acp.watchdog` | `Ignoring …=…: expected whole seconds >= 1; using …x the idle timeout` |
| `benchflow.adapters._toolathlon` | `Toolathlon task …: no MCP config for …; dropping server` |
| `benchflow.agents.install` | `Could not remove agent installer staging: …` |
| `benchflow.agents.install` | `Skills dir not found: …` |
| `benchflow.agents.receipt` | `Could not read …'s installed versions: …` (0.10) |
| `benchflow.batch` | `batch item … not run: …` |
| `benchflow.branch_run` | `Branch trial for …: …` |
| `benchflow.branch_run` | `Job-level results.jsonl aggregation failed: …` |
| `benchflow.branch_run` | `on_event callback failed for …` |
| `benchflow.budget` | `A rollout finished without a USD cost; --max-cost-usd counts only rollouts that report one (see summary.json budget.spent).` |
| `benchflow.budget` | `…; stopping: no new rollouts, … running trial(s) cancelled` |
| `benchflow.budget` | `…; stopping: no new rollouts, … running trial(s) finish` |
| `benchflow.checkpoint_retry` | `Checkpoint … recorded no agent session; the retry starts a fresh one` |
| `benchflow.checkpoint_retry` | `Retry from checkpoint … failed: …` |
| `benchflow.checkpoint_retry` | `Retry from … made no tool calls (reward …): it probably did no work. A fresh session knows only the retry prompt; include @instruction (and @verifier_feedback) in --retry-prompt, or use --retry-resume-session.` |
| `benchflow.checkpoints` | `Checkpoint after prompt … failed (…: …); the run continues` |
| `benchflow.checkpoints` | `Checkpoints requested, but … cannot snapshot; the run continues without them` |
| `benchflow.continue_run.orchestrator` | `Continuation teardown step failed: …` |
| `benchflow.continue_run.replay_proxy` | `replay divergence at turn …: agent sent … messages, recorded turn had …` |
| `benchflow.continue_run.run_folder` | `run … has error_category=…, not a timeout/idle_timeout — continuing it may not be meaningful.` |
| `benchflow.continue_run.run_folder` | `skipping malformed llm_trajectory line …: …` |
| `benchflow.eval_regrade` | `Regrade sandbox cleanup failed` |
| `benchflow.evaluation` | `--verifier-sandbox separate moved … task(s) that declare no artifacts: …. Their verifier gets only the workspace and /logs/artifacts, so an answer kept elsewhere (a service, a database, a file outside the workspace) scores 0 like an agent failure; check each task's oracle under the switch before comparing these zeros with task-mode runs.` |
| `benchflow.evaluation` | `Checkpoint retry of … failed` |
| `benchflow.evaluation` | `Could not persist LearnerStore: …` |
| `benchflow.evaluation` | `Could not write the run card: …` (0.10) |
| `benchflow.evaluation` | `Docker leftover sweep failed: …` |
| `benchflow.evaluation` | `Ignoring malformed … at the tasks-dir root …: …` |
| `benchflow.evaluation` | `Job … also holds results of … task(s) this run does not select (…); its score counts them. Name another job (--job) to score only this selection.` |
| `benchflow.evaluation` | `Job … was made by BenchFlow 0.7 or older (it has no evaluation.json): this run reuses its finished trials and rewrites its summary.json and results files. Its own files are kept in …. Pass --fresh (or another --job-name) to leave it as it was.` (0.10) |
| `benchflow.evaluation` | `Job-level ADP aggregation failed: …` |
| `benchflow.evaluation` | `Job-level results.jsonl aggregation failed: …` |
| `benchflow.evaluation` | `Job-level trainer artifact aggregation failed: …` |
| `benchflow.evaluation` | `Learner store: … skill export failed — skipping generation commit, staying at generation …` |
| `benchflow.evaluation` | `Multiple job directories found (…); resuming into most recent: …` |
| `benchflow.evaluation` | `No run card for this job: …` (0.10) |
| `benchflow.evaluation` | `Resuming into existing job directory: … (finished tasks are reused; --fresh or another --job-name starts a new job, as a new job_name does in Python)` (0.10) |
| `benchflow.evaluation` | `Resuming into existing job directory: … (its settings match the ones this run finally has; finished tasks are reused)` |
| `benchflow.evaluation` | `Resuming without agent_env …: their values are not stored in …. Pass agent_env={...} to Evaluation.resume() if the agent needs them.` |
| `benchflow.evaluation` | `Seeded-rollout report failed: …` |
| `benchflow.evaluation` | `Setting aside … earlier trial(s) of job … in …: they ran …. They are kept there and not counted; this run runs those tasks again with its own settings (bench run --fresh, or another --job-name, keeps two runs as two jobs).` (0.10) |
| `benchflow.evaluation` | `Skipping malformed task …: …` |
| `benchflow.evaluation` | `Skipping task …: …` |
| `benchflow.evaluation` | `Starting a new job …: the latest job in … (…) ran …. Its results stay in their own job; bench eval resume … finishes it with its own settings.` |
| `benchflow.evaluation` | `Starting a new job …: …. An unnamed run never shares a job; pass --job-name to resume that job once it has stopped.` |
| `benchflow.evaluation` | `Taking over a stale job lock from process …, which is gone: …` |
| `benchflow.evaluation` | `The agent's context window is under half the model's: …` (0.10) |
| `benchflow.evaluation` | `The config names agent_env … without values; pass them in agent_env (or the environment) if the agent needs them.` |
| `benchflow.evaluation` | `The job ran pre_agent_hooks …, which … cannot store; pass pre_agent_hooks=[...] to Evaluation.resume() to run them again (without them the run is refused as one with other settings).` (0.10) |
| `benchflow.evaluation` | `The run differs from its reference setup: …` (0.10) |
| `benchflow.evaluation` | `This job's BenchFlow 0.7 summary said mean_reward=…, the mean over scored trials only. BenchFlow 0.8's mean_reward counts every trial, an unscored one as 0: …; its mean_reward_scored_only is 0.7's figure: ….` |
| `benchflow.evaluation` | `Unknown agent … — not in registry. Available: …. Will attempt to use as raw command.` |
| `benchflow.evaluation` | ``… task(s) lost the review or the verifier's sandbox to an infrastructure failure: the trial's result.json names it, and `bench eval score <trial> --tasks-root <tasks> --force` scores it again`` (0.10) |
| `benchflow.evaluation` | `… tasks (…%) failed during verifier dependency install — check verifier_error_category in result.json and fix the task's index policy` |
| `benchflow.evaluation` | `… tasks (…%) had verifier errors — check verifier scripts for bugs` |
| `benchflow.evaluation` | `…job_name …: … is …; this run starts a new job instead` |
| `benchflow.evaluation` | `…n_concurrent_trials: … at the top level wins over orchestrator.n_concurrent_trials: …` |
| `benchflow.experimental.mcp.hooks` | `MCP reviewer server may not be ready on port …` |
| `benchflow.integrity.access` | `access probe exited …: …` |
| `benchflow.integrity.trial` | `Could not load task … for the integrity contract: …` |
| `benchflow.integrity.trial` | `Integrity access probe (…) failed: …` |
| `benchflow.integrity.trial` | `Integrity audit could not be recorded for …: …` |
| `benchflow.integrity.trial` | `Integrity audit failed for … (…: …); its verdict is Rejected, in …/claim_verdict.json` |
| `benchflow.jobs` | `Skipped … unreadable row(s) of … (first: line …)` |
| `benchflow.jobs` | `Skipping unreadable trial …: …` |
| `benchflow.jobs` | `bf.compare: A side mixes several … on … task(s) (e.g. … on …: …), so its reward there is an average over them; pass by=('agent', 'model') to pair like with like.` |
| `benchflow.jobs` | `bf.compare: Settings differ between the sides on … paired task(s): … (…). Pass vary=(...) to declare an intended difference, or on_mismatch='raise' to refuse.` |
| `benchflow.learner_skills` | `Skipping symlinked SKILL.md at … — refusing to follow symlinks` |
| `benchflow.learner_skills` | `Skipping symlinked skill pack … — refusing to follow symlinks` |
| `benchflow.models` | `RolloutResult.messages is deprecated and always empty; read the "agent_message" events in result.trajectory.` |
| `benchflow.models` | `RolloutResult.snapshots is deprecated and always empty; branch checkpoints are recorded in the rollout's tree.json.` |
| `benchflow.models` | `RolloutResult.verified is deprecated; use result.score_outcome in ("passed", "failed").` |
| `benchflow.monitor` | `Monitor mode is not yet implemented — only the API surface is scaffolded. Track progress on https://github.com/benchflow-ai/benchflow/issues/386 ; see docstring on benchflow.monitor for the planned next steps.` |
| `benchflow.native_harness.client` | `Native harness kill failed` |
| `benchflow.native_harness.client` | `Native harness kill of turn … exited …: … …` |
| `benchflow.native_harness.client` | `Native harness parser failed on an event` |
| `benchflow.native_harness.client` | `… did not exit …s after the end of its turn; stopping it` |
| `benchflow.providers.litellm_messages_bridge_patch` | `LiteLLM messages bridge patch not installed` (0.10) |
| `benchflow.providers.litellm_messages_bridge_patch` | `LiteLLM's model map could not be read` (0.10) |
| `benchflow.providers.litellm_messages_bridge_patch` | `documents left as they came` (0.10) |
| `benchflow.providers.litellm_messages_bridge_patch` | `tool result images left as the bridge made them` (0.10) |
| `benchflow.providers.litellm_runtime` | `BENCHFLOW_CAPTURE_TOKEN_LOGPROBS is set but …, so no token ids or logprobs will be recorded (capture runs in the gateway). Use an API key route, e.g. a vllm/ model, for token capture.` |
| `benchflow.providers.litellm_runtime` | `Could not publish …: …` (0.10) |
| `benchflow.providers.litellm_runtime` | `Could not publish …; it holds what the live mirror wrote: …` (0.10) |
| `benchflow.providers.litellm_runtime` | `Installing the proxy in the trial: …` |
| `benchflow.providers.litellm_runtime` | `LiteLLM gateway refused … request(s) for model(s) this run does not serve: …` |
| `benchflow.providers.litellm_runtime` | `Live token counter has stalled: the model-gateway log tail has not advanced past byte … in … polls (… bytes behind). The live token/cost figures now UNDERCOUNT this run; the final totals, imported from the full log, are unaffected.` |
| `benchflow.report` | `Could not write …: …` (0.10) |
| `benchflow.review.evidence` | `Could not remove temporary evidence archive …` |
| `benchflow.review.probe` | `Could not record the reviewer image's tools` (unreleased) |
| `benchflow.review.resume` | `Could not resume scoring for …: …; its saved result is kept` |
| `benchflow.rewards.builtins` | `Judge aggregate reward is invalid (…); returning 0.0` |
| `benchflow.rewards.builtins` | `Judge error on criterion …: …` |
| `benchflow.rewards.builtins` | `No deliverable files found in …` |
| `benchflow.rewards.builtins` | `Rubric has no criteria — returning 0.0` |
| `benchflow.rewards.file_readers` | `Error reading …: …` |
| `benchflow.rewards.file_readers` | `judge deliverables: refusing unsafe directory …` |
| `benchflow.rewards.llm` | `… judge call failed after … attempts` (0.10) |
| `benchflow.rewards.validation` | `… reward map parsed in lenient mode; dropped unrecognized/non-numeric keys: …` |
| `benchflow.rollout.__init__` | `--separate-verifier-include has no effect for …: its verifier runs in the agent's sandbox` |
| `benchflow.rollout.__init__` | `ACP client close failed: …` |
| `benchflow.rollout.__init__` | `ACP tool-evidence reconciliation failed: …` |
| `benchflow.rollout.__init__` | `Agent integration failure [agent_model_catalog]: …` (0.10) |
| `benchflow.rollout.__init__` | `Agent integration failure […]: … — trial unscored` |
| `benchflow.rollout.__init__` | `Branch snapshot has no verifier baseline; re-capturing it from the checkpoint state (tampering before the fork would not be undone at verification)` |
| `benchflow.rollout.__init__` | `Cannot record timed_out in unreadable …` |
| `benchflow.rollout.__init__` | `Cannot record timed_out in …: …` |
| `benchflow.rollout.__init__` | `Cleanup failed: …` |
| `benchflow.rollout.__init__` | `Cloud metadata: the agent runs as root on a sandbox provider; the provider's own metadata policy applies (no BenchFlow block)` |
| `benchflow.rollout.__init__` | `Could not clean up an unused warm-pool sandbox` |
| `benchflow.rollout.__init__` | `Could not keep the sandbox (deleting it): …` (0.10) |
| `benchflow.rollout.__init__` | `Could not write agent/runtime.json: …` (0.10) |
| `benchflow.rollout.__init__` | `Credential cleanup at disconnect failed: …` |
| `benchflow.rollout.__init__` | `Egress denylist proxy stop failed: …` |
| `benchflow.rollout.__init__` | `Installing … in the trial: …` |
| `benchflow.rollout.__init__` | `Kept the sandbox for inspection: … (delete it with …)` (0.10) |
| `benchflow.rollout.__init__` | `LLM trajectory write failed: …` |
| `benchflow.rollout.__init__` | `Partial session-factory trajectory capture failed: …` |
| `benchflow.rollout.__init__` | `Partial trajectory capture failed: …` |
| `benchflow.rollout.__init__` | `Teardown step … failed before the agent's sandbox stopped; cleanup runs it again` |
| `benchflow.rollout.__init__` | `The agent's sandbox did not stop before the verifier; cleanup stops it again: …` |
| `benchflow.rollout.__init__` | `Usage telemetry runtime stop failed: …` |
| `benchflow.rollout.__init__` | `Using scraped trajectory (… events) — UNTRUSTED` |
| `benchflow.rollout.__init__` | `Verifier scored 0: preparation after an earlier test command failed: …` |
| `benchflow.rollout.__init__` | `Verifier scored 0: publication after an earlier test command failed: …` |
| `benchflow.rollout.__init__` | `oracle_access=True without a User — it has no effect: only a User reads the solution.` |
| `benchflow.rollout.__init__` | `sandbox_user=None — agent runs as root with no path lockdown.` |
| `benchflow.rollout._artifacts` | `Collecting the artifacts of an unverified rollout timed out after … s` |
| `benchflow.rollout._artifacts` | `Could not hand the mounted artifacts folder to the host user: …` (0.10) |
| `benchflow.rollout._collect_hooks` | `Collect hook in service … exited …: …` (0.10) |
| `benchflow.rollout._collect_hooks` | `Collect hook in service … failed: …` (0.10) |
| `benchflow.rollout._collect_hooks` | `Sidecar artifact … from … not collected: …` (0.10) |
| `benchflow.rollout._config` | `--agent oracle was requested, but … pins scene agents (agents.roles / document scenes), which take precedence. Oracle mode will NOT run; the task's scene agent(s) will execute instead.` |
| `benchflow.rollout._config` | `loop_strategy=single-shot suppresses the user declared by …/task.md` |
| `benchflow.rollout._config` | `loop_strategy=… overrides the user declared by …/task.md` |
| `benchflow.rollout._deadline` | `BENCHFLOW_ROLLOUT_HARD_DEADLINE=… is not a number; using computed deadline` |
| `benchflow.rollout._keep_sandbox` | `Could not keep Daytona sandbox … (deleting it as usual): …` (0.10) |
| `benchflow.rollout._results` | `ADP artifact write failed: …` |
| `benchflow.rollout._results` | `ATIF artifact write failed: …` |
| `benchflow.rollout._results` | `Cannot record reported models in …: …` |
| `benchflow.rollout._results` | `Cannot record the applied effort in …: …` (0.10) |
| `benchflow.rollout._results` | `Cannot record the harness versions in …: …` (0.10) |
| `benchflow.rollout._results` | `Embodied trainer artifact write failed: …` |
| `benchflow.rollout._results` | `Trainer artifact write failed: …` |
| `benchflow.rollout._results` | `results.jsonl write failed: …` |
| `benchflow.rollout._review` | `Could not read the sandbox clock (…); files written outside the workspace will not be listed` |
| `benchflow.rollout._review` | `… file(s) written outside the workspace … during the agent phase will not reach the separate verifier (by directory: …; e.g. …). If the verifier reads them, pass --separate-verifier-include for their directory.` |
| `benchflow.rollout._separate_verifier` | `--separate-verifier-include …: … path(s) under it do not reach the verifier (…)` |
| `benchflow.rollout._separate_verifier` | `Could not list files outside the workspace (…)` |
| `benchflow.rollout._separate_verifier` | `Could not measure the workspace before the agent ran (…); a separate verifier leaves any capture failure unscored` |
| `benchflow.rollout._separate_verifier` | `Could not move the agent's verifier/ aside: …` |
| `benchflow.rollout._separate_verifier` | `Verifier sandbox cleanup failed` |
| `benchflow.rollout._separate_verifier` | `Verifier scored 0: verifier output publication failed: …` |
| `benchflow.rollout._setup` | `Could not publish script.txt as the agent; trajectory retains its output` |
| `benchflow.rollout._setup` | `Could not write the host copy of …: …` |
| `benchflow.rollout._setup` | `Ending the kept services after grading failed: …` |
| `benchflow.rollout._setup` | `Keeping the timed-out verifier's output failed: …` (0.10) |
| `benchflow.rollout._setup` | `Killing the verifier's processes failed: …` (0.10) |
| `benchflow.rollout._setup` | `Listing git repositories in … for … failed; git may refuse them as 'dubious ownership'` |
| `benchflow.rollout._setup` | `Oracle solve.sh exited with rc=…` |
| `benchflow.rollout._setup` | `Publishing for the verifier: … exited …: …` |
| `benchflow.rollout._setup` | `Publishing for the verifier: … failed: …` |
| `benchflow.rollout._setup` | `Publishing script.txt failed: …` |
| `benchflow.rollout._setup` | `Verifier scored 0: …` |
| `benchflow.rollout._setup` | `Verifier start-receipt probe failed; retrying: …` |
| `benchflow.rollout._setup` | `Verifier test command ended (exit status …) but its exec did not return within the budget; scoring its outputs` |
| `benchflow.rollout._setup` | `Verifier test command start is unconfirmed (timed out with no output); retrying it once` |
| `benchflow.rollout._setup` | `could not read the verifier's refusal stubs: …` |
| `benchflow.rollout._setup` | `could not write verifier/trusted-path.json` |
| `benchflow.rollout._setup` | `script: the root runner printed … unexpected characters around its markers (an image loader or libc setting?)` |
| `benchflow.rollout._setup` | `test.sh ran without the image's …: the root verifier clears these names, so a test that runs the image's own tools with them fails; set the value the test needs in test.sh or [verifier.env]` |
| `benchflow.rollout._setup` | `verifier PATH refusal: a refusal marker names …, which only folders the verifier's PATH leaves out have: …` |
| `benchflow.rollout._teardown_queue` | `Cleanup of … failed: …` |
| `benchflow.rollout._teardown_queue` | `Counting the stop of … failed: …` |
| `benchflow.rollout._user_loop` | `Could not keep round …'s check output: …` (0.10) |
| `benchflow.rollout._user_loop` | `Mid-loop verifier isolation skipped at round …: …. Final verify() still hardens before scoring; next-round feedback may be less isolated.` |
| `benchflow.rollout._user_loop` | `Skill export attempt … failed: …. Retrying in …s...` |
| `benchflow.rollout._verifier_recovery` | `Verifier scored 0: verifier output capture failed: …` |
| `benchflow.rollout_branch` | `Branch child … failed before doing anything (…: …); retrying it once from the checkpoint` |
| `benchflow.rollout_branch` | `Branch child … failed before doing anything (…: …); retrying it once in a new sandbox` |
| `benchflow.rollout_branch` | `` Branch snapshot … was not deleted (…: …); remove it by hand or with `bench sandbox cleanup` `` |
| `benchflow.rollout_branch` | `… during cancellation` |
| `benchflow.run_checks` | `…, at or above …: raise the output cap, or report the rate beside the score` (0.10) |
| `benchflow.runtime` | `RuntimeConfig no longer applies a … s agent timeout by default; this run uses the task's own timeout of … s. Pass RuntimeConfig(timeout=…) to keep the old limit.` |
| `benchflow.runtime` | `RuntimeConfig.… is not used and will be removed; it has no effect on the run.` |
| `benchflow.runtime` | `RuntimeResult is deprecated; Runtime.execute() and bf.run() return benchflow.RolloutResult, which has reward, passed and rollout_dir.` |
| `benchflow.runtime` | `benchflow.RuntimeResult is deprecated: bf.run() and Runtime.execute() return benchflow.RolloutResult, so isinstance(result, RuntimeResult) is False; check RolloutResult (it has reward, passed and rollout_dir).` |
| `benchflow.runtime` | `… will use a Claude login file whose access token has expired (…); if it cannot be refreshed, the run fails after the sandbox starts. Fix: …` |
| `benchflow.sandbox._base` | `Overriding CPU count to … alters the task from its intended configuration.` |
| `benchflow.sandbox._base` | `Overriding GPU count to … alters the task from its intended configuration.` |
| `benchflow.sandbox._base` | `Overriding memory to … MB alters the task from its intended configuration.` |
| `benchflow.sandbox._base` | `Overriding storage to … MB alters the task from its intended configuration.` |
| `benchflow.sandbox._base` | `Skipping env var(s) with non-identifier names (cannot be exported by the shell): …` |
| `benchflow.sandbox._delegate_disarm` | `Cloud metadata: nothing disarmed was put back before grading: processes outside no_new_privs are alive (…); the files stay without their set-ID bits or capability while the verifier runs` (0.10) |
| `benchflow.sandbox._delegate_disarm` | `Cloud metadata: … was not restored before grading (…); it stays without its set-ID bits or capability while the verifier runs` (0.10) |
| `benchflow.sandbox._docker_sweep` | `Could not mark the kept sandbox … on the Docker daemon: a BenchFlow run with another HOME may remove it as a leftover` (0.10) |
| `benchflow.sandbox._docker_sweep` | ``Docker leftover sweep: `docker …` failed: …`` |
| `benchflow.sandbox._docker_sweep` | ``… BenchFlow container(s) are still running although the BenchFlow process that started them has exited: ……. BenchFlow force-removes such a container only when its label proves which process made it; if you did not keep these on purpose, `docker rm -f …` removes them (their networks go at the next run).`` |
| `benchflow.sandbox._prepull` | `Pre-pull of … failed; its trial pulls it: …` |
| `benchflow.sandbox._recovery_baseline` | `Could not release verifier recovery image lease …: …` |
| `benchflow.sandbox._sdk_ops` | `Daytona build keys keep the SDK's folder order: … in daytona … is not the source BenchFlow's sorted context hash was written against, so a task with a COPY folder can get another build key on another machine` (0.10) |
| `benchflow.sandbox._sdk_ops` | `Invalid …=…; pinging every …s` |
| `benchflow.sandbox._sdk_ops` | `daytona get_session_command_logs malformed payload (attempt …/…) for session=… command=…: …` |
| `benchflow.sandbox._snapshot_credentials` | `Credential put-back of … not done: …` |
| `benchflow.sandbox._verifier_runtime` | `verifier runtime (…): …` |
| `benchflow.sandbox._verifier_runtime` | `verifier runtime: BenchFlow's pytest is not in the offline cache; building the plugins without the protected runner` |
| `benchflow.sandbox.agent_otel` | `The agent's OpenTelemetry is not received: …` (0.10) |
| `benchflow.sandbox.agentcore` | `AgentCore runtime … points at …, not … — updating` |
| `benchflow.sandbox.agentcore` | `Failed to stop AgentCore session: …` |
| `benchflow.sandbox.agentcore_builder` | `Could not delete uploaded build context s3://…/…: …. It will expire with the bucket lifecycle policy.` |
| `benchflow.sandbox.agentcore_builder` | `Could not stop unfinished CodeBuild …: …` |
| `benchflow.sandbox.agentcore_builder` | `S3 control-plane mutation conflicted (attempt …); retrying in …s` |
| `benchflow.sandbox.agentcore_reaper` | `BenchFlow-managed AgentCore runtime has no lease tag; treating it as active rather than deleting it` |
| `benchflow.sandbox.agentcore_reaper` | `Failed to reap AgentCore runtime …: …` |
| `benchflow.sandbox.agentcore_reaper` | `Unparseable AgentCore lease …; treating as active` |
| `benchflow.sandbox.agentcore_sealed` | `Skipping non-identifier env key … for AgentCore exec` |
| `benchflow.sandbox.apple_container` | `Apple container … removal failed: …` |
| `benchflow.sandbox.apple_container` | `Apple container … stop failed; continuing cleanup: …` |
| `benchflow.sandbox.apple_container` | `Failed to remove Apple container …: …` |
| `benchflow.sandbox.apple_container` | `Failed to stop Apple container … cleanly: …` |
| `benchflow.sandbox.apple_container` | `Forced Apple container cleanup failed for …: …` |
| `benchflow.sandbox.daytona` | `Cleaning up previous sandbox before retry` |
| `benchflow.sandbox.daytona` | `Could not delete refused Daytona startup sandbox` |
| `benchflow.sandbox.daytona` | `Could not delete sandbox … that failed to start: …` |
| `benchflow.sandbox.daytona` | `Could not look up a sandbox the failed create left: …` |
| `benchflow.sandbox.daytona` | `Could not read Daytona file info for …; treating search hit as a file: …` |
| `benchflow.sandbox.daytona` | `Could not renew the run lease of Daytona sandbox …: …; it lapses at … unless a later renewal goes through` |
| `benchflow.sandbox.daytona` | `Could not write daytona-execs.json: …` |
| `benchflow.sandbox.daytona` | `Daytona batch download failed for … (… files); retrying files individually: …` (0.10) |
| `benchflow.sandbox.daytona` | ``No Daytona sandbox handle to delete, but a create request did not finish, so a sandbox may still exist on Daytona. It is looked for by its create-attempt label until it is found and deleted, up to the job's exit; one still not found then is recorded for `bench sandbox cleanup`.`` |
| `benchflow.sandbox.daytona` | `Sandbox … started from the image rebuilt with Dockerfile salt …; later runs reuse that build` (0.10) |
| `benchflow.sandbox.daytona` | `The create with Dockerfile salt … failed the same way (…): …` (0.10) |
| `benchflow.sandbox.daytona` | `network_block_all=… overrides task config allow_internet=…` |
| `benchflow.sandbox.daytona` | `the Daytona runner holding this image never started the sandbox, and every create of the image goes to it: …. Building the same image again under a new build key (a Dockerfile comment, salts 1 to …)` (0.10) |
| `benchflow.sandbox.daytona_dind` | `Environment vars override BenchFlow compose variable(s): …` |
| `benchflow.sandbox.daytona_dind` | `Retrying docker compose up inside DinD after network create/attach race (attempt …/…, retrying in …s): …` |
| `benchflow.sandbox.daytona_dind` | `docker compose down failed: …` |
| `benchflow.sandbox.daytona_ledger` | `Could not compact …: …` |
| `benchflow.sandbox.daytona_ledger` | `Could not delete sandbox … of create attempt …: …; it stays in the ledger and is retried` |
| `benchflow.sandbox.daytona_ledger` | ``Daytona: … sandbox(es) or create attempt(s) could not be confirmed deleted before exit: …. They are recorded in …; `bench sandbox cleanup` deletes them.`` |
| `benchflow.sandbox.daytona_ledger` | `… (create attempt …)` (0.10) |
| `benchflow.sandbox.daytona_reaper` | `Daytona sandbox … has unparseable created_at …; skipping (possible leak)` |
| `benchflow.sandbox.daytona_reaper` | `Daytona snapshot … has no parseable created_at; skipping` |
| `benchflow.sandbox.daytona_reaper` | `Failed to delete sandbox …` |
| `benchflow.sandbox.daytona_reaper` | `Failed to delete snapshot …` |
| `benchflow.sandbox.daytona_reaper` | `Not deleting Daytona sandbox …: it is still building (…), and a delete now would make Daytona keep its build as failed ("context canceled") for every later create of the image` (0.10) |
| `benchflow.sandbox.daytona_reaper` | ``… Daytona sandbox(es) carry a benchflow label namespace but not the …=… ownership label, so the age-based reaper will never reclaim them (possible orphan leak): …. Not deleting. They may belong to other users of this API key: delete one only if you started it, with `bench sandbox cleanup --id <sandbox id>`.`` (0.10) |
| `benchflow.sandbox.daytona_strategies` | `Could not start from content snapshot …; building the image instead (…)` |
| `benchflow.sandbox.daytona_strategies` | `Daytona could not build the cached install layers on this image; installing in the sandbox instead (…)` |
| `benchflow.sandbox.daytona_strategies` | ``Daytona snapshot … could not be deleted (…); `bench sandbox cleanup` removes it once it is stale`` |
| `benchflow.sandbox.daytona_strategies` | `No content snapshot; building the image (…)` |
| `benchflow.sandbox.daytona_strategies` | `Snapshot template specified but force_build is True. Snapshot will be used instead of building from scratch.` |
| `benchflow.sandbox.docker` | `BenchFlow sets these Compose variables itself; the task's value is not used for Compose: …` |
| `benchflow.sandbox.docker` | ``Both `keep_containers` and `--delete` option are set. keep_containers takes precedence.`` |
| `benchflow.sandbox.docker` | `Chown logs directory timed out; continuing teardown.` |
| `benchflow.sandbox.docker` | `Docker compose down hung/failed (…); force-killing project.` |
| `benchflow.sandbox.docker` | ``Docker verifier-log bind mount is not visible inside the container; verifier outputs will be copied back explicitly. The daemon cannot see …, so it may be remote or reached without path translation (check DOCKER_HOST and `docker context ls`).`` |
| `benchflow.sandbox.docker` | `Failed to chown logs directory: …` |
| `benchflow.sandbox.docker` | `Force-kill of compose project … failed: …` |
| `benchflow.sandbox.docker` | `Force-kill of compose project … stopped: docker network prune gave no answer within … s (Docker is busy); the job's leftover sweep removes what is left` (0.10) |
| `benchflow.sandbox.docker` | `Inspecting the images … built failed: …` |
| `benchflow.sandbox.docker` | `Listing the images … built failed: …` |
| `benchflow.sandbox.docker` | ``Listing the images … built gave no answer within 30 s (Docker is busy), so they stay: `docker image ls --filter label=com.docker.compose.project=…` lists them`` (0.10) |
| `benchflow.sandbox.docker` | `No cached … layer for …; it installs in the trial: …` |
| `benchflow.sandbox.docker` | `Removing the images … built failed: …` |
| `benchflow.sandbox.docker` | `Removing the trial's layer tag … failed: …` |
| `benchflow.sandbox.docker` | `Removing the volumes … made failed: …` |
| `benchflow.sandbox.docker` | `Retrying Docker build for … after transient failure (attempt …/…, retrying in …s): …` |
| `benchflow.sandbox.docker` | `Retrying docker compose up for … after network create/attach race (attempt …/…, retrying in …s): …` |
| `benchflow.sandbox.docker` | ``Snapshot image … is still in use (kept container?); remove it with `docker image rm …` when done`` |
| `benchflow.sandbox.docker` | `Snapshot image … removal failed: …` |
| `benchflow.sandbox.egress_denylist` | `Egress log … not collected: …` (0.10) |
| `benchflow.sandbox.env_layers` | `Not reusing …: no BenchFlow build on this machine made it; building the … layer again` |
| `benchflow.sandbox.env_layers` | `Pruning build records of layer … failed: …` |
| `benchflow.sandbox.env_layers` | `Pruning old … layers failed: …` |
| `benchflow.sandbox.env_layers` | `Removing layer … failed: …` |
| `benchflow.sandbox.env_layers` | `Removing leftover layer tag … failed: …` |
| `benchflow.sandbox.env_layers` | `Removing unreturned layer alias … failed: …` |
| `benchflow.sandbox.env_layers` | `Removing unreturned layer alias …: …` |
| `benchflow.sandbox.env_layers` | `libc probe volumes left in use: …` |
| `benchflow.sandbox.env_layers` | `…=… is not a number of GB; using %g` |
| `benchflow.sandbox.env_layers` | `…=… is not a whole number; keeping …` |
| `benchflow.sandbox.lockdown` | `BenchFlow's root steps leave … off their PATH from now on` |
| `benchflow.sandbox.lockdown` | `Could not check the folders on PATH.…` |
| `benchflow.sandbox.lockdown` | `Egress firewall exec timed out after …s; installing iptables in its own exec, then retrying (… of …)` (0.10) |
| `benchflow.sandbox.lockdown` | `Folders on PATH the agent user may write, which could not be locked to root: …` |
| `benchflow.sandbox.lockdown` | `Ignoring a plugin guard marker with a bad signature: …` |
| `benchflow.sandbox.lockdown` | `Kept service pid … did not live to be graded` |
| `benchflow.sandbox.lockdown` | `Kept service pid … killed before grading: …` |
| `benchflow.sandbox.lockdown` | `Kept service pid … was not frozen before grading (state …)` |
| `benchflow.sandbox.lockdown` | `Kept service pid … writes to the terminal …, which ended with the agent's terminal tool: its writes there fail, so it may not answer; start it with its output in a file` |
| `benchflow.sandbox.lockdown` | ``Killed instead of kept: port … pid … `…` ran without no_new_privs (not the agent's launch), so it could use a set-ID file or a file capability put back before grading`` (0.10) |
| `benchflow.sandbox.lockdown` | `Protected runtime in … holds no Python: …` |
| `benchflow.sandbox.lockdown` | `Root's preparation steps do not run …: …` |
| `benchflow.sandbox.lockdown` | `Sandbox user … setup step failed.…` |
| `benchflow.sandbox.lockdown` | `Soft verification runs with BASH_ENV empty: …` |
| `benchflow.sandbox.lockdown` | `Task … sets PIP_CONFIG_FILE in its verifier env; the verifier's pip reads no configuration file. …` |
| `benchflow.sandbox.lockdown` | `Task …: the image has pip configuration (…); the verifier's pip reads none. …` |
| `benchflow.sandbox.lockdown` | `The compose service … has no protected runtime, so BenchFlow will run no root step there after the agent's turn.…` |
| `benchflow.sandbox.lockdown` | `The kept protected runtime kit could not be placed, so it is built again.…` |
| `benchflow.sandbox.lockdown` | `The separate verifier sandbox runs with BASH_ENV empty: transferred activation has no pre-agent provenance` |
| `benchflow.sandbox.lockdown` | `The verifier runs with BASH_ENV empty: the image's BASH_ENV in service … could not be read (status …).…` |
| `benchflow.sandbox.lockdown` | `The verifier runs with BASH_ENV empty: the image's BASH_ENV in service … could not be read: …` |
| `benchflow.sandbox.lockdown` | `The verifier runs with BASH_ENV empty: … … cannot be resolved: …` |
| `benchflow.sandbox.lockdown` | `The verifier runs with BASH_ENV empty: … … could not be checked.…` |
| `benchflow.sandbox.lockdown` | `The verifier runs with BASH_ENV empty: … … could not be checked: …: …` |
| `benchflow.sandbox.lockdown` | `The verifier runs with BASH_ENV empty: … … has no protected pre-agent inspector` |
| `benchflow.sandbox.lockdown` | `The verifier runs with BASH_ENV empty: … … in service … cannot be checked without shared-mount provenance` |
| `benchflow.sandbox.lockdown` | `The verifier runs with BASH_ENV empty: … … is not a root-owned file outside the agent's reach` |
| `benchflow.sandbox.lockdown` | `The verifier's PATH check could not run in the sandbox, so its PATH keeps only the system folders, without the image's ….…` |
| `benchflow.sandbox.lockdown` | `The verifier's PATH leaves out … folder(s) of the image's PATH: …` |
| `benchflow.sandbox.lockdown` | `benchflow: no protected Python in this sandbox, so the Python start-up hook clean-up was skipped` |
| `benchflow.sandbox.lockdown` | `distro detection failed (…); skipping pip env tweaks` |
| `benchflow.sandbox.lockdown` | `hook clean-up: … entries vanished while they were walked and were skipped, e.g. …` |
| `benchflow.sandbox.lockdown` | `task [verifier.hardening] unknown/invalid: …=…` |
| `benchflow.sandbox.lockdown` | `task.md parse error in …: …` |
| `benchflow.sandbox.lockdown` | `task.toml parse error in …: …` |
| `benchflow.sandbox.metadata` | `Failed to persist sandbox.json for …: …` |
| `benchflow.sandbox.modal_impl` | `Could not end the refused Modal sandbox` |
| `benchflow.sandbox.modal_impl` | `Error terminating Modal sandbox: …` |
| `benchflow.sandbox.modal_impl` | `bfx: the mounted volume is not usable (…)` |
| `benchflow.sandbox.process._base` | `Invalid …=…; using default …s` |
| `benchflow.sandbox.process._base` | `Skipped an output line over … bytes` |
| `benchflow.sandbox.process.agentcore` | `AgentCore shell reader stopped: …` |
| `benchflow.sandbox.process.apple` | `Could not remove staged Apple container env after launch failure` |
| `benchflow.sandbox.process.modal` | `Modal stdout stream stopped: …` |
| `benchflow.sandbox.process.modal` | `Skipped an output line over … bytes` |
| `benchflow.sandbox.remote_docker` | `Compose project … left on … after teardown: …` |
| `benchflow.sandbox.remote_docker` | `Could not confirm cleanup of compose project … on … (host unreachable). Remove what carries its label when the host is back: docker ps -a, docker network ls and docker volume ls with --filter label=com.docker.compose.project=…` |
| `benchflow.sandbox.remote_docker` | `Removing … of compose project … failed: …` |
| `benchflow.sandbox.setup` | `Clamping cpus … -> … for Daytona (override with BENCHFLOW_DAYTONA_MAX_CPUS)` |
| `benchflow.sandbox.setup` | `Clamping memory_mb … -> … for Daytona (override with BENCHFLOW_DAYTONA_MAX_MEMORY_MB)` |
| `benchflow.sandbox.setup` | `Clamping storage_mb … -> … for Daytona (override with BENCHFLOW_DAYTONA_MAX_STORAGE_MB)` |
| `benchflow.sandbox.setup` | `stage_dockerfile_deps: COPY has fewer than 2 arguments, leaving unchanged: …` |
| `benchflow.sandbox.setup` | `stage_dockerfile_deps: could not parse COPY arguments, leaving unchanged: …` |
| `benchflow.sandbox.setup` | `stage_dockerfile_deps: could not parse JSON-form COPY, leaving unchanged: …` |
| `benchflow.sandbox.setup` | `stage_dockerfile_deps: malformed JSON-form COPY, leaving unchanged: …` |
| `benchflow.sandbox.setup` | `stage_dockerfile_deps: refusing non-regular source …` |
| `benchflow.sandbox.setup` | `stage_dockerfile_deps: refusing to stage COPY source … (path could not be resolved)` |
| `benchflow.sandbox.setup` | `stage_dockerfile_deps: refusing to stage COPY source … — resolves outside context_root …` |
| `benchflow.sandbox.setup` | `stage_dockerfile_deps: refusing to stage symlinked source …` |
| `benchflow.sdk` | `The 'trial_name' argument is deprecated. Use 'rollout_name' instead.` |
| `benchflow.sdk` | `benchflow.SDK is deprecated; use bf.run(bf.RolloutConfig(task_path=..., agent=..., model=...)), which takes the same keyword arguments as SDK.run().` |
| `benchflow.task.config` | `The 'memory' field is deprecated. Use 'memory_mb' instead.` |
| `benchflow.task.config` | `The 'storage' field is deprecated. Use 'storage_mb' instead.` |
| `benchflow.task.formats` | `task format … does not implement TaskFormat; ignored` |
| `benchflow.task.formats` | `task format … failed to inspect …: …` |
| `benchflow.task.formats` | `task format … failed to load: …` |
| `benchflow.task.imports` | `…: ignored … task.toml key(s) BenchFlow does not know: …` |
| `benchflow.task.verifier_core` | `Cannot create … for the verifier start receipt; startup of this verifier cannot be confirmed` |
| `benchflow.task.verifier_core` | `Could not hand the mounted verifier outputs to the host user: …` |
| `benchflow.task.verifier_core` | `Criteria runner exited with rc=… after writing its reward; the reward is scored (reward files were cleared before this run)` |
| `benchflow.task.verifier_core` | `Criteria-runner verifier directory download failed; recovered canonical verifier output files individually: …` |
| `benchflow.task.verifier_core` | `Plugin guard …'s markers were not all read (…); deciding on the verdicts that were` |
| `benchflow.task.verifier_core` | `Reading the pytest plugin guard …'s markers failed (…); retrying in …s` (0.10) |
| `benchflow.task.verifier_core` | `Verifier directory download failed; recovered canonical verifier output files individually: …` |
| `benchflow.task.verifier_core` | `Verifier exited with rc=… after writing its reward; the reward is scored (reward files were cleared before this run)` |
| `benchflow.taskmd.verify` | `Could not restore the paths hidden from the judge: …` |
| `benchflow.taskmd.verify` | `The judges' copy of the saved outputs was not published: …` |
| `benchflow.traces.huggingface` | `Unknown format …, skipping row …` |
| `benchflow.traces.task_gen` | `Dropping trace path … (relativized to …) — escapes task workspace` |
| `benchflow.trajectories._capture` | `Could not scrape gemini trajectory: …` |
| `benchflow.trajectories._capture` | `Failed to parse gemini trajectory: …` |
| `benchflow.trajectories._export_common` | `Skipping unreadable trainer artifact …: …` |
| `benchflow.trajectories._llm_capture` | `… holds … of the … lines earlier roles published; this role's lines follow them` (0.10) |
| `benchflow.trajectories.otel_spans` | `Trial spans not sent to …: …` (0.10) |
| `benchflow.trajectories.results` | `Skipping invalid results artifact row …:…: …` |
| `benchflow.trajectories.results` | `Skipping non-object results artifact row …:…` |
| `benchflow.trajectories.results` | `Skipping unreadable results artifact …: …` |
