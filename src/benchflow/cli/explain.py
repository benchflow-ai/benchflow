"""``bench explain [CODE]``: one page per reason a trial has no score.

Every cause the end-of-run summary names (:func:`benchflow.failures.cause_of`)
has a page here, under its code: the code Rust's ``bench`` 0.8.2 uses
where the meaning is the same (``R212``: the model provider rejected the
login), else the cause's key (``verifier_dep_install``). A page says what
happened, whose problem it is, and the next step. The title and the next step
are the summary's own (``failures.py``); the page adds the explanation.

``bench explain`` alone lists every code. ``--json`` prints the page (or the
list) as JSON on stdout; a code with no page is a usage error (exit 2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated

import typer

from benchflow import failures
from benchflow.cli._lazy import UsageError, get_current_context

# Whose problem, in the words Rust's ``bench`` and the JSON errors use
# (``cli/_errors.py``'s ``owner``), with the label a page prints.
OWNER_LABELS: dict[str, str] = {
    "task": "the task",
    "agent": "the agent",
    "model": "the model provider",
    "sandbox": "the sandbox",
    "you": "your setup",
    "benchflow": "BenchFlow (a bug)",
}

# A cause whose fault is "infrastructure" but whose infrastructure is the
# model provider's, not the sandbox's.
_MODEL_PROVIDER = frozenset(
    {"provider_rate_limit", "api_error", "invalid_provider_response"}
)

# What each cause means, beyond its one-line title: what happened and where
# to look. The next step is the summary's (failures.py), printed after it.
_BODIES: dict[str, str] = {
    "kept_service_freezer": (
        "The task declares services that must survive the agent turn and be "
        "frozen until verification, and this kind of sandbox cannot freeze "
        "them for a non-root agent (native Daytona without Compose, for one). "
        "The trial has no score; where the task alone decides it, the "
        "refusal comes before any sandbox is made."
    ),
    "kept_service_freezer:engine": (
        "The task's kept services are frozen in a cgroup between the agent's "
        "turn and verification, and this Docker Engine cannot make one for a "
        "non-root agent: it is older than 28, rootless or remaps user "
        "namespaces. No agent code ran; the trial has no score."
    ),
    "kept_service_freezer:host_uid": (
        "The task keeps services, the sandbox mounts the trial's folders from "
        "the host (local Docker), and the image's agent user has the uid "
        "BenchFlow runs as. While the kept services run, grading closes the "
        "verifier's folder to that uid, which closes it to BenchFlow too, so "
        "a reward BenchFlow writes there would be lost. No agent code ran; "
        "the trial has no score."
    ),
    "kept_service_freezer:probe": (
        "The task's kept services are frozen in a cgroup between the agent's "
        "turn and verification, and the freezer check inside this sandbox "
        "failed (no cgroup v2, or cgroups it cannot write). No agent code "
        "ran; the trial has no score. The check's own reason is in the error."
    ),
    "runtime_sealing": (
        "BenchFlow refused to run root's steps using runtime files whose "
        "ownership or paths could not be protected from the agent. The refusal "
        "preserves isolation; changing the task image is required."
    ),
    "guard_timeout": (
        "The setup guard did not finish or return its report. The trial has "
        "no score. Check the setup evidence before retrying; this is not an "
        "agent task failure."
    ),
    "phase_timeout": (
        "A step of the trial that is not the agent's turn ran out of time: "
        "setting the sandbox up, starting it, installing the agent, "
        "BenchFlow's own set-up for the agent's turn (its network policy, "
        "firewall or model proxy), a multi-step task's step set-up or soft "
        "verify, a script's tear-down after it ran, or the verify step after "
        "the turn. The agent did not run out of its time limit. The trial has "
        "no score and is retried like any infrastructure failure. A step that "
        "fails after the agent's turn had ended the trial is not this: that "
        "ending stays, and post_turn_failure_info in result.json names the "
        "step."
    ),
    "invalid_provider_response": (
        "The proxy observed an upstream HTTP 5xx with an invalid JSON body. "
        "Lifecycle records preserve the actual status, byte counts and elapsed "
        "upstream seconds without copying prompt content."
    ),
    # Unscored: the agent finished, the verifier could not score it.
    "verifier_failure": (
        "The agent finished, but the verifier step failed before a reward "
        "existed, and the trial's error starts 'verifier crashed:'. "
        "BenchFlow names a finer cause when it can (its own infrastructure, "
        "a dependency install, a full disk); this one is left when it "
        "cannot, and usually comes from the task's verifier setup. This "
        "unscored final trial counts as a failure in primary metrics (mean "
        "reward and pass@k), remaining in their denominator; only scored-only "
        "metrics exclude it. A test.sh that itself exits with an error before "
        "writing a reward is not this cause either: that is "
        "verifier_no_usable_reward."
    ),
    "verifier_timeout": (
        "The verifier ran longer than the task's verifier timeout and was "
        "stopped before it wrote a reward. Installing test dependencies while "
        "grading is the usual cause: install them in the image instead, and "
        "raise the timeout only if the tests themselves are slow."
    ),
    "verifier_dep_install": (
        "The verifier installed its dependencies while grading, and the "
        "install failed: a package index it could not reach, or no version "
        "that matched. Grading that depends on the network at grading time "
        "fails whenever the network does."
    ),
    "reward_out_of_range": (
        "The verifier ran and wrote a number outside the range the task "
        "declares (0 to 1 unless task.toml's [verifier] reward_range says "
        "otherwise). Under integrity audit and off, a reward field outside it "
        "is recorded as test.sh wrote it, with a note in "
        "reward_out_of_range_info and in summary.json's rewards_outside_range, "
        "and gets no category; this one is set when the number is one the "
        "reward would be made from (no reward field), and that trial is "
        "unscored. Under integrity strict the trial scores 0, a failure in the "
        "primary metrics (mean reward and pass@k) and in the scored-only ones "
        "alike, and stays in every denominator, with a 'Verifier scored 0:' "
        "warning. reward_out_of_range_info in result.json keeps the raw "
        "value. Every run of the task ends the same way until the verifier or "
        "the declared range is fixed."
    ),
    "registry_auth": (
        "Docker could not pull the task's base image because the registry "
        "refused this machine's login: an expired `docker login`, a login "
        "without access to that repository, or the registry's token service "
        "saying no. It is your setup, not the task's Dockerfile, and it is not "
        "retried. Docker Hub gives the same answer for a repository that does "
        "not exist, so check the FROM line's name too."
    ),
    "task_config": (
        "A file of the task does not load: a typo or an unknown key in "
        "task.toml, a task.md whose front matter does not parse, or a "
        "verifier document or rubric that is not valid. The message names the "
        "file and, for a typo, the key it meant. Every run of the task fails "
        "the same way until the file is fixed, so it is not retried; "
        "`bench check <task>` reports the same problem without running it."
    ),
    "disk_full": (
        "The disk that holds the job folder ran out of space, so BenchFlow "
        "could not write the trial's files (result.json, summary.json), or "
        "the verifier's reward, which the sandbox writes into that folder, "
        "came out empty. It says nothing about the task or the agent, and is "
        "not retried within the run; once space is freed, a resume runs the "
        "trial again."
    ),
    "verifier_infra": (
        "The sandbox the verifier ran in failed, or a separate verifier "
        "sandbox never received the agent's outputs, so nothing was "
        "assessed. It says nothing about the task or the agent."
    ),
    "verifier_other": (
        "The verifier failed in a way BenchFlow has no finer cause for. "
        "The trial's verifier/ folder holds what it printed."
    ),
    "verifier_plugin_trust": (
        "The task's test.sh installs a pytest plugin (usually "
        "pytest-json-ctrf) into a Python environment the agent could write: "
        "the workspace, the agent's home, /tmp, /var/tmp, /logs or /testbed. "
        "The pytest plugin guard refuses to load it, since the agent could "
        "have planted it, so every run of the task scores 0, with a warning "
        "that starts 'Verifier scored 0: pytest plugin guard not confirmed'. "
        "Bake the "
        "plugin into the image, or pin its release in [verifier] "
        "pytest_plugin_releases and install it as root into a Python "
        "environment's site-packages, where the guard trusts it by the bytes "
        "PyPI publishes."
    ),
    "verifier_plugin_unrecorded": (
        "The pytest plugin guard refused a plugin in a place the agent could "
        "write, and the record of the agent's files was missing or "
        "incomplete, so it cannot tell whether the agent put the plugin "
        "there. The trial is a failure in the scores. When test.sh installs "
        "the plugin, pin its release in [verifier] pytest_plugin_releases or "
        "bake it into the image."
    ),
    "verifier_plugin_unproven": (
        "The task's test.sh installed a pytest plugin after the agent stopped "
        "(with uv, uvx or pip, into the workspace or the verifier's uv "
        "state), and the agent had changed files in its trees, any of which "
        "could have chosen what was installed. The pytest plugin guard trusts "
        "such an install only as a release it can check byte for byte, so it "
        "refused the plugin. The trial is a failure in the scores, so a "
        "planted input gains nothing, but it is the task's problem: pin the "
        "plugin's release in [verifier] pytest_plugin_releases, or bake it "
        "into the image."
    ),
    "verifier_plugin_unlisted": (
        "The task's test.sh installed a release of pytest-json-ctrf, the "
        "plugin BenchFlow vouches for, that was missing from the table of "
        "published releases hardening built: this machine could not read "
        "the release list from PyPI, or PyPI published the release after "
        "hardening read it. The pytest plugin guard refused it, and the trial "
        "is a failure in the scores, since the agent can also leave inputs "
        "that install a release of any name. Check that this machine reaches "
        "pypi.org and run the task again."
    ),
    "verifier_plugin_refused": (
        "The pytest plugin guard refused a plugin in a place the agent could "
        "write that the agent left there, or changed, after it started. The "
        "trial counts as a failure. The trial's verifier/ folder names the "
        "plugin and why."
    ),
    "verifier_output_unwritable": (
        "test.sh never ran: its verifier runs as a user other than root that "
        "cannot write the file its output goes to. On Docker BenchFlow makes "
        "that file on the host, as the host's user, before the test command, "
        "and the verifier's user is the image's own USER when the task names "
        "none. That is BenchFlow's failure, not the task's or the agent's: "
        "the trial is unscored and stays out of the error-excluding score. "
        "Until BenchFlow leaves the file to the test command, run "
        'the verifier as root ([verifier] user = "root" in task.toml).'
    ),
    "verifier_no_usable_reward": (
        "The final verify left no reward BenchFlow could use: no reward file, "
        "an empty one, one it could not read, or one without a number. That "
        "is the verifier's error, not a wrong answer, and it is never retried. "
        "The trial still counts as a failure in every score, as a plugin guard "
        "refusal does, because the reward files are evidence the agent can "
        "reach. The trial's verifier/ folder has test.sh's output."
    ),
    "no_verdict": (
        "The trial has neither a reward nor an error: a reviewer's "
        "assessment is still pending, or the run was interrupted before the "
        "trial recorded an outcome."
    ),
    # Errored: the run stopped before a score.
    "provider_auth": (
        "The model provider answered with an authentication error (HTTP 401, "
        "or a key it called invalid): the key or login is wrong, revoked, "
        "expired, or for another provider. The trial is not retried."
    ),
    "provider_rate_limit": (
        "The model provider refused requests for going over its rate limit "
        "(HTTP 429). The trial is not retried within the run."
    ),
    "provider_rejected": (
        "The model provider rejected a request as invalid (HTTP 400), most "
        "often a conversation longer than the model's context window. The "
        "trial is not retried: the same request would be rejected again."
    ),
    "api_error": (
        "The model provider's API failed: a server error, or a service "
        "overloaded or unavailable. api_error_info in result.json has the "
        "status and whether it was transient; transient ones are retried "
        "within the run."
    ),
    "suspected_api_error": (
        "The agent finished without using a token or calling a tool: it most "
        "likely never reached its model (a wrong model id, an expired login, "
        "a provider that refused it without saying so)."
    ),
    "install_failure": (
        "Installing the agent in the sandbox failed, usually on the network "
        "to npm, PyPI or the agent's download site, or on a base image that "
        "lacks a tool the install needs. It is retried within the run."
    ),
    "pipe_closed": (
        "The connection between BenchFlow and the agent process closed: the "
        "Daytona terminal dropped, its websocket closed, or the local or SSH "
        "transport ended. It is retried within the run."
    ),
    "idle_timeout": (
        "The agent sent no tool call, message or thought for longer than the "
        "idle timeout, and the idle watchdog stopped it. A hung tool call or "
        "a stalled model connection are the usual causes. When the verifier "
        "can still score the workspace, the trial is scored instead."
    ),
    "timeout": (
        "The agent ran past its time budget (the task's agent timeout times "
        "--timeout-multiplier, or the host's hard deadline) and was stopped. "
        "When the verifier can still score the workspace, the trial is "
        "scored instead; this cause is a timeout it could not score."
    ),
    "trial_budget": (
        "The trial spent its token or turn budget (a turn is one model call): "
        "BenchFlow's model proxy refused the agent's next call and BenchFlow "
        "stopped the agent. Like a timeout, the verifier then grades what the "
        "agent left, so the trial is usually scored, 0 unless its tests "
        "passed, and it is never retried; this cause is a budget stop it "
        "could not score. result.json's trial_budget says which measure "
        "stopped it, what the trial used and its subagents' share."
    ),
    "infra_failure": (
        "The sandbox, or BenchFlow's connection to it, failed during the "
        "trial: a lost container, or a trial past the host's hard deadline "
        "whose transport or teardown was stuck. It is retried within the run."
    ),
    "sandbox_setup": (
        "The sandbox did not start: the network or the registry failed while "
        "the image was pulled or built, the backend refused it (quota, "
        "capacity, a daemon that is down), or the task's Dockerfile names a "
        "path that is missing. The run summary prints the cause; error in "
        "result.json has the whole message (sandbox_startup_info adds the "
        "sandbox's id and state when the backend gave them)."
    ),
    "image_build": (
        "The task's image did not build, for a reason another attempt would "
        "repeat: a RUN step exited nonzero with no network failure in its "
        "output, the base image in FROM is not "
        "found or not public, or the Dockerfile does not parse. The summary "
        "names the Dockerfile line and BuildKit's reason; the trial is not "
        "retried. A network or registry failure during the build, a RUN "
        "step's included, is sandbox_setup instead and is retried; a registry "
        "that refused the login is registry_auth."
    ),
    "acp_error": (
        "The agent answered a request with an error of its own over the Agent "
        "Client Protocol; the message is the agent's. For Claude Code, 'The "
        "connection to Claude was lost' means its own process or its API "
        "stream ended, not BenchFlow's connection to the sandbox."
    ),
    "script_backend": (
        "An operator script (--agent script:...) needs the trusted root runner "
        "that only the docker and remote-docker backends have, so the trial is "
        "refused before any sandbox or provider call. It is not retried: "
        "another attempt would be refused the same way."
    ),
    "unsupported_task_feature": (
        "The task asks for something the chosen sandbox backend cannot "
        "provide, so the trial is refused before it starts. The message names "
        "each feature and the backend."
    ),
    "task_refused": (
        "The task is on BenchFlow's in-package refusal list: its reference "
        "solution fails under one of BenchFlow's always-on protections, so "
        "any score would be BenchFlow's doing. It is refused before the "
        "turn for every agent, unscored and never retried, and stays in the "
        "task count. The list is keyed by benchmark, task id and the task's "
        "digest: a copy with any byte changed runs."
    ),
    "agent_integration:agent_auth": (
        "The agent started, then did nothing, because its own login or "
        "billing failed (an expired subscription, no credit left)."
    ),
    "agent_integration:agent_model": (
        "The agent does not offer the model the run asked for. The error "
        "lists the models it does offer."
    ),
    "agent_integration:agent_model_catalog": (
        "The agent's engine takes each model's tools from its model catalog, "
        "and the catalog has no entry for this model, so the agent ran on "
        "fallback tools (for Codex: no apply_patch, no code mode). That is "
        "another agent than the model is used with, so the trial is not "
        "scored. Codex's warning ('Model metadata for ... not found') or the "
        "tools its first request offered show it; result.json's tool_surface "
        "lists them."
    ),
    "agent_integration:agent_effort": (
        "The agent offers the model, but not at the reasoning effort the run "
        "asked for (codex-acp lists one model id per effort a model takes). "
        "The run is refused before the agent starts rather than run at "
        "another effort; the error lists the efforts it does offer."
    ),
    "agent_integration:agent_budget": (
        "The trial has a token or turn budget (--token-budget, --turn-budget "
        "or the task's [agent] token_budget and turn_budget), but the agent "
        "calls its model around BenchFlow's model proxy: a subscription "
        "login, an endpoint of its own, or an agent that routes its own "
        "model. The proxy is what counts and enforces a trial's budget, so "
        "the trial is refused before the agent starts rather than run "
        "without one; the error says which route it took."
    ),
    "agent_integration:agent_install": (
        "The agent started, but its install or runtime in the sandbox is "
        "broken: a missing program, or a dependency that does not load."
    ),
    "agent_integration:unknown": (
        "The agent did nothing useful, and BenchFlow has no finer cause for "
        "it; integration_failure_info in result.json has what it saw."
    ),
    "usage_limit": (
        "The login's subscription usage limit was reached (its 5-hour or "
        "7-day window). The job starts no more trials on it; the summary "
        "lists the trials it did not start, the login, the window and the "
        "reset."
    ),
    "pipe_closed:exit": (
        "The agent process exited during its turn; the message has its exit "
        "code, and says whether the sandbox's cgroup counted an OOM kill (an "
        "exit is called a memory kill only with that count, never from its "
        "code alone). Its log has its last words."
    ),
    "provider_unavailable": (
        "The agent stopped with an error after the model provider answered a "
        "call with a server error (HTTP 500, 502, 503, 504 or 529): the "
        "provider's failure, not the agent's or the model's. It is retried "
        "within the run; agent_outcome in result.json counts the failed "
        "calls."
    ),
    "verifier_network_refused": (
        "test.sh's network (verifier.network_mode, else the sandbox's policy) "
        "refused it: a packet at BenchFlow's rules, or a request at its egress "
        "proxy. Its tests did not pass (a non-zero exit, or a reward below 1), "
        "so the trial is unscored rather than a silent 0 that would read as "
        "the agent's failure. It counts 0 in the every-trial mean. "
        "verifier/verifier-network.json names the network, the counts and the "
        "first requests refused. A pass despite refusals keeps its reward."
    ),
    "network_policy_breached": (
        "BenchFlow proves the trial's network rules again at the end of the "
        "agent's turn and at the end of test.sh, from its helper outside the "
        "sandbox: iptables -C on every rule it put in, each rule's counter, "
        "and a connect from inside test.sh's bounded launch to the cloud "
        "metadata server and to TEST-NET. A rule was gone, a counter went "
        "back (zeroed, or the rule was put in again), the check could not be "
        "made or a connect got through, so the trial ran, or would have run, "
        "its tests without the task's network policy. It is unscored, counts "
        "0 in the every-trial mean, is never retried and flags the job. "
        "verifier/network-policy.json names the rules and when they were last "
        "proven present."
    ),
    "verifier_network_unheld": (
        "The task gives the verifier a restricted network (verifier.network_mode, "
        "draft 2's verifier network, else the sandbox's policy), and BenchFlow "
        "could not hold test.sh to it: the rules could not be added, from its "
        "helper or at all, or were not proven. test.sh runs the agent's code, "
        "so it does not run with the network open instead: the trial is "
        "unscored and never retried. The error names what failed."
    ),
    "test_sh_preflight_unavailable": (
        "A branch child restores its source trial's sandbox after that trial's "
        "agent ran, so BenchFlow never judges test.sh's commands there: it "
        "carries the verdict the source decided before its turn. A checkpoint "
        "made before the check recorded none, and BenchFlow could not decide "
        "one on a fresh sandbox of the task's image either (a provider that "
        "starts the child from the snapshot itself, or a check that could not "
        "run), so the child was not run: unscored, never retried, and never "
        "the task's fault. A new checkpoint carries a verdict."
    ),
    "test_sh_command_missing": (
        "BenchFlow runs test.sh in the task's image as a plain, non-login "
        "shell, with the verifier's hardened PATH, as the verifier user. Just "
        "before the agent's turn it reads test.sh's interpreter and the "
        "commands it runs unconditionally with errexit on, before anything "
        "that could install a tool or change PATH, and resolves each there. "
        "One that does not resolve means test.sh can never write a reward, so "
        "the trial is refused before the agent runs: unscored, never retried, "
        "no agent tokens spent. A tool the image puts on PATH only through "
        "a login profile (~/.profile, ~/.bashrc) is not on that PATH. Other "
        "commands test.sh names that do not resolve are warnings in "
        "result.json and change no score."
    ),
    "verifier_tools_refused": (
        "The verifier runs test.sh as root with a PATH of the image's folders "
        "only root can write. Before the agent's turn BenchFlow locks the "
        "image's PATH folders to root (the official rust image leaves "
        "/usr/local/cargo/bin open to every user); a folder still open at "
        "grading time, a relative entry, or one in the workspace, a home or a "
        "temp folder is left out. A task whose test.sh needs programs there "
        "declares them in [verifier] required_tools, and none did, or a "
        "declared one is only in a folder left out. test.sh did not run, so "
        "the trial is unscored, never a 0, and never retried. The error names "
        "each folder and why it was left out."
    ),
    "verifier_launch_refused": (
        "test.sh runs the agent's code as root, so BenchFlow starts it without "
        "NET_ADMIN, NET_RAW and SYS_PTRACE: through its own pinned bfx where "
        "bfx was delivered, else util-linux setpriv, else its kit Python. None "
        "of them could (or the shell it started still held one of the three), "
        "so test.sh did not run and the trial has no score. "
        "verifier/test-stdout.txt says what the launch found."
    ),
    "other": (
        "An error none of the other pages covers. If it repeats on other "
        "tasks, it may be a bug in BenchFlow: report it at "
        "https://github.com/benchflow-ai/benchflow/issues with the trial "
        "folder."
    ),
}


@dataclass(frozen=True)
class Page:
    """One ``bench explain`` page."""

    code: str
    key: str  # the cause's key, the error_category result.json records
    title: str
    owner: str
    body: str
    next: str

    def as_json(self) -> dict[str, str]:
        return {
            "code": self.code,
            "title": self.title,
            "owner": self.owner,
            "body": self.body,
            "next": self.next,
        }


def owner_of(fault: str, key: str = "") -> str:
    """A failures.py fault in the owner words of Rust's ``bench`` and the JSON errors."""
    if key in _MODEL_PROVIDER:
        return "model"
    return {"setup": "you", "infrastructure": "sandbox"}.get(fault, fault)


def pages() -> tuple[Page, ...]:
    """Every page, in the summary's order: unscored causes, then errored ones."""
    found: list[Page] = []
    for key in failures.cause_keys():
        cause = failures.describe(key)
        assert cause is not None, key
        found.append(
            Page(
                code=cause.code,
                key=key,
                title=cause.label,
                owner=owner_of(cause.fault, key),
                body=_BODIES[key],
                next=cause.next_step,
            )
        )
    return tuple(found)


def find(word: str) -> Page | None:
    """The page ``word`` names: its code or, for a Rust code, the cause's key too."""
    wanted = word.strip().lower()
    for page in pages():
        if wanted in (page.code.lower(), page.key.lower()):
            return page
    return None


def _render(page: Page) -> str:
    lines = [f"{page.code}  {page.title}"]
    if page.key != page.code:
        # Where result.json and the 0.8 docs name it.
        lines.append(f"category: {page.key.split(':')[0]}")
    lines += [
        f"whose problem: {OWNER_LABELS[page.owner]}",
        "",
        page.body,
        "",
        f"next: {page.next}",
    ]
    return "\n".join(lines) + "\n"


def _listing() -> str:
    all_pages = pages()
    width = max(len(page.code) for page in all_pages)
    unscored = set(failures.unscored_keys())
    lines: list[str] = []
    for heading, keep in (
        ("Unscored: the agent finished, the verifier could not score it", True),
        ("Errored: the run stopped before a score", False),
    ):
        lines.append(heading)
        lines += [
            f"  {page.code:<{width}}  {page.title}"
            for page in all_pages
            if (page.key in unscored) == keep
        ]
        lines.append("")
    lines.append("`bench explain <code>` prints one page.")
    return "\n".join(lines) + "\n"


def explain(
    code: Annotated[
        str | None,
        typer.Argument(
            metavar="[CODE]",
            help="The code the run summary names, such as R212 or verifier_failure "
            "(omit to list every code).",
            show_default=False,
        ),
    ] = None,
    output_json: Annotated[
        bool, typer.Option("--json", help="Print the page, or the list, as JSON.")
    ] = False,
) -> None:
    """Explain why a trial has no score: what happened, whose problem, what next.

    The code is the one the run summary names; without one, every code is listed.

    Examples:

        bench explain                    # every code
        bench explain R212               # the model provider refused the credential
        bench explain verifier_failure --json
    """
    if code is None:
        if output_json:
            listed = [
                {"code": p.code, "title": p.title, "owner": p.owner} for p in pages()
            ]
            typer.echo(json.dumps(listed, indent=2))
        else:
            typer.echo(_listing(), nl=False)
        return
    page = find(code)
    if page is None:
        from benchflow.cli.main import UsageErrorWithNext

        # A category with a page per cause (agent_integration) names them.
        prefix = code.strip().lower() + ":"
        under = [p.code for p in pages() if p.key.lower().startswith(prefix)]
        error: UsageError = UsageErrorWithNext(
            f"no explain page for {code!r}",
            f"its pages are {', '.join(under)}"
            if under
            else "`bench explain` lists every code",
            ctx=get_current_context(silent=True),
        )
        raise error
    if output_json:
        typer.echo(json.dumps(page.as_json(), indent=2))
    else:
        typer.echo(_render(page), nl=False)


def register_explain(app: typer.Typer) -> None:
    app.command("explain")(explain)
