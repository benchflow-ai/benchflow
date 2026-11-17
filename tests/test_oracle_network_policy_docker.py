"""On a real Docker sandbox: the allowlist covers the oracle, not the verifier.

``tests/fixtures/oracle-network-policy`` allows only pypi.org; its solve.sh
fetches example.com (outside the list) and pypi.org (inside it), and its
test.sh fetches example.com again and scores 1 only when the oracle reached
example.com. Under REPORT B10's bug the oracle reached it and scored 1; with
the policy on the oracle's turn it scores 0, while the oracle still reaches
the listed host. The task runs two ways: the oracle as root and as the task's
[agent].user.

The verifier's network is its own, ``[verifier] network_mode``, else the
task's (#234, EN-609: one reader, held for every process by BenchFlow's
helper). These two variants set it public, so the verifier still reaches
example.com after the oracle's turn. A third, ``inherited``, leaves it unset:
its verifier is held to the task's allowlist, and test.sh's own fetch of
example.com is refused (EN-695: the test assumed the verifier was open after
the oracle's turn, which #234 closed).

Marked ``deterministic`` (a real ``bench`` and a real sandbox, no model);
it needs Docker and the internet (the image, pypi.org, example.com).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.integration.deterministic import harness as h

pytestmark = pytest.mark.deterministic

FIXTURE = Path(__file__).parent / "fixtures" / "oracle-network-policy"
SANDBOX, SKIP_REASON = h.select_sandbox()


# The same task two ways: the oracle as root (it joins the policy's group)
# and as the task's [agent].user (matched by its uid). An image whose USER
# is not root cannot run here at all (the sandbox user's setup needs root),
# so its uid match is covered by the unit tests only.
VARIANTS = {
    "root": ("", ""),
    "task-user": ("RUN useradd -m builder\n", 'user = "builder"\n'),
}
# The verifier's own network, for VARIANTS: public, so the oracle's policy is
# all these trials measure.
PUBLIC_VERIFIER = 'network_mode = "public"\n'
# The oracle as root with no verifier network of its own: the task's.
INHERITED = "inherited"

# R3-R1-A-2: a task whose compose drops SETGID (the net-admin overlay still
# adds NET_ADMIN back, so the rules arm) leaves the root oracle unable to join
# the policy's group. iptables is in the image, so arming itself needs no apt
# (whose own privilege drop also needs SETGID). The trial must error, not
# score the untouched workspace 0.
NO_SETGID = "no-setgid"
NO_SETGID_DOCKERFILE = (
    "RUN apt-get update -qq && apt-get install -y -qq iptables >/dev/null "
    "&& rm -rf /var/lib/apt/lists/*\n"
)
NO_SETGID_COMPOSE = "services:\n  main:\n    cap_drop:\n      - SETGID\n"


def _variant(
    tasks: Path,
    name: str,
    dockerfile_tail: str,
    agent_lines: str,
    compose: str | None = None,
    verifier_lines: str = "",
) -> None:
    task = tasks / f"{FIXTURE.name}-{name}"
    shutil.copytree(FIXTURE, task)
    dockerfile = task / "environment" / "Dockerfile"
    dockerfile.write_text(dockerfile.read_text() + dockerfile_tail)
    if compose is not None:
        (task / "environment" / "docker-compose.yaml").write_text(compose)
    toml = task / "task.toml"
    text = toml.read_text().replace("[agent]\n", "[agent]\n" + agent_lines, 1)
    toml.write_text(text.replace("[verifier]\n", "[verifier]\n" + verifier_lines, 1))


@pytest.fixture(scope="module")
def oracle_trials(tmp_path_factory) -> dict[str, Path]:
    if SANDBOX != "docker":
        pytest.skip(SKIP_REASON if SANDBOX is None else "Docker only")
    root = tmp_path_factory.mktemp("oracle-policy")
    tasks = root / "tasks"
    for name, (dockerfile_tail, agent_lines) in VARIANTS.items():
        _variant(
            tasks, name, dockerfile_tail, agent_lines, verifier_lines=PUBLIC_VERIFIER
        )
    _variant(tasks, INHERITED, "", "")
    _variant(tasks, NO_SETGID, NO_SETGID_DOCKERFILE, "", NO_SETGID_COMPOSE)
    jobs = root / "jobs"
    env = {**os.environ, "BENCHFLOW_SKIP_UPDATE_CHECK": "1"}
    proc = subprocess.run(
        [
            h.bench_executable(),
            "eval",
            "run",
            "--tasks-dir",
            str(tasks),
            "--agent",
            "oracle",
            "--sandbox",
            "docker",
            "--concurrency",
            "3",
            "--jobs-dir",
            str(jobs),
        ],
        cwd=h.REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=1200,
    )
    trials = {
        name: [p.parent for p in jobs.glob(f"*/{FIXTURE.name}-{name}__*/result.json")]
        for name in [*VARIANTS, INHERITED, NO_SETGID]
    }
    assert all(len(found) == 1 for found in trials.values()), proc.stdout[-4000:]
    return {name: found[0] for name, found in trials.items()}


@pytest.fixture(params=list(VARIANTS))
def oracle_trial(request, oracle_trials) -> Path:
    return oracle_trials[request.param]


# What an image whose compose drops SETGID ends with since #234: the root
# oracle joins no group (every-process rules from the helper hold it), and
# hardening cannot switch to the sandbox user to stop its processes.
NO_SETGID_ERROR = (
    "RuntimeError: Verifier hardening failed: sandbox-user process quiescence "
    "check exited with rc=2\nstderr: Cannot send SIGSTOP as the sandbox user: "
    "neither setpriv nor su ran it as UID 1000 GID 1000 (install setpriv or su "
    "in the image and let root switch to that UID and GID)"
)


def test_an_oracle_image_without_setgid_is_an_error_never_a_score(oracle_trials):
    """Was test_an_oracle_that_cannot_join_the_policys_group_is_an_error. Before
    REPORT B10's fix the join exited 1 before solve.sh started, only rc=1 was
    logged, and the verifier scored the untouched workspace 0. Since #234 the
    oracle joins no group, so it cannot fail there; the trial still ends in an
    error and no score, now at hardening's switch to the sandbox user
    (EN-695)."""
    result = json.loads((oracle_trials[NO_SETGID] / "result.json").read_text())
    assert result.get("error") == NO_SETGID_ERROR, result
    assert "cannot join group" not in (result.get("error") or "")
    assert not result.get("rewards"), result


def _verifier_said(trial: Path) -> dict[str, str]:
    said: dict[str, str] = {}
    for path in (trial / "verifier").rglob("*"):
        if not path.is_file():
            continue
        for line in path.read_text(errors="replace").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key in {
                "verifier_reached",
                "oracle_reached",
                "oracle_allowed",
                "oracle_direct",
                "oracle_uid",
                "oracle_apt_lists",
            }:
                said[key] = value
    return said


def test_the_oracle_cannot_reach_a_host_outside_the_allowlist(oracle_trial):
    result = json.loads((oracle_trial / "result.json").read_text())
    assert result["rewards"] == {"reward": 0.0}, result.get("error")
    said = _verifier_said(oracle_trial)
    assert said.get("oracle_reached") == "refused", said
    # The proxy logged the refused attempt in the trial folder.
    blocked = (oracle_trial / "trajectory" / "egress_denylist.jsonl").read_text()
    assert "example.com" in blocked


def test_the_oracle_cannot_get_around_the_proxy(oracle_trial):
    """A direct connection to an address the list does not name is refused by
    the firewall rules, not only steered by the proxy variables."""
    assert _verifier_said(oracle_trial).get("oracle_direct") == "refused"


def test_the_oracle_still_reaches_the_listed_host(oracle_trial):
    assert _verifier_said(oracle_trial).get("oracle_allowed") == "reached"


def test_the_verifier_reaches_its_hosts_after_the_oracles_turn(oracle_trial):
    """Its own network, public: the oracle's policy does not follow it."""
    assert _verifier_said(oracle_trial).get("verifier_reached") == "reached"


def test_a_verifier_with_no_network_of_its_own_is_held_to_the_tasks(oracle_trials):
    """#234's one reader: no [verifier] network_mode, so the task's allowlist
    holds test.sh too, for every process. test.sh's own fetch of example.com
    is refused at the egress proxy, and a test.sh that fails after a refusal
    ends "verifier network refused" (unscored), never a silent 0."""
    trial = oracle_trials[INHERITED]
    said = _verifier_said(trial)
    assert said.get("verifier_reached") == "refused", said
    assert said.get("oracle_reached") == "refused", said
    result = json.loads((trial / "result.json").read_text())
    assert result.get("rewards") is None, result
    assert (result.get("verifier_error") or "").startswith(
        "verifier network refused: test.sh exited 0 with reward 0.0 after the "
        "verifier's network (in-container under allowlist: verifier.network_mode, "
        "else the sandbox's policy) refused it "
    ), result
    record = json.loads((trial / "verifier" / "verifier-network.json").read_text())
    assert record["mode"] == "allowlist", record
    assert record["state"] == "held", record
    assert {
        "method": "CONNECT",
        "rule": "not-allowlisted",
        "url": "example.com:443",
    } in (record["proxy_refusals"]), record


def test_arming_leaves_the_images_apt_lists_as_built(oracle_trial):
    """python:3.12-slim ships without apt lists; installing iptables for the
    policy must not hand the oracle a refreshed package index."""
    assert _verifier_said(oracle_trial).get("oracle_apt_lists") == "0"


def test_the_policy_records_its_cost(oracle_trial):
    timing = json.loads((oracle_trial / "timing.json").read_text())
    assert timing["oracle_network_policy"] > 0


def test_the_oracle_ran_as_the_variant_says(oracle_trials):
    """Root joined the policy's group; the task-user variant, matched by uid,
    ran as builder (so the uid rules, not the group, confined it)."""
    uids = {
        name: _verifier_said(t).get("oracle_uid") for name, t in oracle_trials.items()
    }
    assert uids["root"] == "0", uids
    assert (uids["task-user"] or "").isdigit() and uids["task-user"] != "0", uids


def test_the_finished_trial_keeps_no_scoring_lock(oracle_trials):
    """REPORT B17: an empty .scoring.lock was left in every finished trial."""
    for name in VARIANTS:
        assert not (oracle_trials[name] / ".scoring.lock").exists()
