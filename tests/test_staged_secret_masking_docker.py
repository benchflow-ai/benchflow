"""BenchFlow masks credentials it staged, in a real trial (deterministic tier).

``claude-agent-acp`` runs on the proxy route against the scripted fake
provider, with three staged tokens passed by ``--agent-env``: one with a
quote and a backslash, one whose base64 differs between the two alphabets,
and one long enough to wrap. Its one Bash call reads its own environment and
prints each token and its proxy key as they are, base64-encoded (both
alphabets, at each byte offset) and wrapped (64 and 76 columns, ``\\n`` and
``\\r\\n``); it also prints each reversed, a form outside the slice, which
lets this test recover the proxy key. Then it writes the file the task's
test wants, so the trial still scores through the proxy.

Every file of the job folder, and the job's bundle, is scanned for every
form of every staged value. A file BenchFlow writes through its redaction
layer (``masking.json``'s ``applied_to``) must hold none; a hit is allowed
only where ``masking.json`` says this version does not sweep: what the
sandbox writes into ``verifier/``, ``agent/`` or ``artifacts/``, and the
job's own files.

``bench eval branch`` drives a trial's steps itself, not through
``Rollout.run``: the same task branched, the parent and each child printing
the tokens, leaves none in a covered file of the parent's folder or a
child's, and each child's folder has the record. So does a rollout an SDK
caller drives step by step, as ``docs/examples/branch-agent-run.py`` does.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tests.integration.deterministic import harness as h
from tests.staged_secret_forms import LONG, TOKEN, URLSAFE, scan_forms

pytestmark = pytest.mark.deterministic

SANDBOX, SKIP_REASON = h.select_sandbox()
REQUESTED = os.environ.get(h.SANDBOX_ENV, "").strip().lower() in {"docker", "daytona"}

STAGED_ENV = {
    "BF_STAGED_TOKEN": TOKEN,
    "BF_STAGED_URLSAFE_TOKEN": URLSAFE,
    "BF_STAGED_LONG_TOKEN": LONG,
}


@pytest.fixture(scope="module")
def staged_job(tmp_path_factory) -> h.CliRun:
    if SANDBOX is None:
        if REQUESTED:
            pytest.fail(SKIP_REASON)
        pytest.skip(SKIP_REASON)
    root = tmp_path_factory.mktemp("staged")
    tasks = root / "tasks"
    h.materialize_task(h.TaskVariant("staged-echo", "staged-echo"), tasks)
    agent_env: list[str] = []
    for name, value in STAGED_ENV.items():
        agent_env += ["--agent-env", f"{name}={value}"]
    with h.host_fake_llm(root / "host-fake-llm.jsonl") as url:
        return h.run_bench(
            ["eval", "run"],
            tasks_dir=tasks,
            jobs_dir=root / "jobs",
            sandbox=SANDBOX,
            host_fake_url=url,
            extra=["--retry-attempts", "0", *agent_env],
            env_extra={"BENCHFLOW_KEEP_AGENT_SESSION_LOGS": "1"},
        )


def _trial(run: h.CliRun) -> Path:
    (result,) = [
        p
        for p in run.jobs_dir.glob("**/result.json")
        if (p.parent / "trajectory").is_dir()
    ]
    return result.parent


def _record(trial: Path) -> dict:
    return json.loads((trial / "integrity" / "masking.json").read_text())


def _staged_values(trial: Path) -> dict[str, str]:
    """The staged tokens, and the proxy key the agent printed reversed."""
    text = (trial / "trajectory" / "acp_trajectory.jsonl").read_text()
    found = re.search(
        r"staged (?:ANTHROPIC_AUTH_TOKEN|ANTHROPIC_API_KEY) reversed ([A-Za-z0-9_.~-]+)",
        text,
    )
    assert found, "the agent printed its proxy key reversed"
    return {**STAGED_ENV, "proxy key": found[1][::-1]}


def _hits(data: bytes, values: dict[str, str]) -> list[str]:
    """Each form found, by name, the form's SHA-256 and length: never a
    part of the form."""
    text = data.decode("utf-8", errors="replace")
    return sorted(
        f"{name}: sha256 {hashlib.sha256(form.encode()).hexdigest()[:12]}, "
        f"{len(form)} characters"
        for name, value in values.items()
        for form in scan_forms(value)
        if form in text
    )


def _excused(rel: str, record: dict, *, job_level: bool) -> bool:
    """Whether a hit in *rel* is where ``masking.json`` says it does not sweep."""
    if not record:
        return False
    if job_level:
        return rel in record["not_swept"]["job_files"]
    if any(rel == c or rel.startswith(c) for c in record["applied_to"]):
        return False
    return rel.startswith(tuple(record["not_swept"]["folders"]))


def _trial_folders(job: Path) -> list[str]:
    """Each trial folder of *job*, relative to it, deepest first: each trial
    (finished, or with a record), and each child folder a branch made under
    one."""
    folders = {p.parent for p in job.glob("*/result.json")}
    folders |= {p.parent.parent for p in job.glob("*/integrity/masking.json")}
    folders |= {p for p in job.glob("*/branches/*/children/*") if p.is_dir()}
    return sorted((str(f.relative_to(job)) for f in folders), key=len, reverse=True)


def _leaks(job: Path, files: dict[str, bytes], values: dict[str, str]) -> dict:
    """{file: hits} for each file with a hit where its trial's record does
    not say it goes unswept (none without a record). *files* is keyed by
    the path under the job folder."""
    folders = _trial_folders(job)
    records = {
        folder: json.loads(record.read_text())
        if (record := job / folder / "integrity" / "masking.json").is_file()
        else {}
        for folder in folders
    }
    leaks = {}
    for rel, data in sorted(files.items()):
        hits = _hits(data, values)
        if not hits:
            continue
        owner = next((f for f in folders if rel.startswith(f"{f}/")), None)
        if owner is None:
            excused = folders != [] and _excused(
                rel, records[folders[-1]], job_level=True
            )
        else:
            excused = _excused(rel[len(owner) + 1 :], records[owner], job_level=False)
        if not excused:
            leaks[rel] = hits
    return leaks


def _job_files(job: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(job)): path.read_bytes()
        for path in job.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def test_the_proxy_route_still_scores(staged_job):
    trial = _trial(staged_job)
    result = json.loads((trial / "result.json").read_text())
    assert result["rewards"] == {"reward": 1.0}, staged_job.stdout[-2000:]


def test_no_covered_file_of_the_trial_or_job_holds_a_staged_value(staged_job):
    trial = _trial(staged_job)
    values = _staged_values(trial)
    files = _job_files(trial.parent)
    printout = files[f"{trial.name}/trajectory/acp_trajectory.jsonl"].decode()
    assert "staged BF_STAGED_TOKEN exact" in printout, (
        "the agent's printout is recorded"
    )
    assert _leaks(trial.parent, files, values) == {}


def test_the_jobs_bundle_holds_no_staged_value_outside_what_is_not_swept(
    staged_job, tmp_path
):
    from benchflow.evidence_bundle import build_bundle

    trial = _trial(staged_job)
    values = _staged_values(trial)
    result = build_bundle(trial.parent, tmp_path / "run.zip")
    with zipfile.ZipFile(result.path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    assert any(name.startswith(f"{trial.name}/trajectory/") for name in members)
    assert _leaks(trial.parent, members, values) == {}


def test_the_agents_proxy_key_is_masked_wherever_it_printed_it(staged_job):
    """The key the agent was given for the proxy is staged too: each line
    it printed it on (but the reversed one) holds a mask, not the key."""
    trial = _trial(staged_job)
    text = (trial / "trajectory" / "acp_trajectory.jsonl").read_text()
    # The tool output's lines, as the JSON line holds them (escaped).
    lines = [
        line
        for line in text.split("\\n")
        if line.startswith(("staged ANTHROPIC_AUTH_TOKEN", "staged ANTHROPIC_API_KEY"))
        and not re.search(r" (wrap\d+(?:-crlf)?$|reversed )", line)
    ]
    assert len(lines) >= 7, lines
    assert all("[masked:" in line for line in lines), lines


def test_the_masking_record_counts_by_name_and_names_what_it_did_not_sweep(
    staged_job,
):
    trial = _trial(staged_job)
    record = _record(trial)
    for name in STAGED_ENV:
        assert record["masked"].get(name, 0) >= 10, (name, record["masked"])
    others = {n: c for n, c in record["masked"].items() if n not in STAGED_ENV}
    assert max(others.values(), default=0) >= 10, "the agent's proxy key"
    assert "agent/" in record["not_swept"]["folders"]
    text = (trial / "integrity" / "masking.json").read_bytes()
    assert _hits(text, _staged_values(trial)) == []


@pytest.fixture(scope="module")
def staged_branch(tmp_path_factory) -> h.CliRun:
    """The task branched after its one prompt: the parent prints the staged
    tokens, then two children, each in its own sandbox, print them again.
    Branching takes the native route (it refuses a model proxy)."""
    if SANDBOX is None:
        if REQUESTED:
            pytest.fail(SKIP_REASON)
        pytest.skip(SKIP_REASON)
    root = tmp_path_factory.mktemp("staged-branch")
    tasks = root / "tasks"
    h.materialize_task(h.TaskVariant("staged-echo", "staged-echo"), tasks)
    agent_env: list[str] = []
    for name, value in STAGED_ENV.items():
        agent_env += ["--agent-env", f"{name}={value}"]
    children: list[str] = []
    for label in ("one", "two"):
        prompt = f"label={label},prompt=Print them again. {h.marker('staged-echo')}"
        children += ["--child", prompt]
    with h.host_fake_llm(root / "host-fake-llm.jsonl") as url:
        return h.run_bench(
            ["eval", "branch"],
            tasks_dir=tasks / "staged-echo",
            jobs_dir=root / "jobs",
            sandbox=SANDBOX,
            host_fake_url=url,
            route="native",
            extra=[
                "--prompt",
                "@instruction",
                "--checkpoint-after-prompt",
                "1",
                *children,
                *agent_env,
            ],
            env_extra={"BENCHFLOW_KEEP_AGENT_SESSION_LOGS": "1"},
        )


def test_a_branch_leaves_no_staged_value_in_a_covered_file(staged_branch):
    """The parent's folder and each child's: every file is scanned, and a
    hit is allowed only where that folder's record says it is not swept."""
    assert staged_branch.returncode == 0, staged_branch.output[-4000:]
    (trial,) = staged_branch.trial_dirs()
    files = _job_files(trial.parent)
    printouts = [
        rel
        for rel in files
        if rel.endswith("trajectory/acp_trajectory.jsonl")
        and "staged BF_STAGED_TOKEN exact" in files[rel].decode()
    ]
    assert len(printouts) == 3, sorted(files)  # the parent and two children
    assert sum("/branches/" in rel for rel in printouts) == 2
    assert _leaks(trial.parent, files, STAGED_ENV) == {}


def test_each_branch_child_has_the_trials_masking_record(staged_branch):
    (trial,) = staged_branch.trial_dirs()
    children = sorted(p for p in trial.glob("branches/*/children/*") if p.is_dir())
    assert len(children) == 2, children
    for folder in (trial, *children):
        record = _record(folder)
        for name in STAGED_ENV:
            # One registry masks the branch: each folder holds its totals.
            counts = record["branch_total"]
            assert counts.get(name, 0) >= 10, (folder.name, counts)
        assert (
            _hits((folder / "integrity" / "masking.json").read_bytes(), STAGED_ENV)
            == []
        )


#: ``docs/examples/branch-agent-run.py``'s sequence, on the staged task: setup,
#: start, install, connect, a turn, two branch children (each: connect, a
#: turn, verify), the parent's next turn, verify, finalize. No run(). The config
#: takes the task's declared environment, as ``bench eval branch`` passes it:
#: it starts the scripted provider in the sandbox.
SDK_DRIVER = """
import asyncio, json, sys
from pathlib import Path

from benchflow.evaluation import _environment_manifest_from_task_document
from benchflow.rollout import BranchChild, Rollout, RolloutConfig

task, jobs, agent, model, sandbox, agent_env = sys.argv[1:7]


async def main():
    rollout = Rollout(
        RolloutConfig(
            task_path=Path(task),
            agent=agent,
            model=model,
            environment=sandbox,
            jobs_dir=Path(jobs),
            job_name="manual",
            agent_env=json.loads(agent_env),
            # The scripted provider is the task's service in main, beside
            # the agent's token: accepted, as the deterministic harness does.
            allow_task_processes_with_credentials=True,
            environment_manifest=_environment_manifest_from_task_document(
                Path(task)
            ),
        )
    )

    async def run_child(node, *, child: BranchChild):
        rewards = None
        try:
            await rollout.connect()
            await rollout.execute(None, node=node)
            rewards = await rollout.verify()
        finally:
            await rollout.disconnect()
        return (rewards or {}).get("reward")

    try:
        await rollout.setup()
        await rollout.start()
        await rollout.install_agent()
        await rollout.connect()
        await rollout.execute()
        await rollout.branch(
            2, run_child, snapshot_layers={"sandbox"}, child_labels=["one", "two"]
        )
        await rollout.connect()
        await rollout.execute()
        await rollout.verify()
        await rollout.finalize()
    except BaseException:
        await rollout.cleanup()
        raise


asyncio.run(main())
"""


@pytest.fixture(scope="module")
def staged_sdk(tmp_path_factory) -> Path:
    """The job folder of a rollout driven step by step, as the docs example
    drives it, on the native route; its agent prints the staged tokens."""
    if SANDBOX is None:
        if REQUESTED:
            pytest.fail(SKIP_REASON)
        pytest.skip(SKIP_REASON)
    root = tmp_path_factory.mktemp("staged-sdk")
    tasks = root / "tasks"
    task = h.materialize_task(h.TaskVariant("staged-echo", "staged-echo"), tasks)
    jobs = root / "jobs"
    _, env = h.bench_command(
        ["eval", "run"],
        tasks_dir=task,
        jobs_dir=jobs,
        sandbox=SANDBOX,
        host_fake_url=None,
        route="native",
    )
    env["BENCHFLOW_KEEP_AGENT_SESSION_LOGS"] = "1"
    driver = root / "driver.py"
    driver.write_text(SDK_DRIVER)
    agent_env = {**h.route_env("native", SANDBOX, None), **STAGED_ENV}
    proc = subprocess.run(
        [
            sys.executable,
            str(driver),
            str(task),
            str(jobs),
            h.route_agent("native"),
            h.route_model("native"),
            SANDBOX,
            json.dumps(agent_env),
        ],
        cwd=h.REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1500,
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-4000:]
    (job,) = [p for p in jobs.iterdir() if p.is_dir()]
    return job


def test_a_rollout_driven_step_by_step_leaves_no_staged_value_in_a_covered_file(
    staged_sdk,
):
    """No run(), no verb: setup() bound the trial's registry, so the
    trajectory and every covered file of the parent's folder and the
    children's are masked, and finalize() wrote the record in each."""
    files = _job_files(staged_sdk)
    (trial,) = [p.parent for p in staged_sdk.glob("*/result.json")]
    printout = files[f"{trial.name}/trajectory/acp_trajectory.jsonl"].decode()
    assert "staged BF_STAGED_TOKEN exact" in printout
    assert "[masked:BF_STAGED_TOKEN]" in printout
    assert _leaks(staged_sdk, files, STAGED_ENV) == {}
    children = sorted(p for p in trial.glob("branches/*/children/*") if p.is_dir())
    assert len(children) == 2, children
    for folder in (trial, *children):
        counts = _record(folder)["branch_total"]
        assert counts.get("BF_STAGED_TOKEN", 0) >= 10, (folder.name, counts)


#: Two rollouts of the staged task in one process, each with its own token
#: (under its own name), driven step by step in an order: one after the
#: other; both set up first, then each driven and finished; the first set up
#: and driven, then left unfinished while the second runs (its sandbox goes
#: last, with no finalize).
TWO_DRIVER = """
import asyncio, json, sys
from pathlib import Path

from benchflow.evaluation import _environment_manifest_from_task_document
from benchflow.rollout import Rollout, RolloutConfig

task, jobs, agent, model, sandbox, envs, order = sys.argv[1:8]
envs = json.loads(envs)


def make(label):
    return Rollout(
        RolloutConfig(
            task_path=Path(task),
            agent=agent,
            model=model,
            environment=sandbox,
            jobs_dir=Path(jobs),
            job_name=f"manual-{label}",
            agent_env=envs[label],
            # The scripted provider is the task's service in main, beside
            # the agent's token: accepted, as the deterministic harness does.
            allow_task_processes_with_credentials=True,
            environment_manifest=_environment_manifest_from_task_document(
                Path(task)
            ),
        )
    )


async def prepare(rollout):
    await rollout.setup()
    await rollout.start()
    await rollout.install_agent()


async def drive(rollout, *, finish=True):
    await rollout.connect()
    await rollout.execute()
    if finish:
        await rollout.verify()
        await rollout.finalize()


async def main():
    a, b = make("a"), make("b")
    try:
        if order == "sequence":
            await prepare(a)
            await drive(a)
            await prepare(b)
            await drive(b)
        elif order == "batch":
            await prepare(a)
            await prepare(b)
            await drive(a)
            await drive(b)
        else:
            await prepare(a)
            await drive(a, finish=False)
            await prepare(b)
            await drive(b)
            record = a._rollout_dir / "integrity" / "masking.json"
            (Path(jobs) / "left.json").write_text(
                json.dumps({"record": record.is_file()})
            )
            await a.cleanup()
    except BaseException:
        for rollout in (a, b):
            try:
                await rollout.cleanup()
            except Exception:
                pass
        raise


asyncio.run(main())
"""

#: Each rollout's token, under a name the other doesn't stage.
TWO_TOKENS = {"a": {"BF_STAGED_TOKEN": TOKEN}, "b": {"BF_STAGED_LONG_TOKEN": LONG}}


@pytest.fixture(scope="module", params=["sequence", "batch", "abandon"])
def two_rollouts(request, tmp_path_factory) -> tuple[str, Path]:
    if SANDBOX is None:
        if REQUESTED:
            pytest.fail(SKIP_REASON)
        pytest.skip(SKIP_REASON)
    order = request.param
    root = tmp_path_factory.mktemp(f"two-{order}")
    task = h.materialize_task(
        h.TaskVariant("staged-echo", "staged-echo"), root / "tasks"
    )
    jobs = root / "jobs"
    _, env = h.bench_command(
        ["eval", "run"],
        tasks_dir=task,
        jobs_dir=jobs,
        sandbox=SANDBOX,
        host_fake_url=None,
        route="native",
    )
    driver = root / "driver.py"
    driver.write_text(TWO_DRIVER)
    envs = {
        label: {**h.route_env("native", SANDBOX, None), **tokens}
        for label, tokens in TWO_TOKENS.items()
    }
    proc = subprocess.run(
        [
            sys.executable,
            str(driver),
            str(task),
            str(jobs),
            h.route_agent("native"),
            h.route_model("native"),
            SANDBOX,
            json.dumps(envs),
            order,
        ],
        cwd=h.REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1500,
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-4000:]
    return order, jobs


def test_two_rollouts_in_one_process_are_each_masked_with_their_own_record(
    two_rollouts,
):
    """Whatever the order, each rollout's trajectory masks its own token,
    no covered file of either holds either token, and each folder's record
    names its own token alone. One left unfinished keeps the record its last
    step wrote."""
    order, jobs = two_rollouts
    values = {
        name: value for tokens in TWO_TOKENS.values() for name, value in tokens.items()
    }
    for label, tokens in TWO_TOKENS.items():
        job = jobs / f"manual-{label}"
        (trial,) = [p for p in job.iterdir() if (p / "trajectory").is_dir()]
        (name,) = tokens
        (other,) = set(values) - {name}
        files = _job_files(job)
        printout = files[f"{trial.name}/trajectory/acp_trajectory.jsonl"].decode()
        assert f"[masked:{name}]" in printout, (order, label)
        assert _leaks(job, files, values) == {}, (order, label)
        record = _record(trial)
        assert name in record["registered"], (order, label, record["registered"])
        assert other not in record["registered"], (order, label)
        assert record["masked"].get(name, 0) >= 10, (order, label, record["masked"])
    if order == "abandon":
        assert json.loads((jobs / "left.json").read_text()) == {"record": True}
