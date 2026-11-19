"""a task the sandbox cannot honor errors the same way in a dry run and
in the run.

NETSCAN (rc c03771584): a task with ``[verifier] network_mode = allowlist`` on
Docker passed ``bench … --dry-run``, then every trial errored at once with
``UnsupportedTaskFeatureError`` at the run's sandbox-free plan gate
(``validate_task_plan``, from the rollout's setup). The dry run now calls that
gate for each task and each matrix cell, as ``bench tasks check`` does.

Each sandbox gets a feature its own capability table refuses, read from the
table when the test runs, so the test holds whichever features a sandbox
enforces (#234 starts enforcing verifier allowlists on Docker).
"""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.sandbox.providers import SANDBOX_PROVIDERS, sandbox_sdk_missing
from benchflow.task.config import TaskConfig
from benchflow.task.runtime_capabilities import validate_task_runtime_support

#: Candidate features: (TaskConfig fields, the task.toml [environment] lines).
FEATURES = {
    "gpus": ({"sandbox": {"gpus": 1}}, "gpus = 1\n"),
    "allowlist": (
        {"sandbox": {"network_mode": "allowlist", "allowed_hosts": ["example.com"]}},
        'network_mode = "allowlist"\nallowed_hosts = ["example.com"]\n',
    ),
    "no-network": (
        {"sandbox": {"network_mode": "no-network"}},
        'network_mode = "no-network"\n',
    ),
}


def _refused(sandbox: str) -> str:
    """The first candidate feature ``sandbox``'s capability table refuses."""
    for name, (fields, _) in FEATURES.items():
        if validate_task_runtime_support(
            TaskConfig.model_validate(fields), sandbox=sandbox
        ):
            return name
    raise AssertionError(
        f"{sandbox} refuses none of {sorted(FEATURES)}: add one it refuses"
    )


def _task(root: Path, name: str, environment: str = "", verifier: str = "") -> Path:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text(
        "FROM ubuntu:24.04\nWORKDIR /app\n"
    )
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text(
        "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"
    )
    (task / "solution").mkdir()
    (task / "solution" / "solve.sh").write_text("#!/bin/sh\ntrue\n")
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[agent]\ntimeout_sec = 300\n\n'
        f"[verifier]\ntimeout_sec = 120\n{verifier}\n[environment]\n{environment}"
    )
    (task / "instruction.md").write_text("Do nothing.\n")
    return task


@pytest.fixture
def bench(tmp_path, monkeypatch):
    """``bench eval run`` on tmp_path/tasks, the sandbox's preflight skipped
    (no daemon or account here) and made-up keys: the run's trials reach the
    plan gate, which comes before any sandbox."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BENCHFLOW_SKIP_PREFLIGHT", "1")
    monkeypatch.setenv("DAYTONA_API_KEY", "fake-not-a-key")
    monkeypatch.setenv("MODAL_TOKEN_ID", "fake-not-a-key")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "fake-not-a-key")

    def run(sandbox: str, *flags: str, jobs: str = "jobs", agent: str = "oracle"):
        return CliRunner().invoke(
            app,
            [
                "eval",
                "run",
                "--tasks-dir",
                str(tmp_path / "tasks"),
                "--agent",
                agent,
                "--sandbox",
                sandbox,
                "--jobs-dir",
                str(tmp_path / jobs),
                *flags,
            ],
            prog_name="bench",
        )

    return run


def _plan(result) -> dict:
    text = result.stdout
    return json.JSONDecoder().raw_decode(text[text.index("{") :])[0]


def _recorded(jobs: Path) -> str:
    (result,) = [p for p in jobs.rglob("result.json") if p.parent.name != jobs.name]
    return json.loads(result.read_text())["error"]


def _same_in_dry_run_and_run(
    tmp_path, bench, sandbox: str, name: str, *flags: str
) -> str:
    dry = bench(sandbox, "--dry-run", *flags, jobs="dry")
    run = bench(sandbox, *flags, jobs="run")
    recorded = _recorded(tmp_path / "run")
    assert run.exit_code == 1, run.output
    assert recorded.startswith("UnsupportedTaskFeatureError: "), recorded
    assert dry.exit_code == 1, dry.output
    assert _plan(dry)["errors"] == {name: recorded}
    assert f"{name}: {recorded.splitlines()[0]}" in dry.output
    return recorded


@pytest.mark.parametrize("sandbox", SANDBOX_PROVIDERS)
def test_a_task_the_sandbox_refuses_errors_the_same_in_the_dry_run_and_the_run(
    tmp_path, bench, sandbox
):
    if sandbox_sdk_missing(sandbox):
        pytest.skip(f"{sandbox}'s SDK is an optional extra not installed here")
    feature = _refused(sandbox)
    _task(tmp_path / "tasks", "refused", environment=FEATURES[feature][1])
    _same_in_dry_run_and_run(tmp_path, bench, sandbox, "refused")


#: The reported task's [verifier] lines (NETSCAN).
VERIFIER_ALLOWLIST = 'network_mode = "allowlist"\nallowed_hosts = ["example.com"]\n'


@pytest.mark.parametrize("sandbox", SANDBOX_PROVIDERS)
def test_netscans_verifier_allowlist_is_judged_the_same_in_both(
    tmp_path, bench, sandbox_reached, sandbox
):
    """The reported task. Docker holds a verifier allowlist since #234, so there
    the dry run and the run accept it alike: the dry run plans the task with no
    error, and the run's trial passes every check before its sandbox (a Docker
    run holding it: the test below). A sandbox whose table still refuses it
    errors the same in both, as before."""
    if sandbox_sdk_missing(sandbox):
        pytest.skip(f"{sandbox}'s SDK is an optional extra not installed here")
    fields = {
        "verifier": {"network_mode": "allowlist", "allowed_hosts": ["example.com"]}
    }
    refused = validate_task_runtime_support(
        TaskConfig.model_validate(fields), sandbox=sandbox
    )
    if sandbox == "docker":
        assert not refused, refused
    _task(tmp_path / "tasks", "allowlist-verifier", verifier=VERIFIER_ALLOWLIST)
    if refused:
        recorded = _same_in_dry_run_and_run(
            tmp_path, bench, sandbox, "allowlist-verifier"
        )
        # The verifier's allowlist named: by the capability table, or by the
        # network policy refusal before it (Daytona and Modal).
        assert "allowlist" in recorded and "verifier" in recorded, recorded
        return
    dry, _run, recorded = _both(tmp_path, bench, sandbox)
    assert recorded == REACHED, recorded
    assert dry.exit_code == 0, dry.output
    plan = _plan(dry)
    assert plan["tasks"] == ["allowlist-verifier"]
    assert "errors" not in plan


@pytest.mark.deterministic
def test_netscans_verifier_allowlist_is_held_by_a_docker_run(tmp_path):
    """A real Docker run of the reported task beside its dry run. Both accept
    it (exit 0, the same task), and the run holds the allowlist: test.sh, as
    root, connects straight to an address the list does not name, and
    verifier/verifier-network.json records the allowlist held for every
    process, through the egress proxy's own container, with that refusal
    counted. Needs Docker (CI: the grading-integrity job)."""
    import subprocess

    from tests.integration.deterministic import harness as h

    sandbox, reason = h.select_sandbox()
    if sandbox != "docker":
        pytest.skip(reason if sandbox is None else "Docker only")
    task = _task(tmp_path / "tasks", "allowlist-verifier", verifier=VERIFIER_ALLOWLIST)
    (task / "tests" / "test.sh").write_text(
        "#!/bin/bash\n"
        "# Straight out, past the proxy, to an address the allowlist does not name.\n"
        "timeout 5 bash -c 'exec 3<>/dev/tcp/1.1.1.1/443' 2>/dev/null\n"
        "echo 1 > /logs/verifier/reward.txt\n"
    )

    def bench(*flags: str, jobs: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                h.bench_executable(),
                "eval",
                "run",
                "--tasks-dir",
                str(tmp_path / "tasks"),
                "--agent",
                "oracle",
                "--sandbox",
                "docker",
                "--jobs-dir",
                str(tmp_path / jobs),
                *flags,
            ],
            cwd=h.REPO_ROOT,
            env={**os.environ, "BENCHFLOW_SKIP_UPDATE_CHECK": "1"},
            capture_output=True,
            text=True,
            timeout=1200,
        )

    dry = bench("--dry-run", jobs="dry")
    assert dry.returncode == 0, dry.stdout[-4000:] + dry.stderr[-4000:]
    plan = json.JSONDecoder().raw_decode(dry.stdout[dry.stdout.index("{") :])[0]
    assert plan["tasks"] == ["allowlist-verifier"]
    assert "errors" not in plan
    run = bench(jobs="run")
    assert run.returncode == dry.returncode, run.stdout[-4000:] + run.stderr[-4000:]
    (trial,) = [
        p.parent for p in (tmp_path / "run").glob("*/allowlist-verifier__*/result.json")
    ]
    result = json.loads((trial / "result.json").read_text())
    assert result["error"] is None, result["error"]
    assert result["rewards"]["reward"] == 1, result["rewards"]
    record = json.loads((trial / "verifier" / "verifier-network.json").read_text())
    assert record["mode"] == "allowlist", record
    assert record["state"] == "held", record
    assert record["where"] == "in-container", record
    assert record["proxy"] == "own container", record
    assert record["refused_packets"] >= 1, record


def test_a_supported_task_dry_runs_as_before(tmp_path, bench):
    _task(tmp_path / "tasks", "plain")
    dry = bench("docker", "--dry-run")
    assert dry.exit_code == 0, dry.output
    plan = _plan(dry)
    assert plan["tasks"] == ["plain"]
    assert "errors" not in plan and "unchecked" not in plan


def test_a_matrix_dry_run_names_each_cell_and_task_that_would_error(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    _task(tmp_path / "ds-a", "gpu", environment=FEATURES["gpus"][1])
    _task(tmp_path / "ds-b", "plain")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "agents": [{"name": "oracle"}],
                "datasets": [{"path": "ds-a"}, {"path": "ds-b"}],
            }
        )
    )
    dry = CliRunner().invoke(
        app, ["run", "--config", str(config), "--dry-run"], prog_name="bench"
    )
    assert dry.exit_code == 1, dry.output
    jobs = {job["cell"]: job for job in _plan(dry)["jobs"]}
    refused = [job for job in jobs.values() if "errors" in job]
    assert len(refused) == 1 and list(refused[0]["errors"]) == ["gpu"]
    assert refused[0]["errors"]["gpu"].startswith("UnsupportedTaskFeatureError: ")
    assert "sandbox.gpus" in refused[0]["errors"]["gpu"]


# --- R632: the dry run checks the plan the run builds (round 12) ---------------


@pytest.fixture
def sandbox_reached(monkeypatch):
    """The run stops where it would create its sandbox, so a trial records
    either a refusal from before the sandbox or this error, once every check
    before it has passed. Nothing is built."""
    from benchflow.rollout_planes import DefaultRolloutPlanes

    def reached(self, *args, **kwargs):
        raise RuntimeError("SandboxCreationReached")

    monkeypatch.setattr(DefaultRolloutPlanes, "create_environment", reached)


REACHED = "RuntimeError: SandboxCreationReached"


def _both(tmp_path, bench, sandbox, *flags, agent="oracle"):
    dry = bench(sandbox, "--dry-run", *flags, jobs="dry", agent=agent)
    run = bench(sandbox, *flags, jobs="run", agent=agent)
    return dry, run, _recorded(tmp_path / "run")


def _override(tmp_path, fields) -> str:
    path = tmp_path / "override.json"
    path.write_text(json.dumps(fields))
    return f"@{path}"


def test_an_override_that_removes_the_refused_feature_passes_both(
    tmp_path, bench, sandbox_reached
):
    """R632 1: the dry run checked the task as on disk and refused a run the
    override (gpus 1 -> 0) makes work."""
    _task(tmp_path / "tasks", "gpu", environment=FEATURES["gpus"][1])
    flags = ("--config-override", _override(tmp_path, {"sandbox": {"gpus": 0}}))
    dry, _run, recorded = _both(tmp_path, bench, "docker", *flags)
    assert recorded == REACHED, recorded
    assert dry.exit_code == 0, dry.output
    assert "errors" not in _plan(dry)


def test_an_override_that_adds_a_refused_feature_refuses_both(
    tmp_path, bench, sandbox_reached
):
    _task(tmp_path / "tasks", "plain")
    flags = ("--config-override", _override(tmp_path, {"sandbox": {"gpus": 1}}))
    dry, _run, recorded = _both(tmp_path, bench, "docker", *flags)
    assert recorded.startswith("UnsupportedTaskFeatureError: "), recorded
    assert "sandbox.gpus" in recorded
    assert dry.exit_code == 1, dry.output
    assert _plan(dry)["errors"] == {"plain": recorded}


@pytest.fixture
def claude_login_only(monkeypatch):
    """A (made-up) Claude login and no API key: the native-login route."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake-not-a-token")
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


GATEWAY = {"ANTHROPIC_BASE_URL": "https://gw.example.invalid"}


def test_the_dry_run_checks_the_run_s_agent_env(
    tmp_path, bench, sandbox_reached, claude_login_only
):
    """R632 2: with --agent-env the run's route differs from the host's, and
    the run refused an offline task the dry run passed."""
    _task(tmp_path / "tasks", "offline", environment='network_mode = "no-network"\n')
    flags = (
        "--model",
        "claude-opus-5-5",
        "--agent-env",
        "ANTHROPIC_BASE_URL=https://gw.example.invalid",
    )
    dry, _run, recorded = _both(
        tmp_path, bench, "docker", *flags, agent="claude-agent-acp"
    )
    assert recorded.startswith("ValueError: Native Claude no-web transport"), recorded
    assert dry.exit_code == 1, dry.output
    assert _plan(dry)["errors"] == {"offline": recorded}


def test_the_dry_run_checks_each_matrix_entry_s_agent_env(
    tmp_path, bench, sandbox_reached, claude_login_only
):
    import yaml

    _task(tmp_path / "tasks", "offline", environment='network_mode = "no-network"\n')
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "cc": {
                        "agent": "claude-agent-acp",
                        "model": "claude-opus-5-5",
                        "agent_env": GATEWAY,
                    }
                }
            }
        )
    )
    dry, _run, recorded = _both(tmp_path, bench, "docker", "--matrix", str(matrix))
    assert recorded.startswith("ValueError: Native Claude no-web transport"), recorded
    assert dry.exit_code == 1, dry.output
    assert _plan(dry)["errors"] == {"cc/offline": recorded}


def _two_roles(
    root: Path,
    runner: str = "opencode",
    runner_model: str = "deepseek/deepseek-v4-flash",
) -> Path:
    """A task.md whose two scene roles pin OpenCode, with a solution env the
    oracle would need (R632's two-roles case)."""
    task = root / "two-roles"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text(
        "FROM ubuntu:24.04\nWORKDIR /app\n"
    )
    (task / "oracle").mkdir()
    (task / "oracle" / "solve.sh").write_text("#!/bin/sh\ntrue\n")
    (task / "verifier").mkdir()
    (task / "verifier" / "test.sh").write_text(
        "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"
    )
    (task / "task.md").write_text(
        "---\nschema_version: '1.3'\nagent:\n  timeout_sec: 300\n"
        "verifier:\n  timeout_sec: 60\n"
        'solution:\n  env:\n    SOL_TOKEN: "${R632_SOL}"\n'
        "agents:\n  roles:\n"
        "    writer:\n      agent: opencode\n      model: deepseek/deepseek-v4-flash\n"
        f"    runner:\n      agent: {runner}\n      model: {runner_model}\n"
        "scenes:\n  - name: write\n    turns:\n      - role: writer\n"
        "  - name: run\n    turns:\n      - role: runner\n"
        "---\n# two-roles\n\n## prompt\n\nCount the rows.\n\n"
        "## scene:write\n\nWrite the script.\n\n## scene:run\n\nRun it.\n"
    )
    return task


def test_the_dry_run_checks_the_role_agent_a_task_pins(
    tmp_path, bench, sandbox_reached, monkeypatch
):
    """R632 2: with -a oracle the run's primary agent is the task's role
    agent; checked as the oracle, the dry run refused for the solution env."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-not-a-key")
    monkeypatch.delenv("R632_SOL", raising=False)
    _two_roles(tmp_path / "tasks")
    dry, _run, recorded = _both(
        tmp_path, bench, "docker", "--allow-task-env", "R632_SOL"
    )
    assert recorded == REACHED, recorded
    assert dry.exit_code == 0, dry.output
    assert "errors" not in _plan(dry)


def test_a_role_agent_without_its_key_refuses_both(
    tmp_path, bench, sandbox_reached, monkeypatch
):
    """R632 4 (credentials for task.md role agents): with -a oracle the run's
    agent is the task's role agent, and every trial records its missing key."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("R632_SOL", "made-up")
    _two_roles(tmp_path / "tasks")
    dry, _run, recorded = _both(
        tmp_path, bench, "docker", "--allow-task-env", "R632_SOL"
    )
    assert recorded.startswith("MissingCredentialError: DEEPSEEK_API_KEY"), recorded
    assert dry.exit_code == 1, dry.output
    assert _plan(dry)["errors"] == {"two-roles": recorded}


def test_a_repository_target_is_listed_as_not_checked(tmp_path, monkeypatch):
    """R632 3: a dry run fetches no repository or dataset, so it checks none
    of their tasks, and says so per target."""
    monkeypatch.chdir(tmp_path)
    dry = CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--source-repo",
            "example/none",
            "--source-ref",
            "abc123",
            "--source-path",
            "tasks",
            "--agent",
            "oracle",
            "--dry-run",
        ],
        prog_name="bench",
    )
    assert dry.exit_code == 0, dry.output
    (item,) = _plan(dry)["unchecked"]
    assert (item["target"], item["ref"], item["path"]) == (
        "example/none",
        "abc123",
        "tasks",
    )
    assert item["reason"] == "remote target not fetched in a dry run"
    assert "this dry run checked none of its tasks" in dry.output


# --- Lead 21:46Z: role keys, kept services, the Docker config copy ----------


def test_a_second_role_without_its_key_refuses_both(
    tmp_path, bench, sandbox_reached, monkeypatch
):
    """A scene role whose agent needs another key than the primary's: the
    run checked it only at the role's scene, after the sandbox."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-not-a-key")
    monkeypatch.setenv("R632_SOL", "made-up")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    _two_roles(
        tmp_path / "tasks", runner="claude-agent-acp", runner_model="claude-opus-5-5"
    )
    dry, _run, recorded = _both(
        tmp_path, bench, "docker", "--allow-task-env", "R632_SOL"
    )
    assert recorded.startswith("MissingCredentialError: "), recorded
    assert "claude-opus-5-5" in recorded
    assert dry.exit_code == 1, dry.output
    assert _plan(dry)["errors"] == {"two-roles": recorded}


def test_kept_services_on_daytona_refuse_both(
    tmp_path, bench, sandbox_reached, claude_login_only
):
    """R632 4: kept services on Daytona with a non-root agent (the default
    sandbox user) beside a root verifier, no Compose file."""
    from tests.test_dfx_scale import _kept_task

    task = _kept_task(tmp_path / "tasks", verifier_user="root")
    dry, _run, recorded = _both(
        tmp_path,
        bench,
        "daytona",
        "--model",
        "claude-opus-5-5",
        agent="claude-agent-acp",
    )
    assert recorded.startswith("KeptServiceFreezerUnavailable: "), recorded
    assert dry.exit_code == 1, dry.output
    assert _plan(dry)["errors"] == {task.name: recorded}


MADE_UP_LOGIN = "bWFkZS11cDpyNjMyLW5vdC1hLXJlYWwtbG9naW4="

FAKE_DOCKER = """#!{python}
import json, os, stat, sys
args = sys.argv[1:]
with open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "docker.log"), "a") as log:
    log.write(json.dumps({{"argv": args}}) + "\\n")
    if "--config" in args:
        folder = args[args.index("--config") + 1]
        config = os.path.join(folder, "config.json")
        log.write(json.dumps({{
            "folder": folder,
            "modes": [stat.S_IMODE(os.stat(folder).st_mode), stat.S_IMODE(os.stat(config).st_mode)],
            "holds": "registry.example.invalid" in open(config).read(),
        }}) + "\\n")
if args[-2:] == ["compose", "version"] or args[-1:] == ["version"]:
    print("Docker Compose version v2.40.3")
    sys.exit(0)
if "config" in args:
    sys.stderr.write('service "main" refers to undefined network r632: invalid compose project\\n')
    sys.exit(15)
sys.exit(1)
"""


@pytest.fixture
def docker_login(tmp_path, monkeypatch):
    """A (made-up) registry login in the operator's docker config, a docker
    CLI that records each call and the copy it was given, and a private temp
    folder where the copies are made."""
    home = tmp_path / "home"
    (home / ".docker").mkdir(parents=True)
    (home / ".docker" / "config.json").write_text(
        json.dumps({"auths": {"registry.example.invalid": {"auth": MADE_UP_LOGIN}}})
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("DOCKER_CONFIG", raising=False)
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    docker = bindir / "docker"
    docker.write_text(FAKE_DOCKER.format(python=sys.executable))
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    # Compose runs with PATH, HOME and the docker connection only: the CLI
    # logs beside itself.
    log = bindir / "docker.log"
    temp = tmp_path / "tmp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    return log, temp


def _composed(root: Path) -> Path:
    task = _task(root, "composed")
    (task / "environment" / "docker-compose.yaml").write_text(
        "services:\n  main:\n    build: .\n"
    )
    return task


@pytest.mark.parametrize("outcome", ["compose error", "exception", "clean"])
def test_the_docker_config_copy_is_private_and_gone_after_a_dry_run(
    tmp_path, bench, docker_login, monkeypatch, caplog, outcome
):
    """Lead 21:46Z (5): the copy of the registry logins a Compose check makes
    is 0600 in a 0700 folder, never on argv or in the log, and gone after
    the dry run, whether `docker compose config` fails, the check raises, or
    the dry run passes."""
    import logging

    from benchflow.task import compose_env

    log, temp = docker_login
    seen: list[dict] = []
    if outcome != "compose error":

        def freeze(*args, docker_args=(), **kwargs):
            folder = docker_args[docker_args.index("--config") + 1]
            config = Path(folder) / "config.json"
            seen.append(
                {
                    "argv": list(docker_args),
                    "folder": folder,
                    "modes": [
                        stat.S_IMODE(os.stat(folder).st_mode),
                        stat.S_IMODE(config.stat().st_mode),
                    ],
                    "holds": "registry.example.invalid" in config.read_text(),
                }
            )
            if outcome == "exception":
                raise RuntimeError("R632 compose check stand-in failed")
            return types.SimpleNamespace(refused=[], defaulted=[], project=None)

        monkeypatch.setattr(compose_env, "freeze_compose", freeze)
    caplog.set_level(logging.DEBUG)
    _composed(tmp_path / "tasks")
    dry = bench("docker", "--dry-run", jobs="dry")
    if log.exists():
        seen += [json.loads(line) for line in log.read_text().splitlines()]
    copies = [entry for entry in seen if "folder" in entry]
    assert copies, (outcome, seen, dry.output)
    for entry in copies:
        assert entry["modes"] == [0o700, 0o600], entry
        assert entry["holds"], entry
        assert not Path(entry["folder"]).exists(), entry["folder"]
    assert list(temp.glob("benchflow-docker-config-*")) == []
    for entry in seen:
        assert MADE_UP_LOGIN not in json.dumps(entry.get("argv", []))
    assert MADE_UP_LOGIN not in dry.output
    assert MADE_UP_LOGIN not in caplog.text
    assert dry.exit_code == (0 if outcome == "clean" else 1), dry.output


# --- R632 round 2: matrix mistakes, linked files, a matrix entry's job --------


def _flat(text: str) -> str:
    return " ".join(text.split())


BAD_MATRICES = {
    "not a model": ("models:\n  a: none\n", "is not a model name"),
    "no models mapping": ("agents: [oracle]\n", "--matrix must be a YAML mapping"),
    "agent_env not a mapping": (
        "models:\n  a:\n    agent: oracle\n    agent_env: [1, 2]\n",
        "matrix agent_env must be a mapping",
    ),
}


@pytest.mark.parametrize("shape", sorted(BAD_MATRICES))
def test_a_mistake_in_a_matrix_file_is_refused_as_the_run_refuses_it(
    tmp_path, bench, sandbox_reached, shape
):
    """R632 round 2 (1): the dry run built the matrix's entries before its
    guarded read and crashed as "a bug in BenchFlow"."""
    text, words = BAD_MATRICES[shape]
    _task(tmp_path / "tasks", "plain")
    matrix = tmp_path / "bad.yaml"
    matrix.write_text(text)
    dry = bench("docker", "--matrix", str(matrix), "--dry-run", jobs="dry")
    run = bench("docker", "--matrix", str(matrix), jobs="run")
    for result in (dry, run):
        assert result.exit_code == 1, result.output
        assert isinstance(result.exception, SystemExit), result.exception
        assert words in result.output, result.output
        assert "fix the --matrix file" in result.output


def test_a_task_linking_outside_itself_is_refused_by_both(
    tmp_path, bench, sandbox_reached
):
    """R632 round 2 (2): the tasks-root rule skips a task with a link out of
    its uploaded folders, and the run refuses the job it leaves empty."""
    task = _task(tmp_path / "tasks", "linked")
    outside = tmp_path / "outside.txt"
    outside.write_text("not the task's\n")
    (task / "environment" / "outside.txt").symlink_to(outside)
    dry = bench("docker", "--dry-run", jobs="dry")
    run = bench("docker", jobs="run")
    assert run.exit_code == 1, run.output
    assert dry.exit_code == 1, dry.output
    errors = _plan(dry)["errors"]
    assert list(errors) == ["(job)"], errors
    kind, message = errors["(job)"].split(": ", 1)
    assert kind == "EmptyTaskSelectionError"
    assert message.startswith("No tasks selected"), message
    assert _flat(message)[:100] in _flat(run.output), run.output


def test_a_matrix_entry_the_run_refuses_whole_gets_the_job_hint(
    tmp_path, bench, sandbox_reached, monkeypatch
):
    """R632 round 2 (nit): an entry no route serves is refused before its job
    exists, by the dry run as by the run, not as an error each trial records
    (codex-acp on DeepSeek, which serves no Responses API; the effort PRs let
    Claude Code through on every provider's route)."""
    import yaml

    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-not-a-key")
    _task(tmp_path / "tasks", "plain")
    matrix = tmp_path / "matrix.yaml"
    entry = {"agent": "codex-acp", "model": "deepseek/deepseek-v4-flash"}
    matrix.write_text(yaml.safe_dump({"models": {"ds": entry}}))
    dry = bench("docker", "--matrix", str(matrix), "--dry-run", jobs="dry")
    run = bench("docker", "--matrix", str(matrix), jobs="run")

    def refusal(result) -> list[str]:
        return [
            line
            for line in result.output.splitlines()
            if line.strip() and not line.startswith("note: `bench eval run`")
        ]

    assert run.exit_code == 1, run.output
    assert dry.exit_code == run.exit_code, dry.output
    assert refusal(dry) == refusal(run), (dry.output, run.output)
    assert "No job was created." in run.output
    assert "every trial of this task" not in dry.output


def test_a_matrix_entry_s_effort_and_harness_reach_the_dry_run_gate(
    tmp_path, bench, monkeypatch, claude_login_only
):
    """The effort / seam: the dry run's gate checks the config the run
    builds for a matrix entry with an effort of its own, not the run's."""
    import yaml

    import benchflow.rollout as rollout_mod

    seen: dict[str, list] = {"dry": [], "run": []}
    phase = ["dry"]

    def recording_gate(task, cfg, inputs, **kwargs):
        seen[phase[0]].append(
            (
                cfg.primary_agent,
                cfg.primary_model,
                cfg.primary_reasoning_effort,
                cfg.harness,
            )
        )
        raise RuntimeError("PlanGateReached")

    monkeypatch.setattr(rollout_mod, "run_rollout_plan_gate", recording_gate)
    _task(tmp_path / "tasks", "plain")
    matrix = tmp_path / "matrix.yaml"
    entry = {"agent": "claude-agent-acp", "model": "claude-opus-5-5"}
    matrix.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "hi": {**entry, "reasoning_effort": "high"},
                    "nat": {**entry, "harness": "native"},
                }
            }
        )
    )
    bench("docker", "--matrix", str(matrix), "--dry-run", jobs="dry")
    phase[0] = "run"
    bench("docker", "--matrix", str(matrix), jobs="run")
    assert len(seen["dry"]) == 2 and set(seen["dry"]) == set(seen["run"]), seen
    assert "high" in {cell[2] for cell in seen["dry"]}, seen
    assert "native" in {cell[3] for cell in seen["dry"]}, seen


def test_a_job_whose_tasks_cannot_use_the_separate_verifier_is_refused_whole(
    tmp_path, bench, sandbox_reached
):
    """R632 L1: with --integrity strict every verifier moves to the separate
    verifier sandbox; where the tasks cannot use it the run refuses the whole
    job in one block (_check_verifier_sandbox_plan), and so does its dry run."""
    for name in ("one", "two"):
        _task(tmp_path / "tasks", name)
    flags = ("--integrity", "strict")
    dry = bench("apple-container", *flags, "--dry-run", jobs="dry")
    run = bench("apple-container", *flags, jobs="run")
    assert run.exit_code == 1, run.output
    assert dry.exit_code == 1, dry.output
    errors = _plan(dry)["errors"]
    assert list(errors) == ["(job)"], errors
    message = errors["(job)"].split(": ", 1)[1]
    assert _flat(message)[:120] in _flat(run.output), run.output


def test_an_include_that_names_no_task_exits_2_in_both(
    tmp_path, bench, sandbox_reached
):
    """R632 L2: the run exits 2 with its `bench check` hint; the dry run
    exited 1 with the job hint."""
    _task(tmp_path / "tasks", "plain")
    dry = bench("docker", "--include", "no-such-task", "--dry-run", jobs="dry")
    run = bench("docker", "--include", "no-such-task", jobs="run")
    assert run.exit_code == 2, run.output
    assert dry.exit_code == 2, dry.output
    hints = [
        line for line in run.output.splitlines() if line.strip().startswith("next:")
    ]
    assert hints, run.output
    for hint in hints:
        assert _flat(hint) in _flat(dry.output), dry.output


def test_an_unknown_agent_in_a_matrix_entry_is_refused_by_both(
    tmp_path, bench, sandbox_reached, monkeypatch
):
    """R632 N3: the run and its dry run both crashed with a KeyError on an
    unknown entry agent; both refuse it before launch now."""
    import yaml

    from benchflow.agents import remote_manifests

    # The pinned agents source is not fetched here.
    monkeypatch.setattr(remote_manifests, "autoload_remote_manifest_agents", lambda: 0)
    _task(tmp_path / "tasks", "plain")
    matrix = tmp_path / "matrix.yaml"
    entry = {"agent": "no-such-agent", "model": "claude-opus-5-5"}
    matrix.write_text(yaml.safe_dump({"models": {"x": entry}}))
    dry = bench("docker", "--matrix", str(matrix), "--dry-run", jobs="dry")
    run = bench("docker", "--matrix", str(matrix), jobs="run")
    for result in (dry, run):
        assert result.exit_code == 2, result.output
        assert isinstance(result.exception, SystemExit), result.exception
        words = "matrix model 'x': Unknown agent: 'no-such-agent'"
        assert words in _flat(result.output), result.output
