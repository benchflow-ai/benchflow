"""The operator's credentials a trial puts in its sandbox.

``benchflow.agents.operator_credentials.in_sandbox`` decides what counts, in
one place: an API key the proxy in the sandbox holds, a subscription login or
API key the agent's own route stages, and a cloud credential file. The proxy
on the host (Docker, ``public``) stages none. A cloud credential file (the
gcloud application default credentials) needs the operator's
``--allow-cloud-credential-file``: without it the run is refused before
anything starts, on each path that would stage one (the Gemini CLI's own
Vertex route, the proxy in the sandbox, a file passed in). The check that
main is quiet runs before any of them is staged.

Credentials here are placeholders.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ADC = '{"type": "authorized_user", "refresh_token": "placeholder-refresh-7f3a"}'
KEY = "placeholder-api-key"
TOKEN = "placeholder-oauth-token"
FLAG = "--allow-cloud-credential-file"
HOST_PROXY = "the proxy on the host (Docker, `public`)"


def oc():
    from benchflow.agents import operator_credentials

    return operator_credentials


# ---------------------------------------------------------- the classifier


def test_the_proxy_on_the_host_stages_nothing():
    """Guard: an API key through the proxy on the host is the host's alone."""
    found = oc().in_sandbox(
        "claude-agent-acp",
        "deepseek/deepseek-flash",
        {"DEEPSEEK_API_KEY": KEY},
        proxy_in_sandbox=False,
    )
    assert found == []


def test_the_proxy_in_the_sandbox_holds_the_api_key():
    found = oc().in_sandbox(
        "claude-agent-acp",
        "deepseek/deepseek-flash",
        {"DEEPSEEK_API_KEY": KEY},
        proxy_in_sandbox=True,
    )
    assert found == [
        {
            "kind": "api-key",
            "route": "proxy",
            "names": ["DEEPSEEK_API_KEY"],
            "staged": "the sandbox proxy's process (root)",
        }
    ]


def test_the_agents_own_login_is_in_its_environment():
    """Claude Code on a Claude model with its OAuth token calls Anthropic
    itself: the token is in the agent's environment, in the sandbox, whatever
    the backend."""
    found = oc().in_sandbox(
        "claude-agent-acp",
        "claude-sonnet-4-5",
        {"CLAUDE_CODE_OAUTH_TOKEN": TOKEN},
        proxy_in_sandbox=False,
    )
    assert [(e["kind"], e["route"], e["names"], e["staged"]) for e in found] == [
        (
            "subscription-login",
            "native",
            ["CLAUDE_CODE_OAUTH_TOKEN"],
            "the agent's environment",
        )
    ]


def test_a_direct_endpoints_token_is_an_api_key():
    """Claude Code with an explicit token for another endpoint (the direct
    route) holds that token in its environment."""
    found = oc().in_sandbox(
        "claude-agent-acp",
        "deepseek/deepseek-flash",
        {
            "ANTHROPIC_AUTH_TOKEN": KEY,
            "ANTHROPIC_BASE_URL": "https://api.example.test/anthropic",
        },
        proxy_in_sandbox=False,
    )
    assert [(e["kind"], e["route"], e["names"]) for e in found] == [
        ("api-key", "direct", ["ANTHROPIC_AUTH_TOKEN"])
    ]


def test_claude_codes_api_key_goes_through_the_proxy():
    """Guard: with ANTHROPIC_API_KEY, Claude Code is proxied, so on the host
    nothing of the operator's is in the sandbox."""
    env = {"ANTHROPIC_API_KEY": KEY}
    assert (
        oc().in_sandbox(
            "claude-agent-acp", "claude-sonnet-4-5", env, proxy_in_sandbox=False
        )
        == []
    )
    [found] = oc().in_sandbox(
        "claude-agent-acp", "claude-sonnet-4-5", env, proxy_in_sandbox=True
    )
    assert (found["kind"], found["route"], found["names"]) == (
        "api-key",
        "proxy",
        ["ANTHROPIC_API_KEY"],
    )


def test_the_hosts_login_file_is_a_subscription_login():
    found = oc().in_sandbox(
        "claude-agent-acp",
        "claude-sonnet-4-5",
        {"_BENCHFLOW_SUBSCRIPTION_AUTH": "1"},
        proxy_in_sandbox=False,
    )
    assert [(e["kind"], e["staged"]) for e in found] == [
        ("subscription-login", "{home}/.claude/.credentials.json")
    ]


def test_the_gemini_clis_vertex_route_stages_the_cloud_file():
    found = oc().in_sandbox(
        "gemini",
        "google-vertex/gemini-2.5-pro",
        {"GOOGLE_APPLICATION_CREDENTIALS_JSON": ADC, "GOOGLE_CLOUD_PROJECT": "p"},
        proxy_in_sandbox=False,
    )
    assert oc().cloud_credential_file(found) == {
        "kind": "cloud-credential-file",
        "route": "native",
        "names": ["GOOGLE_APPLICATION_CREDENTIALS_JSON"],
        "staged": "{home}/.config/gcloud/application_default_credentials.json",
    }


@pytest.mark.parametrize(
    ("model", "staged"),
    [
        (
            "anthropic-vertex/claude-sonnet-4-5",
            "the sandbox proxy's runtime folder (root)",
        ),
        ("deepseek/deepseek-flash", "the sandbox proxy's process (root)"),
    ],
    ids=["vertex-model", "passed-in"],
)
def test_the_proxy_in_the_sandbox_holds_a_cloud_file(model, staged):
    env = {"GOOGLE_APPLICATION_CREDENTIALS_JSON": ADC, "GOOGLE_CLOUD_PROJECT": "p"}
    if model.startswith("deepseek"):
        env["DEEPSEEK_API_KEY"] = KEY
    found = oc().in_sandbox("claude-agent-acp", model, env, proxy_in_sandbox=True)
    assert oc().cloud_credential_file(found)["staged"] == staged
    assert oc().in_sandbox("claude-agent-acp", model, env, proxy_in_sandbox=False) == []


def test_a_scripted_agent_stages_nothing():
    assert (
        oc().in_sandbox(
            "oracle", None, {"ANTHROPIC_API_KEY": KEY}, proxy_in_sandbox=True
        )
        == []
    )


def test_no_entry_holds_a_value():
    env = {
        "GOOGLE_APPLICATION_CREDENTIALS_JSON": ADC,
        "GOOGLE_CLOUD_PROJECT": "p",
        "DEEPSEEK_API_KEY": KEY,
        "CLAUDE_CODE_OAUTH_TOKEN": TOKEN,
    }
    text = json.dumps(
        [
            oc().in_sandbox(a, m, env, proxy_in_sandbox=p)
            for a, m in [
                ("claude-agent-acp", "deepseek/deepseek-flash"),
                ("claude-agent-acp", "claude-sonnet-4-5"),
                ("gemini", "google-vertex/gemini-2.5-pro"),
            ]
            for p in (False, True)
        ]
    )
    for value in (ADC, KEY, TOKEN, "placeholder"):
        assert value not in text


def test_one_switch_decides_the_proxy_in_the_sandbox(monkeypatch):
    """One switch decides: with it off, an API key the
    proxy in the sandbox holds no longer counts (a cloud file still does)."""
    monkeypatch.setattr(oc(), "IN_SANDBOX_PROXY_COUNTS", False)
    env = {"DEEPSEEK_API_KEY": KEY}
    assert (
        oc().in_sandbox(
            "claude-agent-acp", "deepseek/deepseek-flash", env, proxy_in_sandbox=True
        )
        == []
    )


# --------------------------------------------------- the cloud file's flag


@pytest.fixture
def adc_home(tmp_path, monkeypatch):
    """A HOME whose gcloud ADC is a placeholder (resolution reads it for a
    Vertex model), and the project a Vertex model needs."""
    home = tmp_path / "home"
    adc = home / ".config" / "gcloud" / "application_default_credentials.json"
    adc.parent.mkdir(parents=True)
    adc.write_text(ADC)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "placeholder-project")
    for name in (
        "GOOGLE_APPLICATION_CREDENTIALS_JSON",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ):
        monkeypatch.delenv(name, raising=False)
    return home


def _task(root: Path, name: str = "vtx", *, allow_internet: bool = True) -> Path:
    task = root / name
    (task / "tests").mkdir(parents=True)
    (task / "solution").mkdir()
    (task / "task.toml").write_text(
        'version = "1.0"\n\n[agent]\ntimeout_sec = 60.0\n\n[verifier]\n'
        'timeout_sec = 60.0\n\n[environment]\ndocker_image = "python:3.12-slim"\n'
        + ("" if allow_internet else "allow_internet = false\n")
    )
    (task / "instruction.md").write_text("Do nothing.\n")
    for path in (task / "tests" / "test.sh", task / "solution" / "solve.sh"):
        path.write_text("#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n")
        path.chmod(0o755)
    return task


#: An ACP agent with no registry entry: its caller's slots are its own.
RAW_COMMAND = "/opt/my-agent --acp"


def _says_why(text: str) -> bool:
    return FLAG in text and HOST_PROXY in text and "cloud credential file" in text


# (agent, model, sandbox, allow_internet, extra env): each stages the file.
STAGED = {
    "gemini-vertex-route": (
        "gemini",
        "google-vertex/gemini-2.5-pro",
        "docker",
        True,
        {},
    ),
    "sandbox-proxy-daytona": (
        "claude-agent-acp",
        "anthropic-vertex/claude-sonnet-4-5",
        "daytona",
        True,
        {},
    ),
    "sandbox-proxy-docker-no-internet": (
        "claude-agent-acp",
        "anthropic-vertex/claude-sonnet-4-5",
        "docker",
        False,
        {},
    ),
    "passed-in-sandbox-proxy": (
        "claude-agent-acp",
        "deepseek/deepseek-flash",
        "daytona",
        True,
        {"GOOGLE_APPLICATION_CREDENTIALS_JSON": ADC, "DEEPSEEK_API_KEY": KEY},
    ),
    "raw-command-env": (
        RAW_COMMAND,
        None,
        "docker",
        True,
        {"GOOGLE_APPLICATION_CREDENTIALS_JSON": ADC},
    ),
}


def _cfg(tmp_path, case, **extra):
    from benchflow.rollout import RolloutConfig

    agent, model, sandbox, internet, env = STAGED[case]
    return RolloutConfig(
        task_path=_task(tmp_path, allow_internet=internet),
        agent=agent,
        model=model,
        environment=sandbox,
        agent_env=dict(env),
        **extra,
    )


@pytest.mark.parametrize("case", sorted(STAGED))
def test_each_path_is_refused_without_the_flag(tmp_path, adc_home, case):
    """``bench run --dry-run`` and ``Rollout.setup`` run this gate before any
    sandbox: refused, naming the flag and the proxy on the host."""
    from benchflow.contracts import default_rollout_planes
    from benchflow.rollout import dry_run_plan_gate

    with pytest.raises(Exception) as caught:
        dry_run_plan_gate(_cfg(tmp_path, case), default_rollout_planes())
    assert _says_why(str(caught.value)), caught.value
    assert ADC not in str(caught.value) and KEY not in str(caught.value)


@pytest.mark.parametrize("case", sorted(STAGED))
def test_the_flag_lets_each_path_through(tmp_path, adc_home, case):
    from benchflow.contracts import default_rollout_planes
    from benchflow.rollout import dry_run_plan_gate

    cfg = _cfg(tmp_path, case, allow_cloud_credential_file=True)
    try:
        dry_run_plan_gate(cfg, default_rollout_planes())
    except Exception as exc:  # another gate may refuse; never this one
        assert "cloud credential file" not in str(exc), exc


def test_the_proxy_on_the_host_needs_no_flag(tmp_path, adc_home):
    """Guard: Docker, ``public``, a Vertex model through the proxy on the
    host: the ADC stays in the host proxy's folder."""
    from benchflow.contracts import default_rollout_planes
    from benchflow.rollout import RolloutConfig, dry_run_plan_gate

    cfg = RolloutConfig(
        task_path=_task(tmp_path),
        agent="claude-agent-acp",
        model="anthropic-vertex/claude-sonnet-4-5",
        environment="docker",
    )
    try:
        dry_run_plan_gate(cfg, default_rollout_planes())
    except Exception as exc:
        assert "cloud credential file" not in str(exc), exc


@pytest.mark.parametrize(
    ("sandbox", "internet"),
    [("daytona", True), ("docker", False)],
    ids=["daytona", "docker-no-internet"],
)
def test_validate_task_plan_refuses_it(tmp_path, adc_home, sandbox, internet):
    """``bench tasks check`` and the SDK's Runtime call this one, with no
    connection policy of their own: the task's own network puts the proxy in
    the sandbox on Docker."""
    from benchflow.task.plan_validation import validate_task_plan

    task_dir = _task(tmp_path, allow_internet=internet)
    with pytest.raises(Exception) as caught:
        validate_task_plan(
            task_dir,
            sandbox=sandbox,
            agent="claude-agent-acp",
            model="anthropic-vertex/claude-sonnet-4-5",
        )
    assert _says_why(str(caught.value)), caught.value
    validate_task_plan(
        task_dir,
        sandbox=sandbox,
        agent="claude-agent-acp",
        model="anthropic-vertex/claude-sonnet-4-5",
        allow_cloud_credential_file=True,
    )


class _Started(RuntimeError):
    pass


def _job(tmp_path: Path, **config):
    from benchflow.evaluation import Evaluation, EvaluationConfig

    root = tmp_path / "tasks"
    _task(root, "vtx")
    base = {
        "agent": "gemini",
        "model": "google-vertex/gemini-2.5-pro",
        "environment": "docker",
        "concurrency": 1,
    }
    base.update(config)
    return Evaluation(
        tasks_dir=str(root),
        jobs_dir=str(tmp_path / "jobs"),
        config=EvaluationConfig(**base),
        preflight=False,
    )


def test_a_job_is_refused_before_its_folder_exists(tmp_path, adc_home, monkeypatch):
    from benchflow.evaluation import Evaluation

    async def started(self):
        raise _Started("the job started")

    monkeypatch.setattr(Evaluation, "_run_unlocked", started)
    with pytest.raises(Exception) as caught:
        _job(tmp_path).run_sync()
    assert not isinstance(caught.value, _Started), "the job started"
    assert _says_why(str(caught.value)) and "vtx" in str(caught.value), caught.value
    assert not (tmp_path / "jobs").exists() or not any((tmp_path / "jobs").iterdir())


def test_a_job_with_the_flag_starts(tmp_path, adc_home, monkeypatch):
    from benchflow.evaluation import Evaluation

    async def started(self):
        raise _Started("the job started")

    monkeypatch.setattr(Evaluation, "_run_unlocked", started)
    with pytest.raises(_Started):
        _job(tmp_path, allow_cloud_credential_file=True).run_sync()


def test_the_refusal_names_the_flag_and_the_proxy_on_the_host():
    entries = [
        {
            "kind": "cloud-credential-file",
            "route": "native",
            "names": ["GOOGLE_APPLICATION_CREDENTIALS_JSON"],
            "staged": "{home}/.config/gcloud/application_default_credentials.json",
        }
    ]
    with pytest.raises(oc().CloudCredentialFileRefused) as caught:
        oc().refuse_cloud_credential_file(entries, task_name="t", allowed=False)
    assert _says_why(str(caught.value))
    assert "proxy route" not in str(caught.value).replace(HOST_PROXY, "")
    oc().refuse_cloud_credential_file(entries, task_name="t", allowed=True)
    oc().refuse_cloud_credential_file([], task_name="t", allowed=False)


# ------------------------------------------------------ the flag's plumbing


def test_the_run_config_key_is_the_flag(tmp_path):
    from benchflow._utils.yaml_loader import rollout_config_from_dict

    cfg = rollout_config_from_dict(
        {
            "task_path": str(_task(tmp_path)),
            "agent": "oracle",
            "allow_cloud_credential_file": True,
        }
    )
    assert cfg.allow_cloud_credential_file is True


@pytest.mark.parametrize(
    ("value", "granted"),
    [
        (True, True),
        (False, False),
        (None, False),
        ("true", None),
        (1, None),
        ("false", None),
    ],
)
def test_a_workers_config_takes_only_a_literal_true(value, granted):
    """Only a boolean true grants; any other value is refused, naming the
    setting (``bool("false")`` is true)."""
    from benchflow.eval_worker import _evaluation_config

    raw = {"agent": "oracle", "environment": "docker"}
    if value is not None:
        raw["allow_cloud_credential_file"] = value
    if granted is None:
        with pytest.raises(ValueError, match="allow_cloud_credential_file must be"):
            _evaluation_config(raw)
        return
    assert _evaluation_config(raw).allow_cloud_credential_file is granted


def test_the_sdk_fields_default_off():
    from benchflow.evaluation import EvaluationConfig
    from benchflow.rollout import RolloutConfig
    from benchflow.runtime import RuntimeConfig

    assert EvaluationConfig().allow_cloud_credential_file is False
    assert (
        RolloutConfig(task_path=".", agent="oracle").allow_cloud_credential_file
        is False
    )
    assert RuntimeConfig().allow_cloud_credential_file is False


def _bench(*args: str, cwd: Path, home: Path) -> subprocess.CompletedProcess[str]:
    bench = Path(sys.executable).parent / "bench"
    if not bench.exists():
        pytest.skip("no bench script beside this Python")
    env = {k: v for k, v in os.environ.items() if k in ("USER", "LANG", "PATH")}
    env.update(
        HOME=str(home),
        GOOGLE_CLOUD_PROJECT="placeholder-project",
        BENCHFLOW_SKIP_PREFLIGHT="1",
        HARBOR_TELEMETRY="off",
        DO_NOT_TRACK="1",
        COLUMNS="200",
    )
    return subprocess.run(
        [str(bench), *args],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_the_cli_dry_run_refuses_and_the_flag_passes(tmp_path, adc_home):
    task_dir = _task(tmp_path)
    args = [
        "run",
        str(task_dir),
        "--agent",
        "gemini",
        "--model",
        "google-vertex/gemini-2.5-pro",
        "--sandbox",
        "docker",
        "--jobs-dir",
        str(tmp_path / "jobs"),
        "--dry-run",
    ]
    done = _bench(*args, cwd=tmp_path, home=adc_home)
    said = done.stdout + done.stderr
    assert done.returncode != 0 and _says_why(said), (done.returncode, said[-1500:])
    assert "placeholder-refresh" not in said
    done = _bench(*args, FLAG, cwd=tmp_path, home=adc_home)
    said = done.stdout + done.stderr
    assert "needs an operator flag" not in said, said[-1500:]
    # The dry run's run card names the grant.
    assert '"setting": "allow_cloud_credential_file"' in said, said[-1500:]


# ----------------------------------------------- no value on a command line


def encodings(value: str) -> list[str]:
    """How a value could ride a command: as is, as the octal escapes the
    Daytona env wrapper writes, as hex, and as base64 at each alignment."""
    import base64

    raw = value.encode()
    found = [
        value,
        "".join(f"\\{b:03o}" for b in raw),
        "".join(f"\\x{b:02x}" for b in raw),
        raw.hex(),
        raw.hex().upper(),
    ]
    for pad in range(3):
        text = base64.b64encode(b"\0" * pad + raw).decode()
        # Drop the characters the padding and the next bytes touch.
        found.append(text[(pad * 4 + 2) // 3 + 1 : -4])
    return [f for f in found if len(f) >= 8]


def leaks(texts: list[str], values: list[str]) -> list[tuple[str, str]]:
    """Each value found in a text in any form, base64 wrapped at 76
    columns (``base64`` the tool, MIME) included."""
    return [
        (value, text[:200])
        for value in values
        for form in encodings(value)
        for text in texts
        if form in text or form in text.replace("\n", "")
    ]


def test_the_scan_finds_each_form():
    """The scan itself: a value as is, octal-escaped, hex or base64 in a
    command is found."""
    import base64

    from benchflow.sandbox._base import wrap_command_with_env_file

    wrapped = wrap_command_with_env_file({"K": KEY}, "true")
    assert KEY not in wrapped and leaks([wrapped], [KEY])
    wrapped76 = base64.encodebytes(("x" * 70 + KEY).encode()).decode()
    assert "\n" in wrapped76.strip()
    for text in (
        f"x {KEY} y",
        KEY.encode().hex(),
        KEY.encode().hex().upper(),
        "".join(f"\\x{b:02x}" for b in KEY.encode()),
        base64.b64encode(f"K={KEY}\n".encode()).decode(),
        base64.b64encode(f"xy{KEY}".encode()).decode(),
        wrapped76,
    ):
        assert leaks([text], [KEY]), text
    assert leaks(["printf ok"], [KEY]) == []


async def test_the_credential_writers_put_no_value_on_a_command_line(
    tmp_path, monkeypatch
):
    """Every file writer BenchFlow stages a credential with (a provider's
    credential file, the Gemini CLI's ADC, the host's login file, Codex's
    auth.json) on a recording sandbox: each value goes up as a file, and no
    exec's command holds it in any form. The writers are the same on every
    backend."""
    from types import SimpleNamespace

    from benchflow.agents import credentials as cr
    from benchflow.agents.registry import AGENTS

    monkeypatch.setattr(cr, "admit", lambda *a, **k: None)
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".claude" / ".credentials.json").write_text(
        json.dumps({"claudeAiOauth": {"accessToken": TOKEN}})
    )
    monkeypatch.setenv("HOME", str(home))
    commands: list[str] = []
    uploads: list[str] = []

    class Recorder:
        async def exec(self, command, **_kw):
            commands.append(str(command))
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        async def upload_file(self, source, target, **_kw):
            uploads.append(Path(source).read_text())

    env = Recorder()
    await cr.write_credential_files(
        env,
        "gemini",
        {"GOOGLE_APPLICATION_CREDENTIALS_JSON": ADC, "GOOGLE_CLOUD_PROJECT": "p"},
        AGENTS["gemini"],
        "google-vertex/gemini-2.5-pro",
        "/home/agent",
    )
    await cr.upload_subscription_auth(env, "claude-agent-acp", "/home/agent")
    auth = json.dumps({"tokens": {"access_token": KEY, "refresh_token": KEY}})
    await cr.write_credential_files(
        env,
        "codex-acp",
        {"CODEX_AUTH_JSON": auth},
        AGENTS["codex-acp"],
        None,
        "/home/agent",
    )
    staged = "\n".join(uploads)
    for value in (ADC, TOKEN, KEY):
        assert value in staged, value
    assert (
        commands
        and leaks(commands, [ADC, TOKEN, KEY, "placeholder-refresh-7f3a"]) == []
    )


# ---------------------------------------- the warning and the trial's record

ADVICE = (
    "With tasks you don't trust, use API keys through the proxy on the host "
    "(Docker, `public`), not a subscription login; rotate a login that ran "
    "such tasks."
)
API_KEY_IN_PROXY = {
    "kind": "api-key",
    "route": "proxy",
    "names": ["DEEPSEEK_API_KEY"],
    "staged": "the sandbox proxy's process (root)",
}
CLOUD_NATIVE = {
    "kind": "cloud-credential-file",
    "route": "native",
    "names": ["GOOGLE_APPLICATION_CREDENTIALS_JSON"],
    "staged": "{home}/.config/gcloud/application_default_credentials.json",
}


def test_the_warning_names_each_credential_and_the_way_out():
    line = oc().warning("primes", [API_KEY_IN_PROXY], [])
    assert line.startswith("primes: operator credentials go into the sandbox")
    assert "api-key (DEEPSEEK_API_KEY) in the sandbox proxy's process (root)" in line
    assert ADVICE in line
    assert oc().warning("primes", [], []) is None


def test_a_granted_cloud_file_is_said_loudly():
    granted = oc().granted([CLOUD_NATIVE], allow_cloud_credential_file=True)
    assert granted == [
        {
            "setting": "allow_cloud_credential_file",
            "run": FLAG,
            "lifts": "a cloud credential file (the gcloud application default "
            "credentials) is staged in the sandbox, where root can read it",
        }
    ]
    line = oc().warning("vtx", [CLOUD_NATIVE], granted)
    assert f"the run allows it with {FLAG}" in line
    assert oc().granted([API_KEY_IN_PROXY], allow_cloud_credential_file=True) == []
    assert oc().granted([CLOUD_NATIVE], allow_cloud_credential_file=False) == []


def _staging_rollout(
    tmp_path,
    monkeypatch,
    *,
    environment,
    agent_env,
    flag=False,
    beside=False,
    sandbox_user=None,
):
    """A Rollout's state at a connection's start, with the quiet-main check
    recorded."""
    from types import SimpleNamespace

    from benchflow.rollout import Rollout
    from benchflow.sandbox import lockdown

    seen: list[bool] = []

    async def check(env, agent_env, *, held=None, credentials=None):
        seen.append((held, bool(credentials)))

    monkeypatch.setattr(lockdown, "refuse_task_processes_before_launch", check)
    rollout = Rollout.__new__(Rollout)
    rollout._env = SimpleNamespace()
    rollout._task = SimpleNamespace(name="primes")
    rollout._config = SimpleNamespace(
        environment=environment,
        allow_cloud_credential_file=flag,
        allow_task_processes_with_credentials=beside,
        task_path=tmp_path,
        sandbox_user=sandbox_user,
    )
    return rollout, seen


async def test_a_connection_warns_once_and_records(tmp_path, monkeypatch, caplog):
    import logging

    rollout, seen = _staging_rollout(
        tmp_path, monkeypatch, environment="daytona", agent_env=None
    )
    env = {"DEEPSEEK_API_KEY": KEY}
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            await rollout._before_staging(
                "claude-agent-acp",
                "deepseek/deepseek-flash",
                env,
                held=False,
                force_sandbox_local=False,
            )
    said = [
        r.getMessage()
        for r in caplog.records
        if "operator credentials" in r.getMessage()
    ]
    assert len(said) == 1 and KEY not in said[0], said
    # Daytona does not list main yet: the record and the warning say so.
    assert rollout._operator_credentials == {
        "in_sandbox": [API_KEY_IN_PROXY],
        "granted": [],
        "quiet_main": "not checked on daytona",
    }
    assert "Main's processes are not checked on daytona." in said[0]
    # An API key the proxy in the sandbox holds: main must be quiet.
    assert seen == [(False, True), (False, True)]


async def test_the_proxy_on_the_host_warns_nothing(tmp_path, monkeypatch, caplog):
    """Guard: Docker, `public`, an API key: no warning, no record."""
    import logging

    rollout, _ = _staging_rollout(
        tmp_path, monkeypatch, environment="docker", agent_env=None
    )
    with caplog.at_level(logging.WARNING):
        await rollout._before_staging(
            "claude-agent-acp",
            "deepseek/deepseek-flash",
            {"DEEPSEEK_API_KEY": KEY},
            held=False,
            force_sandbox_local=False,
        )
    assert not [r for r in caplog.records if "operator credentials" in r.getMessage()]
    assert getattr(rollout, "_operator_credentials", None) is None


async def test_a_roles_cloud_file_is_refused_before_it_is_staged(tmp_path, monkeypatch):
    """A scene role on a Vertex model is past the plan gate (which checks the
    primary agent): its connection refuses before anything of it is staged."""
    rollout, seen = _staging_rollout(
        tmp_path, monkeypatch, environment="docker", agent_env=None
    )
    with pytest.raises(oc().CloudCredentialFileRefused):
        await rollout._before_staging(
            "gemini",
            "google-vertex/gemini-2.5-pro",
            {"GOOGLE_APPLICATION_CREDENTIALS_JSON": ADC, "GOOGLE_CLOUD_PROJECT": "p"},
            held=False,
            force_sandbox_local=False,
        )
    assert seen == []


def test_result_json_has_the_record_beside_sandbox_account(tmp_path):
    from benchflow.rollout._results import _build_rollout_result

    record = {"in_sandbox": [API_KEY_IN_PROXY], "granted": []}
    for given, expected in ((record, record), (None, "absent")):
        rollout_dir = tmp_path / ("with" if given else "without")
        rollout_dir.mkdir()
        _build_rollout_result(
            rollout_dir,
            task_name="primes",
            rollout_name="primes__abc",
            agent="claude-agent-acp",
            agent_name="claude-agent-acp",
            model="deepseek/deepseek-flash",
            n_tool_calls=0,
            prompts=["solve"],
            error=None,
            verifier_error=None,
            trajectory=[],
            partial_trajectory=False,
            rewards=None,
            started_at=__import__("datetime").datetime.now(),
            timing={"agent": 0.0},
            send_spans=False,
            operator_credentials=given,
        )
        data = json.loads((rollout_dir / "result.json").read_text())
        assert data.get("operator_credentials", "absent") == expected


def test_the_run_card_names_each_tasks_credentials(tmp_path):
    from benchflow.agents import operator_credentials as ocs

    root = tmp_path / "tasks"
    open_task = _task(root, "open")
    closed = _task(root, "closed", allow_internet=False)
    env = {"DEEPSEEK_API_KEY": KEY}
    block = ocs.run_card_block(
        [open_task, closed],
        agent="claude-agent-acp",
        model="deepseek/deepseek-flash",
        agent_env=env,
        environment="docker",
        sandbox_user="agent",
        allow_cloud_credential_file=False,
    )
    assert block == {"tasks": {"closed": [API_KEY_IN_PROXY]}, "granted": []}
    assert KEY not in json.dumps(block)
    assert (
        ocs.run_card_block(
            [open_task],
            agent="claude-agent-acp",
            model="deepseek/deepseek-flash",
            agent_env=env,
            environment="docker",
            sandbox_user="agent",
            allow_cloud_credential_file=False,
        )
        is None
    )
    [line] = ocs.run_start_lines(block)
    assert "closed" in line and ADVICE in line


def test_the_docs_give_the_advice_and_the_cloud_file_flag():
    page = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "tasks"
        / "network-and-secrets.md"
    )
    text = page.read_text()
    assert ADVICE in text
    assert FLAG in text and "proxy route" not in text


def _card_job(tmp_path, monkeypatch, environment):
    monkeypatch.setenv("DEEPSEEK_API_KEY", KEY)
    return _job(
        tmp_path,
        agent="claude-agent-acp",
        model="deepseek/deepseek-flash",
        environment=environment,
    )


@pytest.mark.parametrize("environment", ["daytona", "docker"])
def test_the_run_cards_block_and_its_start_line(
    tmp_path, monkeypatch, caplog, environment
):
    """build_run_card carries the block and the run says it at start; on the
    proxy on the host there is neither."""
    import logging

    from benchflow.run_card import build_run_card

    job = _card_job(tmp_path, monkeypatch, environment)
    task_dirs = job._get_task_dirs()
    card = build_run_card(job, task_dirs)
    block = card["operator_credentials"]
    job._run_card = card
    job._job_name = "card"
    (tmp_path / "jobs" / "card").mkdir(parents=True, exist_ok=True)
    with caplog.at_level(logging.WARNING):
        job._log_run_card(done=0)
    said = [
        r.getMessage()
        for r in caplog.records
        if "Operator credentials" in r.getMessage()
    ]
    if environment == "docker":
        assert block is None and said == []
        return
    # Resolution copies the key into Claude Code's own slots; the proxy in the
    # sandbox gets each.
    [entry] = block["tasks"]["vtx"]
    assert entry["kind"] == "api-key" and "DEEPSEEK_API_KEY" in entry["names"]
    assert len(said) == 1 and ADVICE in said[0] and KEY not in said[0]


@pytest.mark.parametrize(
    ("held", "agent_env", "checked"),
    [
        (True, {}, True),
        (False, {"BENCHFLOW_DISALLOW_WEB_TOOLS": "1"}, False),
        (None, {"BENCHFLOW_DISALLOW_WEB_TOOLS": "1"}, True),
        (None, {}, False),
    ],
    ids=["held-given", "not-held-given", "env-held", "env-open"],
)
async def test_the_connections_policy_decides_the_check(
    monkeypatch, held, agent_env, checked
):
    """Before the proxy's wiring the env has no policy markers: the
    connection passes its policy (``held``); without it the env's markers
    decide, as at the ACP launch."""
    from types import SimpleNamespace

    from benchflow.sandbox import lockdown

    seen: list[object] = []

    async def scan(env, **_kwargs):
        seen.append(env)

    monkeypatch.setattr(lockdown, "refuse_task_processes", scan)
    main = SimpleNamespace()
    await lockdown.refuse_task_processes_before_launch(main, agent_env, held=held)
    assert seen == ([main] if checked else [])


# ------------------ main is quiet when an operator credential is in the sandbox

TP_FLAG = "--allow-task-processes-with-credentials"


async def test_a_task_process_beside_a_credential_is_refused_naming_the_way_out(
    monkeypatch,
):
    """Under any policy, `public` included: the refusal names each process,
    the flag and the proxy on the host."""
    from benchflow.sandbox import lockdown
    from tests.test_en698_policy_first_instruction import SCAN, _Main

    async def read(env, command, **kwargs):
        return await env.exec(command, **kwargs)

    monkeypatch.setattr(lockdown, "_read_exec", read)
    main = _Main(SCAN)
    with pytest.raises(RuntimeError) as refused:
        await lockdown.refuse_task_processes_before_launch(
            main, {}, held=False, credentials=oc().QUIET_MAIN_REFUSAL
        )
    text = str(refused.value)
    assert "with an operator credential in the sandbox" in text
    assert TP_FLAG in text and HOST_PROXY in text
    assert "pid 50 (uid 0): python" in text
    assert "proxy route" not in text.replace(HOST_PROXY, "")


async def test_an_unreadable_main_beside_a_credential_names_the_way_out(monkeypatch):
    from benchflow.sandbox import lockdown
    from tests.test_en698_policy_first_instruction import _Main

    async def read(env, command, **kwargs):
        return await env.exec(command, **kwargs)

    monkeypatch.setattr(lockdown, "_read_exec", read)
    with pytest.raises(RuntimeError) as refused:
        await lockdown.refuse_task_processes(
            _Main("BF task-procs status=2\n"), credentials=oc().QUIET_MAIN_REFUSAL
        )
    assert TP_FLAG in str(refused.value) and HOST_PROXY in str(refused.value)


@pytest.mark.parametrize(
    "hold",
    ["_benchflow_start_hold", "_benchflow_offline_rules", "_benchflow_oracle_egress"],
)
async def test_a_held_main_is_still_listed_beside_a_credential(monkeypatch, hold):
    """Every process of main under the policy answers the policy, not a
    credential: a task process beside one is still refused, and without a
    credential the held main is not listed. Mutant: the early return kept
    for a credential."""
    from benchflow.sandbox import lockdown
    from tests.test_en698_policy_first_instruction import SCAN, _Main

    async def read(env, command, **kwargs):
        return await env.exec(command, **kwargs)

    monkeypatch.setattr(lockdown, "_read_exec", read)
    main = _Main(SCAN)
    setattr(main, hold, ("held",))
    with pytest.raises(RuntimeError) as refused:
        await lockdown.refuse_task_processes(main, credentials=oc().QUIET_MAIN_REFUSAL)
    assert "with an operator credential in the sandbox" in str(refused.value)
    quiet = _Main(SCAN)
    setattr(quiet, hold, ("held",))
    await lockdown.refuse_task_processes(quiet)
    assert quiet.commands == []


@pytest.mark.parametrize(
    "hold",
    ["_benchflow_start_hold", "_benchflow_offline_rules", "_benchflow_oracle_egress"],
)
@pytest.mark.parametrize("held", [False, True], ids=["open", "held"])
async def test_a_connection_with_a_credential_lists_a_held_main(
    monkeypatch, hold, held
):
    """Through the launch check: a connection that stages an operator
    credential lists main under an armed hold whether its own policy holds it
    or not, and the refusal gives the credential's reason. Mutant: a held
    connection's credential dropped."""
    from benchflow.sandbox import lockdown
    from tests.test_en698_policy_first_instruction import SCAN, _Main

    async def read(env, command, **kwargs):
        return await env.exec(command, **kwargs)

    monkeypatch.setattr(lockdown, "_read_exec", read)
    main = _Main(SCAN)
    setattr(main, hold, ("held",))
    with pytest.raises(RuntimeError) as refused:
        await lockdown.refuse_task_processes_before_launch(
            main, {}, held=held, credentials=oc().QUIET_MAIN_REFUSAL
        )
    assert "with an operator credential in the sandbox" in str(refused.value)


@pytest.mark.parametrize(
    ("held", "credentials", "checked"),
    [(False, "why", True), (False, None, False), (True, None, True)],
    ids=["credentials", "neither", "held"],
)
async def test_the_credential_trigger_checks_under_any_policy(
    monkeypatch, held, credentials, checked
):
    from types import SimpleNamespace

    from benchflow.sandbox import lockdown

    seen: list[object] = []

    async def scan(env, **kwargs):
        seen.append(kwargs.get("credentials"))

    monkeypatch.setattr(lockdown, "refuse_task_processes", scan)
    await lockdown.refuse_task_processes_before_launch(
        SimpleNamespace(), {}, held=held, credentials=credentials
    )
    assert seen == ([credentials] if checked else [])


async def test_the_flag_lifts_the_credential_trigger_and_is_recorded(
    tmp_path, monkeypatch
):
    """With --allow-task-processes-with-credentials the credential alone no
    longer asks for quiet main; the trial records the acceptance for its
    result.json and its provenance. A held policy still checks."""
    rollout, seen = _staging_rollout(
        tmp_path, monkeypatch, environment="daytona", agent_env=None, beside=True
    )
    for held in (False, True):
        await rollout._before_staging(
            "claude-agent-acp",
            "deepseek/deepseek-flash",
            {"DEEPSEEK_API_KEY": KEY},
            held=held,
            force_sandbox_local=False,
        )
    assert seen == [(False, False), (True, False)]
    record = {
        "setting": "allow_task_processes_with_credentials",
        "run": TP_FLAG,
        "lifts": "a task process may run in main beside the operator's "
        "credentials, where it can read them",
    }
    assert rollout._operator_credentials["granted"] == [record]
    assert rollout._env._benchflow_operator_credentials_lifted == [record]


async def test_the_proxy_on_the_host_asks_for_nothing(tmp_path, monkeypatch):
    """Guard: Docker, `public`, an API key: no credential trigger, no record."""
    rollout, seen = _staging_rollout(
        tmp_path, monkeypatch, environment="docker", agent_env=None
    )
    await rollout._before_staging(
        "claude-agent-acp",
        "deepseek/deepseek-flash",
        {"DEEPSEEK_API_KEY": KEY},
        held=False,
        force_sandbox_local=False,
    )
    assert seen == [(False, False)]
    assert getattr(rollout._env, "_benchflow_operator_credentials_lifted", None) is None


async def test_the_provenance_names_the_acceptance(monkeypatch):
    """The reward's provenance names it beside the verdict, which it does
    not change."""

    from benchflow.rollout import _setup

    text = inspect.getsource(_setup._record_reward_provenance)
    assert 'found["operator_credentials_lifted"]' in text
    assert '"_benchflow_operator_credentials_lifted"' in text


def test_the_flags_plumbing():
    from benchflow._utils.yaml_loader import rollout_config_from_dict
    from benchflow.eval_worker import _evaluation_config
    from benchflow.evaluation import EvaluationConfig
    from benchflow.rollout import RolloutConfig
    from benchflow.runtime import RuntimeConfig

    name = "allow_task_processes_with_credentials"
    assert getattr(EvaluationConfig(), name) is False
    assert getattr(RolloutConfig(task_path=".", agent="oracle"), name) is False
    assert getattr(RuntimeConfig(), name) is False
    cfg = rollout_config_from_dict({"task_path": ".", "agent": "oracle", name: True})
    assert getattr(cfg, name) is True
    raw = {"agent": "oracle", "environment": "docker", name: True}
    assert getattr(_evaluation_config(raw), name) is True
    for value in ("true", 1):
        with pytest.raises(ValueError, match=f"{name} must be"):
            _evaluation_config({**raw, name: value})


def test_the_docs_name_the_task_processes_flag():
    page = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "tasks"
        / "network-and-secrets.md"
    )
    assert TP_FLAG in page.read_text()


def test_the_proxys_own_key_is_not_the_operators():
    """A role's env can carry the proxy's run key from an earlier connection:
    it is BenchFlow's, never listed."""
    found = oc().in_sandbox(
        "claude-agent-acp",
        "deepseek/deepseek-flash",
        {"DEEPSEEK_API_KEY": KEY, "BENCHFLOW_LITELLM_MASTER_KEY": "sk-run"},
        proxy_in_sandbox=True,
    )
    assert [e["names"] for e in found] == [["DEEPSEEK_API_KEY"]]


def test_bench_eval_branch_takes_both_flags(tmp_path, monkeypatch):
    """A branch run stages credentials as a run does: `bench branch`
    takes both flags and its plan carries them to each rollout."""
    from typer.testing import CliRunner

    from benchflow import branch_run
    from benchflow.cli.main import app

    seen = {}

    async def run(plan, task):
        seen.update(
            cloud=plan.allow_cloud_credential_file,
            beside=plan.allow_task_processes_with_credentials,
        )
        return branch_run.BranchTrialOutcome(task=task.name)

    monkeypatch.setattr(branch_run, "run_branch_trial", run)
    monkeypatch.setattr(branch_run, "write_branch_job", lambda plan, outcomes: tmp_path)
    done = CliRunner().invoke(
        app,
        [
            "branch",
            "--tasks-dir",
            str(_task(tmp_path)),
            "--agent",
            "oracle",
            "--child",
            "label=a",
            "--child",
            "label=b",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            FLAG,
            TP_FLAG,
        ],
    )
    assert seen == {"cloud": True, "beside": True}, done.output[-2000:]


def test_a_branch_rollout_gets_both_flags():

    from benchflow import branch_run

    text = inspect.getsource(branch_run.run_branch_trial)
    assert "allow_cloud_credential_file=plan.allow_cloud_credential_file" in text
    assert (
        "allow_task_processes_with_credentials=plan.allow_task_processes_with_credentials"
        in text
    )


# ------------------------------------------------- what each route stages


def test_a_raw_commands_cloud_file_is_in_its_launch_env_and_named(adc_home):
    """A raw ACP command with no model: the slots its caller passes are its
    own, the gcloud credentials included, so its launch env holds them and
    the classifier names a cloud credential file in its environment."""
    from benchflow.agents.env import resolve_agent_env
    from benchflow.agents.launch_env import agent_launch_env

    env = resolve_agent_env(
        RAW_COMMAND, None, {"GOOGLE_APPLICATION_CREDENTIALS_JSON": ADC}
    )
    assert agent_launch_env(RAW_COMMAND, None, dict(env)).get(
        "GOOGLE_APPLICATION_CREDENTIALS_JSON"
    )
    assert oc().in_sandbox(RAW_COMMAND, None, env, proxy_in_sandbox=False) == [
        {
            "kind": "cloud-credential-file",
            "route": "direct",
            "names": ["GOOGLE_APPLICATION_CREDENTIALS_JSON"],
            "staged": "the agent's environment",
        }
    ]


@pytest.mark.parametrize(
    ("agent", "name", "staged"),
    [
        (
            "codex-acp",
            "CODEX_AUTH_JSON",
            "the agent's environment and /home/agent/.codex/auth.json",
        ),
        (
            "muse-code",
            "MUSE_AUTH_JSON",
            "the agent's environment and /home/agent/.config/muse/auth.json",
        ),
    ],
)
def test_a_chatgpt_or_muse_login_is_a_subscription_login_named_with_its_file(
    agent, name, staged
):
    """Both logins carry a refresh token and are also written as files."""
    from benchflow.agents.registry import AGENTS

    if agent not in AGENTS:
        pytest.skip(f"{agent} is not registered here")
    found = oc().in_sandbox(
        agent,
        None,
        {name: '{"tokens": {}}'},
        proxy_in_sandbox=False,
        home="/home/agent",
    )
    assert {"kind": "subscription-login", "names": [name], "staged": staged} in [
        {k: e[k] for k in ("kind", "names", "staged")} for e in found
    ], found


async def test_a_connections_files_are_named_in_its_home(tmp_path, monkeypatch):
    """result.json names a staged file where it is: the connection's home,
    not a template."""
    rollout, _ = _staging_rollout(
        tmp_path,
        monkeypatch,
        environment="docker",
        agent_env=None,
        flag=True,
        sandbox_user="agent",
    )
    await rollout._before_staging(
        "gemini",
        "google-vertex/gemini-2.5-pro",
        {"GOOGLE_APPLICATION_CREDENTIALS_JSON": ADC, "GOOGLE_CLOUD_PROJECT": "p"},
        held=False,
        force_sandbox_local=False,
    )
    [entry] = [
        e
        for e in rollout._operator_credentials["in_sandbox"]
        if e["kind"] == "cloud-credential-file"
    ]
    assert entry["staged"] == (
        "/home/agent/.config/gcloud/application_default_credentials.json"
    )


async def test_a_refused_connection_records_what_it_would_have_staged(
    tmp_path, monkeypatch
):
    """Refused before anything was staged (main not quiet, a cloud file
    without the flag): the trial's record says ``would_stage``, not
    ``in_sandbox``."""
    from benchflow.sandbox import lockdown

    rollout, _ = _staging_rollout(
        tmp_path, monkeypatch, environment="daytona", agent_env=None
    )

    async def refuse(*_a, **_k):
        raise RuntimeError("sandbox startup refused: main must be quiet")

    monkeypatch.setattr(lockdown, "refuse_task_processes_before_launch", refuse)
    with pytest.raises(RuntimeError):
        await rollout._before_staging(
            "claude-agent-acp",
            "deepseek/deepseek-flash",
            {"DEEPSEEK_API_KEY": KEY},
            held=False,
            force_sandbox_local=False,
        )
    record = rollout._operator_credentials
    assert record["in_sandbox"] == [] and record["would_stage"] == [API_KEY_IN_PROXY]
    rollout, _ = _staging_rollout(
        tmp_path, monkeypatch, environment="docker", agent_env=None
    )
    with pytest.raises(oc().CloudCredentialFileRefused):
        await rollout._before_staging(
            "gemini",
            "google-vertex/gemini-2.5-pro",
            {"GOOGLE_APPLICATION_CREDENTIALS_JSON": ADC, "GOOGLE_CLOUD_PROJECT": "p"},
            held=False,
            force_sandbox_local=False,
        )
    assert rollout._operator_credentials["in_sandbox"] == []
    assert [e["kind"] for e in rollout._operator_credentials["would_stage"]] == [
        "cloud-credential-file"
    ]


# ------------------------------------------------------ a continuation's run


def _continuation_env(tmp_path: Path, agent_env: dict[str, str]) -> dict[str, str]:
    """The env a continuation's rollout resolves (its plan gate's inputs):
    the recorded credential table rides in it."""
    from benchflow.agents.env import CREDENTIAL_POLICY_MARKER
    from benchflow.contracts import default_rollout_planes
    from benchflow.rollout import RolloutConfig, rollout_gate_inputs
    from benchflow.task.task import Task

    cfg = RolloutConfig(
        task_path=_task(tmp_path / "tasks", "replay"),
        agent="openhands",
        model=None,
        environment="docker",
        agent_env=agent_env,
    )
    env = rollout_gate_inputs(
        Task(cfg.task_path), cfg, default_rollout_planes()
    ).agent_env
    assert CREDENTIAL_POLICY_MARKER in env
    return env


def test_the_replay_key_is_benchflows_not_the_operators(tmp_path):
    """A continuation's agent holds BenchFlow's own key for its replay
    proxy, by that name and value: never listed, in the env as the rollout
    resolves it (whose recorded table names the slot). The same name with
    another value is the operator's."""
    from benchflow.continue_run.orchestrator import build_agent_env

    replay = _continuation_env(tmp_path / "a", build_agent_env("http://127.0.0.1:9"))
    assert oc().in_sandbox("openhands", None, replay, proxy_in_sandbox=False) == []
    theirs = _continuation_env(
        tmp_path / "b", {**build_agent_env("http://127.0.0.1:9"), "LLM_API_KEY": KEY}
    )
    [entry] = oc().in_sandbox("openhands", None, theirs, proxy_in_sandbox=False)
    assert entry["kind"] == "api-key" and entry["names"] == ["LLM_API_KEY"]


async def test_a_continuation_on_the_host_route_warns_nothing(
    tmp_path, monkeypatch, caplog
):
    """The replay env at connect(): no warning, no record, no credential
    trigger (Docker, `public`)."""
    import logging

    from benchflow.continue_run.orchestrator import build_agent_env

    rollout, seen = _staging_rollout(
        tmp_path, monkeypatch, environment="docker", agent_env=None
    )
    with caplog.at_level(logging.WARNING):
        await rollout._before_staging(
            "openhands",
            None,
            _continuation_env(tmp_path, build_agent_env("http://172.17.0.1:9")),
            held=False,
            force_sandbox_local=False,
        )
    assert not [r for r in caplog.records if "operator credentials" in r.getMessage()]
    assert getattr(rollout, "_operator_credentials", None) is None
    assert seen == [(False, False)]


def _continue_folder(tmp_path: Path, environment: str) -> tuple[Path, Path]:
    from tests.continue_run._helpers import completion, exchange, write_run_folder

    task = _task(tmp_path / "tasks", "demo-task")
    run = write_run_folder(
        tmp_path / "run",
        exchanges=[exchange(completion(content="a"))],
        model="google-vertex/gemini-2.5-pro",
    )
    config = json.loads((run / "config.json").read_text())
    config["environment"] = environment
    (run / "config.json").write_text(json.dumps(config))
    return run, task.parent


def test_continue_on_the_sandbox_route_refuses_a_vertex_cloud_file_first(
    tmp_path, adc_home, monkeypatch
):
    """`bench eval continue` on Daytona puts the live model's proxy in the
    sandbox: a Vertex model's cloud credential file is refused before any
    rollout exists, naming the flag."""
    from typer.testing import CliRunner

    from benchflow.cli.main import app
    from benchflow.rollout import Rollout

    created = []

    async def create(config):
        created.append(config)
        raise AssertionError("no rollout before the refusal")

    monkeypatch.setattr(Rollout, "create", create)
    run, tasks = _continue_folder(tmp_path, "daytona")
    done = CliRunner().invoke(
        app, ["eval", "continue", str(run), "--tasks-dir", str(tasks)]
    )
    assert done.exit_code == 1, done.output[-2000:]
    assert _says_why(" ".join(done.output.split())), done.output[-2000:]
    assert created == [] and ADC not in done.output


@pytest.mark.parametrize("environment", ["daytona", "docker"])
def test_continue_on_the_sandbox_route_checks_the_live_models_credentials_first(
    tmp_path, adc_home, monkeypatch, environment
):
    """Daytona, with both flags: the continuation's rollout gets them, and the
    live model's credentials are classified, recorded and main checked before
    the proxy in the sandbox starts with them. The check and the proxy get one
    placement: on Docker (`--proxy-mode sandbox`) the live proxy is a host
    process, so its Vertex file needs no flag and nothing is recorded."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from typer.testing import CliRunner

    from benchflow.cli.main import app
    from benchflow.rollout import Rollout
    from benchflow.sandbox import lockdown

    order: list[str] = []
    made: list[Rollout] = []

    async def create(config):
        rollout = Rollout.__new__(Rollout)
        rollout._config = config
        rollout._task = SimpleNamespace(name="demo-task")
        rollout._env = SimpleNamespace()
        rollout._rollout_dir = None
        rollout._disallow_web_tools = False
        rollout._egress_denylist = None
        rollout.setup = rollout.start = rollout.install_agent = AsyncMock()
        rollout._planes = SimpleNamespace(
            resolve_agent_env=lambda *a: {
                "GOOGLE_APPLICATION_CREDENTIALS_JSON": ADC,
                "GOOGLE_CLOUD_PROJECT": "p",
            }
        )
        rollout.cleanup = AsyncMock()
        made.append(rollout)
        return rollout

    async def check(env, agent_env, *, held=None, credentials=None):
        order.append(f"main checked (credentials={bool(credentials)})")

    class Stop(Exception):
        pass

    async def proxy(**kwargs):
        order.append(f"proxy starts (forced={kwargs.get('force_sandbox_local')})")
        raise Stop

    monkeypatch.setattr(Rollout, "create", create)
    monkeypatch.setattr(lockdown, "refuse_task_processes_before_launch", check)
    monkeypatch.setattr("benchflow.providers.runtime.ensure_litellm_runtime", proxy)
    run, tasks = _continue_folder(tmp_path, environment)
    args = ["eval", "continue", str(run), "--tasks-dir", str(tasks)]
    if environment == "docker":
        # The sandbox replay on Docker: its live proxy is a host process.
        CliRunner().invoke(app, [*args, "--proxy-mode", "sandbox"])
        [rollout] = made
        assert order == [
            "main checked (credentials=False)",
            "proxy starts (forced=False)",
        ], order
        assert getattr(rollout, "_operator_credentials", None) is None
        return
    CliRunner().invoke(app, [*args, FLAG, TP_FLAG])
    [rollout] = made
    assert rollout._config.allow_cloud_credential_file is True
    assert rollout._config.allow_task_processes_with_credentials is True
    assert order == [
        "main checked (credentials=False)",
        "proxy starts (forced=False)",
    ], order
    record = rollout._operator_credentials
    assert [e["kind"] for e in record["in_sandbox"]] == ["cloud-credential-file"]
    assert record["in_sandbox"][0]["route"] == "proxy"
    assert {g["run"] for g in record["granted"]} == {FLAG, TP_FLAG}


def test_bench_eval_continue_batch_takes_both_flags(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from benchflow.cli.main import app
    from benchflow.continue_run import batch

    seen = {}

    async def run_all(folders, **kwargs):
        seen.update(
            cloud=kwargs["allow_cloud_credential_file"],
            beside=kwargs["allow_task_processes_with_credentials"],
        )
        return []

    monkeypatch.setattr(batch, "continue_batch", run_all)
    monkeypatch.setattr(
        batch, "discover_timeout_run_folders", lambda root, limit=None: [root]
    )
    CliRunner().invoke(app, ["eval", "continue-batch", str(tmp_path), FLAG, TP_FLAG])
    assert seen == {"cloud": True, "beside": True}


# ------------------------------------------------- a grant is read literally


@pytest.mark.parametrize("value", ["false", "no", 1, "true"])
def test_a_rollout_yamls_grant_is_true_or_false_only(value):
    from benchflow._utils.yaml_loader import rollout_config_from_dict

    for name in (
        "allow_cloud_credential_file",
        "allow_task_processes_with_credentials",
    ):
        with pytest.raises(ValueError, match=f"{name} must be true or false"):
            rollout_config_from_dict({"task_path": ".", "agent": "oracle", name: value})


async def test_a_connection_reads_its_grant_literally(tmp_path, monkeypatch):
    """A config object built in Python with ``"false"``: refused, never
    taken for the permission."""
    rollout, seen = _staging_rollout(
        tmp_path, monkeypatch, environment="daytona", agent_env=None, beside="false"
    )
    with pytest.raises(ValueError, match="allow_task_processes_with_credentials"):
        await rollout._before_staging(
            "claude-agent-acp",
            "deepseek/deepseek-flash",
            {"DEEPSEEK_API_KEY": KEY},
            held=False,
            force_sandbox_local=False,
        )
    assert seen == []


def test_a_resumed_job_keeps_its_grants_read_literally():
    from benchflow.evaluation import _config_from_record

    name = "allow_task_processes_with_credentials"
    base = {"agent": "oracle", "environment": "docker"}
    assert getattr(_config_from_record({**base, name: True}, None), name) is True
    assert getattr(_config_from_record(base, None), name) is False
    with pytest.raises(ValueError, match=name):
        _config_from_record({**base, name: "false"}, None)


# ------------------------------- main is listed again after the task's code

_KEEP_ALIVE = "BF proc 1 0 0 0 S 100\t/usr/bin/dash\tsh -c sleep infinity\n"
_DONE = "BF task-procs status=0\n"
# A server the agent's turn left (its shell has exited: its parent is 1).
_AGENT_LEFT = "BF proc 80 1 1000 0 S 500\t/usr/bin/python3\tpython3 -m http.server\n"
# What a step's test.sh left as root.
_TEST_SH_LEFT = "BF proc 90 1 0 0 S 600\t/usr/bin/sleep\tsleep 900\n"


@pytest.mark.parametrize("test_sh_leaves", [True, False], ids=["daemon", "nothing"])
async def test_a_connection_after_the_tasks_code_lists_main_again(
    monkeypatch, test_sh_leaves
):
    """A multi-step task: main is quiet at step one's connection; the agent
    leaves a server; root runs step one's test.sh, which may leave a daemon.
    Step two's connection, with an operator credential, lists main again:
    test.sh's daemon is refused, the agent's server is not."""
    from benchflow.sandbox import lockdown
    from tests.test_en698_policy_first_instruction import _Main

    async def read(env, command, **kwargs):
        return await env.exec(command, **kwargs)

    monkeypatch.setattr(lockdown, "_read_exec", read)
    why = oc().QUIET_MAIN_REFUSAL
    main = _Main(_KEEP_ALIVE + _DONE)
    await lockdown.refuse_task_processes_before_launch(
        main, {}, held=False, credentials=why
    )
    main.scan = _KEEP_ALIVE + _AGENT_LEFT + _DONE
    await lockdown.note_main_before_task_code(main)
    main.scan = (
        _KEEP_ALIVE + _AGENT_LEFT + (_TEST_SH_LEFT if test_sh_leaves else "") + _DONE
    )
    listings = len(main.commands)
    if not test_sh_leaves:
        await lockdown.refuse_task_processes_before_launch(
            main, {}, held=False, credentials=why
        )
        assert len(main.commands) == listings + 1
        # The next turn with no task code between finds what this one left
        # without listing main.
        main.scan = _KEEP_ALIVE + _AGENT_LEFT + _TEST_SH_LEFT + _DONE
        await lockdown.refuse_task_processes_before_launch(
            main, {}, held=False, credentials=why
        )
        assert len(main.commands) == listings + 1
        return
    with pytest.raises(RuntimeError) as refused:
        await lockdown.refuse_task_processes_before_launch(
            main, {}, held=False, credentials=why
        )
    text = str(refused.value)
    assert "pid 90 (uid 0): sleep" in text and "pid 80" not in text, text
    assert TP_FLAG in text


async def test_main_is_not_listed_again_without_the_tasks_code_or_a_credential(
    monkeypatch,
):
    """Another turn with no task code between (an ordinary multi-turn run)
    and a held policy alone keep the check once per sandbox."""
    from benchflow.sandbox import lockdown
    from tests.test_en698_policy_first_instruction import _Main

    async def read(env, command, **kwargs):
        return await env.exec(command, **kwargs)

    monkeypatch.setattr(lockdown, "_read_exec", read)
    why = oc().QUIET_MAIN_REFUSAL
    main = _Main(_KEEP_ALIVE + _DONE)
    await lockdown.refuse_task_processes_before_launch(
        main, {}, held=False, credentials=why
    )
    main.scan = _KEEP_ALIVE + _TEST_SH_LEFT + _DONE
    await lockdown.refuse_task_processes_before_launch(
        main, {}, held=False, credentials=why
    )
    await lockdown.note_main_before_task_code(main)
    listed = len(main.commands)
    await lockdown.refuse_task_processes_before_launch(main, {}, held=True)
    assert len(main.commands) == listed


async def test_each_verify_notes_main_before_the_tasks_code(monkeypatch):
    """A step's soft verify and a verify (a branch's next child connects
    after it) note main before test.sh runs."""
    from types import SimpleNamespace

    from benchflow import rollout as ro
    from benchflow.rollout import Rollout
    from benchflow.sandbox import lockdown

    order: list[str] = []

    async def note(env):
        order.append(f"noted {env.name}")

    class Stop(Exception):
        pass

    async def test_sh(*_a, **_k):
        order.append("test.sh")
        raise Stop

    async def nothing(*_a, **_k):
        return None

    monkeypatch.setattr(lockdown, "note_main_before_task_code", note)
    monkeypatch.setattr(ro, "require_safe_branch_world", lambda _r: None)
    monkeypatch.setattr(ro, "_publish_trajectory_for_verifier", test_sh)
    monkeypatch.setattr(Rollout, "_stop_sandbox_proxies_before_verify", nothing)
    monkeypatch.setattr(Rollout, "_snapshot_session_logs_before_verify", nothing)
    monkeypatch.setattr(Rollout, "_score_finished_solver", test_sh)
    rollout = Rollout.__new__(Rollout)
    rollout._env = SimpleNamespace(name="main")
    rollout._task = SimpleNamespace(config=SimpleNamespace())
    rollout._verify_task = None
    rollout._muse_credentials_staged = False
    rollout._trajectory = []
    rollout._config = SimpleNamespace(
        integrity="off", sandbox_user=None, primary_agent="claude-agent-acp"
    )
    rollout._rollout_paths = SimpleNamespace(
        verifier_dir=Path("/nonexistent-verifier-dir"), agent_dir=Path("/x")
    )
    monkeypatch.setattr(Path, "mkdir", lambda *a, **k: None)
    # The soft verify reports what stopped it as its own error.
    await rollout._soft_verify()
    assert order == ["noted main", "test.sh"], order
    order.clear()
    with pytest.raises(Stop):
        await rollout.verify()
    assert order == ["noted main", "test.sh"], order


async def test_a_continue_batch_hands_each_run_both_flags(tmp_path):
    from benchflow.continue_run.batch import continue_batch
    from benchflow.continue_run.orchestrator import ContinueResult

    seen = []

    async def runner(folder, **kwargs):
        seen.append(
            (
                kwargs["allow_cloud_credential_file"],
                kwargs["allow_task_processes_with_credentials"],
            )
        )
        return ContinueResult(
            rollout_dir=folder,
            rewards=None,
            error=None,
            n_recorded=0,
            n_live=0,
            divergences=0,
        )

    await continue_batch(
        [tmp_path],
        concurrency=1,
        tasks_dir=None,
        model=None,
        timeout=None,
        output_dir=None,
        runner=runner,
        allow_cloud_credential_file=True,
        allow_task_processes_with_credentials=True,
    )
    assert seen == [(True, True)]


def test_bf_branch_hands_its_plan_both_flags(tmp_path, monkeypatch):
    import benchflow as bf
    from benchflow import branch_api

    seen = {}

    async def run(plan, task, **_kwargs):
        seen.update(
            cloud=plan.allow_cloud_credential_file,
            beside=plan.allow_task_processes_with_credentials,
        )
        from benchflow.branch_run import BranchTrialOutcome

        return BranchTrialOutcome(task=task.name)

    monkeypatch.setattr(branch_api, "run_branch_trial", run)
    monkeypatch.setattr(branch_api, "write_branch_job", lambda plan, outcomes: tmp_path)
    monkeypatch.setattr(
        branch_api,
        "_result",
        lambda outcome, job_dir: SimpleNamespace(value=None, error=None),
    )
    bf.branch(
        _task(tmp_path),
        agent="oracle",
        children={"a": None, "b": None},
        jobs_dir=tmp_path / "jobs",
        allow_cloud_credential_file=True,
        allow_task_processes_with_credentials=True,
    )
    assert seen == {"cloud": True, "beside": True}


def test_bench_tasks_check_takes_the_cloud_file_flag(tmp_path, adc_home):
    """A Vertex task on the Gemini CLI's own route: refused without the
    flag, naming it; checked with it."""
    from typer.testing import CliRunner

    from benchflow.cli.main import app

    task = _task(tmp_path / "tasks")
    args = [
        "tasks",
        "check",
        str(task),
        "--agent",
        "gemini",
        "--model",
        "google-vertex/gemini-2.5-pro",
    ]
    refused = CliRunner().invoke(app, args)
    assert refused.exit_code != 0 and FLAG in " ".join(refused.output.split())
    checked = CliRunner().invoke(app, [*args, FLAG])
    assert checked.exit_code == 0, checked.output[-2000:]


def test_a_job_reads_its_grants_literally(tmp_path):
    """An EvaluationConfig built in Python with ``"false"`` is refused before
    the job starts."""
    for name in (
        "allow_cloud_credential_file",
        "allow_task_processes_with_credentials",
    ):
        job = _job(tmp_path / name, agent="oracle", **{name: "false"})
        with pytest.raises(ValueError, match=f"{name} must be true or false"):
            job._check_cloud_credential_files()


@pytest.mark.parametrize(
    ("no_web", "policy", "checked"),
    [(True, None, True), (False, object(), True), (False, None, False)],
    ids=["no-web", "egress-policy", "open"],
)
async def test_connect_checks_with_its_policy_before_anything_is_staged(
    tmp_path, monkeypatch, no_web, policy, checked
):
    """``connect()`` (a branch's child, a retry) runs the staging check before
    its proxy and its logins, with the connection's policy: held under
    no-web or an egress policy, and the proxy then counted as in the
    sandbox."""
    from unittest.mock import AsyncMock

    from benchflow import rollout as ro
    from benchflow.rollout import Rollout
    from benchflow.sandbox import _workspace_grant

    seen = {}

    class Stop(Exception):
        pass

    async def before_staging(agent, model, agent_env, *, held, force_sandbox_local):
        seen.update(held=held, force_sandbox_local=force_sandbox_local)
        raise Stop

    staged = AsyncMock()
    monkeypatch.setattr(ro, "require_safe_branch_world", lambda _r: None)
    monkeypatch.setattr(_workspace_grant, "reopen_granted_workspaces", AsyncMock())
    monkeypatch.setattr(Rollout, "_stage_connection_credentials", staged)
    rollout = Rollout.__new__(Rollout)
    rollout._config = SimpleNamespace(
        primary_agent="claude-agent-acp", primary_model="deepseek/deepseek-flash"
    )
    rollout._env = SimpleNamespace()
    rollout._agent_env = {"DEEPSEEK_API_KEY": KEY}
    rollout._usage_runtime = None
    rollout._disallow_web_tools = no_web
    rollout._end_previous_connection = AsyncMock()
    rollout._require_rollout_dir = lambda: tmp_path
    rollout._agent_egress_policy = AsyncMock(return_value=policy)
    rollout._before_staging = before_staging
    with pytest.raises(Stop):
        await rollout.connect()
    assert seen == {"held": checked, "force_sandbox_local": checked}
    staged.assert_not_called()


# Probe w: a connection between two soft verifies that lists nothing (one
# with no operator credential, or a held policy alone, once per sandbox).
@pytest.mark.parametrize("between", ["no-credential", "held-alone"])
async def test_a_step_that_lists_nothing_leaves_the_next_note_empty(
    monkeypatch, between
):
    """Step one lists main and its note records the agent's server; step
    one's test.sh leaves a root daemon; step two lists nothing, so its note
    must not record the daemon as the agent's; step three, with an operator
    credential, is refused, naming it."""
    from benchflow.sandbox import lockdown
    from tests.test_en698_policy_first_instruction import _Main

    async def read(env, command, **kwargs):
        return await env.exec(command, **kwargs)

    monkeypatch.setattr(lockdown, "_read_exec", read)
    why = oc().QUIET_MAIN_REFUSAL
    main = _Main(_KEEP_ALIVE + _DONE)
    await lockdown.refuse_task_processes_before_launch(
        main, {}, held=False, credentials=why
    )
    main.scan = _KEEP_ALIVE + _AGENT_LEFT + _DONE
    await lockdown.note_main_before_task_code(main)
    main.scan = _KEEP_ALIVE + _AGENT_LEFT + _TEST_SH_LEFT + _DONE
    if between == "no-credential":
        await lockdown.refuse_task_processes_before_launch(
            main, {}, held=False, credentials=None
        )
    else:
        await lockdown.refuse_task_processes_before_launch(main, {}, held=True)
    await lockdown.note_main_before_task_code(main)
    assert main._benchflow_agent_left == [(80, 500)]
    with pytest.raises(RuntimeError) as refused:
        await lockdown.refuse_task_processes_before_launch(
            main, {}, held=False, credentials=why
        )
    text = str(refused.value)
    assert "pid 90 (uid 0): sleep" in text and "pid 80" not in text, text


async def test_a_roles_connection_checks_main_before_anything_is_staged(
    tmp_path, monkeypatch
):
    """``connect_as`` (each step of a multi-step task, each scene role) runs
    the staging check before the role's proxy and logins: Claude Code's own
    token beside a task process in main is refused there, naming the way
    out, and nothing is staged."""
    from unittest.mock import AsyncMock

    from benchflow import rollout as ro
    from benchflow.agents.env import resolve_agent_env
    from benchflow.rollout import Rollout
    from benchflow.sandbox import _workspace_grant, lockdown
    from tests.test_en698_policy_first_instruction import SCAN, _Main

    async def read(env, command, **kwargs):
        return await env.exec(command, **kwargs)

    async def same(_env, policy):
        return policy

    class Staged(Exception):
        pass

    staged = AsyncMock(side_effect=Staged)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(lockdown, "_read_exec", read)
    monkeypatch.setattr(ro, "require_safe_branch_world", lambda _r: None)
    monkeypatch.setattr(ro, "_task_egress_denylist", lambda _t: None)
    monkeypatch.setattr(ro, "task_service_policy", same)
    monkeypatch.setattr(_workspace_grant, "reopen_granted_workspaces", AsyncMock())
    monkeypatch.setattr(Rollout, "_stage_connection_credentials", staged)
    monkeypatch.setattr(
        "benchflow.providers.runtime.ensure_litellm_runtime",
        AsyncMock(side_effect=Staged),
    )
    rollout = Rollout.__new__(Rollout)
    rollout._config = SimpleNamespace(
        primary_agent="claude-agent-acp",
        primary_model="claude-haiku-4-5",
        agent_env={},
        sandbox_user="agent",
        environment="docker",
        task_path=tmp_path,
        allow_cloud_credential_file=False,
        allow_task_processes_with_credentials=False,
    )
    rollout._task = SimpleNamespace(name="steps")
    rollout._env = _Main(SCAN)
    rollout._usage_runtime = None
    rollout._disallow_web_tools = False
    rollout._end_previous_connection = AsyncMock()
    rollout._require_rollout_dir = lambda: tmp_path
    rollout._planes = SimpleNamespace(
        agent_launch=lambda *a, **k: None, resolve_agent_env=resolve_agent_env
    )
    role = SimpleNamespace(
        name="agent",
        agent="claude-agent-acp",
        model="claude-haiku-4-5",
        reasoning_effort=None,
        env={"ANTHROPIC_AUTH_TOKEN": TOKEN, "ANTHROPIC_BASE_URL": "http://127.0.0.1:9"},
    )
    with pytest.raises(RuntimeError) as refused:
        await rollout.connect_as(role)
    text = str(refused.value)
    assert "with an operator credential in the sandbox" in text, text
    assert TP_FLAG in text and TOKEN not in text
    staged.assert_not_called()
    [entry] = rollout._operator_credentials["would_stage"]
    assert entry["names"] == ["ANTHROPIC_AUTH_TOKEN"]


def test_a_job_refuses_a_raw_commands_cloud_file_before_it_starts(tmp_path, adc_home):
    """The job-level check resolves the env as a trial does: a raw command's
    gcloud credentials are its own slot there too."""
    job = _job(
        tmp_path,
        agent=RAW_COMMAND,
        model=None,
        agent_env={"GOOGLE_APPLICATION_CREDENTIALS_JSON": ADC},
    )
    with pytest.raises(oc().CloudCredentialFileRefused) as refused:
        job._check_cloud_credential_files()
    assert ADC not in str(refused.value)


# ------------------------------------------- credentials scoped to the run

SCOPED_FLAG = "--scoped-credential"
NATIVE_ENV = {"ANTHROPIC_AUTH_TOKEN": TOKEN, "ANTHROPIC_BASE_URL": "http://127.0.0.1:9"}


def _resolved(agent: str, model: str | None, env: dict[str, str]) -> dict[str, str]:
    from benchflow.agents.env import resolve_agent_env

    return resolve_agent_env(agent, model, dict(env))


def test_a_scoped_name_is_its_own_entry_marked_scoped(tmp_path, monkeypatch):
    """Claude Code's own token, named with --scoped-credential: still in the
    sandbox, kind and route unchanged, marked scoped."""
    monkeypatch.setenv("HOME", str(tmp_path))
    env = _resolved("claude-agent-acp", "claude-haiku-4-5", NATIVE_ENV)
    [entry] = oc().in_sandbox(
        "claude-agent-acp",
        "claude-haiku-4-5",
        env,
        proxy_in_sandbox=False,
        scoped=("ANTHROPIC_AUTH_TOKEN",),
    )
    assert entry == {
        "kind": "api-key",
        "route": "native",
        "names": ["ANTHROPIC_AUTH_TOKEN"],
        "staged": "the agent's environment",
        "scoped": True,
    }


async def test_main_is_asked_only_for_the_unscoped(tmp_path, monkeypatch):
    """Two keys the proxy in the sandbox holds, one scoped: the quiet-main
    refusal names only the other. Both scoped: main isn't asked, and no
    flag is needed."""
    from benchflow.sandbox import lockdown

    seen: list[str | None] = []

    async def check(env, agent_env, *, held=None, credentials=None):
        seen.append(credentials)

    rollout, _ = _staging_rollout(
        tmp_path, monkeypatch, environment="daytona", agent_env=None
    )
    monkeypatch.setattr(lockdown, "refuse_task_processes_before_launch", check)
    env = {"DEEPSEEK_API_KEY": KEY, "OPENAI_API_KEY": KEY + "-2"}
    rollout._agent_env = dict(env)  # the run's own values (the primary's env)
    for scoped in (("DEEPSEEK_API_KEY",), ("DEEPSEEK_API_KEY", "OPENAI_API_KEY")):
        rollout._config.scoped_credentials = scoped
        await rollout._before_staging(
            "claude-agent-acp",
            "deepseek/deepseek-flash",
            env,
            held=False,
            force_sandbox_local=False,
        )
    mixed, every = seen
    assert "(OPENAI_API_KEY)" in mixed and "DEEPSEEK_API_KEY" not in mixed, mixed
    assert SCOPED_FLAG in mixed and TP_FLAG in mixed
    assert every is None
    record = rollout._operator_credentials
    assert record["scoped"] == ["DEEPSEEK_API_KEY", "OPENAI_API_KEY"]
    assert record["granted"] == []


@pytest.mark.parametrize(
    ("agent", "model", "env", "name", "says"),
    [
        (
            "claude-agent-acp",
            "claude-haiku-4-5",
            NATIVE_ENV,
            "CLAUDE_CODE_OAUTH_TOKEN",
            "does not set CLAUDE_CODE_OAUTH_TOKEN",
        ),
        (
            "codex-acp",
            None,
            {"CODEX_AUTH_JSON": '{"tokens": {}}'},
            "CODEX_AUTH_JSON",
            "is staged as a file",
        ),
        (
            "gemini",
            "google-vertex/gemini-2.5-pro",
            {"GOOGLE_CLOUD_PROJECT": "p"},
            "GOOGLE_APPLICATION_CREDENTIALS_JSON",
            FLAG,
        ),
        (
            "claude-agent-acp",
            "claude-haiku-4-5",
            NATIVE_ENV,
            "ANTHROPIC_BASE_URL",
            "is not a credential variable",
        ),
    ],
    ids=["unset", "login-file", "cloud-file", "not-a-credential"],
)
def test_a_name_that_cant_be_scoped_is_refused_before_any_sandbox(
    tmp_path, adc_home, agent, model, env, name, says
):
    """``bench run --dry-run`` and ``Rollout.setup``'s plan gate: an unset
    name, a login file or the cloud credential file is refused, naming the
    variable and never a value."""
    from benchflow.contracts import default_rollout_planes
    from benchflow.rollout import RolloutConfig, dry_run_plan_gate

    cfg = RolloutConfig(
        task_path=_task(tmp_path),
        agent=agent,
        model=model,
        environment="docker",
        agent_env=dict(env),
        scoped_credentials=(name,),
        allow_cloud_credential_file=True,
    )
    with pytest.raises(oc().ScopedCredentialRefused) as refused:
        dry_run_plan_gate(cfg, default_rollout_planes())
    text = str(refused.value)
    assert f"{SCOPED_FLAG} {name}" in text and says in text, text
    assert TOKEN not in text and ADC not in text


def test_a_job_refuses_an_unset_scoped_name_before_it_starts(tmp_path, adc_home):
    job = _job(
        tmp_path,
        agent="claude-agent-acp",
        model="claude-haiku-4-5",
        agent_env=dict(NATIVE_ENV),
        scoped_credentials=("ANTHROPIC_API_KEY",),
    )
    with pytest.raises(oc().ScopedCredentialRefused, match="ANTHROPIC_API_KEY"):
        job._check_scoped_credentials()
    job = _job(
        tmp_path / "ok",
        agent="claude-agent-acp",
        model="claude-haiku-4-5",
        agent_env=dict(NATIVE_ENV),
        scoped_credentials=("ANTHROPIC_AUTH_TOKEN",),
    )
    job._check_scoped_credentials()


def test_the_records_and_lines_name_the_scoped_names(tmp_path):
    entry = {
        "kind": "api-key",
        "route": "native",
        "names": ["ANTHROPIC_AUTH_TOKEN"],
        "staged": "the agent's environment",
        "scoped": True,
    }
    record = oc().record(None, [entry], [])
    assert record["scoped"] == ["ANTHROPIC_AUTH_TOKEN"]
    line = oc().warning("hosted", [entry], [])
    assert "ANTHROPIC_AUTH_TOKEN, scoped" in line
    assert f"scoped ANTHROPIC_AUTH_TOKEN to this run ({SCOPED_FLAG})" in line
    block = oc().run_card_block(
        [_task(tmp_path)],
        agent="claude-agent-acp",
        model="claude-haiku-4-5",
        agent_env=_resolved("claude-agent-acp", "claude-haiku-4-5", NATIVE_ENV),
        environment="docker",
        sandbox_user="agent",
        allow_cloud_credential_file=False,
        scoped_credentials=("ANTHROPIC_AUTH_TOKEN",),
    )
    assert block["scoped"] == ["ANTHROPIC_AUTH_TOKEN"]
    assert TOKEN not in json.dumps(block)
    [start] = oc().run_start_lines(block)
    assert "scoped ANTHROPIC_AUTH_TOKEN to this run" in start and TOKEN not in start


@pytest.mark.parametrize(
    "value", ["ANTHROPIC_AUTH_TOKEN", [1], [""], {"a": 1}], ids=str
)
def test_scoped_credentials_is_a_list_of_names(value):
    from benchflow._utils.yaml_loader import rollout_config_from_dict
    from benchflow.eval_worker import _evaluation_config

    with pytest.raises(ValueError, match="scoped_credentials must be a list"):
        rollout_config_from_dict(
            {"task_path": ".", "agent": "oracle", "scoped_credentials": value}
        )
    with pytest.raises(ValueError, match="scoped_credentials must be a list"):
        _evaluation_config(
            {"agent": "oracle", "environment": "docker", "scoped_credentials": value}
        )


def test_the_scoped_names_round_trip_through_config_worker_and_resume(tmp_path):
    from benchflow._utils.yaml_loader import rollout_config_from_dict
    from benchflow.eval_sharding import EvalShard, _config_payload
    from benchflow.eval_worker import _evaluation_config
    from benchflow.evaluation import EvaluationConfig, _config_from_record

    names = ["ANTHROPIC_AUTH_TOKEN"]
    cfg = rollout_config_from_dict(
        {"task_path": ".", "agent": "oracle", "scoped_credentials": names}
    )
    assert cfg.scoped_credentials == ("ANTHROPIC_AUTH_TOKEN",)
    raw = {"agent": "oracle", "environment": "docker", "scoped_credentials": names}
    assert _evaluation_config(raw).scoped_credentials == ("ANTHROPIC_AUTH_TOKEN",)
    assert _config_from_record(raw, None).scoped_credentials == (
        "ANTHROPIC_AUTH_TOKEN",
    )
    payload = _config_payload(
        EvaluationConfig(agent="oracle", scoped_credentials=tuple(names)),
        shard=EvalShard(index=0, task_names=("t",), concurrency=1),
    )
    assert payload["scoped_credentials"] == names
    assert _evaluation_config(payload).scoped_credentials == tuple(names)


def test_every_verb_with_the_task_processes_flag_takes_scoped_credential():
    import typer.main

    from benchflow.cli.main import app

    root = typer.main.get_command(app)

    def options(*path):
        command = root
        for name in path:
            command = command.commands[name]
        return {o for p in command.params for o in getattr(p, "opts", ())}

    for verb in (
        ("run",),
        ("eval", "run"),
        ("branch",),
        ("eval", "continue"),
        ("eval", "continue-batch"),
    ):
        assert {TP_FLAG, SCOPED_FLAG} <= options(*verb), verb


def test_bf_branch_hands_its_plan_the_scoped_names(tmp_path, monkeypatch):
    import benchflow as bf
    from benchflow import branch_api

    seen = {}

    async def run(plan, task, **_kwargs):
        seen["scoped"] = plan.scoped_credentials
        from benchflow.branch_run import BranchTrialOutcome

        return BranchTrialOutcome(task=task.name)

    monkeypatch.setattr(branch_api, "run_branch_trial", run)
    monkeypatch.setattr(branch_api, "write_branch_job", lambda plan, outcomes: tmp_path)
    monkeypatch.setattr(
        branch_api,
        "_result",
        lambda outcome, job_dir: SimpleNamespace(value=None, error=None),
    )
    bf.branch(
        _task(tmp_path),
        agent="oracle",
        children={"a": None, "b": None},
        jobs_dir=tmp_path / "jobs",
        scoped_credentials=["ANTHROPIC_AUTH_TOKEN"],
    )
    assert seen == {"scoped": ("ANTHROPIC_AUTH_TOKEN",)}


def test_a_job_with_an_unset_scoped_name_is_refused_before_its_folder_exists(
    tmp_path, adc_home, monkeypatch
):
    from benchflow.evaluation import Evaluation

    async def started(self):
        raise _Started("the job started")

    monkeypatch.setattr(Evaluation, "_run_unlocked", started)
    job = _job(
        tmp_path,
        agent="claude-agent-acp",
        model="claude-haiku-4-5",
        agent_env=dict(NATIVE_ENV),
        scoped_credentials=("ANTHROPIC_API_KEY",),
    )
    with pytest.raises(oc().ScopedCredentialRefused, match="ANTHROPIC_API_KEY"):
        job.run_sync()
    assert not (tmp_path / "jobs").exists() or not any((tmp_path / "jobs").iterdir())


async def test_the_task_processes_grant_is_recorded_only_for_the_unscoped(
    tmp_path, monkeypatch
):
    """Every credential scoped: --allow-task-processes-with-credentials lets
    nothing more in, so neither the trial nor the reward's provenance
    records it."""
    rollout, _ = _staging_rollout(
        tmp_path, monkeypatch, environment="daytona", agent_env=None, beside=True
    )
    rollout._config.scoped_credentials = ("DEEPSEEK_API_KEY",)
    rollout._agent_env = {"DEEPSEEK_API_KEY": KEY}
    await rollout._before_staging(
        "claude-agent-acp",
        "deepseek/deepseek-flash",
        {"DEEPSEEK_API_KEY": KEY},
        held=False,
        force_sandbox_local=False,
    )
    assert rollout._operator_credentials["granted"] == []
    assert rollout._operator_credentials["scoped"] == ["DEEPSEEK_API_KEY"]
    assert getattr(rollout._env, "_benchflow_operator_credentials_lifted", None) is None


@pytest.mark.parametrize(
    ("name", "says"),
    [
        ("NOT_SET_TOKEN", "does not set NOT_SET_TOKEN"),
        ("CODEX_AUTH_JSON", "is staged as a file"),
    ],
    ids=["unset", "login-file"],
)
def test_a_continuation_refuses_a_name_it_cant_scope_before_any_rollout(
    tmp_path, adc_home, monkeypatch, name, says
):
    """`bench eval continue --scoped-credential` names the live model's
    credentials: checked on the live model's env before any rollout exists
    (the plan gate sees only the replay env)."""
    from typer.testing import CliRunner

    from benchflow.agents import env as agent_env
    from benchflow.cli.main import app
    from benchflow.rollout import Rollout

    created = []

    async def create(config):
        created.append(config)
        raise AssertionError("no rollout before the refusal")

    live = {"DEEPSEEK_API_KEY": KEY, "CODEX_AUTH_JSON": '{"tokens": {}}'}
    monkeypatch.setattr(Rollout, "create", create)
    monkeypatch.setattr(agent_env, "resolve_agent_env", lambda *a, **k: dict(live))
    run, tasks = _continue_folder(tmp_path, "daytona")
    done = CliRunner().invoke(
        app,
        ["eval", "continue", str(run), "--tasks-dir", str(tasks), SCOPED_FLAG, name],
    )
    text = " ".join(done.output.split())
    assert done.exit_code == 1 and f"{SCOPED_FLAG} {name}" in text, text[-1500:]
    assert says in text and created == [] and KEY not in text


def test_codexs_scoped_key_names_the_file_its_launcher_writes(tmp_path, monkeypatch):
    """codex-acp on its own route: the launcher also writes OPENAI_API_KEY's
    value to ~/.codex/auth.json, so the entry names that file; scoping still
    applies, to the same value."""
    monkeypatch.setenv("HOME", str(tmp_path))
    env = _resolved("codex-acp", None, {"OPENAI_API_KEY": KEY})
    found = oc().in_sandbox(
        "codex-acp",
        None,
        env,
        proxy_in_sandbox=False,
        home="/home/agent",
        scoped=("OPENAI_API_KEY",),
    )
    [entry] = [e for e in found if "OPENAI_API_KEY" in e["names"]]
    assert entry["staged"] == (
        "the agent's environment and /home/agent/.codex/auth.json"
    ), found
    assert entry.get("scoped") is True


@pytest.mark.parametrize("same", [True, False], ids=["same-value", "other-value"])
async def test_a_name_is_scoped_only_with_the_runs_own_value(
    tmp_path, monkeypatch, same
):
    """A scene role whose env holds the scoped name with another value (an
    operator's own key under that name) is the operator's credential: main
    is asked to be quiet for it."""
    from benchflow.sandbox import lockdown

    seen: list[str | None] = []

    async def check(env, agent_env, *, held=None, credentials=None):
        seen.append(credentials)

    rollout, _ = _staging_rollout(
        tmp_path, monkeypatch, environment="daytona", agent_env=None
    )
    monkeypatch.setattr(lockdown, "refuse_task_processes_before_launch", check)
    rollout._config.scoped_credentials = ("DEEPSEEK_API_KEY",)
    rollout._agent_env = {"DEEPSEEK_API_KEY": KEY}
    role_env = {"DEEPSEEK_API_KEY": KEY if same else "placeholder-operator-key"}
    await rollout._before_staging(
        "claude-agent-acp",
        "deepseek/deepseek-flash",
        role_env,
        held=False,
        force_sandbox_local=False,
    )
    if same:
        assert seen == [None]
        assert rollout._operator_credentials["scoped"] == ["DEEPSEEK_API_KEY"]
    else:
        [why] = seen
        assert why and "(DEEPSEEK_API_KEY)" in why
        assert "scoped" not in rollout._operator_credentials


def test_a_cloud_or_login_file_is_never_marked_scoped(adc_home):
    """Behind the refusal: the classifier never marks the cloud credential
    file or a login file scoped, even when named."""
    from benchflow.agents.env import VERTEX_ADC_ENV

    cloud = oc().in_sandbox(
        "gemini",
        "google-vertex/gemini-2.5-pro",
        {VERTEX_ADC_ENV: ADC, "GOOGLE_CLOUD_PROJECT": "p"},
        proxy_in_sandbox=False,
        scoped=(VERTEX_ADC_ENV,),
    )
    assert cloud and not [e for e in cloud if e.get("scoped")], cloud
    login = oc().in_sandbox(
        "codex-acp",
        None,
        _resolved("codex-acp", None, {"CODEX_AUTH_JSON": '{"tokens": {}}'}),
        proxy_in_sandbox=False,
        scoped=("CODEX_AUTH_JSON",),
    )
    assert login and not [e for e in login if e.get("scoped")], login


def test_the_jobs_scoped_check_reads_the_env_as_a_trial_resolves_it(
    tmp_path, adc_home, monkeypatch
):
    """A name the run's env inherits (the operator's shell sets it) is one
    the run gives the agent: the job's check resolves the env first."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    job = _job(
        tmp_path,
        agent="claude-agent-acp",
        model="claude-haiku-4-5",
        agent_env={},
        scoped_credentials=("ANTHROPIC_API_KEY",),
    )
    job._check_scoped_credentials()


def test_a_continuation_scopes_the_live_models_own_key(tmp_path, adc_home, monkeypatch):
    """On continue's sandbox route the live model's proxy holds its key in
    the sandbox; named with --scoped-credential, that key (the live model's
    value, not the replay env's) is scoped and main isn't asked."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from typer.testing import CliRunner

    from benchflow.agents import env as agent_env
    from benchflow.cli.main import app
    from benchflow.rollout import Rollout
    from benchflow.sandbox import lockdown

    live = {"DEEPSEEK_API_KEY": KEY}
    made: list[Rollout] = []
    asked: list[str | None] = []

    async def create(config):
        rollout = Rollout.__new__(Rollout)
        rollout._config = config
        rollout._task = SimpleNamespace(name="demo-task")
        rollout._env = SimpleNamespace()
        rollout._rollout_dir = None
        rollout._disallow_web_tools = False
        rollout._egress_denylist = None
        rollout.setup = rollout.start = rollout.install_agent = AsyncMock()
        rollout._planes = SimpleNamespace(resolve_agent_env=lambda *a: dict(live))
        rollout.cleanup = AsyncMock()
        made.append(rollout)
        return rollout

    async def check(env, agent_env, *, held=None, credentials=None):
        asked.append(credentials)

    class Stop(Exception):
        pass

    async def proxy(**kwargs):
        raise Stop

    monkeypatch.setenv("DEEPSEEK_API_KEY", KEY)  # the live forwarder's own check
    monkeypatch.setattr(Rollout, "create", create)
    monkeypatch.setattr(agent_env, "resolve_agent_env", lambda *a, **k: dict(live))
    monkeypatch.setattr(lockdown, "refuse_task_processes_before_launch", check)
    monkeypatch.setattr("benchflow.providers.runtime.ensure_litellm_runtime", proxy)
    run, tasks = _continue_folder(tmp_path, "daytona")
    done = CliRunner().invoke(
        app,
        [
            "eval",
            "continue",
            str(run),
            "--tasks-dir",
            str(tasks),
            "--model",
            "deepseek/deepseek-flash",
            SCOPED_FLAG,
            "DEEPSEEK_API_KEY",
        ],
    )
    assert made, done.output[-2000:]
    [rollout] = made
    assert asked == [None], asked
    assert rollout._operator_credentials["scoped"] == ["DEEPSEEK_API_KEY"]
