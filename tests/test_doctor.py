"""``benchflow.doctor`` checks, driven entirely through fake probes.

No test here touches Docker, the network, Daytona or a real credential file:
every side effect goes through :class:`benchflow.doctor.DoctorProbes`.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from benchflow import doctor
from benchflow.doctor import CommandResult, DoctorProbes, run_doctor

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
GIB = 1024**3

# Credential-shaped values that must never appear in any doctor output.
CLAUDE_TOKEN = "sk-ant-oat01-SECRETclaudetokenvalue-0123456789"
CLAUDE_FILE_TOKEN = "sk-ant-oat01-SECRETfiletokenvalue-9876543210"
OPENAI_KEY = "sk-proj-SECRETopenaikeyvalue-abcdefghijkl"
DAYTONA_KEY = "dtn_SECRETdaytonakeyvalue_0123456789"
REFRESH = "rt-SECRETrefreshvalue-0123456789abcdef"


def _jwt(claims: dict) -> str:
    def enc(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")

    return f"{enc({'alg': 'none'})}.{enc(claims)}.SECRETsignaturevalue"


def _docker_info(mem: int = 8 * GIB, cpus: int = 4) -> CommandResult:
    info = {
        "ServerVersion": "29.5.2",
        "NCPU": cpus,
        "MemTotal": mem,
        "OperatingSystem": "Ubuntu 24.04",
    }
    return CommandResult(0, json.dumps(info), "")


DOCKER_DOWN = CommandResult(
    1,
    json.dumps({"ServerVersion": ""}),
    "failed to connect to the docker API at unix:///x/.colima/default/docker.sock",
)


def _colima_list(status: str = "Running", mem: int = 8 * GIB) -> CommandResult:
    rows = [
        {"name": "default", "status": status, "cpus": 4, "memory": mem},
        {"name": "other", "status": "Stopped", "cpus": 2, "memory": GIB},
    ]
    return CommandResult(0, "\n".join(json.dumps(r) for r in rows), "")


def make_probes(
    tmp_path: Path,
    *,
    env: Mapping[str, str] | None = None,
    dotenv: Mapping[str, str] | None = None,
    files: Mapping[str, object] | None = None,
    binaries: Sequence[str] = ("docker", "uv"),
    commands: Mapping[tuple[str, ...], CommandResult] | None = None,
    http: Mapping[str, object] | None = None,
    dists: Mapping[str, str] | None = None,
    daytona_error: Exception | None = None,
    calls: list | None = None,
) -> DoctorProbes:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    for rel, content in (files or {}).items():
        path = home / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content if isinstance(content, str) else json.dumps(content))
    environ = {**(dotenv or {}), **(env or {})}
    origins = {k: ".env" for k in dotenv or {}} | {k: "env" for k in env or {}}
    default_commands: dict[tuple[str, ...], CommandResult] = {
        ("uv", "--version"): CommandResult(0, "uv 0.11.15\n", ""),
        ("docker", "context", "show"): CommandResult(0, "desktop-linux\n", ""),
        ("docker", "info", "--format", "{{json .}}"): _docker_info(),
        ("docker", "buildx", "version"): CommandResult(
            0, "github.com/docker/buildx v0.29.1\n", ""
        ),
    }
    default_commands.update(commands or {})
    record = calls if calls is not None else []

    def run(argv: Sequence[str], timeout: float) -> CommandResult:
        record.append(("run", tuple(argv)))
        return default_commands.get(tuple(argv), CommandResult(None, "", "not found"))

    def http_status(url: str, timeout: float) -> int:
        record.append(("http", url))
        value = (http or {}).get(url, 200)
        if isinstance(value, Exception):
            raise value
        assert isinstance(value, int)
        return value

    def daytona_check(environ: Mapping[str, str], timeout: float) -> None:
        record.append(("daytona", None))
        if daytona_error is not None:
            raise daytona_error

    return DoctorProbes(
        environ=environ,
        origins=origins,
        home=home,
        which=lambda name: f"/usr/bin/{name}" if name in binaries else None,
        run=run,
        http_status=http_status,
        dist_version=lambda dist: (dists or {}).get(dist),
        daytona_check=daytona_check,
        now=lambda: NOW,
        python_version=(3, 12, 9),
        python_executable="/venv/bin/python",
    )


def by_id(report) -> dict:
    return {c.id: c for c in report.checks}


def all_output(report) -> str:
    return json.dumps(report.to_dict())


# ── Sandbox ─────────────────────────────────────────────────────────────


def test_stopped_colima_fails_docker_with_start_hint(tmp_path):
    probes = make_probes(
        tmp_path,
        env={"CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN},
        binaries=("docker", "colima", "uv"),
        commands={
            ("docker", "context", "show"): CommandResult(0, "colima\n", ""),
            ("docker", "info", "--format", "{{json .}}"): DOCKER_DOWN,
            ("colima", "list", "--json"): _colima_list("Stopped"),
        },
    )
    report = run_doctor(probes=probes, offline=True)
    checks = by_id(report)
    assert checks["colima"].status == "fail"
    assert "Stopped" in checks["colima"].summary
    assert checks["colima"].fix == "colima start default"
    assert checks["docker"].status == "fail"
    assert checks["docker"].fix == "colima start default"
    assert not report.ok
    # The cause (Colima) is listed before its symptom (Docker).
    ids = [c.id for c in report.checks]
    assert ids.index("colima") < ids.index("docker")


def test_colima_profile_from_named_context_and_docker_host(tmp_path):
    for context_cmd, env, profile in (
        (CommandResult(0, "colima-demo\n", ""), {}, "demo"),
        (
            CommandResult(0, "ignored\n", ""),
            {"DOCKER_HOST": "unix:///Users/me/.colima/demo/docker.sock"},
            "demo",
        ),
    ):
        probes = make_probes(
            tmp_path,
            env={**env, "CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN},
            binaries=("docker", "colima"),
            commands={
                ("docker", "context", "show"): context_cmd,
                ("colima", "list", "--json"): CommandResult(
                    0,
                    json.dumps(
                        {
                            "name": "demo",
                            "status": "Running",
                            "cpus": 2,
                            "memory": 8 * GIB,
                        }
                    ),
                    "",
                ),
            },
        )
        colima = by_id(run_doctor(probes=probes, offline=True))["colima"]
        assert colima.status == "pass"
        assert colima.details["profile"] == profile


def test_low_memory_colima_warns_with_resize_command(tmp_path):
    probes = make_probes(
        tmp_path,
        env={"CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN},
        binaries=("docker", "colima"),
        commands={
            ("docker", "context", "show"): CommandResult(0, "colima\n", ""),
            ("docker", "info", "--format", "{{json .}}"): _docker_info(mem=2 * GIB),
            ("colima", "list", "--json"): _colima_list("Running", mem=2 * GIB),
        },
    )
    report = run_doctor(probes=probes, offline=True)
    checks = by_id(report)
    assert checks["colima"].status == "warn"
    assert "--memory 8" in checks["colima"].fix
    assert checks["docker"].status == "pass"
    assert checks["docker-memory"].status == "warn"
    assert "colima start default --memory 8" in checks["docker-memory"].fix
    assert report.ok  # warnings never fail the run


def test_low_memory_docker_desktop_points_at_settings(tmp_path):
    probes = make_probes(
        tmp_path,
        env={"CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN},
        commands={
            ("docker", "info", "--format", "{{json .}}"): _docker_info(mem=2 * GIB)
        },
    )
    checks = by_id(run_doctor(probes=probes, offline=True))
    assert "colima" not in checks
    assert "Docker Desktop" in checks["docker-memory"].fix


def test_missing_buildx_warns_and_explains_the_compose_line(tmp_path):
    """Regression test.

    Without the buildx plugin, every Docker build printed Compose's "Docker
    Compose requires buildx plugin to be installed" and fell back to the
    legacy builder, and doctor said nothing about it.
    """
    probes = make_probes(
        tmp_path,
        env={"CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN},
        commands={
            ("docker", "buildx", "version"): CommandResult(
                1, "", "docker: unknown command: docker buildx"
            )
        },
    )
    report = run_doctor(probes=probes, offline=True)
    check = by_id(report)["docker-buildx"]
    assert check.status == "warn"
    assert "requires buildx plugin" in check.summary
    assert "legacy builder" in check.summary
    assert "brew install docker-buildx" in check.fix
    assert report.ok


def test_buildx_is_not_probed_when_the_daemon_is_down(tmp_path):
    calls: list = []
    probes = make_probes(
        tmp_path,
        env={"CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN},
        commands={("docker", "info", "--format", "{{json .}}"): DOCKER_DOWN},
        calls=calls,
    )
    checks = by_id(run_doctor(probes=probes, offline=True))
    assert "docker-buildx" not in checks
    assert ("run", ("docker", "buildx", "version")) not in calls


def test_missing_docker_is_only_a_warning_for_daytona_runs(tmp_path):
    env = {"CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN, "DAYTONA_API_KEY": DAYTONA_KEY}
    docker_run = run_doctor(
        probes=make_probes(tmp_path, env=env, binaries=()), offline=True
    )
    assert by_id(docker_run)["docker"].status == "fail"
    daytona_run = run_doctor(
        sandbox="daytona",
        probes=make_probes(
            tmp_path, env=env, binaries=(), dists={"daytona": "0.184.0"}
        ),
        offline=True,
    )
    assert by_id(daytona_run)["docker"].status == "warn"
    assert daytona_run.ok


def test_daytona_not_configured_is_skipped(tmp_path):
    report = run_doctor(
        probes=make_probes(tmp_path, env={"CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN}),
        offline=True,
    )
    assert by_id(report)["daytona"].status == "skip"


@pytest.mark.parametrize(
    ("sandbox", "expected"),
    [("docker", "warn"), ("daytona", "fail")],
)
def test_daytona_sdk_missing_names_the_extra(tmp_path, sandbox, expected):
    probes = make_probes(
        tmp_path,
        env={"CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN, "DAYTONA_API_KEY": DAYTONA_KEY},
    )
    check = by_id(run_doctor(sandbox=sandbox, probes=probes, offline=True))["daytona"]
    assert check.status == expected
    assert "benchflow[sandbox-daytona]" in check.fix
    assert "--extra sandbox-daytona" in check.fix


def test_daytona_old_sdk_and_missing_key_fail_when_required(tmp_path):
    old = make_probes(
        tmp_path,
        env={"CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN, "DAYTONA_API_KEY": DAYTONA_KEY},
        dists={"daytona": "0.183.2"},
    )
    check = by_id(run_doctor(sandbox="daytona", probes=old, offline=True))["daytona"]
    assert check.status == "fail"
    assert "older than 0.184.0" in check.summary
    no_key = make_probes(
        tmp_path,
        env={"CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN},
        dists={"daytona": "0.184.0"},
    )
    check = by_id(run_doctor(sandbox="daytona", probes=no_key, offline=True))["daytona"]
    assert check.status == "fail"
    assert "DAYTONA_API_KEY is not set" in check.summary


def test_daytona_rejected_key_fails_and_never_echoes_the_key(tmp_path):
    calls: list = []
    probes = make_probes(
        tmp_path,
        env={"CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN, "DAYTONA_API_KEY": DAYTONA_KEY},
        dists={"daytona": "0.184.0"},
        daytona_error=RuntimeError(f"401 Unauthorized for key {DAYTONA_KEY}"),
        calls=calls,
    )
    report = run_doctor(sandbox="daytona", probes=probes)
    check = by_id(report)["daytona"]
    assert check.status == "fail"
    assert "401 Unauthorized" in check.summary
    assert DAYTONA_KEY not in all_output(report)
    assert ("daytona", None) in calls


def test_daytona_offline_skips_the_live_call(tmp_path):
    calls: list = []
    probes = make_probes(
        tmp_path,
        env={"CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN},
        dotenv={"DAYTONA_API_KEY": DAYTONA_KEY},
        dists={"daytona": "0.184.0"},
        calls=calls,
    )
    report = run_doctor(sandbox="daytona", probes=probes, offline=True)
    check = by_id(report)["daytona"]
    assert check.status == "pass"
    assert "(.env)" in check.summary
    assert not [c for c in calls if c[0] in ("daytona", "http")]


def test_unchecked_sandbox_is_listed_as_skip(tmp_path):
    probes = make_probes(tmp_path, env={"CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN})
    report = run_doctor(sandbox="modal", probes=probes, offline=True)
    assert by_id(report)["sandbox.modal"].status == "skip"
    assert by_id(report)["docker"].status == "pass"


# ── Agent credentials ───────────────────────────────────────────────────


def _claude_file(expires: datetime, token: str = CLAUDE_FILE_TOKEN) -> dict:
    return {
        "claudeAiOauth": {
            "accessToken": token,
            "refreshToken": REFRESH,
            "expiresAt": int(expires.timestamp() * 1000),
            "scopes": ["user:inference"],
        }
    }


def test_expired_claude_file_alone_warns_with_setup_token_fix(tmp_path):
    """Guards this failure mode: an
    expired ~/.claude/.credentials.json was silently used as the Claude login
    and the run failed later with an opaque auth error."""
    probes = make_probes(
        tmp_path,
        files={".claude/.credentials.json": _claude_file(NOW - timedelta(days=30))},
    )
    report = run_doctor(probes=probes, offline=True)
    check = by_id(report)["auth.claude-agent-acp"]
    assert check.status == "warn"
    assert "expired 2025-12-02" in check.summary
    assert "(30 days ago)" in check.summary
    assert "claude setup-token" in check.fix
    assert "CLAUDE_CODE_OAUTH_TOKEN" in check.fix
    assert not report.agents["claude-agent-acp"].ready
    assert CLAUDE_FILE_TOKEN not in all_output(report)
    assert REFRESH not in all_output(report)


def test_env_oauth_token_wins_over_stale_file(tmp_path):
    probes = make_probes(
        tmp_path,
        env={"CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN},
        files={".claude/.credentials.json": _claude_file(NOW - timedelta(days=3))},
    )
    report = run_doctor(probes=probes, offline=True)
    check = by_id(report)["auth.claude-agent-acp"]
    assert check.status == "pass"
    assert check.summary.startswith("CLAUDE_CODE_OAUTH_TOKEN (env)")
    assert "ignored ~/.claude/.credentials.json" in check.summary
    assert report.agents["claude-agent-acp"].ready
    assert CLAUDE_TOKEN not in all_output(report)


def test_valid_claude_file_passes_with_expiry(tmp_path):
    probes = make_probes(
        tmp_path,
        files={".claude/.credentials.json": _claude_file(NOW + timedelta(hours=5))},
    )
    check = by_id(run_doctor(probes=probes, offline=True))["auth.claude-agent-acp"]
    assert check.status == "pass"
    assert "valid until" in check.summary


def test_claude_api_key_overrides_oauth_token_note(tmp_path):
    probes = make_probes(
        tmp_path,
        env={"ANTHROPIC_API_KEY": OPENAI_KEY, "CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN},
    )
    report = run_doctor(probes=probes, offline=True)
    check = by_id(report)["auth.claude-agent-acp"]
    assert check.status == "pass"
    assert report.agents["claude-agent-acp"].effective.name == "ANTHROPIC_API_KEY"
    assert "overrides the OAuth token" in check.summary


def _codex_auth(**overrides) -> dict:
    data = {
        "auth_mode": "chatgpt",
        "OPENAI_API_KEY": None,
        "tokens": {
            "id_token": _jwt({"email": "person@example.com"}),
            "access_token": _jwt(
                {
                    "exp": int((NOW - timedelta(days=1)).timestamp()),
                    "https://api.openai.com/auth": {"chatgpt_plan_type": "pro"},
                }
            ),
            "refresh_token": REFRESH,
            "account_id": "acct-SECRET-0000-1111",
        },
        "last_refresh": (NOW - timedelta(days=2)).isoformat().replace("+00:00", "Z"),
    }
    data.update(overrides)
    return data


def test_codex_chatgpt_login_file_passes_even_with_expired_access_token(tmp_path):
    probes = make_probes(tmp_path, files={".codex/auth.json": _codex_auth()})
    report = run_doctor(probes=probes, offline=True)
    check = by_id(report)["auth.codex-acp"]
    assert check.status == "pass"
    assert "ChatGPT login" in check.summary
    assert "plan pro" in check.summary
    assert "last refreshed 2 days ago" in check.summary
    assert "expired" not in check.summary  # Codex refreshes it itself
    out = all_output(report)
    assert "person@example.com" not in out
    assert "acct-SECRET" not in out
    assert REFRESH not in out


def test_codex_api_key_overrides_login_with_note(tmp_path):
    probes = make_probes(
        tmp_path,
        env={"OPENAI_API_KEY": OPENAI_KEY},
        files={".codex/auth.json": _codex_auth()},
    )
    report = run_doctor(probes=probes, offline=True)
    auth = report.agents["codex-acp"]
    assert auth.effective.name == "OPENAI_API_KEY"
    assert auth.endpoints == ("https://api.openai.com/v1/",)
    assert (
        "unset them to run on the subscription"
        in by_id(report)["auth.codex-acp"].summary
    )
    assert OPENAI_KEY not in all_output(report)


def test_codex_inline_auth_json_env_is_detected(tmp_path):
    probes = make_probes(tmp_path, env={"CODEX_AUTH_JSON": json.dumps(_codex_auth())})
    report = run_doctor(probes=probes, offline=True)
    auth = report.agents["codex-acp"]
    assert auth.ready
    assert auth.effective.name == "CODEX_AUTH_JSON"
    assert auth.endpoints == ("https://chatgpt.com/", "https://auth.openai.com/")
    assert REFRESH not in all_output(report)


def test_codex_file_without_tokens_warns(tmp_path):
    probes = make_probes(
        tmp_path, files={".codex/auth.json": {"auth_mode": "chatgpt", "tokens": None}}
    )
    check = by_id(run_doctor(probes=probes, offline=True))["auth.codex-acp"]
    assert check.status == "warn"
    assert "codex login" in check.fix


def test_gemini_login_alone_is_not_a_working_credential(tmp_path):
    """A Google login cannot run Gemini: the mandatory LiteLLM proxy (#820)
    needs GEMINI_API_KEY. Doctor used to pass it, so `bench eval smoke` picked
    Gemini and failed with "LiteLLM route ... requires GEMINI_API_KEY"."""
    probes = make_probes(
        tmp_path,
        files={
            ".gemini/oauth_creds.json": {
                "access_token": "ya29.SECRETgeminiaccess",
                "refresh_token": REFRESH,
                "expiry_date": int((NOW - timedelta(hours=3)).timestamp() * 1000),
            }
        },
    )
    report = run_doctor(probes=probes, offline=True)
    check = by_id(report)["auth.gemini"]
    assert check.status == "warn"
    assert not report.agents["gemini"].ready
    assert "GEMINI_API_KEY" in check.summary
    assert "GEMINI_API_KEY" in check.fix
    assert "expired" not in check.summary
    assert "ya29" not in all_output(report)

    from benchflow.doctor_smoke import plan_smoke

    targets, skipped = plan_smoke(report)
    assert "gemini" not in [t.agent for t in targets]
    assert "GEMINI_API_KEY" in next(s for s in skipped if s.agent == "gemini").reason


def test_gemini_api_key_wins_over_the_login(tmp_path):
    probes = make_probes(
        tmp_path,
        env={"GEMINI_API_KEY": "AIzaSECRETgeminikeyvalue0123456789"},
        files={".gemini/oauth_creds.json": {"refresh_token": REFRESH}},
    )
    report = run_doctor(probes=probes, offline=True)
    assert report.agents["gemini"].ready
    assert by_id(report)["auth.gemini"].status == "pass"
    assert report.agents["gemini"].endpoints == (
        "https://generativelanguage.googleapis.com/",
    )
    assert "AIzaSECRET" not in all_output(report)


def test_bedrock_token_without_region_warns(tmp_path):
    probes = make_probes(
        tmp_path,
        env={"AWS_BEARER_TOKEN_BEDROCK": "bedrock-SECRET-token-value"},
    )
    report = run_doctor(probes=probes, offline=True)
    check = by_id(report)["auth.bedrock"]
    assert check.status == "warn"
    assert "AWS_REGION" in check.summary
    with_region = make_probes(
        tmp_path,
        env={
            "AWS_BEARER_TOKEN_BEDROCK": "bedrock-SECRET-token-value",
            "AWS_REGION": "us-west-2",
        },
    )
    report = run_doctor(probes=with_region, offline=True)
    assert by_id(report)["auth.bedrock"].status == "pass"
    assert report.agents["bedrock"].endpoints == (
        "https://bedrock-runtime.us-west-2.amazonaws.com/",
    )
    assert "bedrock-SECRET" not in all_output(report)


def test_provider_keys_report_missing_base_url_and_ignore_github_token(tmp_path):
    probes = make_probes(
        tmp_path,
        env={
            "DEEPSEEK_API_KEY": "ds-SECRET-value-123456",
            "AZURE_API_KEY": "az-SECRET-value-123456",
            "AZURE_API_ENDPOINT": "https://res.openai.azure.com/",
            "GITHUB_TOKEN": "ghp_SECRETvalue123456",
        },
    )
    report = run_doctor(probes=probes, offline=True)
    check = by_id(report)["auth.providers"]
    assert check.status == "pass"
    assert (
        "DEEPSEEK_API_KEY (env): deepseek/ also needs DEEPSEEK_BASE_URL"
        in check.summary
    )
    assert "AZURE_API_KEY (env): for azure-foundry-openai/" in check.summary
    assert "GITHUB_TOKEN" not in check.summary
    assert "SECRET" not in all_output(report)


def test_no_credentials_at_all_fails(tmp_path):
    report = run_doctor(probes=make_probes(tmp_path), offline=True)
    check = by_id(report)["auth.any"]
    assert check.status == "fail"
    assert "no credential found for any agent" in check.summary
    assert not report.ok


def test_expired_login_alone_warns_and_fails_the_aggregate(tmp_path):
    """An expired login is not a working credential. Its own line only warns
    (its refresh token might still work), but the aggregate check fails
    because no agent is known to be able to run."""
    probes = make_probes(
        tmp_path,
        files={".claude/.credentials.json": _claude_file(NOW - timedelta(days=1))},
    )
    report = run_doctor(probes=probes, offline=True)
    assert by_id(report)["auth.claude-agent-acp"].status == "warn"
    assert by_id(report)["auth.any"].status == "fail"
    assert "expired, incomplete or unreachable" in by_id(report)["auth.any"].summary


def test_dotenv_origin_is_reported(tmp_path):
    probes = make_probes(tmp_path, dotenv={"CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN})
    check = by_id(run_doctor(probes=probes, offline=True))["auth.claude-agent-acp"]
    assert check.summary.startswith("CLAUDE_CODE_OAUTH_TOKEN (.env)")


def test_from_host_merges_dotenv_under_process_env(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "GEMINI_API_KEY=from-dotenv-value\nOPENAI_API_KEY=dotenv-openai\n"
    )
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(dotenv))
    monkeypatch.setenv("OPENAI_API_KEY", "process-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    probes = DoctorProbes.from_host()
    assert probes.get("GEMINI_API_KEY") == "from-dotenv-value"
    assert probes.origin("GEMINI_API_KEY") == ".env"
    assert probes.get("OPENAI_API_KEY") == "process-openai"
    assert probes.origin("OPENAI_API_KEY") == "env"
    assert probes.get("ANTHROPIC_API_KEY") is None  # blank counts as unset


# ── Versions ────────────────────────────────────────────────────────────


def test_versions_show_sandbox_pin_and_host_cli(tmp_path):
    probes = make_probes(
        tmp_path,
        env={"CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN},
        binaries=("docker", "claude", "codex"),
        commands={
            ("claude", "--version"): CommandResult(0, "2.1.280 (Claude Code)\n", ""),
            ("codex", "--version"): CommandResult(0, "codex-cli 0.155.1\n", ""),
        },
    )
    checks = by_id(run_doctor(probes=probes, offline=True))
    claude = checks["version.claude-agent-acp"]
    assert claude.status == "pass"
    assert "@agentclientprotocol/claude-agent-acp@0.73.0" in claude.summary
    assert "host claude 2.1.280" in claude.summary
    assert "host codex 0.155.1" in checks["version.codex-acp"].summary
    assert "gemini CLI not found" in checks["version.gemini"].summary


def test_versions_flag_a_registry_override_of_a_pinned_agent(tmp_path, monkeypatch):
    from dataclasses import replace

    from benchflow.agents.registry import AGENTS, pinned_npm_package

    _, pinned = pinned_npm_package("codex-acp")
    cfg = AGENTS["codex-acp"]
    overridden = cfg.install_cmd.replace(f"@{pinned}", "@9.9.9")
    assert overridden != cfg.install_cmd, "override must replace the pinned version"
    monkeypatch.setitem(AGENTS, "codex-acp", replace(cfg, install_cmd=overridden))
    probes = make_probes(tmp_path, env={"CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN})
    check = by_id(run_doctor(probes=probes, offline=True))["version.codex-acp"]
    assert check.status == "warn"
    assert "codex-acp@9.9.9" in check.summary
    assert f"codex-acp@{pinned}" in check.summary


# ── Network ─────────────────────────────────────────────────────────────


def test_offline_makes_no_http_calls(tmp_path):
    calls: list = []
    probes = make_probes(
        tmp_path, env={"CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN}, calls=calls
    )
    report = run_doctor(probes=probes, offline=True)
    assert by_id(report)["network"].status == "skip"
    assert not [c for c in calls if c[0] == "http"]


def test_reachable_endpoint_line_does_not_show_the_http_status(tmp_path):
    """Regression test: "PASS reachable (HTTP 404)" read like an
    error. Any HTTP answer proves egress; the status stays in the details."""
    probes = make_probes(
        tmp_path,
        env={"CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN},
        http={"https://api.anthropic.com/": 404},
    )
    check = by_id(run_doctor(probes=probes))["net.api.anthropic.com"]
    assert check.status == "pass"
    assert check.summary == "reachable — Claude model API"
    assert check.details["http_status"] == 404


def test_required_endpoint_unreachable_fails(tmp_path):
    probes = make_probes(
        tmp_path,
        env={"CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN},
        http={"https://registry.npmjs.org/": OSError("Connection refused")},
    )
    report = run_doctor(probes=probes)
    check = by_id(report)["net.registry.npmjs.org"]
    assert check.status == "fail"
    assert "agent install inside the sandbox" in check.summary
    assert not report.ok


def test_model_endpoint_unreachable_blocks_that_agent_only(tmp_path):
    probes = make_probes(
        tmp_path,
        env={"CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN},
        files={".codex/auth.json": _codex_auth()},
        http={"https://api.anthropic.com/": TimeoutError("timed out")},
    )
    report = run_doctor(probes=probes)
    checks = by_id(report)
    assert checks["net.api.anthropic.com"].status == "warn"
    assert report.agent_blocked_by_network("claude-agent-acp") == [
        "https://api.anthropic.com/"
    ]
    assert report.agent_blocked_by_network("codex-acp") == []
    assert "auth.any" not in checks  # Codex can still run
    assert report.ok


def test_every_agent_blocked_fails_the_aggregate(tmp_path):
    probes = make_probes(
        tmp_path,
        env={"CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN},
        http={"https://api.anthropic.com/": TimeoutError("timed out")},
    )
    report = run_doctor(probes=probes)
    assert by_id(report)["auth.any"].status == "fail"


def test_json_report_shape(tmp_path):
    probes = make_probes(tmp_path, env={"CLAUDE_CODE_OAUTH_TOKEN": CLAUDE_TOKEN})
    data = run_doctor(probes=probes, offline=True).to_dict()
    assert data["ok"] is True
    assert data["sandbox"] == "docker"
    assert set(data["counts"]) == {"pass", "warn", "fail", "skip"}
    assert {"id", "group", "name", "status", "summary", "fix", "details"} <= set(
        data["checks"][0]
    )
    claude = data["agents"]["claude-agent-acp"]
    assert claude["ready"] is True
    assert claude["effective"]["name"] == "CLAUDE_CODE_OAUTH_TOKEN"
    assert CLAUDE_TOKEN not in json.dumps(data)


def test_redact_replaces_longest_values_first():
    secrets = doctor.secret_values(
        {
            "A_TOKEN": "abcdefgh",
            "B_KEY": "abcdefghijkl",
            "X": "abcdefghijkl-not-secret-name",
        }
    )
    assert secrets == ["abcdefghijkl", "abcdefgh"]
    assert doctor.redact("key=abcdefghijkl", secrets) == "key=***"


def test_secret_values_include_tokens_inside_inline_auth_json():
    inline = json.dumps({"tokens": {"access_token": REFRESH, "short": "x"}})
    secrets = doctor.secret_values({"CODEX_AUTH_JSON": inline})
    assert inline in secrets
    assert REFRESH in secrets
    assert "x" not in secrets
    assert doctor.redact(f"bad token {REFRESH}", secrets) == "bad token ***"
