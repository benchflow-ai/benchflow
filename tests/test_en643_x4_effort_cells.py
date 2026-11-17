"""X4's cells, each as X4 ran it.

X4 ran ``bench run tasks/hello -a <agent> --harness acp --sandbox docker
--retry-attempts 0 --max-cost-usd 0.25 -m xiaomi/mimo-v2.6-flash
--reasoning-effort <level>`` on rc 5d64b0d44 and rel 53cad31d9, each once with
``--dry-run`` and once without:

- ``*-claude-agent-acp-acp-mimo-hello-{xhigh,max}-{dry,run}``: the dry run
  exited 0, and every upstream request carried ``reasoning_effort: high``. rc
  failed the run check only after the solve; rel exited 0.
- ``*-{opencode,openhands}-acp-mimo-hello-{low,medium,high,max}-*`` and
  ``*-{pi-acp-acp,mimo-acp,harvey-lab-harness-acp,deepagents-acp}-mimo-hello-high-*``:
  the dry run exited 0, and the run failed after the sandbox and the agent's
  install ("does not declare or advertise an ACP effort config option").
- ``rc-codex-acp-acp-codex-hello-minimal-*`` (gpt-6.1-sol): the same, after
  codex-acp's ``session/new``.
- ``*-codex-acp-acp-astra-hello-default-*`` (``-m 'gpt-6-astra[ultra]'``):
  the dry run said "the agent's default", and the session ran medium.

MiMo applies no reasoning effort: measured 2026-11-15 on mimo-v2.6-pro at the
xiaomi/ route's endpoint, reasoning tokens were flat across low, high and max,
xhigh was refused with HTTP 400, and ``thinking.type: disabled`` gave 0
(``effort_request._NO_EFFORT_ROUTES``). So every level asked of the xiaomi/
route is refused before launch, by the dry run and the run alike.

Made-up keys only; ``Evaluation.run`` is replaced, so nothing starts a sandbox,
and the gateway case calls a local stub.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

MIMO = "xiaomi/mimo-v2.6-flash"
LEVELS = ("low", "medium", "high", "xhigh", "max")
#: The variants codex-acp 0.159.3 offered in X4
#: (rc-codex-acp-acp-codex-hello-minimal-run: "it offers: low, medium, high,
#: xhigh, max, ultra"); the session opens on Sol at low.
X4_CODEX_VARIANTS = ("low", "medium", "high", "xhigh", "max", "ultra")
_FAKE_CODEX_LOGIN = json.dumps(
    {
        "auth_mode": "chatgpt",
        "tokens": {
            "id_token": "fake.not-a.token",
            "access_token": "fake-not-a-token",
            "refresh_token": "fake-not-a-token",
            "account_id": "fake-account",
        },
    }
)

#: (agent, model, level) of each X4 cell its dry run passed and its run
#: refused late, or ran at another level.
X4_LATE_CELLS = [
    *(("claude-agent-acp", MIMO, level) for level in ("xhigh", "max")),
    *(("opencode", MIMO, level) for level in ("low", "medium", "high", "max")),
    *(("openhands", MIMO, level) for level in ("low", "medium", "high", "max")),
    *(
        (agent, MIMO, "high")
        for agent in ("pi-acp", "mimo", "harvey-lab-harness", "deepagents")
    ),
    ("codex-acp", "gpt-6.1-sol", "minimal"),
]


def _x4_task(root):
    """X4's tasks/hello, as it ran."""
    task = root / "hello"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "solution").mkdir()
    (task / "task.toml").write_text(
        'version = "1.0"\n[metadata]\nauthor_name = "X4"\ndifficulty = "easy"\n'
        'category = "demo"\n[agent]\ntimeout_sec = 180\n[verifier]\n'
        "timeout_sec = 15\n[environment]\ncpus = 1\nmemory_mb = 2048\n"
    )
    (task / "instruction.md").write_text(
        "Write exactly ok followed by a newline to /app/answer.txt using a shell "
        "command, then finish.\n"
    )
    (task / "environment" / "Dockerfile").write_text(
        "FROM python:3.12-slim\nENV HARBOR_TELEMETRY=off\nWORKDIR /app\n"
    )
    (task / "tests" / "test.sh").write_text(
        "#!/bin/bash\ngrep -qx ok /app/answer.txt && r=1 || r=0\n"
        "echo $r > /logs/verifier/reward.txt\n"
    )
    (task / "solution" / "solve.sh").write_text(
        "#!/bin/bash\necho ok > /app/answer.txt\n"
    )
    return task


def _x4_bench_run(tmp_path, monkeypatch, agent, model, level, *, dry_run):
    """X4's command through the real CLI. ``Evaluation.run`` keeps the config
    and reports one pass: a cell refused before launch never reaches it."""
    from typer.testing import CliRunner

    from benchflow.cli.main import app
    from benchflow.evaluation import Evaluation
    from tests.test_eval_artifact_cli import _write_rollout

    root = tmp_path / ("dry" if dry_run else "run")
    task = _x4_task(root / "tasks")
    launched: list = []

    async def fake_run(self):
        launched.append(self._config)
        _write_rollout(self._jobs_dir / self._job_name / "hello__x4", "hello")
        return SimpleNamespace(
            job_name=self._job_name,
            passed=1,
            failed=0,
            errored=0,
            verifier_errored=0,
            total=1,
            score=1.0,
            score_excl_errors=1.0,
        )

    monkeypatch.setattr(Evaluation, "run", fake_run)
    monkeypatch.setattr("benchflow.cli.doctor.eval_preflight", lambda **kw: None)
    monkeypatch.setattr(
        "benchflow.cli.doctor.start_docker_preflight",
        lambda *a, **kw: None,
        raising=False,
    )
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "no.env"))
    argv = [
        "run",
        str(task),
        "-a",
        agent,
        "--harness",
        "acp",
        "--sandbox",
        "docker",
        "--jobs-dir",
        str(root / "jobs"),
        "--job-name",
        "x4",
        "--retry-attempts",
        "0",
        "--max-cost-usd",
        "0.25",
        "--agent-env",
        "HARBOR_TELEMETRY=off",
        "-m",
        model,
    ]
    if level:
        argv += ["--reasoning-effort", level]
    if dry_run:
        argv.append("--dry-run")
    result = CliRunner().invoke(
        app,
        argv,
        env={
            "COLUMNS": "200",
            "HARBOR_TELEMETRY": "off",
            "XIAOMI_API_KEY": "fake-not-a-key",
            "XIAOMI_BASE_URL": "http://127.0.0.1:9/v1",
            "CODEX_AUTH_JSON": _FAKE_CODEX_LOGIN,
        },
    )
    return result, launched


def _refusal(output: str) -> str:
    """The refusal and its next step, as one line of words."""
    text = " ".join(output.split())
    start = text.find("Refused before launch:")
    return text[start:] if start >= 0 else ""


@pytest.mark.parametrize(("agent", "model", "level"), X4_LATE_CELLS)
def test_an_x4_cell_is_refused_before_launch_by_the_dry_run_and_the_run_alike(
    tmp_path, monkeypatch, agent, model, level
):
    dry, dry_launched = _x4_bench_run(
        tmp_path, monkeypatch, agent, model, level, dry_run=True
    )
    run, launched = _x4_bench_run(
        tmp_path, monkeypatch, agent, model, level, dry_run=False
    )
    assert dry.exit_code == 2, dry.output
    assert run.exit_code == 2, run.output
    assert not launched and not dry_launched
    assert _refusal(dry.output), dry.output
    assert _refusal(dry.output) == _refusal(run.output)
    if model == MIMO:
        assert "thinking.type" in _refusal(run.output)
    else:
        assert "does not take reasoning effort minimal" in _refusal(run.output)


@pytest.mark.parametrize("level", ("low", "medium", "high"))
def test_x4_claude_mimo_levels_that_ran_alike_are_refused_too(
    tmp_path, monkeypatch, level
):
    """X4's low, medium and high Claude Code cells on MiMo sent the level as
    asked, but MiMo runs them alike: refused, as xhigh and max are."""
    dry, _ = _x4_bench_run(
        tmp_path, monkeypatch, "claude-agent-acp", MIMO, level, dry_run=True
    )
    run, launched = _x4_bench_run(
        tmp_path, monkeypatch, "claude-agent-acp", MIMO, level, dry_run=False
    )
    assert (dry.exit_code, run.exit_code, launched) == (2, 2, [])
    assert _refusal(dry.output) == _refusal(run.output)
    assert f"--effort {level} would not run as {level}" in _refusal(run.output)


@pytest.mark.parametrize("level", ("ultra", "minimal", "none"))
def test_a_level_claude_code_lacks_is_refused_for_mimo_naming_its_control(
    tmp_path, monkeypatch, level
):
    """X4's ``*-claude-agent-acp-acp-mimo-hello-ultra-{dry,run}`` cells were
    refused with Claude Code's own levels (low to max), every one of which
    MiMo would refuse too: the refusal names MiMo's control instead."""
    dry, _ = _x4_bench_run(
        tmp_path, monkeypatch, "claude-agent-acp", MIMO, level, dry_run=True
    )
    run, launched = _x4_bench_run(
        tmp_path, monkeypatch, "claude-agent-acp", MIMO, level, dry_run=False
    )
    assert (dry.exit_code, run.exit_code, launched) == (2, 2, []), run.output
    assert _refusal(dry.output) == _refusal(run.output)
    assert "thinking.type" in _refusal(run.output)
    assert "it takes low, medium, high, xhigh, max" not in _refusal(run.output)


@pytest.mark.parametrize(
    ("agent", "level"),
    [
        *(("antigravity", level) for level in ("none", "minimal", "xhigh", "max")),
        ("antigravity", "ultra"),
        ("muse-code", "none"),
    ],
)
def test_an_agent_s_own_level_check_does_not_hide_mimo_s_control(
    tmp_path, monkeypatch, agent, level
):
    """Where an agent refuses a level itself (the Antigravity CLI's three
    tiers, Muse without none), a MiMo run is refused for MiMo's reason first."""
    from benchflow.effort_request import EffortRequestError, settle_effort

    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "no.env"))
    with pytest.raises(EffortRequestError, match=r"thinking\.type"):
        settle_effort(agent, MIMO, level, {"XIAOMI_API_KEY": "fake-not-a-key"})


def test_x4_mimo_without_an_effort_still_runs(tmp_path, monkeypatch):
    """No level asked: nothing to refuse (X4's default cells)."""
    run, launched = _x4_bench_run(
        tmp_path, monkeypatch, "claude-agent-acp", MIMO, None, dry_run=False
    )
    assert run.exit_code == 0, run.output
    assert len(launched) == 1


class _CodexSession:
    """codex-acp's session/new answer in X4."""

    model_state: ClassVar[dict[str, Any]] = {
        "currentModelId": "gpt-6.1-sol[low]",
        "availableModels": [
            {"modelId": f"{model}[{effort}]"}
            for model in ("gpt-6.1-sol", "gpt-6-astra")
            for effort in X4_CODEX_VARIANTS
        ],
    }


def test_x4_astra_ultra_suffix_reaches_the_session_as_ultra(tmp_path, monkeypatch):
    """X4's rc-codex-acp-acp-astra-hello-default cells: ``-m
    'gpt-6-astra[ultra]'`` passed the dry run as "the agent's default", and
    the session set gpt-6-astra[medium]."""
    from benchflow.acp.runtime import _select_acp_model_id

    dry, _ = _x4_bench_run(
        tmp_path, monkeypatch, "codex-acp", "gpt-6-astra[ultra]", None, dry_run=True
    )
    assert dry.exit_code == 0, dry.output
    card = json.loads(dry.stdout[dry.stdout.index("{") :])["run_card"]
    assert (card["model"]["effort"], card["model"]["effort_source"]) == (
        "ultra",
        "model[effort]",
    )
    run, launched = _x4_bench_run(
        tmp_path, monkeypatch, "codex-acp", "gpt-6-astra[ultra]", None, dry_run=False
    )
    assert run.exit_code == 0, run.output
    (cfg,) = launched
    assert (cfg.model, cfg.reasoning_effort) == ("gpt-6-astra", "ultra")
    # The session step the run takes with that config, on X4's variants.
    assert (
        _select_acp_model_id(
            cfg.model, "codex-acp", _CodexSession(), cfg.reasoning_effort
        )
        == "gpt-6-astra[ultra]"
    )
    # A suffix that reaches the session itself is read there too.
    assert (
        _select_acp_model_id("gpt-6-astra[ultra]", "codex-acp", _CodexSession(), None)
        == "gpt-6-astra[ultra]"
    )


# --- the gateway half of X4's MiMo cells -----------------------------------

_TOOLS = [
    {
        "name": "Bash",
        "description": "Run a command",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
        },
    }
]
_CHAT_REPLY = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "mimo-v2.6-flash",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
}


class _ChatUpstream(BaseHTTPRequestHandler):
    seen: ClassVar[list[dict[str, Any]]] = []

    def log_message(self, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append({"path": self.path, "body": body})
        data = json.dumps(_CHAT_REPLY).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module")
def chat_upstream():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ChatUpstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.mark.parametrize("level", LEVELS)
def test_the_gateway_sends_the_chat_completions_route_the_level_it_is_given(
    chat_upstream, monkeypatch, level
):
    """X4's MiMo cells went through LiteLLM's Messages-to-chat-completions
    adapter, which lowered xhigh and max to high (the observer recorded
    ``reasoning_effort: high``). With the proxy's patches loaded, a level that
    reaches the adapter goes upstream as given. The plan now refuses any level
    on xiaomi/ (above), so none reaches MiMo; the adapter is the one every
    OpenAI-compatible route's Claude Code calls go through."""
    import litellm

    import benchflow.providers.litellm_bedrock_patch  # noqa: F401  # the proxy's sitecustomize patches
    from benchflow.providers.litellm_config import (
        litellm_proxy_config,
        resolve_litellm_route,
    )

    # The gateway's own config for the route: its model entry and the
    # litellm_settings the proxy loads (drop_params, and the Messages bridge
    # forced onto /chat/completions).
    env = {"XIAOMI_API_KEY": "fake-not-a-key", "XIAOMI_BASE_URL": chat_upstream}
    config = litellm_proxy_config(
        resolve_litellm_route(MIMO, env), master_key="fake-not-a-key"
    )
    settings = config["litellm_settings"]
    assert isinstance(settings, dict)
    for name in ("drop_params", "use_chat_completions_url_for_anthropic_messages"):
        monkeypatch.setattr(litellm, name, settings[name])
    (entry, *_) = config["model_list"]
    assert isinstance(entry, dict)
    params = dict(entry["litellm_params"])
    params.update(api_base=f"{chat_upstream}/v1", api_key="fake-not-a-key")
    router = litellm.Router(model_list=[{"model_name": "m", "litellm_params": params}])
    _ChatUpstream.seen.clear()
    asyncio.run(
        router.aanthropic_messages(
            model="m",
            messages=[{"role": "user", "content": "Write ok to /app/answer.txt."}],
            max_tokens=1024,
            thinking={"type": "adaptive"},
            output_config={"effort": level},
            tools=_TOOLS,
        )
    )
    assert _ChatUpstream.seen, "nothing reached the upstream"
    sent = _ChatUpstream.seen[-1]
    assert sent["path"].endswith("/chat/completions"), sent["path"]
    assert sent["body"].get("reasoning_effort") == level, sent["body"]
