"""`--source-env` on Prime Intellect verifiers v1 environments.

verifiers was rewritten (Taskset / Harness / Env); from the 0.3.2 dev line
its ``vf-eval`` is a new CLI (``verifiers.v1.cli.eval.main``) that rejects
every flag BenchFlow passed (``--env-args``, ``--num-examples``,
``--sampling-args``, ``--save-results``, ``--disable-tui``: "Extra inputs are
not permitted"), writes ``traces.jsonl`` instead of a stdout summary, and
uploads the run to Prime unless ``--no-push``. The legacy 0.3.1 ``vf-eval``
cannot load a v1 taskset at all ("does not expose load_environment"). BenchFlow now reads which CLI the
installed verifiers ships and speaks it; v1 runs never push.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.hosted_env import (
    HostedEnvRef,
    HostedEnvRunConfig,
    build_vf_eval_command,
    read_v1_traces,
    run_hosted_env,
    vf_eval_cli,
)

FIXTURE = Path(__file__).parent / "fixtures" / "hosted_env" / "v1_traces.jsonl"


def _dist_info(venv: Path, version: str, entry: str) -> None:
    info = (
        venv / "lib" / "python3.12" / "site-packages" / f"verifiers-{version}.dist-info"
    )
    info.mkdir(parents=True)
    (info / "entry_points.txt").write_text(f"[console_scripts]\nvf-eval = {entry}\n")
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.4\nName: verifiers\nVersion: {version}\n"
    )


def test_cli_is_read_from_the_installed_entry_point(tmp_path: Path) -> None:
    legacy, v1, none = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    _dist_info(legacy, "0.3.1", "verifiers.legacy.scripts.eval:main")
    _dist_info(v1, "0.3.2.dev137", "verifiers.v1.cli.eval.main:main")
    assert vf_eval_cli(legacy) == ("legacy", "0.3.1")
    assert vf_eval_cli(v1) == ("v1", "0.3.2.dev137")
    assert vf_eval_cli(none) == ("legacy", None)


def _config(**kw) -> HostedEnvRunConfig:
    base = dict(
        source_env=HostedEnvRef.parse("primeintellect/bf-echo", version="0.1.0"),
        model="gpt-4.1-mini",
        env_args={"word": "OK", "dataset_name": "x/y", "shuffle": True, "n": 3},
        concurrency=2,
        num_examples=1,
        rollouts_per_example=4,
        max_tokens=64,
        temperature=0.5,
        sampling_args={"reasoning_effort": "low"},
    )
    base.update(kw)
    return HostedEnvRunConfig(**base)


def test_v1_command_uses_v1_flags_and_never_pushes(tmp_path: Path) -> None:
    cmd = build_vf_eval_command(
        _config(api_base_url="http://127.0.0.1:1/v1", api_key_var="STUB_KEY"),
        vf_eval="/v/bin/vf-eval",
        output_dir=tmp_path / "out",
        cli="v1",
    )
    assert cmd[:2] == ["/v/bin/vf-eval", "bf-echo"]
    pairs = dict(zip(cmd[2::2], cmd[3::2], strict=False))
    assert pairs["-n"] == "1" and pairs["-r"] == "4" and pairs["-c"] == "2"
    assert pairs["-m"] == "openai/gpt-4.1-mini"
    assert pairs["--sampling.max-tokens"] == "64"
    assert pairs["--sampling.temperature"] == "0.5"
    assert pairs["--sampling.reasoning-effort"] == "low"
    assert pairs["--env.taskset.word"] == "OK"
    assert pairs["--env.taskset.dataset-name"] == "x/y"
    assert pairs["--env.taskset.shuffle"] == "True"
    assert pairs["--env.taskset.n"] == "3"
    assert pairs["--client.base-url"] == "http://127.0.0.1:1/v1"
    assert pairs["--client.api-key-var"] == "STUB_KEY"
    assert pairs["--output-dir"] == str(tmp_path / "out")
    assert pairs["--run.dir"] == "benchflow"
    assert cmd[-2:] == ["--no-push", "--no-rich"]
    for v0_flag in (
        "--env-args",
        "--num-examples",
        "--sampling-args",
        "--save-results",
    ):
        assert v0_flag not in cmd


def test_legacy_command_is_unchanged(tmp_path: Path) -> None:
    cmd = build_vf_eval_command(
        _config(), vf_eval="/v/bin/vf-eval", output_dir=tmp_path / "o", cli="legacy"
    )
    assert "--env-args" in cmd and "--save-results" in cmd and "--disable-tui" in cmd
    assert "--no-push" not in cmd
    with_url = build_vf_eval_command(
        _config(api_base_url="http://h/v1", api_key_var="K"),
        vf_eval="vf",
        output_dir=tmp_path,
        cli="legacy",
    )
    assert with_url[with_url.index("--api-base-url") + 1] == "http://h/v1"
    assert with_url[with_url.index("--api-key-var") + 1] == "K"


def test_v1_traces_give_reward_tokens_and_errors(tmp_path: Path) -> None:
    ok = read_v1_traces(FIXTURE)
    assert ok.reward == 1.0 and ok.rollouts == 1 and ok.errored == 0
    assert ok.total_tokens == 6 and ok.error is None

    row = json.loads(FIXTURE.read_text().splitlines()[0])
    bad = dict(row, ok=False, errors=[{"type": "ProviderError", "message": "boom"}])
    half = dict(row)
    half["traces"] = [
        dict(row["traces"][0], rewards={"a": {"score": 0.5, "weight": 1.0}})
    ]
    path = tmp_path / "traces.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in (row, bad, half)) + "\n")
    mixed = read_v1_traces(path)
    # Errored episodes are left out of the mean, not scored 0.
    assert mixed.reward == pytest.approx(0.75)
    assert mixed.rollouts == 2 and mixed.errored == 1
    assert mixed.error is None

    path.write_text(json.dumps(bad) + "\n")
    none = read_v1_traces(path)
    assert none.reward is None and none.error and "ProviderError" in none.error


def test_run_hosted_env_speaks_v1(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        cmd = [str(c) for c in cmd]
        calls.append(cmd)
        if cmd[1:3] == ["venv", "--python"]:
            _dist_info(Path(cmd[-1]), "0.3.2.dev137", "verifiers.v1.cli.eval.main:main")
        if cmd[0].endswith("vf-eval"):
            out = Path(cmd[cmd.index("--output-dir") + 1]) / "benchflow"
            out.mkdir(parents=True)
            shutil.copy(FIXTURE, out / "traces.jsonl")
            return SimpleNamespace(returncode=0, stdout="rollout done\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("benchflow.hosted_env.shutil.which", lambda b: f"/bin/{b}")
    monkeypatch.setattr("benchflow.hosted_env.subprocess.run", fake_run)
    result = run_hosted_env(
        _config(jobs_dir=tmp_path, verifiers_version="0.3.2.dev137")
    )
    install = calls[1]
    assert "bf_echo==0.1.0" in install and "verifiers==0.3.2.dev137" in install
    assert "--no-push" in calls[2]
    assert result.reward == 1.0 and result.error is None
    payload = json.loads((result.run_dir / "result.json").read_text())
    assert payload["rewards"] == {"reward": 1.0}
    assert payload["agent_result"]["total_tokens"] == 6
    assert json.loads((result.run_dir / "prompts.json").read_text()) == [
        "Reply with the word OK."
    ]
    hosted = json.loads((result.run_dir / "hosted_env" / "hosted_run.json").read_text())
    assert hosted["vf_eval_cli"] == "v1"
    assert hosted["verifiers_version"] == "0.3.2.dev137"
    assert hosted["v1_rollouts"] == 1 and hosted["v1_errored"] == 0


def test_cli_passes_the_pin_and_endpoint(tmp_path: Path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from benchflow.cli.main import app
    from benchflow.hosted_env import HostedEnvRunResult

    seen: dict[str, HostedEnvRunConfig] = {}

    def fake(config: HostedEnvRunConfig) -> HostedEnvRunResult:
        seen["config"] = config
        return HostedEnvRunResult(
            source_env=config.source_env,
            run_dir=tmp_path,
            command=["vf-eval"],
            returncode=0,
            stdout="",
            stderr="",
            model=config.model,
            normalized_model=config.model,
            reward=1.0,
        )

    monkeypatch.setattr("benchflow.hosted_env.run_hosted_env", fake)
    out = CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--source-env",
            "primeintellect/bf-echo",
            "--source-env-version",
            "0.1.0",
            "--source-env-verifiers-version",
            "0.3.2.dev137",
            "--source-env-base-url",
            "http://127.0.0.1:1/v1",
            "--source-env-api-key-var",
            "STUB_KEY",
            "--model",
            "stub-model",
            "--jobs-dir",
            str(tmp_path),
        ],
    )
    assert out.exit_code == 0, out.output
    config = seen["config"]
    assert config.verifiers_version == "0.3.2.dev137"
    assert config.api_base_url == "http://127.0.0.1:1/v1"
    assert config.api_key_var == "STUB_KEY"
