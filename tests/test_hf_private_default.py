"""audit, 2026-11-13: what BenchFlow creates on Hugging Face is private
unless --public.

``publish_folder_to_hf`` and ``publish_file_to_hf`` called ``create_repo``
with no ``private``, so a new repo took the Hub's default (public unless the
organisation's default is private), and ``publish_folder_to_bucket`` created
its bucket with ``private=False``; no flag changed either. Every create call
now sends ``private=True`` unless public is asked for (``--public``,
``private=False``). A repo or bucket that already exists keeps its visibility:
nothing that changes one is called, and the publish line claims none. Fakes
stand in for ``huggingface_hub`` throughout; nothing reaches the Hub.
"""

from __future__ import annotations

import io
import json
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.evaluation import Evaluation
from benchflow.publish import huggingface as hf

runner = CliRunner()


def _http_error(status: int) -> HfHubHTTPError:
    request = httpx.Request("POST", "https://hub.invalid/api")
    return HfHubHTTPError(str(status), response=httpx.Response(status, request=request))


class StrictHfApi:
    """``HfApi`` with only the calls a publish makes; any other use fails."""

    def __init__(self, create_status: int | None = None, info_ok: bool = True):
        self.creates: list[tuple[tuple, dict]] = []
        self.calls: list[str] = []
        self.create_status = create_status
        self.info_ok = info_ok

    def __getattr__(self, name: str):
        raise AssertionError(f"publish used HfApi.{name}")

    def create_repo(self, *args, **kwargs):
        self.calls.append("create_repo")
        self.creates.append((args, kwargs))
        if self.create_status is not None:
            raise _http_error(self.create_status)

    def repo_info(self, *args, **kwargs):
        self.calls.append("repo_info")
        if not self.info_ok:
            raise _http_error(404)
        return SimpleNamespace(id=args[0] if args else kwargs.get("repo_id"))

    def upload_folder(self, **kwargs):
        self.calls.append("upload_folder")
        root = Path(kwargs["folder_path"])
        self.uploaded = sorted(
            p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
        )
        return SimpleNamespace(commit_url="https://hub.invalid/commit/1")

    def upload_file(self, **kwargs):
        self.calls.append("upload_file")
        return SimpleNamespace(commit_url="https://hub.invalid/commit/1")


def _fake_api(monkeypatch, **kwargs) -> StrictHfApi:
    api = StrictHfApi(**kwargs)
    monkeypatch.setattr(hf, "_require_hf_api", lambda: api)
    return api


def _fake_hub(monkeypatch, *, exists: bool = False) -> dict[str, list]:
    """A stand-in ``huggingface_hub`` that has create_bucket and sync_bucket
    only; any other name a publish reads from it is recorded."""
    seen: dict[str, list] = {"creates": [], "synced": [], "other": []}

    def create_bucket(bucket_id, **kwargs):
        seen["creates"].append((bucket_id, kwargs))
        if exists:
            raise _http_error(409)

    def sync_bucket(source, remote):
        seen["synced"].append(remote)

    def other(name: str):
        if not name.startswith("__"):
            seen["other"].append(name)
        raise AttributeError(name)

    hub = types.ModuleType("huggingface_hub")
    errors = types.ModuleType("huggingface_hub.errors")
    errors.HfHubHTTPError = HfHubHTTPError
    hub.errors = errors
    hub.create_bucket = create_bucket
    hub.sync_bucket = sync_bucket
    hub.__getattr__ = other
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    return seen


def _folder(tmp_path: Path) -> Path:
    folder = tmp_path / "job"
    folder.mkdir()
    (folder / "result.json").write_text("{}")
    return folder


def _publish_repo(kind: str, tmp_path: Path, **kwargs: Any) -> hf.HfPublishResult:
    folder = _folder(tmp_path)
    if kind == "folder":
        return hf.publish_folder_to_hf(
            folder, repo_id="org/runs", path_in_repo="job", **kwargs
        )
    return hf.publish_file_to_hf(
        folder / "result.json",
        repo_id="org/runs",
        path_in_repo="job/result.json",
        **kwargs,
    )


def _asked(public: bool | None) -> dict[str, bool]:
    """No argument (the default), or private= as the CLI's --public sets it."""
    return {} if public is None else {"private": not public}


# The library: each create call.


@pytest.mark.parametrize("public", [None, False, True])
@pytest.mark.parametrize("kind", ["folder", "file"])
def test_repo_create_sends_private_unless_public_is_asked(
    tmp_path, monkeypatch, kind, public
):
    api = _fake_api(monkeypatch)
    result = _publish_repo(kind, tmp_path, **_asked(public))
    assert api.creates == [
        (("org/runs",), {"repo_type": "dataset", "private": not public})
    ]
    assert result.created == ("public" if public else "private")
    assert result.visibility == f"new {result.created} dataset repo"


@pytest.mark.parametrize("public", [None, False, True])
def test_bucket_create_sends_private_unless_public_is_asked(
    tmp_path, monkeypatch, public
):
    seen = _fake_hub(monkeypatch)
    result = hf.publish_folder_to_bucket(
        _folder(tmp_path), bucket_id="org/runs", path_in_repo="job", **_asked(public)
    )
    assert seen["creates"] == [("org/runs", {"private": not public})]
    assert seen["synced"] == ["hf://buckets/org/runs/job"]
    assert seen["other"] == []
    assert result.created == ("public" if public else "private")
    assert result.visibility == f"new {result.created} bucket"


@pytest.mark.parametrize("public", [None, True])
@pytest.mark.parametrize("kind", ["folder", "file"])
def test_existing_repo_keeps_its_visibility(tmp_path, monkeypatch, kind, public):
    """The Hub refuses the create (409): nothing else is asked of it, and the
    result claims no visibility."""
    api = _fake_api(monkeypatch, create_status=409)
    result = _publish_repo(kind, tmp_path, **_asked(public))
    assert api.calls == ["create_repo", f"upload_{kind}"]
    assert result.created is None
    assert result.visibility == "existing dataset repo, its visibility left as it is"


@pytest.mark.parametrize("kind", ["folder", "file"])
def test_existing_repo_without_create_right_is_still_written(
    tmp_path, monkeypatch, kind
):
    """A 403 on create, for a repo that is there: what exist_ok=True allowed."""
    api = _fake_api(monkeypatch, create_status=403)
    result = _publish_repo(kind, tmp_path)
    assert api.calls == ["create_repo", "repo_info", f"upload_{kind}"]
    assert result.created is None


@pytest.mark.parametrize("kind", ["folder", "file"])
def test_create_refused_for_a_missing_repo_raises_and_uploads_nothing(
    tmp_path, monkeypatch, kind
):
    api = _fake_api(monkeypatch, create_status=403, info_ok=False)
    with pytest.raises(HfHubHTTPError) as raised:
        _publish_repo(kind, tmp_path)
    assert raised.value.response.status_code == 403
    assert api.calls == ["create_repo", "repo_info"]


@pytest.mark.parametrize("public", [None, True])
def test_existing_bucket_keeps_its_visibility(tmp_path, monkeypatch, public):
    seen = _fake_hub(monkeypatch, exists=True)
    result = hf.publish_folder_to_bucket(
        _folder(tmp_path), bucket_id="org/runs", **_asked(public)
    )
    assert seen["creates"] == [("org/runs", {"private": not public})]
    assert seen["synced"] == ["hf://buckets/org/runs"]
    assert seen["other"] == []
    assert result.created is None
    assert result.visibility == "existing bucket, its visibility left as it is"


@pytest.mark.parametrize("exists", [False, True])
@pytest.mark.parametrize("kind", ["folder", "file"])
def test_public_read_check_names_a_repo_this_publish_made_private(
    tmp_path, monkeypatch, kind, exists
):
    _fake_api(monkeypatch, create_status=409 if exists else None)
    refused = httpx.Response(401, request=httpx.Request("GET", "https://hub.invalid"))
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: refused)
    monkeypatch.setattr(httpx, "head", lambda *a, **kw: refused)
    with pytest.raises(ValueError, match="HF public read check failed") as raised:
        _publish_repo(kind, tmp_path, public_read_check=True)
    named = "this publish created org/runs private" in str(raised.value)
    assert named is not exists


# The CLI: bench eval run, bench eval create, bench run and a --config run.


def _tasks(tmp_path: Path) -> Path:
    task = tmp_path / "tasks" / "task-a"
    task.mkdir(parents=True)
    (task / "task.toml").write_text('version = "1.0"\n', encoding="utf-8")
    return task.parent


def _fake_eval(monkeypatch) -> None:
    async def fake_run(self):
        trial = Path(self._jobs_dir) / self._job_name / "task-a__abc"
        trial.mkdir(parents=True)
        (trial / "result.json").write_text(
            json.dumps({"task_name": "task-a", "rewards": {"reward": 1.0}})
        )
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


def _eval_args(route: str, tmp_path: Path) -> list[str]:
    tasks, jobs = _tasks(tmp_path), tmp_path / "jobs"
    if route == "bench eval run --config":
        config = tmp_path / "run.yaml"
        config.write_text(f"tasks_dir: {tasks}\njobs_dir: {jobs}\nagent: oracle\n")
        return ["eval", "run", "--config", str(config)]
    head = {
        "bench eval run": ["eval", "run", "--tasks-dir", str(tasks)],
        "bench eval create": ["eval", "create", "--tasks-dir", str(tasks)],
        "bench run": ["run", str(tasks)],
    }[route]
    return [*head, "--agent", "oracle", "--jobs-dir", str(jobs)]


ROUTES = ["bench eval run", "bench eval create", "bench run", "bench eval run --config"]


def _said(result) -> str:
    return " ".join(result.output.split())


@pytest.mark.parametrize("public", [False, True])
@pytest.mark.parametrize("route", ROUTES)
def test_eval_publish_hf_creates_private_unless_public(
    tmp_path, monkeypatch, route, public
):
    _fake_eval(monkeypatch)
    api = _fake_api(monkeypatch)
    args = [*_eval_args(route, tmp_path), "--publish-hf", "org/runs"]
    result = runner.invoke(app, [*args, *(["--public"] if public else [])])
    assert result.exit_code == 0, result.output
    assert api.creates == [
        (("org/runs",), {"repo_type": "dataset", "private": not public})
    ]
    kind = "public" if public else "private"
    assert re.search(
        r"Published HF artifacts: https://huggingface\.co/datasets/org/runs/tree/"
        rf"main/\S+ \(new {kind} dataset repo\)",
        _said(result),
    ), result.output


@pytest.mark.parametrize("public", [False, True])
@pytest.mark.parametrize("route", ROUTES)
def test_eval_publish_bucket_creates_private_unless_public(
    tmp_path, monkeypatch, route, public
):
    _fake_eval(monkeypatch)
    seen = _fake_hub(monkeypatch)
    args = [*_eval_args(route, tmp_path), "--publish-bucket", "org/runs"]
    result = runner.invoke(app, [*args, *(["--public"] if public else [])])
    assert result.exit_code == 0, result.output
    assert seen["creates"] == [("org/runs", {"private": not public})]
    assert seen["other"] == []
    kind = "public" if public else "private"
    assert re.search(
        r"Published HF bucket: https://huggingface\.co/buckets/org/runs/tree/\S+ "
        rf"\(new {kind} bucket\)",
        _said(result),
    ), result.output


@pytest.mark.parametrize("public", [False, True])
def test_eval_trials_publish_each_trial_job_as_asked(tmp_path, monkeypatch, public):
    """--trials runs one job per trial and publishes each: every create asks
    for the same visibility."""
    _fake_eval(monkeypatch)
    seen = _fake_hub(monkeypatch)
    args = [*_eval_args("bench eval run", tmp_path), "--trials", "2"]
    args += ["--publish-bucket", "org/runs", *(["--public"] if public else [])]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert seen["creates"] == [("org/runs", {"private": not public})] * 2
    kind = "public" if public else "private"
    assert _said(result).count(f"(new {kind} bucket)") == 2


@pytest.mark.parametrize("flag", ["--publish-hf", "--publish-bucket"])
def test_eval_publish_to_an_existing_repo_or_bucket_claims_no_visibility(
    tmp_path, monkeypatch, flag
):
    _fake_eval(monkeypatch)
    if flag == "--publish-hf":
        api = _fake_api(monkeypatch, create_status=409)
        kind = "dataset repo"
    else:
        seen = _fake_hub(monkeypatch, exists=True)
        kind = "bucket"
    args = [*_eval_args("bench eval run", tmp_path), flag, "org/runs", "--public"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    said = _said(result)
    assert f"(existing {kind}, its visibility left as it is)" in said
    assert "new public" not in said and "new private" not in said
    if flag == "--publish-hf":
        assert api.calls == ["create_repo", "upload_folder"]
    else:
        assert seen["other"] == []


@pytest.mark.parametrize("route", ROUTES)
def test_eval_public_without_a_publish_flag_is_refused_before_anything_runs(
    tmp_path, monkeypatch, route
):
    async def must_not_run(self):
        raise AssertionError("the run started")

    monkeypatch.setattr(Evaluation, "run", must_not_run)
    api = _fake_api(monkeypatch)
    result = runner.invoke(app, [*_eval_args(route, tmp_path), "--public"])
    assert result.exit_code == 2, result.output
    assert "--public requires --publish-hf or --publish-bucket" in _said(result)
    assert not (tmp_path / "jobs").exists()
    assert api.calls == []


# --source-env: a hosted run never reaches the publish step.

SOURCE_ENV_COMMANDS = {
    "bench eval run": ["eval", "run"],
    "bench eval create": ["eval", "create"],
    "bench run": ["run"],
}
PUBLISH_FLAGS = {
    "--publish-hf": ["--publish-hf", "org/runs"],
    "--publish-bucket": ["--publish-bucket", "org/runs"],
    "--public": ["--public"],
    "--hf-prefix": ["--hf-prefix", "runs"],
    "--hf-public-read-check": ["--hf-public-read-check"],
}


def _source_env_args(command: str, tmp_path: Path) -> list[str]:
    return [
        *SOURCE_ENV_COMMANDS[command],
        "--source-env",
        "primeintellect/general-agent",
        "--source-env-version",
        "0.1.1",
        "--model",
        "gemini-3.1-flash-lite-preview",
        "--jobs-dir",
        str(tmp_path / "jobs"),
    ]


def _fake_hosted_run(monkeypatch, tmp_path: Path) -> list[Any]:
    from benchflow.hosted_env import HostedEnvRunResult

    started: list[Any] = []

    def fake_run_hosted_env(config):
        started.append(config)
        return HostedEnvRunResult(
            source_env=config.source_env,
            run_dir=tmp_path / "run",
            command=["vf-eval"],
            returncode=0,
            stdout="",
            stderr="",
            model=config.model,
            normalized_model=config.model,
            reward=1.0,
        )

    monkeypatch.setattr("benchflow.hosted_env.run_hosted_env", fake_run_hosted_env)
    return started


@pytest.mark.parametrize("flag", list(PUBLISH_FLAGS))
@pytest.mark.parametrize("command", list(SOURCE_ENV_COMMANDS))
def test_source_env_refuses_a_publish_flag_before_anything_runs(
    tmp_path, monkeypatch, command, flag
):
    """audit, 2026-11-13: a --source-env run never reached the publish
    step, so these flags uploaded nothing and said nothing. --hf-prefix and
    --hf-public-read-check are part of the same publish request."""
    started = _fake_hosted_run(monkeypatch, tmp_path)
    api = _fake_api(monkeypatch)
    hub = _fake_hub(monkeypatch)
    args = [*_source_env_args(command, tmp_path), *PUBLISH_FLAGS[flag]]
    result = runner.invoke(app, args)
    assert result.exit_code == 2, result.output
    said = _said(result)
    assert (
        f"{flag} is not supported with --source-env: a --source-env run is never "
        "published, so nothing would be uploaded"
    ) in said
    assert (
        "run it without the publish flags. No bench command publishes a "
        "--source-env run; in Python, publish_folder_to_hf or "
        "publish_folder_to_bucket (benchflow.publish.huggingface) takes its folder, "
        "<jobs-dir>/hosted-env/<run>, and leaves its raw hosted_env/ output out"
    ) in said
    assert started == []
    assert api.calls == []
    assert hub == {"creates": [], "synced": [], "other": []}
    assert not (tmp_path / "jobs").exists()


def test_source_env_names_every_publish_flag_it_refuses(tmp_path, monkeypatch):
    started = _fake_hosted_run(monkeypatch, tmp_path)
    args = [*_source_env_args("bench run", tmp_path), "--publish-hf", "org/runs"]
    result = runner.invoke(app, [*args, "--publish-bucket", "org/runs", "--public"])
    assert result.exit_code == 2, result.output
    said = _said(result)
    assert (
        "--publish-hf, --publish-bucket, --public are not supported with --source-env"
    ) in said
    assert started == []


@pytest.mark.parametrize("command", list(SOURCE_ENV_COMMANDS))
def test_source_env_without_publish_flags_runs_as_before(
    tmp_path, monkeypatch, command
):
    started = _fake_hosted_run(monkeypatch, tmp_path)
    api = _fake_api(monkeypatch)
    result = runner.invoke(app, _source_env_args(command, tmp_path))
    assert result.exit_code == 0, result.output
    assert [c.source_env.env_uid for c in started] == [
        "primeintellect:primeintellect/general-agent@0.1.1"
    ]
    assert "Reward: 1.0" in _said(result)
    assert api.calls == []


def test_the_python_publish_the_refusal_names_leaves_raw_output_out(
    tmp_path, monkeypatch
):
    """The refusal's next step: publish_folder_to_hf on a hosted run's folder
    uploads its run files without hosted_env/ (or a .venv beside it)."""
    run = tmp_path / "jobs" / "hosted-env" / "env__run"
    for name in ("result.json", "hosted_env/stdout.log", ".venv/pyvenv.cfg"):
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        (run / name).write_text("{}")
    api = _fake_api(monkeypatch)
    hf.publish_folder_to_hf(run, repo_id="org/runs", path_in_repo="env__run")
    assert api.uploaded == ["result.json"]


# The CLI: bench train run sft's --publish-model and --publish-artifacts.


class _FakeTrainer:
    def __init__(self, argv: list[str], **kwargs: Any) -> None:
        self.stdout = io.StringIO("trainer ok\n")
        self.stderr = io.StringIO("")

    def wait(self) -> int:
        return 0


def _train(tmp_path: Path, monkeypatch, *flags: str):
    import benchflow.training.backends.prime_rl as prime_rl

    monkeypatch.setattr(prime_rl.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(prime_rl.subprocess, "Popen", _FakeTrainer)
    config = tmp_path / "sft.toml"
    config.write_text("max_steps = 1\n", encoding="utf-8")
    output_dir = tmp_path / "prime-output"
    output_dir.mkdir()
    return runner.invoke(
        app,
        [
            "train",
            "run",
            "sft",
            "--config",
            str(config),
            "--output-dir",
            str(output_dir),
            "--work-dir",
            str(tmp_path / "train-run"),
            *flags,
        ],
    )


TRAIN_FLAGS = {
    "--publish-model": (
        "benchflow/model",
        "model",
        "https://huggingface.co/benchflow/model/tree/main",
    ),
    "--publish-artifacts": (
        "benchflow/artifacts",
        "dataset",
        "https://huggingface.co/datasets/benchflow/artifacts/tree/main/train-run",
    ),
}


@pytest.mark.parametrize("public", [False, True])
@pytest.mark.parametrize("flag", list(TRAIN_FLAGS))
def test_train_publish_creates_private_unless_public(
    tmp_path, monkeypatch, flag, public
):
    repo, repo_type, url = TRAIN_FLAGS[flag]
    api = _fake_api(monkeypatch)
    result = _train(
        tmp_path, monkeypatch, flag, repo, *(["--public"] if public else [])
    )
    assert result.exit_code == 0, result.output
    assert api.creates == [((repo,), {"repo_type": repo_type, "private": not public})]
    kind = "public" if public else "private"
    assert f"Published HF {repo_type}: {url} (new {kind} {repo_type} repo)" in (
        _said(result)
    )
    manifest = json.loads((tmp_path / "train-run" / "train-run.json").read_text())
    assert [p["created"] for p in manifest["extra"]["published"]] == [kind]


@pytest.mark.parametrize("flag", list(TRAIN_FLAGS))
def test_train_publish_to_an_existing_repo_claims_no_visibility(
    tmp_path, monkeypatch, flag
):
    repo, repo_type, url = TRAIN_FLAGS[flag]
    api = _fake_api(monkeypatch, create_status=409)
    result = _train(tmp_path, monkeypatch, flag, repo, "--public")
    assert result.exit_code == 0, result.output
    assert api.calls == ["create_repo", "upload_folder"]
    said = _said(result)
    assert (
        f"Published HF {repo_type}: {url} (existing {repo_type} repo, its "
        "visibility left as it is)"
    ) in said
    assert "new public" not in said and "new private" not in said
    manifest = json.loads((tmp_path / "train-run" / "train-run.json").read_text())
    assert [p["created"] for p in manifest["extra"]["published"]] == [None]


def test_train_public_without_a_publish_flag_is_refused_before_anything_runs(
    tmp_path, monkeypatch
):
    import benchflow.training.backends.prime_rl as prime_rl

    def must_not_launch(*args, **kwargs):
        raise AssertionError("the trainer started")

    api = _fake_api(monkeypatch)
    monkeypatch.setattr(prime_rl.subprocess, "Popen", must_not_launch)
    config = tmp_path / "sft.toml"
    config.write_text("max_steps = 1\n", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "train",
            "run",
            "sft",
            "--config",
            str(config),
            "--work-dir",
            str(tmp_path / "train-run"),
            "--public",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "--public requires --publish-model or --publish-artifacts" in _said(result)
    assert not (tmp_path / "train-run").exists()
    assert api.calls == []
