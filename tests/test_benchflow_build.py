"""Records name the build that wrote them: ``benchflow_build`` (PR #146).

Guards against the regression an audit of PR #120 found (finding V1):
result.json, summary.json and run-card.json recorded only
``benchflow_version``, so the 19 commits of behaviour change after
v0.10.0rc3 on the release branch were indistinguishable from rc3 in every
record. Each now carries ``{"version", "commit", "dirty"}``; the commit is
baked into the wheel by ``hatch_build.py`` because an installed wheel has no
``.git``, and an unknown commit is null, never a guess.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tomllib
import zipfile
from datetime import datetime
from pathlib import Path

import pytest

from benchflow import __version__, build_info
from benchflow.build_info import (
    RELEASE_RECORDS,
    baked_identity,
    benchflow_build,
    checkout_identity,
    direct_url_identity,
    git_identity,
)

ROOT = Path(__file__).resolve().parents[1]
SHA = "0123456789abcdef0123456789abcdef01234567"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=BenchFlow Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo(path: Path) -> Path:
    """A checkout shaped like BenchFlow's: src/benchflow inside it."""
    package = path / "src" / "benchflow"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (path / ".gitignore").write_text("*.log\n")
    _git(path, "init", "-q")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "source")
    return path


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


# The checkout's identity


def test_a_clean_checkout_names_its_head_commit(tmp_path):
    repo = _repo(tmp_path / "repo")
    assert git_identity(repo) == {
        "commit": _git(repo, "rev-parse", "HEAD"),
        "dirty": False,
    }


@pytest.mark.parametrize(
    ("change", "dirty"),
    [
        (lambda r: (r / "src/benchflow/__init__.py").write_text("x = 1\n"), True),
        # An untracked file under src/ would ship in a wheel built from here.
        (lambda r: (r / "src/benchflow/new.py").write_text(""), True),
        (lambda r: (r / "debug.log").write_text("ignored\n"), False),
    ],
    ids=["tracked-change", "untracked-file", "ignored-file"],
)
def test_uncommitted_changes_make_the_checkout_dirty(tmp_path, change, dirty):
    repo = _repo(tmp_path / "repo")
    change(repo)
    assert git_identity(repo)["dirty"] is dirty


def test_an_acceptance_record_commit_keeps_the_source_commit(tmp_path):
    """The release tag's commit adds only the acceptance record, and its
    rebuilt wheel must hash as the accepted one: both name the source."""
    repo = _repo(tmp_path / "repo")
    source = _git(repo, "rev-parse", "HEAD")
    records = repo / RELEASE_RECORDS
    records.mkdir(parents=True)
    (records / "1.0.0.json").write_text("{}\n")
    assert git_identity(repo) == {"commit": source, "dirty": False}
    tag = _commit(repo, "acceptance record")
    assert tag != source
    assert git_identity(repo) == {"commit": source, "dirty": False}
    (repo / "src/benchflow/__init__.py").write_text("x = 2\n")
    later = _commit(repo, "a code change")
    assert git_identity(repo) == {"commit": later, "dirty": False}


def _merge_no_ff(repo: Path) -> tuple[str, str]:
    """GitHub's merge of a PR branched from the base's head: ``--no-ff``
    makes a merge commit whose tree equals the PR head's. Returns (merge,
    PR head)."""
    base = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    _git(repo, "checkout", "-q", "-b", "pr")
    (repo / "src" / "benchflow" / "feature.py").write_text("y = 1\n")
    pr_head = _commit(repo, "the PR's change")
    _git(repo, "checkout", "-q", base)
    _git(repo, "merge", "-q", "--no-ff", "-m", "Merge pull request #1", "pr")
    merge = _git(repo, "rev-parse", "HEAD")
    assert merge != pr_head
    assert _git(repo, "rev-parse", "HEAD^{tree}") == _git(
        repo, "rev-parse", f"{pr_head}^{{tree}}"
    )
    return merge, pr_head


def test_a_merge_commit_names_itself_not_the_pr_head(tmp_path):
    """Review of PR #146 (M2): at a merge commit whose tree equals its PR
    side, git log's history simplification followed the PR side, so the rc
    head's records named the PR head instead of the commit built."""
    repo = _repo(tmp_path / "repo")
    merge, _pr_head = _merge_no_ff(repo)
    assert git_identity(repo) == {"commit": merge, "dirty": False}
    assert checkout_identity(repo / "src" / "benchflow") == {
        "commit": merge,
        "dirty": False,
    }
    # The release tag's commit on top still names the merge, its source.
    records = repo / RELEASE_RECORDS
    records.mkdir(parents=True)
    (records / "1.0.0.json").write_text("{}\n")
    _commit(repo, "acceptance record")
    assert git_identity(repo) == {"commit": merge, "dirty": False}


def test_a_package_outside_the_checkouts_src_gets_no_commit(tmp_path):
    """A virtual environment inside some other repository never lends the
    installed package that repository's commit."""
    repo = _repo(tmp_path / "repo")
    assert checkout_identity(repo / "src" / "benchflow") == git_identity(repo)
    venv_package = repo / ".venv" / "lib" / "site-packages" / "benchflow"
    venv_package.mkdir(parents=True)
    assert checkout_identity(venv_package) is None
    assert checkout_identity(tmp_path) is None


def test_the_callers_git_environment_is_not_followed(tmp_path, monkeypatch):
    repo = _repo(tmp_path / "repo")
    head = _git(repo, "rev-parse", "HEAD")
    other = _repo(tmp_path / "other")
    (other / "src/benchflow/__init__.py").write_text("y = 1\n")
    _commit(other, "other")
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    assert git_identity(repo) == {"commit": head, "dirty": False}


def test_no_git_is_no_commit(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))  # no git on PATH
    assert git_identity(tmp_path) is None


# The baked file and direct_url.json


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (json.dumps({"commit": SHA, "dirty": False}), {"commit": SHA, "dirty": False}),
        (json.dumps({"commit": SHA, "dirty": "no"}), {"commit": SHA, "dirty": None}),
        (json.dumps({"commit": None, "dirty": None}), None),
        (json.dumps({"commit": "abc123", "dirty": False}), None),
        ("not json", None),
        (json.dumps([SHA]), None),
    ],
)
def test_the_baked_file_needs_a_full_commit(tmp_path, text, expected):
    (tmp_path / build_info.BAKED_FILE).write_text(text)
    assert baked_identity(tmp_path) == expected


def test_no_baked_file_is_no_identity(tmp_path):
    assert baked_identity(tmp_path) is None


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (
            {"url": "https://x/y.git", "vcs_info": {"vcs": "git", "commit_id": SHA}},
            {"commit": SHA, "dirty": None},
        ),
        ({"url": "https://x/y.git", "vcs_info": {"vcs": "hg", "commit_id": SHA}}, None),
        (
            {"url": "https://x/y.git", "vcs_info": {"vcs": "git", "commit_id": "main"}},
            None,
        ),
        ({"url": "file:///src", "dir_info": {"editable": True}}, None),
        ({"url": "https://x/y.tar.gz", "archive_info": {}}, None),
    ],
)
def test_direct_url_gives_only_a_recorded_git_commit(data, expected):
    assert direct_url_identity(json.dumps(data)) == expected


def test_no_direct_url_is_no_identity():
    assert direct_url_identity(None) is None
    assert direct_url_identity("{") is None


def test_an_unknown_build_is_null_never_a_guess(tmp_path, monkeypatch):
    build_info._identity.cache_clear()
    monkeypatch.setattr(build_info, "_PACKAGE_DIR", tmp_path)
    monkeypatch.setattr(build_info, "_direct_url_text", lambda: None)
    try:
        assert benchflow_build() == {
            "version": __version__,
            "commit": None,
            "dirty": None,
        }
    finally:
        build_info._identity.cache_clear()


def test_the_running_build_has_the_three_fields():
    build = benchflow_build()
    assert list(build) == ["version", "commit", "dirty"]
    assert build["version"] == __version__
    assert build["commit"] is None or len(build["commit"]) in (40, 64)


# The records


def test_result_json_names_the_build(tmp_path):
    from benchflow.rollout import _build_rollout_result

    _build_rollout_result(
        tmp_path,
        task_name="t",
        rollout_name="t__1",
        agent="oracle",
        agent_name="oracle",
        model=None,
        n_tool_calls=0,
        prompts=["solve"],
        error=None,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        started_at=datetime.now(),
        timing={},
    )
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["benchflow_build"] == benchflow_build()


def _task(root: Path) -> Path:
    task = root / "t1"
    task.mkdir(parents=True)
    (task / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n[agent]\ntimeout_sec = 60\n[environment]\n'
    )
    (task / "instruction.md").write_text("x\n")
    return task


def test_summary_and_run_card_name_the_build(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from benchflow.cli.main import app
    from benchflow.evaluation import Evaluation
    from benchflow.models import RolloutResult

    async def fake(self, task_path, _cfg):
        trial = self._jobs_dir / self._job_name / f"{task_path.name}__00000001"
        trial.mkdir(parents=True)
        (trial / "result.json").write_text(
            json.dumps({"task_name": task_path.name, "rollout_name": trial.name})
        )
        return RolloutResult(
            task_path.name,
            rollout_name=trial.name,
            rewards={"reward": 1.0},
            agent="oracle",
            rollout_dir=trial,
        )

    monkeypatch.setattr(Evaluation, "_run_single_task", fake)
    task = _task(tmp_path)
    run = CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(task),
            "--agent",
            "oracle",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--sandbox",
            "daytona",
        ],
        terminal_width=200,
    )
    assert run.exit_code == 0, run.output
    job = next(p for p in (tmp_path / "jobs").iterdir() if p.is_dir())
    for name in ("summary.json", "run-card.json"):
        record = json.loads((job / name).read_text())
        assert record["benchflow_build"] == benchflow_build(), name
        assert record["benchflow_version"] == __version__, name


def test_a_trial_without_a_folder_names_the_build(tmp_path):
    from benchflow.evaluation import Evaluation, EvaluationConfig

    task = _task(tmp_path / "tasks")
    evaluation = Evaluation(
        tasks_dir=str(task.parent),
        jobs_dir=str(tmp_path / "jobs"),
        config=EvaluationConfig(agent="oracle"),
        preflight=False,
    )
    job_dir = tmp_path / "jobs" / "job"
    job_dir.mkdir(parents=True)
    payload = {"task_name": "t1", "rewards": None, "error": "task.toml typo"}
    evaluation._record_trial_without_folder(job_dir, "t1", payload)
    (folder,) = [p for p in job_dir.iterdir() if p.is_dir()]
    result = json.loads((folder / "result.json").read_text())
    assert result["benchflow_build"] == benchflow_build()
    assert "benchflow_build" not in payload  # the summary's row is unchanged


# The build hook, end to end: sdist, the wheel built from it, the install


def _project(path: Path) -> Path:
    """A one-module project built with BenchFlow's own hook and backend pin."""
    real = tomllib.loads((ROOT / "pyproject.toml").read_text())
    hatch = real["tool"]["hatch"]["build"]
    assert "custom" in hatch["hooks"]
    assert "hatch_build.py" in hatch["targets"]["sdist"]["only-include"]
    package = path / "src" / "benchflow"
    package.mkdir(parents=True)
    shutil.copy(ROOT / "hatch_build.py", path / "hatch_build.py")
    shutil.copy(ROOT / "src" / "benchflow" / "build_info.py", package)
    (package / "__init__.py").write_text('__version__ = "1.0.0"\n')
    # Hatchling writes hatch_build.py's bytecode beside it before the hook
    # runs; BenchFlow's .gitignore keeps that out of the dirty state.
    assert "__pycache__/" in (ROOT / ".gitignore").read_text().splitlines()
    (path / ".gitignore").write_text("__pycache__/\n")
    (path / "pyproject.toml").write_text(
        '[project]\nname = "benchflow"\nversion = "1.0.0"\n'
        "[build-system]\n"
        f"requires = {json.dumps(real['build-system']['requires'])}\n"
        f'build-backend = "{real["build-system"]["build-backend"]}"\n'
        "[tool.hatch.build.hooks.custom]\n"
        "[tool.hatch.build.targets.sdist]\n"
        'only-include = ["src", "pyproject.toml", "hatch_build.py"]\n'
    )
    _git(path, "init", "-q")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "source")
    return path


def _build(project: Path, out: Path) -> dict[str, str]:
    """``uv build`` as the release does: the sdist, then the wheel from it."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("UV_", "GIT_"))}
    subprocess.run(
        [
            "uv",
            "--no-config",
            "build",
            "--no-sources",
            "--out-dir",
            str(out),
            "--quiet",
        ],
        cwd=project,
        env=env,
        check=True,
        capture_output=True,
        timeout=600,
    )
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.iterdir())
        if p.suffix in (".whl", ".gz")
    }


def _baked(out: Path) -> tuple[dict, dict]:
    (wheel,) = out.glob("*.whl")
    (sdist,) = out.glob("*.tar.gz")
    with zipfile.ZipFile(wheel) as zf:
        in_wheel = json.loads(zf.read("benchflow/_build.json"))
    with tarfile.open(sdist) as tf:
        member = tf.extractfile("benchflow-1.0.0/src/benchflow/_build.json")
        assert member is not None
        in_sdist = json.loads(member.read())
    return in_wheel, in_sdist


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to build")
def test_the_wheel_carries_its_commit_and_rebuilds_byte_for_byte(tmp_path):
    project = _project(tmp_path / "project")
    source = _git(project, "rev-parse", "HEAD")
    first = _build(project, tmp_path / "dist1")
    in_wheel, in_sdist = _baked(tmp_path / "dist1")
    assert in_wheel == in_sdist == {"commit": source, "dirty": False}

    # The release tag's commit adds only its acceptance record: same bytes.
    records = project / RELEASE_RECORDS
    records.mkdir(parents=True)
    (records / "1.0.0.json").write_text("{}\n")
    _commit(project, "acceptance record")
    assert _build(project, tmp_path / "dist2") == first

    # A change that is not committed is a different, dirty build.
    (project / "src/benchflow/__init__.py").write_text('__version__ = "1.0.0"\nx = 1\n')
    _build(project, tmp_path / "dist3")
    assert _baked(tmp_path / "dist3")[0] == {"commit": source, "dirty": True}


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to build")
def test_an_installed_wheel_reports_the_commit_it_was_built_from(tmp_path):
    project = _project(tmp_path / "project")
    source = _git(project, "rev-parse", "HEAD")
    _build(project, tmp_path / "dist")
    (wheel,) = (tmp_path / "dist").glob("*.whl")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("UV_", "GIT_"))}
    venv = tmp_path / "venv"
    subprocess.run(
        ["uv", "--no-config", "venv", "--quiet", str(venv)], env=env, check=True
    )
    python = venv / "bin" / "python"
    subprocess.run(
        [
            "uv",
            "--no-config",
            "pip",
            "install",
            "--quiet",
            "--python",
            str(python),
            str(wheel),
        ],
        env=env,
        check=True,
    )
    # Run away from the project: the installed package, not the source.
    printed = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import json; from benchflow.build_info import benchflow_build;"
            " print(json.dumps(benchflow_build()))",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert json.loads(printed) == {"version": "1.0.0", "commit": source, "dirty": False}


def _installed_build(tmp_path: Path, target: str) -> dict:
    """``benchflow_build()`` of ``target`` installed with uv into a fresh
    venv, run away from the project."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("UV_", "GIT_"))}
    venv = tmp_path / "venv"
    subprocess.run(
        ["uv", "--no-config", "venv", "--quiet", str(venv)], env=env, check=True
    )
    python = venv / "bin" / "python"
    subprocess.run(
        [
            "uv",
            "--no-config",
            "pip",
            "install",
            "--quiet",
            "--no-cache",
            "--python",
            str(python),
            target,
        ],
        env=env,
        check=True,
        timeout=600,
    )
    printed = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import json; from benchflow.build_info import benchflow_build;"
            " print(json.dumps(benchflow_build()))",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(printed)


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to build")
def test_a_wheel_and_a_git_install_of_a_merge_commit_name_the_merge(tmp_path):
    """Review of PR #146 (M2): the wheel built at a ``--no-ff`` merge, and
    ``pip install git+URL@MERGE``, named the PR head."""
    project = _project(tmp_path / "project")
    merge, _pr_head = _merge_no_ff(project)
    _build(project, tmp_path / "dist")
    in_wheel, in_sdist = _baked(tmp_path / "dist")
    assert in_wheel == in_sdist == {"commit": merge, "dirty": False}
    installed = _installed_build(
        tmp_path / "git-install", f"benchflow @ git+file://{project}@{merge}"
    )
    assert installed == {"version": "1.0.0", "commit": merge, "dirty": False}


def test_stamping_never_raises(monkeypatch):
    """Review of PR #146: a git that printed bytes that are not UTF-8, or any
    other surprise, would have failed the record being written."""

    def broken(_package_dir):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    build_info._identity.cache_clear()
    monkeypatch.setattr(build_info, "checkout_identity", broken)
    try:
        assert benchflow_build()["commit"] is None
    finally:
        build_info._identity.cache_clear()


def test_git_output_that_is_not_utf8_is_no_answer(tmp_path, monkeypatch):
    fake = tmp_path / "bin" / "git"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\nprintf '\\377\\376'\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake.parent))
    assert build_info._git(tmp_path, "rev-parse", "HEAD") is None


def test_the_build_is_read_when_a_run_starts(tmp_path, monkeypatch):
    from benchflow.evaluation import Evaluation, EvaluationConfig

    reads = []
    monkeypatch.setattr(build_info, "read_at_start", lambda: reads.append(1))
    task = _task(tmp_path / "tasks")
    Evaluation(
        tasks_dir=str(task.parent),
        jobs_dir=str(tmp_path / "jobs"),
        config=EvaluationConfig(agent="oracle"),
        preflight=False,
    )
    assert reads == [1]
