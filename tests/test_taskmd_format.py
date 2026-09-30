"""The built-in ``taskmd`` task format: detection, materialization, refusals, families, stages."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from benchflow.task.formats import detect_task_format, materialize_task_dir
from benchflow.taskmd import (
    TaskMdError,
    TaskMdFormat,
    family,
    is_taskmd_package,
    taskmd_metadata,
)
from benchflow.taskmd.reference import parse_package
from tests._taskmd_helpers import EXAMPLES


@pytest.fixture(autouse=True)
def format_cache(tmp_path, monkeypatch) -> Path:
    cache = tmp_path / "format-cache"
    monkeypatch.setenv("BENCHFLOW_TASK_FORMAT_CACHE", str(cache))
    return cache


def _frontmatter(package: Path) -> dict:
    text = (package / "task.md").read_text()
    end = text.find("\n---\n", 4)
    return yaml.safe_load(text[4:end])


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _package(
    root: Path,
    task_md: str,
    *,
    dockerfile: str = "FROM ubuntu:24.04\nWORKDIR /app\n",
    test_sh: str | None = None,
) -> Path:
    _write(root / "task.md", task_md)
    _write(root / "sandbox" / "Dockerfile", dockerfile)
    _write(
        root / "verifier" / "test.sh",
        test_sh or "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n",
    )
    return root


# Detection -----------------------------------------------------------------------------------------


def test_detects_draft_2_and_never_a_native_or_frontmatter_task(tmp_path) -> None:
    assert is_taskmd_package(EXAMPLES / "hello-world")
    assert detect_task_format(EXAMPLES / "hello-world").name == "taskmd"
    canary = _write(
        tmp_path / "canary" / "task.md", "<!-- task.md canary 1234 -->\n\nDo it.\n"
    )
    assert is_taskmd_package(canary.parent)
    native = _write(
        tmp_path / "native" / "task.md", "---\nschema_version: '1.3'\n---\n\nDo it.\n"
    )
    assert not is_taskmd_package(native.parent)
    bom = _write(tmp_path / "bom" / "task.md", "﻿---\nagent: {}\n---\nDo it.\n")
    assert not is_taskmd_package(bom.parent)
    blank_first = _write(
        tmp_path / "blank" / "task.md", "\n\n---\nagent: {}\n---\nDo it.\n"
    )
    assert not is_taskmd_package(blank_first.parent)
    robouse = _write(
        tmp_path / "robouse" / "task.md", "---\nrobouse:\n  id: reach\n---\nReach.\n"
    )
    assert not is_taskmd_package(robouse.parent)
    empty = _write(tmp_path / "empty" / "task.md", "")
    assert not is_taskmd_package(empty.parent)
    legacy = _write(tmp_path / "legacy" / "task.toml", "[agent]\ntimeout_sec = 10\n")
    _write(legacy.parent / "instruction.md", "Do it.\n")
    assert not is_taskmd_package(legacy.parent)
    assert not is_taskmd_package(tmp_path / "missing")
    folder = tmp_path / "folder-named-task.md"
    (folder / "task.md").mkdir(parents=True)
    assert not is_taskmd_package(folder)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads unreadable files")
def test_detection_never_raises(tmp_path) -> None:
    unreadable = _write(tmp_path / "locked" / "task.md", "Do it.\n")
    unreadable.chmod(0)
    try:
        assert TaskMdFormat().detect(unreadable.parent) is False
    finally:
        unreadable.chmod(0o644)


# Materializing -------------------------------------------------------------------------------------


def test_hello_world_materializes_as_a_native_package() -> None:
    from benchflow.task import Task

    source = EXAMPLES / "hello-world"
    native = materialize_task_dir(source)
    assert native.name == "hello-world"
    assert detect_task_format(native) is None
    task = Task(native)
    assert task.instruction.strip() == parse_package(source).instruction.strip()
    assert task.config.agent.timeout_sec == 120
    assert (native / "environment" / "Dockerfile").read_text() == (
        source / "sandbox" / "Dockerfile"
    ).read_text()
    assert (native / "oracle" / "solve.sh").read_bytes() == (
        source / "oracle" / "solve.sh"
    ).read_bytes()
    assert (native / "verifier" / "rubric.json").is_file()
    strategy = yaml.safe_load(
        (native / "verifier" / "verifier.md").read_text().split("---")[1]
    )
    assert strategy["verifier"]["strategies"]["taskmd"] == {
        "type": "taskmd",
        "grading": "rubric",
        "isolation": "shared",
        "offline": False,
        "command": "test.sh",
    }
    meta = taskmd_metadata(native)
    assert meta is not None
    assert meta["verifier_mount"] == "/verifier" and meta["grading"] == "rubric"
    assert meta["source_tree"].startswith("sha256:")
    assert (native / ".taskmd" / "package" / "task.md").read_bytes() == (
        source / "task.md"
    ).read_bytes()
    assert materialize_task_dir(source) == native  # content-addressed


def test_the_review_leaves_a_task_md_rubric_to_its_verifier() -> None:
    from benchflow.review.config import find_task_rubrics

    native = materialize_task_dir(EXAMPLES / "hello-world")
    assert find_task_rubrics(native) == []


def test_harbor_import_keeps_its_mounts_and_prebuilt_image() -> None:
    from benchflow.task import Task

    native = materialize_task_dir(EXAMPLES / "regex-log")
    assert (native / "tests" / "test.sh").is_file() and not (
        native / "verifier"
    ).exists()
    assert (native / "solution" / "solve.sh").is_file() and not (
        native / "oracle"
    ).exists()
    task = Task(native)
    assert task.config.sandbox.docker_image == "alexgshaw/regex-log:20251031"
    assert (
        task.config.sandbox.memory_mb == 2048
        and task.config.sandbox.storage_mb == 10240
    )
    assert taskmd_metadata(native)["grading"] == "script"


def test_mcp_servers_map_and_an_empty_list_means_none(tmp_path) -> None:
    from benchflow.task import Task

    empty = _package(
        tmp_path / "empty",
        "Do it.\n\n```toml task\n[sandbox]\nmcp = []\ngpus = 0\n```\n",
    )
    assert Task(materialize_task_dir(empty)).config.sandbox.mcp_servers == []
    served = _package(
        tmp_path / "served",
        "Do it.\n\n```toml task\n[[sandbox.mcp]]\n"
        'name = "mcp-server"\ntransport = "streamable-http"\nurl = "http://mcp-server:8000/mcp"\n```\n',
    )
    servers = Task(materialize_task_dir(served)).config.sandbox.mcp_servers
    assert [(s.name, s.transport, s.url) for s in servers] == [
        ("mcp-server", "streamable-http", "http://mcp-server:8000/mcp")
    ]
    broken = _package(
        tmp_path / "broken",
        'Do it.\n\n```toml task\n[[sandbox.mcp]]\nname = "x"\ntransport = "stdio"\n```\n',
    )
    with pytest.raises(TaskMdError, match=r"\[sandbox\] mcp\[0\].*command"):
        materialize_task_dir(broken)


def test_skills_leave_scripted_seats_alone_and_refuse_an_agent(tmp_path) -> None:
    from benchflow.taskmd.launch import TaskMdLaunchRefused, check_launch

    package = _package(
        tmp_path / "skills",
        'Do it.\n\n```toml task\n[sandbox]\nskills = "/skills"\n```\n',
    )
    native = materialize_task_dir(package)
    check_launch(native, primary_agent="oracle", sandbox_user=None)
    check_launch(native, primary_agent="nop", sandbox_user=None)
    with pytest.raises(TaskMdLaunchRefused, match=r"\[sandbox\] skills"):
        check_launch(native, primary_agent="claude-agent-acp", sandbox_user=None)


@pytest.mark.parametrize(
    ("image", "verifier_dockerfile", "declared", "source"),
    [
        # verifier/Dockerfile comes before the task's image, declared or not.
        (True, True, True, "tests/Dockerfile"),
        (True, True, False, "tests/Dockerfile"),
        (False, True, False, "tests/Dockerfile"),
        # Without one, the task's image: prebuilt ...
        (True, False, True, "verifier.sandbox.docker_image"),
        (True, False, False, "sandbox.docker_image"),
        # ... or built from sandbox/Dockerfile.
        (False, False, False, "environment/Dockerfile"),
    ],
)
def test_a_separate_verifier_runs_in_the_image_the_spec_names(
    tmp_path, image, verifier_dockerfile, declared, source
) -> None:
    """docs/document.md, "Discovery and the verifier's image", as BenchFlow's own planner picks it."""
    from benchflow.task import Task
    from benchflow.task.verifier_sandbox import plan_verifier_image

    config = '[verifier]\nisolation = "separate"\nmount = "/tests"\n'
    if image:
        config = '[sandbox]\nimage = "ubuntu:24.04"\n\n' + config
    if declared:
        config += "\n[verifier.sandbox]\ncpus = 2\n"
    package = _package(tmp_path / "p", f"Do it.\n\n```toml task\n{config}```\n")
    if verifier_dockerfile:
        _write(
            package / "verifier" / "Dockerfile",
            "FROM ubuntu:24.04\nCOPY test.sh /tests/test.sh\n",
        )
    native = materialize_task_dir(package)
    chosen = plan_verifier_image(Task(native).config, native)
    assert chosen.source == source
    if source.endswith("docker_image"):
        assert chosen.sandbox.docker_image == "ubuntu:24.04"
    if declared:
        assert chosen.sandbox.cpus == 2


def _verifier_decision(tmp_path, config: str, *, verifier_dockerfile: bool = False):
    from benchflow.taskmd.plan import plan_package
    from benchflow.taskmd.reference import errors

    package = _package(tmp_path / "p", f"Do it.\n\n```toml task\n{config}```\n")
    if verifier_dockerfile:
        _write(package / "verifier" / "Dockerfile", "FROM ubuntu:24.04\n")
    document = parse_package(package)
    assert not errors(document)
    return package, plan_package(document, package)


@pytest.mark.parametrize(
    ("config", "offline"),
    [
        # A shared verifier can be taken offline, and is open under an agent allowlist.
        ('[sandbox]\nnetwork = "open"\n\n[verifier]\nnetwork = "none"\n', True),
        ('[sandbox]\nnetwork = ["pypi.org"]\n\n[verifier]\nnetwork = "open"\n', False),
        ('[sandbox]\nnetwork = ["pypi.org"]\n\n[verifier]\nnetwork = "none"\n', True),
    ],
)
def test_a_shared_verifier_network_that_benchflow_enforces(
    tmp_path, config, offline
) -> None:
    package, plan = _verifier_decision(tmp_path, config)
    assert not plan.refused, plan.refused
    strategy = yaml.safe_load(
        (materialize_task_dir(package) / "verifier" / "verifier.md")
        .read_text()
        .split("---")[1]
    )["verifier"]["strategies"]["taskmd"]
    assert strategy["offline"] is offline


@pytest.mark.parametrize(
    ("config", "field"),
    [
        # The agent's container is offline: a shared verifier cannot get a network.
        (
            '[sandbox]\nnetwork = "none"\n\n[verifier]\nnetwork = "open"\n',
            "[verifier] network",
        ),
        # A shared verifier runs as root, outside the agent's allowlist.
        (
            '[sandbox]\nnetwork = ["pypi.org"]\n',
            "[verifier] network (inherited from [sandbox] network)",
        ),
        (
            '[verifier]\nnetwork = ["pypi.org"]\n',
            "[verifier] network",
        ),
        # No host list is enforced in a separate verifier's sandbox.
        (
            '[sandbox]\nimage = "ubuntu:24.04"\n\n[verifier]\nisolation = "separate"\nnetwork = ["pypi.org"]\n',
            "[verifier] network",
        ),
        # A separate verifier's own network needs an image of its own or the task's.
        (
            '[sandbox]\nnetwork = "none"\n\n[verifier]\nisolation = "separate"\nnetwork = "open"\n',
            "[verifier] network",
        ),
    ],
)
def test_a_verifier_network_that_benchflow_cannot_enforce_is_refused(
    tmp_path, config, field
) -> None:
    _, plan = _verifier_decision(tmp_path, config)
    assert field in [f.field for f in plan.refused], plan.refused


@pytest.mark.parametrize(
    ("config", "verifier_dockerfile", "mode", "image"),
    [
        (
            '[sandbox]\nimage = "ubuntu:24.04"\nnetwork = "none"\n\n[verifier]\nisolation = "separate"\n\n'
            '[verifier.sandbox]\nnetwork = "open"\n',
            False,
            "public",
            "ubuntu:24.04",
        ),
        (
            '[sandbox]\nimage = "ubuntu:24.04"\nnetwork = "none"\n\n[verifier]\nisolation = "separate"\nnetwork = "open"\n',
            False,
            "public",
            "ubuntu:24.04",
        ),
        (
            '[sandbox]\nnetwork = "open"\n\n[verifier]\nisolation = "separate"\nnetwork = "none"\n',
            True,
            "no-network",
            None,
        ),
    ],
)
def test_a_separate_verifier_gets_its_own_network(
    tmp_path, config, verifier_dockerfile, mode, image
) -> None:
    from benchflow.task import Task
    from benchflow.task.verifier_sandbox import plan_verifier_image

    package, plan = _verifier_decision(
        tmp_path, config, verifier_dockerfile=verifier_dockerfile
    )
    assert not plan.refused, plan.refused
    native = materialize_task_dir(package)
    chosen = plan_verifier_image(Task(native).config, native)
    assert chosen.sandbox.network_mode.value == mode
    assert chosen.sandbox.docker_image == image


def test_a_separate_verifier_with_its_own_settings_needs_an_image(tmp_path) -> None:
    package = _package(
        tmp_path / "p",
        'Do it.\n\n```toml task\n[verifier]\nisolation = "separate"\n\n[verifier.sandbox]\ncpus = 2\n```\n',
    )
    with pytest.raises(
        TaskMdError, match=r"\[verifier.sandbox\]: BenchFlow gives a verifier"
    ):
        materialize_task_dir(package)


def test_separate_verifier_offline_and_judge_time() -> None:
    from benchflow.task import Task

    native = materialize_task_dir(EXAMPLES / "analysis-judge")
    task = Task(native)
    assert task.config.verifier.sandbox_mode.value == "separate"
    assert task.config.sandbox.network_mode.value == "no-network"
    # 5m of scripts, plus 3 agent sessions of 20m each rejudged once, plus the setup margin.
    assert task.config.verifier.timeout_sec == 300 + 3 * 2 * 1200 + 600
    strategy = yaml.safe_load(
        (native / "verifier" / "verifier.md").read_text().split("---")[1]
    )["verifier"]["strategies"]["taskmd"]
    assert (
        strategy["offline"] is True
        and strategy["workdir"] == "/work"
        and strategy["script_timeout"] == 300
    )
    assert "verifier/answers.json" in json.dumps(taskmd_metadata(native)["config"])


@pytest.mark.parametrize(
    ("name", "fields"),
    [
        (
            "calc-quarterly",
            [
                "[sandbox] clock",
                "[world]",
                "[agent] network",
                "bar-chart",
                "verifier/behaviors.json",
            ],
        ),
        ("flaky-retry", ["[agent] network", "extends", "verifier/behaviors.json"]),
        (
            "ising-exponent",
            [
                "[sandbox] network",
                "[stages.analysis] submit",
                "[stages.analysis] mounts",
            ],
        ),
    ],
)
def test_refusals_name_every_field(name: str, fields: list[str]) -> None:
    with pytest.raises(TaskMdError) as caught:
        materialize_task_dir(EXAMPLES / name)
    message = str(caught.value)
    for field in fields:
        assert field in message, field


def test_a_reference_error_refuses_the_package(tmp_path) -> None:
    package = _package(
        tmp_path / "bad",
        'Do it.\n\n```toml task\n[agent]\ntimeout = "2m"\nbudgett = 3\n```\n',
    )
    with pytest.raises(TaskMdError, match="reference parser reports errors") as caught:
        materialize_task_dir(package)
    assert "budgett" in str(caught.value)


def test_an_instruction_with_reserved_native_headings_is_kept(tmp_path) -> None:
    from benchflow.task import Task

    instruction = (
        "Write the play.\n\n## role:narrator\n\nSay hello.\n\n## prompt\n\nThe end."
    )
    package = _package(
        tmp_path / "play",
        instruction + '\n\n```toml task\n[agent]\ntimeout = "1m"\n```\n',
    )
    native = materialize_task_dir(package)
    assert Task(native).instruction.strip() == instruction


def test_templated_env_values_are_refused(tmp_path) -> None:
    package = _package(
        tmp_path / "env",
        'Do it.\n\n```toml task\n[sandbox]\nenv = { TOKEN = "${GITHUB_TOKEN}", MODE = "fast" }\n```\n',
    )
    with pytest.raises(TaskMdError, match=r"\[sandbox\] env.TOKEN"):
        materialize_task_dir(package)


def test_a_batch_skips_a_refused_package_and_runs_the_rest(tmp_path, caplog) -> None:
    from benchflow.evaluation import Evaluation, EvaluationConfig

    suite = tmp_path / "suite"
    shutil.copytree(EXAMPLES / "hello-world", suite / "hello-world")
    shutil.copytree(EXAMPLES / "calc-quarterly", suite / "calc-quarterly")
    ev = Evaluation(
        tasks_dir=suite,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(agent="oracle"),
    )
    with caplog.at_level("WARNING"):
        dirs = ev._get_task_dirs()
    assert [d.name for d in dirs] == ["hello-world"]
    assert "calc-quarterly" in caplog.text and "[sandbox] clock" in caplog.text


# Families ------------------------------------------------------------------------------------------


def _host_docker(
    args: list[str], *, timeout: float, what: str
) -> subprocess.CompletedProcess[str]:
    """Stand-in for the docker CLI: runs the generator on the host with /package and /out mapped."""
    del timeout, what
    if args[0] in ("build", "rmi"):
        return subprocess.CompletedProcess(args, 0, "", "")
    assert args[0] == "run"
    assert "--network" in args and args[args.index("--network") + 1] == "none"
    assert "no-new-privileges" in args
    mounts = [args[i + 1] for i, a in enumerate(args) if a == "-v"]
    package = next(m.split(":")[0] for m in mounts if ":/package" in m)
    out = next(m.split(":")[0] for m in mounts if m.endswith(":/out"))
    at = args.index("--entrypoint")
    entrypoint, tail = args[at + 1], args[at + 3 :]  # skip the image
    tail = [
        package + a[len("/package") :]
        if a.startswith("/package/")
        else out
        if a == "/out"
        else a
        for a in tail
    ]
    return subprocess.run(
        [entrypoint, *tail], capture_output=True, text=True, cwd=package, check=False
    )


def test_a_family_seed_materializes_its_instance(monkeypatch) -> None:
    from benchflow.task import Task

    monkeypatch.setattr(family, "_docker", _host_docker)
    source = EXAMPLES / "sql-family"
    with pytest.raises(TaskMdError, match="--seeds"):
        materialize_task_dir(source)
    native = materialize_task_dir(source, seed=3)
    assert native.name == "sql-family--seed-3"
    task = Task(native)
    assert "{{" not in task.instruction
    meta = taskmd_metadata(native)
    params = meta["family"]["params"]
    assert meta["family"]["seed"] == 3 and meta["family"]["role"] == "train"
    assert (
        params["metric"] in task.instruction
        and f"top {params['k']}" in task.instruction
    )
    assert (native / "environment" / "taskmd-instance" / "data" / "shop.db").is_file()
    assert (
        "COPY taskmd-instance/ /" in (native / "environment" / "Dockerfile").read_text()
    )
    assert (native / "verifier" / "instance" / "hidden.db").is_file()
    assert not (native / "environment" / "taskmd-instance" / "verifier").exists()
    fm = _frontmatter(native)
    expected = {
        "TASKMD_SEED": "3",
        "TASKMD_PARAMS": json.dumps(params, sort_keys=True, separators=(",", ":")),
    }
    assert fm["oracle"]["env"] == expected and fm["verifier"]["env"] == expected
    assert meta["agent_refused"][0]["field"] == "[agent] budget"
    # seed 9 is in the test split
    assert (
        taskmd_metadata(materialize_task_dir(source, seed=9))["family"]["role"]
        == "test"
    )


def test_a_failing_generator_refuses_the_seed(monkeypatch, tmp_path) -> None:
    def failing(args, *, timeout, what):
        del timeout, what
        return subprocess.CompletedProcess(
            args, 0 if args[0] != "run" else 3, "", "boom"
        )

    monkeypatch.setattr(family, "_docker", failing)
    with pytest.raises(TaskMdError, match="generator-failed"):
        materialize_task_dir(EXAMPLES / "sql-family", seed=1)


def test_a_seed_for_a_task_that_is_no_family_is_refused() -> None:
    with pytest.raises(TaskMdError, match="not a family"):
        materialize_task_dir(EXAMPLES / "hello-world", seed=1)


# Controls ------------------------------------------------------------------------------------------


def test_a_control_runs_as_the_oracle(format_cache) -> None:
    from benchflow.taskmd import controls

    source = EXAMPLES / "analysis-judge"
    assert controls(source) == {"known-bad-line-fit": "controls/line-fit.sh"}
    native = TaskMdFormat().materialize_variant(
        source, format_cache / "taskmd", control="known-bad-line-fit"
    )
    assert native.name == "analysis-judge--control-known-bad-line-fit"
    assert (native / "oracle" / "solve.sh").read_bytes() == (
        source / "controls" / "line-fit.sh"
    ).read_bytes()
    assert not (native / "environment" / "controls").exists()


# Stages (M3) ---------------------------------------------------------------------------------------


def _staged(root: Path) -> Path:
    text = (
        "Write a plan in /app/plan.md.\n\n"
        "```stage build\nNow build it from the plan.\n```\n\n"
        "```stage report\nReport what you built in /app/report.md.\n```\n\n"
        '```toml task\n[stages.build]\nunlock = "on_submit"\n\n[stages.report]\nunlock = "after:build"\n```\n'
    )
    return _package(root, text)


def test_chained_stages_become_turns_of_the_runs_agent(tmp_path) -> None:
    from benchflow.rollout import RolloutConfig

    native = materialize_task_dir(_staged(tmp_path / "staged"))
    assert [t["stage"] for t in taskmd_metadata(native)["turns"]] == ["build", "report"]
    config = RolloutConfig.from_legacy(
        task_path=native, agent="claude-agent-acp", model="claude-haiku-4-5"
    )
    (scene,) = config.scenes
    assert [role.agent for role in scene.roles] == ["claude-agent-acp"]
    assert [t.prompt for t in scene.turns] == [
        None,
        "Now build it from the plan.",
        "Report what you built in /app/report.md.",
    ]
    oracle = RolloutConfig.from_legacy(task_path=native, agent="oracle")
    assert oracle.primary_agent == "oracle"
    assert len(oracle.effective_scenes[0].turns) == 1


def test_stages_out_of_order_are_refused(tmp_path) -> None:
    text = (
        "Plan.\n\n```stage b\nB.\n```\n\n"
        '```toml task\n[stages.b]\nunlock = "on_request"\n```\n'
    )
    with pytest.raises(TaskMdError, match=r"\[stages.b\] unlock"):
        materialize_task_dir(_package(tmp_path / "req", text))


def test_roles_and_users_are_refused(tmp_path) -> None:
    text = 'Build it.\n\n```role reviewer\nReview it.\n```\n\n```user\nYou are a customer.\n```\n\n```toml task\n[roles.reviewer]\ntools = ["shell"]\n```\n'
    with pytest.raises(TaskMdError) as caught:
        materialize_task_dir(_package(tmp_path / "multi", text))
    assert "```role reviewer" in str(caught.value) and "```user" in str(caught.value)


# Launching -----------------------------------------------------------------------------------------


def test_an_agent_is_refused_what_only_a_script_can_honor(monkeypatch) -> None:
    from benchflow.taskmd.launch import TaskMdLaunchRefused, check_launch

    monkeypatch.setattr(family, "_docker", _host_docker)
    native = materialize_task_dir(EXAMPLES / "sql-family", seed=2)
    check_launch(native, primary_agent="oracle", sandbox_user=None)
    check_launch(native, primary_agent="nop", sandbox_user="agent")
    with pytest.raises(TaskMdLaunchRefused, match=r"\[agent\] budget"):
        check_launch(native, primary_agent="claude-agent-acp", sandbox_user="agent")


def test_judges_are_checked_before_the_solver_starts(monkeypatch) -> None:
    from benchflow.taskmd.launch import TaskMdLaunchRefused, check_launch

    native = materialize_task_dir(EXAMPLES / "analysis-judge")
    for name in (
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_OAUTH_TOKEN",
        "BENCHFLOW_TASKMD_JUDGE_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(TaskMdLaunchRefused, match="ANTHROPIC_API_KEY"):
        check_launch(native, primary_agent="oracle", sandbox_user=None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder")
    check_launch(native, primary_agent="oracle", sandbox_user=None)
    monkeypatch.setenv("BENCHFLOW_TASKMD_JUDGE_MODEL", "gpt-5")
    with pytest.raises(TaskMdLaunchRefused, match="Anthropic Messages API only"):
        check_launch(native, primary_agent="oracle", sandbox_user=None)
    monkeypatch.setenv(
        "BENCHFLOW_TASKMD_JUDGE_MODEL", "agent=claude-haiku-4-5-20251001"
    )
    check_launch(native, primary_agent="oracle", sandbox_user=None)
