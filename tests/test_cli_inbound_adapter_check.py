"""CLI coverage for checking foreign benchmark tasks through inbound adapters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app


def _force_ios_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the iOS host-capability probe off so the unsupported path is asserted.

    The iOSWorld adapter reports *supported* on a host that actually has the
    iOS Simulator toolchain. These CLI tests cover the provider-honest
    *unsupported* report, so they force the probe to "missing" regardless of
    the host the suite runs on.
    """
    import benchflow.sandbox.macos_ios_simulator as ios_sim

    monkeypatch.setattr(
        ios_sim,
        "detect_ios_simulator_capabilities",
        lambda: dict.fromkeys(
            ("macos", "xcode-26", "ios-26-simulator-runtime", "appium-xcuitest"),
            False,
        ),
    )


def _write_cookbook_osworld_task(root: Path) -> Path:
    task_dir = root / "use-computer-osworld-task"
    task_dir.mkdir()
    (task_dir / "instruction.md").write_text("Observe the desktop once, then stop.\n")
    (task_dir / "task.toml").write_text(
        """\
version = "1.0"

[task]
name = "osworld/ubuntu-smoke"
description = "OSWorld Ubuntu smoke task"
keywords = ["osworld", "ubuntu", "smoke"]

[metadata]
author_name = "Use.Computer"
difficulty = "smoke"
category = "desktop-automation"
tags = ["osworld", "ubuntu", "gui", "smoke"]

[verifier]
timeout_sec = 180

[agent]
timeout_sec = 180

[environment]
cpus = 4
memory_mb = 8192
allow_internet = true
"""
    )
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "osworld_task.json").write_text(
        json.dumps(
            {
                "id": "smoke__ubuntu-osworld",
                "instruction": "Observe the desktop once, then stop.",
                "config": [
                    {
                        "type": "execute",
                        "parameters": {
                            "command": [
                                "bash",
                                "-lc",
                                "printf 'setup-ok\\n' > /tmp/runner-osworld-setup-ok",
                            ],
                            "shell": False,
                            "until": {"returncode": 0},
                        },
                    }
                ],
                "evaluator": {
                    "func": "exact_match",
                    "result": {
                        "type": "vm_command_line",
                        "command": "cat /tmp/runner-osworld-setup-ok",
                    },
                },
            },
            indent=2,
        )
        + "\n"
    )
    return task_dir


def _write_unsupported_cookbook_task(root: Path) -> Path:
    task_dir = root / "use-computer-cuagym-task"
    task_dir.mkdir()
    (task_dir / "instruction.md").write_text("Take one screenshot, then stop.\n")
    (task_dir / "task.toml").write_text(
        """\
[metadata]
author_name = "Use.Computer"
difficulty = "smoke"
category = "desktop-automation"
tags = ["cuagym", "ubuntu", "smoke"]

[verifier]
timeout_sec = 180

[agent]
timeout_sec = 180

[environment]
cpus = 4
memory_mb = 8192
allow_internet = true
"""
    )
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text("#!/bin/bash\necho unsupported\n")
    return task_dir


def _write_iosworld_repo(root: Path) -> Path:
    task_dir = root / "iosworld"
    (task_dir / "scripts").mkdir(parents=True)
    (task_dir / "iphone" / "bootstrap").mkdir(parents=True)
    (task_dir / "tasks.json").write_text(
        json.dumps(
            [
                {
                    "name": "clock-001",
                    "goal": "Set a new alarm for 6:45 AM labeled Gym.",
                    "apps": ["clock"],
                    "category": "single_app",
                    "difficulty": "easy",
                    "rubric": [{"criterion": "Save the alarm"}],
                }
            ],
            indent=2,
        )
        + "\n"
    )
    (task_dir / "scripts" / "run_task_by_id.sh").write_text("#!/usr/bin/env bash\n")
    (task_dir / "iphone" / "bootstrap" / "bootstrap_ios_apps.sh").write_text(
        "#!/usr/bin/env bash\n"
    )
    return task_dir


def _write_unsupported_cuagym_postconfig_task(root: Path) -> Path:
    task_dir = root / "vscode__postconfig-task"
    original = task_dir / "tests" / "cuagym" / "original"
    setup_files = task_dir / "tests" / "setup" / "files" / "original"
    original.mkdir(parents=True)
    setup_files.mkdir(parents=True)
    task_json = {
        "evaluator": {
            "type": "python",
            "url": "./reward.py",
            "postconfig": [
                {
                    "type": "execute",
                    "parameters": {"command": ["python", "-c", "print('x')"]},
                }
            ],
        },
        "config": [],
        "id": "postconfig-task",
        "instruction": "Observe the desktop.",
        "app_type": "vscode",
    }
    for dest in (original, setup_files):
        (dest / "task.json").write_text(json.dumps(task_json) + "\n")
        (dest / "reward.py").write_text("print('REWARD: 0.0')\n")
    (task_dir / "instruction.md").write_text("Observe the desktop.\n")
    (task_dir / "task.toml").write_text(
        """\
[metadata]
author_name = "CUA-Gym"
difficulty = "smoke"
category = "desktop-automation"
tags = ["cuagym", "vscode", "smoke"]

[verifier]
timeout_sec = 180

[agent]
timeout_sec = 180

[environment]
cpus = 4
memory_mb = 8192
allow_internet = true
"""
    )
    return task_dir


def test_tasks_check_materializes_supported_foreign_task_for_cua(
    tmp_path: Path,
) -> None:
    """Guards BenchFlow 0.7 adapter checks for supported OSWorld cookbook slices."""
    task_dir = _write_cookbook_osworld_task(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "tasks",
            "check",
            str(task_dir),
            "--level",
            "runtime-capability",
            "--sandbox",
            "cua",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "valid (runtime-capability)" in result.output
    assert "use-computer-cookbook" in result.output


def test_tasks_check_json_reports_supported_foreign_task_for_cua(
    tmp_path: Path,
) -> None:
    """JSON task checks give adapter-adoption loops a parseable success record."""
    task_dir = _write_cookbook_osworld_task(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "tasks",
            "check",
            str(task_dir),
            "--level",
            "runtime-capability",
            "--sandbox",
            "cua",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {
        "status": "valid",
        "task": str(task_dir),
        "task_name": task_dir.name,
        "adapter": "use-computer-cookbook",
        "validation_level": "runtime-capability",
        "sandbox": "cua",
        "issues": [],
    }


def test_tasks_check_reports_unsupported_foreign_task_reason(
    tmp_path: Path,
) -> None:
    """Guards BenchFlow 0.7 unsupported-task reporting for cookbook adapters."""
    task_dir = _write_unsupported_cookbook_task(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "tasks",
            "check",
            str(task_dir),
            "--level",
            "runtime-capability",
            "--sandbox",
            "cua",
        ],
    )

    assert result.exit_code == 1
    assert "unsupported adapter task" in result.output
    assert "Adapter: use-computer-cookbook" in result.output
    assert "Dataset: cuagym" in result.output
    assert "CUA-Gym cookbook tasks need provider-honest setup/runtime" in result.output
    assert "Tags: cuagym, ubuntu, smoke" in result.output
    assert "Traceback (most recent call last)" not in result.output


def test_tasks_check_json_reports_unsupported_foreign_task_reason(
    tmp_path: Path,
) -> None:
    """JSON task checks preserve structured unsupported-task reasons."""
    task_dir = _write_unsupported_cookbook_task(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "tasks",
            "check",
            str(task_dir),
            "--level",
            "runtime-capability",
            "--sandbox",
            "cua",
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "unsupported-adapter-task"
    assert payload["task"] == str(task_dir)
    assert payload["task_name"] == task_dir.name
    assert payload["adapter"] == "use-computer-cookbook"
    assert payload["dataset"] == "cuagym"
    assert payload["reason"].startswith(
        "CUA-Gym cookbook tasks need provider-honest setup/runtime"
    )
    assert payload["details"] == {"tags": ["cuagym", "ubuntu", "smoke"]}


def test_tasks_check_reports_raw_cuagym_issue_details(tmp_path: Path) -> None:
    """Unsupported raw CUA-Gym reports include the concrete adapter blocker."""
    task_dir = _write_unsupported_cuagym_postconfig_task(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "tasks",
            "check",
            str(task_dir),
            "--level",
            "runtime-capability",
            "--sandbox",
            "cua",
        ],
    )

    assert result.exit_code == 1
    assert "unsupported adapter task" in result.output
    assert "Task: postconfig-task" in result.output
    assert "Issue: unsupported-evaluator-postconfig" in result.output
    assert "Postconfig Kinds: execute" in result.output


def test_tasks_check_reports_iosworld_provider_requirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """iOSWorld sources are recognized but blocked on a Mac/iOS provider."""
    _force_ios_unsupported(monkeypatch)
    task_dir = _write_iosworld_repo(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "tasks",
            "check",
            str(task_dir),
            "--level",
            "runtime-capability",
            "--sandbox",
            "cua",
        ],
    )

    assert result.exit_code == 1
    assert "unsupported adapter task" in result.output
    assert "Adapter: iosworld" in result.output
    assert "Dataset: iosworld" in result.output
    assert "macOS/iOS Simulator provider mapping" in result.output
    assert "Issue: macos-ios-simulator-provider-required" in result.output
    assert "Task Count: 1" in result.output
    assert "Traceback (most recent call last)" not in result.output


def test_tasks_check_json_reports_iosworld_provider_requirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """iOSWorld JSON reports carry provider-honest unsupported metadata."""
    _force_ios_unsupported(monkeypatch)
    task_dir = _write_iosworld_repo(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "tasks",
            "check",
            str(task_dir),
            "--level",
            "runtime-capability",
            "--sandbox",
            "cua",
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "unsupported-adapter-task"
    assert payload["adapter"] == "iosworld"
    assert payload["dataset"] == "iosworld"
    assert payload["reason"] == (
        "iOSWorld tasks require a macOS/iOS Simulator provider mapping"
    )
    assert payload["details"]["issue"] == "macos-ios-simulator-provider-required"
    assert payload["details"]["required_provider"] == "macos-ios-simulator"
    assert payload["details"]["shape"] == "repository"
    assert payload["details"]["task_count"] == 1
    assert payload["details"]["required_capabilities"] == [
        "macos",
        "xcode-26",
        "ios-26-simulator-runtime",
        "appium-xcuitest",
        "iosworld-app-bootstrap",
    ]
