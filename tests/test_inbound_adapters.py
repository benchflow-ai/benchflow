"""Tests for inbound environment adapters (capability #8 — the edges).

Inbound adapters translate a foreign benchmark's task directory into
BenchFlow's native task format. Covered here:

- ``HarborAdapter`` — native split ``task.toml`` task dirs (near-identity;
  BenchFlow's own ``TaskConfig`` uses the same split shape).
- The shared ``InboundTask`` result type and ``detect_adapter`` dispatch.

These are pure format translators — they read a task directory and return
in-memory BenchFlow native models. No sandbox, no runtime.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from benchflow._utils.task_authoring import check_task
from benchflow.adapters.browser_use import (
    BrowserUseAdapter,
    load_encrypted_benchmark_tasks,
    official_task_descriptor,
)
from benchflow.adapters.computer_use import ComputerUseAdapter
from benchflow.adapters.harbor import HarborAdapter
from benchflow.adapters.inbound import (
    InboundTask,
    UnsupportedInboundTaskError,
    detect_adapter,
    materialize_inbound_task_md,
)
from benchflow.adapters.iosworld import IOSWorldAdapter
from benchflow.adapters.macosworld import MacOSWorldAdapter
from benchflow.adapters.stagehand import (
    StagehandEvalAdapter,
    official_task_descriptor_from_source,
    support_report_from_source,
)
from benchflow.adapters.use_computer_cookbook import UseComputerCookbookAdapter
from benchflow.task import TaskDocument
from benchflow.task.config import TaskConfig

# Fixtures — minimal foreign task dirs written to tmp_path

_HARBOR_TASK_TOML = """\
schema_version = "1.0"

[task]
name = "openmoss/abc-bench__widget"
authors = [
    { name = "Jie Yang", email = "yangj24@example.com" },
    { name = "Li Ji" },
]
keywords = ["abc-bench", "backend"]

[metadata]
difficulty = "medium"
category = "programming"

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 300.0

[environment]
build_timeout_sec = 600.0
cpus = 2
memory_mb = 4096
"""

_BROWSER_USE_TASK_JSON = {
    "task_id": "Open Local Page",
    "benchmark": "browser-use",
    "category": "local-browser",
    "confirmed_task": (
        "Use the browser fixture to report the page status. Final answer must "
        "be exactly: browser-use-smoke: ready"
    ),
    "ground_truth": "Final result must be exactly browser-use-smoke: ready",
    "expected_result": "browser-use-smoke: ready",
    "url": "file:///app/browser_fixture/index.html",
    "timeout_sec": 120,
}

_BROWSER_USE_OFFICIAL_TASK = {
    "task_id": "66c6641b-f949-46a2-8bcc-6d9dd388b534",
    "category": "WebBenchREAD",
    "confirmed_task": (
        "Browse the list of active Q&A communities on https://stackexchange.com "
        "and list the names of the top 5 communities by current activity."
    ),
}

_STAGEHAND_SIGN_IN_SOURCE = """\
import { defineBenchTask } from "../../../framework/defineTask.js";

export default defineBenchTask(
  { name: "agent/sign_in" },
  async ({ debugUrl, sessionUrl, logger, agent, v3 }) => {
    const page = v3.context.pages()[0];
    await page.goto("https://v0-modern-login-flow.vercel.app/");
    const agentResult = await agent.execute({
      instruction:
        "Sign in with the email address 'test@browserbaser.com' and the password 'stagehand=goated' ",
      maxSteps: Number(process.env.AGENT_EVAL_MAX_STEPS) || 15,
    });
    logger.log(agentResult);
    const url = page.url();
    if (url === "https://v0-modern-login-flow.vercel.app/authorized") {
      return { _success: true, observations: url, debugUrl, sessionUrl };
    }
    return { _success: false, observations: url, debugUrl, sessionUrl };
  },
);
"""

_STAGEHAND_STEAM_SOURCE = """\
import { defineBenchTask } from "../../../framework/defineTask.js";

export default defineBenchTask(
  { name: "agent/steam_games" },
  async ({ debugUrl, sessionUrl, logger, agent, v3 }) => {
    const page = v3.context.pages()[0];
    await page.goto("https://store.steampowered.com/");
    await agent.execute({
      instruction:
        "Show most played games in Steam. And tell me the number of players in game at this time",
      maxSteps: Number(process.env.AGENT_EVAL_MAX_STEPS) || 30,
    });
    const success = page.url().includes("https://store.steampowered.com/");
    return { _success: success, debugUrl, sessionUrl, logs: logger.getLogs() };
  },
);
"""

_STAGEHAND_DYNAMIC_VERIFIER_SOURCE = """\
import { defineBenchTask } from "../../../framework/defineTask.js";
import { runWithVerifier } from "../../../framework/verifierAdapter.js";

export default defineBenchTask(
  { name: "agent/dynamic_verifier" },
  async ({ agent, v3 }) => {
    const instruction = "Solve a dynamic verifier task.";
    await runWithVerifier({ v3, agent, taskSpec: { id: "agent/dynamic_verifier", instruction } });
    return { _success: true };
  },
);
"""

_STAGEHAND_EXPECTED_ANSWER_VERIFIER_SOURCE = """\
import { defineBenchTask } from "../../../framework/defineTask.js";
import { runWithVerifier } from "../../../framework/verifierAdapter.js";

export default defineBenchTask(
  { name: "agent/expected_answer" },
  async ({ agent, v3 }) => {
    const instruction = "Find the answer.";
    const expected = "42";
    await runWithVerifier({
      v3,
      agent,
      taskSpec: { id: "agent/expected_answer", instruction, expectedAnswer: expected },
    });
    return { _success: true };
  },
);
"""

_COMPUTER_USE_TASK_JSON = {
    "task_id": "Desktop File Roundtrip",
    "benchmark": "computer-use-smoke",
    "category": "desktop",
    "instruction": (
        "Use the desktop environment to write the expected file. Final answer "
        "must be exactly: computer-use-smoke: ready"
    ),
    "expected_result": "computer-use-smoke: ready",
    "expected_file": "/app/computer_use_result.txt",
    "roundtrip_file": "/app/computer_use_roundtrip.txt",
    "screenshot_required": True,
    "workdir": "/app",
    "timeout_sec": 180,
}

_IOSWORLD_TASK_JSON = {
    "name": "clock-001",
    "goal": "Set a new alarm for 6:45 AM labeled 'Gym' and confirm it is set.",
    "apps": ["clock"],
    "category": "single_app",
    "difficulty": "easy",
    "rubric": [
        {"criterion": "Open Clock app"},
        {"criterion": "Save the alarm"},
    ],
}

# A real macOSWorld task slice (github.com/showlab/macosworld): the upstream
# `task` is a per-language mapping and `grading_command` is a list of
# [command, timeout] pairs producing True/False.
_MACOSWORLD_TASK_JSON = {
    "id": "48cf0af3-0612-dbcd-14da-d5202eed6ce9",
    "snapshot": {"en": "snapshot_used_en", "zh": "snapshot_used_zh"},
    "force_snapshot_recovery": True,
    "task": {
        "en": "Add Ong KC to contact with mobile number 96910380.",
        "zh": "将Ong KC添加到联系人。",
    },
    "before_action_delay_seconds": 10,
    "before_grading_delay_seconds": 30,
    "grading_command": [
        [
            "osascript -e 'tell application \"Contacts\" to get phones' "
            '| grep -q "96910380" && echo "True" || echo "False"',
            100,
        ],
    ],
}


def _write_harbor_task(root: Path) -> Path:
    """Create a Harbor-style task dir; return its path."""
    task_dir = root / "harbor-task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(_HARBOR_TASK_TOML)
    (task_dir / "instruction.md").write_text("Build the widget service.\n")
    env = task_dir / "environment"
    env.mkdir()
    (env / "Dockerfile").write_text("FROM python:3.12-slim\n")
    solution = task_dir / "solution"
    solution.mkdir()
    (solution / "solve.sh").write_text("#!/bin/bash\necho solved\n")
    tests = task_dir / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text(
        "#!/bin/bash\n"
        "mkdir -p /logs/verifier\n"
        "printf '1.0' > /logs/verifier/reward.txt\n"
        "printf '{\"reward\": 1.0}\\n' > /logs/verifier/reward.json\n"
    )
    return task_dir


def _write_browser_use_task(root: Path) -> Path:
    """Create a Browser Use-shaped task slice; return its path."""
    task_dir = root / "browser-use-task"
    task_dir.mkdir()
    (task_dir / "browser-use-task.json").write_text(
        json.dumps(_BROWSER_USE_TASK_JSON, indent=2) + "\n"
    )
    env = task_dir / "environment"
    env.mkdir()
    (env / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
        "WORKDIR /app\n"
        "RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts "
        "/app/browser_fixture\n"
        "COPY browser_fixture/ /app/browser_fixture/\n"
    )
    fixture = env / "browser_fixture"
    fixture.mkdir()
    (fixture / "index.html").write_text("<p>browser-use-smoke: ready</p>\n")
    solution = task_dir / "solution"
    solution.mkdir()
    (solution / "solve.sh").write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "printf 'browser-use-smoke: ready\\n' > /app/final_result.txt\n"
    )
    tests = task_dir / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "mkdir -p /logs/verifier /logs/artifacts\n"
        "if [ -f /app/final_result.txt ] && "
        '[ "$(tr -d \'\\n\' < /app/final_result.txt)" = "browser-use-smoke: ready" ]; '
        "then reward=1.0; else reward=0.0; fi\n"
        'printf "%s\\n" "$reward" > /logs/verifier/reward.txt\n'
        'printf \'{"reward": %s}\\n\' "$reward" > /logs/verifier/reward.json\n'
    )
    return task_dir


def _write_stagehand_task(root: Path) -> Path:
    descriptor = official_task_descriptor_from_source(
        _STAGEHAND_SIGN_IN_SOURCE,
        source_file="packages/evals/tasks/bench/agent/sign_in.ts",
        upstream_commit="f2873cd",
    )
    task_dir = root / "stagehand-task"
    task_dir.mkdir()
    (task_dir / "stagehand-task.json").write_text(
        json.dumps(descriptor, indent=2) + "\n"
    )
    (task_dir / "environment").mkdir()
    (task_dir / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    return task_dir


def _write_computer_use_task(root: Path) -> Path:
    """Create a computer-use-shaped task slice; return its path."""
    task_dir = root / "computer-use-task"
    task_dir.mkdir()
    (task_dir / "computer-use-task.json").write_text(
        json.dumps(_COMPUTER_USE_TASK_JSON, indent=2) + "\n"
    )
    env = task_dir / "environment"
    env.mkdir()
    (env / "Dockerfile").write_text(
        "FROM ubuntu:24.04\n"
        "WORKDIR /app\n"
        "RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts /app\n"
    )
    solution = task_dir / "solution"
    solution.mkdir()
    (solution / "solve.sh").write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "printf 'computer-use-smoke: ready\\n' > /app/computer_use_result.txt\n"
    )
    tests = task_dir / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "mkdir -p /logs/verifier\n"
        "printf '1.0\\n' > /logs/verifier/reward.txt\n"
        "printf '{\"reward\": 1.0}\\n' > /logs/verifier/reward.json\n"
    )
    return task_dir


def _write_iosworld_repo(root: Path) -> Path:
    """Create an iOSWorld repository-shaped source; return its path."""
    repo = root / "iosworld"
    (repo / "scripts").mkdir(parents=True)
    (repo / "iphone" / "bootstrap").mkdir(parents=True)
    (repo / "tasks.json").write_text(json.dumps([_IOSWORLD_TASK_JSON], indent=2) + "\n")
    (repo / "scripts" / "run_task_by_id.sh").write_text("#!/usr/bin/env bash\n")
    (repo / "iphone" / "bootstrap" / "bootstrap_ios_apps.sh").write_text(
        "#!/usr/bin/env bash\n"
    )
    return repo


def _write_iosworld_task_slice(root: Path) -> Path:
    """Create an iOSWorld single-task slice; return its path."""
    task_dir = root / "iosworld-task"
    task_dir.mkdir()
    (task_dir / "iosworld-task.json").write_text(
        json.dumps(_IOSWORLD_TASK_JSON, indent=2) + "\n"
    )
    return task_dir


def _write_macosworld_repo(root: Path) -> Path:
    """Create a macOSWorld repository-shaped source; return its path."""
    repo = root / "macosworld"
    (repo / "tasks" / "sys_apps").mkdir(parents=True)
    (repo / "tasks" / "productivity").mkdir(parents=True)
    (repo / "testbench.py").write_text("# macOSWorld testbench\n")
    (repo / "constants.py").write_text("SCREEN_WIDTH = 1024\n")
    (repo / "tasks" / "sys_apps" / f"{_MACOSWORLD_TASK_JSON['id']}.json").write_text(
        json.dumps(_MACOSWORLD_TASK_JSON, indent=2) + "\n"
    )
    return repo


def _write_macosworld_task_slice(root: Path) -> Path:
    """Create a macOSWorld single-task slice; return its path."""
    task_dir = root / "macosworld-task"
    task_dir.mkdir()
    (task_dir / "macosworld-task.json").write_text(
        json.dumps(_MACOSWORLD_TASK_JSON, indent=2) + "\n"
    )
    return task_dir


def _write_use_computer_cookbook_osworld_task(root: Path) -> Path:
    """Create a use-computer cookbook OSWorld smoke task; return its path."""
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
    tests = task_dir / "tests"
    tests.mkdir()
    (tests / "osworld_task.json").write_text(
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


def _write_use_computer_cookbook_cuagym_smoke_task(root: Path) -> Path:
    """Create a use-computer cookbook CUA-Gym infra-smoke task; return its path."""
    task_dir = root / "smoke__ubuntu-infra"
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
    setup = task_dir / "tests" / "setup"
    setup.mkdir(parents=True)
    (setup / "pre_command.sh").write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "printf 'setup-ok\\n' > /tmp/runner-cuagym-setup-ok\n"
    )
    tests = task_dir / "tests"
    (tests / "test.sh").write_text("#!/bin/bash\necho upstream-verifier-placeholder\n")
    env = task_dir / "environment"
    env.mkdir()
    (env / ".gitkeep").write_text("")
    return task_dir


def _write_use_computer_cookbook_cuagym_python_task(
    root: Path,
    *,
    setup_kind: str = "execute",
) -> Path:
    """Create a no-mock CUA-Gym Python reward task; return its path."""
    task_dir = root / "vscode__d0641d59-1751-5d45-8ced-5e45d615a68c"
    original = task_dir / "tests" / "cuagym" / "original"
    setup_files = task_dir / "tests" / "setup" / "files" / "original"
    original.mkdir(parents=True)
    setup_files.mkdir(parents=True)
    task_json = {
        "evaluator": {"type": "python", "url": "./reward.py"},
        "config": [
            {
                "type": "download",
                "parameters": {
                    "files": [
                        {
                            "url": "./initial_setup.py",
                            "path": "/home/user/initial_setup.py",
                        }
                    ]
                },
            },
            {
                "type": setup_kind,
                "parameters": {"command": "python3 /home/user/initial_setup.py"},
            },
        ],
        "id": "d0641d59-1751-5d45-8ced-5e45d615a68c",
        "difficulty": "medium",
        "instruction": "Use Go to Definition to navigate to calculateTax.",
        "app_type": "vscode",
    }
    setup_py = (
        "from pathlib import Path\n"
        "Path('/home/user/project').mkdir(parents=True, exist_ok=True)\n"
        "Path('/home/user/project/tax.js').write_text('function calculateTax() {}\\n')\n"
        "Path('/home/user/project/main.js').write_text('calculateTax();\\n')\n"
    )
    reward_py = (
        "import json, os, sqlite3\n"
        "path = '/home/user/.config/Code/User/workspaceStorage/state.vscdb'\n"
        "reward = 1.0 if os.path.exists(path) else 0.0\n"
        "print(f'REWARD: {reward}')\n"
    )
    for dest in (original, setup_files):
        (dest / "task.json").write_text(json.dumps(task_json, indent=2) + "\n")
        (dest / "initial_setup.py").write_text(setup_py)
        (dest / "reward.py").write_text(reward_py)
    (task_dir / "instruction.md").write_text(task_json["instruction"] + "\n")
    (task_dir / "task.toml").write_text(
        """\
[metadata]
author_name = "CUA-Gym"
difficulty = "medium"
category = "desktop-automation"
tags = ["cuagym", "vscode", "medium"]

[verifier]
timeout_sec = 600

[agent]
timeout_sec = 600

[environment]
cpus = 4
memory_mb = 8192
allow_internet = true
"""
    )
    return task_dir


def _write_use_computer_cookbook_tagged_task(
    root: Path,
    *,
    dataset: str,
    tags: list[str],
) -> Path:
    """Create a tagged but unsupported cookbook task; return its path."""
    task_dir = root / f"use-computer-{dataset}-task"
    task_dir.mkdir()
    (task_dir / "instruction.md").write_text("Take one screenshot, then stop.\n")
    tags_toml = ", ".join(json.dumps(tag) for tag in tags)
    (task_dir / "task.toml").write_text(
        f"""\
[metadata]
author_name = "Use.Computer"
difficulty = "smoke"
category = "desktop-automation"
tags = [{tags_toml}]

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
    tests = task_dir / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text("#!/bin/bash\necho unsupported\n")
    return task_dir


# HarborAdapter


class TestHarborAdapter:
    def test_returns_inbound_task(self, tmp_path: Path) -> None:
        task_dir = _write_harbor_task(tmp_path)
        result = HarborAdapter.from_task_dir(task_dir)
        assert isinstance(result, InboundTask)

    def test_preserves_task_identity(self, tmp_path: Path) -> None:
        task_dir = _write_harbor_task(tmp_path)
        result = HarborAdapter.from_task_dir(task_dir)
        assert result.name == "openmoss/abc-bench__widget"
        assert result.source == "harbor"

    def test_config_is_native_task_config(self, tmp_path: Path) -> None:
        task_dir = _write_harbor_task(tmp_path)
        result = HarborAdapter.from_task_dir(task_dir)
        assert isinstance(result.config, TaskConfig)
        # Harbor's [environment] section maps to BenchFlow's sandbox.
        assert result.config.sandbox.cpus == 2
        assert result.config.sandbox.memory_mb == 4096
        assert result.config.verifier.timeout_sec == 120.0
        assert result.config.agent.timeout_sec == 300.0

    def test_instruction_read_from_file(self, tmp_path: Path) -> None:
        task_dir = _write_harbor_task(tmp_path)
        result = HarborAdapter.from_task_dir(task_dir)
        assert result.instruction == "Build the widget service.\n"

    def test_authors_and_keywords_preserved(self, tmp_path: Path) -> None:
        task_dir = _write_harbor_task(tmp_path)
        result = HarborAdapter.from_task_dir(task_dir)
        assert result.config.task is not None
        assert [a.name for a in result.config.task.authors] == ["Jie Yang", "Li Ji"]
        assert "abc-bench" in result.config.task.keywords

    def test_unknown_extension_keys_are_preserved_not_native(
        self, tmp_path: Path
    ) -> None:
        task_dir = _write_harbor_task(tmp_path)
        config_path = task_dir / "task.toml"
        config_path.write_text(
            'harbor_ext = "kept"\n'
            + config_path.read_text()
            + """
[[steps]]
name = "phase-one"
runner = "harbor-step-runner"

[environment.modal]
image = "registry.example.com/task:latest"

[verifier.reward_kit]
metric = "exact_match"
"""
        )

        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            TaskConfig.model_validate_toml(config_path.read_text())

        result = HarborAdapter.from_task_dir(task_dir)

        assert result.config.sandbox.cpus == 2
        assert result.compatibility is not None
        assert result.compatibility.config_extra == {
            "harbor_ext": "kept",
            "environment": {
                "modal": {"image": "registry.example.com/task:latest"},
            },
            "steps": [{"runner": "harbor-step-runner"}],
            "verifier": {"reward_kit": {"metric": "exact_match"}},
        }
        assert result.compatibility.config_extra_paths == (
            "environment.modal.image",
            "harbor_ext",
            "steps[0].runner",
            "verifier.reward_kit.metric",
        )
        assert result.config.steps is not None
        assert result.config.steps[0].name == "phase-one"

    def test_file_map_points_at_real_paths(self, tmp_path: Path) -> None:
        task_dir = _write_harbor_task(tmp_path)
        result = HarborAdapter.from_task_dir(task_dir)
        # The Dockerfile and verifier script are carried as a relative-path map.
        assert (
            result.files["environment/Dockerfile"]
            == task_dir / "environment" / "Dockerfile"
        )
        assert result.files["tests/test.sh"] == task_dir / "tests" / "test.sh"
        assert result.files["solution/solve.sh"] == task_dir / "solution" / "solve.sh"
        for src in result.files.values():
            assert src.exists()

    def test_missing_task_toml_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            HarborAdapter.from_task_dir(empty)

    def test_missing_instruction_raises(self, tmp_path: Path) -> None:
        task_dir = _write_harbor_task(tmp_path)
        (task_dir / "instruction.md").unlink()
        with pytest.raises(FileNotFoundError):
            HarborAdapter.from_task_dir(task_dir)


# BrowserUseAdapter


class TestBrowserUseAdapter:
    def test_returns_inbound_task(self, tmp_path: Path) -> None:
        task_dir = _write_browser_use_task(tmp_path)
        result = BrowserUseAdapter.from_task_dir(task_dir)
        assert isinstance(result, InboundTask)

    def test_name_is_slugged_from_task_id(self, tmp_path: Path) -> None:
        task_dir = _write_browser_use_task(tmp_path)
        result = BrowserUseAdapter.from_task_dir(task_dir)
        assert result.name == "open-local-page"
        assert result.source == "browser-use-benchmark"

    def test_instruction_uses_confirmed_task(self, tmp_path: Path) -> None:
        task_dir = _write_browser_use_task(tmp_path)
        result = BrowserUseAdapter.from_task_dir(task_dir)
        assert result.instruction.startswith("Use the browser fixture")
        assert result.instruction.endswith("\n")

    def test_config_is_native_and_namespaced(self, tmp_path: Path) -> None:
        task_dir = _write_browser_use_task(tmp_path)
        result = BrowserUseAdapter.from_task_dir(task_dir)
        assert isinstance(result.config, TaskConfig)
        assert result.config.task is not None
        assert result.config.task.name == "browser-use/open-local-page"
        assert result.config.metadata["browser_use"]["expected_result"] == (
            "browser-use-smoke: ready"
        )
        assert result.config.artifacts == []

    def test_official_task_descriptor_can_use_llm_judge_verifier(
        self, tmp_path: Path
    ) -> None:
        task_dir = tmp_path / "browser-use-official"
        task_dir.mkdir()
        descriptor = official_task_descriptor(
            _BROWSER_USE_OFFICIAL_TASK,
            benchmark="BU_Bench_V1",
            task_index=0,
            judge_model="gemini-2.5-flash",
        )
        (task_dir / "browser-use-task.json").write_text(
            json.dumps(descriptor, indent=2) + "\n"
        )
        tests_dir = task_dir / "tests"
        tests_dir.mkdir()
        (tests_dir / "rubric.toml").write_text(
            '[[criterion]]\nname = "ok"\ndescription = "Judge output."\n'
        )

        result = BrowserUseAdapter.from_task_dir(task_dir)

        assert result.config.verifier.type == "llm-judge"
        assert result.config.verifier.judge.model == "gemini-2.5-flash"
        assert result.config.verifier.judge.rubric_path == "tests/rubric.toml"
        assert result.config.verifier.judge.input_dir == "/logs/artifacts"
        assert result.config.verifier.env == {"GEMINI_API_KEY": "${GEMINI_API_KEY}"}
        assert "ground_truth" not in result.config.metadata["browser_use"]
        assert result.compatibility is not None
        assert result.compatibility.config_extra["upstream_task_index"] == 0
        assert result.files["tests/rubric.toml"] == tests_dir / "rubric.toml"

    def test_load_encrypted_benchmark_tasks_matches_browser_use_order(
        self, tmp_path: Path
    ) -> None:
        tasks = [
            {
                "task_id": f"task-{index}",
                "category": "WebBenchREAD",
                "confirmed_task": f"Task {index}",
            }
            for index in range(100)
        ]
        encrypted_file = tmp_path / "BU_Bench_V1.enc"
        key = base64.urlsafe_b64encode(hashlib.sha256(b"BU_Bench_V1").digest())
        encrypted_file.write_text(
            base64.b64encode(Fernet(key).encrypt(json.dumps(tasks).encode())).decode()
        )

        loaded = load_encrypted_benchmark_tasks(encrypted_file, benchmark="BU_Bench_V1")
        raw_order = load_encrypted_benchmark_tasks(
            encrypted_file,
            benchmark="BU_Bench_V1",
            interleave=False,
        )

        assert [task["task_id"] for task in loaded[:6]] == [
            "task-0",
            "task-20",
            "task-40",
            "task-60",
            "task-80",
            "task-1",
        ]
        assert [task["task_id"] for task in raw_order[:3]] == [
            "task-0",
            "task-1",
            "task-2",
        ]

    def test_native_subtrees_carried(self, tmp_path: Path) -> None:
        task_dir = _write_browser_use_task(tmp_path)
        result = BrowserUseAdapter.from_task_dir(task_dir)
        assert result.files["environment/Dockerfile"] == (
            task_dir / "environment" / "Dockerfile"
        )
        assert result.files["environment/browser_fixture/index.html"] == (
            task_dir / "environment" / "browser_fixture" / "index.html"
        )
        assert result.files["tests/test.sh"] == task_dir / "tests" / "test.sh"
        assert result.files["solution/solve.sh"] == (task_dir / "solution" / "solve.sh")

    def test_compatibility_preserves_foreign_descriptor(self, tmp_path: Path) -> None:
        task_dir = _write_browser_use_task(tmp_path)
        result = BrowserUseAdapter.from_task_dir(task_dir)
        assert result.compatibility is not None
        assert result.compatibility.source == "browser-use-benchmark"
        assert result.compatibility.config_extra["task_id"] == "Open Local Page"
        assert result.compatibility.config_extra_paths == ("browser-use-task.json",)

    def test_missing_descriptor_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            BrowserUseAdapter.from_task_dir(empty)

    def test_missing_instruction_raises(self, tmp_path: Path) -> None:
        task_dir = _write_browser_use_task(tmp_path)
        payload = dict(_BROWSER_USE_TASK_JSON)
        payload.pop("confirmed_task")
        (task_dir / "browser-use-task.json").write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="instruction"):
            BrowserUseAdapter.from_task_dir(task_dir)


# StagehandEvalAdapter


class TestStagehandEvalAdapter:
    def test_official_source_descriptor_extracts_deterministic_agent_task(self) -> None:
        descriptor = official_task_descriptor_from_source(
            _STAGEHAND_SIGN_IN_SOURCE,
            source_file="packages/evals/tasks/bench/agent/sign_in.ts",
            upstream_commit="f2873cd",
        )

        assert descriptor["task_id"] == "agent/sign_in"
        assert descriptor["start_url"] == "https://v0-modern-login-flow.vercel.app/"
        assert "test@browserbaser.com" in descriptor["instruction"]
        assert descriptor["max_steps"] == 15
        assert descriptor["success_check"] == {
            "type": "url_exact",
            "value": "https://v0-modern-login-flow.vercel.app/authorized",
        }
        assert descriptor["original_runner"]["command"] == "evals run agent/sign_in"

    def test_official_source_descriptor_extracts_url_contains_task(self) -> None:
        descriptor = official_task_descriptor_from_source(_STAGEHAND_STEAM_SOURCE)

        assert descriptor["task_id"] == "agent/steam_games"
        assert descriptor["max_steps"] == 30
        assert descriptor["success_check"] == {
            "type": "url_contains",
            "value": "https://store.steampowered.com/",
        }

    def test_dynamic_verifier_source_reports_unsupported_mapping(self) -> None:
        report = support_report_from_source(
            _STAGEHAND_DYNAMIC_VERIFIER_SOURCE,
            source_file="packages/evals/tasks/bench/agent/dynamic_verifier.ts",
        )

        assert report.supported is False
        assert report.task_id == "agent/dynamic_verifier"
        assert report.details["issue"] == "stagehand-verifier-not-mapped"
        assert "required_mapping" in report.details

    def test_expected_answer_verifier_source_reports_unsupported_mapping(
        self,
    ) -> None:
        report = support_report_from_source(
            _STAGEHAND_EXPECTED_ANSWER_VERIFIER_SOURCE,
            source_file="packages/evals/tasks/bench/agent/expected_answer.ts",
        )

        assert report.supported is False
        assert report.task_id == "agent/expected_answer"
        assert (
            report.details["issue"] == "stagehand-expected-answer-verifier-not-mapped"
        )
        assert "verifier/reward contract" in str(report.reason)

    def test_returns_inbound_task_with_generated_url_verifier(
        self, tmp_path: Path
    ) -> None:
        task_dir = _write_stagehand_task(tmp_path)
        result = StagehandEvalAdapter.from_task_dir(task_dir)

        assert isinstance(result, InboundTask)
        assert result.name == "agent-sign_in"
        assert result.source == "stagehand-evals"
        assert result.config.task is not None
        assert result.config.task.name == "stagehand/agent-sign_in"
        assert result.config.metadata["stagehand"]["task_id"] == "agent/sign_in"
        assert result.config.metadata["stagehand"]["upstream_commit"] == "f2873cd"
        assert result.instruction.startswith(
            "Open https://v0-modern-login-flow.vercel.app/."
        )
        assert "Sign in with the email address" in result.instruction
        assert "Stagehand max browser steps: 15." in result.instruction
        assert "tests/test.sh" in result.generated_files
        generated_test = result.generated_files["tests/test.sh"]
        assert isinstance(generated_test, str)
        assert "stagehand_current_url" in generated_test
        assert "/logs/verifier/reward.json" in generated_test
        assert result.compatibility is not None
        assert result.compatibility.source == "stagehand-evals"
        assert result.compatibility.config_extra_paths == ("stagehand-task.json",)

    def test_detect_adapter_routes_stagehand_descriptor(self, tmp_path: Path) -> None:
        task_dir = _write_stagehand_task(tmp_path)

        assert detect_adapter(task_dir) is StagehandEvalAdapter

    def test_materialized_task_preserves_stagehand_compatibility(
        self, tmp_path: Path
    ) -> None:
        task_dir = _write_stagehand_task(tmp_path)
        inbound = StagehandEvalAdapter.from_task_dir(task_dir)
        native_dir = materialize_inbound_task_md(inbound, tmp_path / "native-stagehand")

        assert (native_dir / "verifier" / "test.sh").is_file()
        document = TaskDocument.from_path(native_dir / "task.md")
        compat = document.frontmatter["benchflow"]["compat"]
        assert compat["source"] == "stagehand-evals"
        assert compat["config_extra"]["task_id"] == "agent/sign_in"
        assert check_task(native_dir, validation_level="structural") == []

    def test_missing_instruction_raises(self, tmp_path: Path) -> None:
        task_dir = _write_stagehand_task(tmp_path)
        payload = json.loads((task_dir / "stagehand-task.json").read_text())
        payload.pop("instruction")
        (task_dir / "stagehand-task.json").write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="instruction"):
            StagehandEvalAdapter.from_task_dir(task_dir)


# ComputerUseAdapter


class TestComputerUseAdapter:
    def test_returns_inbound_task(self, tmp_path: Path) -> None:
        task_dir = _write_computer_use_task(tmp_path)
        result = ComputerUseAdapter.from_task_dir(task_dir)
        assert isinstance(result, InboundTask)

    def test_name_is_slugged_from_task_id(self, tmp_path: Path) -> None:
        task_dir = _write_computer_use_task(tmp_path)
        result = ComputerUseAdapter.from_task_dir(task_dir)
        assert result.name == "desktop-file-roundtrip"
        assert result.source == "computer-use-benchmark"

    def test_config_is_native_and_namespaced(self, tmp_path: Path) -> None:
        task_dir = _write_computer_use_task(tmp_path)
        result = ComputerUseAdapter.from_task_dir(task_dir)
        assert isinstance(result.config, TaskConfig)
        assert result.config.task is not None
        assert result.config.task.name == "computer-use/desktop-file-roundtrip"
        assert result.config.sandbox.workdir == "/app"
        assert result.config.metadata["computer_use"]["expected_result"] == (
            "computer-use-smoke: ready"
        )

    def test_native_subtrees_carried(self, tmp_path: Path) -> None:
        task_dir = _write_computer_use_task(tmp_path)
        result = ComputerUseAdapter.from_task_dir(task_dir)
        assert result.files["environment/Dockerfile"] == (
            task_dir / "environment" / "Dockerfile"
        )
        assert result.files["tests/test.sh"] == task_dir / "tests" / "test.sh"
        assert result.files["solution/solve.sh"] == (task_dir / "solution" / "solve.sh")

    def test_compatibility_preserves_foreign_descriptor(self, tmp_path: Path) -> None:
        task_dir = _write_computer_use_task(tmp_path)
        result = ComputerUseAdapter.from_task_dir(task_dir)
        assert result.compatibility is not None
        assert result.compatibility.source == "computer-use-benchmark"
        assert result.compatibility.config_extra["task_id"] == (
            "Desktop File Roundtrip"
        )
        assert result.compatibility.config_extra_paths == ("computer-use-task.json",)

    def test_missing_descriptor_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            ComputerUseAdapter.from_task_dir(empty)


def _force_ios_caps(monkeypatch: pytest.MonkeyPatch, present: bool) -> None:
    """Pin the iOS-Simulator host capability probe to a deterministic value.

    The iOSWorld adapter delegates its supported/unsupported decision to
    ``detect_ios_simulator_capabilities``, which reads the real host. Tests
    pin it so they assert one path regardless of whether the host happens to
    have Xcode/Appium installed.
    """
    import benchflow.sandbox.macos_ios_simulator as ios_sim

    caps = dict.fromkeys(
        ("macos", "xcode-26", "ios-26-simulator-runtime", "appium-xcuitest"),
        present,
    )
    monkeypatch.setattr(ios_sim, "detect_ios_simulator_capabilities", lambda: caps)


class TestIOSWorldAdapter:
    """The provider-honest *unsupported* path (host lacks the iOS prereqs)."""

    @pytest.fixture(autouse=True)
    def _no_ios_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _force_ios_caps(monkeypatch, present=False)

    def test_support_report_for_repo_shape(self, tmp_path: Path) -> None:
        repo = _write_iosworld_repo(tmp_path)

        report = IOSWorldAdapter.support_report(repo)

        assert report is not None
        assert report.supported is False
        assert report.source == "iosworld"
        assert report.dataset == "iosworld"
        assert report.reason == (
            "iOSWorld tasks require a macOS/iOS Simulator provider mapping"
        )
        assert report.details["shape"] == "repository"
        assert report.details["required_provider"] == "macos-ios-simulator"
        assert report.details["task_count"] == 1
        assert report.details["categories"] == {"single_app": 1}
        assert report.details["required_capabilities"] == [
            "macos",
            "xcode-26",
            "ios-26-simulator-runtime",
            "appium-xcuitest",
            "iosworld-app-bootstrap",
        ]

    def test_support_report_for_task_slice_shape(self, tmp_path: Path) -> None:
        task_dir = _write_iosworld_task_slice(tmp_path)

        report = IOSWorldAdapter.support_report(task_dir)

        assert report is not None
        assert report.supported is False
        assert report.task_id == "clock-001"
        assert report.details["shape"] == "task-slice"
        assert report.details["required_provider"] == "macos-ios-simulator"
        assert report.details["apps"] == ["clock"]
        assert report.details["rubric_count"] == 2

    def test_from_task_dir_reports_provider_requirement(self, tmp_path: Path) -> None:
        repo = _write_iosworld_repo(tmp_path)

        with pytest.raises(
            UnsupportedInboundTaskError,
            match="macOS/iOS Simulator provider mapping",
        ) as exc:
            IOSWorldAdapter.from_task_dir(repo)

        assert exc.value.report.source == "iosworld"
        assert exc.value.report.details["issue"] == (
            "macos-ios-simulator-provider-required"
        )


class TestIOSWorldAdapterSupported:
    """The provider-honest *supported* path (host advertises the iOS prereqs)."""

    @pytest.fixture(autouse=True)
    def _ios_host_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _force_ios_caps(monkeypatch, present=True)

    def test_task_slice_reports_supported(self, tmp_path: Path) -> None:
        task_dir = _write_iosworld_task_slice(tmp_path)

        report = IOSWorldAdapter.support_report(task_dir)

        assert report is not None
        assert report.supported is True
        assert report.task_id == "clock-001"
        assert report.dataset == "iosworld"
        assert report.reason is None
        assert report.details["shape"] == "task-slice"
        assert report.details["provider"] == "macos-ios-simulator"
        assert report.details["apps"] == ["clock"]
        assert report.details["rubric_count"] == 2
        # App bootstrap is a follow-up step, not a host capability — it is
        # reported as pending, never silently claimed as done.
        assert report.details["pending_capabilities"] == ["iosworld-app-bootstrap"]

    def test_from_task_dir_translates_slice(self, tmp_path: Path) -> None:
        task_dir = _write_iosworld_task_slice(tmp_path)

        task = IOSWorldAdapter.from_task_dir(task_dir)

        assert isinstance(task, InboundTask)
        assert task.name == "clock-001"
        assert task.source == "iosworld"
        assert "alarm" in task.instruction.lower()
        # The rubric maps to an LLM-judge verifier, mirroring the Browser
        # Use / computer-use criteria reward shape.
        assert task.config.verifier.type == "llm-judge"
        assert task.config.verifier.judge.rubric_path == "tests/rubric.md"
        assert task.config.sandbox.os.value == "macos"
        # The rubric criteria are materialized as a generated rubric file.
        rubric = task.generated_files["tests/rubric.md"]
        assert isinstance(rubric, str)
        assert "Open Clock app" in rubric
        assert "Save the alarm" in rubric

    def test_repo_shape_supported_host_is_not_a_single_task(
        self, tmp_path: Path
    ) -> None:
        repo = _write_iosworld_repo(tmp_path)

        report = IOSWorldAdapter.support_report(repo)

        # A whole-repository source is a suite, not one translatable task —
        # even on a capable host it is not a single InboundTask.
        assert report is not None
        assert report.supported is False
        assert report.details["issue"] == (
            "iosworld-repository-suite-not-a-single-task"
        )


def _force_macos_caps(monkeypatch: pytest.MonkeyPatch, present: bool) -> None:
    """Pin the cua macOS-VM host capability probe to a deterministic value.

    The macOSWorld adapter delegates its supported/unsupported decision to
    ``detect_cua_macos_capabilities``, which reads the real host. Tests pin it
    so they assert one path regardless of whether the host happens to be an
    Apple-Silicon Mac with the cua SDK installed.
    """
    import benchflow.adapters.macosworld as macosworld

    caps = {"macos": present, "cua-macos-vm": present}
    monkeypatch.setattr(macosworld, "detect_cua_macos_capabilities", lambda: caps)


class TestMacOSWorldAdapter:
    """The provider-honest *unsupported* path (host lacks the cua macOS VM)."""

    @pytest.fixture(autouse=True)
    def _no_macos_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _force_macos_caps(monkeypatch, present=False)

    def test_support_report_for_repo_shape(self, tmp_path: Path) -> None:
        repo = _write_macosworld_repo(tmp_path)

        report = MacOSWorldAdapter.support_report(repo)

        assert report is not None
        assert report.supported is False
        assert report.source == "macosworld"
        assert report.dataset == "macosworld"
        assert report.reason == (
            "macOSWorld tasks require a cua macOS VM provider mapping"
        )
        assert report.details["shape"] == "repository"
        assert report.details["required_provider"] == "cua"
        assert report.details["task_count"] == 1
        assert report.details["categories"] == {"productivity": 0, "sys_apps": 1}
        # productivity has no task files in this fixture; sys_apps has one.
        assert report.details["issue"] == "cua-macos-vm-provider-required"
        assert report.details["required_capabilities"] == [
            "macos",
            "cua-macos-vm",
            "macos-vm-exec",
        ]

    def test_support_report_for_task_slice_shape(self, tmp_path: Path) -> None:
        task_dir = _write_macosworld_task_slice(tmp_path)

        report = MacOSWorldAdapter.support_report(task_dir)

        assert report is not None
        assert report.supported is False
        assert report.task_id == "48cf0af3-0612-dbcd-14da-d5202eed6ce9"
        assert report.details["shape"] == "task-slice"
        assert report.details["required_provider"] == "cua"
        assert report.details["languages"] == ["en", "zh"]
        assert report.details["grading_command_count"] == 1

    def test_support_report_for_invalid_json(self, tmp_path: Path) -> None:
        task_dir = tmp_path / "macosworld-bad"
        task_dir.mkdir()
        (task_dir / "macosworld-task.json").write_text("{not json")

        report = MacOSWorldAdapter.support_report(task_dir)

        assert report is not None
        assert report.supported is False
        assert report.details["issue"] == "invalid-macosworld-task-json"

    def test_from_task_dir_reports_provider_requirement(self, tmp_path: Path) -> None:
        repo = _write_macosworld_repo(tmp_path)

        with pytest.raises(
            UnsupportedInboundTaskError,
            match="cua macOS VM provider mapping",
        ) as exc:
            MacOSWorldAdapter.from_task_dir(repo)

        assert exc.value.report.source == "macosworld"
        assert exc.value.report.details["issue"] == "cua-macos-vm-provider-required"

    def test_from_task_dir_unrecognized_raises_file_not_found(
        self, tmp_path: Path
    ) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            MacOSWorldAdapter.from_task_dir(empty)


class TestMacOSWorldAdapterSupported:
    """The provider-honest *supported* path (host advertises the cua macOS VM)."""

    @pytest.fixture(autouse=True)
    def _macos_host_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _force_macos_caps(monkeypatch, present=True)

    def test_task_slice_reports_supported(self, tmp_path: Path) -> None:
        task_dir = _write_macosworld_task_slice(tmp_path)

        report = MacOSWorldAdapter.support_report(task_dir)

        assert report is not None
        assert report.supported is True
        assert report.task_id == "48cf0af3-0612-dbcd-14da-d5202eed6ce9"
        assert report.dataset == "macosworld"
        assert report.reason is None
        assert report.details["shape"] == "task-slice"
        assert report.details["provider"] == "cua"
        assert report.details["languages"] == ["en", "zh"]
        assert report.details["grading_command_count"] == 1
        # The macOS exec bridge (#19) is a follow-up step, not a host
        # capability — it is reported as pending, never silently claimed.
        assert report.details["pending_capabilities"] == ["macos-vm-exec"]

    def test_from_task_dir_translates_slice(self, tmp_path: Path) -> None:
        task_dir = _write_macosworld_task_slice(tmp_path)

        task = MacOSWorldAdapter.from_task_dir(task_dir)

        assert isinstance(task, InboundTask)
        assert task.name == "48cf0af3-0612-dbcd-14da-d5202eed6ce9"
        assert task.source == "macosworld"
        # The English instruction is the translated goal.
        assert task.instruction.startswith("Add Ong KC to contact")
        # The grading commands map to an LLM-judge verifier, mirroring the
        # iOSWorld / Browser Use / computer-use criteria reward shape.
        assert task.config.verifier.type == "llm-judge"
        assert task.config.verifier.judge.rubric_path == "tests/rubric.md"
        assert task.config.sandbox.os.value == "macos"
        assert task.config.task is not None
        assert (
            task.config.task.name == "macosworld/48cf0af3-0612-dbcd-14da-d5202eed6ce9"
        )
        # The grading commands are materialized as a generated rubric file.
        rubric = task.generated_files["tests/rubric.md"]
        assert isinstance(rubric, str)
        assert "Add Ong KC to contact" in rubric
        assert "96910380" in rubric
        # The full upstream descriptor is preserved as compatibility metadata.
        assert task.compatibility is not None
        assert task.compatibility.source == "macosworld"
        assert task.compatibility.config_extra_paths == ("macosworld-task.json",)
        assert (
            task.compatibility.config_extra["grading_command"]
            == (_MACOSWORLD_TASK_JSON["grading_command"])
        )

    def test_string_task_instruction_is_accepted(self, tmp_path: Path) -> None:
        task_dir = tmp_path / "macosworld-string-task"
        task_dir.mkdir()
        payload = dict(_MACOSWORLD_TASK_JSON)
        payload["task"] = "Open Stocks and view AAPL."
        (task_dir / "macosworld-task.json").write_text(json.dumps(payload))

        task = MacOSWorldAdapter.from_task_dir(task_dir)

        assert task.instruction.startswith("Open Stocks and view AAPL.")

    def test_missing_instruction_raises(self, tmp_path: Path) -> None:
        task_dir = tmp_path / "macosworld-no-task"
        task_dir.mkdir()
        payload = dict(_MACOSWORLD_TASK_JSON)
        payload.pop("task")
        (task_dir / "macosworld-task.json").write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="instruction"):
            MacOSWorldAdapter.from_task_dir(task_dir)

    def test_repo_shape_supported_host_is_not_a_single_task(
        self, tmp_path: Path
    ) -> None:
        repo = _write_macosworld_repo(tmp_path)

        report = MacOSWorldAdapter.support_report(repo)

        # A whole-repository source is a suite, not one translatable task —
        # even on a capable host it is not a single InboundTask.
        assert report is not None
        assert report.supported is False
        assert report.details["issue"] == (
            "macosworld-repository-suite-not-a-single-task"
        )

    def test_materialized_task_preserves_macosworld_compatibility(
        self, tmp_path: Path
    ) -> None:
        task_dir = _write_macosworld_task_slice(tmp_path)
        inbound = MacOSWorldAdapter.from_task_dir(task_dir)

        native_dir = materialize_inbound_task_md(
            inbound, tmp_path / "native-macosworld"
        )

        document = TaskDocument.from_path(native_dir / "task.md")
        compat = document.frontmatter["benchflow"]["compat"]
        assert compat["source"] == "macosworld"
        assert compat["config_extra_paths"] == ["macosworld-task.json"]


class TestUseComputerCookbookAdapter:
    def test_returns_inbound_task(self, tmp_path: Path) -> None:
        task_dir = _write_use_computer_cookbook_osworld_task(tmp_path)
        result = UseComputerCookbookAdapter.from_task_dir(task_dir)
        assert isinstance(result, InboundTask)

    def test_osworld_metadata_is_preserved(self, tmp_path: Path) -> None:
        task_dir = _write_use_computer_cookbook_osworld_task(tmp_path)
        result = UseComputerCookbookAdapter.from_task_dir(task_dir)

        assert result.name == "smoke__ubuntu-osworld"
        assert result.source == "use-computer-cookbook"
        assert result.config.task is not None
        assert result.config.task.name == "osworld/ubuntu-smoke"
        assert result.config.sandbox.workdir == "/app"
        assert result.config.metadata["use_computer_cookbook"] == {
            "task_id": "smoke__ubuntu-osworld",
            "dataset": "osworld",
            "expected_result": "setup-ok",
            "osworld": True,
            "cuagym_smoke": False,
            "cuagym_task": False,
            "source_dir_name": "use-computer-osworld-task",
        }
        assert result.compatibility is not None
        assert result.compatibility.config_extra["task_id"] == ("smoke__ubuntu-osworld")
        assert result.compatibility.config_extra_paths == ("tests/osworld_task.json",)

    def test_instruction_gets_exact_final_answer_contract(self, tmp_path: Path) -> None:
        task_dir = _write_use_computer_cookbook_osworld_task(tmp_path)
        result = UseComputerCookbookAdapter.from_task_dir(task_dir)

        assert result.instruction.startswith("Observe the desktop once")
        assert result.instruction.endswith("Final answer must be exactly: setup-ok\n")

    def test_generates_native_runtime_files(self, tmp_path: Path) -> None:
        task_dir = _write_use_computer_cookbook_osworld_task(tmp_path)
        result = UseComputerCookbookAdapter.from_task_dir(task_dir)

        assert "environment/Dockerfile" in result.generated_files
        assert "tests/setup/pre_command.sh" in result.generated_files
        assert "tests/test.sh" in result.generated_files
        assert "solution/solve.sh" in result.generated_files
        assert "runner-osworld-setup-ok" in str(
            result.generated_files["tests/setup/pre_command.sh"]
        )
        assert "computer-use-smoke-trace.json" in str(
            result.generated_files["tests/test.sh"]
        )

    def test_cuagym_infra_smoke_is_supported(self, tmp_path: Path) -> None:
        task_dir = _write_use_computer_cookbook_cuagym_smoke_task(tmp_path)

        result = UseComputerCookbookAdapter.from_task_dir(task_dir)

        assert result.name == "smoke__ubuntu-infra"
        assert result.config.metadata["use_computer_cookbook"] == {
            "task_id": "smoke__ubuntu-infra",
            "dataset": "cuagym",
            "expected_result": "setup-ok",
            "osworld": False,
            "cuagym_smoke": True,
            "cuagym_task": False,
            "source_dir_name": "smoke__ubuntu-infra",
        }
        assert result.compatibility is not None
        assert result.compatibility.config_extra_paths == (
            "tests/setup/pre_command.sh",
        )
        assert result.files["tests/setup/pre_command.sh"] == (
            task_dir / "tests" / "setup" / "pre_command.sh"
        )
        assert "tests/test.sh" not in result.files
        assert "tests/test.sh" in result.generated_files
        assert "/tmp/runner-cuagym-setup-ok" in str(
            result.generated_files["tests/test.sh"]
        )

    def test_cuagym_python_reward_task_is_supported(self, tmp_path: Path) -> None:
        task_dir = _write_use_computer_cookbook_cuagym_python_task(tmp_path)

        result = UseComputerCookbookAdapter.from_task_dir(task_dir)

        assert result.name == "d0641d59-1751-5d45-8ced-5e45d615a68c"
        assert result.config.verifier.user == "root"
        assert result.config.metadata["use_computer_cookbook"] == {
            "task_id": "d0641d59-1751-5d45-8ced-5e45d615a68c",
            "dataset": "cuagym",
            "expected_result": "observed",
            "osworld": False,
            "cuagym_smoke": False,
            "cuagym_task": True,
            "source_dir_name": "vscode__d0641d59-1751-5d45-8ced-5e45d615a68c",
        }
        assert result.compatibility is not None
        assert result.compatibility.config_extra["cuagym_task"] == {
            "task_id": "d0641d59-1751-5d45-8ced-5e45d615a68c",
            "app_type": "vscode",
            "difficulty": "medium",
            "setup_kinds": ["download", "execute"],
            "postconfig_kinds": [],
            "postconfig_save_hotkey": False,
            "reward_imports": ["json", "os", "sqlite3"],
            "reward_dependencies": [],
            "original_dir": "tests/cuagym/original",
        }
        assert "tests/cuagym/original/task.json" in result.files
        assert "tests/setup/files/original/task.json" in result.files
        assert "tests/setup/pre_command.sh" in result.generated_files
        assert "tests/test.sh" in result.generated_files
        assert "reward.py" in str(result.generated_files["tests/test.sh"])

    def test_cuagym_python_reward_task_with_launch_setup_is_supported(
        self, tmp_path: Path
    ) -> None:
        task_dir = _write_use_computer_cookbook_cuagym_python_task(
            tmp_path,
            setup_kind="launch",
        )

        result = UseComputerCookbookAdapter.from_task_dir(task_dir)

        assert result.compatibility is not None
        cuagym_task = result.compatibility.config_extra["cuagym_task"]
        assert cuagym_task["setup_kinds"] == ["download", "launch"]
        setup_script = str(result.generated_files["tests/setup/pre_command.sh"])
        assert "def launch_command(command):" in setup_script
        assert "subprocess.Popen(" in setup_script

    def test_support_report_for_supported_cuagym_infra_smoke(
        self, tmp_path: Path
    ) -> None:
        task_dir = _write_use_computer_cookbook_cuagym_smoke_task(tmp_path)

        report = UseComputerCookbookAdapter.support_report(task_dir)

        assert report is not None
        assert report.supported is True
        assert report.dataset == "cuagym"
        assert report.task_id == "smoke__ubuntu-infra"
        assert report.details == {
            "signature": "tests/setup/pre_command.sh:cuagym-smoke"
        }

    def test_support_report_for_supported_cuagym_python_reward(
        self, tmp_path: Path
    ) -> None:
        task_dir = _write_use_computer_cookbook_cuagym_python_task(tmp_path)

        report = UseComputerCookbookAdapter.support_report(task_dir)

        assert report is not None
        assert report.supported is True
        assert report.dataset == "cuagym"
        assert report.task_id == "d0641d59-1751-5d45-8ced-5e45d615a68c"
        assert report.details == {
            "signature": "tests/cuagym/original/task.json:python-reward"
        }

    def test_rejects_cuagym_python_reward_with_non_stdlib_import(
        self, tmp_path: Path
    ) -> None:
        task_dir = _write_use_computer_cookbook_cuagym_python_task(tmp_path)
        reward = task_dir / "tests" / "cuagym" / "original" / "reward.py"
        reward.write_text("import requests\nprint('REWARD: 0.0')\n")

        with pytest.raises(UnsupportedInboundTaskError):
            UseComputerCookbookAdapter.from_task_dir(task_dir)
        report = UseComputerCookbookAdapter.support_report(task_dir)
        assert report is not None
        assert report.supported is False
        assert report.task_id == "d0641d59-1751-5d45-8ced-5e45d615a68c"
        assert report.details["issue"] == "unsupported-reward-imports"
        assert report.details["unsupported_reward_imports"] == ["requests"]

    def test_rejects_cuagym_python_reward_that_cannot_compile(
        self, tmp_path: Path
    ) -> None:
        task_dir = _write_use_computer_cookbook_cuagym_python_task(tmp_path)
        reward = task_dir / "tests" / "cuagym" / "original" / "reward.py"
        reward.write_text(
            "print('not a future import yet')\n"
            "from __future__ import annotations\n"
            "print('REWARD: 0.0')\n"
        )

        with pytest.raises(UnsupportedInboundTaskError):
            UseComputerCookbookAdapter.from_task_dir(task_dir)
        report = UseComputerCookbookAdapter.support_report(task_dir)
        assert report is not None
        assert report.supported is False
        assert report.task_id == "d0641d59-1751-5d45-8ced-5e45d615a68c"
        assert report.details["issue"] == "invalid-reward-python"
        assert "from __future__ imports must occur" in str(
            report.details["reward_compile_error"]
        )

    def test_cuagym_python_reward_with_pypdf2_dependency_is_supported(
        self, tmp_path: Path
    ) -> None:
        task_dir = _write_use_computer_cookbook_cuagym_python_task(tmp_path)
        reward = task_dir / "tests" / "cuagym" / "original" / "reward.py"
        reward.write_text(
            "from pathlib import Path\n"
            "from PyPDF2 import PdfReader\n"
            "print('REWARD: 0.0')\n"
        )

        result = UseComputerCookbookAdapter.from_task_dir(task_dir)

        assert result.compatibility is not None
        cuagym_task = result.compatibility.config_extra["cuagym_task"]
        assert cuagym_task["reward_imports"] == ["PyPDF2", "pathlib"]
        assert cuagym_task["reward_dependencies"] == ["PyPDF2"]
        verifier_script = str(result.generated_files["tests/test.sh"])
        assert "'PyPDF2': 'PyPDF2'" in verifier_script

    def test_cuagym_python_reward_with_stdlib_imports_is_supported(
        self, tmp_path: Path
    ) -> None:
        task_dir = _write_use_computer_cookbook_cuagym_python_task(tmp_path)
        reward = task_dir / "tests" / "cuagym" / "original" / "reward.py"
        reward.write_text(
            "from __future__ import annotations\n"
            "import traceback, zipfile\n"
            "from difflib import SequenceMatcher\n"
            "from pathlib import Path\n"
            "from PyPDF2 import PdfReader\n"
            "print('REWARD: 0.0')\n"
        )

        result = UseComputerCookbookAdapter.from_task_dir(task_dir)

        assert result.compatibility is not None
        cuagym_task = result.compatibility.config_extra["cuagym_task"]
        assert cuagym_task["reward_imports"] == [
            "PyPDF2",
            "__future__",
            "difflib",
            "pathlib",
            "traceback",
            "zipfile",
        ]
        assert cuagym_task["reward_dependencies"] == ["PyPDF2"]

    def test_cuagym_python_reward_with_pillow_dependency_is_supported(
        self, tmp_path: Path
    ) -> None:
        task_dir = _write_use_computer_cookbook_cuagym_python_task(tmp_path)
        reward = task_dir / "tests" / "cuagym" / "original" / "reward.py"
        reward.write_text("import os\nfrom PIL import Image\nprint('REWARD: 0.0')\n")

        result = UseComputerCookbookAdapter.from_task_dir(task_dir)

        assert result.compatibility is not None
        cuagym_task = result.compatibility.config_extra["cuagym_task"]
        assert cuagym_task["reward_imports"] == ["PIL", "os"]
        assert cuagym_task["reward_dependencies"] == ["Pillow"]
        assert "'PIL': 'Pillow'" in str(result.generated_files["tests/test.sh"])

    def test_cuagym_python_reward_with_openpyxl_dependency_is_supported(
        self, tmp_path: Path
    ) -> None:
        task_dir = _write_use_computer_cookbook_cuagym_python_task(tmp_path)
        reward = task_dir / "tests" / "cuagym" / "original" / "reward.py"
        reward.write_text("import os\nimport openpyxl\nprint('REWARD: 0.0')\n")

        result = UseComputerCookbookAdapter.from_task_dir(task_dir)

        assert result.compatibility is not None
        cuagym_task = result.compatibility.config_extra["cuagym_task"]
        assert cuagym_task["reward_imports"] == ["openpyxl", "os"]
        assert cuagym_task["reward_dependencies"] == ["openpyxl"]
        assert "'openpyxl': 'openpyxl'" in str(result.generated_files["tests/test.sh"])

    def test_cuagym_python_reward_with_gimpformats_dependency_is_supported(
        self, tmp_path: Path
    ) -> None:
        task_dir = _write_use_computer_cookbook_cuagym_python_task(tmp_path)
        reward = task_dir / "tests" / "cuagym" / "original" / "reward.py"
        reward.write_text(
            "import os\n"
            "from gimpformats.gimpXcfDocument import GimpDocument\n"
            "print('REWARD: 0.0')\n"
        )

        result = UseComputerCookbookAdapter.from_task_dir(task_dir)

        assert result.compatibility is not None
        cuagym_task = result.compatibility.config_extra["cuagym_task"]
        assert cuagym_task["reward_imports"] == ["gimpformats", "os"]
        assert cuagym_task["reward_dependencies"] == ["gimpformats"]
        assert "'gimpformats': 'gimpformats'" in str(
            result.generated_files["tests/test.sh"]
        )

    def test_cuagym_python_reward_with_docx_dependency_is_supported(
        self, tmp_path: Path
    ) -> None:
        task_dir = _write_use_computer_cookbook_cuagym_python_task(tmp_path)
        reward = task_dir / "tests" / "cuagym" / "original" / "reward.py"
        reward.write_text(
            "import os\nfrom docx import Document\nprint('REWARD: 0.0')\n"
        )

        result = UseComputerCookbookAdapter.from_task_dir(task_dir)

        assert result.compatibility is not None
        cuagym_task = result.compatibility.config_extra["cuagym_task"]
        assert cuagym_task["reward_imports"] == ["docx", "os"]
        assert cuagym_task["reward_dependencies"] == ["python-docx"]
        assert "'docx': 'python-docx'" in str(result.generated_files["tests/test.sh"])

    def test_cuagym_python_reward_with_numpy_pandas_dependencies_is_supported(
        self, tmp_path: Path
    ) -> None:
        task_dir = _write_use_computer_cookbook_cuagym_python_task(tmp_path)
        reward = task_dir / "tests" / "cuagym" / "original" / "reward.py"
        reward.write_text(
            "import numpy as np\nimport os\nimport pandas as pd\nprint('REWARD: 0.0')\n"
        )

        result = UseComputerCookbookAdapter.from_task_dir(task_dir)

        assert result.compatibility is not None
        cuagym_task = result.compatibility.config_extra["cuagym_task"]
        assert cuagym_task["reward_imports"] == ["numpy", "os", "pandas"]
        assert cuagym_task["reward_dependencies"] == ["numpy", "pandas"]
        generated_test = str(result.generated_files["tests/test.sh"])
        assert "'numpy': 'numpy'" in generated_test
        assert "'pandas': 'pandas'" in generated_test

    def test_cuagym_python_reward_with_odf_pptx_pyperclip_dependencies_is_supported(
        self, tmp_path: Path
    ) -> None:
        task_dir = _write_use_computer_cookbook_cuagym_python_task(tmp_path)
        reward = task_dir / "tests" / "cuagym" / "original" / "reward.py"
        reward.write_text(
            "import os\n"
            "import pyperclip\n"
            "from odf.opendocument import load\n"
            "from pptx import Presentation\n"
            "print('REWARD: 0.0')\n"
        )

        result = UseComputerCookbookAdapter.from_task_dir(task_dir)

        assert result.compatibility is not None
        cuagym_task = result.compatibility.config_extra["cuagym_task"]
        assert cuagym_task["reward_imports"] == ["odf", "os", "pptx", "pyperclip"]
        assert cuagym_task["reward_dependencies"] == [
            "odfpy",
            "pyperclip",
            "python-pptx",
        ]
        generated_test = str(result.generated_files["tests/test.sh"])
        assert "'odf': 'odfpy'" in generated_test
        assert "'pptx': 'python-pptx'" in generated_test
        assert "'pyperclip': 'pyperclip'" in generated_test

    def test_rejects_cuagym_python_reward_with_unmapped_setup_launcher(
        self, tmp_path: Path
    ) -> None:
        task_dir = _write_use_computer_cookbook_cuagym_python_task(
            tmp_path,
            setup_kind="launch",
        )
        setup = task_dir / "tests" / "cuagym" / "original" / "initial_setup.py"
        setup.write_text(
            "import subprocess\n"
            "subprocess.Popen(['code', '/home/user/project'])\n"
            "print('GUI_READY')\n"
        )

        with pytest.raises(UnsupportedInboundTaskError):
            UseComputerCookbookAdapter.from_task_dir(task_dir)
        report = UseComputerCookbookAdapter.support_report(task_dir)
        assert report is not None
        assert report.supported is False
        assert report.details["issue"] == "unmapped-setup-launchers"
        assert "subprocess.Popen(" in report.details["unmapped_setup_markers"]

    def test_cuagym_python_reward_with_save_hotkey_postconfig_is_supported(
        self, tmp_path: Path
    ) -> None:
        task_dir = _write_use_computer_cookbook_cuagym_python_task(tmp_path)
        task_json_path = task_dir / "tests" / "cuagym" / "original" / "task.json"
        task_json = json.loads(task_json_path.read_text())
        task_json["evaluator"]["postconfig"] = [
            {
                "type": "execute",
                "parameters": {
                    "command": [
                        "python",
                        "-c",
                        'import pyautogui; pyautogui.hotkey("ctrl", "s");',
                    ]
                },
            },
            {"type": "sleep", "parameters": {"seconds": 0.5}},
        ]
        task_json_path.write_text(json.dumps(task_json, indent=2) + "\n")

        result = UseComputerCookbookAdapter.from_task_dir(task_dir)

        assert result.compatibility is not None
        cuagym_task = result.compatibility.config_extra["cuagym_task"]
        assert cuagym_task["postconfig_kinds"] == ["execute", "sleep"]
        assert cuagym_task["postconfig_save_hotkey"] is True
        assert "XAUTHORITY=/home/cua/.Xauthority" in str(
            result.generated_files["tests/test.sh"]
        )

    def test_rejects_cuagym_python_reward_with_unknown_postconfig_command(
        self, tmp_path: Path
    ) -> None:
        task_dir = _write_use_computer_cookbook_cuagym_python_task(tmp_path)
        task_json_path = task_dir / "tests" / "cuagym" / "original" / "task.json"
        task_json = json.loads(task_json_path.read_text())
        task_json["evaluator"]["postconfig"] = [
            {
                "type": "execute",
                "parameters": {"command": ["python", "-c", "print('x')"]},
            }
        ]
        task_json_path.write_text(json.dumps(task_json, indent=2) + "\n")

        with pytest.raises(UnsupportedInboundTaskError):
            UseComputerCookbookAdapter.from_task_dir(task_dir)
        report = UseComputerCookbookAdapter.support_report(task_dir)
        assert report is not None
        assert report.supported is False
        assert report.details["issue"] == "unsupported-evaluator-postconfig"
        assert report.details["postconfig_kinds"] == ["execute"]

    def test_rejects_non_osworld_cookbook_task(self, tmp_path: Path) -> None:
        task_dir = _write_use_computer_cookbook_tagged_task(
            tmp_path,
            dataset="cuagym",
            tags=["cuagym", "ubuntu", "smoke"],
        )

        with pytest.raises(
            UnsupportedInboundTaskError,
            match="CUA-Gym cookbook tasks need provider-honest setup/runtime",
        ) as exc:
            UseComputerCookbookAdapter.from_task_dir(task_dir)

        report = exc.value.report
        assert report.to_dict()["source"] == "use-computer-cookbook"
        assert report.supported is False
        assert report.dataset == "cuagym"
        assert report.task_id == "use-computer-cuagym-task"
        assert report.details["tags"] == ["cuagym", "ubuntu", "smoke"]

    @pytest.mark.parametrize(
        ("dataset", "tags", "reason"),
        [
            (
                "cuagym",
                ["cuagym", "ubuntu", "smoke"],
                "CUA-Gym cookbook tasks need provider-honest setup/runtime",
            ),
            (
                "macosworld",
                ["macosworld", "macos", "smoke"],
                "macOSWorld cookbook tasks need a macOS desktop provider mapping",
            ),
            (
                "waa",
                ["waa", "windows", "smoke"],
                "WindowsAgentArena cookbook tasks need Windows setup/evaluator",
            ),
        ],
    )
    def test_support_report_for_unsupported_cookbook_datasets(
        self,
        tmp_path: Path,
        dataset: str,
        tags: list[str],
        reason: str,
    ) -> None:
        task_dir = _write_use_computer_cookbook_tagged_task(
            tmp_path,
            dataset=dataset,
            tags=tags,
        )

        report = UseComputerCookbookAdapter.support_report(task_dir)

        assert report is not None
        assert report.supported is False
        assert report.dataset == dataset
        assert report.reason is not None
        assert report.reason.startswith(reason)

    def test_support_report_for_supported_osworld(self, tmp_path: Path) -> None:
        task_dir = _write_use_computer_cookbook_osworld_task(tmp_path)

        report = UseComputerCookbookAdapter.support_report(task_dir)

        assert report is not None
        assert report.supported is True
        assert report.dataset == "osworld"
        assert report.task_id == "smoke__ubuntu-osworld"
        assert report.details == {"signature": "tests/osworld_task.json"}


# detect_adapter — format dispatch


class TestDetectAdapter:
    def test_detects_harbor(self, tmp_path: Path) -> None:
        task_dir = _write_harbor_task(tmp_path)
        assert detect_adapter(task_dir) is HarborAdapter

    def test_task_yaml_is_not_recognized(self, tmp_path: Path) -> None:
        # The flat Terminal-Bench task.yaml format is no longer supported;
        # a directory carrying only a task.yaml matches no known format.
        task_dir = tmp_path / "tb-task"
        task_dir.mkdir()
        (task_dir / "task.yaml").write_text("instruction: do the thing\n")
        with pytest.raises(ValueError, match=r"[Uu]nrecognized"):
            detect_adapter(task_dir)

    def test_detects_browser_use(self, tmp_path: Path) -> None:
        task_dir = _write_browser_use_task(tmp_path)
        assert detect_adapter(task_dir) is BrowserUseAdapter

    def test_detects_computer_use(self, tmp_path: Path) -> None:
        task_dir = _write_computer_use_task(tmp_path)
        assert detect_adapter(task_dir) is ComputerUseAdapter

    def test_detects_iosworld_repo(self, tmp_path: Path) -> None:
        task_dir = _write_iosworld_repo(tmp_path)
        assert detect_adapter(task_dir) is IOSWorldAdapter

    def test_detects_iosworld_task_slice(self, tmp_path: Path) -> None:
        task_dir = _write_iosworld_task_slice(tmp_path)
        assert detect_adapter(task_dir) is IOSWorldAdapter

    def test_detects_macosworld_repo(self, tmp_path: Path) -> None:
        task_dir = _write_macosworld_repo(tmp_path)
        assert detect_adapter(task_dir) is MacOSWorldAdapter

    def test_detects_macosworld_task_slice(self, tmp_path: Path) -> None:
        task_dir = _write_macosworld_task_slice(tmp_path)
        assert detect_adapter(task_dir) is MacOSWorldAdapter

    def test_detects_use_computer_cookbook_osworld(self, tmp_path: Path) -> None:
        task_dir = _write_use_computer_cookbook_osworld_task(tmp_path)
        assert detect_adapter(task_dir) is UseComputerCookbookAdapter

    def test_detects_unsupported_use_computer_cookbook_before_harbor(
        self,
        tmp_path: Path,
    ) -> None:
        task_dir = _write_use_computer_cookbook_tagged_task(
            tmp_path,
            dataset="macosworld",
            tags=["macosworld", "macos", "smoke"],
        )

        assert detect_adapter(task_dir) is UseComputerCookbookAdapter

    def test_unknown_format_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match=r"[Uu]nrecognized"):
            detect_adapter(empty)

    def test_detect_then_convert_harbor(self, tmp_path: Path) -> None:
        task_dir = _write_harbor_task(tmp_path)
        adapter = detect_adapter(task_dir)
        result = adapter.from_task_dir(task_dir)
        assert result.source == "harbor"

    def test_detect_then_convert_browser_use(self, tmp_path: Path) -> None:
        task_dir = _write_browser_use_task(tmp_path)
        adapter = detect_adapter(task_dir)
        result = adapter.from_task_dir(task_dir)
        assert result.source == "browser-use-benchmark"

    def test_detect_then_convert_computer_use(self, tmp_path: Path) -> None:
        task_dir = _write_computer_use_task(tmp_path)
        adapter = detect_adapter(task_dir)
        result = adapter.from_task_dir(task_dir)
        assert result.source == "computer-use-benchmark"

    def test_detect_then_convert_iosworld_reports_unsupported(
        self, tmp_path: Path
    ) -> None:
        task_dir = _write_iosworld_repo(tmp_path)
        adapter = detect_adapter(task_dir)
        with pytest.raises(UnsupportedInboundTaskError) as exc:
            adapter.from_task_dir(task_dir)
        assert exc.value.report.source == "iosworld"

    def test_detect_then_convert_macosworld_reports_unsupported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _force_macos_caps(monkeypatch, present=False)
        task_dir = _write_macosworld_repo(tmp_path)
        adapter = detect_adapter(task_dir)
        with pytest.raises(UnsupportedInboundTaskError) as exc:
            adapter.from_task_dir(task_dir)
        assert exc.value.report.source == "macosworld"

    def test_detect_then_convert_use_computer_cookbook(self, tmp_path: Path) -> None:
        task_dir = _write_use_computer_cookbook_osworld_task(tmp_path)
        adapter = detect_adapter(task_dir)
        result = adapter.from_task_dir(task_dir)
        assert result.source == "use-computer-cookbook"

    def test_both_signature_files_tie_break_to_harbor(self, tmp_path: Path) -> None:
        """A directory carrying BOTH a task.toml and a task.yaml resolves to
        the native split adapter — task.toml is the native superset format
        and wins the tie."""
        task_dir = _write_harbor_task(tmp_path)
        # Add a stray task.yaml alongside the native split signature file.
        (task_dir / "task.yaml").write_text("instruction: do the thing\n")
        assert detect_adapter(task_dir) is HarborAdapter


# InboundTask result type


class TestInboundTask:
    def test_native_task_dir_layout(self, tmp_path: Path) -> None:
        # The file map keys are the BenchFlow-native relative layout, so a
        # consumer can materialize a runnable task directory from any source.
        task_dir = _write_harbor_task(tmp_path)
        result = HarborAdapter.from_task_dir(task_dir)
        for rel in result.files:
            assert not rel.startswith("/")
            # Every mapped destination is a BenchFlow-native location.
            assert rel.split("/", 1)[0] in {"environment", "tests", "solution"}

    @pytest.mark.parametrize(
        ("adapter", "writer"),
        [
            (HarborAdapter, _write_harbor_task),
            (BrowserUseAdapter, _write_browser_use_task),
            (ComputerUseAdapter, _write_computer_use_task),
            (UseComputerCookbookAdapter, _write_use_computer_cookbook_osworld_task),
        ],
        ids=[
            "native-toml",
            "browser-use",
            "computer-use",
            "use-computer-cookbook",
        ],
    )
    def test_materializes_native_task_md_publication_package(
        self,
        tmp_path: Path,
        adapter,
        writer,
    ) -> None:
        """Guards PR #1's inbound adapter -> native task.md dogfood path."""
        foreign_task = writer(tmp_path)
        inbound = adapter.from_task_dir(foreign_task)
        native_task = materialize_inbound_task_md(inbound, tmp_path / "native")

        assert (native_task / "task.md").is_file()
        assert not (native_task / "task.toml").exists()
        assert not (native_task / "instruction.md").exists()
        assert not (native_task / "tests").exists()
        assert not (native_task / "solution").exists()
        assert (native_task / "environment" / "Dockerfile").is_file()
        if adapter is UseComputerCookbookAdapter:
            assert (native_task / "verifier" / "setup" / "pre_command.sh").is_file()
        assert (native_task / "verifier" / "test.sh").is_file()
        assert (native_task / "verifier" / "verifier.md").is_file()
        assert (native_task / "verifier" / "rubrics" / "verifier.md").is_file()
        assert (native_task / "oracle" / "solve.sh").is_file()
        assert TaskDocument.from_path(native_task / "task.md").instruction == (
            inbound.instruction.strip()
        )
        assert check_task(native_task, validation_level="publication-grade") == []

    def test_materialized_harbor_task_md_preserves_foreign_extensions(
        self,
        tmp_path: Path,
    ) -> None:
        """Foreign extension keys survive materialization in benchflow.compat."""
        foreign_task = _write_harbor_task(tmp_path)
        config_path = foreign_task / "task.toml"
        config_path.write_text('harbor_ext = "kept"\n' + config_path.read_text())
        inbound = HarborAdapter.from_task_dir(foreign_task)

        native_task = materialize_inbound_task_md(inbound, tmp_path / "native")
        document = TaskDocument.from_path(native_task / "task.md")

        assert document.benchflow["compat"]["source"] == "harbor"
        assert document.benchflow["compat"]["config_extra"] == {"harbor_ext": "kept"}
        assert document.benchflow["compat"]["config_extra_paths"] == ["harbor_ext"]

    def test_materialization_rejects_unsafe_file_map_paths(
        self,
        tmp_path: Path,
    ) -> None:
        """Inbound materialization must not copy outside the native package."""
        foreign_task = _write_harbor_task(tmp_path)
        inbound = HarborAdapter.from_task_dir(foreign_task)
        escaped = InboundTask(
            name=inbound.name,
            source=inbound.source,
            instruction=inbound.instruction,
            manifest=inbound.manifest,
            config=inbound.config,
            files={"../escape.txt": foreign_task / "environment" / "Dockerfile"},
            generated_files={},
            compatibility=inbound.compatibility,
        )

        with pytest.raises(ValueError, match="not safe relative"):
            materialize_inbound_task_md(escaped, tmp_path / "native")

        assert not (tmp_path / "native").exists()
        assert not (tmp_path / "escape.txt").exists()

    def test_materialization_rejects_generated_file_collision(
        self,
        tmp_path: Path,
    ) -> None:
        """Generated and copied files must not fight over the same native path."""
        foreign_task = _write_computer_use_task(tmp_path)
        inbound = ComputerUseAdapter.from_task_dir(foreign_task)
        colliding = InboundTask(
            name=inbound.name,
            source=inbound.source,
            instruction=inbound.instruction,
            manifest=inbound.manifest,
            config=inbound.config,
            files=inbound.files,
            generated_files={"tests/test.sh": "#!/bin/bash\n"},
            compatibility=inbound.compatibility,
        )

        with pytest.raises(ValueError, match="generated file collision"):
            materialize_inbound_task_md(colliding, tmp_path / "native")
