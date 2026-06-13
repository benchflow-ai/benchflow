from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from typer.testing import CliRunner

import benchflow.cli.environment as cli_environment
from benchflow.cli.main import app


def _write_task(root: Path) -> Path:
    task_dir = root / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        """\
schema_version = "1.3"

[task]
name = "benchflow/env-cli-smoke"

[environment]
cpus = 1
memory_mb = 2048
"""
    )
    (task_dir / "instruction.md").write_text("Write the expected file.\n")
    env = task_dir / "environment"
    env.mkdir()
    (env / "Dockerfile").write_text(
        "FROM python:3.12-slim\nRUN mkdir -p /logs/verifier /logs/artifacts /app\n"
    )
    solution = task_dir / "solution"
    solution.mkdir()
    (solution / "solve.sh").write_text("#!/bin/bash\nprintf ok > /app/result.txt\n")
    tests = task_dir / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text(
        "#!/bin/bash\nmkdir -p /logs/verifier\n"
        "printf '1.0\\n' > /logs/verifier/reward.txt\n"
        "printf '{\"reward\": 1.0}\\n' > /logs/verifier/reward.json\n"
    )
    return task_dir


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


def _write_browser_use_task(root: Path) -> Path:
    task_dir = root / "browser-use-task"
    task_dir.mkdir()
    (task_dir / "browser-use-task.json").write_text(
        json.dumps(
            {
                "task_id": "open-local-page",
                "benchmark": "browser-use",
                "confirmed_task": "Open the local page and report ready.",
                "expected_result": "browser-use-smoke: ready",
            },
            indent=2,
        )
        + "\n"
    )
    env = task_dir / "environment"
    env.mkdir()
    (env / "Dockerfile").write_text(
        "FROM python:3.12-slim\nRUN mkdir -p /logs/verifier /logs/artifacts /app\n"
    )
    tests = task_dir / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text(
        "#!/bin/bash\nmkdir -p /logs/verifier\n"
        "printf '{\"reward\": 1.0}\\n' > /logs/verifier/reward.json\n"
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


def _install_fake_cua(monkeypatch, sandboxes: list[Any]):
    deleted: list[str] = []
    calls: list[tuple[bool, str | None]] = []

    class FakeImage:
        pass

    class FakeSandbox:
        @classmethod
        async def list(cls, *, local: bool = False, api_key: str | None = None):
            calls.append((local, api_key))
            return sandboxes

        @classmethod
        async def delete(
            cls,
            name: str,
            *,
            local: bool = False,
            api_key: str | None = None,
        ) -> None:
            calls.append((local, api_key))
            deleted.append(name)

    monkeypatch.setitem(
        sys.modules,
        "cua_sandbox",
        SimpleNamespace(Image=FakeImage, Sandbox=FakeSandbox),
    )
    return calls, deleted


def _install_fake_docker(monkeypatch):
    calls: list[list[str]] = []
    deleted: list[list[str]] = []
    container_rows = [
        {
            "ID": "container-old",
            "Names": "benchflow-task-main-1",
            "Status": "Exited (0) 2 hours ago",
            "CreatedAt": "2020-01-01 00:00:00 +0000 UTC",
            "Labels": (
                "benchflow.owned=true,com.docker.compose.project=benchflow-task"
            ),
        },
        {
            "ID": "container-fresh",
            "Names": "benchflow-fresh-main-1",
            "Status": "Up 1 second",
            "CreatedAt": "2099-01-01 00:00:00 +0000 UTC",
            "Labels": (
                "benchflow.owned=true,com.docker.compose.project=benchflow-fresh"
            ),
        },
    ]
    network_rows = [
        {
            "ID": "network-old",
            "Name": "benchflow-task_default",
            "Driver": "bridge",
            "CreatedAt": "2020-01-01 00:00:00.123456789 +0000 UTC",
            "Labels": (
                "benchflow.owned=true,com.docker.compose.project=benchflow-task"
            ),
        }
    ]
    image_rows = [
        {
            "ID": "image-old",
            "Repository": "bf-snap-benchflow-task",
            "Tag": "latest",
            "Size": "120MB",
            "CreatedAt": "2020-01-01 00:00:00 +0000 UTC",
        },
        {
            "ID": "image-fresh",
            "Repository": "bf-snap-benchflow-fresh",
            "Tag": "latest",
            "Size": "120MB",
            "CreatedAt": "2099-01-01 00:00:00 +0000 UTC",
        },
    ]

    def fake_run(cmd, **_kwargs):
        command = [str(part) for part in cmd]
        calls.append(command)
        if command[:4] == ["docker", "container", "ls", "-a"]:
            assert "--filter" in command
            assert "label=benchflow.owned=true" in command
            return SimpleNamespace(
                returncode=0,
                stdout="\n".join(json.dumps(row) for row in container_rows) + "\n",
                stderr="",
            )
        if command[:3] == ["docker", "network", "ls"]:
            assert "--filter" in command
            assert "label=benchflow.owned=true" in command
            return SimpleNamespace(
                returncode=0,
                stdout="\n".join(json.dumps(row) for row in network_rows) + "\n",
                stderr="",
            )
        if command[:3] == ["docker", "image", "ls"]:
            assert "--filter" in command
            assert "label=benchflow.owned=true" in command
            return SimpleNamespace(
                returncode=0,
                stdout="\n".join(json.dumps(row) for row in image_rows) + "\n",
                stderr="",
            )
        if command[:3] in (
            ["docker", "rm", "-f"],
            ["docker", "network", "rm"],
            ["docker", "image", "rm"],
        ):
            deleted.append(command)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected docker command: {command}")

    monkeypatch.setattr(
        cli_environment.shutil, "which", lambda name: f"/usr/bin/{name}"
    )
    monkeypatch.setattr(cli_environment.subprocess, "run", fake_run)
    return calls, deleted


def _expected_desktop_environment_adapter(
    *,
    provider_mode: str = "cloud",
    provider_support: str = "runtime-probe-required",
    verified_sandboxes: list[str] | None = None,
    note: str | None = (
        "desktop environment adapter is verified on local Cua; Cua cloud "
        "must pass `bench environment check --probe-runtime --sandbox cua` "
        "before scale"
    ),
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": "desktop",
        "world": "desktop",
        "benchmark_adapter": "use-computer-cookbook",
        "status": "ready",
        "provider_support": provider_support,
        "required_capabilities": [
            "shell",
            "file-transfer",
            "screenshot",
            "display-dimensions",
            "cleanup",
        ],
        "verified_sandboxes": verified_sandboxes or ["cua:local"],
        "provider_mode": provider_mode,
    }
    if note is not None:
        payload["note"] = note
    return payload


def test_environment_create_dry_run_does_not_create_environment(tmp_path: Path) -> None:
    task_dir = _write_task(tmp_path)

    with patch("benchflow.runtime.Environment.from_task") as from_task:
        result = CliRunner().invoke(
            app,
            [
                "environment",
                "create",
                str(task_dir),
                "--sandbox",
                "docker",
                "--dry-run",
            ],
        )

    assert result.exit_code == 0
    assert "Environment dry-run passed" in result.output
    assert "Created: no" in result.output
    from_task.assert_not_called()


def test_environment_create_dry_run_accepts_foreign_adapter_task(
    tmp_path: Path,
) -> None:
    task_dir = _write_cookbook_osworld_task(tmp_path)

    with patch("benchflow.runtime.Environment.from_task") as from_task:
        result = CliRunner().invoke(
            app,
            [
                "environment",
                "create",
                str(task_dir),
                "--sandbox",
                "cua",
                "--dry-run",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "Environment dry-run passed" in result.output
    assert "Adapter: use-computer-cookbook" in result.output
    assert "Sandbox: cua" in result.output
    from_task.assert_not_called()


def test_environment_create_dry_run_json_accepts_foreign_adapter_task(
    tmp_path: Path,
) -> None:
    task_dir = _write_cookbook_osworld_task(tmp_path)

    with patch("benchflow.runtime.Environment.from_task") as from_task:
        result = CliRunner().invoke(
            app,
            [
                "environment",
                "create",
                str(task_dir),
                "--sandbox",
                "cua",
                "--dry-run",
                "--json",
            ],
        )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {
        "status": "dry-run",
        "task": str(task_dir),
        "task_name": task_dir.name,
        "adapter": "use-computer-cookbook",
        "environment_adapter": _expected_desktop_environment_adapter(),
        "sandbox": "cua",
        "created": False,
    }
    from_task.assert_not_called()


def test_environment_create_materializes_foreign_task_for_runtime(
    tmp_path: Path,
) -> None:
    task_dir = _write_cookbook_osworld_task(tmp_path)
    calls: list[tuple[str, str]] = []

    def fake_from_task(path, *, sandbox: str):
        native_path = Path(path)
        assert native_path != task_dir
        assert (native_path / "task.md").is_file()
        assert (native_path / "environment" / "Dockerfile").is_file()
        assert (native_path / "verifier" / "test.sh").is_file()
        calls.append((native_path.name, sandbox))
        return SimpleNamespace(task_path=native_path, sandbox=sandbox)

    with patch("benchflow.runtime.Environment.from_task", side_effect=fake_from_task):
        result = CliRunner().invoke(
            app,
            ["environment", "create", str(task_dir), "--sandbox", "cua"],
        )

    assert result.exit_code == 0, result.output
    assert calls == [("smoke__ubuntu-osworld", "cua")]
    assert "Environment created" in result.output
    assert "Adapter: use-computer-cookbook" in result.output
    assert "Native:" in result.output


def test_environment_create_json_reports_materialized_foreign_task(
    tmp_path: Path,
) -> None:
    task_dir = _write_cookbook_osworld_task(tmp_path)

    def fake_from_task(path, *, sandbox: str):
        native_path = Path(path)
        assert native_path != task_dir
        assert (native_path / "task.md").is_file()
        return SimpleNamespace(task_path=native_path, sandbox=sandbox)

    with patch("benchflow.runtime.Environment.from_task", side_effect=fake_from_task):
        result = CliRunner().invoke(
            app,
            ["environment", "create", str(task_dir), "--sandbox", "cua", "--json"],
        )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {
        "status": "created",
        "task": str(task_dir),
        "task_name": task_dir.name,
        "adapter": "use-computer-cookbook",
        "environment_adapter": _expected_desktop_environment_adapter(),
        "sandbox": "cua",
        "created": True,
        "native": "materialized-temporary",
    }


def test_environment_check_validates_task_and_provider(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_dir = _write_task(tmp_path)
    checked: list[str] = []
    monkeypatch.setattr(
        cli_environment,
        "_check_provider_or_exit",
        lambda sandbox, **_kwargs: checked.append(sandbox),
    )

    result = CliRunner().invoke(
        app,
        ["environment", "check", str(task_dir), "--sandbox", "cua"],
    )

    assert result.exit_code == 0
    assert checked == ["cua"]
    assert "Environment check passed" in result.output


def test_environment_check_accepts_foreign_adapter_task(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_dir = _write_cookbook_osworld_task(tmp_path)
    checked: list[str] = []
    monkeypatch.setattr(
        cli_environment,
        "_check_provider_or_exit",
        lambda sandbox, **_kwargs: checked.append(sandbox),
    )

    result = CliRunner().invoke(
        app,
        ["environment", "check", str(task_dir), "--sandbox", "cua"],
    )

    assert result.exit_code == 0, result.output
    assert checked == ["cua"]
    assert "Environment check passed" in result.output
    assert "Adapter: use-computer-cookbook" in result.output


def test_environment_check_json_reports_adapter_and_provider(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_dir = _write_cookbook_osworld_task(tmp_path)
    checked: list[tuple[str, bool, bool, bool]] = []

    def fake_provider(
        sandbox: str,
        *,
        quiet: bool = False,
        runtime_probe: bool = False,
        output_json: bool = False,
    ):
        checked.append((sandbox, quiet, runtime_probe, output_json))
        return {"provider": sandbox, "status": "ready", "fake": True}

    monkeypatch.setattr(cli_environment, "_check_provider_or_exit", fake_provider)

    result = CliRunner().invoke(
        app,
        ["environment", "check", str(task_dir), "--sandbox", "cua", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {
        "status": "ready",
        "task": str(task_dir),
        "task_name": task_dir.name,
        "adapter": "use-computer-cookbook",
        "environment_adapter": _expected_desktop_environment_adapter(),
        "sandbox": "cua",
        "provider": {"provider": "cua", "status": "ready", "fake": True},
    }
    assert checked == [("cua", True, False, True)]


def test_environment_check_json_reports_local_cua_as_verified(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_dir = _write_cookbook_osworld_task(tmp_path)
    monkeypatch.setenv("BENCHFLOW_CUA_LOCAL", "1")
    monkeypatch.setattr(
        cli_environment,
        "_check_provider_or_exit",
        lambda sandbox, **_kwargs: {"provider": sandbox, "status": "ready"},
    )

    result = CliRunner().invoke(
        app,
        ["environment", "check", str(task_dir), "--sandbox", "cua", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["environment_adapter"] == _expected_desktop_environment_adapter(
        provider_mode="local",
        provider_support="verified",
        note=None,
    )


def test_environment_check_json_reports_browser_environment_adapter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_dir = _write_browser_use_task(tmp_path)
    checked: list[tuple[str, bool, bool, bool]] = []

    def fake_provider(
        sandbox: str,
        *,
        quiet: bool = False,
        runtime_probe: bool = False,
        output_json: bool = False,
    ):
        checked.append((sandbox, quiet, runtime_probe, output_json))
        return {"provider": sandbox, "status": "ready", "fake": True}

    monkeypatch.setattr(cli_environment, "_check_provider_or_exit", fake_provider)

    result = CliRunner().invoke(
        app,
        ["environment", "check", str(task_dir), "--sandbox", "docker", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["adapter"] == "browser-use-benchmark"
    assert payload["environment_adapter"] == {
        "name": "browser",
        "world": "browser",
        "benchmark_adapter": "browser-use-benchmark",
        "status": "ready",
        "provider_support": "verified",
        "required_capabilities": [
            "browser-runtime",
            "local-http-fixture",
            "screenshot-artifacts",
            "trace-artifacts",
        ],
        "verified_sandboxes": ["docker"],
    }
    assert payload["provider"] == {
        "provider": "docker",
        "status": "ready",
        "fake": True,
    }
    assert checked == [("docker", True, False, True)]


def test_environment_create_dry_run_reports_browser_environment_adapter(
    tmp_path: Path,
) -> None:
    task_dir = _write_browser_use_task(tmp_path)

    with patch("benchflow.runtime.Environment.from_task") as from_task:
        result = CliRunner().invoke(
            app,
            [
                "environment",
                "create",
                str(task_dir),
                "--sandbox",
                "docker",
                "--dry-run",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "Adapter: browser-use-benchmark" in result.output
    assert "Environment adapter: browser" in result.output
    from_task.assert_not_called()


def test_environment_check_json_can_probe_cua_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_dir = _write_cookbook_osworld_task(tmp_path)
    checked: list[tuple[str, bool, bool, bool]] = []

    def fake_provider(
        sandbox: str,
        *,
        quiet: bool = False,
        runtime_probe: bool = False,
        output_json: bool = False,
    ):
        checked.append((sandbox, quiet, runtime_probe, output_json))
        return {
            "provider": sandbox,
            "status": "ready",
            "runtime_probe": {
                "status": "ready",
                "checks": {"shell": {"ok": True}},
                "cleanup": {"attempted": True, "ok": True},
            },
        }

    monkeypatch.setattr(cli_environment, "_check_provider_or_exit", fake_provider)

    result = CliRunner().invoke(
        app,
        [
            "environment",
            "check",
            str(task_dir),
            "--sandbox",
            "cua",
            "--probe-runtime",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "ready"
    assert payload["environment_adapter"] == _expected_desktop_environment_adapter(
        provider_support="verified",
        verified_sandboxes=["cua:local", "cua:cloud-probed"],
        note=None,
    )
    assert payload["provider"]["runtime_probe"]["status"] == "ready"
    assert payload["provider"]["runtime_probe"]["cleanup"]["ok"] is True
    assert checked == [("cua", True, True, True)]


def test_cua_provider_runtime_probe_payload_is_included(monkeypatch) -> None:
    _install_fake_cua(monkeypatch, [])
    monkeypatch.setenv("BENCHFLOW_CUA_LOCAL", "1")
    monkeypatch.setattr(
        cli_environment,
        "_probe_cua_runtime",
        lambda: {
            "status": "ready",
            "local": True,
            "checks": {
                "shell": {"ok": True},
                "dimensions": {"ok": True, "width": 1024, "height": 768},
                "screenshot": {"ok": True, "bytes": 42},
                "display_url": {
                    "ok": True,
                    "available": True,
                    "scheme": "http",
                    "length": 32,
                },
            },
            "cleanup": {"attempted": True, "ok": True},
        },
    )

    payload = cli_environment._check_provider_or_exit(
        "cua",
        quiet=True,
        runtime_probe=True,
        output_json=False,
    )

    assert payload["provider"] == "cua"
    assert payload["status"] == "ready"
    runtime_probe = cast("dict[str, Any]", payload["runtime_probe"])
    runtime_checks = cast("dict[str, Any]", runtime_probe["checks"])
    screenshot = cast("dict[str, Any]", runtime_checks["screenshot"])
    assert runtime_probe["status"] == "ready"
    assert screenshot["bytes"] == 42


def test_cua_probe_captures_background_task_errors() -> None:
    async def noisy_probe() -> dict[str, object]:
        async def background() -> None:
            raise TimeoutError("cloud command endpoint returned 404")

        task = asyncio.create_task(background())
        await asyncio.sleep(0)
        del task
        return {"status": "not-ready"}

    payload, background_errors = (
        cli_environment._run_async_with_background_error_capture(noisy_probe())
    )

    assert payload == {"status": "not-ready"}
    assert background_errors == [
        {
            "message": "Task exception was never retrieved",
            "error_type": "TimeoutError",
            "reason": "cloud command endpoint returned 404",
        }
    ]


def test_cua_probe_failure_classifies_cloud_cmd_404() -> None:
    payload = {
        "status": "not-ready",
        "reason": "Cua sandbox failed to initialize BenchFlow runtime directories",
        "background_errors": [
            {
                "error_type": "TimeoutError",
                "reason": (
                    "Computer-server for VM 'benchflow-probe' not reachable: "
                    "Client error '404 Not Found' for url "
                    "'https://benchflow-probe-api.cua.sh/cmd'"
                ),
            }
        ],
    }

    failure_class = cli_environment._classify_cua_probe_failure(payload)

    assert failure_class == "cloud-computer-server-cmd-404"


def test_cua_runtime_probe_reports_failed_capabilities(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import benchflow.sandbox.cua as cua_module

    stopped: list[bool] = []

    class FakeCuaSandbox:
        def __init__(self, *args, **kwargs) -> None:
            self.sandbox_id = "benchflow-cua-runtime-probe"

        @property
        def task_env_config(self):
            return SimpleNamespace(cpus=1, memory_mb=2048, storage_mb=10240)

        @property
        def environment_name(self):
            return "cua-runtime-probe"

        @property
        def session_id(self):
            return "probe"

        @property
        def _sandbox(self):
            return None

        @_sandbox.setter
        def _sandbox(self, value):
            pass

        async def start(self) -> None:
            raise RuntimeError("cloud command endpoint returned 404")

        async def stop(self, *, delete: bool = True) -> None:
            stopped.append(delete)

    monkeypatch.setattr(cua_module, "CuaSandbox", FakeCuaSandbox)
    env_dir = tmp_path / "environment"
    env_dir.mkdir()

    payload = asyncio.run(cli_environment._probe_cua_runtime_async(env_dir))

    assert payload["status"] == "not-ready"
    assert payload["local"] is False
    sdk = cast("dict[str, Any]", payload["sdk"])
    request = cast("dict[str, Any]", payload["request"])
    assert sdk["available"] is True
    assert sdk["supports_ephemeral"] is True
    assert request["linux_kind"] == "vm"
    assert payload["sandbox_id"] == "benchflow-cua-runtime-probe"
    assert payload["required_capabilities"] == [
        "shell",
        "file_transfer",
        "dimensions",
        "screenshot",
    ]
    assert payload["failed_capabilities"] == [
        "startup",
        "shell",
        "file_transfer",
        "dimensions",
        "screenshot",
    ]
    assert payload["reason"] == "cloud command endpoint returned 404"
    assert payload["error_type"] == "RuntimeError"
    assert payload["cleanup"] == {"attempted": True, "ok": True}
    assert stopped == [True]


def test_environment_check_reports_unsupported_foreign_adapter_task(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_dir = _write_unsupported_cookbook_task(tmp_path)
    checked: list[str] = []
    monkeypatch.setattr(
        cli_environment,
        "_check_provider_or_exit",
        lambda sandbox, **_kwargs: checked.append(sandbox),
    )

    result = CliRunner().invoke(
        app,
        ["environment", "check", str(task_dir), "--sandbox", "cua"],
    )

    assert result.exit_code == 1
    assert checked == []
    assert "unsupported adapter task" in result.output
    assert "Adapter: use-computer-cookbook" in result.output
    assert "Dataset: cuagym" in result.output
    assert "CUA-Gym cookbook tasks need provider-honest setup/runtime" in result.output
    assert "Traceback (most recent call last)" not in result.output


def test_environment_check_json_reports_unsupported_foreign_adapter_task(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_dir = _write_unsupported_cookbook_task(tmp_path)
    checked: list[str] = []
    monkeypatch.setattr(
        cli_environment,
        "_check_provider_or_exit",
        lambda sandbox, **_kwargs: checked.append(sandbox),
    )

    result = CliRunner().invoke(
        app,
        ["environment", "check", str(task_dir), "--sandbox", "cua", "--json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "unsupported-adapter-task"
    assert payload["adapter"] == "use-computer-cookbook"
    assert payload["dataset"] == "cuagym"
    assert checked == []


def test_environment_check_json_reports_iosworld_provider_requirement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_dir = _write_iosworld_repo(tmp_path)
    checked: list[str] = []
    monkeypatch.setattr(
        cli_environment,
        "_check_provider_or_exit",
        lambda sandbox, **_kwargs: checked.append(sandbox),
    )

    result = CliRunner().invoke(
        app,
        ["environment", "check", str(task_dir), "--sandbox", "cua", "--json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "unsupported-adapter-task"
    assert payload["adapter"] == "iosworld"
    assert payload["dataset"] == "iosworld"
    assert payload["reason"] == (
        "iOSWorld tasks require a macOS/iOS Simulator provider mapping"
    )
    assert payload["details"]["required_provider"] == "macos-ios-simulator"
    assert payload["details"]["required_capabilities"] == [
        "macos",
        "xcode-26",
        "ios-26-simulator-runtime",
        "appium-xcuitest",
        "iosworld-app-bootstrap",
    ]
    assert checked == []


def test_environment_check_json_reports_missing_task(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    result = CliRunner().invoke(
        app,
        ["environment", "check", str(missing), "--sandbox", "cua", "--json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload == {
        "status": "error",
        "task": str(missing),
        "task_name": missing.name,
        "sandbox": "cua",
        "reason": "task directory not found",
    }


def test_environment_list_docker_json_uses_owned_label(monkeypatch) -> None:
    calls, _deleted = _install_fake_docker(monkeypatch)

    result = CliRunner().invoke(
        app,
        ["environment", "list", "--sandbox", "docker", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["provider"] == "docker"
    assert payload["ownership_label"] == "benchflow.owned=true"
    assert [item["id"] for item in payload["containers"]] == [
        "container-old",
        "container-fresh",
    ]
    assert [item["id"] for item in payload["networks"]] == ["network-old"]
    assert [item["id"] for item in payload["images"]] == [
        "image-old",
        "image-fresh",
    ]
    # Snapshot images are named Repository:Tag, not by container Names.
    assert [item["name"] for item in payload["images"]] == [
        "bf-snap-benchflow-task:latest",
        "bf-snap-benchflow-fresh:latest",
    ]
    assert all("label=benchflow.owned=true" in command for command in calls)


def test_environment_cleanup_docker_json_dry_run_reports_candidates(
    monkeypatch,
) -> None:
    _calls, deleted = _install_fake_docker(monkeypatch)

    result = CliRunner().invoke(
        app,
        [
            "environment",
            "cleanup",
            "--sandbox",
            "docker",
            "--dry-run",
            "--max-age",
            "60",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["provider"] == "docker"
    assert payload["status"] == "dry-run"
    assert payload["dry_run"] is True
    assert payload["ownership_label"] == "benchflow.owned=true"
    assert payload["found"] == 5
    assert payload["matched"] == 3
    assert payload["skipped"] == 2
    assert payload["deleted"] == []
    assert [(item["type"], item["id"]) for item in payload["candidates"]] == [
        ("container", "container-old"),
        ("network", "network-old"),
        ("image", "image-old"),
    ]
    assert all(item["would_delete"] is True for item in payload["candidates"])
    assert deleted == []


def test_environment_cleanup_docker_deletes_matching_owned_resources(
    monkeypatch,
) -> None:
    _calls, deleted = _install_fake_docker(monkeypatch)

    result = CliRunner().invoke(
        app,
        [
            "environment",
            "cleanup",
            "--sandbox",
            "docker",
            "--max-age",
            "60",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["provider"] == "docker"
    assert payload["status"] == "deleted"
    assert payload["dry_run"] is False
    assert [(item["type"], item["id"]) for item in payload["deleted"]] == [
        ("container", "container-old"),
        ("network", "network-old"),
        ("image", "image-old"),
    ]
    assert deleted == [
        ["docker", "rm", "-f", "container-old"],
        ["docker", "network", "rm", "network-old"],
        ["docker", "image", "rm", "-f", "image-old"],
    ]


def test_environment_list_cua_uses_sdk(monkeypatch) -> None:
    sandboxes = [
        SimpleNamespace(
            name="benchflow-old",
            status="running",
            source="cloud",
            os_type="linux",
            created_at="2026-06-12T00:00:00Z",
        )
    ]
    calls, _deleted = _install_fake_cua(monkeypatch, sandboxes)
    monkeypatch.setenv("CUA_API_KEY", "test-key")

    result = CliRunner().invoke(app, ["environment", "list", "--sandbox", "cua"])

    assert result.exit_code == 0
    assert "benchflow-old" in result.output
    assert "1 environment(s)" in result.output
    assert calls == [(False, "test-key")]


def test_environment_list_cua_json_is_parseable(monkeypatch) -> None:
    sandboxes = [
        SimpleNamespace(
            name="benchflow-old",
            status="running",
            source="cloud",
            os_type="linux",
            created_at="2026-06-12T00:00:00Z",
            api_url="https://example.invalid/a/very/long/cua/api/url/that/must/not/wrap",
        )
    ]
    calls, _deleted = _install_fake_cua(monkeypatch, sandboxes)
    monkeypatch.setenv("CUA_API_KEY", "test-key")

    result = CliRunner().invoke(
        app,
        ["environment", "list", "--sandbox", "cua", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload[0]["name"] == "benchflow-old"
    assert payload[0]["api_url"].startswith("https://example.invalid")
    assert calls == [(False, "test-key")]


def test_environment_cleanup_cua_dry_run_does_not_delete(monkeypatch) -> None:
    sandboxes = [
        SimpleNamespace(
            name="benchflow-old",
            status="running",
            source="cloud",
            os_type="linux",
            created_at="2020-01-01T00:00:00Z",
        ),
        SimpleNamespace(
            name="foreign-old",
            status="running",
            source="cloud",
            os_type="linux",
            created_at="2020-01-01T00:00:00Z",
        ),
    ]
    _calls, deleted = _install_fake_cua(monkeypatch, sandboxes)

    result = CliRunner().invoke(
        app,
        [
            "environment",
            "cleanup",
            "--sandbox",
            "cua",
            "--dry-run",
            "--max-age",
            "60",
        ],
    )

    assert result.exit_code == 0
    assert "benchflow-old" in result.output
    assert "foreign-old" not in result.output
    assert "1 matching prefix" in result.output
    assert deleted == []


def test_environment_cleanup_cua_json_dry_run_reports_candidates(monkeypatch) -> None:
    sandboxes = [
        SimpleNamespace(
            name="benchflow-old",
            status="running",
            source="cloud",
            os_type="linux",
            created_at="2020-01-01T00:00:00Z",
        ),
        SimpleNamespace(
            name="foreign-old",
            status="running",
            source="cloud",
            os_type="linux",
            created_at="2020-01-01T00:00:00Z",
        ),
    ]
    _calls, deleted = _install_fake_cua(monkeypatch, sandboxes)

    result = CliRunner().invoke(
        app,
        [
            "environment",
            "cleanup",
            "--sandbox",
            "cua",
            "--dry-run",
            "--max-age",
            "60",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["provider"] == "cua"
    assert payload["status"] == "dry-run"
    assert payload["dry_run"] is True
    assert payload["found"] == 2
    assert payload["matched"] == 1
    assert payload["skipped"] == 1
    assert payload["deleted"] == []
    assert [item["name"] for item in payload["candidates"]] == ["benchflow-old"]
    assert payload["candidates"][0]["would_delete"] is True
    assert deleted == []


def test_environment_cleanup_cua_deletes_matching_prefix(monkeypatch) -> None:
    sandboxes = [
        SimpleNamespace(
            name="benchflow-old",
            status="running",
            source="cloud",
            os_type="linux",
            created_at="2020-01-01T00:00:00Z",
        )
    ]
    _calls, deleted = _install_fake_cua(monkeypatch, sandboxes)

    result = CliRunner().invoke(
        app,
        ["environment", "cleanup", "--sandbox", "cua", "--max-age", "60"],
    )

    assert result.exit_code == 0
    assert "1 Cua environment(s) deleted" in result.output
    assert deleted == ["benchflow-old"]


def test_environment_cleanup_cua_json_reports_deleted_names(monkeypatch) -> None:
    sandboxes = [
        SimpleNamespace(
            name="benchflow-old",
            status="running",
            source="cloud",
            os_type="linux",
            created_at="2020-01-01T00:00:00Z",
        )
    ]
    _calls, deleted = _install_fake_cua(monkeypatch, sandboxes)

    result = CliRunner().invoke(
        app,
        [
            "environment",
            "cleanup",
            "--sandbox",
            "cua",
            "--max-age",
            "60",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["provider"] == "cua"
    assert payload["status"] == "deleted"
    assert payload["dry_run"] is False
    assert payload["found"] == 1
    assert payload["matched"] == 1
    assert payload["deleted"] == ["benchflow-old"]
    assert payload["candidates"][0]["would_delete"] is False
    assert deleted == ["benchflow-old"]
