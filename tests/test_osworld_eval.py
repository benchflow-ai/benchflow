"""Unit tests for the real OSWorld evaluator orchestration.

Driven by the real OSWorld ``os/5812b315`` (create SSH user) task shape, with a
recording fake ``run_command`` standing in for the desktop ``/execute`` server.
"""

from __future__ import annotations

import pytest

from benchflow.adapters.osworld_eval import UnsupportedGetterError, evaluate

# The real os/5812b315 evaluator (create SSH user) — postconfig installs expect +
# downloads a check script, the result getter runs the check, the metric requires
# the success line to be present.
_SSH_USER_EVALUATOR = {
    "postconfig": [
        {
            "type": "execute",
            "parameters": {
                "command": "echo pw | sudo -S apt-get install -y expect",
                "shell": True,
            },
        },
        {
            "type": "download",
            "parameters": {
                "files": [
                    {
                        "url": "https://example.invalid/check_password.sh",
                        "path": "check_password.sh",
                    }
                ]
            },
        },
        {"type": "execute", "parameters": {"command": "chmod +x check_password.sh", "shell": True}},
    ],
    "func": "check_include_exclude",
    "result": {
        "type": "vm_command_line",
        "command": "./check_password.sh charles 'pw'",
        "shell": True,
    },
    "expected": {
        "type": "rule",
        "rules": {
            "include": ["Password, home directory, and write permission check passed"],
            "exclude": [],
        },
    },
}

_SUCCESS = "Password, home directory, and write permission check passed\n"


def _recorder(check_output: str):
    calls: list[tuple] = []

    def run_command(command, shell):
        calls.append((command, shell))
        # The result getter's check command returns the simulated check output.
        if isinstance(command, str) and command.startswith("./check_password.sh"):
            return check_output
        return ""

    return run_command, calls


def test_ssh_user_task_passes_when_check_output_matches() -> None:
    run_command, calls = _recorder(_SUCCESS)
    assert evaluate({"evaluator": _SSH_USER_EVALUATOR}, run_command) == 1.0
    # postconfig ran: apt-get install, the download (curl), chmod, then the check.
    joined = " ".join(str(c[0]) for c in calls)
    assert "apt-get install -y expect" in joined
    assert "curl -fsSL" in joined and "check_password.sh" in joined
    assert "chmod +x check_password.sh" in joined


def test_ssh_user_task_fails_when_check_output_is_failure() -> None:
    run_command, _ = _recorder("Check failed\n")
    assert evaluate({"evaluator": _SSH_USER_EVALUATOR}, run_command) == 0.0


def test_exact_match_path() -> None:
    evaluator = {
        "func": "exact_match",
        "result": {"type": "vm_command_line", "command": "echo hi", "shell": True},
        "expected": {"type": "rule", "rules": {"expected": "hi\n"}},
    }

    def run_command(command, shell):
        return "hi\n"

    assert evaluate({"evaluator": evaluator}, run_command) == 1.0


def test_multi_metric_and_conj_requires_all() -> None:
    evaluator = {
        "func": ["check_include_exclude", "check_include_exclude"],
        "conj": "and",
        "result": [
            {"type": "vm_command_line", "command": "a", "shell": True},
            {"type": "vm_command_line", "command": "b", "shell": True},
        ],
        "expected": [
            {"type": "rule", "rules": {"include": ["ok"]}},
            {"type": "rule", "rules": {"include": ["MISSING"]}},
        ],
    }

    def run_command(command, shell):
        return "ok\n"  # satisfies the first include, not the second

    assert evaluate({"evaluator": evaluator}, run_command) == 0.0


def test_multi_metric_or_conj_needs_one() -> None:
    evaluator = {
        "func": ["check_include_exclude", "check_include_exclude"],
        "conj": "or",
        "result": [
            {"type": "vm_command_line", "command": "a", "shell": True},
            {"type": "vm_command_line", "command": "b", "shell": True},
        ],
        "expected": [
            {"type": "rule", "rules": {"include": ["ok"]}},
            {"type": "rule", "rules": {"include": ["MISSING"]}},
        ],
    }

    def run_command(command, shell):
        return "ok\n"

    assert evaluate({"evaluator": evaluator}, run_command) == 1.0


def test_unsupported_getter_raises() -> None:
    evaluator = {
        "func": "exact_match",
        "result": {"type": "accessibility_tree", "command": "x"},
        "expected": {"type": "rule", "rules": {"expected": "x"}},
    }
    with pytest.raises(UnsupportedGetterError):
        evaluate({"evaluator": evaluator}, lambda c, s: "")


def test_template_vars_substituted_in_commands() -> None:
    seen: list = []

    def run_command(command, shell):
        seen.append(command)
        return "ok\n"

    evaluator = {
        "postconfig": [
            {
                "type": "execute",
                "parameters": {
                    "command": "echo {CLIENT_PASSWORD} | sudo -S true",
                    "shell": True,
                },
            }
        ],
        "func": "check_include_exclude",
        "result": {
            "type": "vm_command_line",
            "command": ["python", "-c", "print({SCREEN_WIDTH_HALF}, {SCREEN_HEIGHT_HALF})"],
            "shell": False,
        },
        "expected": {"type": "rule", "rules": {"include": ["ok"]}},
    }
    evaluate({"evaluator": evaluator}, run_command, password="s3cret", screen=(1920, 1080))
    # postconfig command had {CLIENT_PASSWORD} substituted; result list parts too.
    assert "echo s3cret | sudo -S true" in seen
    assert ["python", "-c", "print(960, 540)"] in seen
    # the raw tokens are gone
    assert not any("{CLIENT_PASSWORD}" in str(c) for c in seen)
    assert not any("{SCREEN_WIDTH_HALF}" in str(c) for c in seen)
