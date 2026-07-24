"""Shared helpers for bench init CLI tests."""

from __future__ import annotations

from typer.testing import CliRunner

from benchflow.cli.main import app

__all__ = ["_init_args", "app", "runner"]

runner = CliRunner()


def _init_args(home, extra=()):
    return [
        "init",
        "--model",
        "deepseek/deepseek-v4-flash",
        "--agent",
        "pi-acp",
        "--dataset",
        "skillsbench@1.1",
        "--sandbox",
        "docker",
        "--api-key",
        "sk-test-123",
        "--skip-smoke",
        *extra,
    ]
