"""The installed agy version in a trial's harness_versions.

0.10.0rc4's dogfood recorded Gemini 4 Hydrogen trials on the antigravity
agent with adapter, cli and engine all "unavailable": the receipt probe only
read npm packages, and agy is a pinned native binary behind BenchFlow's own
ACP shim. The probe now asks the installed binary.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from benchflow import __version__ as benchflow_version
from benchflow.agents.antigravity_config import AGY_BIN, AGY_VERSION
from benchflow.agents.receipt import probe_command, probe_receipt


class _Env:
    def __init__(self, answers):
        self.answers = answers
        self.commands = []

    async def exec(self, command, **_kw):
        self.commands.append(command)
        for needle, (rc, out) in self.answers.items():
            if needle in command:
                return SimpleNamespace(stdout=out, stderr="", return_code=rc)
        return SimpleNamespace(stdout="", stderr="", return_code=127)


def test_the_probe_asks_the_installed_binary():
    command = probe_command("antigravity")
    assert command is not None and AGY_BIN in command and "--version" in command


def test_the_receipt_names_agys_version():
    env = _Env(
        {"--version": (0, f'{{"version": "{AGY_VERSION}", "sha512": "{"f" * 128}"}}\n')}
    )
    receipt = asyncio.run(probe_receipt(env, "antigravity", "acp"))
    for role in ("cli", "engine"):
        assert receipt[role]["version"] == AGY_VERSION, receipt
        assert receipt[role]["source"] == "installed-binary", receipt
    assert receipt["adapter"]["source"] == "bundled"
    assert receipt["adapter"]["version"] == benchflow_version


def test_a_probe_that_fails_stays_unavailable():
    receipt = asyncio.run(probe_receipt(_Env({}), "antigravity", "acp"))
    assert receipt["cli"]["source"] == "unavailable"
