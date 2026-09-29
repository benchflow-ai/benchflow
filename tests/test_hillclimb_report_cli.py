"""The static report and ``bench hillclimb``'s command line."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import click
import pytest
import typer.main
from typer.testing import CliRunner

import benchflow as bf
from benchflow.cli.hillclimb import FLAG_FIELDS
from benchflow.cli.main import app
from benchflow.hillclimbing import HillclimbConfig, ProposerSettings
from tests._hillclimb_fakes import FakeAgent, FakeProposer, append_to_skill
from tests.test_hillclimb_engine import TEST, TRAIN, _config, _setup

REPO = Path(__file__).resolve().parents[1]


def _run(tmp_path, monkeypatch):
    def reward(task, surface, trial):
        return 1.0 if "FIX" in surface and task != "t0" else 0.0

    FakeAgent(reward).install(monkeypatch)
    FakeProposer(
        [append_to_skill('FIX: validate <outputs> & "quote" them')],
        root=tmp_path / "sandboxes",
    ).install(monkeypatch)
    return bf.hillclimb(_config(tmp_path))


def test_the_report_is_self_contained_and_escaped(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch)
    page = result.report.read_text()
    # No network: nothing is loaded from elsewhere.
    assert not re.search(r"(src|href)=\"?(https?:)?//", page)
    assert "@import" not in page and "url(http" not in page
    # The verdict, the curve, the decisions and the diff are all there.
    assert "Gain exceeds noise" in page
    assert page.count("<svg") >= 1 and "Score by round" in page
    assert "r01-c1" in page and "Kept" in page
    # Text from a run is escaped, never interpreted as markup.
    assert "&lt;outputs&gt; &amp; &quot;quote&quot;" in page
    assert "<outputs>" not in page


def test_bf_hillclimb_is_public():
    assert bf.hillclimb is bf.hillclimbing.hillclimb
    assert {"hillclimb", "ahillclimb", "HillclimbConfig", "HillclimbResult"} <= set(
        bf.__all__
    )


def _command() -> click.Command:
    root = typer.main.get_command(app)
    return root.get_command(click.Context(root), "hillclimb")


def test_every_flag_sets_a_config_field_and_every_field_has_a_flag():
    flags = {
        next(o for o in p.opts if o.startswith("--"))
        for p in _command().params
        if isinstance(p, click.Option)
    }
    flags -= {"--help", "--quiet"}
    assert flags == set(FLAG_FIELDS)
    top = {f.name for f in dataclasses.fields(HillclimbConfig)}
    inner = {f.name for f in dataclasses.fields(ProposerSettings)}
    for flag, target in FLAG_FIELDS.items():
        head, _, attr = target.partition(".")
        assert head in top, flag
        if attr:
            assert attr in inner, flag
    # Python-only knobs, on purpose.
    assert top - set(FLAG_FIELDS.values()) - {"proposer"} == {"preflight"}
    assert inner - {t.split(".", 1)[1] for t in FLAG_FIELDS.values() if "." in t} == {
        "extra_instructions"
    }


def test_every_flag_is_documented():
    doc = (REPO / "docs/hillclimb.md").read_text()
    missing = sorted(f for f in FLAG_FIELDS if f"`{f}" not in doc)
    assert missing == []


def test_the_cli_runs_a_climb_and_exits_2_when_the_gate_refuses(tmp_path, monkeypatch):
    import random

    tasks, skills, split = _setup(tmp_path)
    FakeAgent(
        lambda task, surface, trial: float(
            random.Random(f"{task}{trial}").random() < 0.5
        )
    ).install(monkeypatch)
    FakeProposer().install(monkeypatch)
    out = tmp_path / "run"
    result = CliRunner().invoke(
        app,
        [
            "hillclimb",
            "--tasks-dir",
            str(tasks),
            "--surface",
            str(skills),
            "--split-file",
            str(split),
            "--out",
            str(out),
            "--sandbox",
            "modal",
            "--proposer-sandbox",
            "modal",
            "--model",
            "claude-haiku-4-5",
            "--trials",
            "2",
            "--min-gain",
            "0.05",
            "--retry-attempts",
            "0",
            "--bootstrap-samples",
            "200",
            "--quiet",
        ],
    )
    assert result.exit_code == 2, result.output
    assert "Refusing to climb" in result.output
    doc = bf.hillclimbing.load(out).record
    assert doc.status == "refused"
    assert doc.split.train == TRAIN and doc.split.test == TEST
    assert doc.config.model == "claude-haiku-4-5"


def test_the_cli_needs_exactly_one_task_source(tmp_path):
    result = CliRunner().invoke(app, ["hillclimb", "--surface", str(tmp_path)])
    assert result.exit_code == 1
    assert "--tasks-dir" in result.output


@pytest.mark.parametrize(
    "flag,value", [("--objective", "speed"), ("--leak-check", "maybe")]
)
def test_the_cli_refuses_unknown_choices(tmp_path, flag, value):
    result = CliRunner().invoke(
        app,
        [
            "hillclimb",
            "--tasks-dir",
            str(tmp_path),
            "--surface",
            str(tmp_path),
            flag,
            value,
        ],
    )
    assert result.exit_code == 1
    assert flag in result.output
