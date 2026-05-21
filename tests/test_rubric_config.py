"""Tests for rubric.toml / rubric.json parsing (ENG-55, #270)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow.rewards.rubric_config import (
    Criterion,
    JudgeConfig,
    RubricConfig,
    ScoringConfig,
    load_rubric,
    load_rubric_json,
    load_rubric_toml,
)

# ---------------------------------------------------------------------------
# Criterion normalization
# ---------------------------------------------------------------------------


class TestCriterionNormalize:
    def test_binary_pass(self) -> None:
        c = Criterion(description="test", type="binary")
        assert c.normalize(1.0) == 1.0
        assert c.normalize(0.7) == 1.0

    def test_binary_fail(self) -> None:
        c = Criterion(description="test", type="binary")
        assert c.normalize(0.0) == 0.0
        assert c.normalize(0.4) == 0.0

    def test_likert_normalization(self) -> None:
        c = Criterion(description="test", type="likert", points=5)
        assert c.normalize(1) == pytest.approx(0.0)
        assert c.normalize(5) == pytest.approx(1.0)
        assert c.normalize(3) == pytest.approx(0.5)

    def test_likert_single_point(self) -> None:
        c = Criterion(description="test", type="likert", points=1)
        assert c.normalize(1) == 0.0

    def test_numeric_normalization(self) -> None:
        c = Criterion(description="test", type="numeric", min=0.0, max=100.0)
        assert c.normalize(0) == pytest.approx(0.0)
        assert c.normalize(100) == pytest.approx(1.0)
        assert c.normalize(50) == pytest.approx(0.5)

    def test_numeric_clamp(self) -> None:
        c = Criterion(description="test", type="numeric", min=0.0, max=10.0)
        assert c.normalize(15) == 1.0
        assert c.normalize(-5) == 0.0

    def test_numeric_zero_span(self) -> None:
        c = Criterion(description="test", type="numeric", min=5.0, max=5.0)
        assert c.normalize(5) == 0.0


# ---------------------------------------------------------------------------
# Criterion id
# ---------------------------------------------------------------------------


class TestCriterionId:
    def test_explicit_name(self) -> None:
        c = Criterion(description="some long description", name="my-crit")
        assert c.id == "my-crit"

    def test_truncated_description(self) -> None:
        c = Criterion(description="a" * 100)
        assert c.id == "a" * 40


# ---------------------------------------------------------------------------
# TOML parsing
# ---------------------------------------------------------------------------


class TestLoadRubricToml:
    def test_full_rubric(self, tmp_path: Path) -> None:
        toml_content = """\
[judge]
model = "gpt-4o"
mode = "batched"
files = ["/app/output/report.md"]
timeout = 60

[[criterion]]
description = "Is the code correct?"
type = "binary"

[[criterion]]
description = "How readable?"
type = "likert"
points = 5
weight = 2.0

[[criterion]]
description = "Rate coverage"
type = "numeric"
min = 0
max = 100
name = "coverage"

[scoring]
aggregation = "all_pass"
"""
        rubric_file = tmp_path / "rubric.toml"
        rubric_file.write_text(toml_content)

        cfg = load_rubric_toml(rubric_file)

        assert cfg.judge.model == "gpt-4o"
        assert cfg.judge.mode == "batched"
        assert cfg.judge.files == ["/app/output/report.md"]
        assert cfg.judge.timeout == 60

        assert len(cfg.criteria) == 3
        assert cfg.criteria[0].type == "binary"
        assert cfg.criteria[1].type == "likert"
        assert cfg.criteria[1].points == 5
        assert cfg.criteria[1].weight == 2.0
        assert cfg.criteria[2].type == "numeric"
        assert cfg.criteria[2].name == "coverage"
        assert cfg.criteria[2].id == "coverage"

        assert cfg.scoring.aggregation == "all_pass"

    def test_minimal_rubric(self, tmp_path: Path) -> None:
        toml_content = """\
[[criterion]]
description = "Does it work?"
"""
        rubric_file = tmp_path / "rubric.toml"
        rubric_file.write_text(toml_content)

        cfg = load_rubric_toml(rubric_file)

        # Defaults
        assert cfg.judge.model == "claude-sonnet-4-6"
        assert cfg.judge.mode == "individual"
        assert cfg.scoring.aggregation == "weighted_mean"
        assert len(cfg.criteria) == 1
        assert cfg.criteria[0].type == "binary"

    def test_empty_rubric(self, tmp_path: Path) -> None:
        rubric_file = tmp_path / "rubric.toml"
        rubric_file.write_text("")

        cfg = load_rubric_toml(rubric_file)
        assert len(cfg.criteria) == 0


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_judge_config_defaults(self) -> None:
        j = JudgeConfig()
        assert j.model == "claude-sonnet-4-6"
        assert j.mode == "individual"
        assert j.files == []
        assert j.timeout == 120

    def test_scoring_config_defaults(self) -> None:
        s = ScoringConfig()
        assert s.aggregation == "weighted_mean"
        assert s.threshold == 0.7

    def test_rubric_config_defaults(self) -> None:
        r = RubricConfig()
        assert r.criteria == []
        assert r.judge.model == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Harvey LAB style rubric.json parsing (#270)
# ---------------------------------------------------------------------------


class TestLoadRubricJson:
    def test_harvey_lab_rubric(self, tmp_path: Path) -> None:
        """#270: a Harvey LAB rubric.json maps id/title/match_criteria."""
        rubric_file = tmp_path / "rubric.json"
        rubric_file.write_text(
            json.dumps(
                {
                    "title": "Corporate M&A Analysis",
                    "criteria": [
                        {
                            "id": "criterion-1",
                            "title": "Identifies the purchase price",
                            "match_criteria": "The memo states the $X price.",
                        },
                        {
                            "id": "criterion-2",
                            "title": "Flags the indemnity cap",
                            "match_criteria": "The memo discusses the cap.",
                        },
                    ],
                }
            )
        )

        cfg = load_rubric_json(rubric_file)

        assert len(cfg.criteria) == 2
        assert cfg.criteria[0].name == "criterion-1"
        assert cfg.criteria[0].id == "criterion-1"
        assert cfg.criteria[0].description == "The memo states the $X price."
        assert cfg.criteria[0].type == "binary"
        # judge/scoring defaults apply when absent
        assert cfg.judge.model == "claude-sonnet-4-6"
        assert cfg.scoring.aggregation == "weighted_mean"

    def test_rubric_json_with_judge_and_scoring(self, tmp_path: Path) -> None:
        """#270: a JSON rubric may carry its own judge/scoring config."""
        rubric_file = tmp_path / "rubric.json"
        rubric_file.write_text(
            json.dumps(
                {
                    "criteria": [{"id": "c1", "match_criteria": "ok"}],
                    "judge": {"model": "gpt-4o"},
                    "scoring": {"aggregation": "all_pass"},
                }
            )
        )
        cfg = load_rubric_json(rubric_file)
        assert cfg.judge.model == "gpt-4o"
        assert cfg.scoring.aggregation == "all_pass"

    def test_rubric_json_title_fallback(self, tmp_path: Path) -> None:
        """A criterion with no match_criteria falls back to its title."""
        rubric_file = tmp_path / "rubric.json"
        rubric_file.write_text(
            json.dumps({"criteria": [{"id": "c1", "title": "Be correct"}]})
        )
        cfg = load_rubric_json(rubric_file)
        assert cfg.criteria[0].description == "Be correct"


class TestLoadRubricDispatch:
    def test_dispatches_to_json(self, tmp_path: Path) -> None:
        """#270: load_rubric routes .json files to the JSON parser."""
        rubric_file = tmp_path / "rubric.json"
        rubric_file.write_text(
            json.dumps({"criteria": [{"id": "c1", "match_criteria": "ok"}]})
        )
        cfg = load_rubric(rubric_file)
        assert len(cfg.criteria) == 1
        assert cfg.criteria[0].id == "c1"

    def test_dispatches_to_toml(self, tmp_path: Path) -> None:
        """#270: load_rubric routes .toml files to the TOML parser."""
        rubric_file = tmp_path / "rubric.toml"
        rubric_file.write_text('[[criterion]]\ndescription = "Does it work?"\n')
        cfg = load_rubric(rubric_file)
        assert len(cfg.criteria) == 1
