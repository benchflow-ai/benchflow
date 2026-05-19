"""Regression tests for bundled docs/example tasks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_nanofirm_evaluator():
    path = Path("docs/examples/nanofirm-task/tests/evaluate.py")
    spec = importlib.util.spec_from_file_location("nanofirm_evaluate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nanofirm_perfect_analysis_reaches_full_reward(tmp_path, monkeypatch):
    """Guards ENG-87: visible scoring increments should allow reward 1.0."""
    module = _load_nanofirm_evaluator()
    analysis = {
        "risks": [
            {
                "clause": "1",
                "severity": "high",
                "issue": "uncapped liability",
                "recommendation": "cap liability",
            },
            {
                "clause": "2",
                "severity": "medium",
                "issue": "broad assignment",
                "recommendation": "narrow assignment",
            },
            {
                "clause": "3",
                "severity": "medium",
                "issue": "weak termination rights",
                "recommendation": "add termination trigger",
            },
        ],
        "compound_risks": [
            {
                "clauses": ["1", "3"],
                "severity": "high",
                "issue": "uncapped post-termination exposure",
                "recommendation": "align cap and termination language",
            }
        ],
        "deal_breakers": ["uncapped liability"],
        "summary": "This analysis identifies the main commercial risks clearly.",
    }
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(analysis))
    monkeypatch.setattr(module, "ANALYSIS_PATH", str(analysis_path))

    assert module.evaluate() == 1.0
