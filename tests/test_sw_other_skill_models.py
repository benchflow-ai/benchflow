"""Guards FF-sw-other model aggregation and trace identity at 363d815b."""

import json

from benchflow.skill_eval import (
    CaseResult,
    EvalCase,
    EvalDataset,
    SkillEvalResult,
    SkillEvaluator,
    export_gepa_traces,
)


def test_skill_scores_keep_agent_and_model_identity(tmp_path):
    dataset = EvalDataset("dummy", tmp_path, [EvalCase("case-1", "dummy")])
    evaluator = SkillEvaluator.__new__(SkillEvaluator)
    evaluator.dataset = dataset
    results = [
        CaseResult(
            "case-1",
            "agent",
            model,
            mode,
            float(model == "provider-a/model") if mode else 0.0,
        )
        for model in ("provider-a/model", "provider-b/model", "")
        for mode in (True, False)
    ]
    lifts = evaluator._compute_lifts(
        results, ["agent"] * 3, ["provider-a/model", "provider-b/model", None], False
    )
    assert [x.with_skill_score for x in lifts] == [1.0, 0.0, 0.0]
    assert [x.lift for x in lifts] == [1.0, 0.0, 0.0]
    result = SkillEvalResult("dummy", 1, ["agent"], case_results=results)
    out = export_gepa_traces(result, dataset, tmp_path / "out")
    files = list((out / "traces").glob("*.json"))
    assert len(files) == 6
    assert {
        (x["model"], x["with_skill"])
        for x in (json.loads(f.read_text()) for f in files)
    } == {
        (model, mode)
        for model in ("provider-a/model", "provider-b/model", "")
        for mode in (True, False)
    }


def test_trace_identity_survives_basename_and_delimiter_collisions(tmp_path):
    """Guards FF-sw-other trace identities at 363d815b, including partial exports."""
    dataset = EvalDataset(
        "dummy", tmp_path, [EvalCase("a", "dummy"), EvalCase("a-b", "dummy")]
    )
    rows = [
        CaseResult("a", "one/agent", "x/model", True, 1.0),
        CaseResult("a", "two/agent", "x/model", True, 0.0),
        CaseResult("a-b", "agent", "x/model", True, 0.5),
        CaseResult("a", "b-agent", "x/model", True, 0.25),
    ]
    result = SkillEvalResult("dummy", 2, [], case_results=rows)
    out = export_gepa_traces(result, dataset, tmp_path / "out")
    before = {p.name: p.read_text() for p in (out / "traces").glob("*.json")}
    assert len(before) == 4
    for row in rows:
        export_gepa_traces(
            SkillEvalResult("dummy", 2, [], case_results=[row]), dataset, out
        )
    assert {p.name: p.read_text() for p in (out / "traces").glob("*.json")} == before
