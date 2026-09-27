"""The full rubric review in the benchflow.trial export.

Viewers read rubric reviews through the versioned trial document, not through the on-disk layout. Before 1.1 the
document carried only the one detached ``bench review`` verdict it could find
(names, outcomes, scores, explanations): no criterion descriptions or scales,
no reviewer effort, no weighted arithmetic, and nothing from the automatic
reviewer's scoring revisions (``bench eval run`` with a task rubric, or
``bench eval score``). ``rubric_reviews`` now lists every revision and every
detached audit of the trial, each with the rubric definition, per-criterion
verdicts with evidence references, the reviewer and the reward computation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import benchflow as bf
from benchflow import job_export
from tests.test_python_sdk_load_job import _trial

jsonschema = pytest.importorskip("jsonschema")
SCHEMAS = Path(__file__).resolve().parents[1] / "docs/reference/schemas"

RUBRIC = {
    "criteria": [
        {
            "name": "answer_correct",
            "blocker": 1,
            "weight": 1,
            "description": "The answer is 42.",
            "guidance": "PASS when answer.txt holds 42.",
        },
        {
            "name": "explanation_clear",
            "blocker": 0,
            "weight": 3,
            "description": "The explanation names the factors.",
            "guidance": "Score 2 when both factors are named.",
        },
        {
            "name": "tests_added",
            "blocker": 0,
            "weight": 1,
            "description": "A test covers the answer.",
            "guidance": "Score 2 when a test exists.",
        },
    ]
}

REVIEWER = {
    "agent": "claude-agent-acp",
    "model": "claude-sonnet-4-6",
    "reasoning_effort": "high",
    "environment": "daytona",
    "timeout_sec": 300,
    "concurrency": 1,
    "image": "reviewer:1",
    "open_network": False,
    "agent_env": {"BENCHFLOW_PROVIDER_MODEL": "claude-sonnet-4-6"},
    "agent_env_keys": ["BENCHFLOW_PROVIDER_MODEL", "CLAUDE_CODE_OAUTH_TOKEN"],
}

CHECKS = {
    "answer_correct": {
        "outcome": "pass",
        "explanation": "/evidence/workspace/answer.txt holds '42'; "
        "/evidence/trial/verifier/reward.txt is 1.",
    },
    "explanation_clear": {
        "score": 1,
        "explanation": "explanation.txt gives 42 but names only 6 "
        "(/evidence/workspace/explanation.txt:1).",
    },
    "tests_added": {
        "score": 2,
        "explanation": "/evidence/workspace/tests/test_answer.py checks 42.",
    },
}


def _revision(trial: Path, attempt: str, scoring: dict, **extra) -> None:
    (trial / "reviews" / attempt).mkdir(parents=True)
    (trial / "reviews" / attempt / "rubric.json").write_text(json.dumps(RUBRIC))
    details = {
        "attempt": attempt,
        "rubric_path": "rubric.json",
        "rubric_sha256": "9ac3",
        "contract": "v0.2",
        "reviewer": REVIEWER,
        "task_digest": "sha256:3cf7",
        "rubric_snapshot": f"reviews/{attempt}/rubric.json",
        **extra,
        "scoring": {**scoring, "revision": f"scoring/{attempt}.json"},
    }
    (trial / "scoring").mkdir(exist_ok=True)
    (trial / "scoring" / f"{attempt}.json").write_text(json.dumps(details))


def _reviewed_trial(root: Path) -> Path:
    trial = _trial(root / "jobs" / "j", "hello")
    run = "reviews/bbb/runtime/hello/x/review-hello__1"
    # An earlier revision whose reviewer failed, then the committed one.
    _revision(
        trial,
        "aaa",
        {
            "schema_version": 1,
            "policy": "tests-blockers-quality-v1",
            "status": "error",
            "tests_pass": True,
            "verifier_reward": 1.0,
            "error": "reviewer timed out",
            "failed_blockers": [],
        },
        review_valid=False,
    )
    complete = {
        "schema_version": 1,
        "policy": "tests-blockers-quality-v1",
        "status": "complete",
        "passed": True,
        "tests_pass": True,
        "all_blockers_pass": True,
        "failed_blockers": [],
        "verifier_reward": 1.0,
        "rubric_reward": 0.625,
        "reviewer_run": run,
        "error": None,
    }
    _revision(
        trial,
        "bbb",
        complete,
        checks=CHECKS,
        summary="Right answer, half an explanation.",
        review_valid=True,
    )
    result = json.loads((trial / "result.json").read_text())
    result["scoring"] = {**complete, "revision": "scoring/bbb.json"}
    result["rewards"] = {
        "reward": 0.625,
        "verifier_reward": 1.0,
        "rubric_reward": 0.625,
    }
    (trial / "result.json").write_text(json.dumps(result))
    # A detached `bench review` audit of the same trial.
    task = root / "tasks" / "hello"
    task.mkdir(parents=True)
    (task / "rubric.json").write_text(json.dumps(RUBRIC))
    report = root / "jobs" / "review-1" / "review_report.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps(
            {
                "path": str(trial.parent),
                "rubric": {
                    "path": str(task / "rubric.json"),
                    "criteria": [c["name"] for c in RUBRIC["criteria"]],
                    "contracts": ["v0.2"],
                },
                "reviewer": {
                    "agent": "codex-acp",
                    "model": "gpt-5.5",
                    "environment": "daytona",
                    "network": "no-internet",
                },
                "trials": [
                    {
                        "trial_name": trial.name,
                        "source_rollout": str(trial),
                        "review_valid": True,
                        "summary": "Audit agrees.",
                        "checks": {
                            **CHECKS,
                            "tests_added": {**CHECKS["tests_added"], "score": 0},
                        },
                        "error": None,
                        "reviewer_rollout": "review-1/runtime/r1",
                        "rubric_path": str(task / "rubric.json"),
                        "rubric_contract": "v0.2",
                        "criteria": [c["name"] for c in RUBRIC["criteria"]],
                        "criterion_metadata": [
                            {
                                "name": c["name"],
                                "blocker": c["blocker"],
                                "weight": c["weight"],
                            }
                            for c in RUBRIC["criteria"]
                        ],
                        "scoring": {
                            "deterministic_pass": True,
                            "all_blockers_pass": True,
                            "failed_blockers": [],
                            "weighted_points": 3,
                            "max_weighted_points": 8,
                            "raw_quality": 0.375,
                            "gated_quality": 0.375,
                            "decision": "not_publishable",
                        },
                        "notes": [],
                    }
                ],
            }
        )
    )
    return trial


def test_the_trial_export_carries_every_rubric_review(tmp_path: Path) -> None:
    doc = bf.load_trial(_reviewed_trial(tmp_path)).to_json_dict()
    schema = json.loads((SCHEMAS / job_export.schema_filename("trial")).read_text())
    jsonschema.validate(doc, schema, cls=jsonschema.Draft202012Validator)
    assert doc["schema_version"] == 1 and doc["schema_minor"] == 1

    reviews = doc["rubric_reviews"]
    assert [(r["kind"], r["id"], r["current"]) for r in reviews] == [
        ("revision", "aaa", False),
        ("revision", "bbb", True),
        ("audit", "review-1", False),
    ]
    failed, current, audit = reviews

    # The committed revision: rubric, reviewer, verdicts, arithmetic.
    assert current["source"] == "scoring/bbb.json" and current["status"] == "complete"
    assert current["reviewer"] == {
        "agent": "claude-agent-acp",
        "model": "claude-sonnet-4-6",
        "reasoning_effort": "high",
        "environment": "daytona",
        "run": "reviews/bbb/runtime/hello/x/review-hello__1",
    }
    rubric = current["rubric"]
    assert rubric["contract"] == "v0.2" and rubric["sha256"] == "9ac3"
    assert rubric["snapshot"] == "reviews/bbb/rubric.json"
    assert [
        (c["name"], c["kind"], c["weight"], c["scale"]) for c in rubric["criteria"]
    ] == [
        ("answer_correct", "blocker", 1, ["pass", "fail"]),
        ("explanation_clear", "scored", 3, [0, 1, 2]),
        ("tests_added", "scored", 1, [0, 1, 2]),
    ]
    assert rubric["criteria"][1]["guidance"] == "Score 2 when both factors are named."

    verdicts = {v["name"]: v for v in current["verdicts"]}
    assert verdicts["answer_correct"]["outcome"] == "pass"
    assert verdicts["answer_correct"]["points"] is None  # blockers gate, not score
    assert verdicts["explanation_clear"]["score"] == 1
    assert verdicts["explanation_clear"]["points"] == 3
    assert verdicts["explanation_clear"]["max_points"] == 6
    assert verdicts["explanation_clear"]["evidence"] == [
        {"area": "workspace", "path": "explanation.txt", "line": 1}
    ]
    assert verdicts["answer_correct"]["evidence"] == [
        {"area": "workspace", "path": "answer.txt", "line": None},
        {"area": "trial", "path": "verifier/reward.txt", "line": None},
    ]

    reward = current["reward"]
    assert reward["weighted_points"] == 5 and reward["max_weighted_points"] == 8
    assert reward["rubric_reward"] == 0.625 and reward["verifier_reward"] == 1.0
    assert reward["tests_pass"] and reward["all_blockers_pass"]
    assert reward["reward"] == 0.625 and reward["passed"] is True
    assert reward["formula"] == (
        "reward = rubric_reward if tests pass and every blocker passes, else 0; "
        "rubric_reward = weighted_points / max_weighted_points = 5 / 8"
    )

    assert failed["status"] == "error" and failed["error"] == "reviewer timed out"
    assert failed["verdicts"] == [] and failed["reward"]["reward"] is None

    # The detached audit: no effort recorded, its own arithmetic, not current.
    assert audit["source"].endswith("review-1/review_report.json")
    assert audit["reviewer"]["model"] == "gpt-5.5"
    assert audit["reviewer"]["reasoning_effort"] is None
    assert audit["reviewer"]["run"] == "review-1/runtime/r1"
    assert audit["rubric"]["criteria"][0]["description"] == "The answer is 42."
    assert audit["reward"]["weighted_points"] == 3
    assert audit["reward"]["decision"] == "not_publishable"
    assert audit["reward"]["reward"] is None  # an audit never changes the reward


def test_a_trial_without_reviews_has_an_empty_list(tmp_path: Path) -> None:
    doc = bf.load_trial(_trial(tmp_path / "jobs" / "j", "hello")).to_json_dict()
    assert doc["rubric_reviews"] == [] and doc["schema_minor"] == 1
