"""Deterministic wrong-answer / equivalence battery (``bench tasks check --level equivalence``).

Runs on BenchFlow's task and sandbox APIs. The sandbox half needs a real sandbox backend; these tests pin the
variant generation and the verdict logic with an in-process fake verifier.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from benchflow.task.equivalence import (
    EquivalenceReport,
    GradeOutcome,
    run_equivalence_battery,
)
from benchflow.task.equivalence_variants import (
    build_variants,
    extract_graded_literals,
    paraphrase,
)

ANSWER = "/app/answer.json"

ORACLE = {
    "results": [
        {"id": "r1", "value": 12.5, "count": 7, "unit": "usd"},
        {"id": "r2", "value": 3.25, "count": 2, "unit": "usd"},
    ],
    "methods": "Each value is the row total; the residual is reported apart, rounded to six decimals.",
}

LITERAL_VERIFIER = """
import json
ans = json.load(open("/app/answer.json"))
m = str(ans.get("methods", "")).lower()
ok = all(x in m for x in ("residual", "six decimals"))
"""


class FakeGrader:
    """Applies variant changes over the oracle output and scores them in-process."""

    def __init__(
        self,
        oracle: Mapping[str, bytes],
        verify: Callable[[dict[str, bytes]], float | None],
    ) -> None:
        self.oracle = dict(oracle)
        self.verify = verify
        self.calls: list[Mapping[str, bytes | None]] = []

    async def grade(self, changes: Mapping[str, bytes | None]) -> GradeOutcome:
        self.calls.append(changes)
        files = dict(self.oracle)
        for path, content in changes.items():
            if content is None:
                files.pop(path, None)
            else:
                files[path] = content
        try:
            reward = self.verify(files)
        except Exception as exc:  # the verifier crashed on this variant
            return GradeOutcome(reward=None, error=f"verifier crashed: {exc}")
        return GradeOutcome(reward=reward, error=None)


def _answer(files: dict[str, bytes]) -> dict:
    return json.loads(files[ANSWER])


def _strict_values(ans: dict) -> bool:
    rows = {r["id"]: r for r in ans["results"]}
    return (
        set(rows) == {"r1", "r2"}
        and abs(rows["r1"]["value"] - 12.5) < 1e-6
        and abs(rows["r2"]["value"] - 3.25) < 1e-6
        and rows["r1"]["count"] == 7
        and rows["r2"]["count"] == 2
    )


def good_verifier(files: dict[str, bytes]) -> float:
    if ANSWER not in files or not files[ANSWER].strip():
        return 0.0
    try:
        ans = _answer(files)
    except json.JSONDecodeError:
        return 0.0
    return 1.0 if _strict_values(ans) else 0.0


def literal_verifier(files: dict[str, bytes]) -> float:
    """Right on values, but grades the prose field by literal substring."""
    if good_verifier(files) < 1.0:
        return 0.0
    m = str(_answer(files).get("methods", "")).lower()
    return 1.0 if all(x in m for x in ("residual", "six decimals")) else 0.0


def lax_verifier(files: dict[str, bytes]) -> float:
    """Checks only that the answer parses and has two rows."""
    if ANSWER not in files or not files[ANSWER].strip():
        return 0.0
    try:
        ans = _answer(files)
    except json.JSONDecodeError:
        return 0.0
    return 1.0 if len(ans.get("results", [])) == 2 else 0.0


def _task(
    tmp_path: Path, instruction: str, verifier_src: str = LITERAL_VERIFIER
) -> Path:
    task = tmp_path / "task"
    (task / "tests").mkdir(parents=True)
    (task / "environment").mkdir()
    (task / "solution").mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text(instruction)
    (task / "tests" / "verify.py").write_text(verifier_src)
    (task / "tests" / "test.sh").write_text("python3 /tests/verify.py\n")
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    return task


INSTRUCTION = (
    "Write `/app/answer.json` with `results` (one row per request: `id`, "
    "`value`, `count`, `unit`) and a `methods` paragraph describing your method.\n"
)


def _run(
    task: Path, verify: Callable[[dict[str, bytes]], float | None], **kw
) -> tuple[EquivalenceReport, FakeGrader]:
    oracle = {ANSWER: json.dumps(ORACLE, indent=2).encode()}
    grader = FakeGrader(oracle, verify)
    report = asyncio.run(run_equivalence_battery(task, grader, oracle, **kw))
    return report, grader


# ------------------------------------------------------------------ variants


def test_paraphrase_rewrites_a_literal_without_filler() -> None:
    assert paraphrase("residual") == "remainder"
    assert paraphrase("six decimals") == "six decimal places"
    assert paraphrase("July 2025") == "July of 2025"
    assert paraphrase("zzz") is None


def test_extract_graded_literals_finds_prose_membership_checks() -> None:
    lits = extract_graded_literals({"tests/verify.py": LITERAL_VERIFIER})
    assert {g.literal for g in lits} == {"residual", "six decimals"}
    assert all(g.field == "methods" for g in lits)


def test_extract_graded_literals_ignores_non_prose_membership() -> None:
    src = (
        "import json\nans = json.load(open('/app/answer.json'))\n"
        "ok = ans.get('unit') in ('usd', 'eur')\n"
    )
    assert extract_graded_literals({"tests/verify.py": src}) == []


def test_build_variants_covers_every_family_with_exact_labels() -> None:
    answers = {ANSWER: json.dumps(ORACLE, indent=2).encode()}
    variants = build_variants(
        answers,
        instruction=INSTRUCTION,
        verifier_sources={"tests/verify.py": LITERAL_VERIFIER},
    )
    by_family = {v.family: v for v in variants}
    reject = {v.family for v in variants if v.expect == "reject"}
    accept = {v.family for v in variants if v.expect == "accept"}
    assert {
        "empty-output",
        "empty-file",
        "truncated",
        "wrong-number",
        "off-by-one",
        "wrong-unit",
        "swapped-rows",
    } <= reject
    assert {"reformat", "key-order", "precision", "row-order", "paraphrase"} <= accept
    # the empty output deletes the answer file
    assert by_family["empty-output"].changes == {ANSWER: None}
    # the wrong number names the field and the old and new value
    wrong = next(v for v in variants if v.family == "wrong-number")
    assert "results[0].value 12.5 -> 13.75" in wrong.label
    # an off-by-one moves the integer by exactly one
    off = by_family["off-by-one"]
    assert json.loads(off.changes[ANSWER])["results"][0]["count"] == 8
    # the reformat is value-identical
    assert json.loads(by_family["reformat"].changes[ANSWER]) == ORACLE
    assert by_family["reformat"].changes[ANSWER] != answers[ANSWER]


def test_text_answers_get_number_and_whitespace_variants() -> None:
    answers = {"/app/answer.txt": b"42\n"}
    variants = build_variants(
        answers, instruction="Write the count to /app/answer.txt."
    )
    changes = {
        v.family: v.changes["/app/answer.txt"]
        for v in variants
        if v.family != "empty-output"
    }
    assert changes["wrong-number"] != b"42\n"
    assert changes["off-by-one"] == b"43\n"
    assert changes["empty-file"] == b""
    assert changes["whitespace"] in (b"42", b"42\n\n")


def test_row_order_is_marked_fixed_when_the_instruction_orders_rows() -> None:
    answers = {ANSWER: json.dumps(ORACLE).encode()}
    variants = build_variants(
        answers, instruction=INSTRUCTION + "List results in request order.\n"
    )
    row = next(v for v in variants if v.family == "row-order")
    assert row.instruction_fixes_form


# ------------------------------------------------------------------- battery


def test_clean_verifier_has_no_false_negatives_or_positives(tmp_path: Path) -> None:
    task = _task(tmp_path, INSTRUCTION, verifier_src="")
    report, _ = _run(task, good_verifier)
    assert report.status == "clean", report.issues()
    assert report.false_negatives == []
    assert report.false_positives == []
    assert report.oracle_reward == 1.0


def test_literal_prose_verifier_reports_the_paraphrase_false_negative(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path, INSTRUCTION)
    report, _ = _run(task, literal_verifier)
    assert report.status == "defects"
    labels = [r.variant.label for r in report.false_negatives]
    assert any("'residual' -> 'remainder'" in lab for lab in labels), labels
    assert report.false_positives == []
    issue = next(i for i in report.issues() if "remainder" in i)
    assert issue.startswith("equivalence: false negative [paraphrase]")
    assert "scored 0.0" in issue and "oracle 1.0" in issue


def test_mandated_phrase_is_a_warning_not_a_false_negative(tmp_path: Path) -> None:
    ins = (
        INSTRUCTION
        + 'The `methods` field must contain the exact phrase "six decimals" and "residual".\n'
    )
    task = _task(tmp_path, ins)
    report, _ = _run(task, literal_verifier)
    assert report.false_negatives == []
    assert any(r.verdict == "form-fixed-by-instruction" for r in report.results)
    assert report.status == "clean"


def test_lax_verifier_reports_wrong_answers_it_accepts(tmp_path: Path) -> None:
    task = _task(tmp_path, INSTRUCTION, verifier_src="")
    report, _ = _run(task, lax_verifier)
    assert report.status == "defects"
    fams = {r.variant.family for r in report.false_positives}
    assert {"wrong-number", "off-by-one", "wrong-unit", "swapped-rows"} <= fams
    assert "empty-output" not in fams
    assert any(
        i.startswith("equivalence: false positive [wrong-number]")
        for i in report.issues()
    )


def test_blocker_when_oracle_and_empty_output_score_the_same(tmp_path: Path) -> None:
    task = _task(tmp_path, INSTRUCTION, verifier_src="")
    report, grader = _run(task, lambda files: 0.0)
    assert report.status == "blocked"
    assert "fails before reading the answer" in report.reason
    assert len(grader.calls) == 2  # oracle control and the empty output only


def test_oracle_that_does_not_pass_stops_the_battery(tmp_path: Path) -> None:
    task = _task(tmp_path, INSTRUCTION, verifier_src="")
    report, _ = _run(task, lambda files: 0.5 if ANSWER in files else 0.0)
    assert report.status == "oracle-fails"
    assert report.issues() == ["equivalence: oracle output scores 0.5, not 1.0"]


def test_verifier_that_accepts_an_empty_output_is_a_false_positive(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path, INSTRUCTION, verifier_src="")
    report, _ = _run(task, lambda files: 1.0)
    assert report.status == "defects"
    assert "empty-output" in {r.variant.family for r in report.false_positives}


def test_nondeterministic_verifier_is_flagged(tmp_path: Path) -> None:
    task = _task(tmp_path, INSTRUCTION, verifier_src="")
    seen = {"n": 0}

    def flaky(files: dict[str, bytes]) -> float:
        if ANSWER in files and files[ANSWER] == json.dumps(ORACLE, indent=2).encode():
            seen["n"] += 1
            return 1.0 if seen["n"] == 1 else 0.0
        return good_verifier(files)

    report, _ = _run(task, flaky)
    assert report.status == "flaky"
    assert any("oracle output re-scored" in i for i in report.issues())


def test_verifier_crash_on_an_equivalent_variant_is_reported(tmp_path: Path) -> None:
    task = _task(tmp_path, INSTRUCTION, verifier_src="")

    def crashes_on_reformat(files: dict[str, bytes]) -> float:
        # the oracle file is indented; the reformat variant is the compact one
        if ANSWER in files and files[ANSWER].strip() and b"\n" not in files[ANSWER]:
            raise ValueError("expected one key per line")
        return good_verifier(files)

    report, _ = _run(task, crashes_on_reformat)
    errs = [r for r in report.results if r.verdict == "verifier-error"]
    assert errs and errs[0].variant.family == "reformat"
    assert any("verifier error [reformat]" in i for i in report.issues())


def test_families_filter_limits_the_battery(tmp_path: Path) -> None:
    task = _task(tmp_path, INSTRUCTION, verifier_src="")
    report, _ = _run(task, good_verifier, families={"wrong-number"})
    assert {r.variant.family for r in report.results} == {"wrong-number"}


def test_report_round_trips_to_json(tmp_path: Path) -> None:
    task = _task(tmp_path, INSTRUCTION)
    report, _ = _run(task, literal_verifier)
    data = json.loads(report.to_json())
    assert data["status"] == "defects"
    assert data["answer_files"] == [ANSWER]
    assert any(r["verdict"] == "false-negative" for r in data["results"])


# ------------------------------------------------------- check_task + CLI


def test_check_task_equivalence_level_requires_a_sandbox(tmp_path: Path) -> None:
    from benchflow._utils.task_authoring import check_task

    task = _task(tmp_path, INSTRUCTION)
    (task / "solution" / "solve.sh").write_text("echo ok\n")
    issues = check_task(task, validation_level="equivalence")
    assert "equivalence validation requires --sandbox <backend>" in issues


def test_check_task_equivalence_level_returns_battery_issues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchflow._utils import task_authoring
    from benchflow.task import equivalence

    task = _task(tmp_path, INSTRUCTION)
    (task / "solution" / "solve.sh").write_text("echo ok\n")
    seen: dict = {}

    def fake_check(task_dir: Path, *, sandbox_type: str, **kw) -> EquivalenceReport:
        seen["args"] = (task_dir, sandbox_type, kw)
        return EquivalenceReport(
            task=task_dir.name,
            sandbox=sandbox_type,
            status="oracle-fails",
            reason="oracle output scores 0.0, not 1.0",
            oracle_reward=0.0,
        )

    monkeypatch.setattr(equivalence, "check_equivalence", fake_check)
    out = tmp_path / "eq.json"
    issues = task_authoring.check_task(
        task,
        sandbox_type="daytona",
        validation_level="equivalence",
        equivalence_report_output=out,
    )
    assert issues == ["equivalence: oracle output scores 0.0, not 1.0"]
    assert seen["args"][1] == "daytona"
    assert json.loads(out.read_text())["status"] == "oracle-fails"


def test_cli_lists_the_equivalence_level() -> None:
    from typer.testing import CliRunner

    from benchflow.cli.main import app

    result = CliRunner().invoke(app, ["tasks", "check", "--help"], terminal_width=200)
    assert result.exit_code == 0
    assert "equivalence" in result.output


def test_python_api_is_exported_from_benchflow_task() -> None:
    import benchflow.task as task_api

    for name in (
        "EquivalenceReport",
        "check_equivalence",
        "check_equivalence_async",
        "run_equivalence_battery",
    ):
        assert name in task_api.__all__
        assert getattr(task_api, name) is not None


def test_key_order_is_marked_fixed_when_the_instruction_orders_keys() -> None:
    """A task whose instruction says "The top-level keys, in order, must be ...", so the
    reversed-keys variant scoring 0 is a warning, not a false negative."""
    answers = {ANSWER: json.dumps(ORACLE).encode()}
    plain = build_variants(answers, instruction=INSTRUCTION)
    ordered = build_variants(
        answers,
        instruction=INSTRUCTION
        + "The top-level keys, in order, must be `results` and `methods`.\n",
    )
    assert not next(v for v in plain if v.family == "key-order").instruction_fixes_form
    assert next(v for v in ordered if v.family == "key-order").instruction_fixes_form


def test_any_top_level_list_of_objects_counts_as_the_row_list() -> None:
    """A task's rows can live under `scenarios`, a name outside the
    default row-list names; swapped-rows and row-order must still be built."""
    answer = {"scenarios": [{"id": "a", "effect": 1.5}, {"id": "b", "effect": 2.5}]}
    variants = build_variants({ANSWER: json.dumps(answer).encode()}, instruction="")
    fams = {v.family for v in variants}
    assert {"swapped-rows", "row-order"} <= fams
    swapped = next(v for v in variants if v.family == "swapped-rows")
    rows = json.loads(swapped.changes[ANSWER])["scenarios"]
    assert [r["effect"] for r in rows] == [2.5, 1.5]


def test_phrase_mandate_matches_whole_words_only() -> None:
    """A mandated phrase must match whole words: 'mean' is not mandated by a
    sentence that contains 'means'."""
    from benchflow.task.equivalence_variants import instruction_mandates_phrase

    assert not instruction_mandates_phrase("A pass means the values match.", "mean")
    assert instruction_mandates_phrase(
        "`methods` must use the exact word mean.", "mean"
    )


def test_precision_variant_moves_floats_by_one_ulp() -> None:
    """A relative 1e-9 move exceeded legitimate absolute tolerances on large
    values (100000000.0 -> +0.1); one ulp is pure float rounding noise."""
    import math

    answers = {ANSWER: json.dumps(ORACLE).encode()}
    variants = build_variants(answers, instruction=INSTRUCTION)
    prec = next(v for v in variants if v.family == "precision")
    value = json.loads(prec.changes[ANSWER])["results"][0]["value"]
    assert value == math.nextafter(12.5, math.inf)
    assert not prec.instruction_fixes_form


def test_exact_values_in_the_instruction_fix_precision() -> None:
    answers = {ANSWER: json.dumps(ORACLE).encode()}
    variants = build_variants(
        answers,
        instruction=INSTRUCTION
        + "Answers must give exact values in the order shown.\n",
    )
    assert next(v for v in variants if v.family == "precision").instruction_fixes_form


def test_reformat_keeps_the_trailing_newline() -> None:
    """A task that mandates "JSON
    ending in one newline": the compact reformat used to drop it and so changed two
    things at once. The newline is its own `whitespace` variant."""
    raw = (json.dumps(ORACLE, indent=2) + "\n").encode()
    variants = build_variants(
        {ANSWER: raw}, instruction="Write strict JSON ending in one newline."
    )
    reformat = next(v for v in variants if v.family == "reformat")
    assert reformat.changes[ANSWER].endswith(b"}\n")
    assert b"\n  " not in reformat.changes[ANSWER]
    ws = next(v for v in variants if v.family == "whitespace")
    assert ws.changes[ANSWER] == raw[:-1]
    assert ws.instruction_fixes_form
