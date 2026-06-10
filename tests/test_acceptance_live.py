"""Tests for acceptance-live spec parsing and decision helpers.

Execution-path coverage (the faked rollout layer) lives in
test_acceptance_live_execution.py; shared builders in acceptance_live_harness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import benchflow.task.acceptance_live as acceptance_live
from benchflow.task.acceptance_live import (
    LiveAcceptanceCase,
    LiveAcceptanceExpectation,
    LiveAcceptanceLeaderboard,
    LiveAcceptanceRunResult,
    LiveAcceptanceSpec,
    LiveAcceptanceWorkspace,
    parse_live_acceptance_spec,
    run_live_acceptance_checks,
)
from tests.acceptance_live_harness import (
    AL,
    CALIBRATION_REL,
    REPORT_REL,
    case_evidence,
    full_live_evidence,
    green_case,
    write_json,
    write_live_task,
)


@pytest.fixture
def live_task(tmp_path: Path) -> Path:
    return write_live_task(tmp_path / "live-task")


def _rewrite_calibration_report(task_dir: Path, cases: object) -> None:
    write_json(
        task_dir / CALIBRATION_REL, {"kind": "calibration-report", "cases": cases}
    )


PARSE_REJECTS = [
    pytest.param(
        lambda e, td: e.pop(AL),
        "acceptance-live validation requires "
        "benchflow.evidence.acceptance_live mapping",
        id="missing-acceptance-live",
    ),
    pytest.param(
        lambda e, td: e.__setitem__(AL, ["cases"]),
        "acceptance-live validation requires "
        "benchflow.evidence.acceptance_live mapping",
        id="acceptance-live-not-mapping",
    ),
    pytest.param(
        lambda e, td: e[AL].__setitem__("workspace", "current-worktree"),
        "acceptance-live workspace must be a mapping when declared",
        id="workspace-not-mapping",
    ),
    pytest.param(
        lambda e, td: e[AL].__setitem__("workspace", {"source": "git-archive"}),
        "acceptance-live workspace.source currently supports only 'current-worktree'",
        id="workspace-bad-source",
    ),
    pytest.param(
        lambda e, td: e[AL].__setitem__("workspace", {"target": 7}),
        "acceptance-live workspace.target must be a string",
        id="workspace-target-not-string",
    ),
    pytest.param(
        lambda e, td: e[AL].__setitem__("workspace", {"target": "rel/path"}),
        "acceptance-live workspace.target must be an absolute non-root sandbox path",
        id="workspace-target-relative",
    ),
    pytest.param(
        lambda e, td: e[AL].__setitem__("workspace", {"target": "/"}),
        "acceptance-live workspace.target must be an absolute non-root sandbox path",
        id="workspace-target-root",
    ),
    pytest.param(
        lambda e, td: e[AL].__setitem__("cases", "echo ok"),
        "acceptance-live cases must be a list when declared",
        id="cases-not-list",
    ),
    pytest.param(
        lambda e, td: e[AL].__setitem__("cases", []),
        "acceptance-live cases must be non-empty when declared",
        id="cases-empty",
    ),
    pytest.param(
        lambda e, td: (e[AL].pop("cases"), e[AL].pop("calibration")),
        "acceptance-live cases must be a non-empty list or generated calibration cases",
        id="no-cases-at-all",
    ),
    pytest.param(
        lambda e, td: e[AL]["cases"].__setitem__(0, 5),
        "acceptance-live cases[0] must be a mapping",
        id="case-not-mapping",
    ),
    pytest.param(
        lambda e, td: e[AL]["cases"][0].__setitem__("name", "  "),
        "acceptance-live cases[0].name must be a non-empty string",
        id="case-name-blank",
    ),
    pytest.param(
        lambda e, td: e[AL]["cases"][0].__setitem__("type", "fuzz"),
        "acceptance-live cases[0].type must be one of "
        "verifier, oracle, no-op, known-bad, partial, or reference",
        id="case-type-invalid",
    ),
    pytest.param(
        lambda e, td: e[AL]["cases"][0].__setitem__("command", " "),
        "acceptance-live cases[0].command must be a non-empty string when declared",
        id="case-command-blank",
    ),
    pytest.param(
        lambda e, td: e[AL]["cases"][0].__setitem__("command", "echo a\necho b"),
        "acceptance-live cases[0].command must be a single-line sandbox command",
        id="case-command-multiline",
    ),
    pytest.param(
        lambda e, td: e[AL]["cases"][1].__setitem__("command", "echo nope"),
        "acceptance-live cases[1].command is not supported for oracle cases; "
        "acceptance-live oracle uses the selected oracle/solve.sh",
        id="oracle-case-with-command",
    ),
    pytest.param(
        lambda e, td: e[AL]["cases"][0].__setitem__("reruns", 0),
        "acceptance-live cases[0].reruns must be an integer within 1..20",
        id="case-reruns-zero",
    ),
    pytest.param(
        lambda e, td: e[AL]["cases"][0].__setitem__("reruns", 21),
        "acceptance-live cases[0].reruns must be an integer within 1..20",
        id="case-reruns-over-max",
    ),
    pytest.param(
        lambda e, td: e[AL]["cases"][0].__setitem__("reruns", True),
        "acceptance-live cases[0].reruns must be an integer within 1..20",
        id="case-reruns-bool",
    ),
    pytest.param(
        lambda e, td: e[AL]["cases"][0].__setitem__("reruns", "2"),
        "acceptance-live cases[0].reruns must be an integer within 1..20",
        id="case-reruns-string",
    ),
    pytest.param(
        lambda e, td: e[AL]["cases"][0].pop("expect"),
        "acceptance-live cases[0].expect must be a mapping",
        id="expect-missing",
    ),
    pytest.param(
        lambda e, td: e[AL]["cases"][0].__setitem__("expect", {"flake_rate_max": 0.1}),
        "acceptance-live cases[0].expect must declare reward_min, reward_max, "
        "reward_range, or reward_equals",
        id="expect-no-bounds",
    ),
    pytest.param(
        lambda e, td: e[AL]["cases"][0].__setitem__("expect", {"reward_min": 1.5}),
        "acceptance-live cases[0].expect.reward_min must be numeric within 0..1",
        id="expect-reward-min-out-of-range",
    ),
    pytest.param(
        lambda e, td: e[AL]["cases"][0].__setitem__("expect", {"reward_max": True}),
        "acceptance-live cases[0].expect.reward_max must be numeric within 0..1",
        id="expect-reward-max-bool",
    ),
    pytest.param(
        lambda e, td: e[AL]["cases"][0].__setitem__("expect", {"reward_equals": "1"}),
        "acceptance-live cases[0].expect.reward_equals must be numeric within 0..1",
        id="expect-reward-equals-string",
    ),
    pytest.param(
        lambda e, td: e[AL]["cases"][0].__setitem__(
            "expect", {"reward_min": 0.5, "flake_rate_max": 2}
        ),
        "acceptance-live cases[0].expect.flake_rate_max must be numeric within 0..1",
        id="expect-flake-rate-out-of-range",
    ),
    pytest.param(
        lambda e, td: e[AL]["cases"][0].__setitem__("expect", {"reward_range": [0.1]}),
        "acceptance-live cases[0].expect.reward_range must be [min, max]",
        id="expect-range-wrong-shape",
    ),
    pytest.param(
        lambda e, td: e[AL]["cases"][0].__setitem__(
            "expect", {"reward_range": [0.8, 0.2]}
        ),
        "acceptance-live cases[0].expect.reward_range min must be <= max",
        id="expect-range-inverted",
    ),
    pytest.param(
        lambda e, td: e[AL]["cases"][0].__setitem__(
            "expect", {"reward_range": [0.2, 2]}
        ),
        "acceptance-live cases[0].expect.reward_range[1] must be numeric within 0..1",
        id="expect-range-element-out-of-range",
    ),
    pytest.param(
        lambda e, td: e[AL]["cases"].__setitem__(1, dict(e[AL]["cases"][0])),
        "acceptance-live case names must be unique; duplicate case 'greencase'",
        id="duplicate-case-names",
    ),
    pytest.param(
        lambda e, td: e[AL].__setitem__("report", 5),
        "acceptance-live report must be a non-empty relative path",
        id="report-not-string",
    ),
    pytest.param(
        lambda e, td: e[AL].__setitem__("report", "/abs/report.json"),
        "acceptance-live report must be a safe relative file path",
        id="report-absolute",
    ),
    pytest.param(
        lambda e, td: e[AL].__setitem__("report", "../escape.json"),
        "acceptance-live report must be a safe relative file path",
        id="report-traversal",
    ),
    pytest.param(
        lambda e, td: e[AL].__setitem__("leaderboard", "required"),
        "acceptance-live leaderboard must be a mapping",
        id="leaderboard-not-mapping",
    ),
    pytest.param(
        lambda e, td: e[AL].__setitem__("leaderboard", {"required": "yes"}),
        "acceptance-live leaderboard.required must be boolean",
        id="leaderboard-required-not-bool",
    ),
    pytest.param(
        lambda e, td: e[AL].__setitem__(
            "leaderboard", {"required": False, "max_flake_rate": 2}
        ),
        "acceptance-live leaderboard.max_flake_rate must be numeric within 0..1",
        id="leaderboard-flake-out-of-range",
    ),
    pytest.param(
        lambda e, td: e[AL].pop("report"),
        "acceptance-live leaderboard.required requires report",
        id="leaderboard-required-without-report",
    ),
    pytest.param(
        lambda e, td: e[AL].__setitem__("calibration", 5),
        "acceptance-live calibration must be a mapping when declared",
        id="calibration-not-mapping",
    ),
    pytest.param(
        lambda e, td: e[AL].__setitem__("calibration", {"from": "elsewhere"}),
        "acceptance-live calibration.from currently supports only calibration.report",
        id="calibration-from-unsupported",
    ),
    pytest.param(
        lambda e, td: e[AL]["calibration"].__setitem__("reruns", 0),
        "acceptance-live calibration.reruns must be an integer within 1..20",
        id="calibration-reruns-invalid",
    ),
    pytest.param(
        lambda e, td: e[AL]["calibration"].__setitem__("flake_rate_max", 2),
        "acceptance-live calibration.flake_rate_max must be numeric within 0..1",
        id="calibration-flake-invalid",
    ),
    pytest.param(
        lambda e, td: e.pop("calibration"),
        "acceptance-live calibration requires benchflow.evidence.calibration mapping",
        id="evidence-calibration-missing",
    ),
    pytest.param(
        lambda e, td: e["calibration"].pop("report"),
        "acceptance-live calibration.from report must be declared",
        id="calibration-report-undeclared",
    ),
    pytest.param(
        lambda e, td: e["calibration"].__setitem__("report", "/abs.json"),
        "acceptance-live calibration.from report must be a safe relative file path",
        id="calibration-report-unsafe-path",
    ),
    pytest.param(
        lambda e, td: e["calibration"].__setitem__("no_op_reward_max", "high"),
        "acceptance calibration.no_op_reward_max must be numeric within 0..1",
        id="calibration-no-op-threshold-invalid",
    ),
    pytest.param(
        lambda e, td: e["calibration"].__setitem__("known_bad_reward_max", 1.5),
        "acceptance calibration.known_bad_reward_max must be numeric within 0..1",
        id="calibration-known-bad-threshold-invalid",
    ),
    pytest.param(
        lambda e, td: e["calibration"].__setitem__(
            "partial_solution_range", [0.8, 0.2]
        ),
        "acceptance calibration.partial_solution_range must be [min, max] within 0..1",
        id="calibration-partial-range-invalid",
    ),
    pytest.param(
        lambda e, td: e["calibration"].__setitem__("partial_solution_range", "wide"),
        "acceptance calibration.partial_solution_range must be [min, max] within 0..1",
        id="calibration-partial-range-not-list",
    ),
    pytest.param(
        lambda e, td: (td / CALIBRATION_REL).unlink(),
        "acceptance-live calibration.from report is missing: "
        "evidence/calibration-report.json",
        id="calibration-report-file-missing",
    ),
    pytest.param(
        lambda e, td: (td / CALIBRATION_REL).write_text("[]\n"),
        "acceptance-live calibration.from report must be a JSON object",
        id="calibration-report-not-object",
    ),
    pytest.param(
        lambda e, td: _rewrite_calibration_report(td, []),
        "acceptance-live calibration.from report.cases must be a non-empty list",
        id="calibration-report-cases-empty",
    ),
    pytest.param(
        lambda e, td: _rewrite_calibration_report(td, [5]),
        "acceptance-live calibration.from report.cases[0] must be a mapping",
        id="generated-case-not-mapping",
    ),
    pytest.param(
        lambda e, td: _rewrite_calibration_report(
            td, [{"type": "no-op", "command": "true", "reward": 0.0}]
        ),
        "acceptance-live calibration.from report.cases[0].name must be a "
        "non-empty string",
        id="generated-case-name-missing",
    ),
    pytest.param(
        lambda e, td: _rewrite_calibration_report(
            td, [{"name": "x", "type": "oracle", "command": "true", "reward": 0.0}]
        ),
        "acceptance-live calibration.from report.cases[0].type must be "
        "no-op, known-bad, partial, or reference",
        id="generated-case-type-invalid",
    ),
    pytest.param(
        lambda e, td: _rewrite_calibration_report(
            td, [{"name": "x", "type": "no-op", "reward": 0.0}]
        ),
        "acceptance-live calibration.from report.cases[0].command must be "
        "declared for generated no-op live calibration cases",
        id="generated-case-command-missing",
    ),
    pytest.param(
        lambda e, td: _rewrite_calibration_report(
            td, [{"name": "x", "type": "no-op", "command": "a\nb", "reward": 0.0}]
        ),
        "acceptance-live calibration.from report.cases[0].command must be a "
        "single-line sandbox command",
        id="generated-case-command-multiline",
    ),
    pytest.param(
        lambda e, td: _rewrite_calibration_report(
            td, [{"name": "x", "type": "no-op", "command": " ", "reward": 0.0}]
        ),
        "acceptance-live calibration.from report.cases[0].command must be a "
        "non-empty string when declared",
        id="generated-case-command-blank",
    ),
    pytest.param(
        lambda e, td: _rewrite_calibration_report(
            td, [{"name": "x", "type": "no-op", "command": "true", "reward": 1.5}]
        ),
        "acceptance-live calibration.from report.cases[0].reward must be "
        "numeric within 0..1",
        id="generated-case-reward-out-of-range",
    ),
    pytest.param(
        lambda e, td: e["calibration"].__setitem__("no_op_reward_max", "broken"),
        "acceptance-live calibration.from report.cases[0] could not derive "
        "live reward expectations",
        id="generated-case-cannot-derive-expectation",
    ),
    pytest.param(
        lambda e, td: _rewrite_calibration_report(
            td, [{"name": "x", "type": "reference"}]
        ),
        "acceptance-live calibration.from report.cases[0] could not derive "
        "live reward expectations",
        id="generated-reference-without-reward-cannot-derive",
    ),
]


class TestParseLiveAcceptanceSpec:
    @pytest.mark.parametrize(("mutate", "expected"), PARSE_REJECTS)
    def test_rejects(self, live_task: Path, mutate, expected: str) -> None:
        evidence = full_live_evidence()
        mutate(evidence, live_task)
        spec, issues = parse_live_acceptance_spec(live_task, evidence=evidence)
        assert spec is None
        assert expected in issues

    def test_calibration_report_invalid_json(self, live_task: Path) -> None:
        (live_task / CALIBRATION_REL).write_text("{not json")
        spec, issues = parse_live_acceptance_spec(
            live_task, evidence=full_live_evidence()
        )
        assert spec is None
        assert any(
            i.startswith("acceptance-live calibration.from report is not valid JSON:")
            for i in issues
        )

    def test_full_spec_parses(self, live_task: Path) -> None:
        evidence = full_live_evidence(calibration_flake_rate_max=0.25)
        spec, issues = parse_live_acceptance_spec(live_task, evidence=evidence)
        assert issues == []
        assert spec is not None
        assert spec.workspace == LiveAcceptanceWorkspace(
            source="current-worktree", target="/workspace"
        )
        assert [case.name for case in spec.cases] == [
            "greencase",
            "oracleproof",
            "calnoop",
            "calknownbad",
            "calpartial",
            "calreference",
        ]
        assert spec.cases[0] == LiveAcceptanceCase(
            name="greencase",
            case_type="verifier",
            command="echo ok",
            reruns=2,
            expect=LiveAcceptanceExpectation(reward_min=0.5),
            source="declared",
        )
        assert spec.cases[1].command is None
        assert spec.cases[1].reruns == 1
        generated = {case.name: case for case in spec.cases[2:]}
        assert generated["calnoop"].expect == LiveAcceptanceExpectation(
            reward_max=0.1, flake_rate_max=0.25
        )
        assert generated["calknownbad"].expect == LiveAcceptanceExpectation(
            reward_max=0.5, flake_rate_max=0.25
        )
        assert generated["calpartial"].expect == LiveAcceptanceExpectation(
            reward_range=(0.2, 0.8), flake_rate_max=0.25
        )
        assert generated["calreference"].expect == LiveAcceptanceExpectation(
            reward_equals=1.0, flake_rate_max=0.25
        )
        assert generated["calreference"].command is None
        assert generated["calnoop"].command == "true"
        assert all(
            case.source == "calibration-report" and case.reruns == 3
            for case in spec.cases[2:]
        )
        assert spec.leaderboard == LiveAcceptanceLeaderboard(
            required=True, max_flake_rate=0.0
        )
        assert spec.report_path == Path(REPORT_REL)

    def test_case_name_and_command_are_stripped(self, live_task: Path) -> None:
        evidence = case_evidence(
            green_case(name="  padded  ", command="  echo hi  "), report=None
        )
        spec, issues = parse_live_acceptance_spec(live_task, evidence=evidence)
        assert issues == []
        assert spec is not None
        assert spec.cases[0].name == "padded"
        assert spec.cases[0].command == "echo hi"

    def test_workspace_defaults_to_task_workdir(self, tmp_path: Path) -> None:
        task = write_live_task(tmp_path / "live-task", workdir="/srv/code")
        evidence = case_evidence(green_case(), report=None)
        spec, issues = parse_live_acceptance_spec(task, evidence=evidence)
        assert issues == []
        assert spec is not None
        assert spec.workspace.target == "/srv/code"

    def test_workspace_falls_back_to_app_without_workdir(self, tmp_path: Path) -> None:
        task = write_live_task(tmp_path / "live-task", workdir=None)
        evidence = case_evidence(green_case(), report=None)
        spec, _issues = parse_live_acceptance_spec(task, evidence=evidence)
        assert spec is not None
        assert spec.workspace.target == "/app"

    def test_workspace_mapping_without_target_uses_workdir(
        self, tmp_path: Path
    ) -> None:
        task = write_live_task(tmp_path / "live-task", workdir="/srv/code")
        evidence = case_evidence(
            green_case(), report=None, workspace={"source": "current-worktree"}
        )
        spec, _issues = parse_live_acceptance_spec(task, evidence=evidence)
        assert spec is not None
        assert spec.workspace.target == "/srv/code"

    def test_oracle_case_requires_solve_script(self, tmp_path: Path) -> None:
        task = write_live_task(tmp_path / "live-task")
        (task / "oracle" / "solve.sh").unlink()
        spec, issues = parse_live_acceptance_spec(task, evidence=full_live_evidence())
        assert spec is None
        solve = task.resolve() / "oracle" / "solve.sh"
        assert (
            "acceptance-live cases[1].type=oracle requires executable "
            f"oracle/solve.sh or legacy solution/solve.sh: {solve}"
        ) in issues

    def test_oracle_case_requires_executable_solve_script(self, tmp_path: Path) -> None:
        task = write_live_task(tmp_path / "live-task")
        (task / "oracle" / "solve.sh").chmod(0o644)
        spec, issues = parse_live_acceptance_spec(task, evidence=full_live_evidence())
        assert spec is None
        solve = task.resolve() / "oracle" / "solve.sh"
        assert (
            f"acceptance-live cases[1].type=oracle requires executable file: {solve}"
        ) in issues

    def test_report_output_overrides_declared_report(
        self, live_task: Path, tmp_path: Path
    ) -> None:
        override = tmp_path / "out" / "live-report.json"
        spec, issues = parse_live_acceptance_spec(
            live_task, evidence=full_live_evidence(), report_output=override
        )
        assert issues == []
        assert spec is not None
        assert spec.report_path == override

    def test_report_output_relative_resolves_against_cwd(
        self, live_task: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        spec, issues = parse_live_acceptance_spec(
            live_task,
            evidence=full_live_evidence(),
            report_output=Path("rel-report.json"),
        )
        assert issues == []
        assert spec is not None
        assert spec.report_path == (tmp_path / "rel-report.json").resolve()

    def test_report_output_rejects_directory(
        self, live_task: Path, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "outdir"
        out_dir.mkdir()
        spec, issues = parse_live_acceptance_spec(
            live_task, evidence=full_live_evidence(), report_output=out_dir
        )
        assert spec is None
        assert (
            "acceptance-live report output must be a file path, not a directory"
            in issues
        )

    def test_report_output_rejects_nameless_path(self, live_task: Path) -> None:
        spec, issues = parse_live_acceptance_spec(
            live_task, evidence=full_live_evidence(), report_output=Path("/")
        )
        assert spec is None
        assert "acceptance-live report output must be a file path" in issues

    def test_report_output_expanduser_failure(self, live_task: Path) -> None:
        spec, issues = parse_live_acceptance_spec(
            live_task,
            evidence=full_live_evidence(),
            report_output=Path("~benchflow_no_such_user/report.json"),
        )
        assert spec is None
        assert any(
            i.startswith("acceptance-live report output cannot expand user:")
            for i in issues
        )


async def test_wrapper_fails_closed_inside_running_event_loop(
    live_task: Path,
) -> None:
    issues = run_live_acceptance_checks(
        live_task,
        sandbox_type="docker",
        evidence=case_evidence(green_case(), report=None),
    )
    assert issues == [
        "acceptance-live validation cannot run inside an active event loop; "
        "call the async live acceptance runner instead"
    ]


class TestHelpers:
    @pytest.mark.parametrize(
        ("rewards", "expected"),
        [
            ({"reward": 0.5}, 0.5),
            ({"reward": 1}, 1.0),
            ({"reward": True}, None),
            ({"reward": 1.5}, None),
            ({"reward": -0.1}, None),
            ({"reward": "1"}, None),
            ({}, None),
            (None, None),
            ("reward", None),
        ],
    )
    def test_scalar_reward(self, rewards: Any, expected: float | None) -> None:
        assert acceptance_live._scalar_reward(rewards) == expected

    def test_coerce_run_result(self) -> None:
        result = LiveAcceptanceRunResult(reward=1.0, error=None)
        assert acceptance_live._coerce_run_result(result) is result
        assert acceptance_live._coerce_run_result((0.5, None)) == (
            LiveAcceptanceRunResult(reward=0.5, error=None)
        )
        with pytest.raises(TypeError, match="must be LiveAcceptanceRunResult"):
            acceptance_live._coerce_run_result(3)

    @pytest.mark.parametrize(
        ("trajectory", "expected"),
        [
            ([], None),
            ([{"type": "agent", "return_code": 3}], None),
            ([{"type": "oracle", "return_code": True}], None),
            ([{"type": "oracle"}, {"type": "oracle", "return_code": 4}], 4),
            ([{"type": "oracle", "return_code": 0}], 0),
        ],
    )
    def test_oracle_return_code(
        self, trajectory: list[dict], expected: int | None
    ) -> None:
        assert acceptance_live._oracle_return_code(trajectory) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("/app", True),
            ("/", False),
            ("", False),
            ("  ", False),
            ("relative/path", False),
            (7, False),
        ],
    )
    def test_is_safe_sandbox_dir(self, value: Any, expected: bool) -> None:
        assert acceptance_live._is_safe_sandbox_dir(value) is expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("a/b.json", Path("a/b.json")),
            ("/abs.json", None),
            (".", None),
            ("../escape.json", None),
            ("a/../b.json", None),
            ("a/./b.json", Path("a/b.json")),
        ],
    )
    def test_safe_relative_file_path(self, value: str, expected: Path | None) -> None:
        assert acceptance_live._safe_relative_file_path(value) == expected

    async def test_upload_workspace_fails_closed(self, tmp_path: Path) -> None:
        workspace = LiveAcceptanceWorkspace(source="current-worktree", target="/app")
        with pytest.raises(
            RuntimeError, match="acceptance-live current-worktree was not staged"
        ):
            await acceptance_live._upload_workspace(
                object(), workspace=workspace, staged_worktree=None
            )
        unknown = LiveAcceptanceWorkspace(source="weird", target="/app")  # type: ignore[arg-type]
        with pytest.raises(
            RuntimeError,
            match="unsupported acceptance-live workspace source: weird",
        ):
            await acceptance_live._upload_workspace(
                object(), workspace=unknown, staged_worktree=tmp_path
            )

    def test_leaderboard_suitability_with_no_records(self) -> None:
        spec = LiveAcceptanceSpec(
            workspace=LiveAcceptanceWorkspace(source="current-worktree", target="/app"),
            cases=(),
            report_path=None,
            leaderboard=LiveAcceptanceLeaderboard(required=True, max_flake_rate=0.0),
        )
        suitability = acceptance_live._leaderboard_suitability(spec=spec, records=[])
        assert suitability["status"] == "insufficient"
        assert suitability["checks"]["has_live_runs"] is False
        assert suitability["checks"]["flake_rate_within_limit"] is True
        assert suitability["issues"] == [
            "requires at least one live run",
            "requires all live runs to pass",
            "requires a passed oracle live case",
            "requires a passed reference live case",
            "missing generated calibration live case types: "
            "known-bad, no-op, partial, reference",
        ]

    def test_case_flake_expectation_without_threshold_is_noop(self) -> None:
        case = LiveAcceptanceCase(
            name="x",
            case_type="verifier",
            command=None,
            reruns=1,
            expect=LiveAcceptanceExpectation(reward_min=0.5),
        )
        records = [{"case": "x", "status": "failed"}]
        assert (
            acceptance_live._check_case_flake_expectation(case=case, records=records)
            == []
        )
