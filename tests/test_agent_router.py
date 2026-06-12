"""Unit tests for the ``bench agent`` adoption router (create / run / verify)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from typer.testing import CliRunner

from benchflow.agent_router import (
    DEFAULT_REWARD_TOLERANCE,
    AdoptionSkill,
    BenchmarkExistsError,
    BenchmarkNotFound,
    CodexLaunchError,
    InvalidBenchmarkName,
    ParityExperimentMissing,
    assemble_adoption_context,
    build_codex_launch_command,
    build_scaffold_files,
    build_verify_report,
    collect_adoption_skills,
    create_benchmark,
    derive_name_from_source,
    extract_criterion_comparisons,
    extract_reward_samples,
    has_codex_credentials,
    load_parity_experiment,
    prepare_adoption_launch,
    render_divergence_issue,
    roundtrip_conformance_status,
    run_agent_adoption,
    validate_benchmark_name,
)
from benchflow.cli.main import app

# ── name validation ───────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["programbench", "my-bench", "a1", "x-y-z2"])
def test_validate_name_accepts_valid_slugs(name: str) -> None:
    assert validate_benchmark_name(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "",
        "Programbench",  # uppercase
        "my_bench",  # underscore
        "1bench",  # leading digit
        "-bench",  # leading hyphen
        "bench-",  # trailing hyphen
        "my--bench",  # consecutive hyphens
        "my bench",  # whitespace
        "../escape",  # path traversal
        "a/b",  # path separator
        "x" * 65,  # too long
        "good\n",  # trailing newline — `$` matches before it; `fullmatch` rejects
        "good\nbad",  # embedded newline
        "good\r\n",  # CRLF tail
    ],
)
def test_validate_name_rejects_invalid_slugs(name: str) -> None:
    with pytest.raises(InvalidBenchmarkName):
        validate_benchmark_name(name)


def test_validate_name_rejects_trailing_newline_but_accepts_stripped() -> None:
    """A trailing newline must not slip past the slug check.

    Mutation guard: ``re`` ``$`` matches just before a final ``\\n``, so a
    ``match`` (or a pattern anchored with ``$``) would accept ``"good\\n"``.
    Pinning both the rejection *and* the accepted stripped form kills a revert
    to ``match``/``$`` — under that mutation ``"good\\n"`` would validate.
    """
    assert validate_benchmark_name("good") == "good"
    with pytest.raises(InvalidBenchmarkName):
        validate_benchmark_name("good\n")


def test_derive_name_from_source_strips_git_and_slugifies() -> None:
    assert derive_name_from_source("https://github.com/foo/My_Bench.git") == "my-bench"
    assert derive_name_from_source("/local/path/cool-bench/") == "cool-bench"


# ── create: scaffold ──────────────────────────────────────────────────

_EXPECTED_FILES = {
    "__init__.py",
    "benchflow.py",
    "main.py",
    "parity_test.py",
    "run_my_bench.py",
    "my-bench.yaml",
    "benchmark.yaml",
    "parity_experiment.json",
    "README.md",
}


def test_scaffold_emits_full_reference_file_set() -> None:
    files = build_scaffold_files("my-bench")
    assert set(files) == _EXPECTED_FILES


def test_scaffold_converter_has_documented_convert_entrypoint() -> None:
    converter = build_scaffold_files("my-bench")["benchflow.py"]
    assert "def convert(" in converter
    assert "def convert_all(" in converter
    assert "CONVERT.md" in converter
    # token substitution happened — no raw placeholder survives.
    assert "{{NAME}}" not in converter
    assert "my-bench" in converter


def test_scaffold_parity_test_lists_three_modes() -> None:
    parity = build_scaffold_files("my-bench")["parity_test.py"]
    for mode_fn in ("structural_parity", "eval_parity", "side_by_side_parity"):
        assert mode_fn in parity


def test_scaffold_benchmark_yaml_and_parity_json_are_well_formed() -> None:
    files = build_scaffold_files("my-bench")
    assert "name: my-bench" in files["benchmark.yaml"]
    parity = json.loads(files["parity_experiment.json"])
    assert parity["benchmark"] == "my-bench"
    assert parity["status"] == "template"
    assert parity["conversion_parity"]["tasks"] == []
    assert parity["reward_distribution_parity"]["samples"] == []


def test_create_benchmark_writes_files_to_disk(tmp_path: Path) -> None:
    target, written = create_benchmark("my-bench", tmp_path)
    assert target == tmp_path / "my-bench"
    assert written == sorted(_EXPECTED_FILES)
    for rel in _EXPECTED_FILES:
        assert (target / rel).exists()


def test_create_benchmark_refuses_existing_directory(tmp_path: Path) -> None:
    create_benchmark("my-bench", tmp_path)
    sentinel = tmp_path / "my-bench" / "README.md"
    original = sentinel.read_text()
    with pytest.raises(BenchmarkExistsError):
        create_benchmark("my-bench", tmp_path)
    # refusal is fail-closed: existing content is untouched.
    assert sentinel.read_text() == original


def test_create_benchmark_rejects_bad_name_before_touching_disk(tmp_path: Path) -> None:
    with pytest.raises(InvalidBenchmarkName):
        create_benchmark("../escape", tmp_path)
    assert list(tmp_path.iterdir()) == []


# ── run: context assembly + launch command ────────────────────────────


def test_assemble_context_includes_source_guide_skills_and_target() -> None:
    skills = [AdoptionSkill("conversion-guide", "benchmarks/CONVERT.md")]
    prompt = assemble_adoption_context(
        "github.com/foo/bar",
        "my-bench",
        convert_guide="GUIDE-BODY-SENTINEL",
        skills=skills,
    )
    assert "github.com/foo/bar" in prompt
    assert "benchmarks/my-bench/" in prompt
    assert "CONVERT.md" in prompt
    assert "Adoption skills" in prompt
    assert "conversion-guide" in prompt
    assert "GUIDE-BODY-SENTINEL" in prompt


def test_build_launch_command_structure() -> None:
    cmd = build_codex_launch_command(
        "PROMPT-SENTINEL", workdir="/repo", codex_bin="codex", model="gpt-x"
    )
    assert cmd[0] == "codex"
    assert cmd[1] == "exec"
    assert "--cd" in cmd
    assert cmd[cmd.index("--cd") + 1] == "/repo"
    assert "--skip-git-repo-check" in cmd
    assert cmd[cmd.index("--sandbox") + 1] == "workspace-write"
    assert cmd[cmd.index("--model") + 1] == "gpt-x"
    # the assembled prompt is the final positional argument.
    assert cmd[-1] == "PROMPT-SENTINEL"


def test_build_launch_command_omits_model_when_absent() -> None:
    cmd = build_codex_launch_command("P", workdir="/repo")
    assert "--model" not in cmd


def test_collect_adoption_skills_references_convert_guide(tmp_path: Path) -> None:
    skills = collect_adoption_skills()
    refs = {s.reference for s in skills}
    assert any("CONVERT.md" in r for r in refs)


def test_prepare_launch_assembles_command_and_prompt(tmp_path: Path) -> None:
    launch = prepare_adoption_launch(
        "github.com/foo/bar",
        "my-bench",
        repo_root=tmp_path,
        convert_guide="GUIDE",
        codex_bin="codex",
    )
    assert launch.cwd == str(tmp_path)
    assert launch.command[-1] == launch.prompt
    assert "benchmarks/my-bench/" in launch.prompt


# ── run: credentials + fake exec ──────────────────────────────────────


def test_has_codex_credentials_branches(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    assert has_codex_credentials({"OPENAI_API_KEY": "k"}, auth) is True
    assert has_codex_credentials({"CODEX_API_KEY": "k"}, auth) is True
    assert has_codex_credentials({}, auth) is False
    auth.write_text("{}")
    assert has_codex_credentials({}, auth) is True


def test_run_adoption_fails_closed_without_credentials(tmp_path: Path) -> None:
    calls: list = []

    def fake_exec(command, *, cwd, env):
        calls.append(command)
        return 0

    with pytest.raises(CodexLaunchError) as exc:
        run_agent_adoption(
            "github.com/foo/bar",
            "my-bench",
            repo_root=tmp_path,
            exec_fn=fake_exec,
            env={},
            auth_file=tmp_path / "missing-auth.json",
        )
    assert "OPENAI_API_KEY" in str(exc.value)
    # fail-closed: the exec layer is never reached.
    assert calls == []


def test_run_adoption_launches_via_fake_exec_with_credentials(tmp_path: Path) -> None:
    guide = tmp_path / "benchmarks" / "CONVERT.md"
    guide.parent.mkdir(parents=True)
    guide.write_text("# Benchmark Conversion Guide\nCONVERT.md body\n")
    recorded: dict = {}

    def fake_exec(command, *, cwd, env):
        recorded["command"] = command
        recorded["cwd"] = cwd
        return 7

    code = run_agent_adoption(
        "github.com/foo/bar",
        "my-bench",
        repo_root=tmp_path,
        exec_fn=fake_exec,
        env={"OPENAI_API_KEY": "k"},
        auth_file=tmp_path / "missing-auth.json",
    )
    assert code == 7
    assert recorded["cwd"] == str(tmp_path)
    assert recorded["command"][0] == "codex"
    # the launched prompt carries the conversion guide + source.
    assert "github.com/foo/bar" in recorded["command"][-1]
    assert "CONVERT.md" in recorded["command"][-1]


# ── verify: parity extraction ─────────────────────────────────────────


def _criteria_doc(*pairs: tuple[str, str]) -> dict:
    return {
        "conversion_parity": {
            "tasks": [
                {
                    "task_id": "t1",
                    "criteria_results": [
                        {
                            "criterion_id": f"C-{i}",
                            "original_verdict": orig,
                            "adapted_verdict": adapted,
                        }
                        for i, (orig, adapted) in enumerate(pairs)
                    ],
                }
            ]
        }
    }


def test_extract_criterion_comparisons_computes_agreement() -> None:
    comps = extract_criterion_comparisons(
        _criteria_doc(("pass", "pass"), ("pass", "fail"))
    )
    assert len(comps) == 2
    assert comps[0].agreement is True
    assert comps[1].agreement is False


def test_extract_reward_samples_from_agent_parity_shape() -> None:
    doc = {
        "agent_parity": {
            "results": [
                {
                    "task_id": "t1",
                    "programbench": {"reward": 0.50},
                    "benchflow": {"reward": 0.50},
                },
                {
                    "task_id": "t2",
                    "programbench": {"reward": 0.00},
                    "benchflow": {"reward": 0.10},
                },
            ]
        }
    }
    samples = extract_reward_samples(doc)
    deltas = sorted(round(s.delta, 4) for s in samples)
    assert deltas == [0.0, 0.1]


def test_one_sided_legacy_reward_fails_closed() -> None:
    # legacy recorded, converted missing, no explicit reward_delta: the sample
    # is unmeasured and must never confirm parity (the half-recorded bug).
    doc = {"agent_parity": {"results": [{"task_id": "t1", "legacy_reward": 1.0}]}}
    report = build_verify_report("my-bench", doc)
    assert report.verdict != "parity-confirmed"
    assert report.reward is not None
    assert [s.task_id for s in report.reward.exceeding] == ["t1"]
    assert report.reward.samples[0].delta == float("inf")


def test_one_sided_converted_reward_fails_closed() -> None:
    doc = {"agent_parity": {"results": [{"task_id": "t2", "converted_reward": 0.5}]}}
    report = build_verify_report("my-bench", doc)
    assert report.verdict == "parity-divergent"
    assert [s.task_id for s in report.reward.exceeding] == ["t2"]


def test_mixed_full_and_one_sided_reward_is_divergent() -> None:
    doc = {
        "agent_parity": {
            "results": [
                {"task_id": "ok", "legacy_reward": 1.0, "converted_reward": 1.0},
                {"task_id": "half", "legacy_reward": 1.0},
            ]
        }
    }
    report = build_verify_report("my-bench", doc)
    assert report.verdict == "parity-divergent"
    assert [s.task_id for s in report.reward.exceeding] == ["half"]


def test_explicit_reward_delta_override_is_honored_with_one_side() -> None:
    # An author-supplied reward_delta is the recorded measurement and wins even
    # when only one raw reward is present.
    doc = {
        "agent_parity": {
            "results": [
                {"task_id": "t1", "legacy_reward": 1.0, "reward_delta": 0.0},
            ]
        }
    }
    report = build_verify_report("my-bench", doc)
    assert report.verdict == "parity-confirmed"
    assert report.reward.samples[0].delta == 0.0


def test_non_numeric_reward_fields_skipped_not_crash() -> None:
    # All-non-numeric reward fields yield no sample (no float() crash, no
    # phantom zero-delta confirmation).
    doc = {
        "agent_parity": {
            "results": [
                {"task_id": "t1", "legacy_reward": "x", "converted_reward": "y"},
            ]
        }
    }
    report = build_verify_report("my-bench", doc)
    assert report.reward is None
    assert report.verdict == "insufficient-evidence"


# ── verify: verdicts + issue draft ────────────────────────────────────


def test_verify_pass_when_all_criteria_agree() -> None:
    report = build_verify_report(
        "my-bench", _criteria_doc(("pass", "pass"), ("fail", "fail"))
    )
    assert report.verdict == "parity-confirmed"
    assert report.passed is True


def test_verify_fails_on_criterion_disagreement() -> None:
    report = build_verify_report(
        "my-bench", _criteria_doc(("pass", "pass"), ("pass", "fail"))
    )
    assert report.verdict == "parity-divergent"
    assert report.passed is False


def test_verify_pass_via_reward_layer_within_tolerance() -> None:
    doc = {
        "reward_distribution_parity": {
            "samples": [
                {"task_id": "t1", "legacy_reward": 0.50, "converted_reward": 0.51},
            ]
        }
    }
    report = build_verify_report("my-bench", doc, tolerance=0.02)
    assert report.verdict == "parity-confirmed"


def test_verify_fails_when_reward_delta_exceeds_tolerance() -> None:
    doc = {
        "reward_distribution_parity": {
            "samples": [
                {"task_id": "t1", "legacy_reward": 0.50, "converted_reward": 0.90},
            ]
        }
    }
    report = build_verify_report("my-bench", doc, tolerance=0.02)
    assert report.verdict == "parity-divergent"
    assert report.reward is not None
    assert report.reward.exceeding[0].task_id == "t1"


def test_verify_insufficient_evidence_on_empty_data() -> None:
    report = build_verify_report("my-bench", {})
    assert report.verdict == "insufficient-evidence"
    assert report.passed is False


def test_scaffolded_parity_file_is_insufficient_evidence() -> None:
    data = json.loads(build_scaffold_files("my-bench")["parity_experiment.json"])
    report = build_verify_report("my-bench", data)
    assert report.verdict == "insufficient-evidence"


# An adopted benchmark (harvey-lab) ships a parity_experiment.json whose top
# level is a JSON *array* of experiment-summary records carrying ``metrics``
# with original-vs-converted values. The parsers must read those metrics — and
# never call ``.get`` on the array unguarded (the old AttributeError crash).
_TOP_LEVEL_ARRAY_PARITY = [
    {
        "benchmark": "demo-bench",
        "experiment": "end-to-end",
        "metrics": [
            {"name": "mean_pass_rate", "original": "23.0", "converted": "22.2"}
        ],
    },
    {
        "benchmark": "demo-bench",
        "experiment": "prompt-level",
        "metrics": [{"name": "agreement", "original": "100%", "converted": "100%"}],
    },
]

# A top-level array carrying no comparable metric values at all — the parsers
# must tolerate it (no crash) and yield no comparisons, so the gate reports
# insufficient-evidence rather than a false confirmation.
_TOP_LEVEL_ARRAY_NO_METRICS = [
    {"benchmark": "demo-bench", "experiment": "end-to-end", "notes": "n/a"},
    {"benchmark": "demo-bench", "metrics": [{"name": "noisy", "original": "1 ± 1"}]},
]


def test_extract_parsers_read_top_level_array_metrics() -> None:
    comps = extract_criterion_comparisons(_TOP_LEVEL_ARRAY_PARITY)
    # Exact-comparison metrics on both records become comparisons; the matching
    # 100%/100% agrees, the 23.0/22.2 mismatch disagrees.
    by_id = {c.criterion_id: c.agreement for c in comps}
    assert by_id == {"mean_pass_rate": False, "agreement": True}
    # The reward parser must not fabricate phantom zero-delta samples from the
    # array's summary dicts (that would falsely confirm).
    assert extract_reward_samples(_TOP_LEVEL_ARRAY_PARITY) == []


def test_extract_parsers_tolerate_top_level_array_without_metrics() -> None:
    # Non-comparable / distributional-only array: no comparisons, no crash.
    assert extract_criterion_comparisons(_TOP_LEVEL_ARRAY_NO_METRICS) == []
    assert extract_reward_samples(_TOP_LEVEL_ARRAY_NO_METRICS) == []


def test_verify_top_level_array_metrics_flip_divergent() -> None:
    report = build_verify_report("demo-bench", _TOP_LEVEL_ARRAY_PARITY)
    # One metric agrees, one diverges -> the deterministic floor is not all-agree.
    assert report.verdict == "parity-divergent"
    assert report.passed is False


def test_verify_top_level_array_without_metrics_is_insufficient_evidence() -> None:
    report = build_verify_report("demo-bench", _TOP_LEVEL_ARRAY_NO_METRICS)
    assert report.verdict == "insufficient-evidence"
    assert report.passed is False


def test_cli_verify_top_level_array_parity_file_does_not_crash(tmp_path: Path) -> None:
    create_benchmark("demo-bench", tmp_path)
    parity = tmp_path / "demo-bench" / "parity_experiment.json"
    parity.write_text(json.dumps(_TOP_LEVEL_ARRAY_NO_METRICS))
    result = CliRunner().invoke(
        app, ["agent", "verify", "demo-bench", "--benchmarks-dir", str(tmp_path)]
    )
    # The documented support-path exit (1), not an uncaught AttributeError.
    assert result.exit_code == 1, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    out = click.unstyle(result.output)
    assert "insufficient-evidence" in out
    assert "AttributeError" not in out


# ── in-repo benchmarks: every shipped parity file produces a real verdict ──

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SHIPPED_PARITY_FILES = sorted(
    (_REPO_ROOT / "benchmarks").glob("*/parity_experiment.json")
)


def test_shipped_parity_files_are_discovered() -> None:
    # Guard against the glob silently matching nothing (which would make the
    # parametrized test below vacuously pass).
    assert _SHIPPED_PARITY_FILES, "no benchmarks/*/parity_experiment.json found"


@pytest.mark.parametrize(
    "parity_file", _SHIPPED_PARITY_FILES, ids=lambda p: p.parent.name
)
def test_shipped_parity_file_yields_real_verdict(parity_file: Path) -> None:
    """Each in-repo benchmark's recorded parity must parse to a real verdict.

    A newly-adopted benchmark shipping a schema the verify gate can't read would
    report ``insufficient-evidence`` here — telling its maintainer their
    genuinely-validated benchmark has 'no recorded parity comparisons'. Globbing
    (not a hardcoded list) makes a stale-schema benchmark fail CI on add.
    """
    data = json.loads(parity_file.read_text())
    report = build_verify_report(parity_file.parent.name, data)
    assert report.verdict != "insufficient-evidence", (
        f"{parity_file.parent.name} parity_experiment.json parsed to no "
        f"comparisons or samples (schema unsupported by the verify gate)"
    )


def test_divergence_issue_names_failing_criterion_and_is_unfiled() -> None:
    report = build_verify_report("my-bench", _criteria_doc(("pass", "fail")))
    issue = render_divergence_issue(report)
    assert "my-bench" in issue
    assert "parity-divergent" in issue
    assert "original=pass converted=fail" in issue
    assert "NOT been filed" in issue


def test_divergence_issue_lists_reward_breaches() -> None:
    doc = {
        "reward_distribution_parity": {
            "samples": [
                {"task_id": "tx", "legacy_reward": 0.0, "converted_reward": 0.9},
            ]
        }
    }
    report = build_verify_report("my-bench", doc, tolerance=0.02)
    issue = render_divergence_issue(report)
    assert "tx" in issue
    assert "delta" in issue


# ── verify: loaders + harness wiring ──────────────────────────────────


def test_load_parity_experiment_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkNotFound):
        load_parity_experiment(tmp_path, "my-bench")
    create_benchmark("my-bench", tmp_path)
    # scaffolded benchmark ships a parity_experiment.json → loads cleanly.
    data = load_parity_experiment(tmp_path, "my-bench")
    assert data["benchmark"] == "my-bench"
    (tmp_path / "my-bench" / "parity_experiment.json").unlink()
    with pytest.raises(ParityExperimentMissing):
        load_parity_experiment(tmp_path, "my-bench")


def test_load_parity_experiment_malformed_json_is_domain_error(tmp_path: Path) -> None:
    create_benchmark("my-bench", tmp_path)
    (tmp_path / "my-bench" / "parity_experiment.json").write_text("{ bad json")
    # Malformed JSON must surface as the domain error (routed through the CLI's
    # existing fail-closed arm), never a raw json.JSONDecodeError.
    with pytest.raises(ParityExperimentMissing) as excinfo:
        load_parity_experiment(tmp_path, "my-bench")
    assert "not valid JSON" in str(excinfo.value)
    assert not isinstance(excinfo.value, json.JSONDecodeError)


def test_roundtrip_conformance_status_maps_report() -> None:
    fake_report = SimpleNamespace(
        status="drift",
        mismatches=[SimpleNamespace(reason="tests/ differ")],
    )
    status, reasons = roundtrip_conformance_status(
        Path("/nope"), report_fn=lambda _td: fake_report
    )
    assert status == "drift"
    assert reasons == ["tests/ differ"]


def test_roundtrip_conformance_status_real_import_binding() -> None:
    """Exercise the real report_fn=None branch so the import wiring can't rot.

    Every other test injects a fake report_fn; this one runs the actual
    benchflow.task.build_harbor_roundtrip_conformance_report against a real
    example task, locking the import path and the (status, reasons) shape.
    """
    task_dir = Path(__file__).parent / "examples" / "hello-world-task"
    if not task_dir.exists():
        import pytest

        pytest.skip("hello-world-task fixture not present")
    status, reasons = roundtrip_conformance_status(task_dir)
    assert isinstance(status, str) and status
    assert isinstance(reasons, list)


# ── CLI surface ───────────────────────────────────────────────────────


def test_cli_create_then_verify_roundtrip(tmp_path: Path) -> None:
    runner = CliRunner()
    create = runner.invoke(
        app, ["agent", "create", "my-bench", "--benchmarks-dir", str(tmp_path)]
    )
    assert create.exit_code == 0, create.output
    assert (tmp_path / "my-bench" / "benchflow.py").exists()

    # a fresh scaffold has no parity evidence → verify exits non-zero and
    # prints the support-path issue draft.
    verify = runner.invoke(
        app, ["agent", "verify", "my-bench", "--benchmarks-dir", str(tmp_path)]
    )
    assert verify.exit_code == 1
    out = click.unstyle(verify.output)
    assert "insufficient-evidence" in out
    assert "NOT been filed" in out


def test_cli_create_refuses_existing(tmp_path: Path) -> None:
    runner = CliRunner()
    runner.invoke(
        app, ["agent", "create", "my-bench", "--benchmarks-dir", str(tmp_path)]
    )
    again = runner.invoke(
        app, ["agent", "create", "my-bench", "--benchmarks-dir", str(tmp_path)]
    )
    assert again.exit_code == 1
    assert "already exists" in click.unstyle(again.output)


def test_cli_run_dry_run_prints_command(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app, ["agent", "run", "github.com/foo/bar", "--name", "my-bench", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    out = click.unstyle(result.output)
    assert "codex" in out
    assert "exec" in out


def test_cli_verify_pass_exits_zero(tmp_path: Path) -> None:
    create_benchmark("my-bench", tmp_path)
    parity = tmp_path / "my-bench" / "parity_experiment.json"
    parity.write_text(json.dumps(_criteria_doc(("pass", "pass"))))
    result = CliRunner().invoke(
        app, ["agent", "verify", "my-bench", "--benchmarks-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "parity-confirmed" in click.unstyle(result.output)


def test_cli_verify_malformed_json_prints_clean_message(tmp_path: Path) -> None:
    create_benchmark("my-bench", tmp_path)
    (tmp_path / "my-bench" / "parity_experiment.json").write_text("{ bad json")
    result = CliRunner().invoke(
        app, ["agent", "verify", "my-bench", "--benchmarks-dir", str(tmp_path)]
    )
    out = click.unstyle(result.output)
    # Fail-closed: a clean actionable message, no uncaught JSONDecodeError.
    assert "not valid JSON" in out
    assert "JSONDecodeError" not in out
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_cli_verify_nonexistent_roundtrip_task_no_traceback(tmp_path: Path) -> None:
    create_benchmark("my-bench", tmp_path)
    parity = tmp_path / "my-bench" / "parity_experiment.json"
    parity.write_text(json.dumps(_criteria_doc(("pass", "pass"))))
    bad = tmp_path / "does-not-exist"
    result = CliRunner().invoke(
        app,
        [
            "agent",
            "verify",
            "my-bench",
            "--benchmarks-dir",
            str(tmp_path),
            "--roundtrip-task",
            str(bad),
        ],
    )
    assert result.exit_code == 1
    # The clean-exit assertion is the mutation-killer: a bare shutil.copytree
    # FileNotFoundError would surface here instead of SystemExit.
    assert result.exception is None or isinstance(result.exception, SystemExit)
    # Collapse rich's line-wrapping before substring checks.
    out = " ".join(click.unstyle(result.output).split())
    assert "round-trip: error" in out
    assert bad.name in out  # names the --roundtrip-task value the user passed
    assert "Traceback" not in out
    assert "FileNotFoundError" not in out


def test_cli_verify_non_numeric_reward_is_clean(tmp_path: Path) -> None:
    create_benchmark("my-bench", tmp_path)
    parity = tmp_path / "my-bench" / "parity_experiment.json"
    parity.write_text(
        json.dumps(
            {
                "agent_parity": {
                    "results": [
                        {
                            "task_id": "t1",
                            "legacy_reward": "not-a-number",
                            "converted_reward": "also-bad",
                        }
                    ]
                }
            }
        )
    )
    result = CliRunner().invoke(
        app, ["agent", "verify", "my-bench", "--benchmarks-dir", str(tmp_path)]
    )
    # Non-numeric reward fields must not crash on float(); the sample is skipped
    # so the gate degrades to insufficient-evidence, never a ValueError.
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "insufficient-evidence" in click.unstyle(result.output)


def test_default_reward_tolerance_is_small() -> None:
    assert 0 < DEFAULT_REWARD_TOLERANCE < 0.1
