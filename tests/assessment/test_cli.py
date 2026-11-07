"""``bench assess``: opt-in, lazily loaded, and the same calls as the Python API."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from benchflow.cli.main import app
from tests.assessment._fixtures import RUBRIC_ID, RUBRIC_VERSION, world


def _bench(*args: str) -> tuple[int, str, str]:
    result = CliRunner().invoke(app, list(args), prog_name="bench")
    return result.exit_code, result.stdout, result.stderr


def test_bench_help_does_not_load_the_assessment_code() -> None:
    script = (
        "import sys\n"
        "from typer.testing import CliRunner\n"
        "from benchflow.cli.main import app\n"
        "result = CliRunner().invoke(app, ['--help'], prog_name='bench')\n"
        "assert result.exit_code == 0, result.output\n"
        "assert 'assessment' in result.output\n"
        "print(sorted(m for m in sys.modules if 'assessment' in m))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    ).stdout
    assert out.strip() == "[]"


def test_a_full_run_through_the_cli(tmp_path: Path) -> None:
    w = world(tmp_path)
    refs = []
    for index, rollout in enumerate((w.first, w.second)):
        out = tmp_path / f"attempt-{index}.json"
        code, stdout, _ = _bench(
            "assess",
            "attempt-ref",
            str(rollout.root),
            "--output",
            str(rollout.output),
            "--review-artifact",
            str(rollout.preview),
            "--sandbox-provider",
            "docker",
            "--out",
            str(out),
        )
        assert code == 0, stdout
        refs.append(out)
    code, _, stderr = _bench(
        "assess", "attempt-ref", str(w.first.root), "--output", str(w.first.output),
        "--review-artifact", str(w.first.preview), "--sandbox-provider", "docker", "--out", str(refs[0]),
    )  # fmt: skip
    assert code == 1 and "never overwritten" in stderr
    sidecar = str(w.sidecar)
    code, stdout, _ = _bench(
        "assess",
        "create",
        sidecar,
        "--attempt",
        str(refs[0]),
        "--attempt",
        str(refs[1]),
        "--rubric",
        str(w.rubric),
        "--rubric-id",
        RUBRIC_ID,
        "--rubric-version",
        RUBRIC_VERSION,
        "--protocol",
        str(w.protocol),
    )
    assert code == 0 and stdout.startswith("appended: assessment.created #1"), stdout
    code, stdout, _ = _bench(
        "assess", "create", sidecar, "--attempt", str(refs[1]), "--attempt", str(refs[0]),
        "--rubric", str(w.rubric), "--rubric-id", RUBRIC_ID, "--rubric-version", RUBRIC_VERSION,
        "--protocol", str(w.protocol),
    )  # fmt: skip
    assert code == 0 and stdout.startswith("already recorded: assessment.created #1"), (
        stdout
    )

    assignment = w.collector_assignment()
    code, _, stderr = _bench(
        "assess", "assign", sidecar, "--collector-db", str(w.collector.db),
        "--collector-code", str(w.collector.code), "--assignment", assignment,
    )  # fmt: skip
    assert code == 1 and "compute.released first" in stderr
    assert "next:" in stderr

    code, stdout, _ = _bench("assess", "release", sidecar, "--proof", str(w.proof))
    assert code == 0, stdout
    assignment = w.collector_assignment()  # a fresh one, after the release
    code, stdout, _ = _bench(
        "assess", "assign", sidecar, "--collector-db", str(w.collector.db),
        "--collector-code", str(w.collector.code), "--assignment", assignment, "--json",
    )  # fmt: skip
    assert code == 0, stdout
    assert json.loads(stdout)["state"] == "awaiting_human"

    w.vote("tie", reason="both fine")
    code, stdout, _ = _bench(
        "assess", "import-vote", sidecar, "--collector-db", str(w.collector.db),
        "--collector-code", str(w.collector.code),
    )  # fmt: skip
    assert code == 0, stdout
    code, stdout, _ = _bench("assess", "reduce", sidecar)
    assert code == 0 and "0.5 (tie)" in stdout, stdout

    code, stdout, _ = _bench("assess", "show", sidecar, "--json")
    shown = json.loads(stdout)
    assert shown["summary"]["state"] == "label_ready" and shown["label"]["value"] == 0.5

    code, stdout, _ = _bench(
        "assess",
        "consume",
        sidecar,
        "--revision",
        "1",
        "--consumer",
        "ledger",
        "--receipt",
        "row-1",
    )
    assert code == 0, stdout
    code, _, stderr = _bench(
        "assess",
        "consume",
        sidecar,
        "--revision",
        "1",
        "--consumer",
        "ledger",
        "--receipt",
        "row-2",
    )
    assert code == 3 and "different payload" in stderr

    target = shown["summary"]["effective_answer_event_id"]
    code, stdout, _ = _bench(
        "assess", "correct", sidecar, "--supersedes", target, "--choice", "left",
        "--reason", "A reads better", "--correction-reason", "misclick", "--actor", "held-out-reviewer-01",
    )  # fmt: skip
    assert code == 0, stdout
    code, stdout, _ = _bench("assess", "reduce", sidecar, "--json")
    label = json.loads(stdout)
    assert (label["revision"], label["value"], label["resolved"]) == (2, 1, False)
    code, stdout, _ = _bench(
        "assess", "withdraw", sidecar, "--revision", "1", "--reason", "superseded",
        "--disposition", "queued_compensation", "--actor", "operator-01",
    )  # fmt: skip
    assert code == 0, stdout

    code, stdout, _ = _bench("assess", "verify", sidecar, "--evidence", "--json")
    report = json.loads(stdout)
    assert (
        report["chain_verified"]
        and report["evidence_verified"]
        and report["label_file"] == "current"
    )
    code, stdout, _ = _bench("assess", "label", sidecar, "--check")
    assert code == 0 and json.loads(stdout)["resolved"] is True


def test_a_corrupt_sidecar_exits_1_with_a_next_step(tmp_path: Path) -> None:
    w = world(tmp_path)
    w.create()
    log = w.sidecar / "assessment-events.v1.jsonl"
    log.write_bytes(log.read_bytes()[:-5])
    code, _, stderr = _bench("assess", "show", str(w.sidecar))
    assert code == 1 and "torn append" in stderr and "next:" in stderr
    code, _, stderr = _bench("assess", "label", str(w.sidecar), "--check")
    assert code == 1


def test_every_default_next_step_names_a_real_command() -> None:
    """The steps a refusal falls back to name commands and flags that exist."""
    import re

    from benchflow.cli.assessment import ASSESSMENT_NEXT, NEXT_STEPS

    for step in (ASSESSMENT_NEXT, *NEXT_STEPS.values()):
        for span in re.findall(r"`(bench [^`]*)`", step):
            words = [w for w in span.split()[1:] if not w.startswith("<")]
            code, out, _ = (
                _bench(*words, "--help") if "--help" not in words else _bench(*words)
            )
            assert code == 0, (span, out)


def test_an_evidence_refusal_says_what_to_do(tmp_path: Path) -> None:
    w = world(tmp_path)
    ref = tmp_path / "a.json"
    ref.write_text(json.dumps(w.first.ref()))
    other = tmp_path / "b.json"
    other.write_text(json.dumps(w.second.ref()))
    (w.first.output / "index.html").write_bytes(b"changed")
    code, _, stderr = _bench(
        "assess", "create", str(w.sidecar), "--attempt", str(ref), "--attempt", str(other),
        "--rubric", str(w.rubric), "--rubric-id", RUBRIC_ID, "--rubric-version", RUBRIC_VERSION,
        "--protocol", str(w.protocol),
    )  # fmt: skip
    assert code == 1 and "output changed" in stderr
    assert "bench assess attempt-ref" in stderr


def test_a_directory_without_a_log_is_not_an_empty_assessment(tmp_path: Path) -> None:
    for command in ("show", "verify", "label"):
        code, _, stderr = _bench("assess", command, str(tmp_path / "nothing-here"))
        assert code == 1 and "holds no assessment log" in stderr, command
        assert "bench assess create" in stderr
