"""Tests for the ProgramBench onramp adapter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make `onramp` importable from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from onramp.programbench.benchflow import (
    UpstreamInstance,
    cleanroom_image_name,
    convert,
    render_task,
    sanitize_task_name,
)


def _write_upstream(tmp_path: Path, instance_id: str, *, language: str = "rust",
                    difficulty: str = "easy", branches: dict | None = None) -> Path:
    """Build a minimal ProgramBench-shaped upstream directory."""
    d = tmp_path / instance_id
    d.mkdir(parents=True)
    (d / "task.yaml").write_text(
        f"repository: example/{instance_id.split('__', 1)[0]}\n"
        f"commit: deadbeef\n"
        f"language: {language}\n"
        f"difficulty: {difficulty}\n"
    )
    tests_json = {"branches": branches or {"abc123def456": {"ignored": False, "tests": ["tests.foo.test_a", "tests.foo.test_b"]}}}
    (d / "tests.json").write_text(json.dumps(tests_json))
    return d


class TestSanitizeTaskName:
    def test_underscores_and_dot(self):
        assert (
            sanitize_task_name("abishekvashok__cmatrix.5c082c6")
            == "programbench/abishekvashok-cmatrix-5c082c6"
        )

    def test_uppercase_lowered(self):
        assert sanitize_task_name("Foo__Bar.Baz") == "programbench/foo-bar-baz"

    def test_stable_across_calls(self):
        a = sanitize_task_name("jq__jq.cff5336")
        b = sanitize_task_name("jq__jq.cff5336")
        assert a == b


class TestCleanroomImageName:
    def test_double_underscore_replaced(self):
        # Mirrors programbench.constants.image_name_from_instance_id.
        assert (
            cleanroom_image_name("abishekvashok__cmatrix.5c082c6")
            == "programbench/abishekvashok_1776_cmatrix.5c082c6"
        )


class TestUpstreamInstanceFromDir:
    def test_loads_yaml_and_tests(self, tmp_path):
        d = _write_upstream(tmp_path, "owner__proj.abc1234", language="go", difficulty="medium")
        inst = UpstreamInstance.from_dir(d)
        assert inst.instance_id == "owner__proj.abc1234"
        assert inst.repository == "example/owner"
        assert inst.commit == "deadbeef"
        assert inst.language == "go"
        assert inst.difficulty == "medium"
        assert "abc123def456" in inst.tests_json["branches"]


class TestRenderTask:
    def test_writes_all_required_files(self, tmp_path):
        d = _write_upstream(tmp_path, "owner__proj.abc1234")
        inst = UpstreamInstance.from_dir(d)
        out = tmp_path / "out"
        out.mkdir()
        task_dir = render_task(inst, out)
        assert (task_dir / "task.toml").exists()
        assert (task_dir / "instruction.md").exists()
        assert (task_dir / "environment" / "Dockerfile").exists()
        assert (task_dir / "tests" / "test.sh").exists()

    def test_task_toml_has_registry_name(self, tmp_path):
        d = _write_upstream(tmp_path, "owner__proj.abc1234")
        out = tmp_path / "out"

        out.mkdir()
        render_task(UpstreamInstance.from_dir(d), out)
        toml_text = (out / "owner__proj.abc1234" / "task.toml").read_text()
        # BenchFlow's registry requires a [task].name field.
        assert 'name = "programbench/owner-proj-abc1234"' in toml_text

    def test_dockerfile_uses_cleanroom_tag(self, tmp_path):
        d = _write_upstream(tmp_path, "owner__proj.abc1234")
        out = tmp_path / "out"

        out.mkdir()
        render_task(UpstreamInstance.from_dir(d), out)
        df = (out / "owner__proj.abc1234" / "environment" / "Dockerfile").read_text()
        assert "FROM programbench/owner_1776_proj.abc1234:task_cleanroom" in df
        assert 'ENV BF_INSTANCE_ID="owner__proj.abc1234"' in df

    def test_tests_json_sidecar_round_trips(self, tmp_path):
        # tests.json travels as a sidecar in tests/, not as a Dockerfile ENV —
        # cmatrix-class tasks have hundreds of branches and the b64 string blew
        # past Docker's 65 535-byte ENV-line limit.
        branches = {"deadbeef": {"ignored": False, "tests": ["t1", "t2"]}}
        d = _write_upstream(tmp_path, "owner__proj.abc1234", branches=branches)
        out = tmp_path / "out"

        out.mkdir()
        render_task(UpstreamInstance.from_dir(d), out)
        sidecar = out / "owner__proj.abc1234" / "tests" / "tests.json"
        assert sidecar.exists()
        decoded = json.loads(sidecar.read_text())
        assert decoded == {"branches": branches}
        df = (out / "owner__proj.abc1234" / "environment" / "Dockerfile").read_text()
        assert "BF_TESTS_JSON_B64" not in df  # guards the move from ENV → file
        for line in df.splitlines():
            assert len(line) < 65535

    def test_test_sh_is_static_and_executable(self, tmp_path):
        d = _write_upstream(tmp_path, "owner__proj.abc1234")
        out = tmp_path / "out"

        out.mkdir()
        render_task(UpstreamInstance.from_dir(d), out)
        ts = out / "owner__proj.abc1234" / "tests" / "test.sh"
        assert ts.read_text().startswith("#!/bin/bash")
        assert (ts.stat().st_mode & 0o111) != 0
        # test.sh should NOT contain task-specific values; per-task data is
        # ENV-injected so the script is identical for every generated task.
        body = ts.read_text()
        assert "owner__proj.abc1234" not in body

    def test_difficulty_resources_propagate(self, tmp_path):
        d = _write_upstream(tmp_path, "owner__hard.123abcd", difficulty="hard")
        out = tmp_path / "out"

        out.mkdir()
        render_task(UpstreamInstance.from_dir(d), out)
        toml_text = (out / "owner__hard.123abcd" / "task.toml").read_text()
        assert "memory_mb = 16384" in toml_text
        assert "timeout_sec = 7200" in toml_text

    def test_unknown_difficulty_falls_back_to_unrated(self, tmp_path):
        d = _write_upstream(tmp_path, "owner__weird.123abcd", difficulty="weird-unknown")
        out = tmp_path / "out"

        out.mkdir()
        render_task(UpstreamInstance.from_dir(d), out)
        toml_text = (out / "owner__weird.123abcd" / "task.toml").read_text()
        # Unrated config: memory 8192, agent timeout 3600.
        assert "memory_mb = 8192" in toml_text


class TestConvert:
    def test_full_run_emits_each_task(self, tmp_path):
        for tid in ("aaa__one.deadbee", "bbb__two.beadfee", "ccc__three.feedfee"):
            _write_upstream(tmp_path, tid)
        out = tmp_path / "out"

        out.mkdir()
        generated = convert(upstream_tasks_dir=tmp_path, output_dir=out)
        assert len(generated) == 3
        assert {p.name for p in generated} == {"aaa__one.deadbee", "bbb__two.beadfee", "ccc__three.feedfee"}

    def test_limit_truncates(self, tmp_path):
        for tid in ("aaa__one.deadbee", "bbb__two.beadfee", "ccc__three.feedfee"):
            _write_upstream(tmp_path, tid)
        out = tmp_path / "out"

        out.mkdir()
        generated = convert(upstream_tasks_dir=tmp_path, output_dir=out, limit=2)
        assert len(generated) == 2
        # Sorted, so the first two alphabetical instances should be picked.
        assert {p.name for p in generated} == {"aaa__one.deadbee", "bbb__two.beadfee"}

    def test_task_ids_filter(self, tmp_path):
        for tid in ("aaa__one.deadbee", "bbb__two.beadfee", "ccc__three.feedfee"):
            _write_upstream(tmp_path, tid)
        out = tmp_path / "out"

        out.mkdir()
        generated = convert(
            upstream_tasks_dir=tmp_path,
            output_dir=out,
            task_ids=["bbb__two.beadfee", "ccc__three.feedfee"],
        )
        assert {p.name for p in generated} == {"bbb__two.beadfee", "ccc__three.feedfee"}

    def test_existing_tasks_preserved_without_overwrite(self, tmp_path):
        _write_upstream(tmp_path, "aaa__one.deadbee")
        out = tmp_path / "out"

        out.mkdir()
        convert(upstream_tasks_dir=tmp_path, output_dir=out)
        marker = out / "aaa__one.deadbee" / "instruction.md"
        marker.write_text("HAND-EDITED")
        # Re-run without overwrite — hand edits must survive.
        convert(upstream_tasks_dir=tmp_path, output_dir=out)
        assert marker.read_text() == "HAND-EDITED"

    def test_overwrite_regenerates(self, tmp_path):
        _write_upstream(tmp_path, "aaa__one.deadbee")
        out = tmp_path / "out"

        out.mkdir()
        convert(upstream_tasks_dir=tmp_path, output_dir=out)
        marker = out / "aaa__one.deadbee" / "instruction.md"
        marker.write_text("HAND-EDITED")
        convert(upstream_tasks_dir=tmp_path, output_dir=out, overwrite=True)
        assert marker.read_text() != "HAND-EDITED"


class TestGeneratedTaskShape:
    """Sanity check that generated tasks would pass `bench tasks check`."""

    def test_generated_task_passes_check(self, tmp_path):
        from benchflow.tasks import check_task
        d = _write_upstream(tmp_path, "owner__proj.abc1234")
        out = tmp_path / "out"

        out.mkdir()
        render_task(UpstreamInstance.from_dir(d), out)
        issues = check_task(out / "owner__proj.abc1234")
        assert issues == [], f"check_task complained: {issues}"
