"""Contract tests for the ASI-Bench integration.

These tests cover the converter boundary and verifier contract.  They do not
run BenchFlow, download source material, or claim scoring parity.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

ROOT = Path(__file__).parents[1]
ADAPTER = ROOT / "benchmarks" / "asi-bench"


def _load_adapter():
    spec = importlib.util.spec_from_file_location(
        "asi_bench_benchflow", ADAPTER / "benchflow.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_scaffold_has_planned_public_files_and_no_materialized_tasks() -> None:
    expected = {
        "__init__.py",
        "benchflow.py",
        "main.py",
        "parity_test.py",
        "run_asi_bench.py",
        "asi-bench.yaml",
        "benchmark.yaml",
        "parity_experiment.json",
        "README.md",
        "verifier_template/test.sh",
        "verifier_template/score_entry.py",
    }
    actual = {
        str(path.relative_to(ADAPTER))
        for path in ADAPTER.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert expected <= actual
    assert not (ADAPTER / "tasks").exists()
    assert not (ADAPTER / "source").exists()
    assert not any(
        path.name in {"generate_gt.py", "precompute_gt.py"}
        for path in ADAPTER.rglob("*")
    )


def test_descriptor_reports_implemented_adapter_and_incomplete_parity() -> None:
    """Guards P5 descriptor drift after converter and verifier reached AWS P3 parity."""
    adapter = _load_adapter()
    descriptor = yaml.safe_load((ADAPTER / "benchmark.yaml").read_text())
    assert descriptor["name"] == "asi-bench"
    assert descriptor["status"] == "parity-pending"
    assert descriptor["tasks"]["seed"] == 31415
    assert descriptor["tasks"]["official"] is False
    assert "runner-task-image" in descriptor["tasks"]["excluded"]
    assert descriptor["conversion"]["status"] == "implemented"
    assert descriptor["verification"]["status"] == "implemented"
    assert descriptor["parity"]["status"] == "insufficient-evidence"

    manifest = json.loads((ADAPTER / "evaluator_files.json").read_text())
    assert manifest["upstream"]["revision"] == adapter.ASI_REVISION

    parity = json.loads((ADAPTER / "parity_experiment.json").read_text())
    assert descriptor["tasks"]["count"] == (
        len(parity["coverage"]["included_task_ids"])
        * len(parity["coverage"]["required_prompt_levels"])
    )
    assert parity["status"] == "template"
    assert parity["coverage"]["status"] == "incomplete"
    assert parity["coverage"]["source_task_count"] == 60
    assert len(parity["coverage"]["included_task_ids"]) == 46
    assert parity["coverage"]["required_prompt_levels"] == ["B1", "B2", "B3", "B4"]
    assert parity["coverage"]["required_harnesses"] == ["opencode", "pi-acp"]
    assert parity["scope"] == {
        "seed": 31415,
        "official": False,
        "asi_revision": adapter.ASI_REVISION,
        "hf_revision": adapter.HF_REVISION,
    }
    assert parity["conversion_parity"]["tasks"] == []
    assert parity["agent_parity"]["results"] == []


def test_adapter_declares_only_supported_seed_and_levels() -> None:
    adapter = _load_adapter()
    assert adapter.SUPPORTED_SEED == 31415
    assert adapter.PROMPT_LEVELS == ("B1", "B2", "B3", "B4")
    # Revisions are now real pinned values (not placeholder strings)
    assert len(adapter.ASI_REVISION) == 40  # 40-char git SHA
    assert not adapter.ASI_REVISION.startswith("REPLACE_WITH_")
    assert adapter.HF_REVISION == "0fa14219cafdbab634d8b3cfbce238a8735a214f"

    valid = adapter.ASIBenchInstance(
        task_id="demo",
        instance_id="demo__seed31415",
        prompt_level="B1",
        seed=31415,
        instance_dir=ROOT,
        task_bundle_dir=ROOT,
    )
    adapter.validate_instance(valid)

    with pytest.raises(ValueError, match="only supports seed 31415"):
        adapter.validate_instance(valid.__class__(**{**valid.__dict__, "seed": 42}))

    with pytest.raises(ValueError, match="prompt_level"):
        adapter.validate_instance(
            valid.__class__(**{**valid.__dict__, "prompt_level": "B5"})
        )


@pytest.mark.parametrize("revision_name", ["ASI_REVISION", "HF_REVISION"])
@pytest.mark.parametrize("moving_revision", ["main", "refs/heads/main", "v1"])
def test_converter_rejects_moving_source_revisions_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    revision_name: str,
    moving_revision: str,
) -> None:
    """Guards the P5 pin to HF commit 0fa14219 after P3 used moving main."""
    adapter = _load_adapter()
    monkeypatch.setattr(adapter, revision_name, moving_revision)
    output = tmp_path / "tasks"
    with pytest.raises(
        ValueError, match=f"{revision_name} must be a 40-character commit"
    ):
        adapter.convert_all(None, output)
    assert not output.exists()


def test_converter_rejects_seed42_and_bad_level(tmp_path: Path) -> None:
    adapter = _load_adapter()

    # seed42 is rejected by validate_instance before any I/O
    with pytest.raises(ValueError, match="only supports seed 31415"):
        adapter.validate_instance(
            adapter.ASIBenchInstance(
                task_id="demo",
                instance_id="demo__seed42",
                prompt_level="B1",
                seed=42,
                instance_dir=tmp_path,
                task_bundle_dir=tmp_path,
            )
        )

    # convert() rejects missing instance_dir
    instance = adapter.ASIBenchInstance(
        task_id="demo",
        instance_id="demo__seed31415",
        prompt_level="B1",
        seed=31415,
        instance_dir=tmp_path / "nonexistent_instance",
        task_bundle_dir=tmp_path / "nonexistent_bundle",
    )
    output_dir = tmp_path / "output"
    with pytest.raises((ValueError, FileNotFoundError)):
        adapter.convert(instance, output_dir)
    # No partial output should be left
    assert not output_dir.exists()


def test_task_runtime_is_installed_in_verifier_environment() -> None:
    """Guards the mpsc B1 verifier failure in AWS job 77319f47 (2026-10-10)."""
    adapter = _load_adapter()
    meta = {"runtime": {"python": ">=3.11", "packages": ["cvxpy>=1.4", "clarabel>=0.7"]}}
    task_dockerfile = adapter._build_dockerfile(meta, {"evaluation": {"runtime": "task"}})
    evaluator_install = task_dockerfile.split("# Task runtime packages")[0]
    assert "'cvxpy>=1.4'" in evaluator_install
    assert "'clarabel>=0.7'" in evaluator_install
    assert "'cvxpy>=1.4'" in task_dockerfile.split("# Task runtime packages")[1]

    ordinary_dockerfile = adapter._build_dockerfile(meta, {"evaluation": {}})
    ordinary_evaluator_install = ordinary_dockerfile.split("# Task runtime packages")[0]
    assert "cvxpy>=1.4" not in ordinary_evaluator_install


def test_custom_scorer_dependencies_are_installed_in_verifier_environment() -> None:
    """Guards the particle B1 pandas verifier failure on AWS (2026-10-10)."""
    adapter = _load_adapter()
    meta = {"runtime": {"python": ">=3.10", "packages": ["pandas>=2.0"]}}
    dockerfile = adapter._build_dockerfile(
        meta, {"evaluation": {}}, custom_scorer=True
    )
    assert "'pandas>=2.0'" in dockerfile.split("# Task runtime packages")[0]


def test_convert_all_requires_task_ids_when_no_source_dir(tmp_path: Path) -> None:
    adapter = _load_adapter()
    # Without source_dir AND without task_ids → should raise
    with pytest.raises((ValueError, TypeError)):
        adapter.convert_all(None, tmp_path / "out")


def test_download_retries_after_partial_instance(tmp_path, monkeypatch) -> None:
    """Guards the c13683ca downloader against the P5 partial-Levin retry failure."""
    adapter = _load_adapter()
    calls = []

    def download_instance(task_id, destination, revision, cache_dir):
        calls.append(task_id)
        (destination / "prompt_b1.md").write_text("partial")
        if len(calls) == 1:
            raise RuntimeError("download interrupted")
        for level in (2, 3, 4):
            (destination / f"prompt_b{level}.md").write_text("ready")
        (destination / "reference").mkdir()
        (destination / "reference" / "answer.json").write_text("{}")
        return revision

    def download_bundle(task_id, destination, revision):
        (destination / "task_meta.yaml").write_text("id: math.demo\n")
        (destination / "task_eval.yaml").write_text("evaluation: {}\n")
        return revision

    monkeypatch.setattr(adapter, "_download_instance", download_instance)
    monkeypatch.setattr(adapter, "_download_task_bundle", download_bundle)
    root = tmp_path / "downloads"

    with pytest.raises(RuntimeError, match="download interrupted"):
        adapter._download_task_assets("math.demo", root, "a" * 40, "b" * 40, None)
    assert not (root / "instance" / "math.demo__seed31415").exists()
    adapter._download_task_assets("math.demo", root, "a" * 40, "b" * 40, None)
    adapter._download_task_assets("math.demo", root, "a" * 40, "b" * 40, None)

    instance = root / "instance" / "math.demo__seed31415"
    assert len(calls) == 2
    assert all((instance / f"prompt_b{level}.md").is_file() for level in (1, 2, 3, 4))


def test_download_retries_after_partial_task_bundle(tmp_path, monkeypatch) -> None:
    """Guards the c13683ca downloader against the P5 partial-bundle retry failure."""
    adapter = _load_adapter()
    bundle_calls = []

    def download_instance(task_id, destination, revision, cache_dir):
        for level in (1, 2, 3, 4):
            (destination / f"prompt_b{level}.md").write_text("ready")
        return revision

    def download_bundle(task_id, destination, revision):
        bundle_calls.append(task_id)
        (destination / "task_meta.yaml").write_text("partial")
        if len(bundle_calls) == 1:
            raise RuntimeError("rate limit exceeded")
        (destination / "task_eval.yaml").write_text("evaluation: {}\n")
        return revision

    monkeypatch.setattr(adapter, "_download_instance", download_instance)
    monkeypatch.setattr(adapter, "_download_task_bundle", download_bundle)
    root = tmp_path / "downloads"

    with pytest.raises(RuntimeError, match="rate limit exceeded"):
        adapter._download_task_assets("math.demo", root, "a" * 40, "b" * 40, None)
    assert not (root / "task_bundle" / "tasks" / "math" / "demo").exists()
    adapter._download_task_assets("math.demo", root, "a" * 40, "b" * 40, None)
    adapter._download_task_assets("math.demo", root, "a" * 40, "b" * 40, None)

    bundle = root / "task_bundle" / "tasks" / "math" / "demo"
    assert len(bundle_calls) == 2
    assert (bundle / "task_eval.yaml").is_file()


def test_verifier_exits_nonzero_without_instance_json(tmp_path: Path) -> None:
    """Verifier must exit non-zero and not write reward.txt when metadata is missing."""
    script = ADAPTER / "verifier_template" / "score_entry.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode != 0
    # No reward files should be created when infrastructure fails
    assert not list(tmp_path.rglob("reward.txt"))
    assert not list(tmp_path.rglob("reward.json"))


def test_parity_placeholder_reports_insufficient_evidence() -> None:
    result = subprocess.run(
        [sys.executable, str(ADAPTER / "parity_test.py"), "--mode", "full"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "insufficient-evidence"


def test_verifier_test_script_is_executable() -> None:
    assert os.access(ADAPTER / "verifier_template" / "test.sh", os.X_OK)


def test_converter_copies_complete_data_to_agent_and_verifier(tmp_path, monkeypatch) -> None:
    """Guards the seed31415 homotopy P3 missing-system.json regression (2026-10-09)."""
    adapter = _load_adapter()
    monkeypatch.setattr(adapter, "_render_task_md", lambda **_: "task")
    monkeypatch.setattr(adapter, "_build_dockerfile", lambda *_, **__: "FROM python:3.11\n")
    monkeypatch.setattr(adapter, "_materialize_evaluator_files", lambda *_: None)
    materialize_helpers = Mock(wraps=adapter._materialize_task_helpers)
    monkeypatch.setattr(adapter, "_materialize_task_helpers", materialize_helpers)
    instance = tmp_path / "instance"
    (instance / "data" / "nested").mkdir(parents=True)
    (instance / "data" / "system.json").write_text('{"system": 1}')
    (instance / "data" / "nested" / "extra.json").write_text("extra")
    reference = instance / "reference"
    reference.mkdir()
    (reference / "answer.json").write_text("{}")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "task_eval.yaml").write_text("evaluation: {}\n")
    staging = tmp_path / "task"
    staging.mkdir()
    adapter._generate_task_dir(
        staging=staging, task_id="math.demo", instance_id="math.demo__seed31415",
        prompt_level="B1", task_meta={"input": {"files": [{"name": "system.json"}]}},
        task_eval={"evaluation": {}}, prompt_text="solve", instance_dir=instance,
        bundle_dir=bundle, reference_dir=reference,
        asi_revision="a" * 40, hf_revision="b" * 40,
    )
    for root in (
        staging / "environment" / "inputs",
        staging / "verifier" / "instance_data",
        staging / "verifier",
    ):
        assert (root / "data" / "system.json").read_text() == '{"system": 1}'
        assert (root / "data" / "nested" / "extra.json").read_text() == "extra"
    metadata = json.loads((staging / "verifier" / "instance.json").read_text())
    assert metadata["requires_instance_data"] is True
    assert materialize_helpers.call_count == 1
    assert materialize_helpers.call_args.args[1:] == (
        "math.demo", bundle, staging / "verifier" / "evaluator"
    )


def test_converter_rejects_missing_declared_data_root(tmp_path, monkeypatch) -> None:
    """Guards the seed31415 homotopy P3 missing-system.json regression (2026-10-09)."""
    adapter = _load_adapter()
    monkeypatch.setattr(adapter, "_render_task_md", lambda **_: "task")
    monkeypatch.setattr(adapter, "_build_dockerfile", lambda *_, **__: "FROM python:3.11\n")
    monkeypatch.setattr(adapter, "_materialize_evaluator_files", lambda *_: None)
    instance = tmp_path / "instance"
    instance.mkdir()
    reference = instance / "reference"
    reference.mkdir()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    staging = tmp_path / "task"
    staging.mkdir()
    with pytest.raises(ValueError, match="data"):
        adapter._generate_task_dir(
            staging=staging, task_id="math.demo", instance_id="math.demo__seed31415",
            prompt_level="B1", task_meta={"input": {"files": [{"name": "system.json"}]}},
            task_eval={"evaluation": {}}, prompt_text="solve", instance_dir=instance,
            bundle_dir=bundle, reference_dir=reference,
            asi_revision="a" * 40, hf_revision="b" * 40,
        )


def test_evaluator_materializer_rejects_unpinned_source_bytes(tmp_path, monkeypatch) -> None:
    """Guards the seed31415 P3 source revision mismatch (2026-10-09)."""
    adapter = _load_adapter()
    source = tmp_path / "asi"
    (source / "ai4sci_bench").mkdir(parents=True)
    (source / "ai4sci_bench" / "core.py").write_text("actual")
    monkeypatch.setenv("ASI_BENCH_SOURCE", str(source))
    spec = {
        "upstream": {"revision": adapter.ASI_REVISION},
        "files": [{"kind": "copy", "path": "ai4sci_bench/core.py",
                   "source": "ai4sci_bench/core.py",
                   "source_sha256": hashlib.sha256(b"expected").hexdigest()}],
    }
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        adapter._materialize_evaluator_files(spec, tmp_path / "evaluator")


def test_converter_materializes_only_pinned_public_task_helpers(tmp_path) -> None:
    """Guards the Levin P3 missing-lts_eval_runtime verifier crash (2026-10-10)."""
    adapter = _load_adapter()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    helper = bundle / "lts_eval_runtime.py"
    helper.write_text("VALUE = 1\n")
    expected = hashlib.sha256(helper.read_bytes()).hexdigest()
    spec = {
        "task_helper_files": {
            "math.levin_context_grid_search": [
                {"path": helper.name, "source_sha256": expected}
            ]
        }
    }
    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()

    adapter._materialize_task_helpers(
        spec, "math.levin_context_grid_search", bundle, evaluator
    )
    assert (evaluator / helper.name).read_bytes() == helper.read_bytes()
    assert sorted(path.name for path in evaluator.iterdir()) == [helper.name]

    with pytest.raises(FileNotFoundError, match="Pinned task helper missing"):
        adapter._materialize_task_helpers(
            spec, "math.levin_context_grid_search", tmp_path / "missing", tmp_path / "other"
        )

    helper.write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        adapter._materialize_task_helpers(
            spec, "math.levin_context_grid_search", bundle, tmp_path / "other"
        )
