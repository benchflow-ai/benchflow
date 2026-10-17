"""Contract tests for the ASI-Bench verifier (score_entry.py).

Covers plan §4 requirements:
- Missing output files → submission failure (reward 0, exit 0), NOT evaluator error
- Symlinks in workspace → not copied, treated as absent
- Path traversal in output spec → rejected
- Data/ covered by output → staging refuses to overwrite data inputs
- Evaluator infrastructure error → no reward written, exit 1
- reward.txt and reward.json scalar must be equal
- reward ∈ [0, 1] on successful scoring
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCORE_ENTRY = ROOT / "benchmarks" / "asi-bench" / "verifier_template" / "score_entry.py"


def _load_score_entry() -> object:
    """Import score_entry.py as a module without executing main()."""
    spec = importlib.util.spec_from_file_location("asi_score_entry", SCORE_ENTRY)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["asi_score_entry"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_task_scorer_uses_prebuilt_verifier_runtime(monkeypatch) -> None:
    """Guards the mpsc B1 verifier failure in AWS job 77319f47 (2026-10-10)."""
    mod = _load_score_entry()
    names = (
        "AI4SCI_TASK_RUNTIME_ACTIVE",
        "AI4SCI_TASK_RUNTIME_PYTHON",
        "AI4SCI_TASK_RUNTIME_BIN",
        "AI4SCI_TRUSTED_SITE_PACKAGES",
    )
    original = {name: os.environ.get(name) for name in names}
    for name in names:
        monkeypatch.delenv(name, raising=False)
    try:
        mod._activate_task_runtime({})
        assert "AI4SCI_TASK_RUNTIME_ACTIVE" not in os.environ
        mod._activate_task_runtime({"runtime": "task"})
        assert os.environ["AI4SCI_TASK_RUNTIME_ACTIVE"] == "1"
        assert os.environ["AI4SCI_TASK_RUNTIME_PYTHON"] == sys.executable
        assert os.environ["AI4SCI_TASK_RUNTIME_BIN"] == str(Path(sys.executable).parent)
        assert os.environ["AI4SCI_TRUSTED_SITE_PACKAGES"]
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_instance_json(tmp_path: Path, **overrides) -> Path:
    data = {
        "schema_version": 1,
        "task_id": "demo.task",
        "instance_id": "demo.task__seed31415",
        "prompt_level": "B1",
        "seed": 31415,
        "official": False,
        "asi_revision": "a" * 40,
        "hf_revision": "main",
        "parameters": {},
        **overrides,
    }
    verifier_dir = tmp_path / "verifier"
    verifier_dir.mkdir(exist_ok=True)
    p = verifier_dir / "instance.json"
    p.write_text(json.dumps(data))
    return p


def _make_output_specs(tmp_path: Path, specs: list[dict]) -> None:
    (tmp_path / "verifier" / "output_specs.json").write_text(json.dumps(specs))


def _make_score_divisor(tmp_path: Path, divisor: float = 1.0) -> None:
    (tmp_path / "verifier" / "score_divisor.json").write_text(
        json.dumps({"score_divisor": divisor})
    )


# ── _safe_rel tests ───────────────────────────────────────────────────────────

class TestSafeRel:
    def setup_method(self):
        self.mod = _load_score_entry()

    def test_normal_path_accepted(self):
        p = self.mod._safe_rel("results/output.csv")
        assert str(p) == "results/output.csv"

    def test_simple_filename_accepted(self):
        p = self.mod._safe_rel("output.csv")
        assert str(p) == "output.csv"

    def test_absolute_path_rejected(self):
        with pytest.raises(ValueError, match="unsafe"):
            self.mod._safe_rel("/etc/passwd")

    def test_traversal_rejected(self):
        with pytest.raises(ValueError, match="unsafe"):
            self.mod._safe_rel("../secret")

    def test_traversal_in_middle_rejected(self):
        with pytest.raises(ValueError, match="unsafe"):
            self.mod._safe_rel("a/../../b")

    def test_empty_string_rejected(self):
        with pytest.raises(ValueError, match="unsafe"):
            self.mod._safe_rel("")

    def test_backslash_rejected(self):
        with pytest.raises(ValueError, match="unsafe"):
            self.mod._safe_rel("a\\b")

    def test_null_byte_rejected(self):
        with pytest.raises(ValueError, match="unsafe"):
            self.mod._safe_rel("a\x00b")


# ── _stage_outputs tests ──────────────────────────────────────────────────────

class TestStageOutputs:
    def setup_method(self):
        self.mod = _load_score_entry()

    def test_present_file_is_staged(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "result.csv").write_text("a,b\n1,2\n")
        staging = tmp_path / "staging"
        staging.mkdir()

        specs = [{"name": "result.csv", "type": "data"}]
        present = self.mod._stage_outputs(workspace, specs, staging)

        assert "result.csv" in present
        assert (staging / "result.csv").is_file()

    def test_missing_file_not_in_present(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        staging = tmp_path / "staging"
        staging.mkdir()

        specs = [{"name": "missing.csv", "type": "data"}]
        present = self.mod._stage_outputs(workspace, specs, staging)

        assert "missing.csv" not in present
        assert not (staging / "missing.csv").exists()

    def test_symlink_not_staged(self, tmp_path):
        """Symlinks in workspace must be silently skipped (not staged)."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        real_file = tmp_path / "real.csv"
        real_file.write_text("data")
        link = workspace / "result.csv"
        link.symlink_to(real_file)

        staging = tmp_path / "staging"
        staging.mkdir()
        specs = [{"name": "result.csv", "type": "data"}]
        present = self.mod._stage_outputs(workspace, specs, staging)

        assert "result.csv" not in present
        assert not (staging / "result.csv").exists()

    def test_nested_output_path_staged(self, tmp_path):
        workspace = tmp_path / "workspace"
        (workspace / "results").mkdir(parents=True)
        (workspace / "results" / "x.csv").write_text("col\n1\n")
        staging = tmp_path / "staging"
        staging.mkdir()

        specs = [{"name": "results/x.csv", "type": "data"}]
        present = self.mod._stage_outputs(workspace, specs, staging)

        assert "results/x.csv" in present
        assert (staging / "results" / "x.csv").is_file()

    def test_traversal_spec_silently_skipped(self, tmp_path):
        """Output spec with path traversal must be silently skipped."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        staging = tmp_path / "staging"
        staging.mkdir()

        specs = [{"name": "../escape.txt", "type": "data"}]
        present = self.mod._stage_outputs(workspace, specs, staging)
        assert "../escape.txt" not in present
        # Ensure nothing was written outside staging
        assert not (tmp_path / "escape.txt").exists()

    def test_sha256_recorded_for_present_file(self, tmp_path):
        import hashlib

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        content = b"hello verifier"
        (workspace / "out.bin").write_bytes(content)
        staging = tmp_path / "staging"
        staging.mkdir()

        specs = [{"name": "out.bin", "type": "data"}]
        present = self.mod._stage_outputs(workspace, specs, staging)

        expected_sha = hashlib.sha256(content).hexdigest()
        assert present.get("out.bin") == expected_sha

    def test_declared_output_cannot_replace_evaluator_data(self, tmp_path):
        """Guards the seed31415 homotopy P3 input boundary (2026-10-09)."""
        workspace = tmp_path / "workspace"
        (workspace / "data").mkdir(parents=True)
        (workspace / "data" / "system.json").write_text("agent version")
        staging = tmp_path / "staging"
        (staging / "data").mkdir(parents=True)
        (staging / "data" / "system.json").write_text("trusted version")
        with pytest.raises(self.mod.MissingEvaluatorInputError, match="overlaps"):
            self.mod._stage_outputs(workspace, [{"name": "data/system.json"}], staging)
        assert (staging / "data" / "system.json").read_text() == "trusted version"

    def test_private_data_symlink_is_verifier_error(self, tmp_path):
        """Guards the seed31415 homotopy P3 input boundary (2026-10-09)."""
        private = tmp_path / "private"
        private.mkdir()
        (private / "system.json").symlink_to(tmp_path / "outside.json")
        (tmp_path / "staging").mkdir()
        with pytest.raises(self.mod.MissingEvaluatorInputError, match="symlink"):
            self.mod._stage_data_inputs(private, tmp_path / "staging", required=True)


# ── _write_reward / _write_error tests ───────────────────────────────────────

class TestWriteReward:
    def setup_method(self):
        self.mod = _load_score_entry()

    def test_reward_txt_and_json_scalar_match(self, tmp_path, monkeypatch):
        monkeypatch.setattr(self.mod, "LOGS_DIR", tmp_path / "logs" / "verifier")

        instance = {"task_id": "t", "instance_id": "i", "prompt_level": "B1",
                    "seed": 31415, "official": False, "asi_revision": "a"*40,
                    "hf_revision": "main"}
        result = {"final_score": 0.75, "max_score": 1.0, "hard_gates_passed": True,
                  "gate_results": [], "score_results": [], "score_divisor": 1.0}

        self.mod._write_reward(0.75, result, instance, {}, 1.0)

        txt_val = float((tmp_path / "logs" / "verifier" / "reward.txt").read_text().strip())
        json_val = json.loads(
            (tmp_path / "logs" / "verifier" / "reward.json").read_text()
        )["reward"]

        assert abs(txt_val - 0.75) < 1e-9
        assert abs(json_val - 0.75) < 1e-9
        assert abs(txt_val - json_val) < 1e-12

    def test_reward_in_zero_one(self, tmp_path, monkeypatch):
        monkeypatch.setattr(self.mod, "LOGS_DIR", tmp_path / "logs" / "verifier")
        instance = {"task_id": "t", "instance_id": "i", "prompt_level": "B1",
                    "seed": 31415, "official": False, "asi_revision": "a"*40,
                    "hf_revision": "main"}
        result = {"final_score": 0.42, "max_score": 1.0, "hard_gates_passed": False,
                  "gate_results": [], "score_results": [], "score_divisor": 1.0}

        self.mod._write_reward(0.42, result, instance, {}, 1.0)
        reward_json = json.loads(
            (tmp_path / "logs" / "verifier" / "reward.json").read_text()
        )
        assert 0.0 <= reward_json["reward"] <= 1.0

    def test_reward_json_has_no_flat_non_reward_scalars(self, tmp_path, monkeypatch):
        """BenchFlow rejects top-level non-[0,1] scalars; all ASI fields must be in metadata."""
        monkeypatch.setattr(self.mod, "LOGS_DIR", tmp_path / "logs" / "verifier")
        instance = {"task_id": "t", "instance_id": "i", "prompt_level": "B1",
                    "seed": 31415, "official": False, "asi_revision": "a"*40,
                    "hf_revision": "main"}
        result = {"final_score": 77.3, "max_score": 100.0, "hard_gates_passed": True,
                  "gate_results": [], "score_results": [], "score_divisor": 1.0}

        self.mod._write_reward(0.773, result, instance, {}, 1.0)
        reward_json = json.loads(
            (tmp_path / "logs" / "verifier" / "reward.json").read_text()
        )
        # Top-level keys: only "reward" may be a numeric scalar
        for key, val in reward_json.items():
            if isinstance(val, (int, float)):
                assert key == "reward", (
                    f"Flat numeric field {key!r}={val} would be rejected by BenchFlow"
                )

    def test_write_error_does_not_write_reward(self, tmp_path, monkeypatch):
        """Evaluator/infra error must NOT produce reward.txt or reward.json."""
        logs_dir = tmp_path / "logs" / "verifier"
        monkeypatch.setattr(self.mod, "LOGS_DIR", logs_dir)

        exc = RuntimeError("evaluator crashed")
        self.mod._write_error(exc, "test context")

        assert not (logs_dir / "reward.txt").exists()
        assert not (logs_dir / "reward.json").exists()
        assert (logs_dir / "asi_error.json").exists()

    def test_write_error_cleans_existing_reward_files(self, tmp_path, monkeypatch):
        """If reward files exist from a previous run, _write_error must remove them."""
        logs_dir = tmp_path / "logs" / "verifier"
        logs_dir.mkdir(parents=True)
        (logs_dir / "reward.txt").write_text("0.5")
        (logs_dir / "reward.json").write_text('{"reward": 0.5}')
        monkeypatch.setattr(self.mod, "LOGS_DIR", logs_dir)

        self.mod._write_error(ValueError("oops"), "cleanup test")

        assert not (logs_dir / "reward.txt").exists()
        assert not (logs_dir / "reward.json").exists()


# ── _normalize_reward tests ───────────────────────────────────────────────────

class TestNormalizeReward:
    def setup_method(self):
        self.mod = _load_score_entry()

    def test_basic_normalization(self):
        result = {"final_score": 50.0, "max_score": 100.0}
        assert abs(self.mod._normalize_reward(result) - 0.5) < 1e-9

    def test_score_divisor_override(self):
        # The framework applies the divisor to both score and maximum.
        result = {"final_score": 99.75 / 1.05, "max_score": 100.0 / 1.05}
        reward = self.mod._normalize_reward(result)
        expected = 99.75 / 100.0
        assert abs(reward - expected) < 1e-9
        assert 0.0 <= reward <= 1.0

    def test_reward_clamped_to_zero_one(self):
        result = {"final_score": 200.0, "max_score": 100.0}
        assert self.mod._normalize_reward(result) == 1.0

        result2 = {"final_score": -5.0, "max_score": 100.0}
        assert self.mod._normalize_reward(result2) == 0.0

    def test_zero_score_returns_zero(self):
        result = {"final_score": 0.0, "max_score": 100.0}
        assert self.mod._normalize_reward(result) == 0.0


def test_verifier_stages_private_instance_data_and_ignores_agent_data(tmp_path, monkeypatch):
    """Guards seed31415 P3 input and prediction persistence after 111b3748."""
    mod = _load_score_entry()
    verifier = tmp_path / "verifier"
    private_data = verifier / "instance_data" / "data"
    private_data.mkdir(parents=True)
    (private_data / "system.json").write_text('{"trusted": true}')
    workspace = tmp_path / "workspace"
    (workspace / "data").mkdir(parents=True)
    (workspace / "data" / "system.json").write_text('{"trusted": false}')
    (workspace / "roots.npy").write_bytes(b"prediction")
    _make_instance_json(tmp_path, requires_instance_data=True)
    _make_output_specs(tmp_path, [{"name": "roots.npy"}])
    monkeypatch.setattr(mod, "VERIFIER_DIR", verifier)
    monkeypatch.setattr(mod, "INSTANCE_JSON", verifier / "instance.json")
    monkeypatch.setattr(mod, "OUTPUT_SPECS_JSON", verifier / "output_specs.json")
    monkeypatch.setattr(mod, "SCORE_DIVISOR_JSON", verifier / "score_divisor.json")
    monkeypatch.setattr(mod, "EVALUATOR_DIR", verifier / "evaluator")
    monkeypatch.setattr(mod, "REFERENCE_DIR", verifier / "reference")
    monkeypatch.setattr(mod, "WORKSPACE_DIR", workspace)
    monkeypatch.setattr(mod, "LOGS_DIR", tmp_path / "logs" / "verifier")

    def fake_score(_evaluator, pred_dir, _reference, _instance):
        assert (pred_dir / "data" / "system.json").read_text() == '{"trusted": true}'
        assert (pred_dir / "roots.npy").read_bytes() == b"prediction"
        return {"final_score": 5.0, "max_score": 10.0,
                "hard_gates_passed": True, "gate_results": [], "score_results": [],
                "scorer_internal_error": False}

    monkeypatch.setattr(mod, "_run_scoring", fake_score)
    assert mod.main() == 0
    assert json.loads((mod.LOGS_DIR / "reward.json").read_text())["reward"] == 0.5
    # Guards P3 parity: BenchFlow deletes the sandbox after scoring, so the
    # verifier must preserve the exact declared outputs in its host-mounted logs.
    assert (mod.LOGS_DIR / "predictions" / "roots.npy").read_bytes() == b"prediction"
    assert not (mod.LOGS_DIR / "predictions" / "data").exists()


def test_missing_private_instance_data_is_verifier_error(tmp_path, monkeypatch):
    """Guards the seed31415 homotopy P3 missing-system.json regression (2026-10-09)."""
    mod = _load_score_entry()
    _make_instance_json(tmp_path, requires_instance_data=True)
    _make_output_specs(tmp_path, [])
    verifier = tmp_path / "verifier"
    monkeypatch.setattr(mod, "INSTANCE_JSON", verifier / "instance.json")
    monkeypatch.setattr(mod, "OUTPUT_SPECS_JSON", verifier / "output_specs.json")
    monkeypatch.setattr(mod, "SCORE_DIVISOR_JSON", verifier / "score_divisor.json")
    monkeypatch.setattr(mod, "EVALUATOR_DIR", verifier / "evaluator")
    monkeypatch.setattr(mod, "REFERENCE_DIR", verifier / "reference")
    monkeypatch.setattr(mod, "WORKSPACE_DIR", tmp_path / "workspace")
    monkeypatch.setattr(mod, "LOGS_DIR", tmp_path / "logs" / "verifier")
    monkeypatch.setattr(mod, "_run_scoring", lambda *_: pytest.fail("scorer must not run"))
    assert mod.main() == 1
    assert not (mod.LOGS_DIR / "reward.json").exists()
    assert json.loads((mod.LOGS_DIR / "asi_error.json").read_text())["failure_kind"] == "missing_evaluator_input"


# ── end-to-end subprocess tests ───────────────────────────────────────────────

def _run_score_entry(tmp_path: Path, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(SCORE_ENTRY)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_missing_instance_json_exits_nonzero_no_reward(tmp_path):
    """Without instance.json the verifier must exit non-zero and write no reward."""
    result = _run_score_entry(tmp_path)
    assert result.returncode != 0
    assert not list(tmp_path.rglob("reward.txt"))
    assert not list(tmp_path.rglob("reward.json"))


def test_wrong_seed_exits_nonzero_no_reward(tmp_path):
    """seed != 31415 is an evaluator error — no reward, exit non-zero."""
    _make_instance_json(tmp_path, seed=42)
    _make_output_specs(tmp_path, [])
    _make_score_divisor(tmp_path)
    result = _run_score_entry(tmp_path)
    assert result.returncode != 0
    logs = tmp_path / "logs" / "verifier"
    assert not (logs / "reward.txt").exists()
    assert not (logs / "reward.json").exists()


def test_missing_evaluator_exits_nonzero_no_reward(tmp_path):
    """Missing evaluator directory → infrastructure error → no reward, exit non-zero."""
    _make_instance_json(tmp_path)
    _make_output_specs(tmp_path, [{"name": "result.csv", "type": "data"}])
    _make_score_divisor(tmp_path)
    # No evaluator/ directory, no task_eval.yaml
    result = _run_score_entry(tmp_path)
    assert result.returncode != 0
    logs = tmp_path / "logs" / "verifier"
    assert not (logs / "reward.txt").exists()
    assert not (logs / "reward.json").exists()
