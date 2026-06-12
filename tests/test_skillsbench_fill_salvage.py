from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_salvage_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "experiments" / "skillsbench-fill" / "salvage_timeout_partials.py"
    spec = importlib.util.spec_from_file_location("skillsbench_fill_salvage", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data)
    else:
        path.write_text(json.dumps(data))


def _extract_script(path: Path) -> Path:
    script = path / "extract.py"
    script.write_text(
        "import json\n"
        "print(json.dumps({'task_skills_loading': 0, "
        "'task_skills_loading_status': 'not_expected_without_skills'}))\n"
    )
    return script


def test_salvage_rejects_timeout_with_final_provider_error(tmp_path: Path) -> None:
    """Guards PR #638 follow-up against crediting provider-failed timeouts."""
    salvage = _load_salvage_module()
    hf_path = "root/citation-check__abc"
    root = tmp_path / hf_path
    _write(
        root / "result.json",
        {
            "error": "Agent timed out after 900s",
            "error_category": "timeout",
            "rewards": {"reward": 0.0},
            "timing": {"total": 901.0},
            "agent_result": {"total_tokens": 123},
            "trajectory_summary": {"partial_trajectory": True},
        },
    )
    _write(root / "config.json", {"environment": "daytona"})
    _write(root / "timing.json", {"total": 901.0})
    _write(
        root / "trajectory/acp_trajectory.jsonl",
        '{"type":"tool_call","status":"completed"}\n',
    )
    _write(
        root / "trajectory/llm_trajectory.jsonl",
        json.dumps({"response": {"status_code": 200}})
        + "\n"
        + json.dumps({"response": {"status_code": 500}})
        + "\n",
    )

    def fake_download(_repo, path_in_repo, **_kwargs):
        return str(tmp_path / path_in_repo)

    rec = salvage.inspect_row(
        {
            "cell_id": "minimax-m3__without__citation-check__t1",
            "hf_path": hf_path,
            "model": "minimax-m3",
            "skill_mode": "without",
            "task": "citation-check",
            "trial_slot": 1,
            "error": "Agent timed out after 900s",
            "sandbox": "daytona",
        },
        hf_hub_download=fake_download,
        token=None,
        extract_script=str(_extract_script(tmp_path)),
        tasks_root=str(tmp_path),
    )

    assert rec["llm_last_status_code"] == 500
    assert rec["llm_non_2xx_response_count"] == 1
    assert "llm_final_response_ok" in rec["failed_checks"]
    assert rec["timeout_complete_artifacts"] is False
