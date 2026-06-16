import importlib.util
import os
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".github/scripts/select_integration_provider.py"
)
spec = importlib.util.spec_from_file_location("select_integration_provider", SCRIPT)
assert spec is not None
selector = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = selector
spec.loader.exec_module(selector)


def test_select_candidate_picks_first_successful_provider():
    env = {"DEEPSEEK_API_KEY": "deepseek-key", "GLM_API_KEY": "glm-key"}
    seen = []

    def probe(candidate, _env):
        seen.append(candidate.name)
        return candidate.name == "glm", "ok" if candidate.name == "glm" else "nope"

    selected, attempts = selector.select_candidate(selector.candidates(env), env, probe)

    assert selected.name == "glm"
    assert seen == ["deepseek", "glm"]
    assert attempts == [("deepseek", "nope"), ("glm", "ok")]


def test_deepseek_exports_openai_compatible_judge_env():
    env = {
        "DEEPSEEK_API_KEY": "deepseek-key",
        "DEEPSEEK_BASE_URL": "https://api.deepseek.example",
    }

    candidate = selector.candidates(env)[0]

    assert candidate.name == "deepseek"
    assert candidate.rollout_model == "deepseek/deepseek-v4-flash"
    assert candidate.judge_model == "openai/deepseek-v4-flash"
    assert candidate.exports["DEEPSEEK_API_KEY"] == "deepseek-key"
    assert candidate.exports["OPENAI_API_KEY"] == "deepseek-key"
    assert candidate.exports["OPENAI_BASE_URL"] == "https://api.deepseek.example"


def test_github_env_writer_uses_multiline_format(tmp_path, monkeypatch):
    path = tmp_path / "github_env"
    monkeypatch.setenv("GITHUB_ENV", str(path))

    selector._append_github_env({"OPENAI_API_KEY": "secret\nvalue"}, os.environ)

    assert path.read_text() == (
        "OPENAI_API_KEY<<__BENCHFLOW_ENV__\nsecret\nvalue\n__BENCHFLOW_ENV__\n"
    )
