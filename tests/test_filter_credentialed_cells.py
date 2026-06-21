"""Tests for the env-aware credential filter (filter_credentialed_cells.py).

The credential filter is the workflow step that runs BETWEEN the pure planner
and run-matrix. It drops any planned cell whose agent+model required credential
is absent so an un-keyed agent never spins up a (doomed, sandbox-burning) cell
that the grader would log as a false-red slot.

Two layers are covered:

* The benchflow-backed path (guarded by ``importorskip`` exactly like the other
  production-path tests): with DeepSeek keyed and Bedrock UNkeyed, a deepseek
  cell is KEPT and a Bedrock/Claude cell is DROPPED into
  ``skipped_uncredentialed`` carrying ``missing_key == AWS_BEARER_TOKEN_BEDROCK``.
* The fail-open pass-through path: when ``benchflow`` cannot be imported the
  matrix passes through UNCHANGED — exercised by faking the import failure, so
  it runs whether or not benchflow is installed.

Credentials are controlled via ``os.environ`` and restored after each test.
``BENCHFLOW_DOTENV_PATH`` is pointed at a non-existent file so a developer's
local ``.env`` (which may carry AWS_BEARER_TOKEN_BEDROCK) cannot leak in and mask
the missing-credential drop.
"""

from __future__ import annotations

import builtins
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "filter_credentialed_cells.py"

DEEPSEEK_MODEL = "deepseek/deepseek-v4-flash"
BEDROCK_MODEL = "aws-bedrock/us.anthropic.claude-haiku-4-5-20251001"


def _load_module():
    """Import the script by path (it lives outside the importable package)."""
    spec = importlib.util.spec_from_file_location(
        "filter_credentialed_cells", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fc = _load_module()


@pytest.fixture
def isolate_dotenv(monkeypatch, tmp_path):
    """Stop the repo .env from supplying credentials under test."""
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "nonexistent.env"))


def _sample_plan() -> dict:
    """A plan with one deepseek cell and one bedrock/claude cell."""
    return {
        "schema_version": "1",
        "scope": "all-agents",
        "matrix": [
            {
                "id": "openhands-deepseek",
                "agent": "openhands",
                "model": DEEPSEEK_MODEL,
                "task": "hello-world",
            },
            {
                "id": "claude-bedrock",
                "agent": "claude-agent-acp",
                "model": BEDROCK_MODEL,
                "task": "hello-world",
            },
        ],
    }


# ---------------------------------------------------------------------------
# benchflow-backed path: real resolve_agent_env credential resolution.
# ---------------------------------------------------------------------------


def test_drops_uncredentialed_keeps_credentialed(monkeypatch, isolate_dotenv):
    """DeepSeek keyed + Bedrock unkeyed -> deepseek KEPT, bedrock DROPPED."""
    pytest.importorskip("benchflow.agents.env")

    # DeepSeek proxy creds present; AWS Bedrock bearer token absent.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)

    filtered = fc.filter_matrix(_sample_plan())

    kept = filtered["matrix"]
    skipped = filtered["skipped_uncredentialed"]

    kept_agents = {c["agent"] for c in kept}
    assert "openhands" in kept_agents, "credentialed deepseek cell must survive"
    assert "claude-agent-acp" not in kept_agents, (
        "uncredentialed bedrock cell must be dropped from the run matrix"
    )

    assert len(skipped) == 1
    dropped = skipped[0]
    assert dropped["agent"] == "claude-agent-acp"
    assert dropped["model"] == BEDROCK_MODEL
    assert dropped["missing_key"] == "AWS_BEARER_TOKEN_BEDROCK"


def test_keeps_all_when_every_credential_present(monkeypatch, isolate_dotenv):
    """With Bedrock ALSO keyed, nothing is skipped."""
    pytest.importorskip("benchflow.agents.env")

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-token-test")

    filtered = fc.filter_matrix(_sample_plan())

    assert filtered["skipped_uncredentialed"] == []
    assert {c["agent"] for c in filtered["matrix"]} == {
        "openhands",
        "claude-agent-acp",
    }


def test_drops_deepseek_when_base_url_missing(monkeypatch, isolate_dotenv):
    """filter_matrix DROPS a cell when resolve_agent_env raises the base-URL
    builder message ("... requires DEEPSEEK_BASE_URL to build the provider base
    URL") — the shape it raises when DEEPSEEK_BASE_URL is absent, BEFORE the
    api-key "not set" check. Since openhands (deepseek) is the baseline agent,
    missing this would false-red almost every run instead of logging a skip.

    resolve_agent_env is patched so the missing-base-URL condition is exercised
    DETERMINISTICALLY: whether a bare DEEPSEEK_BASE_URL is truly absent depends on
    the ambient env (a populated CI job can supply a fallback base), which is
    benchflow's concern, not the filter's. This pins the filter's own routing of
    that ValueError shape to a documented drop + correct missing_key.
    """
    pytest.importorskip("benchflow.agents.env")
    import benchflow.agents.env as bfenv

    real = bfenv.resolve_agent_env

    def fake_resolve(agent, model, agent_env):
        if model == DEEPSEEK_MODEL:
            raise ValueError(
                f"Provider 'deepseek' for model {model!r} requires "
                "DEEPSEEK_BASE_URL to build the provider base URL."
            )
        return real(agent, model, agent_env)

    # filter_matrix re-imports the name from the module on each call, so patching
    # the module attribute is picked up.
    monkeypatch.setattr(bfenv, "resolve_agent_env", fake_resolve)
    # Keep the non-deepseek (bedrock) cell credentialed so it survives.
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-token-test")

    filtered = fc.filter_matrix(_sample_plan())

    kept_agents = {c["agent"] for c in filtered["matrix"]}
    assert "openhands" not in kept_agents, (
        "deepseek cell with no DEEPSEEK_BASE_URL must be a documented skip, not red"
    )
    dropped = {s["agent"]: s["missing_key"] for s in filtered["skipped_uncredentialed"]}
    assert dropped.get("openhands") == "DEEPSEEK_BASE_URL"
    # Bedrock cell resolves cleanly, so claude survives.
    assert "claude-agent-acp" in kept_agents


def test_cli_writes_filtered_matrix(monkeypatch, isolate_dotenv, tmp_path):
    """End-to-end CLI: reads matrix.json, writes the filtered plan back."""
    pytest.importorskip("benchflow.agents.env")

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)

    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(_sample_plan()), encoding="utf-8")

    rc = fc.main(["--matrix", str(matrix_path), "--out", str(matrix_path)])
    assert rc == 0

    out = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert [c["agent"] for c in out["matrix"]] == ["openhands"]
    assert out["skipped_uncredentialed"][0]["missing_key"] == (
        "AWS_BEARER_TOKEN_BEDROCK"
    )
    # Non-matrix top-level keys are preserved verbatim.
    assert out["scope"] == "all-agents"


# ---------------------------------------------------------------------------
# fail-open path: benchflow import unavailable -> pass through unchanged.
# ---------------------------------------------------------------------------


def test_pass_through_when_benchflow_absent(monkeypatch):
    """If benchflow cannot be imported, the matrix is returned UNCHANGED."""
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "benchflow.agents.env" or name.startswith("benchflow"):
            raise ModuleNotFoundError("No module named 'benchflow'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    plan = _sample_plan()
    filtered = fc.filter_matrix(plan)

    # Every cell preserved; nothing dropped; empty skip list added.
    assert [c["agent"] for c in filtered["matrix"]] == [
        "openhands",
        "claude-agent-acp",
    ]
    assert filtered["skipped_uncredentialed"] == []
    # Original input is not mutated in place.
    assert "skipped_uncredentialed" not in plan


def test_extract_missing_key_shapes():
    """The missing-key extractor handles every real message shape."""
    bedrock_msg = (
        "AWS_BEARER_TOKEN_BEDROCK required for Bedrock model 'x' but not set. "
        "Export it or pass via agent_env."
    )
    generic_msg = (
        "GEMINI_API_KEY required for model 'y' but not set. Pass it explicitly."
    )
    # Missing base-URL env: "requires" (not "required") + "to build ... base URL".
    base_url_msg = (
        "Provider 'deepseek' for model 'deepseek/deepseek-v4-flash' requires "
        "DEEPSEEK_BASE_URL to build the provider base URL."
    )
    azure_msg = (
        "Azure AI Foundry model 'z' requires AZURE_OPENAI_RESOURCE or "
        "AZURE_OPENAI_ENDPOINT to build the provider base URL. Export ..."
    )
    assert fc._is_missing_credential(bedrock_msg)
    assert fc._is_missing_credential(generic_msg)
    assert fc._is_missing_credential(base_url_msg)
    assert fc._is_missing_credential(azure_msg)
    assert fc._extract_missing_key(bedrock_msg) == "AWS_BEARER_TOKEN_BEDROCK"
    assert fc._extract_missing_key(generic_msg) == "GEMINI_API_KEY"
    assert fc._extract_missing_key(base_url_msg) == "DEEPSEEK_BASE_URL"
    # The first listed env stands in for the "X or Y" alternatives.
    assert fc._extract_missing_key(azure_msg) == "AZURE_OPENAI_RESOURCE"

    # An unrelated ValueError is NOT a missing-credential signal.
    assert not fc._is_missing_credential("provider 'vllm' base url is malformed")
