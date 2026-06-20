"""Tests for the L3 codex equivalence reviewer (.github/scripts/codex_review.py).

Stdlib-only (no ``benchflow`` import at module load): the script is loaded as a
file module exactly like the other .github/scripts tests, so this runs in the L0
``pytest tests/`` lane. The load-bearing properties under test are:

  * worst() is ADVISORY-STRICTER-ONLY (codex can never upgrade).
  * the verdict parser prefers the machine-readable footer and fails closed on
    garbage.
  * the deterministic verdict reader fails closed when verdict.md is missing.
  * auth handling is fail-closed (no key + no auth.json => not authenticated).
  * build_codex_command mirrors agent_router.build_codex_launch_command.
  * main() fails closed (non-zero exit + 'not mergeable (codex unavailable)')
    when the skill is missing, codex auth is absent, or codex output is
    unparseable — never a silent pass.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "codex_review.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("codex_review", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cr = _load_module()


# ------------------------------------------------------------------
# _codex_env — isolate the codex CLI auth from the DeepSeek judge clobber
# ------------------------------------------------------------------


def test_codex_env_prefers_codex_api_key_and_drops_deepseek_base():
    # The review-pack job points OPENAI_* at DeepSeek for the Pass-1 judge; the
    # host codex CLI must instead use the REAL OpenAI key (CODEX_API_KEY) and the
    # default OpenAI endpoint (DeepSeek base URL dropped). The judge model stays.
    out = cr._codex_env(
        {
            "OPENAI_API_KEY": "ds-key",
            "OPENAI_BASE_URL": "https://api.deepseek.com",
            "CODEX_API_KEY": "real-openai-key",
            "BENCHFLOW_JUDGE_MODEL": "openai/deepseek-v4-flash",
        }
    )
    assert out["OPENAI_API_KEY"] == "real-openai-key"
    assert "OPENAI_BASE_URL" not in out
    assert out["BENCHFLOW_JUDGE_MODEL"] == "openai/deepseek-v4-flash"


def test_codex_env_unchanged_without_codex_api_key():
    # Backward compatible: with no CODEX_API_KEY (host where OPENAI_API_KEY is
    # already the real key), the env is returned untouched.
    src = {"OPENAI_API_KEY": "x", "OPENAI_BASE_URL": "y"}
    assert cr._codex_env(src) == src


def test_reasoning_config_sets_effort_override():
    # CODEX_REASONING_EFFORT -> a model_reasoning_effort `-c` override.
    assert cr._reasoning_config({"CODEX_REASONING_EFFORT": "xhigh"}) == [
        'model_reasoning_effort="xhigh"'
    ]


def test_reasoning_config_empty_when_unset():
    # No env -> no override (codex uses its default effort).
    assert cr._reasoning_config({}) == []
    assert cr._reasoning_config({"CODEX_REASONING_EFFORT": "  "}) == []


# ------------------------------------------------------------------
# worst() — advisory-stricter-only composition
# ------------------------------------------------------------------
def test_worst_codex_can_downgrade():
    assert cr.worst("mergeable", "not mergeable") == "not mergeable"
    assert (
        cr.worst("mergeable", "mergeable with quarantines")
        == "mergeable with quarantines"
    )


def test_worst_codex_cannot_upgrade():
    # The core safety property: a deterministic 'not mergeable' stays no matter
    # what codex says.
    assert cr.worst("not mergeable", "mergeable") == "not mergeable"
    assert cr.worst("mergeable with quarantines", "mergeable") == (
        "mergeable with quarantines"
    )


def test_worst_unavailable_is_strictest():
    assert (
        cr.worst("mergeable", "not mergeable (codex unavailable)")
        == "not mergeable (codex unavailable)"
    )
    # An unknown verdict on either side is treated as strictest (fail closed).
    assert cr.worst("mergeable", "garbage") == "garbage"
    assert cr.worst("garbage", "mergeable") == "garbage"


def test_worst_identity():
    assert cr.worst("mergeable", "mergeable") == "mergeable"


# ------------------------------------------------------------------
# _parse_codex_verdict — footer preferred, fail-closed on garbage
# ------------------------------------------------------------------
def test_parse_prefers_verdict_json_footer():
    out = (
        "Verdict: mergeable\n\n"  # prose says mergeable...
        "```verdict-json\n"
        '{"verdict": "not mergeable", "blockers": ["x"]}\n'
        "```\n"
    )
    # ...but the machine-readable footer (not mergeable) is authoritative.
    assert cr._parse_codex_verdict(out) == "not mergeable"


def test_parse_quarantines_phrase():
    assert (
        cr._parse_codex_verdict("Verdict: mergeable with quarantines")
        == "mergeable with quarantines"
    )


def test_parse_plain_verdict_line():
    assert cr._parse_codex_verdict("...\nVerdict: mergeable\n") == "mergeable"


def test_parse_returns_none_on_garbage():
    assert cr._parse_codex_verdict("no verdict anywhere here") is None
    assert cr._parse_codex_verdict("") is None


# ------------------------------------------------------------------
# _read_deterministic_verdict — explicit wins, else verdict.md, else closed
# ------------------------------------------------------------------
def test_read_deterministic_explicit_wins(tmp_path: Path):
    assert (
        cr._read_deterministic_verdict(tmp_path, "mergeable with quarantines")
        == "mergeable with quarantines"
    )


def test_read_deterministic_from_verdict_md(tmp_path: Path):
    pack = tmp_path / "review-pack"
    pack.mkdir()
    (pack / "verdict.md").write_text("## Verdict\n\nnot mergeable\n")
    assert cr._read_deterministic_verdict(pack, None) == "not mergeable"


def test_read_deterministic_missing_is_fail_closed(tmp_path: Path):
    # No explicit flag, no verdict.md => fail closed to 'not mergeable'.
    assert cr._read_deterministic_verdict(tmp_path, None) == "not mergeable"


# ------------------------------------------------------------------
# auth handling — fail-closed
# ------------------------------------------------------------------
def test_has_codex_auth_requires_key_or_file(tmp_path: Path):
    env = {"HOME": str(tmp_path)}
    assert cr.has_codex_auth(env, None) is False
    assert cr.has_codex_auth({"OPENAI_API_KEY": "x", "HOME": str(tmp_path)}, None)
    assert cr.has_codex_auth({"CODEX_API_KEY": "x", "HOME": str(tmp_path)}, None)


def test_write_codex_auth_from_secret(tmp_path: Path):
    env = {"CODEX_HOME": str(tmp_path / ".codex"), "CODEX_AUTH_JSON": '{"k": 1}'}
    path = cr.write_codex_auth(env)
    assert path is not None and path.exists()
    assert json.loads(path.read_text()) == {"k": 1}
    assert cr.has_codex_auth(env, path) is True


def test_write_codex_auth_from_openai_api_key(tmp_path: Path):
    # The durable CI path: OPENAI_API_KEY (no CODEX_AUTH_JSON blob) is written as
    # an apikey auth.json so `codex exec` authenticates with a stable, revocable
    # key instead of a rotating personal-OAuth blob.
    env = {"CODEX_HOME": str(tmp_path / ".codex"), "OPENAI_API_KEY": "sk-test-123"}
    path = cr.write_codex_auth(env)
    assert path is not None and path.exists()
    data = json.loads(path.read_text())
    assert data["OPENAI_API_KEY"] == "sk-test-123"
    assert data["auth_mode"] == "apikey"
    assert cr.has_codex_auth(env, path) is True


def test_write_codex_auth_no_secret_no_key(tmp_path: Path):
    env = {"CODEX_HOME": str(tmp_path / ".codex")}
    path = cr.write_codex_auth(env)
    assert path is None
    assert cr.has_codex_auth(env, path) is False


# ------------------------------------------------------------------
# build_codex_command — mirrors agent_router.build_codex_launch_command
# ------------------------------------------------------------------
def test_build_codex_command_shape(tmp_path: Path):
    cmd = cr.build_codex_command(
        "the prompt", workdir=tmp_path, model="o4", config_overrides=["a=b"]
    )
    assert cmd[0] == "codex"
    assert cmd[1] == "exec"
    assert "--cd" in cmd and str(tmp_path) in cmd
    assert "--skip-git-repo-check" in cmd
    assert cmd[cmd.index("--sandbox") + 1] == "read-only"  # reviewer reads only
    assert cmd[cmd.index("--model") + 1] == "o4"
    assert cmd[cmd.index("-c") + 1] == "a=b"
    assert cmd[-1] == "the prompt"  # prompt is the trailing positional


# ------------------------------------------------------------------
# main() fail-closed paths (no network; fakes injected)
# ------------------------------------------------------------------
def _make_pack(tmp_path: Path, verdict: str = "mergeable") -> Path:
    pack = tmp_path / "review-pack"
    pack.mkdir()
    (pack / "verdict.md").write_text(f"## Verdict\n\n{verdict}\n")
    artifacts = tmp_path / "jobs"
    artifacts.mkdir()
    return pack


def test_main_missing_skill_fails_closed(tmp_path: Path, monkeypatch, capsys):
    pack = _make_pack(tmp_path, "mergeable")
    rc = cr.main(
        [
            "--review-pack",
            str(pack),
            "--artifacts",
            str(tmp_path / "jobs"),
            "--skill",
            str(tmp_path / "nonexistent-SKILL.md"),
        ]
    )
    assert rc == 1
    out = capsys.readouterr().out
    assert "not mergeable (codex unavailable)" in out


def test_main_missing_auth_fails_closed(tmp_path: Path, monkeypatch, capsys):
    pack = _make_pack(tmp_path, "mergeable")
    skill = tmp_path / "SKILL.md"
    skill.write_text("# rubric\n")
    # No deepseek work (no rollouts under artifacts), no codex auth.
    monkeypatch.setattr(cr, "run_deepseek_findings", lambda *a, **k: [])
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_AUTH_JSON", raising=False)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "home" / ".codex"))
    rc = cr.main(
        [
            "--review-pack",
            str(pack),
            "--artifacts",
            str(tmp_path / "jobs"),
            "--skill",
            str(skill),
        ]
    )
    assert rc == 1
    assert "not mergeable (codex unavailable)" in capsys.readouterr().out


def test_main_unparseable_codex_fails_closed(tmp_path: Path, monkeypatch, capsys):
    pack = _make_pack(tmp_path, "mergeable")
    skill = tmp_path / "SKILL.md"
    skill.write_text("# rubric\n")
    monkeypatch.setattr(cr, "run_deepseek_findings", lambda *a, **k: [])
    # Provide auth so we reach the codex step, then return garbage so the parser
    # yields None => fail closed.
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(cr, "run_codex_verdict", lambda *a, **k: (None, "garbage"))
    rc = cr.main(
        [
            "--review-pack",
            str(pack),
            "--artifacts",
            str(tmp_path / "jobs"),
            "--skill",
            str(skill),
        ]
    )
    assert rc == 1
    assert "not mergeable (codex unavailable)" in capsys.readouterr().out


def test_looks_transient_classification():
    # Regression (#794): a transient bwrap sandbox failure must be recognized so
    # codex is retried rather than falsely fail-closed.
    assert cr._looks_transient(
        "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted"
    )
    assert cr._looks_transient("stream error: connection refused")
    assert cr._looks_transient("HTTP 429 rate limit exceeded")
    # A real verdict or plain prose is NOT transient.
    assert not cr._looks_transient("Verdict: not mergeable")
    assert not cr._looks_transient("the rollout looks fine, no issues")
    # "429" as an incidental substring (line/attempt numbers) is NOT a transient
    # rate-limit signal — the marker is bounded, not a bare "429".
    assert not cr._looks_transient("AssertionError at line 429 in module")
    assert not cr._looks_transient("attempt 4290 of the loop")


def test_assemble_prompt_inlines_review_pack(tmp_path: Path):
    # Regression (#794): the review-pack file CONTENTS are inlined into the prompt
    # so codex never needs a sandboxed shell to read them from disk.
    pack = tmp_path / "review-pack"
    pack.mkdir()
    (pack / "verdict.md").write_text("## Verdict\n\nnot mergeable\n")
    (pack / "manifest.json").write_text('{"marker": "MANIFEST_MARKER_XYZ"}')
    prompt = cr._assemble_codex_prompt("SKILL", "TEMPLATE", [{"f": 1}], pack)
    assert "MANIFEST_MARKER_XYZ" in prompt  # manifest.json inlined verbatim
    assert "not mergeable" in prompt  # verdict.md inlined
    assert "----- verdict.md -----" in prompt
    assert "do NOT need to run any shell command" in prompt


def _codex_args(pack: Path, jobs: Path, skill: Path) -> list[str]:
    return ["--review-pack", str(pack), "--artifacts", str(jobs), "--skill", str(skill)]


def test_main_retries_transient_codex_then_succeeds(tmp_path, monkeypatch, capsys):
    # Regression (#794): a transient sandbox failure on the first codex attempt is
    # retried; the recovered verdict is honored instead of "codex unavailable".
    pack = _make_pack(tmp_path, "mergeable")
    skill = tmp_path / "SKILL.md"
    skill.write_text("# rubric\n")
    monkeypatch.setattr(cr, "run_deepseek_findings", lambda *a, **k: [])
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    calls: list[int] = []

    def fake(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            return (
                None,
                "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted",
            )
        return ("mergeable", '```verdict-json\n{"verdict": "mergeable"}\n```')

    monkeypatch.setattr(cr, "run_codex_verdict", fake)
    rc = cr.main(_codex_args(pack, tmp_path / "jobs", skill))
    assert len(calls) == 2  # retried once, then succeeded
    assert "codex_verdict=mergeable" in capsys.readouterr().out
    assert rc == 0


def test_main_persistent_transient_codex_fails_closed(tmp_path, monkeypatch, capsys):
    # If the transient never clears, exhaust the retries and STILL fail closed.
    pack = _make_pack(tmp_path, "mergeable")
    skill = tmp_path / "SKILL.md"
    skill.write_text("# rubric\n")
    monkeypatch.setattr(cr, "run_deepseek_findings", lambda *a, **k: [])
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("CODEX_MAX_ATTEMPTS", "3")
    calls: list[int] = []

    def fake(*a, **k):
        calls.append(1)
        return (None, "bwrap: Operation not permitted")

    monkeypatch.setattr(cr, "run_codex_verdict", fake)
    rc = cr.main(_codex_args(pack, tmp_path / "jobs", skill))
    assert len(calls) == 3  # exhausted the cap
    assert rc == 1
    assert "not mergeable (codex unavailable)" in capsys.readouterr().out


def test_main_nontransient_unparseable_not_retried(tmp_path, monkeypatch, capsys):
    # A non-transient unparseable output is NOT retried — fail closed immediately.
    pack = _make_pack(tmp_path, "mergeable")
    skill = tmp_path / "SKILL.md"
    skill.write_text("# rubric\n")
    monkeypatch.setattr(cr, "run_deepseek_findings", lambda *a, **k: [])
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("CODEX_MAX_ATTEMPTS", "3")
    calls: list[int] = []

    def fake(*a, **k):
        calls.append(1)
        return (None, "plain prose, no verdict and no transient marker")

    monkeypatch.setattr(cr, "run_codex_verdict", fake)
    rc = cr.main(_codex_args(pack, tmp_path / "jobs", skill))
    assert len(calls) == 1  # not retried
    assert rc == 1
    assert "not mergeable (codex unavailable)" in capsys.readouterr().out


def test_main_codex_downgrade_is_honored(tmp_path: Path, monkeypatch, capsys):
    # Deterministic mergeable + codex 'not mergeable' => final not mergeable, rc 1.
    pack = _make_pack(tmp_path, "mergeable")
    skill = tmp_path / "SKILL.md"
    skill.write_text("# rubric\n")
    monkeypatch.setattr(cr, "run_deepseek_findings", lambda *a, **k: [])
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(
        cr, "run_codex_verdict", lambda *a, **k: ("not mergeable", "raw")
    )
    rc = cr.main(
        [
            "--review-pack",
            str(pack),
            "--artifacts",
            str(tmp_path / "jobs"),
            "--skill",
            str(skill),
        ]
    )
    assert rc == 1
    assert "final_verdict=not mergeable" in capsys.readouterr().out


def test_main_codex_cannot_upgrade_not_mergeable(tmp_path: Path, monkeypatch, capsys):
    # Deterministic not mergeable + codex 'mergeable' => final stays not mergeable.
    pack = _make_pack(tmp_path, "not mergeable")
    skill = tmp_path / "SKILL.md"
    skill.write_text("# rubric\n")
    monkeypatch.setattr(cr, "run_deepseek_findings", lambda *a, **k: [])
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(cr, "run_codex_verdict", lambda *a, **k: ("mergeable", "raw"))
    rc = cr.main(
        [
            "--review-pack",
            str(pack),
            "--artifacts",
            str(tmp_path / "jobs"),
            "--skill",
            str(skill),
        ]
    )
    assert rc == 1
    out = capsys.readouterr().out
    assert "final_verdict=not mergeable" in out


def test_main_all_mergeable_passes(tmp_path: Path, monkeypatch, capsys):
    pack = _make_pack(tmp_path, "mergeable")
    skill = tmp_path / "SKILL.md"
    skill.write_text("# rubric\n")
    monkeypatch.setattr(cr, "run_deepseek_findings", lambda *a, **k: [])
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(cr, "run_codex_verdict", lambda *a, **k: ("mergeable", "raw"))
    rc = cr.main(
        [
            "--review-pack",
            str(pack),
            "--artifacts",
            str(tmp_path / "jobs"),
            "--skill",
            str(skill),
        ]
    )
    assert rc == 0
    assert "final_verdict=mergeable" in capsys.readouterr().out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
