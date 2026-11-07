"""OpenCode JSONC must retain the same resume identity as its ordered JSON."""

import json
import logging

import pytest

from benchflow._utils.config_redaction import (
    _inline_config_fingerprint,
    _inline_config_structure,
    _should_record_env_entry,
)
from benchflow.evaluation import (
    ResumeMismatchError,
    _check_resume_mismatch,
    _recorded_run_differs,
)
from tests.test_resume_settings_match import (
    _config,
    _evaluation,
    _mark_reviewed,
    _run,
)

NAME = "OPENCODE_CONFIG_CONTENT"
FORMATS = ["line-lf", "line-cr", "line-crlf", "block", "trailing", "combined"]


def _jsonc(node, syntax):
    """Render fixture objects with trivia at every structural boundary."""
    if syntax == "json":
        return json.dumps(node)
    comment = {
        "line-lf": "// policy\n",
        "line-cr": "// policy\r",
        "line-crlf": "// policy\r\n",
        "block": "/* policy\n, } ] // */",
        "trailing": " ",
        "combined": "/* policy */ // policy\r\n",
    }[syntax]
    trailing = syntax in {"trailing", "combined"}

    def render(value):
        if isinstance(value, dict):
            parts = [
                json.dumps(k) + comment + ":" + comment + render(v)
                for k, v in value.items()
            ]
            left, right = "{", "}"
        elif isinstance(value, list):
            parts = [render(v) for v in value]
            left, right = "[", "]"
        else:
            return json.dumps(value)
        body = ("," + comment).join(parts)
        return (
            left
            + comment
            + body
            + ("," if parts and trailing else "")
            + comment
            + right
        )

    return comment + render(node) + comment


@pytest.mark.parametrize("syntax", FORMATS)
@pytest.mark.parametrize("owner", ["solver", "reviewer", "override"])
@pytest.mark.parametrize("scope", [(), ("agent", "build"), ("mode", "build")])
@pytest.mark.parametrize("policy", ["tools", "permission"])
@pytest.mark.parametrize(
    "backend",
    ["docker", "remote-docker", "daytona", "modal", "apple-container", "agentcore"],
)
def test_jsonc_policy_resume(tmp_path, caplog, syntax, owner, scope, policy, backend):
    """Guards P2-CONFIRM-4 at ade564d3 across formats, paths and backends."""
    switches = (
        {"write": True, "edit": False}
        if policy == "tools"
        else {"*": "deny", "edit": "allow"}
    )

    def config(values, marker, form):
        node = {policy: values}
        for key in reversed(scope):
            node = {key: node}
        node["provider"] = {"local": {"options": {"apiKey": marker}}}
        node["instructions"] = ["policy.md"]
        env = {NAME: _jsonc(node, form)}
        if owner == "solver":
            return _config(environment=backend, agent_env=env)
        if owner == "reviewer":
            return _config(environment=backend, reviewer={"agent_env": env})
        return _config(environment=backend, config_override={"sandbox": {"env": env}})

    first = _evaluation(
        tmp_path, config(switches, "24681357", syntax), job_name="named"
    )
    assert _run(first).ran == 2
    _mark_reviewed(first.job_dir)
    # Rotation and trivia-only changes preserve matching on both resume paths.
    same = config(switches, "13572468", syntax)
    assert _recorded_run_differs(first.job_dir, same, None) is None
    assert _evaluation(tmp_path, same).job_dir == first.job_dir
    assert _check_resume_mismatch(first.job_dir, same) is None
    changed = config(dict(reversed(switches.items())), "13572468", syntax)
    assert _recorded_run_differs(first.job_dir, changed, None) is not None
    assert _evaluation(tmp_path, changed).job_dir != first.job_dir
    same_json = config(switches, "13572468", "json")
    assert _recorded_run_differs(first.job_dir, same_json, None) is None
    assert _evaluation(tmp_path, same_json).job_dir == first.job_dir
    assert _check_resume_mismatch(first.job_dir, same_json) is None
    with caplog.at_level(logging.WARNING, logger="benchflow.evaluation"):
        dest = _check_resume_mismatch(first.job_dir, changed, restart_changed=True)
    assert dest is not None
    reason = (dest / "set-aside.json").read_text()
    assert NAME in reason
    # Solver/reviewer env artifacts omit the raw credential-bearing value.
    # config_override persistence is separate from its comparison policy.
    record = (
        (first.job_dir / "evaluation.json").read_text() if owner != "override" else ""
    )
    for marker in ("24681357", "13572468"):
        assert marker not in record + reason + caplog.text


@pytest.mark.parametrize("syntax", FORMATS)
def test_jsonc_strings_and_credentials(syntax):
    """Guards P2-CONFIRM-4 at ade564d3: JSONC trivia cannot alter strings."""
    public = {
        "instructions": [
            "https://example.invalid/a/*b*/?x=,}",
            'quote" slash\\ // /* ,]',
        ],
        "tools": {"write": True, "edit": False},
        "provider": {"local": {"options": {}}},
        "mcp": {"docs": {"environment": {}, "headers": {}}},
    }
    original = json.loads(json.dumps(public))
    original["provider"]["local"]["options"]["apiKey"] = "24681357"
    original["mcp"]["docs"]["environment"]["ACCESS_TOKEN"] = "24681357"
    original["mcp"]["docs"]["headers"]["Authorization"] = "Bearer 24681357"
    raw = _jsonc(original, syntax)
    structure = _inline_config_structure(NAME, raw)
    assert structure == (original, public)
    assert list(structure[0]["tools"]) == ["write", "edit"]
    assert _inline_config_fingerprint(NAME, raw) == _inline_config_fingerprint(
        NAME, json.dumps(public)
    )
    assert not _should_record_env_entry(NAME, raw)
    assert _should_record_env_entry(NAME, _jsonc(public, syntax))
    assert _inline_config_structure("CODEX_CONFIG", raw) is None


@pytest.mark.parametrize(
    "comment", ['// "apiKey": "24681357"\n', '/* "apiKey": "24681357" */']
)
def test_jsonc_comment_credentials_are_never_recorded(comment):
    """Guards P2-CONFIRM-4 at ade564d3: ignored comments remain private."""
    raw = comment + '{"tools":{"write":true,"edit":false}}'
    assert not _should_record_env_entry(NAME, raw)
    assert _inline_config_fingerprint(NAME, raw) == _inline_config_fingerprint(
        NAME, raw[len(comment) :]
    )


@pytest.mark.parametrize(
    "raw",
    [
        "{,}",
        '{"tools":,}',
        '{"tools":[,]}',
        '{"tools":[true,,]}',
        '{"tools":true false}',
        '{"tools":1/*gap*/2}',
        '{"tools":tru/*gap*/e}',
        '{"tools":true} /* unclosed',
        '{"tools":"unclosed}',
        '{"tools":"bad\\q"}',
        "{'tools':true}",
        "{tools:true}",
        '{"tools":True}',
        '{"tools":true} {}',
        '{"tools":[true,}',
    ],
)
def test_jsonc_invalid_syntax_is_not_repaired(raw):
    """Guards P2-CONFIRM-4 at ade564d3: only JSONC extensions are accepted."""
    assert _inline_config_structure(NAME, raw) is None


@pytest.mark.parametrize("owner", ["solver", "reviewer"])
@pytest.mark.parametrize("has_secret", [False, True])
@pytest.mark.parametrize("syntax", FORMATS)
def test_jsonc_legacy_record_needs_known_policy(tmp_path, owner, has_secret, syntax):
    """Guards P2-CONFIRM-4 at ade564d3 when old records lost JSONC policy."""
    node = {"tools": {"write": True, "edit": False}}
    if has_secret:
        node["provider"] = {"local": {"options": {"apiKey": "24681357"}}}
    env = {NAME: _jsonc(node, syntax)}
    config = (
        _config(agent_env=env)
        if owner == "solver"
        else _config(reviewer={"agent_env": env})
    )
    first = _evaluation(tmp_path, config, job_name="named")
    assert _run(first).ran == 2
    _mark_reviewed(first.job_dir)
    path = first.job_dir / "evaluation.json"
    record = json.loads(path.read_text())
    role = record["config"] if owner == "solver" else record["config"]["reviewer"]
    role.pop("agent_env_fingerprints", None)
    if has_secret:
        # The old parser could drop the whole value, retaining only its name.
        role["agent_env"].pop(NAME, None)
    path.write_text(json.dumps(record))
    if has_secret:
        assert _recorded_run_differs(first.job_dir, config, None) is not None
        assert _evaluation(tmp_path, config).job_dir != first.job_dir
        with pytest.raises(ResumeMismatchError):
            _check_resume_mismatch(first.job_dir, config)
    else:
        # A public legacy value is sufficient to reconstruct its identity.
        assert _recorded_run_differs(first.job_dir, config, None) is None
        assert _evaluation(tmp_path, config).job_dir == first.job_dir
        assert _check_resume_mismatch(first.job_dir, config) is None
