"""Guard unsupported custom-tool captures against silent loss in 16ae05afd."""

import copy
import json

import pytest

from benchflow.trajectories.export_prime_sft import (
    export_prime_sft_jsonl,
    normalize_prime_sft_exchange,
)
from benchflow.trajectories.export_trl_sft import export_trl_sft_jsonl
from benchflow.trajectories.results import (
    _training_success_exchange_indices,
    build_rollout_results_record,
)


def _text(text="The check passed."):
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def _custom_call():
    return {
        "type": "custom_tool_call",
        "call_id": "custom-call",
        "name": "exec",
        "input": 'console.log("checked");\n',
    }


def _custom_result():
    return {
        "type": "custom_tool_call_output",
        "call_id": "custom-call",
        "output": "checked",
    }


def _function_call():
    return {
        "type": "function_call",
        "call_id": "function-call",
        "name": "exec",
        "arguments": '{"command":"echo checked"}',
    }


def _exchange(kind="text"):
    body = {
        "model": "test-model",
        "input": [{"role": "user", "content": "Check the answer."}],
        "tools": [
            {
                "type": "function",
                "name": "exec",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                },
            }
        ],
    }
    output = [_text()]
    if kind == "function":
        output = [_function_call()]
    elif kind == "custom":
        output = [_custom_call()]
    elif kind == "mixed":
        output.append(_custom_call())
    elif kind in {"input_history", "messages_history", "output_history"}:
        body["input"].extend(
            [_custom_result()]
            if kind == "output_history"
            else [_custom_call(), _custom_result()]
        )
        if kind == "messages_history":
            body["messages"] = body.pop("input")
    elif kind == "custom_declaration":
        body["tools"] = [
            {
                "type": "custom",
                "name": "payload-name-must-not-leak",
                "format": {"type": "text"},
            }
        ]
    return {
        "request": {"body": body},
        "response": {
            "status_code": 200,
            "body": {"status": "completed", "output": output},
        },
    }


def _rollout(tmp_path, records):
    trial = tmp_path / "task__trial"
    (trial / "trajectory").mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps({"task_name": "task", "rewards": {"reward": 1}})
    )
    capture = trial / "trajectory/llm_trajectory.jsonl"
    capture.write_text("\n".join(map(json.dumps, records)))
    return trial, capture


@pytest.mark.parametrize(
    "kind", ["custom", "mixed", "input_history", "messages_history", "output_history"]
)
def test_custom_protocol_refused_before_normalization(kind):
    """Guards 16ae05afd: custom strings cannot silently become function JSON or text."""
    record = _exchange(kind)
    original = copy.deepcopy(record)
    normalized, reason = normalize_prime_sft_exchange(record)
    assert normalized is None
    assert reason.startswith("unsupported_tool_protocol:")
    assert "custom_tool_call" in reason
    assert "console.log" not in reason
    assert record == original


@pytest.mark.parametrize(
    "kind", ["custom", "mixed", "input_history", "custom_declaration"]
)
def test_results_remain_untrainable_with_explicit_custom_protocol_error(tmp_path, kind):
    """Guards 16ae05afd: unsupported selected calls keep canonical results untrainable."""
    trial, capture = _rollout(tmp_path, [_exchange(kind)])
    original = capture.read_bytes()
    row = build_rollout_results_record(
        trial,
        task_name="task",
        rollout_name="trial",
        agent="codex-acp",
        agent_name="codex-acp",
        model="test-model",
        n_tool_calls=1,
        prompts=["Check the answer."],
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 1},
        error=None,
        verifier_error=None,
    )
    assert row["info"]["training_ready"] is False
    assert row["info"]["training_ready_reason"] == "export_error"
    assert "unsupported_tool_protocol:" in row["error"]["error_chain_str"]
    assert row["trajectory"] == []
    assert capture.read_bytes() == original


@pytest.mark.parametrize("export", [export_prime_sft_jsonl, export_trl_sft_jsonl])
@pytest.mark.parametrize("kind", ["custom", "mixed", "input_history"])
@pytest.mark.parametrize("row_mode", ["exchange", "rollout"])
@pytest.mark.parametrize("existing_output", [False, True])
def test_directory_export_refuses_before_publishing_partial_file(
    tmp_path, export, kind, row_mode, existing_output
):
    """Guards 16ae05afd: a later unsupported call must not publish earlier text rows."""
    prior = _exchange("text")
    prior["request"]["body"]["input"][0]["content"] = "Earlier independent prompt."
    _, capture = _rollout(tmp_path, [prior, _exchange(kind)])
    original = capture.read_bytes()
    out = tmp_path / "out.jsonl"
    manifest = tmp_path / "manifest.json"
    if existing_output:
        out.write_text("previous output\n")
        manifest.write_text("previous manifest\n")
    with pytest.raises(ValueError, match="unsupported_tool_protocol:"):
        export(tmp_path, out, manifest=manifest, row_mode=row_mode)
    if existing_output:
        assert out.read_text() == "previous output\n"
        assert manifest.read_text() == "previous manifest\n"
    else:
        assert not out.exists()
        assert not manifest.exists()
    assert capture.read_bytes() == original


@pytest.mark.parametrize("kind", ["text", "function"])
def test_supported_protocols_and_literal_custom_type_text_still_normalize(kind):
    """Guards 16ae05afd: refusal examines protocol items, not words in user content."""
    record = _exchange(kind)
    record["request"]["body"]["input"][0]["content"] = (
        "Explain custom_tool_call_output."
    )
    record["response"]["body"]["output"].insert(0, {"type": "reasoning", "summary": []})
    normalized, reason = normalize_prime_sft_exchange(record)
    assert reason is None
    assert normalized is not None
    assert bool(normalized.messages[-1].get("tool_calls")) is (kind == "function")


@pytest.mark.parametrize("field", ["tools", "tool_defs"])
def test_unused_custom_tool_declaration_is_not_rewritten_as_a_function(field):
    """Guards 16ae05afd: even a text response was conditioned on the declared custom tool."""
    record = _exchange("custom_declaration")
    if field == "tool_defs":
        body = record["request"]["body"]
        body[field] = body.pop("tools")
    original = copy.deepcopy(record)
    normalized, reason = normalize_prime_sft_exchange(record)
    assert normalized is None
    assert reason == "unsupported_tool_protocol: custom"
    assert record == original


@pytest.mark.parametrize("export", [export_prime_sft_jsonl, export_trl_sft_jsonl])
@pytest.mark.parametrize("field", ["tools", "tool_defs"])
def test_directory_export_refuses_unused_custom_declaration(tmp_path, export, field):
    """Guards 16ae05afd: a declared custom tool cannot become a published function schema."""
    record = _exchange("custom_declaration")
    if field == "tool_defs":
        body = record["request"]["body"]
        body[field] = body.pop("tools")
    _rollout(tmp_path, [record])
    out = tmp_path / "out.jsonl"
    with pytest.raises(ValueError, match="unsupported_tool_protocol: custom"):
        export(tmp_path, out)
    assert not out.exists()


@pytest.mark.parametrize(
    "definition",
    [
        {"type": "function", "name": "exec", "parameters": {"type": "object"}},
        {
            "type": "function",
            "function": {"name": "exec", "parameters": {"type": "object"}},
        },
        {"name": "exec", "input_schema": {"type": "object"}},
    ],
)
def test_supported_function_and_anthropic_definitions_remain_accepted(definition):
    """Guards 16ae05afd: the custom refusal preserves supported tool declaration forms."""
    record = _exchange("text")
    record["request"]["body"]["tools"] = [definition]
    normalized, reason = normalize_prime_sft_exchange(record)
    assert reason is None
    assert normalized is not None
    assert normalized.tool_defs[0]["function"]["name"] == "exec"


def _nested_custom_declaration(carrier):
    record = _exchange("custom_declaration")
    body = record["request"]["body"]
    tools = [{"type": "namespace", "name": "functions", "tools": body.pop("tools")}]
    if carrier in {"tools", "tool_defs"}:
        body[carrier] = tools
    else:
        body["input"].insert(0, {"type": "additional_tools", "tools": tools})
        if carrier == "messages":
            body["messages"] = body.pop("input")
    return record


@pytest.mark.parametrize("carrier", ["tools", "tool_defs", "input", "messages"])
def test_custom_declarations_in_typed_namespaces_are_refused(carrier):
    """Guards 16ae05afd: namespaced and additional_tools declarations retain their type."""
    record = _nested_custom_declaration(carrier)
    original = copy.deepcopy(record)
    normalized, reason = normalize_prime_sft_exchange(record)
    assert normalized is None
    assert reason == "unsupported_tool_protocol: custom"
    assert record == original


@pytest.mark.parametrize("export", [export_prime_sft_jsonl, export_trl_sft_jsonl])
@pytest.mark.parametrize("carrier", ["tools", "tool_defs", "input", "messages"])
def test_directory_export_refuses_nested_custom_declaration(tmp_path, export, carrier):
    """Guards 16ae05afd: namespace declarations cannot vanish from a published text row."""
    _rollout(tmp_path, [_nested_custom_declaration(carrier)])
    out = tmp_path / "out.jsonl"
    with pytest.raises(ValueError, match="unsupported_tool_protocol: custom"):
        export(tmp_path, out)
    assert not out.exists()


@pytest.mark.parametrize("carrier", ["message", "arguments", "schema_default"])
def test_quoted_or_payload_custom_declarations_remain_data(carrier):
    """Guards 16ae05afd: inspect only typed declaration lists, never arbitrary payloads."""
    data = {
        "type": "additional_tools",
        "tools": [
            {"type": "namespace", "tools": [{"type": "custom", "name": "quoted"}]}
        ],
    }
    record = _exchange("function" if carrier == "arguments" else "text")
    if carrier == "message":
        record["request"]["body"]["input"][0]["content"] = [
            {"type": "text", "text": json.dumps(data)}
        ]
    elif carrier == "arguments":
        record["response"]["body"]["output"][0]["arguments"] = json.dumps(data)
    else:
        record["request"]["body"]["tools"][0]["parameters"]["default"] = data
    normalized, reason = normalize_prime_sft_exchange(record)
    assert reason is None
    assert normalized is not None
    if carrier == "arguments":
        call = normalized.messages[-1]["tool_calls"][0]
        assert json.loads(call["function"]["arguments"]) == data
    elif carrier == "schema_default":
        assert normalized.tool_defs[0]["function"]["parameters"]["default"] == data


@pytest.mark.parametrize("export", [export_prime_sft_jsonl, export_trl_sft_jsonl])
def test_discarded_custom_retry_does_not_block_supported_consumed_calls(
    tmp_path, export
):
    """Guards 16ae05afd: an unconsumed retry outside the selected calls stays excluded."""
    first = _exchange("function")
    followup = _exchange("text")
    followup["request"]["body"]["input"].extend(
        [
            _function_call(),
            {
                "type": "function_call_output",
                "call_id": "function-call",
                "output": "checked",
            },
        ]
    )
    retry = _exchange("custom")
    records = [first, followup, retry]
    assert _training_success_exchange_indices(records) == {0, 1}
    _rollout(tmp_path, records)
    out = tmp_path / "out.jsonl"
    stats = export(tmp_path, out, row_mode="exchange")
    assert stats.rows_written == 2
    assert len(out.read_text().splitlines()) == 2


@pytest.mark.parametrize("export", [export_prime_sft_jsonl, export_trl_sft_jsonl])
def test_incomplete_custom_response_stays_outside_selection(tmp_path, export):
    """Guards 16ae05afd: an unsuccessful response is not newly admitted or exported."""
    incomplete = _exchange("custom")
    incomplete["response"]["body"]["status"] = "incomplete"
    records = [incomplete, _exchange("text")]
    assert _training_success_exchange_indices(records) == {1}
    _rollout(tmp_path, records)
    stats = export(tmp_path, tmp_path / "out.jsonl", row_mode="exchange")
    assert stats.rows_written == 1
