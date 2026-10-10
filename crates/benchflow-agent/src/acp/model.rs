//! Choosing the model and effort ids an agent's session accepts (0.8's
//! `acp/runtime.py`).

use serde_json::Value;

use super::error::AcpError;

/// The model name of a Codex id: `gpt-5[high]` is `gpt-5`.
pub(crate) fn codex_model_name(id: &str) -> &str {
    id.split('[').next().unwrap_or(id)
}

/// The effort of a Codex id: `gpt-5[high]` is `high`; empty when none.
pub(crate) fn codex_effort(id: &str) -> &str {
    match (id.rfind('['), id.strip_suffix(']')) {
        (Some(open), Some(body)) if open < body.len() => &body[open + 1..],
        _ => "",
    }
}

/// The exact `model[effort]` id to send codex-acp for `model`, from the
/// models its session advertises (`session/new`'s `models`). codex-acp
/// validates `session/set_model` against those ids and answers anything
/// else with an opaque `-32603`, so a model it does not offer is refused
/// here, naming what it offers. A requested effort picks the matching
/// variant; otherwise medium, then high, low, minimal, none.
pub(crate) fn codex_model_id(
    agent: &str,
    model: &str,
    models: Option<&Value>,
    effort: Option<&str>,
) -> Result<String, AcpError> {
    let Some(state) = models.and_then(Value::as_object) else {
        return Ok(model.to_string());
    };
    let requested = codex_model_name(model);
    if let Some(current) = state.get("currentModelId").and_then(Value::as_str)
        && codex_model_name(current) == requested
        && effort.is_none_or(|e| codex_effort(current) == e)
    {
        return Ok(current.to_string());
    }
    let Some(available) = state.get("availableModels").and_then(Value::as_array) else {
        return Ok(model.to_string());
    };
    let ids: Vec<&str> = available
        .iter()
        .filter_map(|entry| entry.get("modelId").and_then(Value::as_str))
        .collect();
    let candidates: Vec<&str> = ids
        .iter()
        .copied()
        .filter(|id| codex_model_name(id) == requested)
        .collect();
    if candidates.is_empty() {
        let mut offered: Vec<String> = ids
            .iter()
            .map(|id| codex_model_name(id).to_string())
            .collect();
        offered.sort();
        offered.dedup();
        if offered.is_empty() {
            return Ok(model.to_string());
        }
        return Err(AcpError::ModelNotOffered {
            agent: agent.to_string(),
            model: requested.to_string(),
            offered,
        });
    }
    let preferred = effort
        .into_iter()
        .chain(["medium", "high", "low", "minimal", "none"]);
    for wanted in preferred {
        if let Some(candidate) = candidates.iter().find(|c| codex_effort(c) == wanted) {
            return Ok(candidate.to_string());
        }
    }
    Ok(candidates[0].to_string())
}

/// The ids of the config options a session offers.
pub(crate) fn config_option_ids(options: &[Value]) -> Vec<&str> {
    options
        .iter()
        .filter_map(|o| o.get("id").and_then(Value::as_str))
        .collect()
}

/// The values a `select` config option offers, groups flattened in order.
pub(crate) fn config_option_values(options: &[Value], config_id: &str) -> Vec<String> {
    let Some(option) = options
        .iter()
        .find(|o| o.get("id").and_then(Value::as_str) == Some(config_id))
    else {
        return Vec::new();
    };
    fn walk(choices: &[Value], out: &mut Vec<String>) {
        for choice in choices {
            if let Some(group) = choice.get("options").and_then(Value::as_array) {
                walk(group, out);
            } else if let Some(value) = choice.get("value").and_then(Value::as_str) {
                out.push(value.to_string());
            }
        }
    }
    let mut values = Vec::new();
    if let Some(choices) = option.get("options").and_then(Value::as_array) {
        walk(choices, &mut values);
    }
    values
}

/// Whether an agent's error says `value` itself was refused, not that
/// something transient went wrong: it reads as a refusal and names the
/// value ("model temporarily not available" stays a retried error).
pub(crate) fn refuses_value(error: &AcpError, value: &str) -> bool {
    let AcpError::Agent { message, data, .. } = error else {
        return false;
    };
    let details = match data.as_deref() {
        Some(Value::Object(map)) => map.get("details").map(value_text).unwrap_or_default(),
        Some(other) => value_text(other),
        None => String::new(),
    };
    let text = format!("{message} {details}").to_lowercase();
    if !value.is_empty() && !text.contains(&value.to_lowercase()) {
        return false;
    }
    ["invalid value", "not offered", "unknown model"]
        .iter()
        .any(|m| text.contains(m))
}

fn value_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn codex_models() -> Value {
        json!({
            "currentModelId": "gpt-5.6-sol[medium]",
            "availableModels": [
                {"modelId": "gpt-5.6-sol[low]", "name": "x"},
                {"modelId": "gpt-5.6-sol[medium]", "name": "x"},
                {"modelId": "gpt-5.6-sol[high]", "name": "x"},
                {"modelId": "gpt-6.1-sol[high]", "name": "x"},
                {"modelId": "gpt-6.1-sol[xhigh]", "name": "x"},
            ],
        })
    }

    #[test]
    fn codex_ids_carry_the_effort_the_session_offers() {
        let models = codex_models();
        let pick = |model, effort| codex_model_id("codex", model, Some(&models), effort);
        assert_eq!(pick("gpt-5.6-sol", None).unwrap(), "gpt-5.6-sol[medium]");
        assert_eq!(
            pick("gpt-5.6-sol", Some("high")).unwrap(),
            "gpt-5.6-sol[high]"
        );
        assert_eq!(
            pick("gpt-6.1-sol", Some("xhigh")).unwrap(),
            "gpt-6.1-sol[xhigh]"
        );
        // No medium: high comes next.
        assert_eq!(pick("gpt-6.1-sol", None).unwrap(), "gpt-6.1-sol[high]");
        // Nothing preferred matches: the first variant.
        assert_eq!(
            pick("gpt-6.1-sol", Some("max")).unwrap(),
            "gpt-6.1-sol[high]"
        );
        let err = pick("gpt-4o", None).unwrap_err();
        assert_eq!(
            err.to_string(),
            "agent integration failure [agent_model]: codex does not offer model 'gpt-4o' for this login; \
             it offers: gpt-5.6-sol, gpt-6.1-sol"
        );
        // Without advertised models the id goes as given.
        assert_eq!(
            codex_model_id("codex", "gpt-5", None, None).unwrap(),
            "gpt-5"
        );
        assert_eq!(codex_effort("gpt-5[high]"), "high");
        assert_eq!(codex_effort("gpt-5"), "");
        assert_eq!(codex_model_name("gpt-5[high]"), "gpt-5");
    }

    #[test]
    fn select_values_flatten_groups() {
        let options = vec![json!({
            "id": "model",
            "type": "select",
            "currentValue": "default",
            "options": [
                {"value": "default", "name": "Default"},
                {"group": "More", "name": "More", "options": [{"value": "opus"}, {"value": "sonnet[1m]"}]},
                {"value": "haiku"},
            ],
        })];
        assert_eq!(
            config_option_values(&options, "model"),
            ["default", "opus", "sonnet[1m]", "haiku"]
        );
        assert!(config_option_values(&options, "effort").is_empty());
        assert_eq!(config_option_ids(&options), ["model"]);
    }

    #[test]
    fn only_a_refusal_naming_the_value_counts() {
        let error = |message: &str, data: Option<Value>| AcpError::Agent {
            code: -32603,
            message: message.into(),
            data: data.map(Box::new),
            method: "session/set_config_option".into(),
        };
        let refused = error(
            "Internal error",
            Some(json!({"details": "Invalid value for config option model: claude-x"})),
        );
        assert!(refuses_value(&refused, "claude-x"));
        assert!(!refuses_value(&refused, "claude-y"));
        assert!(!refuses_value(
            &error("model temporarily not available: claude-x", None),
            "claude-x"
        ));
    }
}
