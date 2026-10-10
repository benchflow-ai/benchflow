//! A JSON Schema for task.md's config block (the ` ```toml task ` block),
//! built from the same vocabulary the checker uses: the tables, their closed
//! keys, nested tables (`*` for names the task picks, `[]` for arrays of
//! tables), and documented values. Values are not typed beyond that; `bench
//! check` remains the full validator.

use serde_json::{Map, Value, json};

use super::vocab;

/// The schema, as a JSON value.
pub fn config_schema() -> Value {
    let mut props = Map::new();
    for key in vocab::IDENTITY {
        let v = match *key {
            "authors" | "keywords" => json!({"type": "array"}),
            _ => json!({"type": "string"}),
        };
        props.insert((*key).to_string(), v);
    }
    props.insert("$schema".into(), json!({"type": "string"}));
    for table in vocab::TABLES {
        props.insert((*table).to_string(), child("", table));
    }
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "task.md config",
        "description": "The ```toml task block of a task.md (draft 2). Generated from the checker's vocabulary; `bench check` validates values in full.",
        "type": "object",
        "properties": props,
        "additionalProperties": false,
    })
}

/// The schema of `key` in the table at rule path `parent`.
fn child(parent: &str, key: &str) -> Value {
    let path = if parent.is_empty() {
        key.to_string()
    } else {
        format!("{parent}.{key}")
    };
    if closed(&path).is_some() {
        return table(&path);
    }
    let items = format!("{path}[]");
    if closed(&items).is_some() || vocab::ARRAYS.contains(&path.as_str()) {
        return json!({"type": "array", "items": table(&items)});
    }
    let named = format!("{path}.*");
    if closed(&named).is_some() {
        return json!({"type": "object", "additionalProperties": table(&named)});
    }
    match vocab::enum_values(&path) {
        Some(values) => json!({"enum": values}),
        None => json!({}),
    }
}

/// A table's schema: its closed keys, or any content the task defines.
fn table(path: &str) -> Value {
    let Some(keys) = closed(path) else {
        return json!({"type": "object"});
    };
    let props: Map<String, Value> = keys
        .iter()
        .map(|k| ((*k).to_string(), child(path, k)))
        .collect();
    json!({"type": "object", "properties": props, "additionalProperties": false})
}

fn closed(path: &str) -> Option<&'static [&'static str]> {
    vocab::keys(path).or_else(|| vocab::nested(path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_tables_nesting_and_values() {
        let s = config_schema();
        let p = &s["properties"];
        assert_eq!(p["sandbox"]["additionalProperties"], false);
        assert!(p["sandbox"]["properties"]["image"].is_object());
        assert_eq!(
            p["sandbox"]["properties"]["clock"]["properties"]["advance"]["enum"][0],
            "real"
        );
        assert_eq!(p["credits"]["type"], "array");
        assert_eq!(p["tiers"]["type"], "object");
        assert!(p["tiers"]["additionalProperties"].is_object());
        assert_eq!(p["name"]["type"], "string");
    }
}
