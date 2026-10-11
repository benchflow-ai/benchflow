//! The trajectory writer's float formatting against Python's `repr(float)`
//! (0.8 wrote acp_trajectory.jsonl with `json.dumps`, and files are
//! compared byte for byte). The fixture holds 9,111 floats (edge values,
//! random bit patterns, rounded decimals, wide magnitudes, and 4,000 ties
//! between equally short digit strings, half of them rounding down) with
//! the `repr` 0.8's Python 3.12 gave each; `fixtures/make_float_repr.py`
//! writes it. Guards the D5 fix 132fcbfd (ties broken upward).

use benchflow_agent::trajectory::python_float_repr;

#[test]
fn python_float_repr_matches_python() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/python-float-repr.jsonl");
    let text = std::fs::read_to_string(path).unwrap();
    let mut n = 0;
    for line in text.lines() {
        let case: serde_json::Value = serde_json::from_str(line).unwrap();
        let value = f64::from_bits(case["bits"].as_u64().unwrap());
        assert_eq!(
            python_float_repr(value),
            case["repr"].as_str().unwrap(),
            "{value:e}"
        );
        n += 1;
    }
    assert!(n > 9000);
}
