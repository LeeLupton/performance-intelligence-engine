//! Exhaustive timestamp-acceptance parity: for every shape in the committed
//! battery (verdicts computed by the reference Python interpreter), the
//! bridge must agree on both accept/reject and the UTC instant. This is the
//! pinned form of review finding #7 — parity is a fixture, not a claim.

use std::path::Path;

use chrono::DateTime;
use serde_json::{Value, json};

use idr_intelligence_rt::event::RawEvent;

#[test]
fn timestamp_acceptance_matches_reference_interpreter() {
    run_battery(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/timestamp_battery.json"),
    );
}

/// Optional larger battery for differential sweeps: point IDR_RT_EXTRA_BATTERY
/// at a JSON file with the same {shape, utc} entries (generated from the
/// reference interpreter) to run it in addition to the committed one.
#[test]
fn extra_battery_if_provided() {
    if let Ok(extra) = std::env::var("IDR_RT_EXTRA_BATTERY") {
        run_battery(Path::new(&extra));
    }
}

fn run_battery(path: &Path) {
    let battery: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let entries = battery.as_array().unwrap();
    assert!(
        entries.len() > 100,
        "battery unexpectedly small: {}",
        entries.len()
    );
    let (mut accepted, mut rejected) = (0, 0);
    for entry in entries {
        let shape = entry["shape"].as_str().unwrap();
        let event = json!({
            "id": "battery", "timestamp": shape, "source": "s", "severity": "HIGH",
            "kind": {"type": "t"}, "metadata": null,
        });
        let got = RawEvent::from_value(&event);
        match (got, entry["utc"].as_str()) {
            (Ok(parsed), Some(want)) => {
                let want = DateTime::parse_from_rfc3339(want).unwrap();
                assert_eq!(parsed.timestamp, want, "instant diverges for {shape:?}");
                accepted += 1;
            }
            (Err(_), None) => rejected += 1,
            (Ok(parsed), None) => {
                panic!(
                    "Rust accepts {shape:?} (as {}) but Python rejects it",
                    parsed.timestamp
                )
            }
            (Err(error), Some(want)) => {
                panic!("Python accepts {shape:?} (as {want}) but Rust rejects it: {error}")
            }
        }
    }
    println!("battery: {accepted} accepted + {rejected} rejected, all in agreement");
}
