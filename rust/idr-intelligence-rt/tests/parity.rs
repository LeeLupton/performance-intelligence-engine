//! Golden-stream parity: the Rust bridge must reproduce the torch
//! `StreamingScorer` finding on the committed fixtures — every discrete field
//! exactly, probabilities and drift within 1e-4.

use std::path::{Path, PathBuf};

use serde_json::Value;

use idr_intelligence_rt::event::read_ndjson;
use idr_intelligence_rt::scorer::StreamingScorer;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn score(case: &str, max_nodes: Option<usize>, suppressions: &[String]) -> (Value, Value, Value) {
    let dir = fixtures();
    let text = std::fs::read_to_string(dir.join(format!("events_{case}.ndjson"))).unwrap();
    let mut events = read_ndjson(&text).unwrap();
    events.sort_by(|a, b| (a.timestamp, &a.id).cmp(&(b.timestamp, &b.id)));
    let mut scorer = StreamingScorer::open(&dir, max_nodes).unwrap();
    for event in &events {
        scorer.ingest(event).unwrap();
    }
    let finding = scorer.finding(None, suppressions).unwrap();
    (
        serde_json::to_value(&finding).unwrap(),
        serde_json::to_value(&scorer.evictions).unwrap(),
        events.len().into(),
    )
}

fn expected(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(fixtures().join(name)).unwrap()).unwrap()
}

const EXACT_FIELDS: &[&str] = &[
    "campaign_id",
    "calibration",
    "predicted_next_stage",
    "observed_attack_stages",
    "related_entities",
    "entity_evidence",
    "applied_suppressions",
    "evidence_event_ids",
    "model_version",
    "graph_nodes",
    "graph_relations",
    "engine_version",
    "feature_schema_hash",
    "continues_campaign",
    "windows_observed",
];

fn assert_finding_matches(got: &Value, want: &Value, case: &str) {
    for field in EXACT_FIELDS {
        assert_eq!(got[field], want[field], "field {field} diverges for {case}");
    }
    for field in ["escalation_probability", "raw_escalation_probability"] {
        let (g, w) = (got[field].as_f64().unwrap(), want[field].as_f64().unwrap());
        assert!(
            (g - w).abs() <= 1e-4,
            "{field} diverges for {case}: {g} vs {w}"
        );
    }
    match (&got["feature_drift"], &want["feature_drift"]) {
        (Value::Null, Value::Null) => {}
        (g, w) => {
            assert_eq!(
                g["flagged_features"], w["flagged_features"],
                "flagged_features diverge for {case}"
            );
            for field in ["psi_max", "psi_mean"] {
                let (g, w) = (g[field].as_f64().unwrap(), w[field].as_f64().unwrap());
                assert!(
                    (g - w).abs() <= 1e-4,
                    "{field} diverges for {case}: {g} vs {w}"
                );
            }
        }
    }
}

#[test]
fn malicious_stream_matches_python_reference() {
    let (got, _, _) = score("malicious", None, &[]);
    assert_finding_matches(&got, &expected("expected_malicious.json"), "malicious");
}

#[test]
fn benign_stream_matches_python_reference() {
    let (got, _, _) = score("benign", None, &[]);
    assert_finding_matches(&got, &expected("expected_benign.json"), "benign");
}

#[test]
fn budget_and_suppressions_match_python_reference() {
    let (got, evictions, _) = score("malicious", Some(6), &["ip:".to_string()]);
    let want = expected("expected_budget.json");
    assert_finding_matches(&got, &want, "budget");
    assert_eq!(
        evictions, want["evictions"],
        "eviction audit trail diverges"
    );
}

#[test]
fn out_of_order_events_are_rejected() {
    let dir = fixtures();
    let text = std::fs::read_to_string(dir.join("events_malicious.ndjson")).unwrap();
    let mut events = read_ndjson(&text).unwrap();
    events.sort_by(|a, b| (b.timestamp, &b.id).cmp(&(a.timestamp, &a.id)));
    let mut scorer = StreamingScorer::open(&dir, None).unwrap();
    scorer.ingest(&events[0]).unwrap();
    let error = scorer.ingest(events.last().unwrap()).unwrap_err();
    assert!(error.contains("out-of-order"), "unexpected error: {error}");
}

#[test]
fn finding_without_events_is_an_error() {
    let mut scorer = StreamingScorer::open(&fixtures(), None).unwrap();
    assert_eq!(scorer.finding(None, &[]).unwrap_err(), "no events ingested");
}
