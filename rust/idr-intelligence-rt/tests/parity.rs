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
    score_in(&fixtures(), case, max_nodes, suppressions)
}

fn score_in(
    dir: &std::path::Path,
    case: &str,
    max_nodes: Option<usize>,
    suppressions: &[String],
) -> (Value, Value, Value) {
    let text = std::fs::read_to_string(fixtures().join(format!("events_{case}.ndjson"))).unwrap();
    let mut events = read_ndjson(&text).unwrap();
    events.sort_by(|a, b| (a.timestamp, &a.id).cmp(&(b.timestamp, &b.id)));
    let mut scorer = StreamingScorer::open(dir, max_nodes).unwrap();
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

// Probability/PSI tolerance: goldens are rounded to 6 decimals, so runtime
// drift (~1e-7 measured) is quantized to 1e-6 steps — 2e-6 tolerates exactly
// one rounding-boundary flip and nothing more.
const PROBABILITY_TOLERANCE: f64 = 2e-6;

fn assert_finding_matches(got: &Value, want: &Value, case: &str) {
    // Field-set equality first: EXACT_FIELDS is a closed list, and indexing a
    // key absent from both sides compares Null==Null — a field added to the
    // finding (or lost from the golden) must fail here, not pass silently.
    let got_keys: std::collections::BTreeSet<&String> = got.as_object().unwrap().keys().collect();
    let want_keys: std::collections::BTreeSet<&String> = want
        .as_object()
        .unwrap()
        .keys()
        .filter(|key| key.as_str() != "evictions")
        .collect();
    assert_eq!(got_keys, want_keys, "finding field sets diverge for {case}");
    for field in EXACT_FIELDS {
        assert_eq!(got[field], want[field], "field {field} diverges for {case}");
    }
    for field in ["escalation_probability", "raw_escalation_probability"] {
        let (g, w) = (got[field].as_f64().unwrap(), want[field].as_f64().unwrap());
        assert!(
            (g - w).abs() <= PROBABILITY_TOLERANCE,
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
                    (g - w).abs() <= PROBABILITY_TOLERANCE,
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

#[test]
fn nognn_bundle_matches_python_reference() {
    // The no-GNN ablation's exported head has no adjacency input; this golden
    // pins the manifest-driven conditional-feed path end to end. Ranked order
    // is NOT compared: without graph mixing, structurally identical entities
    // tie exactly, so tie order is runtime-dependent by nature — the main
    // bundle's goldens own order pinning.
    let dir = fixtures().join("nognn");
    let (got, _, _) = score_in(&dir, "malicious", None, &[]);
    let want: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("expected_malicious.json")).unwrap(),
    )
    .unwrap();
    for field in [
        "campaign_id",
        "calibration",
        "predicted_next_stage",
        "observed_attack_stages",
        "model_version",
        "graph_nodes",
        "graph_relations",
        "engine_version",
        "feature_schema_hash",
        "continues_campaign",
        "windows_observed",
        "feature_drift",
        "entity_evidence",
        "applied_suppressions",
    ] {
        assert_eq!(got[field], want[field], "field {field} diverges for nognn");
    }
    for field in ["escalation_probability", "raw_escalation_probability"] {
        let (g, w) = (got[field].as_f64().unwrap(), want[field].as_f64().unwrap());
        assert!(
            (g - w).abs() <= PROBABILITY_TOLERANCE,
            "{field} diverges for nognn: {g} vs {w}"
        );
    }
    let got_related: std::collections::BTreeSet<&str> = got["related_entities"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap())
        .collect();
    let want_related: std::collections::BTreeSet<&str> = want["related_entities"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap())
        .collect();
    assert_eq!(
        got_related, want_related,
        "related entity sets diverge for nognn"
    );
}

#[test]
fn inconsistent_manifests_are_rejected() {
    // Truncated feature names / zero half-life would panic or NaN at scoring
    // time; Manifest::load must refuse them up front.
    let source = fixtures();
    let scratch = std::env::temp_dir().join(format!("idr-rt-manifest-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    for file in ["step.onnx", "head.onnx"] {
        std::fs::copy(source.join(file), scratch.join(file)).unwrap();
    }
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(source.join("manifest.json")).unwrap())
            .unwrap();

    let mut truncated = manifest.clone();
    let names = truncated["features"]["names"].as_array().unwrap()[..12].to_vec();
    truncated["features"]["names"] = Value::Array(names);
    std::fs::write(scratch.join("manifest.json"), truncated.to_string()).unwrap();
    let error = match StreamingScorer::open(&scratch, None) {
        Ok(_) => panic!("truncated feature names should be rejected"),
        Err(error) => error,
    };
    assert!(
        error.contains("features.names"),
        "unexpected error: {error}"
    );

    let mut zero_decay = manifest.clone();
    zero_decay["model"]["decay_half_life"] = serde_json::json!(0.0);
    std::fs::write(scratch.join("manifest.json"), zero_decay.to_string()).unwrap();
    let error = match StreamingScorer::open(&scratch, None) {
        Ok(_) => panic!("zero decay_half_life should be rejected"),
        Err(error) => error,
    };
    assert!(
        error.contains("decay_half_life"),
        "unexpected error: {error}"
    );

    std::fs::remove_dir_all(&scratch).ok();
}
