//! Low-level port parity: entities, edges, and feature vectors must match the
//! Python `features.project_event` output exactly on every fixture event.
//! This catches string-handling or truthiness divergence long before it could
//! show up as a scoring difference.

use std::path::Path;

use serde_json::Value;

use idr_intelligence_rt::event::RawEvent;
use idr_intelligence_rt::features::project_event;
use idr_intelligence_rt::manifest::Manifest;

#[test]
fn feature_vectors_match_python() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let manifest = Manifest::load(&dir).unwrap();
    let raw: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("feature_vectors.json")).unwrap())
            .unwrap();
    let entries = raw.as_array().unwrap();
    assert!(
        entries.len() >= 20,
        "fixture unexpectedly small: {} entries",
        entries.len()
    );
    for (index, entry) in entries.iter().enumerate() {
        let event = RawEvent::from_value(&entry["event"]).unwrap();
        let projection = project_event(
            &event,
            entry["delta_seconds"].as_f64().unwrap(),
            &manifest.features,
        );
        let want_entities: Vec<&str> = entry["entities"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(
            projection.entities, want_entities,
            "entities diverge at entry {index} ({})",
            event.id
        );
        let want_edges: Vec<Vec<&str>> = entry["edges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|edge| {
                edge.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap())
                    .collect()
            })
            .collect();
        let got_edges: Vec<Vec<&str>> = projection
            .edges
            .iter()
            .map(|(left, right, relation)| vec![left.as_str(), right.as_str(), relation.as_str()])
            .collect();
        assert_eq!(
            got_edges, want_edges,
            "edges diverge at entry {index} ({})",
            event.id
        );
        let want_features: Vec<f64> = entry["features"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        assert_eq!(
            projection.features.len(),
            want_features.len(),
            "feature length at entry {index}"
        );
        for (channel, (got, want)) in projection.features.iter().zip(&want_features).enumerate() {
            assert!(
                (f64::from(*got) - want).abs() <= 1e-6,
                "feature {channel} ({}) diverges at entry {index} ({}): {got} vs {want}",
                manifest.features.names[channel],
                event.id
            );
        }
    }
}
