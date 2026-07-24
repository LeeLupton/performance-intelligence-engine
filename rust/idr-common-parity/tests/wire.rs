//! The W17 integration claim, tested against the real thing: events serialized
//! by `idr_common::IdrEvent` are consumed by the bridge, and the committed
//! golden streams are valid idr-main wire format.

use std::path::{Path, PathBuf};

use chrono::{TimeZone, Utc};
use idr_common::events::{BgpAnomalyKind, EventKind, EventSource, IdrEvent, Severity};
use idr_intelligence_rt::event::RawEvent;
use idr_intelligence_rt::scorer::StreamingScorer;
use uuid::Uuid;

fn bridge_fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../idr-intelligence-rt/tests/fixtures")
}

/// A small kill-chain slice built directly from idr_common types.
fn idr_common_events() -> Vec<IdrEvent> {
    let base = Utc.with_ymd_and_hms(2026, 7, 1, 12, 0, 0).unwrap();
    let mut events = vec![
        IdrEvent {
            id: Uuid::from_u128(1),
            timestamp: base,
            source: EventSource::KernelEbpf,
            severity: Severity::High,
            kind: EventKind::SocketLineage {
                pid: 4242,
                tgid: 4242,
                exe_path: "/tmp/.cache/update".to_string(),
                exe_sha256: "AB".repeat(32),
                dst_ip: "203.0.113.77".to_string(),
                dst_port: 443,
                is_signed: false,
            },
            metadata: serde_json::json!({"host": "workstation-77"}),
        },
        IdrEvent {
            id: Uuid::from_u128(2),
            timestamp: base + chrono::Duration::minutes(3),
            source: EventSource::SentinelCorrelation,
            severity: Severity::Critical,
            kind: EventKind::BgpAnomaly {
                kind: BgpAnomalyKind::SubprefixHijackLocalInfra {
                    covered_local_prefix: "203.0.113.0/24".to_string(),
                    hijacker_asn: 64580,
                },
                prefix: "203.0.113.0/25".to_string(),
                observed_origin_asn: 64580,
                legitimate_origin_asn: Some(64500),
                confidence: "high".to_string(),
            },
            metadata: serde_json::json!({"host": "collector-1"}),
        },
        IdrEvent {
            id: Uuid::from_u128(3),
            timestamp: base + chrono::Duration::minutes(7),
            source: EventSource::HardwareNvme,
            severity: Severity::High,
            kind: EventKind::NvmeLatencyAnomaly {
                device: "nvme0n1".to_string(),
                baseline_us: 120,
                observed_us: 2100,
                deviation_pct: 1650.0,
                concurrent_exfil: true,
            },
            metadata: serde_json::json!({"host": "workstation-77"}),
        },
        IdrEvent {
            id: Uuid::from_u128(4),
            timestamp: base + chrono::Duration::minutes(9),
            source: EventSource::ExternalTriage,
            severity: Severity::Warning,
            kind: EventKind::TriageClassification {
                family: "generic_loader".to_string(),
                variant: None,
                family_type: "loader".to_string(),
                confidence: "medium".to_string(),
                score: 61,
                source_path: "/tmp/.cache/update".to_string(),
                sha256: Some("ab".repeat(32)),
                dest_ips: vec!["203.0.113.77".to_string()],
            },
            metadata: serde_json::Value::Null,
        },
    ];
    events.sort_by_key(|event| event.timestamp);
    events
}

#[test]
fn bridge_parses_idr_common_serialized_events() {
    for event in idr_common_events() {
        let wire = serde_json::to_value(&event).unwrap();
        let parsed = RawEvent::from_value(&wire).unwrap_or_else(|error| {
            panic!("bridge rejected idr_common wire format: {error}\n{wire}")
        });
        assert_eq!(parsed.id, event.id.to_string());
        assert_eq!(parsed.timestamp, event.timestamp);
        assert_eq!(
            parsed.source,
            serde_json::to_value(event.source)
                .unwrap()
                .as_str()
                .unwrap()
        );
        assert_eq!(
            parsed.severity,
            serde_json::to_value(event.severity)
                .unwrap()
                .as_str()
                .unwrap()
        );
        assert_eq!(
            Some(parsed.kind_type().as_str()),
            wire["kind"]["type"].as_str()
        );
    }
}

#[test]
fn bridge_scores_an_idr_common_stream_end_to_end() {
    let mut scorer = StreamingScorer::open(&bridge_fixtures(), None).unwrap();
    for event in idr_common_events() {
        let wire = serde_json::to_value(&event).unwrap();
        scorer
            .ingest(&RawEvent::from_value(&wire).unwrap())
            .unwrap();
    }
    let finding = scorer.finding(None, &[]).unwrap();
    assert!(finding.graph_nodes > 0);
    assert!(finding.escalation_probability > 0.0 && finding.escalation_probability < 1.0);
    assert_eq!(
        finding.campaign_id,
        format!("idr-campaign-{}", &Uuid::from_u128(1).to_string()[..8])
    );
    // socket_lineage(execution) + bgp(collection) + nvme(exfiltration) -> impact next.
    assert_eq!(finding.predicted_next_stage, "impact");
    assert!(
        finding
            .related_entities
            .iter()
            .any(|entity| entity == "ip:203.0.113.77")
    );
}

#[test]
fn committed_goldens_deserialize_into_idr_common() {
    // The malicious golden is built from production-shaped kinds only. (The
    // benign golden intentionally carries a calibration-style nested bgp kind
    // idr-common does not define, so it is not asserted here.)
    let path = bridge_fixtures().join("events_malicious.ndjson");
    let text = std::fs::read_to_string(&path).unwrap();
    let mut count = 0;
    for (index, line) in text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
    {
        let event: IdrEvent = serde_json::from_str(line).unwrap_or_else(|error| {
            panic!(
                "golden line {} is not idr_common wire format: {error}",
                index + 1
            )
        });
        assert!(!event.id.is_nil());
        count += 1;
    }
    assert!(count >= 20, "unexpectedly small golden: {count} events");
}
