//! Red-team label driver: generate synthetic multi-modal windows, run EACH one
//! through idr-main's REAL SentinelCorrelator, and label it by the correlator's
//! own verdict (an emitted ImpossibleState = confirmed campaign). The label is
//! the detection platform's deterministic ground truth, not an assertion.
//!
//! Output: LabeledWindow NDJSON ({window_id,label,events:[IdrEvent...]}) the
//! engine's `--data` / `validate` path consumes directly.
//!
//! Safety: IdrConfig::default() leaves auto_panic_enabled=false, so the panic
//! responder's execute() returns before any `ip`/`nvme` command — verified.
use idr_common::config::IdrConfig;
use idr_common::events::{EventKind, EventSource, IdrEvent, Severity};
use idr_common::reputation::ReputationDb;
use idr_sentinel::correlator::SentinelCorrelator;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc};
use uuid::Uuid;

async fn correlator_confirms(window: &[IdrEvent]) -> bool {
    let correlator = SentinelCorrelator::new(IdrConfig::default(), Arc::new(ReputationDb::new()));
    let (event_tx, event_rx) = mpsc::channel::<IdrEvent>(4096);
    let (dash_tx, mut dash_rx) = broadcast::channel::<IdrEvent>(16384);
    let run = tokio::spawn(correlator.run(event_rx, dash_tx));
    for e in window {
        event_tx.send(e.clone()).await.unwrap();
    }
    drop(event_tx);
    run.await.unwrap().unwrap();
    let mut confirmed = false;
    while let Ok(ev) = dash_rx.try_recv() {
        if matches!(ev.kind, EventKind::ImpossibleState { .. }) {
            confirmed = true;
        }
    }
    confirmed
}

fn host_meta(event: IdrEvent, host: &str) -> IdrEvent {
    event.with_metadata(json!({ "host": host }))
}

/// A multi-modal window. `malicious` decides attributes AND whether the
/// correlator-tripping pair (physics anomaly + IGMP->QUIC correlation) is
/// present, so the real correlator's verdict tracks the intent — but the LABEL
/// written is always the correlator's, not `malicious`.
fn build_window(index: usize, malicious: bool) -> Vec<IdrEvent> {
    let host = format!("host-{index:04}");
    let c2 = format!("198.51.{}.{}", index % 256, 10 + index % 200);
    let sha = format!("{}{index:062x}", if malicious { "de" } else { "5a" });
    let sha = sha[sha.len() - 64..].to_string();
    let mut w = Vec::new();
    if malicious {
        // Confirming pair FIRST: physics single-hop intercept on the path to a
        // HIGH-TRUST service (seeded in the reputation DB) + IGMP->QUIC. Emitted
        // before the exfil signal so the correlator confirms ImpossibleState
        // before firmware_anomaly_active diverts into the panic-condition path.
        let trusted = ["142.250.80.46", "8.8.8.8", "1.1.1.1", "157.240.1.35"][index % 4];
        w.push(host_meta(
            IdrEvent::new(
                EventSource::KernelEbpf,
                Severity::High,
                EventKind::PhysicsAnomaly {
                    dst_ip: trusted.into(),
                    expected_ttl_range: (48, 58),
                    observed_ttl: 63,
                    rtt_ms: 2.0,
                    reason: "single-hop intercept".into(),
                },
            ),
            &host,
        ));
        w.push(host_meta(
            IdrEvent::new(
                EventSource::SentinelCorrelation,
                Severity::High,
                EventKind::IgmpQuicCorrelation {
                    igmp_event_id: Uuid::new_v4(),
                    quic_event_id: Uuid::new_v4(),
                    window_ms: 200,
                },
            ),
            &host,
        ));
    }
    // Multi-modal payload (malicious or benign attributes on the same modalities).
    w.push(host_meta(
        IdrEvent::new(
            EventSource::KernelEbpf,
            if malicious {
                Severity::High
            } else {
                Severity::Info
            },
            EventKind::SocketLineage {
                pid: 4000 + index as u32,
                tgid: 4000 + index as u32,
                exe_path: (if malicious {
                    "/tmp/.x/update"
                } else {
                    "/usr/bin/updater"
                })
                .into(),
                exe_sha256: sha,
                dst_ip: c2.clone(),
                dst_port: 443,
                is_signed: !malicious,
            },
        ),
        &host,
    ));
    w.push(host_meta(
        IdrEvent::new(
            EventSource::NetworkZeek,
            Severity::High,
            EventKind::NtpTimeShift {
                offset_seconds: if malicious { 90.0 } else { 0.3 },
                ntp_server: c2.clone(),
            },
        ),
        &host,
    ));
    w.push(host_meta(
        IdrEvent::new(
            EventSource::HardwareNvme,
            if malicious {
                Severity::Critical
            } else {
                Severity::Info
            },
            EventKind::NvmeLatencyAnomaly {
                device: "/dev/nvme0".into(),
                baseline_us: 100,
                observed_us: if malicious { 450 } else { 110 },
                deviation_pct: if malicious { 350.0 } else { 8.0 },
                concurrent_exfil: malicious,
            },
        ),
        &host,
    ));
    w
}

#[tokio::main]
async fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: harness <out.labeled.ndjson> [n_each]");
    let n: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    let mut lines = Vec::new();
    let (mut pos, mut neg) = (0usize, 0usize);
    for index in 0..(2 * n) {
        let intended_malicious = index % 2 == 0;
        let window = build_window(index, intended_malicious);
        let label = u8::from(correlator_confirms(&window).await); // REAL correlator assigns the label
        if label == 1 {
            pos += 1
        } else {
            neg += 1
        }
        let events: Vec<serde_json::Value> = window
            .iter()
            .map(|e| serde_json::to_value(e).unwrap())
            .collect();
        lines.push(
            serde_json::to_string(&json!({
                "window_id": format!("w{index:04}"), "label": label, "events": events,
            }))
            .unwrap(),
        );
    }
    std::fs::write(&out, lines.join("\n") + "\n").unwrap();
    eprintln!(
        "wrote {} windows to {out}: {pos} correlator-confirmed (label 1), {neg} not (label 0)",
        2 * n
    );
}
