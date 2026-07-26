use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Severity levels for the detection pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Severity {
    Info,
    Warning,
    High,
    Critical,
    /// Impossible state detected — cross-layer anomaly confirmed
    Impossible,
}

/// Source layer that generated the event
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventSource {
    KernelEbpf,
    NetworkZeek,
    NetworkSuricata,
    HardwareNvme,
    HardwareMoca,
    HardwareRtc,
    SentinelCorrelation,
    /// External offline classifier (the triage CLI tailer)
    ExternalTriage,
}

/// Canonical event envelope for all telemetry flowing through the IDR pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdrEvent {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub source: EventSource,
    pub severity: Severity,
    pub kind: EventKind,
    /// Free-form metadata for layer-specific details
    pub metadata: serde_json::Value,
}

impl IdrEvent {
    pub fn new(source: EventSource, severity: Severity, kind: EventKind) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            source,
            severity,
            kind,
            metadata: serde_json::Value::Null,
        }
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Discriminated union of all event types in the detection pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EventKind {
    // === Kernel Layer (eBPF) ===
    /// IGMP v3 multicast detected on 224.0.0.1
    IgmpTrigger {
        src_ip: String,
        group_addr: String,
    },
    /// QUIC heartbeat (UDP 443) detected within IGMP correlation window
    QuicHeartbeat {
        src_ip: String,
        dst_ip: String,
        dst_port: u16,
        pid: u32,
        exe_path: String,
    },
    /// IGMP → QUIC correlation confirmed within 500ms window
    IgmpQuicCorrelation {
        igmp_event_id: Uuid,
        quic_event_id: Uuid,
        window_ms: u64,
    },
    /// Socket opened by a process — lineage tracking
    SocketLineage {
        pid: u32,
        tgid: u32,
        exe_path: String,
        exe_sha256: String,
        dst_ip: String,
        dst_port: u16,
        is_signed: bool,
    },
    /// Unsigned/non-standard binary beaconing to high-trust IP
    SuspiciousBeacon {
        pid: u32,
        exe_path: String,
        exe_sha256: String,
        dst_ip: String,
        asn_owner: String,
    },
    /// TTL or RTT anomaly indicating physical intercept
    PhysicsAnomaly {
        dst_ip: String,
        expected_ttl_range: (u8, u8),
        observed_ttl: u8,
        rtt_ms: f64,
        reason: String,
    },

    // === Network Layer (Zeek/Suricata) ===
    /// DNS PTR query with octet reversal detected
    OctetReversalDetected {
        forward_ip: String,
        reversed_ip: String,
        forward_asn: String,
        reversed_asn: String,
        ptr_query: String,
    },
    /// NTP time shift exceeds threshold
    NtpTimeShift {
        offset_seconds: f64,
        ntp_server: String,
    },
    /// Expired TLS certificate accepted during NTP time-shift window
    HstsTimeManipulation {
        domain: String,
        cert_expiry: String,
        ntp_shift_seconds: f64,
    },

    // === Hardware & Bus Layer ===
    /// NVMe I/O latency deviation from baseline
    NvmeLatencyAnomaly {
        device: String,
        baseline_us: u64,
        observed_us: u64,
        deviation_pct: f64,
        concurrent_exfil: bool,
    },
    /// Gateway MAC address flapping (MoCA/ARP MitM indicator)
    MacFlapping {
        gateway_ip: String,
        old_mac: String,
        new_mac: String,
        flap_count: u32,
        window_seconds: u64,
    },
    /// Software clock diverged from hardware RTC
    RtcClockDivergence {
        software_time: String,
        rtc_time: String,
        drift_seconds: f64,
    },

    // === Sentinel Engine ===
    /// Cross-layer correlation — "impossible state" detected
    ImpossibleState {
        correlated_event_ids: Vec<Uuid>,
        description: String,
        kill_chain_stage: String,
    },
    /// Panic response triggered
    PanicResponse {
        reason: String,
        actions_taken: Vec<String>,
    },

    // === External Classifier (offline triage tool) ===
    /// Triage CLI emitted a family classification for a sample / pcap / CAPE report
    TriageClassification {
        family: String,
        variant: Option<String>,
        /// Family-type metadata from the KB ("ransomware", "rat", "loader",
        /// "stealer", "c2", "apt", "phishing_tracker", etc.). Drives stage
        /// mapping in the correlator; the correlator no longer needs to
        /// hardcode family-name lists.
        #[serde(default)]
        family_type: String,
        confidence: String,
        score: u32,
        source_path: String,
        sha256: Option<String>,
        /// Destination IPs the artifact contacted (from pcap or extracted-URI hosts).
        /// Empty for pure binary/CAPE-text scans where no network state is observed.
        /// Used by the SentinelCorrelator to advance kill-chain tracks.
        #[serde(default)]
        dest_ips: Vec<String>,
    },

    // === BGP Data Plane (idr-bgpd) ===
    /// BGP anomaly emitted by the BGP data plane. Spec §6.2 / §6.3.
    /// Production variants advance the kill chain; `Observed*` calibration
    /// variants are emitted for analyst review without state advancement.
    BgpAnomaly {
        kind: BgpAnomalyKind,
        /// IP-network string (e.g. "8.8.8.0/24"). Stored as String here
        /// to keep `idr-common` free of `ipnet` as a hard dep.
        prefix: String,
        observed_origin_asn: u32,
        legitimate_origin_asn: Option<u32>,
        /// Confidence band: high (ROA-backed), medium (multi-collector
        /// corroborated heuristic), low (single-source heuristic).
        #[serde(default = "default_bgp_confidence")]
        confidence: String,
    },
}

fn default_bgp_confidence() -> String {
    "medium".to_string()
}

/// BGP anomaly classification. Production variants advance the kill
/// chain in the correlator; `Observed*` calibration variants emit
/// only (per spec §6.3).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BgpAnomalyKind {
    // ─── Production rules — advance kill-chain ──────────────────────
    /// Spec §6.3 production rule. Subprefix advertised by an AS not
    /// in the legitimate origin or upstream-AS allowlist for a
    /// covering local-infra prefix; recent first_seen; ≥2 collectors.
    SubprefixHijackLocalInfra {
        covered_local_prefix: String,
        hijacker_asn: u32,
    },
    /// Bogon origin — collector noise; correlator drops these.
    Bogon,

    // ─── Calibration rules — emit but DO NOT advance kill-chain ────
    ObservedMoas {
        observed_origins: Vec<u32>,
    },
    ObservedRpkiInvalid {
        authorized_origin: Option<u32>,
    },
    ObservedOriginFlap {
        previous_origin: u32,
        new_origin: u32,
    },
    ObservedSquatBurst {
        lifetime_secs: u64,
    },
    ObservedSquatDormant {
        silent_days: u64,
    },
    ObservedAsPathPrepend {
        n_self_prepends: u8,
        path: Vec<u32>,
    },
    ObservedValleyFreeViolation {
        path: Vec<u32>,
        violation_at: usize,
    },
    /// Calibration variant of subprefix-MOAS-with-different-origin.
    /// Emitted whenever a more-specific prefix is announced by an
    /// origin different from its cover's origin and not in the cover's
    /// transit upstream set, and the cover prefix is NOT in the
    /// operator's local-infra allowlist. Most of these are normal
    /// traffic-engineering events; `SubprefixHijackLocalInfra` is the
    /// production-tier counterpart, fired only for prefixes the
    /// operator has explicitly claimed.
    ObservedSubprefixMoreSpecific {
        covered_prefix: String,
        cover_origin_asn: u32,
        new_origin_asn: u32,
    },

    // ─── Informational / always-pass-through ────────────────────────
    RpkiTransition {
        from: String,
        to: String,
    },
}

/// Kill chain stages for the DPRK-001 campaign
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KillChainStage {
    /// Stage 1: IGMP multicast trigger for C2 wake
    IgmpTrigger,
    /// Stage 2: QUIC heartbeat beacon to C2
    QuicHeartbeat,
    /// Stage 3: DNS PTR octet reversal for DPI evasion
    PtrOctetReversal,
    /// Stage 4: BGP adjacency sinkhole
    BgpSinkhole,
    /// Stage 5: Data exfiltration via NVMe controller
    NvmeExfiltration,
}
