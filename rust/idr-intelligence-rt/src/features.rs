//! Entity, edge, and feature extraction — a line-for-line port of `features.py`.
//!
//! The feature-vector parity fixture (`tests/fixtures/feature_vectors.json`,
//! generated from the Python implementation) pins this port; any semantic
//! drift from `project_event` fails that test before it can skew scoring.

use blake2::Blake2sVar;
use blake2::digest::{Update, VariableOutput};
use serde_json::{Map, Value};

use crate::event::{RawEvent, stringify};
use crate::manifest::FeatureSpec;

/// `delta_seconds_log` position in the feature vector (graph.py `_DELTA_FEATURE_INDEX`).
pub const DELTA_FEATURE_INDEX: usize = 2;

/// One event decomposed into entities, typed edges, and a feature vector.
#[derive(Debug, Clone)]
pub struct Projection {
    pub entities: Vec<String>,
    pub edges: Vec<(String, String, String)>,
    pub features: Vec<f32>,
}

/// Python truthiness for the JSON values feature extraction branches on.
fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(flag)) => *flag,
        Some(Value::Number(number)) => number.as_f64().is_some_and(|value| value != 0.0),
        Some(Value::String(text)) => !text.is_empty(),
        Some(Value::Array(items)) => !items.is_empty(),
        Some(Value::Object(object)) => !object.is_empty(),
    }
}

/// Python `a or b` over two optional kind values: first truthy, else the second.
fn truthy_chain<'v>(kind: &'v Map<String, Value>, first: &str, second: &str) -> Option<&'v Value> {
    if truthy(kind.get(first)) {
        return kind.get(first);
    }
    kind.get(second)
}

/// `log1p(max(seconds, 0)) / divisor` — matches `graph._normalize_delta`.
pub fn normalize_delta(seconds: f64, divisor: f64) -> f64 {
    seconds.max(0.0).ln_1p() / divisor
}

/// Project one event onto the fixed feature vector plus its graph pieces.
pub fn project_event(event: &RawEvent, delta_seconds: f64, spec: &FeatureSpec) -> Projection {
    let entities = extract_entities(event);
    let edges = derive_edges(&entities);
    let kind = &event.kind;
    let kind_type = event.kind_type();
    let mut features = vec![0.0f32; spec.names.len()];
    features[0] = spec
        .severity_weight
        .get(&event.severity)
        .copied()
        .unwrap_or(spec.severity_default) as f32;
    features[1] = spec
        .kind_prior
        .get(&kind_type)
        .copied()
        .unwrap_or(spec.kind_prior_default) as f32;
    features[DELTA_FEATURE_INDEX] = normalize_delta(delta_seconds, spec.delta_log_divisor) as f32;
    features[3] = flag(kind.contains_key("pid") || kind.contains_key("tgid"));
    features[4] = flag(kind.get("is_signed") == Some(&Value::Bool(false)));
    features[5] = flag(
        kind.keys()
            .any(|key| key.ends_with("_ip") || key == "dest_ips" || key == "ntp_server"),
    );
    features[6] = flag(kind.contains_key("exe_sha256") || truthy(kind.get("sha256")));
    features[7] = flag(
        ["domain", "sni", "ptr_query"]
            .iter()
            .any(|key| kind.contains_key(*key)),
    );
    features[8] = flag(
        ["prefix", "observed_origin_asn", "asn_owner"]
            .iter()
            .any(|key| kind.contains_key(*key)),
    );
    features[9] = flag(matches!(
        kind_type.as_str(),
        "nvme_latency_anomaly" | "mac_flapping" | "rtc_clock_divergence"
    ));
    features[10] = flag(truthy(kind.get("concurrent_exfil")));
    features[11] = flag(is_production_bgp_anomaly(kind));
    features[12 + source_group(&event.source)] = 1.0;
    features[16 + (blake2s_one_byte(&kind_type) % 4) as usize] = 1.0;
    let has_user = entities.iter().any(|entity| entity.starts_with("user:"));
    features[20] = flag(has_user);
    // Identity pivot: an authenticated actor reaching a remote host/resource.
    features[21] = flag(
        has_user
            && ["dst_ip", "dest_ips", "target_host", "cloud_resource", "arn"]
                .iter()
                .any(|key| kind.contains_key(*key)),
    );
    Projection {
        entities,
        edges,
        features,
    }
}

fn flag(condition: bool) -> f32 {
    if condition { 1.0 } else { 0.0 }
}

/// blake2s with a one-byte digest, identical to `hashlib.blake2s(digest_size=1)`.
fn blake2s_one_byte(text: &str) -> u8 {
    let mut hasher = Blake2sVar::new(1).expect("one-byte blake2s digest");
    hasher.update(text.as_bytes());
    let mut digest = [0u8; 1];
    hasher
        .finalize_variable(&mut digest)
        .expect("one-byte blake2s digest");
    digest[0]
}

/// List the deduplicated typed entities (host:, process:, ip:, ...) an event mentions.
pub fn extract_entities(event: &RawEvent) -> Vec<String> {
    let kind = &event.kind;
    let host = if truthy(event.metadata.get("host")) {
        stringify(&event.metadata["host"])
    } else if truthy(event.metadata.get("hostname")) {
        stringify(&event.metadata["hostname"])
    } else {
        "unknown-host".to_string()
    };
    let mut entities: Vec<String> = vec![format!("host:{host}")];
    let push = |entities: &mut Vec<String>, entity: String| {
        if !entities.contains(&entity) {
            entities.push(entity);
        }
    };
    // Python: `pid = kind.get("pid") or kind.get("tgid"); if pid is not None:` —
    // a falsy-but-present tgid (e.g. 0) still yields a process entity.
    if let Some(pid) = truthy_chain(kind, "pid", "tgid")
        && !pid.is_null()
    {
        push(&mut entities, format!("process:{host}:{}", stringify(pid)));
    }
    for key in ["exe_sha256", "sha256"] {
        if truthy(kind.get(key)) {
            push(
                &mut entities,
                format!("hash:{}", stringify(&kind[key]).to_lowercase()),
            );
        }
    }
    for key in [
        "src_ip",
        "dst_ip",
        "forward_ip",
        "reversed_ip",
        "ntp_server",
        "gateway_ip",
    ] {
        if truthy(kind.get(key)) {
            push(&mut entities, format!("ip:{}", stringify(&kind[key])));
        }
    }
    if let Some(Value::Array(items)) = kind.get("dest_ips") {
        for item in items {
            push(&mut entities, format!("ip:{}", stringify(item)));
        }
    }
    if truthy(kind.get("prefix")) {
        push(
            &mut entities,
            format!("prefix:{}", stringify(&kind["prefix"])),
        );
    }
    for key in ["observed_origin_asn", "legitimate_origin_asn"] {
        if let Some(value) = kind.get(key)
            && !value.is_null()
        {
            push(&mut entities, format!("asn:{}", stringify(value)));
        }
    }
    for key in ["domain", "ptr_query"] {
        if truthy(kind.get(key)) {
            let name = stringify(&kind[key]).to_lowercase();
            push(
                &mut entities,
                format!("domain:{}", name.trim_end_matches('.')),
            );
        }
    }
    if truthy(kind.get("device")) {
        push(
            &mut entities,
            format!("device:{host}:{}", stringify(&kind["device"])),
        );
    }
    // user: is global (no host prefix) so the same actor links events across hosts.
    if let Some(user) = ["user", "username", "account"]
        .iter()
        .find(|key| truthy(kind.get(**key)))
    {
        push(
            &mut entities,
            format!("user:{}", stringify(&kind[*user]).to_lowercase()),
        );
    }
    if let Some(session) = truthy_chain(kind, "session_id", "sid")
        && !session.is_null()
    {
        push(
            &mut entities,
            format!("session:{host}:{}", stringify(session)),
        );
    }
    for key in ["cloud_resource", "arn"] {
        if truthy(kind.get(key)) {
            push(
                &mut entities,
                format!("cloud:{}", stringify(&kind[key]).to_lowercase()),
            );
        }
    }
    entities
}

/// Yield (left, right, relation) edges implied by one event's entities.
fn derive_edges(entities: &[String]) -> Vec<(String, String, String)> {
    let of = |prefix: &str| -> Vec<&String> {
        entities
            .iter()
            .filter(|entity| entity.starts_with(prefix))
            .collect()
    };
    let host = entities
        .iter()
        .find(|entity| entity.starts_with("host:"))
        .expect("host entity always present");
    let process = entities
        .iter()
        .find(|entity| entity.starts_with("process:"));
    let anchor = process.unwrap_or(host);
    let mut edges = Vec::new();
    let mut add = |left: &str, right: &str, relation: &str| {
        edges.push((left.to_string(), right.to_string(), relation.to_string()));
    };
    if let Some(process) = process {
        add(host, process, "executes");
    }
    for digest in of("hash:") {
        add(anchor, digest, "identified_by");
    }
    let ips = of("ip:");
    for ip in &ips {
        add(anchor, ip, "connects_to");
    }
    for prefix in of("prefix:") {
        for ip in &ips {
            add(ip, prefix, "belongs_to");
        }
        for asn in of("asn:") {
            add(prefix, asn, "originated_by");
        }
    }
    for domain in of("domain:") {
        for ip in &ips {
            add(domain, ip, "resolves_or_connects");
        }
        add(host, domain, "queries_or_visits");
    }
    for device in of("device:") {
        add(host, device, "contains");
    }
    for user in of("user:") {
        add(user, host, "authenticates_to");
        if let Some(process) = process {
            add(user, process, "owns");
        }
        for cloud in of("cloud:") {
            add(user, cloud, "accesses");
        }
    }
    for session in of("session:") {
        add(host, session, "contains");
        if let Some(process) = process {
            add(session, process, "spawns");
        }
    }
    edges
}

/// True only for the nested sub-prefix hijack shape emitted by idr-main.
fn is_production_bgp_anomaly(kind: &Map<String, Value>) -> bool {
    if kind.get("type").and_then(Value::as_str) != Some("bgp_anomaly") {
        return false;
    }
    kind.get("kind")
        .and_then(Value::as_object)
        .and_then(|nested| nested.get("kind"))
        .and_then(Value::as_str)
        == Some("subprefix_hijack_local_infra")
}

fn source_group(source: &str) -> usize {
    match source {
        "kernel_ebpf" => 0,
        "network_zeek" | "network_suricata" => 1,
        _ if source.starts_with("hardware_") => 2,
        _ => 3,
    }
}
