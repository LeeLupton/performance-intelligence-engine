//! The `idr_common::IdrEvent` wire envelope, validated the way `schema.py` does.

use chrono::{DateTime, NaiveDate, NaiveDateTime, Timelike, Utc};
use serde_json::{Map, Value};

use crate::features::truthy;

/// One validated event in the canonical shape serialized by `idr_common::IdrEvent`.
///
/// `kind` stays a tagged JSON object rather than a typed enum so the bridge —
/// like the Python engine — keeps scoring streams that carry event kinds newer
/// than the crate. Feature extraction only reads string keys.
#[derive(Debug, Clone)]
pub struct RawEvent {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub source: String,
    pub severity: String,
    pub kind: Map<String, Value>,
    pub metadata: Map<String, Value>,
}

impl RawEvent {
    /// Validate one decoded JSON object; the error names what is wrong.
    pub fn from_value(raw: &Value) -> Result<Self, String> {
        let object = raw.as_object().ok_or("event must be a JSON object")?;
        let mut missing: Vec<&str> = ["id", "timestamp", "source", "severity", "kind"]
            .into_iter()
            .filter(|key| !object.contains_key(*key))
            .collect();
        missing.sort_unstable();
        if !missing.is_empty() {
            return Err(format!("missing IdrEvent fields: {}", missing.join(", ")));
        }
        let kind = object["kind"]
            .as_object()
            .filter(|kind| kind.contains_key("type"))
            .ok_or("kind must be a tagged object containing 'type'")?;
        // schema.py does `raw.get("metadata") or {}` before the type check, so
        // any falsy value ([], "", 0, false) silently becomes empty metadata;
        // only truthy non-objects are rejected.
        let metadata = match object.get("metadata") {
            None | Some(Value::Null) => Map::new(),
            Some(Value::Object(metadata)) => metadata.clone(),
            Some(other) if !truthy(Some(other)) => Map::new(),
            Some(_) => return Err("metadata must be an object or null".to_string()),
        };
        Ok(Self {
            id: stringify(&object["id"]),
            timestamp: parse_timestamp(&object["timestamp"])?,
            source: stringify(&object["source"]),
            severity: stringify(&object["severity"]).to_uppercase(),
            kind: kind.clone(),
            metadata,
        })
    }

    /// The tag inside the kind object, e.g. `"socket_lineage"`.
    pub fn kind_type(&self) -> String {
        self.kind
            .get("type")
            .map(stringify)
            .unwrap_or_else(|| "unknown".to_string())
    }
}

/// Python `str()` for the JSON values that appear in entity ids and tags.
pub fn stringify(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Null => "None".to_string(),
        other => other.to_string(),
    }
}

fn parse_timestamp(value: &Value) -> Result<DateTime<Utc>, String> {
    let raw = value
        .as_str()
        .ok_or("timestamp must be an ISO-8601 string")?;
    // Python's fromisoformat treats a comma as the decimal separator.
    let normalized = raw.replacen(',', ".", 1);
    let text = normalized.as_str();
    if let Ok(parsed) = DateTime::parse_from_rfc3339(text) {
        return Ok(truncate_to_micros(parsed.with_timezone(&Utc)));
    }
    // Offset without a colon (+0000), which RFC 3339 parsing rejects.
    for format in ["%Y-%m-%dT%H:%M:%S%.f%z", "%Y-%m-%d %H:%M:%S%.f%z"] {
        if let Ok(parsed) = DateTime::parse_from_str(text, format) {
            return Ok(truncate_to_micros(parsed.with_timezone(&Utc)));
        }
    }
    // Naive timestamps are assumed UTC, matching schema.py — every remaining
    // shape Python's fromisoformat accepts: space separators, minute or hour
    // precision, and the basic (separator-free) forms.
    for format in [
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M",
        "%Y%m%dT%H%M%S%.f",
        "%Y%m%d %H%M%S%.f",
    ] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(text, format) {
            return Ok(truncate_to_micros(naive.and_utc()));
        }
    }
    // Hour-only precision: chrono cannot parse a bare hour, so complete it.
    if text.len() == 13 && (text.as_bytes()[10] == b'T' || text.as_bytes()[10] == b' ') {
        let completed = format!("{text}:00");
        for format in ["%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M"] {
            if let Ok(naive) = NaiveDateTime::parse_from_str(&completed, format) {
                return Ok(naive.and_utc());
            }
        }
    }
    for format in ["%Y-%m-%d", "%Y%m%d"] {
        if let Ok(date) = NaiveDate::parse_from_str(text, format) {
            return Ok(date
                .and_hms_opt(0, 0, 0)
                .expect("midnight is valid")
                .and_utc());
        }
    }
    Err(format!("unparseable timestamp {raw:?}"))
}

/// Python datetime carries microseconds; chrono carries nanoseconds. idr-main
/// serializes `Utc::now()` at nanosecond precision, so without truncation the
/// two engines would disagree on ordering, gaps, and first-event identity.
fn truncate_to_micros(timestamp: DateTime<Utc>) -> DateTime<Utc> {
    let nanos = timestamp.timestamp_subsec_nanos();
    timestamp
        .with_nanosecond((nanos / 1000) * 1000)
        .expect("truncated nanoseconds are in range")
}

/// Parse newline-delimited IdrEvent JSON, naming the offending line on failure.
pub fn read_ndjson(text: &str) -> Result<Vec<RawEvent>, String> {
    let mut events = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line)
            .map_err(|error| format!("invalid event at line {}: {error}", index + 1))?;
        events.push(
            RawEvent::from_value(&value)
                .map_err(|error| format!("invalid event at line {}: {error}", index + 1))?,
        );
    }
    Ok(events)
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};
    use serde_json::json;

    use super::parse_timestamp;

    #[test]
    fn accepts_every_python_fromisoformat_shape() {
        // Each pair verified against datetime.fromisoformat on the reference
        // interpreter (see the review's timestamp-coverage finding).
        let cases = [
            ("2026-03-01T00:09:30+00:00", (0, 9, 30, 0)),
            ("2026-03-01T00:09:30Z", (0, 9, 30, 0)),
            ("2026-03-01T00:09:30+0000", (0, 9, 30, 0)),
            ("2026-03-01T00:09:30.123456789+00:00", (0, 9, 30, 123_456)), // ns -> us truncation
            ("2026-03-01T00:09:30,5", (0, 9, 30, 500_000)),
            ("2026-03-01 00:09:30", (0, 9, 30, 0)),
            ("2026-03-01T00:09", (0, 9, 0, 0)),
            ("20260301T000930", (0, 9, 30, 0)),
            ("2026-03-01T11", (11, 0, 0, 0)),
            ("2026-03-01 11", (11, 0, 0, 0)),
            ("2026-03-01", (0, 0, 0, 0)),
            ("20260301", (0, 0, 0, 0)),
        ];
        for (text, (hour, minute, second, micros)) in cases {
            let parsed = parse_timestamp(&json!(text)).unwrap_or_else(|e| panic!("{text}: {e}"));
            let want = Utc
                .with_ymd_and_hms(2026, 3, 1, hour, minute, second)
                .unwrap()
                + chrono::Duration::microseconds(micros);
            assert_eq!(parsed, want, "shape {text:?}");
        }
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse_timestamp(&json!("not-a-time")).is_err());
        assert!(parse_timestamp(&json!(1234)).is_err());
    }
}
