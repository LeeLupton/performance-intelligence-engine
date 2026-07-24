//! The `idr_common::IdrEvent` wire envelope, validated the way `schema.py` does.

use chrono::{DateTime, NaiveDateTime, Utc};
use serde_json::{Map, Value};

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
        let metadata = match object.get("metadata") {
            None | Some(Value::Null) => Map::new(),
            Some(Value::Object(metadata)) => metadata.clone(),
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
    let text = value
        .as_str()
        .ok_or("timestamp must be an ISO-8601 string")?;
    if let Ok(parsed) = DateTime::parse_from_rfc3339(text) {
        return Ok(parsed.with_timezone(&Utc));
    }
    // Naive timestamps are assumed UTC, matching schema.py.
    NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S%.f")
        .map(|naive| naive.and_utc())
        .map_err(|error| format!("unparseable timestamp {text:?}: {error}"))
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
