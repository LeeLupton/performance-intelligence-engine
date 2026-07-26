//! Dependency-free structured logging for the bridge, mirroring the Python
//! engine's observability contract: JSON lines to **stderr**, off by default,
//! never touching the stdout finding. Gated by `--log-level` (or the
//! `IDR_RT_LOG` env var); reuses serde_json/chrono, already dependencies, so
//! no logging crate is pulled in.

use std::io::Write;

use serde_json::{Map, Value};

/// Severity, ordered by the numeric convention the Python engine uses
/// (debug < info < warning < error): an event is emitted when its severity is
/// at or above the configured threshold.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Debug,
    Info,
    Warn,
    Error,
}

impl Level {
    pub fn parse(text: &str) -> Result<Level, String> {
        match text.to_ascii_lowercase().as_str() {
            "debug" => Ok(Level::Debug),
            "info" => Ok(Level::Info),
            "warning" | "warn" => Ok(Level::Warn),
            "error" => Ok(Level::Error),
            other => Err(format!(
                "unknown log level {other:?} (expected debug|info|warning|error)"
            )),
        }
    }

    fn severity(self) -> u8 {
        match self {
            Level::Debug => 10,
            Level::Info => 20,
            Level::Warn => 30,
            Level::Error => 40,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Level::Debug => "DEBUG",
            Level::Info => "INFO",
            Level::Warn => "WARNING",
            Level::Error => "ERROR",
        }
    }
}

pub struct Logger {
    threshold: u8,
}

impl Logger {
    pub fn new(level: Level) -> Logger {
        Logger {
            threshold: level.severity(),
        }
    }

    /// Resolve the level from the flag, else `IDR_RT_LOG`, else `warning`.
    pub fn from_flag(flag: Option<&str>) -> Result<Logger, String> {
        let text = flag
            .map(str::to_string)
            .or_else(|| std::env::var("IDR_RT_LOG").ok())
            .unwrap_or_else(|| "warning".to_string());
        Ok(Logger::new(Level::parse(&text)?))
    }

    /// Emit one structured record if `level` clears the threshold. `fields`
    /// should be a JSON object; its keys are merged after ts/level/event.
    pub fn event(&self, level: Level, event: &str, fields: Value) {
        if level.severity() < self.threshold {
            return;
        }
        let mut record = Map::new();
        record.insert("ts".into(), Value::from(now_iso()));
        record.insert("level".into(), Value::from(level.label()));
        record.insert("event".into(), Value::from(event));
        if let Value::Object(extra) = fields {
            for (key, value) in extra {
                record.insert(key, value);
            }
        }
        let line =
            serde_json::to_string(&Value::Object(record)).unwrap_or_else(|_| "{}".to_string());
        let _ = writeln!(std::io::stderr(), "{line}");
    }
}

fn now_iso() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S").to_string()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{Level, Logger};

    #[test]
    fn level_parse_and_threshold() {
        assert!(Level::parse("warn").is_ok());
        assert!(Level::parse("WARNING").is_ok());
        assert!(Level::parse("nope").is_err());
        // Info threshold suppresses debug, admits info/warning/error.
        assert!(!Level::Info.severity_visible_from(Level::Debug));
        assert!(Level::Info.severity_visible_from(Level::Warn));
    }

    #[test]
    fn from_flag_defaults_to_warning() {
        // No flag, no env override in this process -> warning threshold, so an
        // info event is suppressed (verified structurally via severity()).
        let logger = Logger::from_flag(None).unwrap();
        assert!(logger.suppresses(super::Level::Info));
        assert!(!logger.suppresses(super::Level::Error));
    }

    #[test]
    fn merges_fields_without_panicking() {
        // Non-object field payloads are ignored, never panic.
        Logger::new(Level::Debug).event(Level::Info, "unit", json!(42));
        Logger::new(Level::Debug).event(Level::Info, "unit", json!({"k": "v"}));
    }

    // Test-only introspection helpers keep the production API minimal.
    impl Level {
        fn severity_visible_from(self, other: Level) -> bool {
            other.severity() >= self.severity()
        }
    }
    impl Logger {
        fn suppresses(&self, level: Level) -> bool {
            level.severity() < self.threshold
        }
    }
}
