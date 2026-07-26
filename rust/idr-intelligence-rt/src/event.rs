//! The `idr_common::IdrEvent` wire envelope, validated the way `schema.py` does.

use chrono::{DateTime, NaiveDate, Utc};
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
    parse_isoformat(raw).ok_or_else(|| format!("unparseable timestamp {raw:?}"))
}

/// Structural mirror of CPython 3.11+ `datetime.fromisoformat`, followed by
/// schema.py's normalization (naive -> assumed UTC, aware -> converted to
/// UTC). Acceptance parity with the reference interpreter is pinned
/// exhaustively by `tests/fixtures/timestamp_battery.json` — the enumeration
/// of strftime formats this replaces under-accepted by the hundreds.
///
/// Grammar (verified against the reference interpreter): date is
/// `YYYY-MM-DD` | `YYYYMMDD` | `YYYY-Www[-D]` | `YYYYWww[D]`; when a time
/// follows, it is separated by exactly one character (any); time is
/// `HH[:MM[:SS]]` or `HH[MM[SS]]` with an optional `.`/`,` fraction only
/// after seconds (zero fraction digits allowed, truncated to microseconds);
/// hour 24 rolls to next-day midnight when minutes/seconds/fraction are zero;
/// the offset is `Z` (uppercase only) or a sign plus the same time grammar
/// (hour <= 23). Python datetimes span years 1..=9999, so conversions past
/// either end are rejected like Python's OverflowError.
/// A parsed date, with calendar validation deferred: the reference parser
/// increments the RAW day number on an hour-24 rollover before validating, so
/// "2026-03-00T24:00:00" is 2026-03-01 while "2026-03-00T11:00" is invalid.
enum IsoDate {
    Ymd(i32, u32, u32),
    Week(NaiveDate),
}

fn parse_isoformat(text: &str) -> Option<DateTime<Utc>> {
    let bytes = text.as_bytes();
    let date_len = find_datetime_separator(bytes)?;
    let iso_date = parse_iso_date(&bytes[..date_len.min(bytes.len())])?;
    let (mut hour, minute, second, micros, offset) = if bytes.len() == date_len {
        (0, 0, 0, 0, None)
    } else {
        // Exactly one separator character (any character), then the time.
        let rest = &text[date_len..];
        let mut chars = rest.chars();
        chars.next()?;
        parse_iso_time(chars.as_str())?
    };
    let rollover = hour == 24;
    if rollover {
        if minute != 0 || second != 0 || micros != 0 {
            return None;
        }
        hour = 0;
    }
    let date = match iso_date {
        IsoDate::Week(date) => {
            if rollover {
                date.succ_opt()?
            } else {
                date
            }
        }
        IsoDate::Ymd(year, month, day) => {
            if rollover && day == 0 {
                NaiveDate::from_ymd_opt(year, month, 1)?
            } else if rollover {
                NaiveDate::from_ymd_opt(year, month, day)?.succ_opt()?
            } else {
                NaiveDate::from_ymd_opt(year, month, day)?
            }
        }
    };
    let mut naive = date.and_hms_micro_opt(hour, minute, second, micros)?;
    if let Some((offset_seconds, offset_micros)) = offset {
        naive = naive
            - chrono::Duration::seconds(offset_seconds)
            - chrono::Duration::microseconds(offset_micros);
    }
    in_python_range(naive)
}

/// Python datetimes span years 1..=9999; schema.py's astimezone(UTC) raises
/// OverflowError past either end, so the mirror rejects those instants.
fn in_python_range(naive: chrono::NaiveDateTime) -> Option<DateTime<Utc>> {
    use chrono::Datelike;
    if !(1..=9999).contains(&naive.year()) {
        return None;
    }
    Some(naive.and_utc())
}

/// Structural mirror of CPython's `_find_isoformat_datetime_separator`: the
/// date length is decided up front by documented lookahead heuristics — there
/// is no backtracking. Notably "YYYY-Www-<digit><digit>" resolves to the
/// no-day form (separator is the hyphen at 8), and for basic week dates the
/// parity of the trailing digit run decides between YYYYWww and YYYYWwwD.
fn find_datetime_separator(b: &[u8]) -> Option<usize> {
    let len = b.len();
    if len == 7 {
        return Some(7);
    }
    if len < 8 {
        return Some(len); // shorter strings fail in the date parser itself
    }
    if b[4] == b'-' {
        if b[5] == b'W' {
            if len > 8 && b[8] == b'-' {
                if len == 9 {
                    return None; // "YYYY-Www-" is explicitly invalid
                }
                if len > 10 && b[10].is_ascii_digit() {
                    return Some(8);
                }
                return Some(10);
            }
            Some(8)
        } else {
            Some(10)
        }
    } else if b[4] == b'W' {
        let mut index = 7;
        while index < len && b[index].is_ascii_digit() {
            index += 1;
        }
        if index < 9 {
            return Some(index);
        }
        if index % 2 == 0 { Some(7) } else { Some(8) }
    } else {
        Some(8)
    }
}

/// Mirror of `_parse_isoformat_date`, applied to the exact substring the
/// separator finder selected: calendar or week date, with the dash usage
/// required to be internally consistent.
fn parse_iso_date(b: &[u8]) -> Option<IsoDate> {
    let year = fixed_digits(b, 0, 4)? as i32;
    let has_sep = b.len() > 4 && b[4] == b'-';
    let mut pos = 4 + usize::from(has_sep);
    if b.get(pos) == Some(&b'W') {
        pos += 1;
        let week = fixed_digits(b, pos, 2)?;
        pos += 2;
        let mut day = 1;
        if b.len() > pos {
            if (b[pos] == b'-') != has_sep {
                return None; // inconsistent use of dash separator
            }
            pos += usize::from(has_sep);
            day = fixed_digits(b, pos, 1)?;
            if b.len() != pos + 1 {
                return None;
            }
        }
        iso_week_date(year, week, day).map(IsoDate::Week)
    } else {
        let month = fixed_digits(b, pos, 2)?;
        pos += 2;
        if b.len() > pos {
            if (b[pos] == b'-') != has_sep {
                return None;
            }
        } else if has_sep {
            return None; // "YYYY-MM" has a dash but no day
        }
        pos += usize::from(has_sep);
        let day = fixed_digits(b, pos, 2)?;
        if b.len() != pos + 2 {
            return None;
        }
        Some(IsoDate::Ymd(year, month, day))
    }
}

fn iso_week_date(year: i32, week: u32, day: u32) -> Option<NaiveDate> {
    use chrono::Weekday::*;
    let weekday = match day {
        1 => Mon,
        2 => Tue,
        3 => Wed,
        4 => Thu,
        5 => Fri,
        6 => Sat,
        7 => Sun,
        _ => return None,
    };
    NaiveDate::from_isoywd_opt(year, week, weekday)
}

type TimeParts = (u32, u32, u32, u32, Option<(i64, i64)>);

fn parse_iso_time(time: &str) -> Option<TimeParts> {
    if time.len() < 2 {
        return None;
    }
    // First-of-any scan (the C accelerator's behavior, probe-verified:
    // "11W-05" parses but "11Z-05" does not — 'Z' found first is a malformed
    // tz when anything follows it, while 'W' is a tolerable junk character).
    let tz_start = time.find(['Z', '+', '-']);
    let (base, tz) = match tz_start {
        Some(position) => (&time[..position], Some(&time[position..])),
        None => (time, None),
    };
    let (hour, minute, second, micros) = parse_hh_mm_ss_ff(base.as_bytes(), tz.is_some())?;
    // Time components are range-checked (hour 24 is the caller's special case).
    if hour > 24 || minute > 59 || second > 59 {
        return None;
    }
    let offset = match tz {
        None => None,
        Some("Z") => Some((0, 0)),
        Some(tz) => {
            let sign: i64 = match tz.as_bytes()[0] {
                b'+' => 1,
                b'-' => -1,
                _ => return None, // 'Z' followed by more characters
            };
            // Offsets reuse the time grammar with no trailing-junk tolerance
            // and no per-component caps — the reference interpreter builds a
            // timedelta ("+09:90" is 10:30) and only requires the total to be
            // strictly inside a day.
            let (oh, om, os, of) = parse_hh_mm_ss_ff(&tz.as_bytes()[1..], false)?;
            let seconds = i64::from(oh) * 3600 + i64::from(om) * 60 + i64::from(os);
            if seconds * 1_000_000 + i64::from(of) >= 24 * 3600 * 1_000_000 {
                return None;
            }
            Some((sign * seconds, sign * i64::from(of)))
        }
    };
    Some((hour, minute, second, micros, offset))
}

/// The reference parser's time grammar, reconstructed mechanism-for-mechanism
/// (every rule below is pinned by the committed battery):
/// - `HH[:MM[:SS]]` or `HH[MM[SS]]`, with the separator style sticky — mixing
///   "11:0930" or "1109:30" fails;
/// - exactly one trailing character of ANY kind is tolerated, but only when an
///   offset follows ("11W+00" is 11:00+00:00; "11W" alone fails);
/// - a fraction (after seconds only) demands min(remaining, 6) DIGITS — so
///   ".5W+00" fails (quota 2 hits 'W') while ".123456:+09" succeeds (quota met,
///   ':' becomes the tolerated trailing character) — and extra digits truncate;
/// - basic-format seconds may be followed by bare fraction digits under the
///   same quota ("1109012345" is 11:09:01.2345).
fn parse_hh_mm_ss_ff(base: &[u8], offset_follows: bool) -> Option<(u32, u32, u32, u32)> {
    #[derive(PartialEq)]
    enum Mode {
        Unknown,
        Extended,
        Basic,
    }
    let len = base.len();
    let mut comps = [0u32; 3];
    let mut position = 0;
    let mut index = 0;
    let mut mode = Mode::Unknown;
    loop {
        comps[index] = fixed_digits(base, position, 2)?;
        position += 2;
        if position == len {
            return Some((comps[0], comps[1], comps[2], 0));
        }
        let c = base[position];
        position += 1;
        if position == len {
            // c is the single tolerated trailing character.
            return if offset_follows {
                Some((comps[0], comps[1], comps[2], 0))
            } else {
                None
            };
        }
        if index < 2 {
            if c == b':' && mode != Mode::Basic {
                mode = Mode::Extended;
                index += 1;
            } else if c.is_ascii_digit() && mode != Mode::Extended {
                position -= 1;
                mode = Mode::Basic;
                index += 1;
            } else {
                return None;
            }
        } else {
            // After seconds: only a fraction may continue.
            if c == b'.' || c == b',' {
                break;
            } else if c.is_ascii_digit() && mode != Mode::Extended {
                position -= 1;
                break;
            } else {
                return None;
            }
        }
    }
    // Fraction: min(remaining, 6) characters MUST be digits; further digits
    // truncate away; any remaining tail (of any length) is tolerated only
    // when an offset follows — unlike the exactly-one-character tolerance
    // after bare components.
    let quota = (len - position).min(6);
    let digits = fixed_digits(base, position, quota)?;
    position += quota;
    let micros = digits * 10u32.pow(6 - quota as u32);
    while position < len && base[position].is_ascii_digit() {
        position += 1;
    }
    if position == len || offset_follows {
        Some((comps[0], comps[1], comps[2], micros))
    } else {
        None
    }
}

fn fixed_digits(b: &[u8], start: usize, count: usize) -> Option<u32> {
    if b.len() < start + count {
        return None;
    }
    let mut value = 0;
    for &byte in &b[start..start + count] {
        if !byte.is_ascii_digit() {
            return None;
        }
        value = value * 10 + u32::from(byte - b'0');
    }
    Some(value)
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
