"""Export real idr-sentinel BGP-anomaly audit records as IdrEvent NDJSON.

idr-sentinel writes every emitted (post-dedup) BGP anomaly to a JSONL audit log
(default /var/lib/idr-sentinel/state/anomalies.jsonl). Those records are the
raw BgpAnomaly payload, not the IdrEvent envelope, so this adapter normalizes
each into the canonical IdrEvent the engine ingests — synthesizing a stable id,
mapping the SystemTime stamp, and wrapping the anomaly kind. It is the concrete
bridge from live idr-main state to `idr-intelligence score/stream`.

The audit log is a firehose (tens of GB) of mostly calibration-tier anomalies,
so this reads a BOUNDED prefix of the file and can group by BGP prefix to form
coherent per-prefix windows. It never loads the whole file.

Usage:
    python scripts/export_sentinel_events.py \
        --input /var/lib/idr-sentinel/state/anomalies.jsonl \
        --limit 200000 --prefix 197.214.36.0/22 --out window.ndjson
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

_NS = uuid.UUID("6f4d9d2e-0000-4000-8000-000000000001")  # stable namespace for derived ids

# The sentinel serializes BgpAnomalyKind as an externally-tagged enum
# ({"observed_origin_flap": {...}}); the engine expects the idr-common internal
# tag ({"kind": "observed_origin_flap", ...}). None of the Observed* variants
# are the production subprefix hijack, so every exported event is calibration
# tier (production_bgp_anomaly feature = 0), which is the honest label.
_CALIBRATION_KINDS = {
    "observed_origin_flap",
    "observed_moas",
    "observed_squat_burst",
    "observed_squat_dormant",
    "observed_as_path_prepend",
    "observed_valley_free_violation",
    "observed_subprefix_more_specific",
    "observed_rpki_invalid",
    "rpki_transition",
    "bogon",
}


def _timestamp(record: dict) -> str:
    ts = record.get("ts")
    if isinstance(ts, dict) and "secs_since_epoch" in ts:
        seconds = ts["secs_since_epoch"] + ts.get("nanos_since_epoch", 0) / 1e9
        return datetime.fromtimestamp(seconds, tz=UTC).isoformat()
    if isinstance(ts, str):
        return ts
    return datetime.now(UTC).isoformat()


def to_idr_event(record: dict) -> dict | None:
    """Normalize one sentinel anomaly audit record into a canonical IdrEvent."""
    raw_kind = record.get("kind")
    if not isinstance(raw_kind, dict) or not raw_kind:
        return None
    tag = next(iter(raw_kind))
    inner = raw_kind[tag] if isinstance(raw_kind[tag], dict) else {}
    timestamp = _timestamp(record)
    prefix = record.get("prefix", "")
    origin = record.get("observed_origin_asn")
    # Deterministic id so re-exports are byte-stable and dedupe-able.
    event_id = str(uuid.uuid5(_NS, f"{prefix}|{timestamp}|{tag}|{origin}"))
    kind: dict = {
        "type": "bgp_anomaly",
        "kind": {"kind": tag, **inner},
        "prefix": prefix,
        "observed_origin_asn": origin,
        "confidence": "low" if tag in _CALIBRATION_KINDS else "high",
    }
    legitimate = record.get("legitimate_origin_asn")
    if legitimate is not None:
        kind["legitimate_origin_asn"] = legitimate
    return {
        "id": event_id,
        "timestamp": timestamp,
        "source": "sentinel_correlation",
        "severity": "WARNING" if tag in _CALIBRATION_KINDS else "HIGH",
        "kind": kind,
        "metadata": {"host": "bgp-collector"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="/var/lib/idr-sentinel/state/anomalies.jsonl")
    parser.add_argument("--limit", type=int, default=200000, help="max audit lines to scan (bounded read)")
    parser.add_argument("--prefix", default=None, help="keep only anomalies for this BGP prefix (one coherent window)")
    parser.add_argument("--max-events", type=int, default=64, help="cap exported events (a scoring window)")
    parser.add_argument("--out", default="-", help="output NDJSON path, or - for stdout")
    args = parser.parse_args()

    source = Path(args.input)
    if not source.is_file():
        raise SystemExit(f"audit log not found: {source}")

    exported: list[dict] = []
    with source.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle):
            if line_number >= args.limit or len(exported) >= args.max_events:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if args.prefix and record.get("prefix") != args.prefix:
                continue
            event = to_idr_event(record)
            if event is not None:
                exported.append(event)

    payload = "\n".join(json.dumps(event) for event in exported) + ("\n" if exported else "")
    if args.out == "-":
        sys.stdout.write(payload)
    else:
        Path(args.out).write_text(payload)
    print(f"exported {len(exported)} IdrEvents"
          + (f" for prefix {args.prefix}" if args.prefix else "")
          + (f" -> {args.out}" if args.out != "-" else ""), file=sys.stderr)


if __name__ == "__main__":
    main()
