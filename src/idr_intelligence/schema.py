from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

SEVERITY_WEIGHT = {
    "INFO": 0.05,
    "WARNING": 0.25,
    "HIGH": 0.65,
    "CRITICAL": 0.85,
    "IMPOSSIBLE": 1.0,
}

KIND_PRIOR = {
    "socket_lineage": 0.12,
    "suspicious_beacon": 0.72,
    "physics_anomaly": 0.68,
    "octet_reversal_detected": 0.74,
    "ntp_time_shift": 0.42,
    "hsts_time_manipulation": 0.78,
    "nvme_latency_anomaly": 0.52,
    "mac_flapping": 0.64,
    "rtc_clock_divergence": 0.38,
    "triage_classification": 0.60,
    "bgp_anomaly": 0.48,
    "impossible_state": 1.0,
}


@dataclass(frozen=True)
class IdrEvent:
    id: str
    timestamp: datetime
    source: str
    severity: str
    kind: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def kind_type(self) -> str:
        return str(self.kind.get("type", "unknown"))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "IdrEvent":
        required = {"id", "timestamp", "source", "severity", "kind"}
        missing = sorted(required - raw.keys())
        if missing:
            raise ValueError(f"missing IdrEvent fields: {', '.join(missing)}")
        if not isinstance(raw["kind"], dict) or "type" not in raw["kind"]:
            raise ValueError("kind must be a tagged object containing 'type'")
        metadata = raw.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object or null")
        return cls(
            id=str(raw["id"]),
            timestamp=_parse_timestamp(raw["timestamp"]),
            source=str(raw["source"]),
            severity=str(raw["severity"]).upper(),
            kind=dict(raw["kind"]),
            metadata=dict(metadata),
        )


def parse_events(rows: Iterable[dict[str, Any]]) -> list[IdrEvent]:
    return sorted((IdrEvent.from_dict(row) for row in rows), key=lambda event: (event.timestamp, event.id))


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError("timestamp must be an ISO-8601 string")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
