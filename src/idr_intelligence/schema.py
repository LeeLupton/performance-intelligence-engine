"""Canonical IdrEvent shape, validation, and scoring priors."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .config import DEFAULT_KIND_PRIOR, DEFAULT_SEVERITY_WEIGHT

SEVERITY_WEIGHT = DEFAULT_SEVERITY_WEIGHT
KIND_PRIOR = DEFAULT_KIND_PRIOR


@dataclass(frozen=True)
class IdrEvent:
    """One validated event in the canonical shape serialized by idr_common::IdrEvent."""

    id: str
    timestamp: datetime
    source: str
    severity: str
    kind: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def kind_type(self) -> str:
        """The tag inside the kind object, e.g. "socket_lineage"."""
        return str(self.kind.get("type", "unknown"))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> IdrEvent:
        """Validate one decoded JSON object; raise ValueError naming what's wrong."""
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
    """Validate many raw objects and return them in chronological order."""
    return sorted((IdrEvent.from_dict(row) for row in rows), key=lambda event: (event.timestamp, event.id))


@dataclass(frozen=True)
class LabeledWindow:
    """One labeled scoring window of real events — the training unit for W8 ingestion."""

    window_id: str
    label: int
    events: tuple[IdrEvent, ...]

    @property
    def start(self) -> datetime:
        return self.events[0].timestamp

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LabeledWindow:
        """Validate one decoded window object; raise ValueError naming what's wrong."""
        missing = sorted({"window_id", "label", "events"} - raw.keys())
        if missing:
            raise ValueError(f"missing LabeledWindow fields: {', '.join(missing)}")
        if raw["label"] not in (0, 1):
            raise ValueError("label must be 0 or 1")
        if not isinstance(raw["events"], list) or not raw["events"]:
            raise ValueError("events must be a non-empty list")
        return cls(
            window_id=str(raw["window_id"]),
            label=int(raw["label"]),
            events=tuple(parse_events(raw["events"])),
        )


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value)
    else:
        raise ValueError("timestamp must be an ISO-8601 string")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
