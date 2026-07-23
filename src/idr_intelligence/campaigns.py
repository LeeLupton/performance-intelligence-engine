"""Stable campaign identity across scoring windows.

Today a campaign gets a fresh id per scoring call (derived from its first
event), so idr-sentinel cannot accumulate corroboration for one hypothesis
across windows. The registry matches each new window's durable entity
fingerprint against known campaigns by weighted Jaccard and either continues
the existing identity or mints a content-addressed new one. Deterministic
set-matching — auditable, no model in the loop.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Durable infrastructure weighs most: an executable hash or C2 domain is a far
# stronger continuity signal than a workstation name. Ephemeral per-window
# entities (process ids, sessions) are excluded entirely.
ENTITY_WEIGHTS = {
    "hash": 3.0,
    "domain": 2.5,
    "user": 2.5,
    "ip": 2.0,
    "cloud": 2.0,
    "prefix": 1.5,
    "asn": 1.5,
    "device": 1.0,
    "host": 0.5,
}

MATCH_THRESHOLD = 0.5


def fingerprint(entities: tuple[str, ...] | list[str]) -> dict[str, float]:
    """Durable entities with their continuity weights; ephemeral kinds dropped."""
    prints: dict[str, float] = {}
    for entity in entities:
        kind = entity.split(":", 1)[0]
        weight = ENTITY_WEIGHTS.get(kind)
        if weight:
            prints[entity] = weight
    return prints


def weighted_jaccard(left: dict[str, float], right: dict[str, float]) -> float:
    """Σ weights of shared entities / Σ weights of all entities (0 when either empty)."""
    if not left or not right:
        return 0.0
    shared = sum(weight for entity, weight in left.items() if entity in right)
    union = sum(left.values()) + sum(weight for entity, weight in right.items() if entity not in left)
    return shared / union


def content_address(prints: dict[str, float]) -> str:
    """Deterministic campaign id from the sorted durable entity set."""
    digest = hashlib.blake2s("\n".join(sorted(prints)).encode(), digest_size=6).hexdigest()
    return f"idr-campaign-{digest}"


@dataclass
class CampaignRecord:
    """One known campaign's identity and continuity metadata."""

    campaign_id: str
    fingerprint: dict[str, float]
    windows_observed: int
    first_seen: str
    last_seen: str


@dataclass
class CampaignRegistry:
    """Weighted-Jaccard matcher over known campaign fingerprints, JSON-persistable."""

    records: list[CampaignRecord] = field(default_factory=list)

    def match_or_register(self, entities: tuple[str, ...] | list[str], scored_at: str) -> tuple[str, bool, int]:
        """Return (campaign_id, continues_campaign, windows_observed) for a window.

        A match at or above MATCH_THRESHOLD continues the existing campaign
        (fingerprint unioned, window count incremented); otherwise a new
        content-addressed identity is registered. Ties resolve to the highest
        score, then the earliest-registered campaign, deterministically.
        """
        prints = fingerprint(entities)
        if not prints:
            return "idr-campaign-unfingerprinted", False, 1
        best_record, best_score = None, 0.0
        for record in self.records:
            score = weighted_jaccard(prints, record.fingerprint)
            if score > best_score:
                best_record, best_score = record, score
        if best_record is not None and best_score >= MATCH_THRESHOLD:
            best_record.fingerprint.update(prints)
            best_record.windows_observed += 1
            best_record.last_seen = scored_at
            return best_record.campaign_id, True, best_record.windows_observed
        record = CampaignRecord(
            campaign_id=content_address(prints),
            fingerprint=prints,
            windows_observed=1,
            first_seen=scored_at,
            last_seen=scored_at,
        )
        self.records.append(record)
        return record.campaign_id, False, 1

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"records": [asdict(record) for record in self.records]}, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "CampaignRegistry":
        """Load a registry; a missing file yields an empty registry."""
        file = Path(path)
        if not file.exists():
            return cls()
        raw = json.loads(file.read_text())
        return cls(records=[CampaignRecord(**record) for record in raw["records"]])
