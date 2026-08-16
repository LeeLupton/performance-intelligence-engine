"""Deterministic MITRE ATT&CK mapping for observed kinds and next-stage prediction.

This intentionally stays a rule table, not a model: the next-stage field is the
one idr-sentinel corroborates, so it must remain auditable. The table gives each
event kind a primary enterprise tactic and a representative technique; mapping
choices are domain judgments recorded here, not learned artifacts.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from .schema import IdrEvent


def _load_attack_reference() -> dict[str, Any]:
    """The committed MITRE ATT&CK reference (canonical tactic order + techniques).

    Distilled from the MITRE ATT&CK Enterprise STIX bundle by
    scripts/ground_attack_reference.py, so the deterministic machinery
    idr-sentinel corroborates is authoritative rather than hand-guessed.
    """
    with resources.files("idr_intelligence").joinpath("data/attack_reference.json").open(encoding="utf-8") as handle:
        return json.load(handle)


ATTACK_REFERENCE = _load_attack_reference()

# Canonical enterprise kill-chain order, straight from the MITRE ATT&CK matrix
# (includes the two pre-compromise tactics the hand-typed order omitted).
TACTIC_ORDER = tuple(ATTACK_REFERENCE["enterprise_tactic_order"])

KIND_TO_ATTACK = {
    # Corrected for SEMANTIC fit against real MITRE ATT&CK (audit in
    # reports/AUDIT.md): each technique describes what the sensor actually
    # observes, not a tactically-adjacent guess. Every entry is validated
    # technique- and tactic-consistent against data/attack_reference.json by
    # validate_mapping_against_reference().
    "socket_lineage": {"tactic": "command-and-control", "technique": "T1071"},        # process opens an outbound app-layer socket (beaconing lineage)
    "suspicious_beacon": {"tactic": "command-and-control", "technique": "T1102"},     # unsigned binary beaconing to a high-trust web service (carrier)
    "bgp_anomaly": {"tactic": "collection", "technique": "T1557"},                    # routing interception / sinkhole = adversary-in-the-middle (parent)
    "ntp_time_shift": {"tactic": "defense-evasion", "technique": "T1562"},            # rogue-NTP clock shift to defeat time-based checks
    "hsts_time_manipulation": {"tactic": "defense-evasion", "technique": "T1553"},    # expired cert accepted via time rollback = subvert trust controls
    "nvme_latency_anomaly": {"tactic": "collection", "technique": "T1005"},           # bulk local-disk read footprint (the sensor sees I/O, not egress)
    "mac_flapping": {"tactic": "collection", "technique": "T1557.002"},               # ARP cache poisoning (MoCA/ARP MitM)
    "rtc_clock_divergence": {"tactic": "defense-evasion", "technique": "T1562"},       # software clock vs hardware RTC divergence = impair defenses
    "physics_anomaly": {"tactic": "collection", "technique": "T1557"},                # TTL/RTT single-hop intercept on a high-trust path = adversary-in-the-middle
    "octet_reversal_detected": {"tactic": "command-and-control", "technique": "T1001"},  # DNS PTR octet reversal hides the C2 destination = data obfuscation
    "igmp_trigger": {"tactic": "command-and-control", "technique": "T1205"},          # IGMPv3 magic-packet wake for a dormant implant = traffic signaling
    "quic_heartbeat": {"tactic": "command-and-control", "technique": "T1071.001"},    # QUIC/UDP-443 beacon disguised as web traffic = application layer protocol
    # Deliberately unmapped:
    #   triage_classification  — a correlator meta-event, not an adversary action.
    #   impossible_state       — the sentinel's OWN confirmation verdict; no ATT&CK
    #     technique means "my correlator fired", so any mapping (it was T1499
    #     Endpoint Denial of Service) is an analytic error in an intel product.
    #   igmp_quic_correlation  — the sentinel's OWN correlation of igmp_trigger +
    #     quic_heartbeat within a time window; same category as triage_classification,
    #     not an independently observed adversary technique.
    #   panic_response         — the sentinel's OWN countermeasure action (what the
    #     defender did), not anything the adversary did; mapping it would describe
    #     our response as their technique.
}

def technique_name(technique_id: str) -> str | None:
    """Human-readable MITRE name for a technique id, or None if unknown."""
    entry = ATTACK_REFERENCE["techniques"].get(technique_id)
    return entry["name"] if entry else None


def validate_mapping_against_reference() -> list[str]:
    """Inconsistencies between KIND_TO_ATTACK and the MITRE reference (empty = consistent).

    Each mapped kind must name a real, current technique whose real MITRE
    tactics include the tactic the engine assigns it — so the auditable
    next-stage field can never drift from the authoritative dataset unnoticed.
    """
    problems: list[str] = []
    techniques = ATTACK_REFERENCE["techniques"]
    tactics = set(ATTACK_REFERENCE["enterprise_tactic_order"])
    for kind, mapping in KIND_TO_ATTACK.items():
        tactic, technique = mapping["tactic"], mapping["technique"]
        if tactic not in tactics:
            problems.append(f"{kind}: tactic {tactic!r} is not a MITRE enterprise tactic")
        real = techniques.get(technique)
        if real is None:
            problems.append(f"{kind}: technique {technique} is not a current MITRE technique")
        elif tactic not in real["tactics"]:
            problems.append(f"{kind}: {technique} assigned {tactic!r}, but MITRE tactics are {real['tactics']}")
    return problems


def stage_record(kind_type: str, first_event_id: str) -> dict[str, Any] | None:
    """The ATT&CK stage a mapped kind contributes, enriched with the MITRE name.

    Single source of truth for the stage dict, shared by the batch, streaming,
    and ONNX-runner paths so the finding's `observed_attack_stages` shape can
    never drift between them. Returns None for unmapped kinds.
    """
    mapping = KIND_TO_ATTACK.get(kind_type)
    if mapping is None:
        return None
    return {
        "tactic": mapping["tactic"],
        "technique": mapping["technique"],
        "technique_name": technique_name(mapping["technique"]),
        "kind_type": kind_type,
        "first_event_id": first_event_id,
    }


def observed_attack_stages(events: list[IdrEvent]) -> tuple[dict[str, Any], ...]:
    """Timestamp-ordered attack-stage observations with evidence, one per kind.

    Deduplication is by event kind, not by (tactic, technique): each mapped kind
    contributes at most one stage entry, anchored to the first event that
    exhibited it. Distinct kinds sharing a tactic/technique (e.g. ntp_time_shift
    and rtc_clock_divergence both map to defense-evasion/T1562) therefore each
    produce their own entry.
    """
    stages: dict[str, dict[str, Any]] = {}
    for event in sorted(events, key=lambda item: (item.timestamp, item.id)):
        if event.kind_type in stages:
            continue
        record = stage_record(event.kind_type, event.id)
        if record is not None:
            stages[event.kind_type] = record
    return tuple(stages.values())


def next_stage_from_stages(stages: tuple[dict[str, Any], ...]) -> str:
    """Next unobserved kill-chain tactic after the furthest tactic in `stages`.

    Shared by batch (predict_next_stage over events) and the streaming scorer
    (which accumulates stage observations incrementally).
    """
    observed_indices = {TACTIC_ORDER.index(stage["tactic"]) for stage in stages}
    if not observed_indices:
        return "unknown"
    for index in range(max(observed_indices) + 1, len(TACTIC_ORDER)):
        if index not in observed_indices:
            return TACTIC_ORDER[index]
    return "kill-chain-complete"


def predict_next_stage(events: list[IdrEvent]) -> str:
    """Next unobserved kill-chain tactic after the furthest tactic observed.

    Progression-based, not presence-based: prediction advances past the
    furthest tactic already seen, whatever order the events arrived in, rather
    than reporting an earlier stage the campaign has already moved beyond.
    """
    return next_stage_from_stages(observed_attack_stages(events))
