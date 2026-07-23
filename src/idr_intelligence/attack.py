"""Deterministic MITRE ATT&CK mapping for observed kinds and next-stage prediction.

This intentionally stays a rule table, not a model: the next-stage field is the
one idr-sentinel corroborates, so it must remain auditable. The table gives each
event kind a primary enterprise tactic and a representative technique; mapping
choices are domain judgments recorded here, not learned artifacts.
"""

from __future__ import annotations

from typing import Any

from .schema import IdrEvent

TACTIC_ORDER = (
    "initial-access",
    "execution",
    "persistence",
    "privilege-escalation",
    "defense-evasion",
    "credential-access",
    "discovery",
    "lateral-movement",
    "collection",
    "command-and-control",
    "exfiltration",
    "impact",
)

KIND_TO_ATTACK = {
    "socket_lineage": {"tactic": "execution", "technique": "T1059"},
    "suspicious_beacon": {"tactic": "command-and-control", "technique": "T1071"},
    "bgp_anomaly": {"tactic": "collection", "technique": "T1557"},
    "ntp_time_shift": {"tactic": "defense-evasion", "technique": "T1562"},
    "hsts_time_manipulation": {"tactic": "credential-access", "technique": "T1557"},
    "nvme_latency_anomaly": {"tactic": "exfiltration", "technique": "T1041"},
    "mac_flapping": {"tactic": "collection", "technique": "T1557.002"},
    "rtc_clock_divergence": {"tactic": "defense-evasion", "technique": "T1562"},
    "physics_anomaly": {"tactic": "impact", "technique": "T1495"},
    "octet_reversal_detected": {"tactic": "defense-evasion", "technique": "T1027"},
    "impossible_state": {"tactic": "impact", "technique": "T1499"},
    # triage_classification is a correlator meta-event, deliberately unmapped
}


def observed_attack_stages(events: list[IdrEvent]) -> tuple[dict[str, Any], ...]:
    """Timestamp-ordered, deduplicated (tactic, technique) observations with evidence.

    Each mapped kind contributes at most one stage entry, anchored to the first
    event that exhibited it.
    """
    stages: dict[str, dict[str, Any]] = {}
    for event in sorted(events, key=lambda item: (item.timestamp, item.id)):
        mapping = KIND_TO_ATTACK.get(event.kind_type)
        if mapping is None or event.kind_type in stages:
            continue
        stages[event.kind_type] = {
            "tactic": mapping["tactic"],
            "technique": mapping["technique"],
            "kind_type": event.kind_type,
            "first_event_id": event.id,
        }
    return tuple(stages.values())


def predict_next_stage(events: list[IdrEvent]) -> str:
    """Next unobserved kill-chain tactic after the furthest tactic observed.

    Unlike the former presence lookup, this respects progression: an
    exfiltration-stage observation predicts impact next, whatever order the
    events arrived in, and a lone execution-stage event predicts persistence —
    not a stage the campaign already passed.
    """
    observed_indices = {
        TACTIC_ORDER.index(stage["tactic"]) for stage in observed_attack_stages(events)
    }
    if not observed_indices:
        return "unknown"
    for index in range(max(observed_indices) + 1, len(TACTIC_ORDER)):
        if index not in observed_indices:
            return TACTIC_ORDER[index]
    return "kill-chain-complete"
