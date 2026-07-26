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
# (includes the two pre-compromise tactics the hand-typed order omitted). The
# relative order of every tactic the engine maps to is unchanged, so next-stage
# predictions are identical — but the order is now provably canonical.
TACTIC_ORDER = tuple(ATTACK_REFERENCE["enterprise_tactic_order"])

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

    Unlike the former presence lookup, this respects progression: an
    exfiltration-stage observation predicts impact next, whatever order the
    events arrived in, and a lone execution-stage event predicts persistence —
    not a stage the campaign already passed.
    """
    return next_stage_from_stages(observed_attack_stages(events))
