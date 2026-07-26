"""Regenerate src/idr_intelligence/data/attack_reference.json from the MITRE
ATT&CK Enterprise STIX bundle.

The engine's deterministic ATT&CK machinery (tactic order + kind->technique
mapping in attack.py) is the field idr-sentinel corroborates, so it must be
authoritative, not hand-guessed. This script distills the large STIX bundle
into a compact, committed reference: the canonical enterprise tactic order and
every current technique's id -> {name, tactics}. A test asserts the engine's
tables stay consistent with it.

Usage:
    python scripts/ground_attack_reference.py [BUNDLE_PATH]

BUNDLE_PATH defaults to $IDR_ATTACK_BUNDLE, then the local cti checkout. The
committed reference is what the engine and its tests use; this script only
needs the bundle when regenerating (e.g. after a MITRE ATT&CK release).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DEFAULT_BUNDLE = os.environ.get(
    "IDR_ATTACK_BUNDLE", "/home/lee/Desktop/cti/enterprise-attack/enterprise-attack.json"
)
OUT = Path(__file__).resolve().parent.parent / "src" / "idr_intelligence" / "data" / "attack_reference.json"


def _attack_id(obj: dict) -> str | None:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id")
    return None


def build_reference(bundle_path: str) -> dict:
    data = json.loads(Path(bundle_path).read_text())
    objects = data["objects"]
    by_id = {obj["id"]: obj for obj in objects}

    matrix = next(obj for obj in objects if obj["type"] == "x-mitre-matrix")
    tactic_order = [by_id[ref]["x_mitre_shortname"] for ref in matrix["tactic_refs"]]

    techniques: dict[str, dict] = {}
    for obj in objects:
        if obj["type"] != "attack-pattern" or obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        technique_id = _attack_id(obj)
        if not technique_id:
            continue
        tactics = [
            phase["phase_name"]
            for phase in obj.get("kill_chain_phases", [])
            if phase.get("kill_chain_name") == "mitre-attack"
        ]
        techniques[technique_id] = {"name": obj["name"], "tactics": tactics}

    return {
        "source": "MITRE ATT&CK Enterprise STIX",
        "spec_version": data.get("spec_version", "2.1"),
        "attack_spec_version": next((o.get("x_mitre_attack_spec_version") for o in objects if o.get("x_mitre_attack_spec_version")), None),
        "enterprise_tactic_order": tactic_order,
        "techniques": dict(sorted(techniques.items())),
    }


def main() -> None:
    bundle = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BUNDLE
    if not Path(bundle).is_file():
        raise SystemExit(f"MITRE bundle not found: {bundle}\nPass a path or set IDR_ATTACK_BUNDLE.")
    reference = build_reference(bundle)
    OUT.write_text(json.dumps(reference, indent=2, sort_keys=False) + "\n")
    print(f"wrote {OUT} — {len(reference['enterprise_tactic_order'])} tactics, {len(reference['techniques'])} techniques")


if __name__ == "__main__":
    main()
