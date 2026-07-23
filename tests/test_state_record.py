"""Guards the canonical engineering state record against drift."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = json.loads((ROOT / "state.json").read_text())


def test_state_record_has_canonical_sections():
    assert STATE["version"] == 1
    for section in ("objective", "constraints", "inputs", "outputs", "architecture", "tasks", "evidence", "risks", "artifacts", "compressed_state"):
        assert section in STATE, f"state.json missing section: {section}"


def test_state_component_modules_exist():
    for component in STATE["architecture"]["components"]:
        assert (ROOT / component["module"]).is_file(), f"stale component path: {component['module']}"


def test_state_interfaces_are_versioned():
    interfaces = STATE["architecture"]["interfaces"]
    assert interfaces
    for interface in interfaces:
        assert interface["version"].startswith("v")
        assert interface["contract"]


def test_advisory_boundary_recorded_as_constraint():
    assert any("PanicResponse" in constraint for constraint in STATE["constraints"])


def test_evidence_entries_carry_results():
    for entry in STATE["evidence"]["commands_run"]:
        assert entry["cmd"] and entry["result"]
