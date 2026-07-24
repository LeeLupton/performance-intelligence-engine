"""Regenerate the cross-language parity fixtures for idr-intelligence-rt.

Run from the repository root with the project environment:

    .venv/bin/python rust/idr-intelligence-rt/tests/fixtures/generate.py

Everything here is deterministic (seeded init, simulator streams), and the
script asserts its own consistency before writing: the ONNX reference runner
must agree with the torch StreamingScorer, and the ranked entity probabilities
must be separated by comfortable gaps so cross-runtime float drift can never
reorder them. The committed outputs pin the Rust bridge to the Python engine.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from idr_intelligence.bounded_graph import GraphBudget
from idr_intelligence.export import OnnxStreamScorer, export_streaming_bundle
from idr_intelligence.features import FEATURE_DIM, project_event
from idr_intelligence.models import CampaignModel
from idr_intelligence.simulator import simulate_campaign
from idr_intelligence.streaming import StreamingScorer

FIXTURES = Path(__file__).resolve().parent
# Min separation between consecutive top-k node probabilities. Cross-runtime
# (torch vs onnxruntime vs tract) drift on these graphs is ~1e-7; 5e-5 keeps
# two orders of magnitude of margin so the ranking can never reorder.
RANKING_GAP_FLOOR = 5e-5


def event_rows(events):
    return [
        {
            "id": event.id,
            "timestamp": event.timestamp.isoformat(),
            "source": event.source,
            "severity": event.severity,
            "kind": event.kind,
            "metadata": event.metadata,
        }
        for event in events
    ]


def node_probability_gaps(model, scorer):
    """Recompute ranked node probabilities the way finding() does, for the gap check."""
    from idr_intelligence.graph import degree_normalize

    node_ids = tuple(scorer.entities)
    index_of = {entity: index for index, entity in enumerate(node_ids)}
    adjacency = np.eye(len(node_ids), dtype=np.float32)
    for (left, right), seen in scorer._edge_last_seen.items():
        weight = np.float32(1.0)
        if model.decay_half_life is not None:
            age = (scorer._previous_time - seen).total_seconds()
            weight = np.float32(0.5 ** (age / model.decay_half_life))
        adjacency[index_of[left], index_of[right]] = adjacency[index_of[right], index_of[left]] = weight
    adjacency = degree_normalize(adjacency)
    with torch.no_grad():
        outputs = torch.cat([state.output for state in scorer.entities.values()], dim=0)
        node_state = model.temporal.norm(outputs).unsqueeze(0)
        active = torch.ones(1, len(node_ids), 1)
        output = model.relational_head(node_state, active, torch.from_numpy(adjacency).unsqueeze(0))
        probabilities = torch.sigmoid(output.node_logits[0]).numpy()
    # Only the top-k boundary decides the finding; deep-tail ties (structurally
    # identical entities share exact logits) never surface in related/evidence.
    from idr_intelligence.config import DEFAULT_CONFIG

    ranked = np.sort(probabilities)[::-1][: DEFAULT_CONFIG.scoring.top_k + 1]
    return np.diff(-ranked) if len(ranked) > 1 else np.array([np.inf])


def main() -> None:
    torch.manual_seed(20260723)
    model = CampaignModel(FEATURE_DIM, hidden_dim=24, state_dim=6, time_mode="time_aware", decay_half_life=3600.0)
    with torch.no_grad():
        model.temperature.fill_(1.35)
        model.cal_bias.fill_(-0.18)
    model.feature_stats = {
        "bin_edges": np.linspace(0.0, 1.0, 11).tolist(),
        "histograms": [[1] * 10 for _ in range(FEATURE_DIM)],
    }
    export_streaming_bundle(model, FIXTURES, model_version="fixture-v1")

    # Mixed multi-campaign streams: single-scenario streams are full of
    # structurally identical (isomorphic) entities whose probabilities tie
    # exactly, which no seed can fix. The mixes below have structurally
    # distinct top-k entities, verified by the gap floor.
    def mix(parts):
        events = []
        for label, seed, scenario in parts:
            events.extend(simulate_campaign(label, seed, scenario=scenario))
        return events

    cases = {
        "malicious": mix([(1, 11, "distractor"), (1, 8, "lateral_movement"), (1, 4, "stale_preamble")]),
        "benign": mix([(0, 7, "distractor"), (0, 2, "legit_update")]),
    }
    vectors = []
    for case, events in cases.items():
        ordered = sorted(events, key=lambda event: (event.timestamp, event.id))
        (FIXTURES / f"events_{case}.ndjson").write_text("\n".join(json.dumps(row) for row in event_rows(ordered)) + "\n")

        reference = StreamingScorer(model, model_version="fixture-v1")
        exported = OnnxStreamScorer(FIXTURES)
        for event in ordered:
            reference.ingest(event)
            exported.ingest(event)
        want = reference.finding()
        got = exported.finding()
        assert got.related_entities == want.related_entities, f"{case}: ORT runner reorders entities"
        assert abs(got.escalation_probability - want.escalation_probability) <= 1e-4, case
        assert got.feature_drift == want.feature_drift, case
        gaps = node_probability_gaps(model, reference)
        assert gaps.min() >= RANKING_GAP_FLOOR, f"{case}: ranked probabilities too close ({gaps.min():.2e}); reseed"
        (FIXTURES / f"expected_{case}.json").write_text(json.dumps(want.to_dict(), indent=2) + "\n")

        previous = None
        for event in ordered:
            delta = (event.timestamp - previous).total_seconds() if previous is not None else 0.0
            previous = event.timestamp
            projection = project_event(event, delta_seconds=delta)
            vectors.append({
                "event": event_rows([event])[0],
                "delta_seconds": delta,
                "entities": list(projection.entities),
                "edges": [list(edge) for edge in projection.edges],
                "features": projection.features.tolist(),
            })
    (FIXTURES / "feature_vectors.json").write_text(json.dumps(vectors, indent=2) + "\n")

    # Budget + suppression golden: bounded entity memory with an audited
    # eviction trail, and an ip: prefix suppressed out of the ranking.
    ordered = sorted(cases["malicious"], key=lambda event: (event.timestamp, event.id))
    reference = StreamingScorer(model, budget=GraphBudget(max_nodes=6), model_version="fixture-v1")
    exported = OnnxStreamScorer(FIXTURES, budget=GraphBudget(max_nodes=6))
    for event in ordered:
        reference.ingest(event)
        exported.ingest(event)
    want = reference.finding(suppressions=["ip:"])
    got = exported.finding(suppressions=["ip:"])
    assert got.related_entities == want.related_entities
    assert [record.entity for record in exported.evictions] == [record.entity for record in reference.evictions]
    payload = want.to_dict()
    payload["evictions"] = [
        {"entity": record.entity, "last_seen": record.last_seen.isoformat(), "reason": record.reason}
        for record in reference.evictions
    ]
    (FIXTURES / "expected_budget.json").write_text(json.dumps(payload, indent=2) + "\n")

    print(f"fixtures written to {FIXTURES}")


if __name__ == "__main__":
    main()
