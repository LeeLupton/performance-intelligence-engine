import numpy as np

from idr_intelligence.features import FEATURE_DIM
from idr_intelligence.graph import build_temporal_graph
from idr_intelligence.models import CampaignModel
from idr_intelligence.pipeline import score_events
from idr_intelligence.schema import IdrEvent
from idr_intelligence.simulator import simulate_campaign
from idr_intelligence.training import make_dataset


def test_parses_canonical_idr_event():
    event = IdrEvent.from_dict({
        "id": "d34db33f-0000-0000-0000-000000000001",
        "timestamp": "2026-06-18T12:00:00Z",
        "source": "kernel_ebpf",
        "severity": "HIGH",
        "kind": {"type": "socket_lineage", "pid": 12, "dst_ip": "203.0.113.9"},
        "metadata": {"host": "alpha"},
    })
    assert event.kind_type == "socket_lineage"
    assert event.metadata["host"] == "alpha"


def test_graph_is_symmetric_and_contains_relational_entities():
    graph = build_temporal_graph(simulate_campaign(label=1, seed=2))
    assert any(node.startswith("process:") for node in graph.node_ids)
    assert any(node.startswith("asn:") for node in graph.node_ids)
    assert any(node.startswith("device:") for node in graph.node_ids)
    assert np.allclose(graph.adjacency, graph.adjacency.T)


def test_all_ablation_paths_forward_and_backpropagate():
    batch = make_dataset(samples=20, seed=4, max_nodes=24, max_steps=8)
    for use_s6, use_gnn in ((False, False), (True, False), (False, True), (True, True)):
        model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4, use_s6=use_s6, use_gnn=use_gnn)
        output = model(batch.sequences[:2], batch.mask[:2], batch.adjacency[:2])
        assert output.graph_logit.shape == (2,)
        output.graph_logit.sum().backward()
        assert any(parameter.grad is not None for parameter in model.parameters())


def test_finding_retains_primary_evidence():
    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4)
    finding = score_events(simulate_campaign(label=1, seed=8), model)
    assert 0.0 <= finding.escalation_probability <= 1.0
    assert finding.evidence_event_ids
    assert finding.related_entities
    assert finding.predicted_next_stage == "impact_or_exfiltration"
