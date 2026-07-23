import json
from datetime import timezone

import numpy as np
import pytest
import torch

from idr_intelligence import cli
from idr_intelligence.features import FEATURE_DIM
from idr_intelligence.graph import build_temporal_graph
from idr_intelligence.models import CampaignModel, load_campaign_model, save_checkpoint
from idr_intelligence.pipeline import score_events
from idr_intelligence.schema import IdrEvent
from idr_intelligence.simulator import simulate_campaign
from idr_intelligence.training import make_dataset, train_ablation


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


def test_schema_rejects_missing_fields():
    with pytest.raises(ValueError, match="severity") as excinfo:
        IdrEvent.from_dict({"id": "x", "timestamp": "2026-06-18T12:00:00Z", "source": "kernel_ebpf", "kind": {"type": "socket_lineage"}})
    assert "missing IdrEvent fields" in str(excinfo.value)


def test_schema_rejects_untagged_kind():
    with pytest.raises(ValueError, match="tagged object"):
        IdrEvent.from_dict({
            "id": "x", "timestamp": "2026-06-18T12:00:00Z", "source": "kernel_ebpf",
            "severity": "HIGH", "kind": {"pid": 12},
        })


def test_schema_rejects_non_object_metadata():
    with pytest.raises(ValueError, match="metadata"):
        IdrEvent.from_dict({
            "id": "x", "timestamp": "2026-06-18T12:00:00Z", "source": "kernel_ebpf",
            "severity": "HIGH", "kind": {"type": "socket_lineage"}, "metadata": ["not", "a", "dict"],
        })


def test_timestamps_normalize_to_utc():
    base = {"id": "x", "source": "kernel_ebpf", "severity": "HIGH", "kind": {"type": "socket_lineage"}}
    zulu = IdrEvent.from_dict({**base, "timestamp": "2026-06-18T12:00:00Z"})
    naive = IdrEvent.from_dict({**base, "timestamp": "2026-06-18T12:00:00"})
    offset = IdrEvent.from_dict({**base, "timestamp": "2026-06-18T14:00:00+02:00"})
    assert zulu.timestamp == naive.timestamp == offset.timestamp
    assert zulu.timestamp.tzinfo == timezone.utc


def test_train_ablation_smoke_and_checkpoint_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = train_ablation(samples=20, epochs=1, output="reports/demo.json")
    assert set(report["metrics"]) == {"static_baseline", "s6_only", "gnn_only", "s6_gnn"}
    assert report["split"] == {"train": 12, "validation": 4, "test": 4}
    assert (tmp_path / "reports/demo.json").exists()
    model = load_campaign_model(tmp_path / "artifacts/hybrid_model.pt")
    finding = score_events(simulate_campaign(1, 3), model)
    assert 0.0 <= finding.escalation_probability <= 1.0


def test_legacy_raw_state_dict_still_loads(tmp_path):
    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4)
    path = tmp_path / "legacy.pt"
    torch.save(model.state_dict(), path)
    loaded = load_campaign_model(path)
    assert loaded.hidden_dim == 12
    assert loaded.state_dim == 4
    assert loaded.feature_dim == FEATURE_DIM


def test_cli_score_roundtrip(tmp_path, monkeypatch, capsys):
    events_path = tmp_path / "events.ndjson"
    lines = []
    for event in simulate_campaign(1, 5):
        lines.append(json.dumps({
            "id": event.id, "timestamp": event.timestamp.isoformat(), "source": event.source,
            "severity": event.severity, "kind": event.kind, "metadata": event.metadata,
        }))
    events_path.write_text("\n".join(lines) + "\n")
    weights = tmp_path / "model.pt"
    save_checkpoint(CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4), weights)
    monkeypatch.setattr("sys.argv", ["idr-intelligence", "score", str(events_path), "--weights", str(weights)])
    cli.main()
    finding = json.loads(capsys.readouterr().out)
    assert finding["model_version"] == "model.pt"
    assert finding["graph_nodes"] > 0
    assert finding["evidence_event_ids"]


def test_cli_score_reports_bad_line(tmp_path, monkeypatch):
    events_path = tmp_path / "events.ndjson"
    events_path.write_text('{"id": "x"}\n')
    monkeypatch.setattr("sys.argv", ["idr-intelligence", "score", str(events_path)])
    with pytest.raises(SystemExit, match="line 1"):
        cli.main()


def test_campaign_id_is_order_independent():
    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4)
    events = simulate_campaign(label=1, seed=8)
    forward = score_events(list(events), model)
    reversed_order = score_events(list(reversed(events)), model)
    assert forward.campaign_id == reversed_order.campaign_id


def test_padding_nodes_stay_finite():
    batch = make_dataset(samples=20, seed=4, max_nodes=24, max_steps=8)
    for use_s6, use_gnn in ((False, False), (True, True)):
        model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4, use_s6=use_s6, use_gnn=use_gnn)
        output = model(batch.sequences[:2], batch.mask[:2], batch.adjacency[:2])
        assert torch.isfinite(output.graph_logit).all()
        assert torch.isfinite(output.node_logits).all()
