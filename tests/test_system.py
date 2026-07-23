import json
from datetime import datetime, timezone

import numpy as np
import pytest
import torch

from idr_intelligence import cli
from idr_intelligence.config import DEFAULT_CONFIG, ENGINE_VERSION, load_config
from idr_intelligence.features import FEATURE_DIM
from idr_intelligence.registry import ModelManifest, SchemaMismatchError, feature_schema_hash
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
    assert finding.predicted_next_stage == "impact"
    assert {stage["tactic"] for stage in finding.observed_attack_stages} >= {"execution", "command-and-control", "exfiltration"}


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
    assert set(report["metrics"]) == {"static_baseline", "s6_only", "gnn_only", "s6_gnn", "s6_gnn_uniform_pool"}
    assert set(report["evidence_precision_at_5"]) == set(report["metrics"])
    assert all(0.0 <= value <= 1.0 for value in report["evidence_precision_at_5"].values())
    assert report["split"] == {"train": 12, "validation": 4, "test": 4}
    assert (tmp_path / "reports/demo.json").exists()
    saved = torch.load(tmp_path / "artifacts/hybrid_model.pt", weights_only=True)
    model = load_campaign_model(tmp_path / "artifacts/hybrid_model.pt")
    torch.testing.assert_close(dict(model.state_dict())["graph_head.0.weight"], saved["state_dict"]["graph_head.0.weight"])
    finding = score_events(simulate_campaign(1, 3), model)
    assert 0.0 <= finding.escalation_probability <= 1.0


def test_checkpoint_roundtrips_every_ablation_variant(tmp_path):
    batch = make_dataset(samples=20, seed=6, max_nodes=24, max_steps=8)
    variants = [
        {"use_s6": False, "use_gnn": False},
        {"use_s6": True, "use_gnn": False},
        {"use_s6": False, "use_gnn": True},
        {"use_s6": True, "use_gnn": True},
        {"use_s6": True, "use_gnn": True, "pooling": "uniform"},
    ]
    for index, kwargs in enumerate(variants):
        model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4, **kwargs)
        path = tmp_path / f"variant{index}.pt"
        save_checkpoint(model, path)
        loaded = load_campaign_model(path)
        assert (loaded.use_s6, loaded.use_gnn, loaded.pooling) == (model.use_s6, model.use_gnn, model.pooling)
        model.eval()
        loaded.eval()
        with torch.no_grad():
            expected = model(batch.sequences[:2], batch.mask[:2], batch.adjacency[:2])
            actual = loaded(batch.sequences[:2], batch.mask[:2], batch.adjacency[:2])
        torch.testing.assert_close(actual.graph_logit, expected.graph_logit)


def test_legacy_raw_state_dict_still_loads(tmp_path):
    modern = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4, pooling="uniform")
    legacy = dict(modern.state_dict())
    legacy_static = torch.nn.Sequential(
        torch.nn.Linear(FEATURE_DIM * 2, 12), torch.nn.GELU(), torch.nn.LayerNorm(12)
    ).state_dict()
    for key, value in legacy_static.items():
        legacy[f"static.{key}"] = value
    assert any(key.startswith("static.") for key in legacy)
    path = tmp_path / "legacy.pt"
    torch.save(legacy, path)
    loaded = load_campaign_model(path)
    assert loaded.hidden_dim == 12
    assert loaded.state_dim == 4
    assert loaded.feature_dim == FEATURE_DIM
    assert loaded.pooling == "uniform"
    torch.testing.assert_close(dict(loaded.state_dict())["node_head.weight"], legacy["node_head.weight"])


def test_default_toml_reproduces_default_config():
    loaded = load_config("configs/default.toml")
    assert loaded == DEFAULT_CONFIG
    assert loaded.config_hash() == DEFAULT_CONFIG.config_hash()


def test_config_hash_tracks_content():
    from idr_intelligence.config import EngineConfig, ModelConfig

    assert EngineConfig().config_hash() == EngineConfig().config_hash()
    assert EngineConfig(model=ModelConfig(hidden_dim=99)).config_hash() != EngineConfig().config_hash()


def test_checkpoint_manifest_guards_feature_schema(tmp_path):
    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4)
    path = tmp_path / "model.pt"
    save_checkpoint(model, path)
    payload = torch.load(path, weights_only=True)
    manifest = ModelManifest.from_dict(payload["manifest"])
    assert manifest.engine_version == ENGINE_VERSION
    assert manifest.feature_schema_hash == feature_schema_hash()
    assert load_campaign_model(path) is not None
    payload["manifest"]["feature_schema_hash"] = "0" * 32
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)
    with pytest.raises(SchemaMismatchError, match="feature schema"):
        load_campaign_model(tampered)


def test_manifestless_checkpoint_still_loads(tmp_path):
    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4)
    path = tmp_path / "v3.pt"
    save_checkpoint(model, path)
    payload = torch.load(path, weights_only=True)
    del payload["manifest"]
    torch.save(payload, path)
    loaded = load_campaign_model(path)
    assert loaded.pooling == "attention"


def test_finding_carries_provenance():
    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4)
    finding = score_events(simulate_campaign(label=1, seed=8), model)
    assert finding.engine_version == ENGINE_VERSION
    assert finding.feature_schema_hash == feature_schema_hash()
    assert datetime.fromisoformat(finding.scored_at).tzinfo is not None
    assert len(finding.related_entities) <= DEFAULT_CONFIG.scoring.top_k


def test_attack_table_covers_every_scored_kind():
    from idr_intelligence.attack import KIND_TO_ATTACK, TACTIC_ORDER
    from idr_intelligence.schema import KIND_PRIOR

    unmapped = set(KIND_PRIOR) - set(KIND_TO_ATTACK)
    assert unmapped == {"triage_classification"}
    for mapping in KIND_TO_ATTACK.values():
        assert mapping["tactic"] in TACTIC_ORDER
        assert mapping["technique"].startswith("T")


def test_attack_stages_are_time_ordered_with_evidence():
    from idr_intelligence.attack import observed_attack_stages

    events = simulate_campaign(label=1, seed=8)
    stages = observed_attack_stages(events)
    ids = [stage["first_event_id"] for stage in stages]
    by_id = {event.id: event for event in events}
    times = [by_id[event_id].timestamp for event_id in ids]
    assert times == sorted(times)
    assert stages[0]["tactic"] == "execution"
    assert stages[-1]["tactic"] == "exfiltration"


def test_next_stage_respects_progression_not_presence():
    from idr_intelligence.attack import predict_next_stage

    base = {"source": "kernel_ebpf", "severity": "HIGH", "metadata": {"host": "alpha"}}
    socket = IdrEvent.from_dict({**base, "id": "a" * 8 + "-0000-0000-0000-000000000001", "timestamp": "2026-06-18T12:00:00Z", "kind": {"type": "socket_lineage", "pid": 1}})
    nvme = IdrEvent.from_dict({**base, "id": "b" * 8 + "-0000-0000-0000-000000000002", "timestamp": "2026-06-18T11:00:00Z", "kind": {"type": "nvme_latency_anomaly", "device": "nvme0n1"}})
    assert predict_next_stage([socket]) == "persistence"
    assert predict_next_stage([nvme, socket]) == "impact"
    assert predict_next_stage([socket, nvme]) == "impact"
    assert predict_next_stage([]) == "unknown"
    impossible = IdrEvent.from_dict({**base, "id": "c" * 8 + "-0000-0000-0000-000000000003", "timestamp": "2026-06-18T13:00:00Z", "kind": {"type": "impossible_state"}})
    assert predict_next_stage([impossible]) == "kill-chain-complete"


def test_labels_are_random_but_split_safe():
    from idr_intelligence.training import _draw_labels

    labels = _draw_labels(40, seed=7, malicious_rate=0.3)
    assert not np.array_equal(labels, np.arange(40) % 2 == 1)
    for start, end in ((0, 24), (24, 32), (32, 40)):
        segment = labels[start:end]
        assert 0 < segment.sum() < end - start
    assert np.array_equal(labels, _draw_labels(40, seed=7, malicious_rate=0.3))


def test_operating_point_metrics_on_toys():
    from idr_intelligence.training import _precision_at_k, _recall_at_fpr

    labels = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    perfect = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert _recall_at_fpr(labels, perfect, 0.01) == 1.0
    assert _precision_at_k(labels, perfect, 2) == 1.0
    inverted = 1.0 - perfect
    assert _recall_at_fpr(labels, inverted, 0.01) == 0.0
    assert _precision_at_k(labels, inverted, 2) == 0.0


def test_ablation_runs_imbalanced(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = train_ablation(samples=24, epochs=1, seed=11, malicious_rate=0.3)
    assert report["malicious_rate"] == 0.3
    row = report["metrics"]["s6_gnn"]
    for key in ("recall_at_fpr_1pct", "recall_at_fpr_0p1pct", "precision_at_5"):
        assert 0.0 <= row[key] <= 1.0


def test_temperature_fit_never_worsens_validation_nll():
    from idr_intelligence.training import _fit_temperature, chronological_split

    batch = make_dataset(samples=20, seed=9, max_nodes=24, max_steps=8)
    _, validation, _ = chronological_split(batch)
    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4)
    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        logits = model(validation.sequences, validation.mask, validation.adjacency).graph_logit
        pre = loss_fn(logits, validation.labels).item()
    _fit_temperature(model, validation)
    with torch.no_grad():
        post = loss_fn(logits / model.temperature, validation.labels).item()
    assert post <= pre + 1e-9


def test_calibration_errors_zero_when_perfect():
    from idr_intelligence.training import _calibration_errors

    labels = np.array([0.0, 0.0, 1.0, 1.0])
    ece, mce = _calibration_errors(labels, np.array([0.0, 0.0, 1.0, 1.0]))
    assert ece == 0.0 and mce == 0.0
    ece_bad, mce_bad = _calibration_errors(labels, np.array([0.9, 0.9, 0.1, 0.1]))
    assert ece_bad > 0.5 and mce_bad > 0.5


def test_temperature_roundtrips_and_defaults(tmp_path):
    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4)
    model.temperature.fill_(2.5)
    path = tmp_path / "calibrated.pt"
    save_checkpoint(model, path)
    payload = torch.load(path, weights_only=True)
    assert payload["manifest"]["calibration"] == "temperature:2.500000"
    loaded = load_campaign_model(path)
    assert float(loaded.temperature.item()) == 2.5
    stripped = dict(payload["state_dict"])
    del stripped["temperature"]
    payload["state_dict"] = stripped
    del payload["manifest"]
    older = tmp_path / "older.pt"
    torch.save(payload, older)
    assert float(load_campaign_model(older).temperature.item()) == 1.0


def test_finding_reports_raw_and_calibrated():
    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4)
    model.temperature.fill_(2.0)
    finding = score_events(simulate_campaign(label=1, seed=8), model)
    assert finding.calibration == "temperature:2.000000"
    raw, calibrated = finding.raw_escalation_probability, finding.escalation_probability
    assert abs(calibrated - 0.5) <= abs(raw - 0.5) + 1e-9


def test_attention_ranking_receives_training_gradient():
    batch = make_dataset(samples=20, seed=4, max_nodes=24, max_steps=8)
    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4)
    loss = torch.nn.BCEWithLogitsLoss()(model(batch.sequences, batch.mask, batch.adjacency).graph_logit, batch.labels)
    loss.backward()
    assert model.attention is not None
    grad = model.attention.score.weight.grad
    assert grad is not None
    assert grad.abs().sum().item() > 0.0


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
    # Distinct id prefixes matter: simulator UUIDs all share "00000000", which
    # would make this assertion pass no matter which event were picked.
    base = {"source": "kernel_ebpf", "severity": "HIGH", "kind": {"type": "socket_lineage", "pid": 1, "dst_ip": "203.0.113.9"}, "metadata": {"host": "alpha"}}
    early = IdrEvent.from_dict({**base, "id": "aaaaaaaa-0000-0000-0000-000000000001", "timestamp": "2026-06-18T12:00:00Z"})
    late = IdrEvent.from_dict({**base, "id": "bbbbbbbb-0000-0000-0000-000000000002", "timestamp": "2026-06-18T13:00:00Z"})
    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4)
    forward = score_events([early, late], model)
    reversed_order = score_events([late, early], model)
    assert forward.campaign_id == reversed_order.campaign_id == "idr-campaign-aaaaaaaa"


def test_padding_nodes_stay_finite():
    batch = make_dataset(samples=20, seed=4, max_nodes=24, max_steps=8)
    for use_s6, use_gnn in ((False, False), (True, True)):
        model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4, use_s6=use_s6, use_gnn=use_gnn)
        output = model(batch.sequences[:2], batch.mask[:2], batch.adjacency[:2])
        assert torch.isfinite(output.graph_logit).all()
        assert torch.isfinite(output.node_logits).all()
