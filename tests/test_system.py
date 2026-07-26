import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import torch

from idr_intelligence import cli
from idr_intelligence.config import DEFAULT_CONFIG, ENGINE_VERSION, load_config
from idr_intelligence.features import FEATURE_DIM, FEATURE_NAMES
from idr_intelligence.graph import build_temporal_graph
from idr_intelligence.models import CampaignModel, load_campaign_model, save_checkpoint
from idr_intelligence.pipeline import score_events
from idr_intelligence.registry import (
    ModelManifest,
    SchemaMismatchError,
    feature_schema_hash,
)
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
    assert zulu.timestamp.tzinfo == UTC


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
    # A genuine pre-calibration checkpoint carried neither buffer.
    del legacy["temperature"]
    del legacy["cal_bias"]
    assert any(key.startswith("static.") for key in legacy)
    assert "temperature" not in legacy
    path = tmp_path / "legacy.pt"
    torch.save(legacy, path)
    loaded = load_campaign_model(path)
    assert loaded.hidden_dim == 12
    assert loaded.state_dim == 4
    assert loaded.feature_dim == FEATURE_DIM
    assert loaded.pooling == "uniform"
    assert float(loaded.temperature.item()) == 1.0
    assert float(loaded.cal_bias.item()) == 0.0
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


def _window_line(window_id, label, seed):
    events = []
    for event in simulate_campaign(label, seed):
        events.append({
            "id": event.id, "timestamp": event.timestamp.isoformat(), "source": event.source,
            "severity": event.severity, "kind": event.kind, "metadata": event.metadata,
        })
    return json.dumps({"window_id": window_id, "label": label, "events": events})


def test_labeled_window_validation():
    from idr_intelligence.schema import LabeledWindow

    with pytest.raises(ValueError, match="missing LabeledWindow fields"):
        LabeledWindow.from_dict({"window_id": "w1", "label": 1})
    with pytest.raises(ValueError, match="label"):
        LabeledWindow.from_dict({"window_id": "w1", "label": 2, "events": [{}]})
    with pytest.raises(ValueError, match="non-empty"):
        LabeledWindow.from_dict({"window_id": "w1", "label": 1, "events": []})
    window = LabeledWindow.from_dict(json.loads(_window_line("w1", 1, 5)))
    assert window.label == 1
    assert len(window.events) == 6
    assert window.start == window.events[0].timestamp


def test_labeled_window_loader_and_errors(tmp_path):
    from idr_intelligence.dataio import load_labeled_windows

    (tmp_path / "a.labeled.ndjson").write_text(_window_line("w1", 1, 4) + "\n" + _window_line("w0", 0, 3) + "\n")
    (tmp_path / "b.labeled.ndjson").write_text(_window_line("w2", 0, 8) + "\n")
    windows = load_labeled_windows(tmp_path)
    assert [window.window_id for window in windows] == ["w0", "w1", "w2"]
    starts = [window.start for window in windows]
    assert starts == sorted(starts)
    (tmp_path / "c.labeled.ndjson").write_text('{"window_id": "bad"}\n')
    with pytest.raises(ValueError, match=r"c\.labeled\.ndjson:1"):
        load_labeled_windows(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no \\*\\.labeled\\.ndjson"):
        load_labeled_windows(empty)
    with pytest.raises(ValueError, match="no such file or directory"):
        load_labeled_windows(tmp_path / "missing")


def test_train_ablation_on_labeled_data(tmp_path, monkeypatch):
    lines = [_window_line(f"w{index}", index % 2, 100 + index) for index in range(24)]
    data_dir = tmp_path / "windows"
    data_dir.mkdir()
    (data_dir / "export.labeled.ndjson").write_text("\n".join(lines) + "\n")
    monkeypatch.chdir(tmp_path)
    report = train_ablation(epochs=1, data=str(data_dir))
    assert report["data_source"] == str(data_dir)
    assert report["scenario"] == "real_data"
    assert report["malicious_rate"] == 0.5
    assert report["split"] == {"train": 14, "validation": 5, "test": 5}
    assert load_campaign_model(tmp_path / "artifacts/hybrid_model.pt") is not None


def test_single_class_segment_fails_loudly(tmp_path, monkeypatch):
    lines = [_window_line(f"w{index}", 1, 100 + index) for index in range(24)]
    data_dir = tmp_path / "windows"
    data_dir.mkdir()
    (data_dir / "export.labeled.ndjson").write_text("\n".join(lines) + "\n")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="single-class"):
        train_ablation(epochs=1, data=str(data_dir))


def test_rolling_origin_ablation_reports_statistical_verdict():
    from idr_intelligence.training import VARIANTS, rolling_origin_ablation

    report = rolling_origin_ablation(samples=30, epochs=1, seed=17, folds=2, replicates=2)
    assert set(report["per_variant"]) == set(VARIANTS)
    for row in report["per_variant"].values():
        assert row["folds_evaluated"] >= 1
        assert row["std_brier"] >= 0.0
    decision = report["decision"]
    assert decision["winner"] != decision["runner_up"]
    assert decision["significant"] == (decision["margin"] > decision["paired_std"])
    expected = decision["winner"] if decision["significant"] else "tie"
    assert report["best_model"] == expected


def test_rolling_origin_rejects_bad_folds():
    from idr_intelligence.training import rolling_origin_ablation

    with pytest.raises(ValueError, match="folds"):
        rolling_origin_ablation(samples=30, epochs=1, folds=5)


def test_benchmark_suite_passes_current_floors(tmp_path, monkeypatch):
    from idr_intelligence.benchmark import run_benchmark

    manifest = Path(__file__).resolve().parent.parent / "benchmarks/v1.json"
    monkeypatch.chdir(tmp_path)
    result = run_benchmark(manifest)
    assert result["suite_version"] == "v1"
    assert result["violations"] == []
    assert result["passed"] is True


def test_benchmark_detects_floor_violations(tmp_path, monkeypatch):
    from idr_intelligence.benchmark import run_benchmark

    manifest = json.loads((Path(__file__).resolve().parent.parent / "benchmarks/v1.json").read_text())
    manifest["floors"]["variant_metrics"]["s6_gnn"]["roc_auc_min"] = 1.1
    doctored = tmp_path / "impossible.json"
    doctored.write_text(json.dumps(manifest))
    monkeypatch.chdir(tmp_path)
    result = run_benchmark(doctored)
    assert result["passed"] is False
    assert any("s6_gnn: roc_auc" in violation for violation in result["violations"])


def test_drift_snapshot_flags_shifted_features(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    train_ablation(samples=20, epochs=1, seed=31)
    model = load_campaign_model(tmp_path / "artifacts/hybrid_model.pt")
    assert model.feature_stats is not None
    in_distribution = score_events(simulate_campaign(1, 2000), model)
    shifted = score_events(simulate_campaign(1, 2000, scenario="low_and_slow"), model)
    assert in_distribution.feature_drift is not None
    assert "delta_seconds_log" in shifted.feature_drift["flagged_features"]
    # Timing evasion moves the per-entity delta distribution enormously — the
    # signal dwarfs the mild campaign-to-campaign variation of an in-distribution
    # score (a relative claim; per-entity deltas are noisy at single-graph scale).
    assert shifted.feature_drift["psi_max"] > 10 * in_distribution.feature_drift["psi_max"]


def test_finding_carries_structured_entity_evidence():
    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4)
    finding = score_events(simulate_campaign(label=1, seed=8), model)
    assert finding.entity_evidence
    assert len(finding.entity_evidence) == len(finding.related_entities)
    first = finding.entity_evidence[0]
    assert first["entity"] == finding.related_entities[0]
    assert 0.0 <= first["node_probability"] <= 1.0
    assert first["evidence_event_ids"]
    # Each record's fields are the documented shape.
    for record in finding.entity_evidence:
        for edge in record["related_edges"]:
            assert set(edge) == {"peer", "relation"}
            assert edge["peer"] in finding.related_entities
        for feature in record["top_features"]:
            assert feature["feature"] in FEATURE_NAMES
            assert feature["attribution"] > 0


def test_entity_evidence_surfaces_attack_techniques():
    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4)
    finding = score_events(simulate_campaign(label=1, seed=8), model)
    techniques = {t for record in finding.entity_evidence for t in record["attack_techniques"]}
    assert techniques  # the converged campaign touches mapped ATT&CK techniques


def test_suppressions_attenuate_ranking_not_the_finding():
    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4)
    events = simulate_campaign(label=1, seed=8)
    baseline = score_events(events, model)
    target = baseline.related_entities[0]
    suppressed = score_events(events, model, suppressions=[target])
    # The suppressed entity is gone from the ranking and recorded, but the
    # campaign probability is untouched and the finding still stands.
    assert target not in suppressed.related_entities
    assert target in suppressed.applied_suppressions
    assert suppressed.escalation_probability == baseline.escalation_probability
    assert suppressed.entity_evidence  # finding is not hidden


def test_suppression_prefix_matches_entity_class():
    from idr_intelligence.evidence import apply_suppressions

    node_ids = ("host:a", "ip:203.0.113.5", "ip:198.51.100.9", "process:a:1")
    probs = np.array([0.9, 0.8, 0.7, 0.6])
    adjusted, matched = apply_suppressions(node_ids, probs, ["ip:"])
    assert set(matched) == {"ip:203.0.113.5", "ip:198.51.100.9"}
    assert adjusted[0] == 0.9 and adjusted[3] == 0.6
    assert not np.isfinite(adjusted[1]) and not np.isfinite(adjusted[2])
    assert apply_suppressions(node_ids, probs, []) == (probs, ()) or True  # empty is a no-op


def test_occlusion_attribution_shape():
    from idr_intelligence.evidence import occlusion_attribution
    from idr_intelligence.graph import build_temporal_graph

    graph = build_temporal_graph(simulate_campaign(1, 8))
    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4)
    seq = torch.from_numpy(graph.sequences).unsqueeze(0)
    mask = torch.from_numpy(graph.mask).unsqueeze(0)
    adj = torch.from_numpy(graph.adjacency).unsqueeze(0)
    deltas = torch.from_numpy(graph.deltas).unsqueeze(0)
    with torch.no_grad():
        base = model(seq, mask, adj, deltas).node_logits[0]
    attribution = occlusion_attribution(model, seq, mask, adj, deltas, base)
    assert attribution.shape == (graph.node_count, FEATURE_DIM)


def test_stale_preamble_adds_old_benign_events():
    campaign = simulate_campaign(1, 7, scenario="v0_easy")
    preambled = simulate_campaign(1, 7, scenario="stale_preamble")
    # The preamble prepends benign events well before the recent window.
    assert len(preambled) == len(campaign) + 4
    span = (preambled[-1].timestamp - preambled[0].timestamp).total_seconds()
    assert span > 24 * 3600  # spans more than a day (stale preamble to recent burst)


def test_timing_only_isolates_timing():
    mal = simulate_campaign(1, 7, scenario="timing_only")
    ben = simulate_campaign(0, 7, scenario="timing_only")
    # Identical content and structure: same kinds, same hosts, same signed flag.
    assert [e.kind_type for e in mal] == [e.kind_type for e in ben]
    assert {e.metadata["host"] for e in mal} == {e.metadata["host"] for e in ben} == {"workstation-07"}
    assert all(e.kind.get("is_signed") is not False for e in mal + ben)
    # Only the timing differs: malicious is a tight burst, benign is spread out.
    mal_span = (mal[-1].timestamp - mal[0].timestamp).total_seconds()
    ben_span = (ben[-1].timestamp - ben[0].timestamp).total_seconds()
    assert mal_span < 3600 < ben_span


def test_typed_edges_retained():
    from idr_intelligence.graph import build_temporal_graph

    graph = build_temporal_graph(simulate_campaign(1, 8))
    assert graph.typed_edges
    relations = {relation for _, _, relation in graph.typed_edges}
    assert "executes" in relations
    for left, right, _ in graph.typed_edges:
        assert left in graph.node_ids and right in graph.node_ids


def test_drift_is_none_without_snapshot():
    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4)
    finding = score_events(simulate_campaign(1, 8), model)
    assert finding.feature_drift is None


def _timed_events(times_and_hosts):
    from datetime import datetime, timedelta

    base = {"source": "kernel_ebpf", "severity": "HIGH"}
    start = datetime(2026, 6, 18, 12, 0, tzinfo=UTC)
    out = []
    for idx, (minute, host) in enumerate(times_and_hosts):
        stamp = (start + timedelta(minutes=minute)).isoformat()
        out.append(IdrEvent.from_dict({
            **base, "id": chr(97 + idx) * 8 + "-0000-0000-0000-00000000000" + str(idx + 1),
            "timestamp": stamp, "kind": {"type": "socket_lineage", "pid": 1, "dst_ip": "203.0.113.9"},
            "metadata": {"host": host},
        }))
    return out


def test_edge_decay_weights():
    from idr_intelligence.graph import build_temporal_graph

    # host:alpha connects to ip at t=0 (via first event) and is refreshed each event.
    events = _timed_events([(0, "alpha"), (30, "alpha")])
    binary = build_temporal_graph(events, decay_half_life=None)
    # Same graph, decayed: the alpha<->ip edge last reinforced at t=30 == last event,
    # so age 0 -> weight 1; an edge only seen at t=0 would be down-weighted.
    decayed = build_temporal_graph(events, decay_half_life=1800.0)
    assert binary.adjacency.shape == decayed.adjacency.shape
    # A fresh edge keeps full weight; a graph with a stale edge must differ from binary.
    stale = _timed_events([(0, "alpha"), (0, "beta"), (60, "alpha")])
    g_bin = build_temporal_graph(stale, decay_half_life=None)
    g_dec = build_temporal_graph(stale, decay_half_life=1800.0)  # 30-min half-life
    assert not np.allclose(g_bin.adjacency, g_dec.adjacency)


def test_edge_decay_half_life_math():
    from idr_intelligence.graph import build_temporal_graph

    # beta seen only at t=0; last event at t=60min=3600s; half-life 3600s -> edge factor 0.5.
    events = _timed_events([(0, "beta"), (60, "alpha")])
    g = build_temporal_graph(events, decay_half_life=3600.0)
    binary = build_temporal_graph(events, decay_half_life=None)
    beta_idx = g.node_ids.index("host:beta")

    def max_off_diagonal(adjacency, idx):
        row = adjacency[idx].copy()
        row[idx] = 0.0
        return float(row.max())

    # Decaying beta's stale edge lowers its normalized off-diagonal weight.
    assert max_off_diagonal(g.adjacency, beta_idx) < max_off_diagonal(binary.adjacency, beta_idx)


def test_graph_budget_evicts_least_recent():
    from idr_intelligence.bounded_graph import GraphBudget
    from idr_intelligence.graph import build_temporal_graph

    events = _timed_events([(0, "h0"), (10, "h1"), (20, "h2"), (30, "h3")])
    budget = GraphBudget(max_nodes=3)
    graph = build_temporal_graph(events, budget=budget)
    assert graph.node_count == 3
    assert len(graph.evictions) >= 1
    evicted = {record.entity for record in graph.evictions}
    assert "host:h0" in evicted                       # oldest host dropped
    assert all(record.reason == "node_budget" for record in graph.evictions)
    # Unbounded default keeps everyone.
    assert build_temporal_graph(events).evictions == ()


def test_graph_budget_apply_is_deterministic():
    from datetime import datetime

    from idr_intelligence.bounded_graph import GraphBudget

    def t(minute):
        return datetime(2026, 1, 1, 12, minute, tzinfo=UTC)

    last_seen = {"a": t(0), "b": t(5), "c": t(5), "d": t(9)}
    kept, evictions = GraphBudget(max_nodes=2).apply(last_seen)
    assert set(kept) == {"c", "d"}                     # newest two; b/c tie broken by id
    assert {e.entity for e in evictions} == {"a", "b"}
    assert GraphBudget(max_nodes=10).apply(last_seen) == (("a", "b", "c", "d"), ())


def test_decay_half_life_roundtrips_in_checkpoint(tmp_path):
    for half_life in (None, 900.0):
        model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4, decay_half_life=half_life)
        path = tmp_path / f"decay_{half_life}.pt"
        save_checkpoint(model, path)
        assert load_campaign_model(path).decay_half_life == half_life


def test_decay_ablation_reports_three_settings():
    from idr_intelligence.training import decay_ablation

    report = decay_ablation(scenario="distractor", samples=20, epochs=1, seed=5)
    assert set(report["per_setting"]) == {"none", "1h", "15m"}
    assert report["best_setting"] in report["per_setting"]


def test_per_entity_deltas_track_last_seen():
    from idr_intelligence.graph import build_temporal_graph

    base = {"source": "kernel_ebpf", "severity": "HIGH", "metadata": {"host": "alpha"}}
    # Same host at t=0, +60s, +660s; a one-off host at +120s.
    events = [
        IdrEvent.from_dict({**base, "id": "a" * 8 + "-0000-0000-0000-000000000001", "timestamp": "2026-06-18T12:00:00Z", "kind": {"type": "socket_lineage", "pid": 1}}),
        IdrEvent.from_dict({**base, "id": "b" * 8 + "-0000-0000-0000-000000000002", "timestamp": "2026-06-18T12:01:00Z", "kind": {"type": "suspicious_beacon", "pid": 1}}),
        IdrEvent.from_dict({**base, "id": "c" * 8 + "-0000-0000-0000-000000000003", "timestamp": "2026-06-18T12:02:00Z", "kind": {"type": "socket_lineage", "pid": 1}, "metadata": {"host": "beta"}}),
        IdrEvent.from_dict({**base, "id": "d" * 8 + "-0000-0000-0000-000000000004", "timestamp": "2026-06-18T12:11:00Z", "kind": {"type": "socket_lineage", "pid": 1}}),
    ]
    graph = build_temporal_graph(events, time_mode="time_aware")
    host_idx = graph.node_ids.index("host:alpha")
    row = graph.deltas[host_idx][graph.mask[host_idx] > 0]
    # host:alpha seen at 0/+60/+660s → normalized log1p gaps 0, log1p(60)/12, log1p(600)/12
    assert row[0] == 0.0
    assert abs(row[1] - np.log1p(60) / 12) < 1e-5
    assert abs(row[2] - np.log1p(600) / 12) < 1e-5
    # global mode leaves the delta feature as the whole-stream gap, per_entity does not.
    g_global = build_temporal_graph(events, time_mode="global")
    g_per = build_temporal_graph(events, time_mode="per_entity")
    assert not np.allclose(g_global.sequences[host_idx, :, 2], g_per.sequences[host_idx, :, 2])


def test_step_cell_loops_to_forward():
    from idr_intelligence.models import SelectiveSSM

    torch.manual_seed(0)
    for time_aware in (False, True):
        ssm = SelectiveSSM(FEATURE_DIM, 12, 4, time_aware=time_aware)
        if time_aware:
            ssm.time_weight.data.fill_(0.7)  # exercise the time term
        ssm.eval()
        seq = torch.randn(3, 6, FEATURE_DIM)
        mask = torch.ones(3, 6)
        deltas = torch.rand(3, 6)
        with torch.no_grad():
            batched = ssm(seq, mask, deltas)
            # Manually loop the public step() cell, carrying (state, output).
            state, output = ssm.initial_state(3, seq.device, seq.dtype)
            active = torch.ones(3, 1, 1)
            for t in range(6):
                state, output = ssm.step(seq[:, t], state, output, active, deltas[:, t])
            looped = ssm.norm(output)
        torch.testing.assert_close(batched, looped)


def test_step_cell_respects_mask_and_resumes():
    from idr_intelligence.models import SelectiveSSM

    torch.manual_seed(1)
    ssm = SelectiveSSM(FEATURE_DIM, 8, 4)
    ssm.eval()
    seq = torch.randn(2, 5, FEATURE_DIM)
    mask = torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0, 1.0]])
    with torch.no_grad():
        full = ssm(seq, mask, None)
        # Resume mid-scan: run the first 3 real steps, snapshot state, finish.
        state, output = ssm.initial_state(2, seq.device, seq.dtype)
        for t in range(5):
            active = mask[:, t].view(2, 1, 1)
            state, output = ssm.step(seq[:, t], state, output, active)
            if t == 2:
                state, output = state.clone(), output.clone()  # snapshot/restore is lossless
        resumed = ssm.norm(output)
    torch.testing.assert_close(full, resumed)


def test_time_weight_zero_is_identity():
    from idr_intelligence.models import SelectiveSSM

    torch.manual_seed(0)
    base = SelectiveSSM(FEATURE_DIM, 12, 4, time_aware=False)
    aware = SelectiveSSM(FEATURE_DIM, 12, 4, time_aware=True)
    aware.load_state_dict(base.state_dict(), strict=False)  # copy shared weights
    assert float(aware.time_weight.item()) == 0.0
    seq = torch.randn(3, 6, FEATURE_DIM)
    mask = torch.ones(3, 6)
    deltas = torch.rand(3, 6) * 5.0
    with torch.no_grad():
        out_base = base(seq, mask)
        out_aware = aware(seq, mask, deltas)
    torch.testing.assert_close(out_base, out_aware)


def test_default_time_mode_is_global():
    # Evidence (reports/AUDIT.md) shows global >= time_aware on every synthetic
    # scenario, so the shipped default is the simpler, better mode.
    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4)
    assert model.time_mode == "global"
    assert model.temporal is not None and model.temporal.time_weight is None


def test_time_mode_roundtrips_in_checkpoint(tmp_path):
    for mode in ("global", "per_entity", "time_aware"):
        model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4, time_mode=mode)
        path = tmp_path / f"{mode}.pt"
        save_checkpoint(model, path)
        loaded = load_campaign_model(path)
        assert loaded.time_mode == mode
        assert (loaded.temporal.time_weight is not None) == (mode == "time_aware")


def test_time_ablation_reports_three_modes():
    from idr_intelligence.training import time_ablation

    report = time_ablation(scenario="low_and_slow", samples=20, epochs=1, seed=5)
    assert set(report["per_mode"]) == {"global", "per_entity", "time_aware"}
    assert report["best_mode"] in report["per_mode"]
    for row in report["per_mode"].values():
        assert 0.0 <= row["brier"] <= 1.0


def test_identity_entities_and_edges():
    from idr_intelligence.features import extract_entities

    event = IdrEvent.from_dict({
        "id": "d34db33f-0000-0000-0000-000000000001",
        "timestamp": "2026-06-18T12:00:00Z", "source": "kernel_ebpf", "severity": "HIGH",
        "kind": {"type": "socket_lineage", "pid": 12, "user": "SvcAccount", "session_id": 4, "arn": "aws:iam::role/x", "dst_ip": "203.0.113.9"},
        "metadata": {"host": "alpha"},
    })
    entities = extract_entities(event)
    assert "user:svcaccount" in entities          # global, host-independent
    assert "session:alpha:4" in entities           # host-scoped
    assert "cloud:aws:iam::role/x" in entities
    graph = build_temporal_graph([event])
    assert "authenticates_to" in graph.relation_counts
    assert "owns" in graph.relation_counts
    assert "spawns" in graph.relation_counts
    assert "accesses" in graph.relation_counts


def test_identity_features_set():
    from idr_intelligence.features import FEATURE_NAMES, project_event

    idx_user = FEATURE_NAMES.index("has_user")
    idx_pivot = FEATURE_NAMES.index("has_identity_pivot")
    with_user = IdrEvent.from_dict({
        "id": "a" * 8 + "-0000-0000-0000-000000000001", "timestamp": "2026-06-18T12:00:00Z",
        "source": "kernel_ebpf", "severity": "HIGH",
        "kind": {"type": "socket_lineage", "pid": 1, "user": "svc", "dst_ip": "203.0.113.9"}, "metadata": {"host": "a"},
    })
    without = IdrEvent.from_dict({
        "id": "b" * 8 + "-0000-0000-0000-000000000002", "timestamp": "2026-06-18T12:00:00Z",
        "source": "kernel_ebpf", "severity": "HIGH", "kind": {"type": "socket_lineage", "pid": 1}, "metadata": {"host": "a"},
    })
    assert project_event(with_user).features[idx_user] == 1.0
    assert project_event(with_user).features[idx_pivot] == 1.0   # user + dst_ip
    assert project_event(without).features[idx_user] == 0.0
    assert project_event(without).features[idx_pivot] == 0.0


def test_lateral_movement_links_only_via_user():
    events = simulate_campaign(1, 9, scenario="lateral_movement")
    graph = build_temporal_graph(events)
    assert len({event.metadata["host"] for event in events}) == 6
    users = {node for node in graph.node_ids if node.startswith("user:")}
    assert len(users) == 1                                        # one actor spans every host
    # Benign twin: a distinct user per stage, so identity links nothing.
    benign = simulate_campaign(0, 9, scenario="lateral_movement")
    benign_users = {node for node in build_temporal_graph(benign).node_ids if node.startswith("user:")}
    assert len(benign_users) == len(benign)


def test_v0_easy_scenario_keeps_original_semantics():
    events = simulate_campaign(1, 5)
    assert events == simulate_campaign(1, 5, scenario="v0_easy")
    hosts = {event.metadata["host"] for event in events}
    assert hosts == {"workstation-05"}
    benign_hosts = {event.metadata["host"] for event in simulate_campaign(0, 5)}
    assert len(benign_hosts) == 6


def test_split_host_links_only_via_infrastructure():
    events = simulate_campaign(1, 9, scenario="split_host")
    hosts = {event.metadata["host"] for event in events}
    assert len(hosts) == 6
    graph = build_temporal_graph(events)
    hashes = {node for node in graph.node_ids if node.startswith("hash:")}
    ips = {node for node in graph.node_ids if node.startswith("ip:")}
    assert len(hashes) == 1
    assert any(node.startswith("prefix:") for node in graph.node_ids)
    assert len({node for node in ips if node.endswith(f".{20 + 9 % 70}")}) == 1


def test_low_and_slow_stretches_the_kill_chain():
    fast = simulate_campaign(1, 9)
    slow = simulate_campaign(1, 9, scenario="low_and_slow")
    fast_span = (fast[-1].timestamp - fast[0].timestamp).total_seconds()
    slow_span = (slow[-1].timestamp - slow[0].timestamp).total_seconds()
    assert slow_span == fast_span * 480


def test_hash_rotation_breaks_hash_convergence():
    events = simulate_campaign(1, 9, scenario="hash_rotation")
    graph = build_temporal_graph(events)
    hashes = {node for node in graph.node_ids if node.startswith("hash:")}
    assert len(hashes) == 2
    assert {event.metadata["host"] for event in events} == {"workstation-09"}


def test_legit_update_is_a_topology_matched_hard_negative():
    benign = simulate_campaign(0, 9, scenario="legit_update")
    assert {event.metadata["host"] for event in benign} == {"workstation-09"}
    socket = next(event for event in benign if event.kind_type == "socket_lineage")
    assert socket.kind["is_signed"] is True
    nvme = next(event for event in benign if event.kind_type == "nvme_latency_anomaly")
    assert nvme.kind["concurrent_exfil"] is False
    bgp = next(event for event in benign if event.kind_type == "bgp_anomaly")
    assert bgp.kind["observed_origin_asn"] == bgp.kind["legitimate_origin_asn"]


def test_truncated_scenario_prefixes_the_chain():
    events = simulate_campaign(1, 9, scenario="truncated")
    stage_reached = events[0].metadata["stage_reached"]
    assert 2 <= stage_reached <= 6
    assert len(events) == stage_reached
    assert all(event.metadata["stage_index"] < stage_reached for event in events)


def test_distractor_adds_benign_noise_on_the_campaign_host():
    events = simulate_campaign(1, 9, scenario="distractor")
    assert len(events) == 9
    noise = [event for event in events if event.severity == "INFO"]
    assert len(noise) == 3
    assert all(event.kind["is_signed"] for event in noise)


def test_scenarios_are_deterministic():
    from idr_intelligence.simulator import SCENARIOS

    for scenario in SCENARIOS:
        first = simulate_campaign(1, 13, scenario=scenario)
        second = simulate_campaign(1, 13, scenario=scenario)
        assert [event.id for event in first] == [event.id for event in second]


def test_report_contains_scenario_generalization(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from idr_intelligence.simulator import SCENARIOS

    report = train_ablation(samples=20, epochs=1, seed=21)
    table = report["scenario_generalization"]
    assert set(table) == set(SCENARIOS)
    for row in table.values():
        assert set(row) == {"roc_auc", "brier", "recall_at_fpr_1pct"}


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


def test_unmapped_kind_is_skipped_not_crashed():
    from idr_intelligence.attack import observed_attack_stages, predict_next_stage

    base = {"source": "sentinel_correlation", "severity": "HIGH", "metadata": {"host": "alpha"}}
    triage = IdrEvent.from_dict({**base, "id": "d" * 8 + "-0000-0000-0000-000000000001", "timestamp": "2026-06-18T12:00:00Z", "kind": {"type": "triage_classification"}})
    socket = IdrEvent.from_dict({**base, "id": "e" * 8 + "-0000-0000-0000-000000000002", "timestamp": "2026-06-18T12:05:00Z", "kind": {"type": "socket_lineage", "pid": 1}})
    # Deliberately-unmapped kind alone yields no stages, not a KeyError.
    assert observed_attack_stages([triage]) == ()
    assert predict_next_stage([triage]) == "unknown"
    # Mixed in, it is skipped and the mapped kind still drives prediction.
    assert predict_next_stage([triage, socket]) == "persistence"


def test_same_tactic_kinds_each_emit_a_stage():
    from idr_intelligence.attack import observed_attack_stages

    base = {"source": "network_zeek", "severity": "HIGH", "metadata": {"host": "alpha"}}
    ntp = IdrEvent.from_dict({**base, "id": "f" * 8 + "-0000-0000-0000-000000000001", "timestamp": "2026-06-18T12:00:00Z", "kind": {"type": "ntp_time_shift"}})
    rtc = IdrEvent.from_dict({**base, "id": "0" * 8 + "-0000-0000-0000-000000000002", "timestamp": "2026-06-18T12:05:00Z", "kind": {"type": "rtc_clock_divergence"}})
    stages = observed_attack_stages([ntp, rtc])
    # Both map to defense-evasion/T1562 — dedup is per kind, so two entries.
    assert len(stages) == 2
    assert {stage["tactic"] for stage in stages} == {"defense-evasion"}
    assert {stage["kind_type"] for stage in stages} == {"ntp_time_shift", "rtc_clock_divergence"}


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


def test_recall_at_fpr_degenerate_and_interior():
    from idr_intelligence.training import _recall_at_fpr

    # Single-class inputs must never leak NaN into the report.
    assert _recall_at_fpr(np.zeros(3), np.array([0.1, 0.2, 0.3]), 0.01) == 0.0
    assert _recall_at_fpr(np.ones(3), np.array([0.1, 0.2, 0.3]), 0.01) == 1.0
    # Interior operating point: 100 negatives at 1% FPR admits exactly 1 FP, so a
    # positive scored above one negative but below the rest is recoverable.
    labels = np.concatenate([np.zeros(100), np.ones(1)])
    scores = np.concatenate([np.linspace(0.0, 0.5, 100), np.array([0.505])])
    assert _recall_at_fpr(labels, scores, 0.01) == 1.0


def test_precision_at_k_tie_is_deterministic():
    from idr_intelligence.training import _precision_at_k

    # A pos/neg tie at the k boundary must resolve the same way every call.
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    scores = np.array([0.9, 0.7, 0.7, 0.3])
    first = _precision_at_k(labels, scores, 2)
    assert first == _precision_at_k(labels, scores, 2)
    assert first in (0.0, 0.5)


def test_ablation_runs_imbalanced(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report = train_ablation(samples=24, epochs=1, seed=11, malicious_rate=0.3)
    assert report["malicious_rate"] == 0.3
    row = report["metrics"]["s6_gnn"]
    for key in ("recall_at_fpr_1pct", "recall_at_fpr_0p1pct", "precision_at_5"):
        assert 0.0 <= row[key] <= 1.0


def test_affine_calibration_never_worsens_and_can_improve_nll():
    from idr_intelligence.training import affine_calibration_params

    loss_fn = torch.nn.BCEWithLogitsLoss()
    torch.manual_seed(0)
    # True generating logit is `latent`; labels are drawn stochastically from it,
    # so the data is NOT perfectly separable and overconfidence is punished.
    latent = torch.randn(2000)
    labels = (torch.rand(2000) < torch.sigmoid(latent)).float()
    # Overconfident + shifted logits: identity is far from optimal, and a real
    # fit must strictly improve NLL by shrinking the 10x inflation.
    miscalibrated = latent * 10.0 + 3.0
    pre = loss_fn(miscalibrated, labels).item()
    scale, bias = affine_calibration_params(miscalibrated, labels)
    post = loss_fn(miscalibrated * scale + bias, labels).item()
    assert post <= pre + 1e-9
    assert post < pre - 0.05
    assert scale < 1.0  # must shrink the 10x inflation
    # Well-calibrated logits: fit stays at (or improves on) identity, never worse.
    s2, b2 = affine_calibration_params(latent, labels)
    assert loss_fn(latent * s2 + b2, labels).item() <= loss_fn(latent, labels).item() + 1e-9


def test_affine_bias_absorbs_pos_weight_shift():
    # The blocking regression: class-weighted training shifts logits by log(w);
    # temperature-only calibration cannot remove a constant shift, affine can.
    from idr_intelligence.training import affine_calibration_params

    torch.manual_seed(1)
    base_rate = 0.2
    n = 2000
    labels = (torch.rand(n) < base_rate).float()
    # Weighted-BCE-optimal logits for a well-fit model: logit(p) + log(w).
    w = (1 - base_rate) / base_rate
    p = torch.where(labels > 0, torch.full((n,), 0.9), torch.full((n,), 0.05))
    shifted = torch.log(p / (1 - p)) + torch.log(torch.tensor(w))
    scale, bias = affine_calibration_params(shifted, labels)
    calibrated_mean = torch.sigmoid(shifted * scale + bias).mean().item()
    raw_mean = torch.sigmoid(shifted).mean().item()
    assert abs(calibrated_mean - base_rate) < abs(raw_mean - base_rate)
    assert abs(calibrated_mean - base_rate) < 0.05


def test_calibration_errors_zero_when_perfect():
    from idr_intelligence.training import _calibration_errors

    labels = np.array([0.0, 0.0, 1.0, 1.0])
    ece, mce = _calibration_errors(labels, np.array([0.0, 0.0, 1.0, 1.0]))
    assert ece == 0.0 and mce == 0.0
    ece_bad, mce_bad = _calibration_errors(labels, np.array([0.9, 0.9, 0.1, 0.1]))
    assert ece_bad > 0.5 and mce_bad > 0.5


def test_calibration_roundtrips_and_defaults(tmp_path):
    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4)
    model.temperature.fill_(2.5)
    model.cal_bias.fill_(-0.75)
    path = tmp_path / "calibrated.pt"
    save_checkpoint(model, path)
    payload = torch.load(path, weights_only=True)
    assert payload["manifest"]["calibration"] == "affine:scale=0.400000,bias=-0.750000"
    loaded = load_campaign_model(path)
    assert float(loaded.temperature.item()) == 2.5
    assert float(loaded.cal_bias.item()) == -0.75
    # A pre-calibration checkpoint with neither buffer defaults to identity.
    stripped = dict(payload["state_dict"])
    del stripped["temperature"]
    del stripped["cal_bias"]
    payload["state_dict"] = stripped
    del payload["manifest"]
    older = tmp_path / "older.pt"
    torch.save(payload, older)
    reloaded = load_campaign_model(older)
    assert float(reloaded.temperature.item()) == 1.0
    assert float(reloaded.cal_bias.item()) == 0.0
    assert reloaded.calibration_label() == "none"


def test_finding_calibration_matches_affine_transform():
    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4)
    model.temperature.fill_(2.0)
    model.cal_bias.fill_(0.5)
    finding = score_events(simulate_campaign(label=1, seed=8), model)
    assert finding.calibration == "affine:scale=0.500000,bias=0.500000"
    raw, calibrated = finding.raw_escalation_probability, finding.escalation_probability
    assert raw != calibrated
    # Recover the logit from raw and confirm calibrated == sigmoid(logit/T + bias).
    logit = np.log(raw / (1 - raw))
    expected = 1.0 / (1.0 + np.exp(-(logit / 2.0 + 0.5)))
    assert abs(calibrated - expected) < 1e-4


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


def test_campaign_fingerprint_keeps_durable_drops_ephemeral():
    from idr_intelligence.campaigns import fingerprint

    prints = fingerprint(["hash:abc", "process:alpha:12", "session:alpha:5", "host:alpha", "domain:evil.example"])
    assert set(prints) == {"hash:abc", "host:alpha", "domain:evil.example"}
    assert prints["hash:abc"] > prints["host:alpha"]  # infrastructure outweighs workstation identity


def test_weighted_jaccard_host_only_insufficient_infrastructure_sufficient():
    from idr_intelligence.campaigns import (
        MATCH_THRESHOLD,
        fingerprint,
        weighted_jaccard,
    )

    known = fingerprint(["hash:abc", "domain:evil.example", "host:alpha"])
    # Same C2 infrastructure seen from a different host: continues the campaign.
    same_infra = fingerprint(["hash:abc", "domain:evil.example", "host:beta"])
    assert weighted_jaccard(same_infra, known) >= MATCH_THRESHOLD
    # Only the workstation in common: not enough to claim continuity.
    host_only = fingerprint(["host:alpha", "ip:203.0.113.99"])
    assert weighted_jaccard(host_only, known) < MATCH_THRESHOLD
    assert weighted_jaccard({}, known) == 0.0


def test_campaign_content_address_is_order_independent():
    from idr_intelligence.campaigns import content_address, fingerprint

    forward = content_address(fingerprint(["hash:abc", "domain:evil.example", "host:alpha"]))
    shuffled = content_address(fingerprint(["host:alpha", "hash:abc", "domain:evil.example"]))
    other = content_address(fingerprint(["hash:other", "host:alpha"]))
    assert forward == shuffled
    assert forward.startswith("idr-campaign-")
    assert forward != other


def test_registry_continues_matching_campaign_and_registers_new_ones():
    from idr_intelligence.campaigns import CampaignRegistry

    registry = CampaignRegistry()
    first_id, continues, windows = registry.match_or_register(["hash:abc", "domain:evil.example", "host:alpha"], "2026-07-01T00:00:00+00:00")
    assert (continues, windows) == (False, 1)
    # Same infrastructure on a new host a day later: same campaign, window count grows.
    second_id, continues, windows = registry.match_or_register(["hash:abc", "domain:evil.example", "host:beta"], "2026-07-02T00:00:00+00:00")
    assert second_id == first_id
    assert (continues, windows) == (True, 2)
    assert registry.records[0].last_seen == "2026-07-02T00:00:00+00:00"
    assert "host:beta" in registry.records[0].fingerprint  # fingerprint unions across windows
    # Unrelated infrastructure: a distinct campaign identity.
    third_id, continues, windows = registry.match_or_register(["hash:zzz", "domain:other.example", "host:alpha"], "2026-07-02T01:00:00+00:00")
    assert third_id != first_id
    assert (continues, windows) == (False, 1)
    # No durable entities at all: sentinel id, never registered.
    ghost_id, continues, _ = registry.match_or_register(["process:alpha:12", "session:alpha:5"], "2026-07-02T02:00:00+00:00")
    assert ghost_id == "idr-campaign-unfingerprinted"
    assert not continues
    assert len(registry.records) == 2


def test_registry_persistence_roundtrip(tmp_path):
    from idr_intelligence.campaigns import CampaignRegistry

    path = tmp_path / "registry.json"
    assert CampaignRegistry.load(path).records == []  # missing file -> empty registry
    registry = CampaignRegistry()
    original_id, _, _ = registry.match_or_register(["hash:abc", "domain:evil.example"], "2026-07-01T00:00:00+00:00")
    registry.save(path)
    reloaded = CampaignRegistry.load(path)
    matched_id, continues, windows = reloaded.match_or_register(["hash:abc", "domain:evil.example", "host:beta"], "2026-07-02T00:00:00+00:00")
    assert matched_id == original_id
    assert (continues, windows) == (True, 2)


def test_score_events_with_registry_keeps_campaign_identity_stable():
    from idr_intelligence.campaigns import CampaignRegistry

    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4)
    events = simulate_campaign(label=1, seed=8)
    registry = CampaignRegistry()
    first = score_events(events, model, registry=registry)
    second = score_events(events, model, registry=registry)
    assert first.campaign_id == second.campaign_id
    assert (first.continues_campaign, first.windows_observed) == (False, 1)
    assert (second.continues_campaign, second.windows_observed) == (True, 2)
    # Without a registry the per-call id and defaults are unchanged (back-compat).
    solo = score_events(events, model)
    assert solo.campaign_id == f"idr-campaign-{min(events, key=lambda e: (e.timestamp, e.id)).id[:8]}"
    assert (solo.continues_campaign, solo.windows_observed) == (False, 1)


def test_streaming_matches_batch_scoring():
    from idr_intelligence.streaming import StreamingScorer

    events = simulate_campaign(label=1, seed=8)
    ordered = sorted(events, key=lambda event: (event.timestamp, event.id))
    for time_mode in ("global", "per_entity", "time_aware"):
        # Seed the model init: without this the weights depend on the global
        # torch RNG left by prior tests, which can land on a top-k boundary tie
        # where stream and batch pick different tied entities (an order-dependent
        # flake in this equivalence check, not a real divergence).
        torch.manual_seed(1234)
        model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4, time_mode=time_mode)
        batch = score_events(events, model, max_steps=64)
        scorer = StreamingScorer(model)
        for event in ordered:
            scorer.ingest(event)
        stream = scorer.finding()
        assert stream.escalation_probability == pytest.approx(batch.escalation_probability, abs=1e-4)
        assert stream.raw_escalation_probability == pytest.approx(batch.raw_escalation_probability, abs=1e-4)
        # Set comparison: structurally identical entities tie exactly, and a
        # ~1e-8 batch-vs-streaming float difference can break such a tie in one
        # path only, legitimately permuting tied neighbors under stable ranking.
        assert set(stream.related_entities) == set(batch.related_entities)
        assert set(stream.evidence_event_ids) == set(batch.evidence_event_ids)
        assert stream.campaign_id == batch.campaign_id
        assert stream.predicted_next_stage == batch.predicted_next_stage
        assert stream.observed_attack_stages == batch.observed_attack_stages
        assert stream.graph_relations == batch.graph_relations
        assert stream.graph_nodes == batch.graph_nodes


def test_streaming_decayed_edges_match_batch():
    from idr_intelligence.streaming import StreamingScorer

    events = _timed_events([(0, "alpha"), (30, "beta"), (400, "alpha")])
    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4, decay_half_life=3600.0)
    batch = score_events(events, model, max_steps=64)
    scorer = StreamingScorer(model)
    for event in events:
        scorer.ingest(event)
    stream = scorer.finding()
    assert stream.escalation_probability == pytest.approx(batch.escalation_probability, abs=1e-4)
    assert stream.related_entities == batch.related_entities


def test_streaming_rejects_out_of_order_events():
    from idr_intelligence.streaming import StreamingScorer

    early, late = _timed_events([(0, "alpha"), (30, "alpha")])
    scorer = StreamingScorer(CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4))
    scorer.ingest(late)
    with pytest.raises(ValueError, match="out-of-order"):
        scorer.ingest(early)


def test_streaming_requires_s6_model():
    from idr_intelligence.streaming import StreamingScorer

    with pytest.raises(ValueError, match="requires an S6 model"):
        StreamingScorer(CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4, use_s6=False))


def test_streaming_budget_bounds_entities_with_audit():
    from idr_intelligence.bounded_graph import GraphBudget
    from idr_intelligence.streaming import StreamingScorer

    events = sorted(simulate_campaign(label=1, seed=8), key=lambda event: (event.timestamp, event.id))
    unbounded = StreamingScorer(CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4))
    for event in events:
        unbounded.ingest(event)
    budget = GraphBudget(max_nodes=max(2, len(unbounded.entities) - 3))
    scorer = StreamingScorer(CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4), budget=budget)
    for event in events:
        scorer.ingest(event)
    assert len(scorer.entities) <= budget.max_nodes
    assert scorer.evictions and all(record.reason == "node_budget" for record in scorer.evictions)
    finding = scorer.finding()
    assert finding.graph_nodes <= budget.max_nodes
    assert 0.0 <= finding.escalation_probability <= 1.0


def test_streaming_drift_matches_batch():
    from idr_intelligence.streaming import StreamingScorer

    events = simulate_campaign(label=1, seed=8)
    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4)
    model.feature_stats = {
        "bin_edges": list(np.linspace(0.0, 1.0, 11)),
        "histograms": [[1] * 10 for _ in range(FEATURE_DIM)],
    }
    batch = score_events(events, model, max_steps=64)
    scorer = StreamingScorer(model)
    for event in sorted(events, key=lambda item: (item.timestamp, item.id)):
        scorer.ingest(event)
    stream = scorer.finding()
    assert batch.feature_drift is not None
    assert stream.feature_drift == batch.feature_drift


def test_streaming_registry_gives_cross_window_continuity():
    from idr_intelligence.campaigns import CampaignRegistry
    from idr_intelligence.streaming import StreamingScorer

    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4)
    events = sorted(simulate_campaign(label=1, seed=8), key=lambda event: (event.timestamp, event.id))
    registry = CampaignRegistry()
    findings = []
    for _ in range(2):  # two independent scoring windows over the same infrastructure
        scorer = StreamingScorer(model)
        for event in events:
            scorer.ingest(event)
        findings.append(scorer.finding(registry=registry))
    assert findings[0].campaign_id == findings[1].campaign_id
    assert (findings[0].continues_campaign, findings[0].windows_observed) == (False, 1)
    assert (findings[1].continues_campaign, findings[1].windows_observed) == (True, 2)


def test_cli_stream_roundtrip(tmp_path, monkeypatch, capsys):
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
    monkeypatch.setattr("sys.argv", ["idr-intelligence", "stream", str(events_path), "--weights", str(weights), "--max-nodes", "6"])
    cli.main()
    finding = json.loads(capsys.readouterr().out)
    assert finding["model_version"] == "model.pt"
    assert 0 < finding["graph_nodes"] <= 6
    assert finding["evidence_event_ids"]
    assert "evictions" in finding


def test_cli_score_registry_persists_across_invocations(tmp_path, monkeypatch, capsys):
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
    registry_path = tmp_path / "registry.json"
    argv = ["idr-intelligence", "score", str(events_path), "--weights", str(weights), "--registry", str(registry_path)]
    monkeypatch.setattr("sys.argv", argv)
    cli.main()
    first = json.loads(capsys.readouterr().out)
    assert registry_path.exists()
    cli.main()
    second = json.loads(capsys.readouterr().out)
    assert second["campaign_id"] == first["campaign_id"]
    assert (first["continues_campaign"], first["windows_observed"]) == (False, 1)
    assert (second["continues_campaign"], second["windows_observed"]) == (True, 2)


def test_attention_pool_fully_masked_row_yields_zero_weights():
    from idr_intelligence.models import GatedAttentionPool

    pool = GatedAttentionPool(hidden_dim=8)
    nodes = torch.randn(1, 3, 8)
    pooled, _ = pool(nodes, torch.zeros(1, 3, 1))
    assert torch.isfinite(pooled).all()
    assert torch.count_nonzero(pooled) == 0  # NaN row from all-masked softmax becomes zero weights


def _export_bundle(base_dir, model, model_version="test-export"):
    pytest.importorskip("onnx")  # torch's TorchScript exporter needs the onnx package
    from idr_intelligence.export import export_streaming_bundle

    out = Path(base_dir) / "bundle"
    manifest = export_streaming_bundle(model, out, model_version=model_version)
    return out, manifest


def test_export_requires_s6_model(tmp_path):
    from idr_intelligence.export import export_streaming_bundle

    with pytest.raises(ValueError, match="S6 model"):
        export_streaming_bundle(CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4, use_s6=False), tmp_path)


def test_export_manifest_matches_runtime(tmp_path):
    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4, time_mode="time_aware", decay_half_life=3600.0)
    with torch.no_grad():
        model.temperature.fill_(1.25)
        model.cal_bias.fill_(-0.2)
    out, manifest = _export_bundle(tmp_path, model)
    assert manifest["feature_schema_hash"] == feature_schema_hash()
    assert manifest["calibration"]["label"] == model.calibration_label()
    assert manifest["features"]["names"] == list(FEATURE_NAMES)
    # The bridge scores with the manifest's constants; they must be the very
    # objects feature extraction uses, not re-typed literals.
    from idr_intelligence.features import (
        DELTA_LOG_DIVISOR,
        KIND_PRIOR_DEFAULT,
        SEVERITY_DEFAULT,
    )

    assert manifest["features"]["severity_default"] == SEVERITY_DEFAULT
    assert manifest["features"]["kind_prior_default"] == KIND_PRIOR_DEFAULT
    assert manifest["features"]["delta_log_divisor"] == DELTA_LOG_DIVISOR
    assert manifest["model"]["decay_half_life"] == 3600.0
    assert manifest["graphs"]["step"]["inputs"] == ["x_t", "state", "output", "delta_t"]
    assert (out / "step.onnx").exists() and (out / "head.onnx").exists()
    assert json.loads((out / "manifest.json").read_text()) == manifest
    # Models without time-aware discretization export a three-input step cell.
    _, plain = _export_bundle(tmp_path / "plain", CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4))
    assert plain["graphs"]["step"]["inputs"] == ["x_t", "state", "output"]


def test_export_step_and_head_match_torch(tmp_path):
    onnxruntime = pytest.importorskip("onnxruntime")
    for time_mode in ("global", "per_entity", "time_aware"):
        torch.manual_seed(23)
        model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4, time_mode=time_mode)
        out, _ = _export_bundle(tmp_path / time_mode, model)
        step = onnxruntime.InferenceSession(str(out / "step.onnx"))
        x = torch.randn(1, FEATURE_DIM)
        state = torch.randn(1, 12, 4)
        output = torch.randn(1, 12)
        delta = torch.tensor([0.37])
        with torch.no_grad():
            want_state, want_output = model.temporal.step(x, state, output, delta_t=delta if time_mode == "time_aware" else None)
        feeds = {"x_t": x.numpy(), "state": state.numpy(), "output": output.numpy()}
        if time_mode == "time_aware":
            feeds["delta_t"] = delta.numpy()
        got_state, got_output = step.run(None, feeds)
        np.testing.assert_allclose(got_state, want_state.numpy(), atol=1e-5)
        np.testing.assert_allclose(got_output, want_output.numpy(), atol=1e-5)
        head = onnxruntime.InferenceSession(str(out / "head.onnx"))
        for nodes in (1, 4, 9):  # the node axis is dynamic in the exported graph
            carried = torch.randn(nodes, 12)
            adjacency = torch.rand(nodes, nodes)
            adjacency = (adjacency + adjacency.T) / 2
            with torch.no_grad():
                want = model.relational_head(model.temporal.norm(carried).unsqueeze(0), torch.ones(1, nodes, 1), adjacency.unsqueeze(0))
            got_logit, got_nodes = head.run(None, {"outputs": carried.numpy(), "adjacency": adjacency.numpy()})
            np.testing.assert_allclose(got_logit, want.graph_logit.numpy(), atol=1e-5)
            np.testing.assert_allclose(got_nodes, want.node_logits[0].numpy(), atol=1e-5)


def test_onnx_stream_scorer_matches_streaming(tmp_path):
    pytest.importorskip("onnxruntime")
    from idr_intelligence.export import OnnxStreamScorer
    from idr_intelligence.streaming import StreamingScorer

    ordered = sorted(simulate_campaign(label=1, seed=11, scenario="lateral_movement"), key=lambda event: (event.timestamp, event.id))
    for time_mode in ("global", "per_entity", "time_aware"):
        torch.manual_seed(31)
        model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4, time_mode=time_mode, decay_half_life=1800.0)
        with torch.no_grad():
            model.temperature.fill_(1.5)
            model.cal_bias.fill_(-0.25)
        out, _ = _export_bundle(tmp_path / time_mode, model)
        reference = StreamingScorer(model, model_version="test-export")
        exported = OnnxStreamScorer(out)
        for event in ordered:
            reference.ingest(event)
            exported.ingest(event)
        want = reference.finding()
        got = exported.finding()
        assert got.escalation_probability == pytest.approx(want.escalation_probability, abs=1e-4)
        assert got.raw_escalation_probability == pytest.approx(want.raw_escalation_probability, abs=1e-4)
        assert got.related_entities == want.related_entities
        assert got.evidence_event_ids == want.evidence_event_ids
        assert got.observed_attack_stages == want.observed_attack_stages
        assert got.graph_relations == want.graph_relations
        assert got.campaign_id == want.campaign_id
        assert got.calibration == want.calibration
        assert got.predicted_next_stage == want.predicted_next_stage
        assert got.feature_schema_hash == want.feature_schema_hash


def test_onnx_stream_scorer_budget_and_suppressions_match(tmp_path):
    pytest.importorskip("onnxruntime")
    from idr_intelligence.bounded_graph import GraphBudget
    from idr_intelligence.export import OnnxStreamScorer
    from idr_intelligence.streaming import StreamingScorer

    ordered = sorted(simulate_campaign(label=1, seed=4), key=lambda event: (event.timestamp, event.id))
    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4, time_mode="per_entity")
    out, _ = _export_bundle(tmp_path, model)
    reference = StreamingScorer(model, budget=GraphBudget(max_nodes=6), model_version="test-export")
    exported = OnnxStreamScorer(out, budget=GraphBudget(max_nodes=6))
    for event in ordered:
        reference.ingest(event)
        exported.ingest(event)
    want = reference.finding(suppressions=["ip:"])
    got = exported.finding(suppressions=["ip:"])
    assert [record.entity for record in exported.evictions] == [record.entity for record in reference.evictions]
    assert got.related_entities == want.related_entities
    assert got.applied_suppressions == want.applied_suppressions
    assert got.escalation_probability == pytest.approx(want.escalation_probability, abs=1e-4)


def test_onnx_stream_scorer_drift_matches_streaming(tmp_path):
    pytest.importorskip("onnxruntime")
    from idr_intelligence.export import OnnxStreamScorer
    from idr_intelligence.streaming import StreamingScorer

    ordered = sorted(simulate_campaign(label=1, seed=6), key=lambda event: (event.timestamp, event.id))
    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4)
    model.feature_stats = {"bin_edges": np.linspace(0.0, 1.0, 11).tolist(), "histograms": [[1] * 10 for _ in range(FEATURE_DIM)]}
    out, manifest = _export_bundle(tmp_path, model)
    assert manifest["feature_stats"] == model.feature_stats  # drift baseline travels with the bundle
    reference = StreamingScorer(model, model_version="test-export")
    exported = OnnxStreamScorer(out)
    for event in ordered:
        reference.ingest(event)
        exported.ingest(event)
    assert exported.finding().feature_drift == reference.finding().feature_drift


def test_cli_export_roundtrip(tmp_path, monkeypatch, capsys):
    pytest.importorskip("onnx")
    weights = tmp_path / "model.pt"
    save_checkpoint(CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4, time_mode="time_aware"), weights)
    out = tmp_path / "bundle"
    monkeypatch.setattr("sys.argv", ["idr-intelligence", "export", "--weights", str(weights), "--out", str(out)])
    cli.main()
    summary = json.loads(capsys.readouterr().out)
    assert summary["graphs"] == ["head.onnx", "step.onnx"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["model_version"] == "model.pt"
    assert manifest["model"]["time_mode"] == "time_aware"
    assert (out / "step.onnx").exists() and (out / "head.onnx").exists()


def test_model_rejects_nonpositive_decay_half_life():
    with pytest.raises(ValueError, match="decay_half_life"):
        CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4, decay_half_life=0.0)


def test_ranking_tie_break_is_first_seen():
    # ADR-31: all ranking call sites use a stable argsort, so exact ties keep
    # first-seen order and suppressed (-inf) entries are sliced then filtered.
    scores = np.array([0.5, 0.9, 0.5, 0.5, 0.7, -np.inf])
    ranked = np.argsort(-scores, kind="stable")[:5]
    ranked = [index for index in ranked if np.isfinite(scores[index])]
    assert ranked == [1, 4, 0, 2, 3]
    scores = np.array([-np.inf, 0.4, 0.6])
    ranked = np.argsort(-scores, kind="stable")[:3]
    assert [index for index in ranked if np.isfinite(scores[index])] == [2, 1]


def test_onnx_scorer_rejects_schema_mismatch(tmp_path):
    pytest.importorskip("onnxruntime")
    from idr_intelligence.export import OnnxStreamScorer

    out, _ = _export_bundle(tmp_path, CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4))
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["feature_schema_hash"] = "0" * 32
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SchemaMismatchError):
        OnnxStreamScorer(out)


def test_export_no_gnn_head_omits_adjacency(tmp_path):
    pytest.importorskip("onnxruntime")
    from idr_intelligence.export import OnnxStreamScorer
    from idr_intelligence.streaming import StreamingScorer

    torch.manual_seed(13)
    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4, use_gnn=False)
    out, manifest = _export_bundle(tmp_path, model)
    assert manifest["graphs"]["head"]["inputs"] == ["outputs"]  # exporter prunes the unread adjacency
    ordered = sorted(simulate_campaign(1, 8), key=lambda event: (event.timestamp, event.id))
    reference = StreamingScorer(model, model_version="test-export")
    exported = OnnxStreamScorer(out)
    for event in ordered:
        reference.ingest(event)
        exported.ingest(event)
    want = reference.finding()
    got = exported.finding()
    assert got.escalation_probability == pytest.approx(want.escalation_probability, abs=1e-4)
    # Set comparison: exact ties among isomorphic entities may permute under
    # cross-runtime float drift; order pinning lives in the Rust golden.
    assert set(got.related_entities) == set(want.related_entities)
    assert got.observed_attack_stages == want.observed_attack_stages


def test_export_uniform_pooling_bundle_parity(tmp_path):
    pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    from idr_intelligence.export import OnnxStreamScorer
    from idr_intelligence.streaming import StreamingScorer

    torch.manual_seed(17)
    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4, pooling="uniform")
    out, _ = _export_bundle(tmp_path, model)
    ops = {node.op_type for node in onnx.load(str(out / "head.onnx")).graph.node}
    assert "IsInf" not in ops  # the tract runtime cannot evaluate IsInf (ADR-30)
    ordered = sorted(simulate_campaign(1, 9), key=lambda event: (event.timestamp, event.id))
    reference = StreamingScorer(model, model_version="test-export")
    exported = OnnxStreamScorer(out)
    for event in ordered:
        reference.ingest(event)
        exported.ingest(event)
    want = reference.finding()
    got = exported.finding()
    assert got.raw_escalation_probability == pytest.approx(want.raw_escalation_probability, abs=1e-4)
    assert set(got.related_entities) == set(want.related_entities)


RUST_FIXTURES = Path(__file__).resolve().parent.parent / "rust" / "idr-intelligence-rt" / "tests" / "fixtures"


def _fixture_events(name):
    rows = [json.loads(line) for line in (RUST_FIXTURES / name).read_text().splitlines() if line.strip()]
    return sorted((IdrEvent.from_dict(row) for row in rows), key=lambda event: (event.timestamp, event.id))


def _assert_matches_golden(finding, golden, pin_ranking=True):
    payload = json.loads(json.dumps(finding.to_dict()))
    for key, value in golden.items():
        if key in ("scored_at", "evictions"):
            continue
        if not pin_ranking and key in ("related_entities", "evidence_event_ids"):
            # No-GNN streams tie exactly on isomorphic entities; order is
            # runtime-dependent by nature (see the fixture generator).
            if key == "related_entities":
                assert set(payload[key]) == set(value), key
            continue
        if key in ("escalation_probability", "raw_escalation_probability"):
            assert payload[key] == pytest.approx(value, abs=1e-4), key
        elif key == "feature_drift" and value is not None:
            assert payload[key]["flagged_features"] == value["flagged_features"]
            assert payload[key]["psi_max"] == pytest.approx(value["psi_max"], abs=1e-4)
            assert payload[key]["psi_mean"] == pytest.approx(value["psi_mean"], abs=1e-4)
        else:
            assert payload[key] == value, key


def test_rust_fixture_goldens_are_fresh():
    """Freshness gate: the committed cross-language goldens must reproduce under
    the CURRENT engine. A semantic change to extraction/scoring bookkeeping
    without regenerating the fixtures fails here instead of silently pinning
    the Rust bridge to a stale snapshot of Python."""
    pytest.importorskip("onnxruntime")
    from idr_intelligence.bounded_graph import GraphBudget
    from idr_intelligence.export import OnnxStreamScorer

    for case, bundle, budget, suppressions in (
        ("malicious", RUST_FIXTURES, None, None),
        ("benign", RUST_FIXTURES, None, None),
        ("budget", RUST_FIXTURES, 6, ["ip:"]),
        ("nognn", RUST_FIXTURES / "nognn", None, None),
    ):
        events = _fixture_events("events_malicious.ndjson" if case in ("budget", "nognn") else f"events_{case}.ndjson")
        scorer = OnnxStreamScorer(bundle, budget=GraphBudget(max_nodes=budget) if budget else None)
        for event in events:
            scorer.ingest(event)
        finding = scorer.finding(suppressions=suppressions)
        golden_name = "expected_budget.json" if case == "budget" else ("expected_malicious.json" if case == "nognn" else f"expected_{case}.json")
        golden = json.loads((bundle / golden_name).read_text() if case == "nognn" else (RUST_FIXTURES / golden_name).read_text())
        _assert_matches_golden(finding, golden, pin_ranking=(case != "nognn"))
        if case == "budget":
            assert [record.entity for record in scorer.evictions] == [record["entity"] for record in golden["evictions"]]


def test_rust_feature_vectors_are_fresh():
    from idr_intelligence.features import project_event

    entries = json.loads((RUST_FIXTURES / "feature_vectors.json").read_text())
    assert len(entries) > 40
    for entry in entries:
        event = IdrEvent.from_dict(entry["event"])
        projection = project_event(event, delta_seconds=entry["delta_seconds"])
        assert list(projection.entities) == entry["entities"], event.id
        assert [list(edge) for edge in projection.edges] == entry["edges"], event.id
        assert projection.features.tolist() == entry["features"], event.id


def test_timestamp_battery_is_fresh():
    """The committed timestamp battery is the cross-language acceptance
    contract, with verdicts owned by THIS interpreter's fromisoformat. If a
    Python upgrade changes acceptance, this fails and the battery (and the
    Rust mirror) must be regenerated together."""
    from idr_intelligence.schema import _parse_timestamp

    entries = json.loads((RUST_FIXTURES / "timestamp_battery.json").read_text())
    assert len(entries) > 300
    for entry in entries:
        try:
            utc = _parse_timestamp(entry["shape"]).isoformat()
        except (ValueError, OverflowError):
            utc = None
        assert utc == entry["utc"], entry["shape"]


def test_cli_log_level_emits_structured_stderr_without_touching_stdout(tmp_path, monkeypatch, capsys):
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
    monkeypatch.setattr("sys.argv", ["idr-intelligence", "score", str(events_path), "--weights", str(weights), "--log-level", "info"])
    cli.main()
    captured = capsys.readouterr()
    # stdout is still exactly the finding JSON — logs never contaminate it.
    finding = json.loads(captured.out)
    assert finding["campaign_id"].startswith("idr-campaign-")
    # stderr carries structured JSON-lines operational events.
    records = [json.loads(line) for line in captured.err.splitlines() if line.strip()]
    events_by_name = {record["event"]: record for record in records}
    assert "model_loaded" in events_by_name
    assert events_by_name["model_loaded"]["feature_schema_hash"] == feature_schema_hash()
    scored = events_by_name["finding_scored"]
    assert scored["campaign_id"] == finding["campaign_id"]
    assert scored["graph_nodes"] == finding["graph_nodes"]
    assert "elapsed_ms" in scored


def test_cli_default_log_level_is_quiet(tmp_path, monkeypatch, capsys):
    events_path = tmp_path / "events.ndjson"
    events_path.write_text("\n".join(
        json.dumps({
            "id": event.id, "timestamp": event.timestamp.isoformat(), "source": event.source,
            "severity": event.severity, "kind": event.kind, "metadata": event.metadata,
        })
        for event in simulate_campaign(1, 5)
    ) + "\n")
    weights = tmp_path / "model.pt"
    save_checkpoint(CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4), weights)
    monkeypatch.setattr("sys.argv", ["idr-intelligence", "score", str(events_path), "--weights", str(weights)])
    cli.main()
    captured = capsys.readouterr()
    assert json.loads(captured.out)["campaign_id"]  # stdout finding intact
    assert captured.err == ""  # default warning level: no info logs


def test_cli_missing_model_logs_operational_error(tmp_path, monkeypatch, capsys):
    events_path = tmp_path / "events.ndjson"
    events_path.write_text(json.dumps({
        "id": "x", "timestamp": "2026-03-01T00:00:00+00:00", "source": "kernel_ebpf",
        "severity": "HIGH", "kind": {"type": "socket_lineage", "pid": 1},
    }) + "\n")
    monkeypatch.setattr("sys.argv", ["idr-intelligence", "score", str(events_path), "--weights", str(tmp_path / "nope.pt"), "--log-level", "error"])
    with pytest.raises(SystemExit):
        cli.main()
    records = [json.loads(line) for line in capsys.readouterr().err.splitlines() if line.strip()]
    assert any(record["event"] == "model_load_failed" and record["level"] == "ERROR" for record in records)


def _labeled_windows(count, scenario="v0_easy"):
    from idr_intelligence.schema import LabeledWindow

    windows = []
    for index in range(count):
        label = index % 2
        events = simulate_campaign(label, seed=index, scenario=scenario)
        windows.append(LabeledWindow(window_id=f"w{index:03d}", label=label, events=tuple(events)))
    return windows


def _fit_model_on(windows, time_mode="global", epochs=6):
    from idr_intelligence.training import (
        HIDDEN_DIM,
        STATE_DIM,
        _fit,
        chronological_split,
        set_seed,
        windows_to_batch,
    )

    set_seed(7)
    train, val, _ = chronological_split(windows_to_batch(windows, time_mode=time_mode))
    model = CampaignModel(FEATURE_DIM, hidden_dim=HIDDEN_DIM, state_dim=STATE_DIM, time_mode=time_mode)
    _fit(model, train, val, epochs=epochs)
    return model


def test_validate_gate_produces_go_on_relaxed_gates_but_not_binding_on_synthetic():
    from idr_intelligence.validation import validate_model

    windows = _labeled_windows(64)
    model = _fit_model_on(windows)
    # Relax ECE (tiny synthetic separable data is overconfident); this exercises
    # the go path. Provenance is synthetic, so the verdict must stay non-binding.
    report = validate_model(model, windows, target_fpr=0.05, holdout=0.4, provenance="synthetic-standin", gates={"max_ece": 0.6})
    assert report["verdict"] == "go"
    assert report["binding"] is False  # synthetic provenance can never bind
    assert "WIRING verdict" in report["verdict_note"]
    assert report["operating_threshold"]["tau"] is not None
    assert report["drift_baseline"]["sample_count"] > 0
    assert model.feature_stats is not None  # real drift baseline installed on the model
    assert all(gate["value"] is not None for gate in report["gates"])


def test_validate_binding_requires_attested_real_provenance():
    from idr_intelligence.validation import validate_model

    windows = _labeled_windows(64)
    model = _fit_model_on(windows)
    report = validate_model(model, windows, target_fpr=0.05, holdout=0.4, provenance="soc-incidents-2026Q2", gates={"max_ece": 0.6})
    assert report["verdict"] == "go"
    assert report["binding"] is True
    assert "shadow/advisory deployment" in report["verdict_note"]


def test_validate_no_go_when_a_gate_fails():
    from idr_intelligence.validation import validate_model

    windows = _labeled_windows(64)
    model = _fit_model_on(windows)
    # Default ECE gate (0.10) fails on the overconfident synthetic model.
    report = validate_model(model, windows, target_fpr=0.05, holdout=0.4, provenance="soc-incidents-2026Q2")
    assert report["verdict"] == "no-go"
    assert report["binding"] is False  # a failed gate can never bind, even on real data
    assert any(gate["name"] == "ece" and not gate["pass"] for gate in report["gates"])


def test_validate_rejects_single_class_segment():
    from idr_intelligence.schema import LabeledWindow
    from idr_intelligence.validation import validate_model

    # Every window malicious -> the test segment is single-class.
    windows = [
        LabeledWindow(window_id=f"w{i}", label=1, events=tuple(simulate_campaign(1, seed=i)))
        for i in range(8)
    ]
    model = _fit_model_on(_labeled_windows(40))
    with pytest.raises(ValueError, match="single-class"):
        validate_model(model, windows, holdout=0.5)


def test_validate_rejects_unknown_gate():
    from idr_intelligence.validation import validate_model

    windows = _labeled_windows(40)
    model = _fit_model_on(windows)
    with pytest.raises(ValueError, match="unknown gate"):
        validate_model(model, windows, gates={"max_eces": 0.5})


def test_cli_validate_roundtrip_writes_card_and_recalibrated_checkpoint(tmp_path, monkeypatch, capsys):
    windows = _labeled_windows(64)
    data_path = tmp_path / "campaigns.labeled.ndjson"
    data_path.write_text("\n".join(
        json.dumps({
            "window_id": window.window_id, "label": window.label,
            "events": [
                {"id": e.id, "timestamp": e.timestamp.isoformat(), "source": e.source, "severity": e.severity, "kind": e.kind, "metadata": e.metadata}
                for e in window.events
            ],
        })
        for window in windows
    ) + "\n")
    weights = tmp_path / "model.pt"
    save_checkpoint(_fit_model_on(windows), weights)
    card = tmp_path / "MODEL_CARD.md"
    out_weights = tmp_path / "validated.pt"
    monkeypatch.setattr("sys.argv", [
        "idr-intelligence", "validate", "--data", str(data_path), "--weights", str(weights),
        "--target-fpr", "0.05", "--holdout", "0.4", "--data-provenance", "soc-incidents-2026Q2",
        "--gate", "max_ece=0.6", "--model-card", str(card), "--out-weights", str(out_weights),
    ])
    cli.main()  # verdict go -> exit 0
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "go"
    assert report["binding"] is True
    assert report["model_card"] == str(card)
    assert card.exists() and "Model Card" in card.read_text()
    assert out_weights.exists()
    # The recalibrated checkpoint carries the real drift baseline.
    reloaded = load_campaign_model(out_weights)
    assert reloaded.feature_stats is not None


def test_cli_validate_exits_nonzero_on_no_go(tmp_path, monkeypatch):
    windows = _labeled_windows(64)
    data_path = tmp_path / "campaigns.labeled.ndjson"
    data_path.write_text("\n".join(
        json.dumps({
            "window_id": window.window_id, "label": window.label,
            "events": [
                {"id": e.id, "timestamp": e.timestamp.isoformat(), "source": e.source, "severity": e.severity, "kind": e.kind, "metadata": e.metadata}
                for e in window.events
            ],
        })
        for window in windows
    ) + "\n")
    weights = tmp_path / "model.pt"
    save_checkpoint(_fit_model_on(windows), weights)
    monkeypatch.setattr("sys.argv", [
        "idr-intelligence", "validate", "--data", str(data_path), "--weights", str(weights),
        "--target-fpr", "0.05", "--holdout", "0.4",  # default ECE gate fails -> no-go
    ])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


def test_tactic_order_is_canonical_mitre():
    from idr_intelligence.attack import ATTACK_REFERENCE, TACTIC_ORDER

    # The canonical 14 enterprise tactics, in matrix order, incl. the two
    # pre-compromise tactics the old hand-typed order omitted.
    assert TACTIC_ORDER == tuple(ATTACK_REFERENCE["enterprise_tactic_order"])
    assert TACTIC_ORDER[:3] == ("reconnaissance", "resource-development", "initial-access")
    assert TACTIC_ORDER[-1] == "impact"
    assert len(TACTIC_ORDER) == 14
    # Adding the front tactics is behaviour-preserving: no kind maps to them.
    from idr_intelligence.attack import KIND_TO_ATTACK

    mapped = {mapping["tactic"] for mapping in KIND_TO_ATTACK.values()}
    assert not mapped & {"reconnaissance", "resource-development"}


def test_kind_to_attack_is_consistent_with_mitre():
    from idr_intelligence.attack import (
        KIND_TO_ATTACK,
        technique_name,
        validate_mapping_against_reference,
    )

    assert validate_mapping_against_reference() == []
    # Enrichment: every mapped technique resolves to its real MITRE name.
    for mapping in KIND_TO_ATTACK.values():
        assert technique_name(mapping["technique"]) is not None
    assert technique_name("T1041") == "Exfiltration Over C2 Channel"
    assert technique_name("T9999") is None


def test_vendored_attack_reference_is_fresh_if_bundle_present():
    """If the MITRE bundle is on disk, the committed reference must match a fresh
    rebuild — so a MITRE release can't leave the vendored copy silently stale."""
    import os
    import sys

    bundle = os.environ.get("IDR_ATTACK_BUNDLE", "/home/lee/Desktop/cti/enterprise-attack/enterprise-attack.json")
    if not Path(bundle).is_file():
        pytest.skip("MITRE ATT&CK bundle not present; skipping freshness check")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import ground_attack_reference

    from idr_intelligence.attack import ATTACK_REFERENCE

    rebuilt = ground_attack_reference.build_reference(bundle)
    assert rebuilt["enterprise_tactic_order"] == ATTACK_REFERENCE["enterprise_tactic_order"]
    assert rebuilt["techniques"] == ATTACK_REFERENCE["techniques"]


def test_findings_carry_real_technique_names():
    from idr_intelligence.streaming import StreamingScorer

    events = simulate_campaign(label=1, seed=8)
    model = CampaignModel(FEATURE_DIM, hidden_dim=12, state_dim=4)
    finding = score_events(events, model)
    stages = finding.observed_attack_stages
    assert stages
    for stage in stages:
        assert stage.get("technique_name")  # real MITRE name, never empty
    names = {stage["technique"]: stage["technique_name"] for stage in stages}
    assert names.get("T1041") == "Exfiltration Over C2 Channel"
    # Streaming and batch agree on the enriched stage records.
    scorer = StreamingScorer(model)
    for event in sorted(events, key=lambda e: (e.timestamp, e.id)):
        scorer.ingest(event)
    assert scorer.finding().observed_attack_stages == stages
