"""Regenerate the cross-language parity fixtures for idr-intelligence-rt.

Run from the repository root with the project environment:

    .venv/bin/python rust/idr-intelligence-rt/tests/fixtures/generate.py

Everything here is deterministic (seeded init, simulator streams, content-
addressed event ids), and the script asserts its own consistency before
writing:

- the ONNX reference runner must agree with the torch StreamingScorer on every
  golden (including the budget/suppression and no-GNN cases);
- ranked node probabilities must clear RANKING_GAP_FLOOR at the top-k boundary
  of every golden, so cross-runtime float drift can never reorder a ranking;
- the drift baseline is a real feature histogram (not uniform), and the
  malicious golden must flag some-but-not-all features, so the PSI threshold
  is exercised on both sides of the line;
- event ids are content-addressed per case, so campaign_id (first-event
  selection) genuinely discriminates between implementations.

The committed outputs pin the Rust bridge to the Python engine;
tests/test_system.py's freshness gate replays them against the current engine
so a semantic change without regeneration fails CI.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import uuid
from pathlib import Path

import numpy as np
import torch

from idr_intelligence.bounded_graph import GraphBudget
from idr_intelligence.config import DEFAULT_CONFIG
from idr_intelligence.export import OnnxStreamScorer, export_streaming_bundle
from idr_intelligence.features import FEATURE_DIM, project_event
from idr_intelligence.graph import degree_normalize
from idr_intelligence.models import CampaignModel
from idr_intelligence.schema import parse_events
from idr_intelligence.simulator import simulate_campaign
from idr_intelligence.streaming import StreamingScorer

FIXTURES = Path(__file__).resolve().parent

# Min separation between consecutive top-k node probabilities. Cross-runtime
# (torch vs onnxruntime vs tract) drift on these graphs is ~1e-7; 5e-5 keeps
# two orders of magnitude of margin so the ranking can never reorder.
RANKING_GAP_FLOOR = 5e-5


def remap_ids(events, salt):
    """Content-addressed per-case event ids.

    The simulator's uuid.UUID(int=small) ids all share a '00000000' prefix,
    which would make campaign_id (first 8 chars of the first event's id) a
    constant no matter which event an implementation picks as first — ids must
    discriminate for the golden to pin first-event selection. The salt also
    perturbs same-timestamp ordering, which the gap search below exploits.
    """
    return [
        dataclasses.replace(
            event,
            id=str(uuid.UUID(bytes=hashlib.blake2s(f"{salt}:{event.id}".encode(), digest_size=16).digest())),
        )
        for event in events
    ]


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


def node_probabilities(model, scorer):
    """Node probabilities exactly as finding() computes them, for gap checks."""
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
        probabilities = torch.sigmoid(output.node_logits[0]).numpy().astype(np.float64)
    return node_ids, probabilities


def assert_ranked_gaps(model, scorer, label, suppressions=()):
    """Only the top-k boundary decides the finding; deep-tail ties (structurally
    identical entities share exact logits) never surface in related/evidence."""
    node_ids, probabilities = node_probabilities(model, scorer)
    for index, entity in enumerate(node_ids):
        if any(entity == rule or (rule.endswith(":") and entity.startswith(rule)) for rule in suppressions):
            probabilities[index] = -np.inf
    ranked = np.sort(probabilities[np.isfinite(probabilities)])[::-1][: DEFAULT_CONFIG.scoring.top_k + 1]
    gaps = np.diff(-ranked) if len(ranked) > 1 else np.array([np.inf])
    assert gaps.min() >= RANKING_GAP_FLOOR, f"{label}: ranked probabilities too close ({gaps.min():.2e}); reseed"


def timestamp_battery():
    """Exhaustive timestamp-acceptance contract, decided by the reference
    interpreter: for every shape, record the UTC instant schema.py produces or
    null for rejection. The Rust parser must agree on every entry — both the
    accept/reject verdict and the instant — so acceptance parity is a pinned
    fixture, not a claim."""
    from idr_intelligence.schema import _parse_timestamp

    dates = ["2026-03-01", "20260301", "2026-W09-7", "2026W097", "2026-W09", "2026W09",
             "2026-03", "2026", "2026-3-01", "2026-03-1", "2026-13-01", "2026-02-30"]
    seps = ["T", "t", " ", "_", "x", "+"]
    times = ["11", "1109", "11:09", "110930", "11:09:30", "11:09:30.5", "11:09:30,5",
             "11:09:30.123456789", "110930.5", "24:00", "24:00:00", "24:00:01",
             "11:60:00", "11:09:60", "9:09:30"]
    offsets = ["", "Z", "z", "+00", "+0000", "+00:00", "+09", "+0930", "+09:30", "-05",
               "+09:30:15", "+09:30:15.123456", "+093015", "+5", "+24:00"]
    shapes = list(dates)
    shapes += [f"2026-03-01T{time}{offset}" for time in times for offset in offsets]
    shapes += [f"{date}T11:09:30{offset}" for date in dates for offset in ("", "Z", "+09:30")]
    shapes += [f"2026-03-01{sep}{time}{offset}" for sep in seps for time in ("11:09", "11:09:30") for offset in ("", "Z")]
    shapes += [f"20260301T{time}{offset}" for time in times for offset in ("", "Z", "+0000", "+09")]
    shapes += ["", "notatime", "2026-03-01T", "2026-03-01T00:09:30X", "2026-03-01TT00:09",
               "99999999", "2026-03-01 T 00:09", "2026-03-01T00:09:30.",
               # fraction placement, mixed basic/extended, week+time, Python range limits
               "2026-03-01T11.5", "2026-03-01T11:09.5", "2026-03-01T11:0930", "2026-03-01T1109:30",
               "2026-W09T11:09", "2026W097T11:09Z", "2026-03-01T11:09:30.+00",
               "2026-03-01T11:09:30+09:30.5", "2026-03-01T0009", "20260301T11:09",
               "0001-01-01T00:00+23:59", "9999-12-31T23:59+00", "9999-12-31T00:00-23:59",
               "2026-03-01T11 ", "2026-W60-1", "2026-W09-8", "2026-W09-77T00",
               # discovered mechanism edges: junk separators, digit-run week
               # heuristics, quota fractions, tz-scan asymmetries, day-0 rollover
               "2026-03-01T11W+00", "2026-03-01T11:09:30XZ", "2026-03-01T11109+09:30",
               "2026-03-01T119+00", "2026-03-01T1109401+00", "2026-03-01T1109012345",
               "2026-03-01T11:09:30.123456:+09", "2026-03-01T11:09:30.5W+00",
               "2026-03-01T11:09:30,2.-0530", "2026-03-01T11:09:30.123456z89+09",
               "2026W09711:09", "2026W097311:09", "2026W09791109305Z", "2026-W09-1109",
               "2026-W09911Z+09", "2026-03-01T11Z-05", "2026-03-01T11W-05",
               "20260300T24:00:00", "2026-01-31T24:00:00", "20260232T24:00:00",
               "2026-03-01T1109305Z", "2026-03-01T11:0930", "2026-03-01T1109:30"]
    entries = []
    for shape in dict.fromkeys(shapes):
        try:
            parsed = _parse_timestamp(shape)
            utc = parsed.isoformat()
        except (ValueError, OverflowError):
            utc = None
        entries.append({"shape": shape, "utc": utc})
    return entries


def drift_baseline():
    """Histogram real projected feature rows as the training snapshot, so the
    PSI goldens are non-degenerate and the flag threshold is exercised."""
    events = []
    for label, seed in ((0, 21), (1, 22), (0, 23), (1, 24)):
        events.extend(simulate_campaign(label, seed))
    ordered = sorted(events, key=lambda event: (event.timestamp, event.id))
    rows, previous = [], None
    for event in ordered:
        delta = (event.timestamp - previous).total_seconds() if previous is not None else 0.0
        previous = event.timestamp
        rows.append(project_event(event, delta_seconds=delta).features)
    stacked = np.stack(rows)
    edges = np.linspace(0.0, 1.0, 11)
    histograms = [np.histogram(stacked[:, index], bins=edges)[0].tolist() for index in range(FEATURE_DIM)]
    return {"bin_edges": edges.tolist(), "histograms": histograms}


# Hand-authored wire events covering the extraction branches the simulator
# never emits: dest_ips, bare sha256, every *_ip key, ptr_query/sni,
# username/account, session_id/sid, cloud_resource/arn, hostname metadata,
# tgid-only and pid=0 processes, falsy values, unknown kinds/severities, and
# the timestamp shapes (naive, space-separated, nanosecond) whose parsing the
# Rust port must mirror.
EDGE_CASE_EVENTS = [
    {"id": "e1000000-0000-4000-8000-000000000001", "timestamp": "2026-03-01T00:00:00+00:00", "source": "kernel_ebpf",
     "severity": "HIGH", "metadata": {"host": "edge-host"},
     "kind": {"type": "triage_classification", "family": "loader", "sha256": "ABCDEF" + "0" * 58,
              "dest_ips": ["203.0.113.9", "203.0.113.9", "198.51.100.7"], "source_path": "/tmp/x"}},
    {"id": "e1000000-0000-4000-8000-000000000002", "timestamp": "2026-03-01T00:01:00.123456789+00:00",
     "source": "network_zeek", "severity": "WARNING", "metadata": {"hostname": "fallback-host"},
     "kind": {"type": "octet_reversal_detected", "forward_ip": "142.251.211.170", "reversed_ip": "170.211.251.142",
              "ptr_query": "MiXeD.PTR.Example.COM..", "forward_asn": "AS15169", "reversed_asn": "AS21852"}},
    {"id": "e1000000-0000-4000-8000-000000000003", "timestamp": "2026-03-01 00:02:00", "source": "network_suricata",
     "severity": "high", "metadata": None,
     "kind": {"type": "unknown_future_kind", "sni": "cdn.example", "src_ip": "192.0.2.1", "gateway_ip": "192.0.2.254"}},
    {"id": "e1000000-0000-4000-8000-000000000004", "timestamp": "2026-03-01T00:03:00", "source": "custom_source",
     "severity": "WEIRD", "metadata": [],
     "kind": {"type": "mac_flapping", "gateway_ip": "192.0.2.254", "old_mac": "aa:bb", "new_mac": "cc:dd"}},
    {"id": "e1000000-0000-4000-8000-000000000005", "timestamp": "2026-03-01T00:04:00+00:00", "source": "kernel_ebpf",
     "severity": "INFO", "metadata": {"host": "edge-host"},
     "kind": {"type": "socket_lineage", "tgid": 4100, "exe_path": "/usr/bin/x", "exe_sha256": "",
              "dst_ip": "", "dst_port": 443, "is_signed": True}},
    {"id": "e1000000-0000-4000-8000-000000000006", "timestamp": "2026-03-01T00:05:00+00:00", "source": "kernel_ebpf",
     "severity": "CRITICAL", "metadata": {"host": "edge-host"},
     "kind": {"type": "suspicious_beacon", "pid": 0, "tgid": 0, "exe_path": "/tmp/y",
              "exe_sha256": "DEADBEEF" + "1" * 56, "dst_ip": "203.0.113.9", "asn_owner": "Example Transit",
              "user": "SVC-Account", "session_id": 0, "cloud_resource": "ARN:AWS:S3:::Bucket/Key"}},
    {"id": "e1000000-0000-4000-8000-000000000007", "timestamp": "2026-03-01T00:06:00+00:00", "source": "hardware_rtc",
     "severity": "HIGH", "metadata": {"host": "edge-host"},
     "kind": {"type": "rtc_clock_divergence", "software_time": "x", "rtc_time": "y", "drift_seconds": 90.0,
              "username": "Backup-Operator", "sid": "S-1-5-21", "arn": "arn:aws:iam::1:role/Admin",
              "target_host": "server-1", "device": "rtc0"}},
    {"id": "e1000000-0000-4000-8000-000000000008", "timestamp": "2026-03-01T00:07:00+00:00",
     "source": "sentinel_correlation", "severity": "HIGH", "metadata": {"host": "edge-host"},
     "kind": {"type": "bgp_anomaly", "kind": {"kind": "route_leak_benign"}, "prefix": "203.0.113.0/25",
              "observed_origin_asn": 0, "legitimate_origin_asn": None, "confidence": "low"}},
    {"id": "e1000000-0000-4000-8000-000000000009", "timestamp": "2026-03-01T00:08:00+00:00", "source": "network_zeek",
     "severity": "HIGH", "metadata": {"host": "edge-host"},
     "kind": {"type": "hsts_time_manipulation", "domain": "Update-CDN.Example.", "cert_expiry": "2025-12-01",
              "ntp_shift_seconds": 90.0, "account": "", "user": None}},
    {"id": "e1000000-0000-4000-8000-00000000000a", "timestamp": "2026-03-01T00:09:30,5", "source": "network_zeek",
     "severity": "WARNING", "metadata": {"host": "edge-host"},
     "kind": {"type": "ntp_time_shift", "offset_seconds": 90.0, "ntp_server": "192.0.2.9"}},
    {"id": "e1000000-0000-4000-8000-00000000000b", "timestamp": "20260301T001000", "source": "kernel_ebpf",
     "severity": "IMPOSSIBLE", "metadata": {"host": "edge-host"},
     "kind": {"type": "impossible_state", "correlated_event_ids": [], "description": "x", "kill_chain_stage": "y"}},
    {"id": "e1000000-0000-4000-8000-00000000000c", "timestamp": "2026-03-01T11", "source": "hardware_nvme",
     "severity": "HIGH", "metadata": {"host": "edge-host"},
     "kind": {"type": "nvme_latency_anomaly", "device": "nvme0n1", "baseline_us": 120, "observed_us": 500,
              "deviation_pct": 316.0, "concurrent_exfil": False}},
]


def make_bundle(out_dir, seed, use_gnn, feature_stats):
    torch.manual_seed(seed)
    model = CampaignModel(
        FEATURE_DIM, hidden_dim=24, state_dim=6, use_gnn=use_gnn, time_mode="time_aware", decay_half_life=3600.0
    )
    with torch.no_grad():
        model.temperature.fill_(1.35)
        model.cal_bias.fill_(-0.18)
    model.feature_stats = feature_stats
    export_streaming_bundle(model, out_dir, model_version="fixture-v1")
    return model


def golden(model, bundle_dir, ordered, out_path, budget_nodes=None, suppressions=(), pin_ranking=True):
    """Score with torch + the ONNX runner, assert agreement, write the expected file.

    pin_ranking=False is for goldens whose stream has exact ties in the ranked
    region by construction (the no-GNN model cannot separate isomorphic
    entities): probabilities and discrete graph fields still pin, ranked order
    does not — the main bundle's goldens own order pinning.
    """
    budget = GraphBudget(max_nodes=budget_nodes) if budget_nodes else None
    reference = StreamingScorer(model, budget=budget, model_version="fixture-v1")
    exported = OnnxStreamScorer(bundle_dir, budget=GraphBudget(max_nodes=budget_nodes) if budget_nodes else None)
    for event in ordered:
        reference.ingest(event)
        exported.ingest(event)
    want = reference.finding(suppressions=list(suppressions) or None)
    got = exported.finding(suppressions=list(suppressions) or None)
    if pin_ranking:
        assert got.related_entities == want.related_entities, f"{out_path.name}: ORT runner reorders entities"
        assert_ranked_gaps(model, reference, out_path.name, suppressions=suppressions)
    else:
        assert set(got.related_entities) == set(want.related_entities), out_path.name
    assert abs(got.escalation_probability - want.escalation_probability) <= 1e-4, out_path.name
    assert got.feature_drift == want.feature_drift, out_path.name
    assert [record.entity for record in exported.evictions] == [record.entity for record in reference.evictions]
    payload = want.to_dict()
    if budget_nodes:
        payload["evictions"] = [
            {"entity": record.entity, "last_seen": record.last_seen.isoformat(), "reason": record.reason}
            for record in reference.evictions
        ]
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return want


def stream(model, ordered, budget_nodes=None):
    scorer = StreamingScorer(
        model, budget=GraphBudget(max_nodes=budget_nodes) if budget_nodes else None, model_version="fixture-v1"
    )
    for event in ordered:
        scorer.ingest(event)
    return scorer


def main() -> None:
    stats = drift_baseline()
    model = make_bundle(FIXTURES, seed=20260723, use_gnn=True, feature_stats=stats)
    nognn_dir = FIXTURES / "nognn"
    nognn_dir.mkdir(exist_ok=True)
    nognn_model = make_bundle(nognn_dir, seed=20260724, use_gnn=False, feature_stats=None)

    # Mixed multi-campaign streams: single-scenario streams are full of
    # structurally identical (isomorphic) entities whose probabilities tie
    # exactly, which no seed can fix. The mixes below have structurally
    # distinct top-k entities; because the id remap perturbs same-timestamp
    # ordering, the salt is searched deterministically until every golden that
    # will be generated from the case clears the gap floor.
    def mix(case, parts, checks):
        events = []
        for label, seed, scenario in parts:
            events.extend(simulate_campaign(label, seed, scenario=scenario))
        for attempt in range(1, 33):
            salt = case if attempt == 1 else f"{case}#{attempt}"
            ordered = sorted(remap_ids(events, salt), key=lambda event: (event.timestamp, event.id))
            try:
                for check_model, budget_nodes, suppressions in checks:
                    assert_ranked_gaps(check_model, stream(check_model, ordered, budget_nodes), f"{case}/{salt}", suppressions)
            except AssertionError:
                continue
            print(f"{case}: salt {salt!r} clears the gap floor")
            return ordered
        raise AssertionError(f"{case}: no salt in 32 attempts clears the gap floor; change the mix")

    cases = {
        "malicious": mix(
            "malicious",
            [(1, 11, "distractor"), (1, 8, "lateral_movement"), (1, 4, "stale_preamble")],
            checks=[(model, None, ()), (model, 6, ("ip:",))],
        ),
        "benign": mix(
            "benign",
            [(0, 7, "distractor"), (0, 2, "legit_update")],
            checks=[(model, None, ())],
        ),
    }
    vectors = []
    for case, ordered in cases.items():
        (FIXTURES / f"events_{case}.ndjson").write_text("\n".join(json.dumps(row) for row in event_rows(ordered)) + "\n")
        finding = golden(model, FIXTURES, ordered, FIXTURES / f"expected_{case}.json")
        if case == "malicious":
            flagged = finding.feature_drift["flagged_features"]
            assert 0 < len(flagged) < FEATURE_DIM, (
                f"drift golden is degenerate ({len(flagged)}/{FEATURE_DIM} flagged); pick a different baseline"
            )
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

    # Edge-case rows: wire-shaped dicts parsed through IdrEvent.from_dict so
    # the recorded expectations reflect envelope validation (severity
    # uppercasing, falsy metadata, naive timestamps) exactly as Rust sees them.
    edge_events = parse_events(EDGE_CASE_EVENTS)
    previous = None
    for raw, event in zip(EDGE_CASE_EVENTS, sorted(edge_events, key=lambda item: (item.timestamp, item.id)), strict=True):
        delta = (event.timestamp - previous).total_seconds() if previous is not None else 0.0
        previous = event.timestamp
        projection = project_event(event, delta_seconds=delta)
        vectors.append({
            "event": raw,
            "delta_seconds": delta,
            "entities": list(projection.entities),
            "edges": [list(edge) for edge in projection.edges],
            "features": projection.features.tolist(),
        })
    (FIXTURES / "feature_vectors.json").write_text(json.dumps(vectors, indent=2) + "\n")
    (FIXTURES / "timestamp_battery.json").write_text(json.dumps(timestamp_battery(), indent=2) + "\n")

    # Budget + suppression golden: bounded entity memory with an audited
    # eviction trail, and an ip: prefix suppressed out of the ranking.
    golden(model, FIXTURES, cases["malicious"], FIXTURES / "expected_budget.json", budget_nodes=6, suppressions=("ip:",))

    # No-GNN ablation bundle: the exported head has no adjacency input, so this
    # golden pins the conditional-feed path in both runners.
    manifest = json.loads((nognn_dir / "manifest.json").read_text())
    assert manifest["graphs"]["head"]["inputs"] == ["outputs"], "no-GNN head should drop the adjacency input"
    golden(nognn_model, nognn_dir, cases["malicious"], nognn_dir / "expected_malicious.json", pin_ranking=False)

    print(f"fixtures written to {FIXTURES}")


if __name__ == "__main__":
    main()
