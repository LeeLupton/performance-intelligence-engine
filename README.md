# Performance Intelligence Engine — IDR Intelligence

[![CI](https://github.com/LeeLupton/performance-intelligence-engine/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/LeeLupton/performance-intelligence-engine/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.1.0-informational)](https://github.com/LeeLupton/performance-intelligence-engine)
[![License](https://img.shields.io/github/license/LeeLupton/performance-intelligence-engine)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.2-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

A portfolio-grade temporal-graph research system for the `idr-main` intrusion detection and response pipeline.

The project consumes the canonical JSON shape emitted by `idr_common::IdrEvent`, reconstructs relationships among hosts, processes, executable hashes, IPs, prefixes, ASNs, domains, gateways, and storage devices, then combines:

- a selective S6-style state-space encoder for ordered entity histories;
- a residual graph neural network for cross-entity campaign context;
- evidence-preserving output suitable for corroboration by `idr-sentinel`.

It is designed to answer a concrete question:

> Do individually weak signals become a credible campaign when their order and infrastructure relationships are evaluated together?

## What's built

- **Model** — a selective S6 state-space encoder over each entity's ordered history feeding a residual GNN over the entity graph, with gated-attention pooling whose scores *are* the evidence ranking (so the ranking is trained, not a random projection).
- **Calibrated, provenanced findings** — every `IntelligenceFinding` carries an affine-calibrated probability (plus the raw value and calibration string), an ATT&CK stage narrative, per-entity evidence (the events implicating each entity, its typed edges to other flagged entities, the features that drove it by occlusion attribution, and its ATT&CK techniques), a feature-schema hash, and an advisory feature-drift block.
- **Honest evaluation** — a five-arm chronological ablation with calibration (ECE, log-loss) and operating-point (recall@FPR, precision@k) metrics; cross-scenario generalization over 11 simulator families; rolling-origin cross-validation that reports a statistical verdict (or an honest "tie"); and a frozen benchmark manifest whose regression floors fail CI on every build.
- **Real-data path** — `LabeledWindow` ingestion (`--data`) swaps the simulator for real `*.labeled.ndjson` campaigns with zero model changes.
- **Deployment controls** — an analyst suppression allowlist that attenuates entities out of the ranking without hiding the finding or touching the campaign probability; a bounded, audited node budget for streaming memory.

The simulator uses event families already present in `idr-main` (`socket_lineage`, `suspicious_beacon`, `bgp_anomaly`, `ntp_time_shift`, `hsts_time_manipulation`, `nvme_latency_anomaly`, …) and 11 scenario families spanning graded difficulty, hard negatives, evasion (low-and-slow, split-host, hash-rotation), identity-pivot lateral movement, and timing-only discrimination.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
idr-intelligence demo --samples 80 --epochs 3 --output reports/demo.json
```

`demo` prints the ablation benchmark and an evidence-linked `IntelligenceFinding`. Other subcommands:

```bash
idr-intelligence score events.ndjson --weights artifacts/hybrid_model.pt   # score an NDJSON export (each line a serialized IdrEvent)
idr-intelligence score events.ndjson --suppress 'ip:' --suppress host:known-scanner  # analyst allowlist
idr-intelligence benchmark --manifest benchmarks/v1.json                    # frozen regression floors; exit 1 on violation (runs in CI)
idr-intelligence ablation --folds 3 --replicates 3                          # rolling-origin CV with a statistical best-model verdict
idr-intelligence time-ablation --scenario timing_only                       # global vs per-entity vs time-aware S6
idr-intelligence decay-ablation --scenario distractor                       # edge-decay half-life comparison
```

Synthetic metrics demonstrate that the software and evaluation workflow function; they are not claims of production detection accuracy — see `reports/AUDIT.md`.

## A note on the temporal-physics workstreams

The engine ships per-entity time deltas, time-aware S6 discretization, and time-decayed edges — architecturally correct and fully tested, but **verified (not assumed) to be undifferentiated on synthetic data**: a single `delta_seconds_log` feature already captures the inter-event gap, so on the `timing_only` scenario (structure held identical, timing the sole discriminator) the simpler `global` mode wins. They stay as opt-in modes for real high-cardinality streams; the shipped default is the simpler mode. This is documented in `reports/AUDIT.md` so the machinery is never mistaken for a demonstrated accuracy gain.

## Integration boundary

```text
idr-main IdrEvent NDJSON
          ↓
entity extraction + temporal graph
          ↓
S6 temporal state + GNN relational state
          ↓
IntelligenceFinding with evidence IDs
          ↓
idr-sentinel deterministic corroboration
```

The model must not trigger `PanicResponse` by itself. It is an advisory layer that raises a hypothesis and points the existing correlator to the exact source evidence.

## Repository map

```text
src/idr_intelligence/schema.py      IdrEvent + LabeledWindow validation
src/idr_intelligence/config.py      EngineConfig: typed tunables, prior tables, config hash
src/idr_intelligence/registry.py    ModelManifest, feature-schema hash, SchemaMismatchError
src/idr_intelligence/features.py    entity extraction (incl. identity), typed edges, features
src/idr_intelligence/graph.py       temporal graph: per-entity time, decayed edges, node budget
src/idr_intelligence/bounded_graph.py  GraphBudget + audited eviction
src/idr_intelligence/models.py      S6, GNN, gated-attention pooling, checkpoints
src/idr_intelligence/attack.py      deterministic kind→ATT&CK mapping + next-stage
src/idr_intelligence/simulator.py   11 scenario families with stage-level ground truth
src/idr_intelligence/training.py    ablation, calibration, scenario gen, rolling-origin CV
src/idr_intelligence/benchmark.py   frozen-manifest regression floors (CI gate)
src/idr_intelligence/dataio.py      *.labeled.ndjson real-campaign ingestion
src/idr_intelligence/pipeline.py    evidence-linked, calibrated scoring
src/idr_intelligence/evidence.py    per-entity evidence (occlusion, edges, ATT&CK) + suppression
src/idr_intelligence/cli.py         demo · score · benchmark · ablation · time/decay-ablation
benchmarks/v1.json                  frozen benchmark manifest with regression floors
docs/ARCHITECTURE.md                integration design + Rust EventKind contract
reports/AUDIT.md                    model-risk audit + verified findings
state.json                          canonical engineering state record (ADR log, evidence)
```

## Resume bullet

> Built a temporal-graph threat-intelligence engine for a Rust-based cross-layer IDR platform: selective S6 state-space modeling with GNN propagation and trained gated-attention evidence ranking across host, process, hash, IP, prefix, ASN, domain, identity, and hardware entities. Shipped calibrated, provenanced, ATT&CK-mapped findings with per-entity occlusion evidence; a CI-gated regression benchmark over 11 adversarial scenario families with rolling-origin cross-validation; real labeled-data ingestion; and model-risk controls — including adversarial self-review that caught a genuine calibration bug and a verified finding that the temporal-physics elaborations add no measurable value on synthetic data.
