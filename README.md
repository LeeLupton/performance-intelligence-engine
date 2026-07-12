# Performance Intelligence Engine — IDR Intelligence

A portfolio-grade temporal-graph research system for the `idr-main` intrusion detection and response pipeline.

The project consumes the canonical JSON shape emitted by `idr_common::IdrEvent`, reconstructs relationships among hosts, processes, executable hashes, IPs, prefixes, ASNs, domains, gateways, and storage devices, then combines:

- a selective S6-style state-space encoder for ordered entity histories;
- a residual graph neural network for cross-entity campaign context;
- evidence-preserving output suitable for corroboration by `idr-sentinel`.

It is designed to answer a concrete question:

> Do individually weak signals become a credible campaign when their order and infrastructure relationships are evaluated together?

## What the demo does

The demo creates chronological IDR-style benign and malicious campaign streams, trains four models under the same split, compares their held-out metrics, and emits an evidence-linked finding:

1. static baseline;
2. S6-only;
3. GNN-only;
4. S6 + GNN.

The simulator uses event families already present in `idr-main`, including `socket_lineage`, `suspicious_beacon`, `bgp_anomaly`, `ntp_time_shift`, `hsts_time_manipulation`, and `nvme_latency_anomaly`.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
idr-intelligence demo --samples 80 --epochs 3 --output reports/demo.json
```

The command prints a benchmark and an `IntelligenceFinding` containing a campaign probability, next-stage hypothesis, ranked entities, exact source event IDs, and model version.

Synthetic metrics demonstrate that the software and ablation workflow function; they are not claims of production detection accuracy.

## Score an IDR NDJSON export

After the demo creates `artifacts/hybrid_model.pt`:

```bash
idr-intelligence score examples/idr_campaign.ndjson \
  --weights artifacts/hybrid_model.pt
```

Each line must be a serialized `IdrEvent`.

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
src/idr_intelligence/schema.py      IdrEvent validation
src/idr_intelligence/features.py    entity and feature extraction
src/idr_intelligence/graph.py       temporal graph construction
src/idr_intelligence/models.py      S6, GNN, and hybrid model
src/idr_intelligence/training.py    chronological ablation benchmark
src/idr_intelligence/pipeline.py    evidence-linked inference
src/idr_intelligence/cli.py         demo and NDJSON scoring commands
docs/ARCHITECTURE.md                integration design
reports/AUDIT.md                    model-risk and engineering audit
state.json                          compact engineering state record
```

## Resume bullet

> Built a temporal-graph threat intelligence engine for a Rust-based cross-layer IDR platform, combining selective S6 state-space sequence modeling with GNN propagation across host, process, IP, prefix, ASN, domain, and hardware entities; implemented chronological ablations, evidence-linked findings, CI, and model-risk controls.
