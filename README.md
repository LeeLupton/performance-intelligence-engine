# Performance Intelligence Engine — IDR Intelligence

[![CI](https://github.com/LeeLupton/performance-intelligence-engine/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/LeeLupton/performance-intelligence-engine/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/rust-1.96-000000?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Version](https://img.shields.io/badge/version-0.1.0-informational)](https://github.com/LeeLupton/performance-intelligence-engine)
[![License](https://img.shields.io/github/license/LeeLupton/performance-intelligence-engine)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.2-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

A temporal-graph campaign-inference engine that sits beside the `idr-main`
intrusion-detection platform. It reads the platform's event stream, reconstructs
the relationships among hosts, processes, executable hashes, IPs, prefixes,
ASNs, domains, identities, and hardware, and answers one question:

> Do individually weak signals become a credible campaign when their **order**
> and **infrastructure relationships** are evaluated together?

Two different kinds of context, and the architecture maps onto them one-for-one:
a **selective S6 state-space encoder** for order, a **residual graph neural
network** for relationships. Output is an advisory `IntelligenceFinding` — a
calibrated campaign probability, a ranked evidence trail, and an ATT&CK stage
narrative — handed to `idr-sentinel` for deterministic corroboration.

---

## Status — read this first

This is **production-grade software wrapped around a research-grade model**, and
the README keeps those two apart on purpose:

- **Software readiness — strong.** 131 Python tests + a Rust serving bridge +
  wire-parity tests, all CI-gated; strict input validation, model provenance,
  bounded memory, structured logging, reproducible builds, and an adversarial
  review that fixed 19 findings.
- **Detector readiness — blocked on data, not code.** Every accuracy number in
  this repo comes from a synthetic simulator. That proves the *pipeline and
  evaluation workflow* function; it is **not** evidence of real-world detection
  accuracy. A binding go/no-go needs real labeled campaign windows, which do not
  yet exist on the platform.

The full assessment, and the staged path to go-live, is in
[`reports/PRODUCTION_READINESS.md`](reports/PRODUCTION_READINESS.md).

---

## What's built

**Model & findings**
- A selective **S6 state-space encoder** over each entity's ordered history,
  feeding a **residual GNN** over the entity graph, with **gated-attention
  pooling** whose scores *are* the evidence ranking (trained, not a random
  projection).
- Every `IntelligenceFinding` carries an **affine-calibrated probability** (plus
  the raw sigmoid and the calibration string), an **ATT&CK stage narrative**
  (deterministic, grounded in real MITRE data — see below), **per-entity
  evidence** (implicating events, typed edges, top features by occlusion
  attribution, techniques), a **feature-schema hash** for provenance, and an
  advisory **feature-drift** block (PSI vs a training baseline).

**Streaming & serving**
- **`StreamingScorer`** ingests events one at a time, carrying each entity's S6
  state forward with the same `step()` cell training uses — O(1) work and state
  per event, no history replay — and runs the relational head on demand.
  Findings are test-pinned equivalent to batch scoring across all time modes,
  edge decay, and drift.
- **ONNX export + pure-Rust serving bridge.** `idr-intelligence export` writes
  the streaming surface as two ONNX graphs (the `step()` cell and the relational
  head) plus a manifest carrying every scoring constant.
  [`rust/idr-intelligence-rt`](rust/idr-intelligence-rt) runs the bundle on
  [tract](https://github.com/sonos/tract) — no Python, no ONNX Runtime — and
  reproduces `StreamingScorer` findings field-for-field, pinned by committed
  golden streams and a fuzzed `datetime.fromisoformat`-mirroring timestamp
  parser.

**Real data & validation**
- **Real-data ingestion** (`--data`) swaps the simulator for real
  `*.labeled.ndjson` campaign windows with zero model changes.
- **Validation gate** (`idr-intelligence validate`) — the go/no-go step to
  production: on a strictly time-ordered holdout it re-fits calibration, picks
  an operating threshold at a target FPR, snapshots a real drift baseline,
  verifies the *realized* FPR on an untouched later segment, and turns
  configurable gates into a verdict + model card. Its verdict is **binding only
  when the operator attests real data** — a synthetic run is an explicit,
  non-binding wiring check.
- **Live-telemetry bridge & label driver** (machine-local, next to `idr-main`) —
  `scripts/export_sentinel_events.py` turns the running sentinel's anomaly audit
  log into `IdrEvent` NDJSON; `scripts/build_kill_chain_dataset.py` and
  `rust/sentinel-label-harness` build labeled multi-signal datasets, the latter
  by driving idr-main's **real `SentinelCorrelator`** and labeling each window
  by the correlator's own `ImpossibleState` verdict.

**Grounding & operations**
- **Real MITRE ATT&CK grounding.** The tactic order and kind→technique mapping
  are validated against the MITRE Enterprise STIX bundle and enriched with real
  technique names; a freshness test rebuilds the vendored reference from the
  bundle when present.
- **Structured logging** (`--log-level`) — opt-in JSON lines to stderr (model
  provenance, timing, finding summaries, drift, evictions), never touching the
  stdout finding contract.
- **Deployment** — `make gates` runs the full CI locally; Dockerfiles package
  the Python engine and the minimal Rust serving binary.
- **Deployment controls** — a suppression allowlist that attenuates entities
  from the ranking without hiding the finding, and a bounded, audited node
  budget for streaming memory.
- **Campaign identity across windows** — an opt-in registry (`--registry`)
  matches each window's durable-entity fingerprint against known campaigns by
  weighted Jaccard, so corroboration accumulates over days. Deterministic and
  recomputable from the registry JSON.

---

## How it works

```text
  idr-main IdrEvent NDJSON
            │
            ▼
  entity extraction + temporal graph      schema.py · features.py · graph.py
            │
            ▼
  S6 temporal state  +  GNN relational state   models.py
            │
            ▼
  IntelligenceFinding (calibrated p, evidence IDs, ATT&CK stages, drift)
            │
            ▼
  idr-sentinel deterministic corroboration
```

The model is advisory: it **must not trigger `PanicResponse` by itself**. It
raises a hypothesis and points the existing correlator to the exact source
evidence. Latent state is never a substitute for primary telemetry.

---

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,export]'
pytest
idr-intelligence demo --samples 80 --epochs 3 --output reports/demo.json
```

`make gates` runs every CI check locally (pytest, ruff, mypy, benchmark floors,
and the Rust bridge's fmt/clippy/test). `make docker` / `make docker-rt` build
the engine and serving images.

`demo` prints the ablation benchmark and an evidence-linked
`IntelligenceFinding`. The CLI (every command also takes `--log-level`):

```bash
# Scoring
idr-intelligence score  events.ndjson --weights artifacts/hybrid_model.pt      # score an NDJSON IdrEvent export
idr-intelligence score  events.ndjson --suppress 'ip:' --suppress host:scanner # analyst suppression allowlist
idr-intelligence score  events.ndjson --registry campaigns.json                # stable campaign ids across windows
idr-intelligence stream events.ndjson --max-nodes 64                           # event-at-a-time over carried S6 state (default bound 4096; 0 = off)

# Serving
idr-intelligence export --weights artifacts/hybrid_model.pt --out artifacts/export   # ONNX bundle for the Rust bridge

# Real data & validation
idr-intelligence validate --data campaigns/ --data-provenance soc-2026Q2 --model-card CARD.md   # go/no-go gate + model card

# Evaluation
idr-intelligence benchmark      --manifest benchmarks/v1.json   # frozen regression floors; exit 1 on violation (CI)
idr-intelligence ablation       --folds 3 --replicates 3        # rolling-origin CV with a statistical best-model verdict
idr-intelligence time-ablation  --scenario timing_only          # global vs per-entity vs time-aware S6
idr-intelligence decay-ablation --scenario distractor           # edge-decay half-life comparison
```

Synthetic metrics demonstrate that the software and evaluation workflow
function; they are not claims of production detection accuracy — see
[`reports/AUDIT.md`](reports/AUDIT.md).

---

## Honest evaluation

The benchmark can say *no*: a five-arm chronological ablation with calibration
(ECE, log-loss) and operating-point (recall@FPR, precision@k) metrics; 11
simulator scenario families spanning graded difficulty, hard negatives, evasion
(low-and-slow, split-host, hash-rotation), identity-pivot lateral movement, and
timing-only discrimination; rolling-origin cross-validation that returns a blunt
**tie** rather than crowning single-split noise; and a frozen benchmark manifest
whose regression floors fail CI on every build.

**On the temporal-physics workstreams:** the engine ships per-entity time
deltas, time-aware S6 discretization, and time-decayed edges — correct and fully
tested, but **verified (not assumed) to be undifferentiated on synthetic data**.
A single `delta_seconds_log` feature already captures the inter-event gap, so on
the `timing_only` scenario (structure held identical, timing the sole
discriminator) the simpler `global` mode wins. They remain opt-in modes for real
high-cardinality streams; the shipped default is the simpler one. Recorded in
`reports/AUDIT.md` so the machinery is never mistaken for an accuracy gain.

---

## Repository map

```text
Core pipeline
  src/idr_intelligence/schema.py      IdrEvent + LabeledWindow validation, ISO-8601 timestamps
  src/idr_intelligence/features.py    entity extraction (incl. identity), typed edges, 22-dim features
  src/idr_intelligence/graph.py       temporal graph: per-entity time, decayed edges, node budget
  src/idr_intelligence/bounded_graph.py  GraphBudget + audited eviction
  src/idr_intelligence/pipeline.py    evidence-linked, calibrated scoring
  src/idr_intelligence/evidence.py    per-entity evidence (occlusion, edges, ATT&CK) + suppression
  src/idr_intelligence/streaming.py   event-at-a-time scoring over carried S6 state

Model & config
  src/idr_intelligence/models.py      S6, GNN, gated-attention pooling, checkpoints
  src/idr_intelligence/config.py      EngineConfig: typed tunables, prior tables, config hash
  src/idr_intelligence/registry.py    ModelManifest, feature-schema hash, SchemaMismatchError
  src/idr_intelligence/attack.py      deterministic kind→ATT&CK mapping + next-stage (real MITRE)
  src/idr_intelligence/data/attack_reference.json  tactic order + technique catalog from MITRE STIX

Evaluation & data
  src/idr_intelligence/simulator.py   11 scenario families with stage-level ground truth
  src/idr_intelligence/training.py    ablation, calibration, scenario gen, rolling-origin CV
  src/idr_intelligence/benchmark.py   frozen-manifest regression floors (CI gate)
  src/idr_intelligence/dataio.py      *.labeled.ndjson real-campaign ingestion
  src/idr_intelligence/validation.py  real-data go/no-go gate: recalibrate, threshold, drift, model card

Serving & operations
  src/idr_intelligence/export.py      ONNX bundle export + torch-free reference runner (serving contract)
  src/idr_intelligence/observability.py  opt-in JSON-lines stderr logging
  src/idr_intelligence/cli.py         demo · score · stream · export · validate · benchmark · ablation · time/decay-ablation

Rust
  rust/idr-intelligence-rt/           serving bridge on tract, golden-pinned to StreamingScorer (CI)
  rust/idr-common/                    vendored idr_common wire types from idr-main (see its VENDOR.md)
  rust/idr-common-parity/             wire parity vs the real idr_common types (CI)
  rust/sentinel-label-harness/        red-team label driver: synthetic windows → real SentinelCorrelator → labels (machine-local)

Real-data bridges (machine-local, next to idr-main)
  scripts/export_sentinel_events.py   live sentinel anomaly audit log → IdrEvent NDJSON
  scripts/build_kill_chain_dataset.py labeled multi-signal dataset from idr-sim + real BGP
  scripts/ground_attack_reference.py  regenerate the ATT&CK reference from a MITRE STIX bundle

Docs & record
  docs/ARCHITECTURE.md                integration design + Rust EventKind contract
  reports/AUDIT.md                    model-risk audit + verified findings
  reports/PRODUCTION_READINESS.md     software-vs-detector readiness + staged go-live path
  state.json                          canonical engineering state record (ADR log, evidence)
  Makefile · Dockerfile · rust/idr-intelligence-rt/Dockerfile   reproducible gates + images
```

---

## Resume bullet

> Built a temporal-graph threat-intelligence engine for a Rust-based cross-layer
> IDR platform: selective S6 state-space modeling with GNN propagation and
> trained gated-attention evidence ranking across host, process, hash, IP,
> prefix, ASN, domain, identity, and hardware entities. Shipped calibrated,
> ATT&CK-grounded, provenanced findings with per-entity occlusion evidence; a
> CI-gated regression benchmark over 11 scenario families spanning graded
> difficulty, hard negatives, and evasion, with rolling-origin cross-validation;
> an ONNX export + pure-Rust (tract) serving bridge proven field-for-field
> against the Python scorer; a validation gate that recalibrates and selects an
> operating threshold on a temporal holdout, emitting go/no-go verdicts and
> model cards — binding only on attested real data; and a label driver that
> adjudicates synthetic campaigns through the platform's *real* correlator. Kept
> a scrupulous software-vs-detector honesty line throughout — including
> adversarial self-review that caught a genuine calibration bug and a verified
> finding that the temporal-physics elaborations add no measurable value on
> synthetic data.
