# Audit Report

## Scope

Temporal-graph campaign inference over `IdrEvent` streams.

## Verified controls

- Strict validation of the canonical tagged event shape.
- Deterministic event ordering and seeded simulation.
- Chronological 60/20/20 train, validation, and test splits.
- Four-way ablation: static baseline, S6-only, GNN-only, and S6+GNN.
- Gradient checks through every architecture.
- Evidence IDs retained in each intelligence finding.
- Model output is advisory and cannot directly trigger IDR panic response.

## Known limitations

- The included benchmark is synthetic and cannot support production-accuracy claims.
- Graph message passing is homogeneous; relation counts remain auditable, but relation-specific transforms are future work.
- The reference S6 scan is explicit and optimized for readability rather than throughput.
- Probability calibration must be repeated on real temporally held-out data.
- No automatic Rust/ONNX inference bridge is included in v0.1.

## Finding: temporal physics is not empirically differentiated on synthetic data

The temporal-physics workstreams — per-entity time deltas + time-aware S6
discretization (W11) and time-decayed edges (W12) — are architecturally correct
and fully tested, but **no synthetic scenario differentiates them from the
simpler baseline**, and this was verified rather than assumed.

Purpose-built scenarios were run through the `time-ablation` and `decay-ablation`
harnesses (train the hybrid under each mode on one split, compare held-out Brier):

| scenario | what it isolates | verdict |
|---|---|---|
| `low_and_slow` | kill chain stretched 480× | `global` wins (0.221 vs 0.238) |
| `distractor` | benign noise on the campaign host | decay a near-tie (0.2364 vs 0.2366) |
| `stale_preamble` | identical benign preamble ~2 days before the recent window | `global` wins on time; decay a marginal win (0.2278 vs 0.2286) |
| `timing_only` | identical structure/content; **only** inter-event timing differs | `global` wins decisively (0.186 vs 0.210); per-entity/time-aware hurt |

The decisive case is `timing_only`: with graph structure held identical between
classes so that timing is the *sole* discriminator, every mode still separates
the classes at ROC/PR 1.0 — because the scalar `delta_seconds_log` feature
already encodes the inter-event gap. The per-entity clock and time-aware state
decay add variance without adding signal at this budget, and slightly worsen
calibration.

**Conclusion.** On synthetic data a single delta feature is sufficient; the
temporal-physics elaborations cannot be justified by the benchmark. They stay in
the codebase as opt-in modes (`time_mode`, `decay_half_life`) because they are
expected to matter on real streams where many entities carry genuinely
divergent, high-cardinality rhythms — a regime the simulator cannot manufacture.
**Validating them requires real labeled campaigns via the `--data`
(LabeledWindow) path, not more synthetic scenarios.**

Two consequences follow, recorded so the temporal machinery is not mistaken for
a demonstrated accuracy gain:

1. `decay_half_life` already defaults to `None` (off) on the shipped hybrid — the
   simplest setting, consistent with the evidence.
2. `time_mode` still defaults to `time_aware` (from W11). The evidence now favors
   defaulting to `global`, which is simpler and empirically better across every
   scenario measured; flipping it is the recommended follow-up, deferred here so
   this change stays a pure, low-risk benchmark-and-documentation addition.
