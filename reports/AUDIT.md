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
