# W17: ONNX export + Rust serving bridge — roadmap complete (17/17)

Two commits: the W17 slice (`f40637a`) and the adversarial-review fix slice (`c619db9`).

## What this ships

**Python export surface** — `idr-intelligence export` writes the streaming model as an ONNX bundle: `step.onnx` (the W13 `step()` cell; fixed shapes, recurrence driven by the consumer — no ONNX Scan/Loop), `head.onnx` (the shared relational head; symbolic node axis), and `manifest.json` carrying every scoring constant (prior tables, ATT&CK mapping, calibration, dimensions, IO signatures) so consumers hardcode nothing. `export.OnnxStreamScorer` is the torch-free reference runner — the executable spec of the serving contract — test-pinned to `StreamingScorer` across time modes, budget eviction, suppressions, and drift, and it verifies the bundle's `feature_schema_hash` against the host runtime before scoring.

**Rust bridge** — `rust/idr-intelligence-rt` consumes the bundle on tract (pure Rust; no Python, no ONNX Runtime library): envelope validation matching `schema.py` (including µs timestamp truncation — idr-main serializes `Utc::now()` at ns precision), the full entity/edge/feature extraction port, the streaming state machine with budget LRU + audited evictions, suppressions, and finding assembly matching `IntelligenceFinding` field-for-field. CLI mirrors `idr-intelligence stream`.

**Cross-language parity, pinned** — committed golden streams (content-addressed ids, salt-searched so top-k rankings are tie-free with a 5e-5 gap floor ≈ 500× measured cross-runtime drift), a no-GNN ablation bundle exercising the conditional head signature, per-event feature vectors including 9 hand-authored edge cases covering every extraction branch, and **freshness gates**: pytest replays all committed goldens against the current engine, so semantic drift without fixture regeneration fails CI.

**idr-main integration, proven without touching idr-main** — the machine-local `rust/idr-common-parity` crate (path-dep on the idr-main working tree; not in CI) proves events serialized by the real `idr_common::IdrEvent` — including `TriageClassification` and the `BgpAnomaly` family — parse and score end-to-end, and that the goldens are valid idr-main wire format. The `EventKind::IntelligenceFinding` variant remains the *proposed* idr-main change (`docs/ARCHITECTURE.md`); until it lands the bridge emits finding JSON in exactly that field shape. Note: the published `leelupton/idr` GitHub repo is behind your local tree (it lacks `ExternalTriage`/`TriageClassification`/`BgpAnomaly`), which is why this is a path-dep, not a git-dep.

**Model changes (both behavior-preserving; benchmark floors green)** — `GatedAttentionPool` and `_masked_max` no longer export the `IsInf` op tract can't evaluate (ADR-30), and ranking uses a stable argsort with a defined first-seen tie-break in both languages (ADR-31: exact ties are data-inherent — isomorphic entities score identically under any weights — so tie order must be *defined* to be reproducible). `CampaignModel` now rejects non-positive `decay_half_life`.

## Review

A 65-agent adversarial workflow (5 lenses → 30 raw findings, each challenged by 2 refuters → 19 confirmed) ran against the W17 commit; `c619db9` fixes all confirmed findings or defers them with documentation (per-event entity caps stay parity-with-Python with the trust boundary documented in the crate README; rolling Arch rust toolchain in CI is accepted, same class as the earlier ruff pin — revisit on first breakage).

## Gates

- 114 Python tests, ruff, `python -m mypy src`, benchmark floors: green
- `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, 12 bridge tests: green
- machine-local idr-common wire tests: 3/3
- CI extended: Arch container gains `rust` + pip `onnx`/`onnxruntime`; cargo fmt/clippy/test gate the bridge crate

State record: ADR-30/31, W17 completed — **all 17 roadmap workstreams are done.**

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Session: https://claude.ai/code/session_01C4oEsobmYyfr2gF5ankaNY
