# IDR Intelligence Architecture

`idr-intelligence` is a sidecar research engine for `idr-main`. It consumes newline-delimited JSON serialized from `idr_common::IdrEvent`; it does not require a replacement event envelope.

## Flow

1. Validate and order `IdrEvent` records by timestamp.
2. Extract hosts, processes, executable hashes, IPs, prefixes, ASNs, domains, and devices.
3. Build one chronological feature sequence per entity.
4. Encode each sequence with a selective diagonal S6-style state-space layer.
5. Propagate entity states through a normalized relationship graph.
6. Emit campaign probability, ranked entities, a next-stage hypothesis, and source event IDs.

## Evidence boundary

Model output is advisory. Every finding carries `evidence_event_ids` and should be corroborated by the deterministic IDR correlator before automated response. Latent state is never a substitute for primary telemetry.

## Intended Rust integration

```rust
EventKind::IntelligenceFinding {
    campaign_id: String,
    escalation_probability: f32,       // affine-calibrated; see `calibration`
    raw_escalation_probability: f32,   // uncalibrated sigmoid
    calibration: String,               // "none" or "affine:scale=..,bias=.."
    predicted_next_stage: String,
    related_entities: Vec<String>,
    evidence_event_ids: Vec<Uuid>,
    model_version: String,
    engine_version: String,
    feature_schema_hash: String,
    scored_at: DateTime<Utc>,
    observed_attack_stages: Vec<AttackStage>, // {tactic, technique, kind_type, first_event_id}
}
```

`escalation_probability` is affine-calibrated (temperature scale plus bias, fit
on held-out validation NLL); `raw_escalation_probability` is the uncalibrated
sigmoid and `calibration` records the transform applied. `predicted_next_stage`
and `observed_attack_stages` come from a deterministic kind→ATT&CK table
(`attack.py`), never from the model — the field idr-sentinel corroborates stays
auditable.

`feature_schema_hash` identifies the exact feature semantics (names plus prior
tables) the scoring model was trained under; `idr-sentinel` can deterministically
reject findings whose hash disagrees with the deployed engine's manifest. The
same hash is embedded in every checkpoint, and `load_campaign_model` refuses a
checkpoint whose manifest disagrees with the runtime (`SchemaMismatchError`).

## Serving bridge (W17)

The ONNX inference adapter exists: `idr-intelligence export` writes the
streaming surface as `step.onnx` (the W13 `step()` cell), `head.onnx` (the
shared relational head), and `manifest.json` carrying every scoring constant —
prior tables, the ATT&CK mapping above, calibration, dimensions, and IO
signatures. `rust/idr-intelligence-rt` consumes the bundle on tract (pure
Rust) and reproduces `StreamingScorer` findings field-for-field; committed
golden streams pin the two implementations together, and the machine-local
`rust/idr-common-parity` crate proves the input wire format against the real
`idr_common` types from the idr-main tree.

The `EventKind::IntelligenceFinding` variant above remains the *proposed*
idr-main addition — idr-main is modified only on explicit instruction. Until
it lands, the bridge emits the finding JSON directly (the variant's exact
field shape) rather than wrapping it in an `IdrEvent` envelope.

The current repository is the reproducible development and ablation environment. Production deployment still requires real labeled campaigns, timestamp-versioned graph snapshots, and calibration fit on real data.
