# idr-intelligence Roadmap: From Demo-Grade Advisory to Deployable Intelligence Engine

**Synthesis of 24 judged proposals into 4 phases / 14 workstreams.** Ordering principle: first make the numbers and evidence the engine already emits *true* (cheap, high ROI), then build a benchmark that can *detect* improvement, then fix the temporal-graph physics, then ship stateful serving and the Rust bridge. The advisory-only boundary to idr-sentinel is treated as a hard invariant throughout — several workstreams strengthen it rather than route around it.

---

## Phase 1 — Truthful outputs (all effort-S; target: one sprint)

**Theme:** Everything `IntelligenceFinding` carries today is partly untrustworthy: the entity ranking is an untrained random projection, the probability is uncalibrated, the provenance is a filename, and the constants that produced a score live in five modules. Fix the product before improving the model.

### W1. Attention-gated pooling that trains the evidence ranking *(merges: attention pooling 7.5)*
Replace mean+max pooling in `CampaignModel.forward` (`models.py`) with gated MIL attention over active nodes; pre-softmax attention scores replace `node_head` as `ModelOutput.node_logits`. This fixes the two coupled defects at once: `node_head` currently receives **zero gradient** (`_fit` in `training.py` optimizes only `graph_logit`), so the top-5 entities and `evidence_event_ids` that `pipeline.score_events` hands idr-sentinel are ranked by noise. Add evidence-precision@5 to `_evaluate` (simulator ground truth makes this measurable). Keep `uniform_pool` as an ablation arm.

### W2. EngineConfig + ModelManifest: one config, full provenance *(merges: EngineConfig 7, manifest/schema-hash 7.5)*
These two proposals are one workstream — the config hash is a manifest field. Add `src/idr_intelligence/config.py` (frozen dataclasses + TOML loader + `config_hash`) absorbing the disagreeing `HIDDEN_DIM=24` (`training.py`) vs `hidden_dim=32` (`models.py`), the `max_steps` 24-vs-16 split (`graph.py` vs `make_dataset`), the top-5 cutoff (`pipeline.py`), and the `SEVERITY_WEIGHT`/`KIND_PRIOR` tables out of `schema.py`. Add `src/idr_intelligence/registry.py` with `ModelManifest` (feature-schema blake2s over `FEATURE_NAMES` + priors, config hash, git SHA, calibration state); `save_checkpoint` embeds it, `load_campaign_model` raises `SchemaMismatchError` on hash mismatch. Extend `IntelligenceFinding` with `feature_schema_hash`, `engine_version`, `scored_at`, and mirror them in the Rust `EventKind` sketch in `docs/ARCHITECTURE.md` so idr-sentinel can deterministically reject stale-model findings. Ship `configs/default.toml` reproducing current behavior (no-op verified by existing pytest).

### W3. Calibration layer + calibration metrics *(merges: temperature+ECE 6.5, temperature+ensemble 5.5, the calibration half of FP-management 5.5)*
One temperature-scaling implementation: after early stopping in `_fit`, fit scalar T on the validation split by NLL, store it in the checkpoint (via the W2 manifest), apply `sigmoid(logit/T)` in `_evaluate` and `pipeline.score_events`. Findings report raw + calibrated probability + calibration method/version. Extend `_evaluate` with log-loss, 15-bin ECE, max calibration error, and the reliability-bin table; CI asserts post-scaling ECE ≤ pre-scaling ECE. **Scope cut:** defer the K=3 deep-ensemble / `probability_std` half of the 5.5 proposal until real labeled data exists (Phase 2 W8) — on the saturated synthetic benchmark ensemble spread is meaningless, and it triples the Phase 4 ONNX export surface.

### W4. Honest operating-point evaluation *(from: class imbalance + recall@FPR 6)*
`make_dataset` in `training.py` hardcodes `label = int(index % 2 == 1)` — a 50/50 alternation that also leaks parity across split boundaries. Add `malicious_rate` (seeded RNG labels), `pos_weight` in `_fit`, and extend `_evaluate` with recall@FPR∈{0.01, 0.001}, precision@k, and per-scenario breakdowns. Prerequisite for every later claim of improvement.

### W5. ATT&CK-native next-stage mapping *(from: MITRE mapping 5.5)*
Replace the four if-statements in `pipeline._next_stage` with a deterministic kind→technique/tactic table module; order observed tactics by timestamp (events are already sorted in `graph.py`), emit `observed_attack_stages`, and predict the next unobserved kill-chain tactic. Keeps the rule-based-and-auditable property while fixing the order-blindness (`nvme_latency_anomaly` first ≠ exfil next) and the 8-of-12 kinds falling through to `"unknown"`.

**Phase 1 done means:** entity ranking is trained and measured (evidence-precision@5 in the report); every finding carries calibrated probability, schema hash, config hash, and ATT&CK-vocabulary stage; a checkpoint refuses to load against changed feature semantics; all constants live in one TOML.

---

## Phase 2 — A benchmark that can say no (target: 2–3 sprints)

**Theme:** Every class saturates ROC/PR at n=80, so no modeling change in later phases can be validated today. Build the data and evaluation machinery first; it gates everything in Phases 3–4.

### W6. Graded + adversarial simulator scenarios *(merges: graded-difficulty 5.5, evasion-scenario third of adversarial-robustness 5)*
Parameterize `simulate_campaign(label, seed, scenario, difficulty)` in `simulator.py`: convergence knob p, distractor interleaving on the same `primary_host`, `legit_update` hard negatives, variable-length truncation with `stage_reached`, overlapping campaigns — plus the three evasion families (spaced-out low-and-slow, split-host sharing only prefix/ASN/domain, per-stage hash rotation). Keep the current script as `scenario='v0_easy'`. Emit per-event ground-truth stage labels in metadata. Per-scenario metrics (W4) make robustness regressions visible in CI.

### W7. Versioned benchmark suite + drift snapshot *(from: benchmark suite 4.5)*
`benchmarks/` directory of frozen scenario manifests with per-variant metric floors, a generator command, and a pytest-marked runner that fails CI on floor violations. Store per-feature training stats in the checkpoint; `cli.py score` computes PSI/chi-square drift against them and attaches an advisory drift block — load-bearing because `delta_seconds_log`'s distribution will shift drastically between simulator cadence and real traffic.

### W8. Labeled real-campaign ingestion *(from: LabeledWindow 6.5)*
`LabeledWindow` schema next to `IdrEvent` in `schema.py`, a `*.labeled.ndjson` directory loader reusing `IdrEvent.from_dict` and `build_temporal_graph`, and a `--data` flag on `train_ablation` that swaps the simulator for real windows with zero model changes. This is AUDIT.md's stated blocker for any production-accuracy claim, and it unlocks the deferred ensemble-uncertainty work from W3.

### W9. Statistical rigor: rolling-origin CV + seed replicates *(from: rolling-origin 4)*
Expanding-window folds × seed replicates in `train_ablation`; `best_model` declared only when the winner beats the runner-up beyond paired per-fold std, else "tie". With 16 test samples and one seed, the current ablation winner is noise — this must land before Phase 3 ablations mean anything.

### W10. Identity entities in the vocabulary *(from: user/session entities 5.5)*
Extend `extract_entities`/`_derive_edges` in `features.py` with `user:`, `session:{host}:{sid}`, optional `cloud:` nodes and `authenticates_to`/`owns`/`spawns` edges, plus `has_user`/`cross_host_user` features (`FEATURE_DIM` is derived, checkpoints carry dims, and W2's schema hash catches mismatches automatically). Pair with a W6 lateral-movement scenario where campaigns share only the user entity — otherwise the strongest real-world pivot key stays both invisible and untested.

**Phase 2 done means:** the ablation can rank architectures by ROC/PR/recall@FPR with uncertainty bars, not just Brier; hard-negative and evasion regressions fail CI; a real labeled window trains end-to-end via `--data`.

---

## Phase 3 — Temporal-graph physics (target: 2–3 sprints)

**Theme:** Fix the two structural lies in the current representation — the global clock and the immortal binary adjacency — now measurable against Phase 2's evasion scenarios.

### W11. Per-entity time + time-aware S6 discretization *(merges: per-entity deltas 5.5, per-entity-delta third of adversarial-robustness 5)*
Move delta computation from the global gap in `build_temporal_graph` (`graph.py` lines 36–39) to a per-node last-seen clock written into each history row; make discretization in `SelectiveSSM` explicitly time-aware (`delta_t = softplus(delta(u) + w·log1p(dt))`) so state decay tracks real elapsed entity time. Thread the delta tensor through `Batch`, `CampaignModel.forward`, and `score_events`. Ablate global vs per-entity vs time-aware on the W6 low-and-slow scenario — the one place this is directly measurable.

### W12. Bounded, decayed, sparse entity graph *(merges: bounded graph 5.5, time-decayed edges 5, decayed-adjacency third of adversarial-robustness 5)*
One implementation: `src/idr_intelligence/bounded_graph.py` with `GraphBudget`, half-life edge decay, `deque(maxlen)` histories, LRU eviction with audited `EvictionRecord`s, and sparse k-hop `snapshot()`; `ResidualGraphLayer` switches `torch.bmm` → `torch.sparse.mm`; `build_temporal_graph` becomes a thin replay wrapper so batch and streaming share one graph. The standalone tau-decay proposal is subsumed — ship its ablation (tau ∈ {inf, 1h, 15m}) as the acceptance test, using the W6 benign-preamble scenario so decay has something to suppress.

### W13. S6 step() cell + scan throughput *(from: ONNX prerequisite 6, scan throughput 4.5)*
Refactor `SelectiveSSM` to expose `step(u_t, state)` with `forward` reduced to a loop over it (one code path for training, streaming, and export), then hoist the per-step projections out of the loop (stage 1 of the throughput proposal, ~5–10x on CPU). **Defer** the associative-scan and `torch.compile` stage until profiling on real window sizes shows the hoisted loop is the bottleneck — at hidden_dim 24 it likely is not. Numerical-equivalence pytest keeps the readable loop as the oracle.

### W14. Structured evidence + analyst suppression loop *(merges: structured per-entity evidence 5.5, suppression half of FP-management 5.5)*
Replace the flat top-5 flatten in `score_events` (`pipeline.py:44–46`) with per-entity records: `{entity_id, node_probability (= W1 attention), evidence_event_ids from graph.evidence_ids, typed edges to other ranked entities (retain the per-pair relation labels graph.py currently discards), top-3 occlusion features named via FEATURE_NAMES, ATT&CK techniques from W5}`. Keep flat `evidence_event_ids` for Rust back-compat. Add the per-deployment allowlist file (`ip:...`, `(kind_type, entity)` pairs) that attenuates priors and excludes entities from ranking — never hides the finding from idr-sentinel — with `applied_suppressions` in the finding for audit.

**Phase 3 done means:** the low-and-slow and split-host evasion scenarios show measurable (CI-tracked) robustness gains over the Phase 2 baseline; graph memory is bounded with an eviction audit trail; a finding tells an analyst *which events implicate which entity and why*.

---

## Phase 4 — Stateful serving and the Rust boundary (target: 3–4 sprints)

**Theme:** Everything above still scores batches. Make the engine live against a continuous idr-main stream and deployable outside Python.

### W15. Streaming incremental scorer *(from: streaming 6; depends on W12 + W13)*
`src/idr_intelligence/streaming.py` with `EntityState` (SSM state, last timestamp, evidence deque) and `StreamingScorer.ingest/snapshot/restore`; per-event scoring advances only touched entities' `step()` states and runs the GNN over the k-hop `BoundedEntityGraph.snapshot()`. Snapshot/restore is what idr-sentinel integration actually requires for process restarts.

### W16. Stable campaign identity registry *(from: fingerprint matching 6; can start any time after Phase 1 — it is deterministic and independent)*
Replace `campaign_id = f"idr-campaign-{first.id[:8]}"` (`pipeline.py:48`) with a small registry: weighted-Jaccard matching on durable entity fingerprints (hash/domain weighted above host, `process:` excluded), content-addressed IDs, and continuity metadata (`continues_campaign`, `windows_observed`, first/last seen). Deterministic set-matching — auditable, and lets idr-sentinel accumulate corroboration for one hypothesis across windows.

### W17. ONNX export bundle + Rust `idr-intelligence-rt` + parity CI *(from: ONNX bridge 6; unblocked by W13)*
`src/idr_intelligence/export.py` emitting `s6_cell.onnx` / `gnn.onnx` / `heads.onnx` plus `features.json` (FEATURE_NAMES + prior tables so Rust reproduces `project_event` bit-for-bit) and the W2 manifest; Rust crate on `ort` owning the recurrence loop; `tests/test_onnx_parity.py` asserting 1e-5 agreement in existing CI. Note: W1's attention pooling changes `heads.onnx` and W3's temperature is a scalar in the manifest — sequencing after Phases 1–3 avoids exporting a moving target.

**Phase 4 done means:** an event stream can be scored incrementally with persistent state surviving restart; the same campaign keeps one ID across days; a golden NDJSON fixture produces identical findings through torch and the Rust runtime, enforced by CI.

---

## Three highest-leverage items

1. **W1 — Attention pooling fixing the untrained `node_head`.** The engine's entire value proposition is pointing idr-sentinel at the right entities, and that ranking is currently a random projection with zero gradient flow. One small change makes the core product real *and* adds a metric proving it.
2. **W2 — EngineConfig + ModelManifest provenance.** Without the feature-schema hash, any edit to `features.py` or the prior tables silently corrupts every deployed checkpoint at the same `feature_dim=20`. This is the cheapest way to make the sentinel boundary enforceable rather than aspirational, and every later workstream (calibration state, drift snapshots, ONNX manifests) hangs fields on it.
3. **W6+W4 — Simulator hardening plus operating-point metrics.** Nothing else on this roadmap can be *validated* until the benchmark stops saturating. This is the gate on Phases 3–4: without it, per-entity deltas, decay, and typed anything are unfalsifiable.

## Explicit rejections

- **REJECT: Learned next-stage head (score 3).** It replaces an auditable deterministic rule with a model trained on the simulator's own hardcoded kill-chain order — circular supervision that cannot generalize, and a downgrade in explainability at the exact field idr-sentinel corroborates. The W5 ATT&CK mapping delivers order-awareness and interoperability deterministically. Its one good sub-idea (truncated-prefix samples restore benchmark headroom) is absorbed into W6's variable-length scenarios.
- **REJECT (in proposed form): Relation-typed R-GCN message passing (score 3.5).** A dense `(R=8, N, N)` stacked adjacency multiplies graph memory 8x and directly contradicts W12's bounded sparse direction; and until Phase 2 lands, the saturated benchmark cannot show a `typed_gnn` variant winning, so it would ship unfalsified complexity. Revisit *only* as a sparse per-relation edge-index variant after Phase 3, gated on the split-host evasion scenario demonstrating that the homogeneous GNN actually fails — the typed-edge data is preserved either way by W14's structured evidence.
- **TRIM: K=3 deep ensemble / `probability_std`** (half of the 5.5 temperature+ensemble proposal): deferred, not rejected — meaningless on saturated synthetic data and a 3x export burden; reconsider once W8 provides real labeled windows where OOD spread has something to measure.
