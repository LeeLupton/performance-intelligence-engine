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
2. `time_mode` now defaults to `global` (changed from `time_aware`) — the
   simpler mode, and empirically better across every scenario measured.
   Per-entity and time-aware modes remain available explicitly for real-data
   deployments where entity-relative rhythms are expected to carry signal.

## Finding: ATT&CK mapping audit — tactic-consistency is not semantic correctness

The `KIND_TO_ATTACK` table (the auditable ATT&CK field `idr-sentinel`
corroborates and an analyst puts in intel products) was originally hand-typed at
W5. A later pass "grounded" it against the MITRE Enterprise STIX bundle, but that
check only verified **tactic-consistency** — that each technique exists and its
MITRE tactics include the assigned tactic. That is too weak: it passes a
technique that is tactically valid but describes the wrong *activity*.

Prompted by a user catching `physics_anomaly → T1495 (Firmware Corruption)` — a
TTL/RTT path-intercept signal mapped to a device-firmware-wipe technique — all 11
mappings were re-audited for **semantic** fit (a 21-agent adversarial workflow;
several agents fetched primary MITRE text). Corrections applied:

| kind | was | now | why |
|---|---|---|---|
| `physics_anomaly` | impact / T1495 Firmware Corruption | collection / **T1557** Adversary-in-the-Middle | a single-hop TTL/RTT intercept is on-path interception, not firmware corruption |
| `impossible_state` | impact / T1499 Endpoint DoS | **unmapped** | the sentinel's own confirmation verdict — no adversary technique means "my correlator fired"; the old mapping existed only to make next-stage say "kill-chain-complete" |
| `nvme_latency_anomaly` | exfiltration / T1041 Exfiltration Over C2 | collection / **T1005** Data from Local System | a disk-latency sensor observes local bulk reads, not a C2 egress channel |
| `octet_reversal_detected` | defense-evasion / T1027 Obfuscated Files | command-and-control / **T1001** Data Obfuscation | a DNS-PTR trick hides the C2 destination in transit; there is no file |
| `hsts_time_manipulation` | credential-access / T1557 AitM | defense-evasion / **T1553** Subvert Trust Controls | accepting an expired cert via a time rollback subverts a trust control; T1557 named only the downstream goal |
| `suspicious_beacon` | C2 / T1071 Application Layer Protocol | command-and-control / **T1102** Web Service | the discriminator is beaconing to a high-trust web service used as a C2 carrier |
| `socket_lineage` | execution / T1059 Command and Scripting Interpreter | command-and-control / **T1071** Application Layer Protocol | the event observes an outbound app-layer socket, not a script interpreter |

Kept after primary-source verification: `bgp_anomaly` (collection/T1557 —
parent-level AitM covers routing interception), `ntp_time_shift` and
`rtc_clock_divergence` (defense-evasion/T1562 Impair Defenses), `mac_flapping`
(collection/T1557.002 ARP Cache Poisoning — an exact fit).

Consequence: findings no longer claim an `execution` or `exfiltration` stage the
sensors don't actually observe; `predict_next_stage` shifts accordingly (e.g. a
full synthetic campaign now predicts `exfiltration`, not `impact`). The audit is
reproducible; the `validate_mapping_against_reference()` test still enforces
tactic-consistency, but semantic fit remains a human/committed-record judgment.
