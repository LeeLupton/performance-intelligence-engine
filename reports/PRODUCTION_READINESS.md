# Production Readiness Assessment

Status as of the W17 line (ONNX export + Rust serving bridge) plus operational
logging. This document separates two questions that are easy to conflate:

- **Software readiness** — is the code correct, tested, observable, and
  deployable as infrastructure?
- **Detector readiness** — is the *model* fit to make security decisions on
  real traffic?

The short version: **software readiness is high; detector readiness is
blocked, and the blocker is not code.** No amount of engineering closes it —
it requires real labeled data and a realized integration path. Shipping this
as a live detector today would mean acting on a model whose only evidence of
working is that it separates a simulator from itself.

## Verdict

| Axis | State | Blocker |
|---|---|---|
| Software readiness | **Strong** | packaging + a chosen runtime shape |
| Detector readiness | **Not ready** | real labeled campaigns; real-data calibration; a defensible operating threshold |
| Integration | **Designed, not realized** | findings have no runtime path into `idr-sentinel`; the `EventKind` variant is proposed, not merged |

## Software readiness — what is genuinely done

These are verified in-repo, not aspirational:

- **Correctness under test** — 118 Python tests + 15 Rust tests + 3 machine-local
  wire tests; a frozen benchmark manifest fails CI on regression; gradient
  checks through every architecture arm.
- **Input validation** — strict tagged-envelope validation (`schema.py`), and
  the Rust bridge mirrors it (including a fromisoformat-grammar timestamp
  parser pinned by a fuzzed acceptance battery).
- **Provenance & safety rails** — feature-schema hash embedded in every
  checkpoint and export bundle; a stale bundle refuses to load rather than
  mis-score; model output is advisory and cannot trigger `PanicResponse`.
- **Determinism & reproducibility** — seeded throughout; ONNX export is
  bit-reproducible; cross-language parity pinned by committed golden streams.
- **Bounded resource use** — audited LRU node budget; both stream CLIs default
  to `--max-nodes 4096` so untrusted input cannot blow up the dense adjacency.
- **Drift instrumentation** — per-feature PSI on every finding.
- **Operational logging** *(new)* — opt-in JSON-lines to stderr
  (`--log-level`): model provenance on load, per-run counts/timing, finding
  summaries, drift flags, evictions, and rejections. Off by default; never
  touches the stdout finding contract.
- **CI & toolchain hygiene** — pinned ruff rule set and pinned Rust toolchain
  (`rust-toolchain.toml`) so distro bumps cannot silently change what CI
  enforces; adversarially reviewed (65-agent workflow, 19 findings fixed).

### Software gaps that ARE code work (closeable here)

1. **No runtime.** The CLI is one-shot: read a file, score, exit. There is no
   continuous consumer of `idr-main`'s live stream. The *shape* of that runtime
   — stdin/stdout daemon, file tailer, socket service, or a library call from
   inside `idr-sentinel` — is an architectural decision that depends on how
   `idr-main` emits events, so it is deliberately not guessed here.
2. **No deployment packaging.** No Dockerfile / reproducible artifact build /
   pinned wheel. Low-risk to add once (1) is decided.
3. **No model card.** For an advisory security model, a short card (intended
   use, training data, metrics, known failure modes, non-goals) should ship
   with each checkpoint.

## Detector readiness — the wall

This is the load-bearing section. Every accuracy number in this repo comes
from a **synthetic simulator** the engine's own author wrote. That is the
correct way to prove the *software and evaluation workflow* function — and it
is explicitly not evidence that the model detects real campaigns
(`reports/AUDIT.md` says so, and the README repeats it). The following cannot
be closed by writing code:

1. **No real training data.** The model has never seen a real campaign. Weights
   fit to the simulator encode the simulator's structure, not an adversary's.
2. **Calibration is synthetic.** `escalation_probability` is affine-calibrated
   on held-out *synthetic* NLL. On real traffic with a real base rate those
   probabilities are meaningless until re-fit — the calibration machinery is
   correct, but it has nothing real to calibrate against.
3. **No operating threshold.** There is no real ROC/PR curve, so there is no
   defensible "alert if p > τ" cutoff. Picking one on synthetic data would be a
   fabricated SLA.
4. **Temporal-physics modes unvalidated** *(documented, `AUDIT.md`)* — shipped
   off by default precisely because the simulator cannot manufacture the
   high-cardinality real rhythms they target.

The tooling to *consume* real data already exists (`LabeledWindow` via
`--data`, rolling-origin CV), so the gap is data availability plus a
go-live validation gate, not missing ingestion code.

## Integration — designed, not realized

`docs/ARCHITECTURE.md` sketches the sidecar: consume `idr_common::IdrEvent`
NDJSON, emit a finding for deterministic corroboration by `idr-sentinel`.
Today the finding is printed to stdout and **nothing consumes it**. Two things
are missing: a runtime that routes findings to `idr-sentinel`, and the
`EventKind::IntelligenceFinding` envelope in `idr-main` (proposed in
`ARCHITECTURE.md`; `idr-main` is modify-only-on-instruction, so it awaits an
explicit go-ahead).

## Staged path to go-live

Ordered by dependency. Owners noted where the work is not this repo's.

1. **[repo] Deployment-agnostic hardening — ✅ MOSTLY BUILT.** Rust-side
   structured logging now matches Python (`--log-level`/`IDR_RT_LOG`); a
   `Makefile` runs every CI gate locally (`make gates`); a Python engine
   `Dockerfile` and a minimal multi-stage Rust serving `Dockerfile` package
   both components (CPU-only, non-root, pinned bases). Model cards are produced
   by the validation gate (step 2). *Remaining: baking a provenance card into
   `save_checkpoint`, and CI build-caching — minor.*
2. **[repo] Real-data validation gate — ✅ BUILT** (`idr-intelligence validate`,
   `validation.py`). Takes real `LabeledWindow` data, re-fits calibration on an
   earlier time segment, selects an operating threshold at a target FPR, snapshots
   a real drift baseline, verifies metrics + the *realized* FPR on a strictly-later
   untouched segment, and turns configurable gates into a go/no-go verdict, a model
   card, and (only on `go`) a recalibrated checkpoint. Exits non-zero on no-go so a
   deploy pipeline halts. Honesty is structural: the operator must attest data
   provenance, and a verdict is only `binding` when that attestation names real
   data — running the gate on the simulator produces a full report whose verdict is
   explicitly a non-binding wiring check. **This converts "needs real data" into a
   one-command gate: drop in real labeled campaigns and it renders a defensible
   verdict.**
3. **[Lee + org] Supply real labeled campaigns.** The wall. Nothing downstream
   is defensible without it.
4. **[Lee] Choose the runtime shape** and build it (this repo), wiring findings
   toward `idr-sentinel`.
5. **[Lee] Merge `EventKind::IntelligenceFinding`** into `idr-main` (explicit
   instruction required), or keep the bare-JSON contract.
6. **[org] Shadow deployment** — run advisory-only against live traffic beside
   the deterministic correlator; compare, tune τ, watch drift, *then* consider
   any automated action (still gated by the advisory boundary).

## Bottom line

This is production-grade *engineering* wrapped around a *research-grade*
model. It is ready to be deployed **in shadow, advisory-only, once a runtime
exists** — and it is not ready to make decisions until it has been trained and
validated on real campaigns. The most valuable code work remaining (steps 1–2)
makes the go-live path a one-command gate; the decisive step (3) is data, not
code.
