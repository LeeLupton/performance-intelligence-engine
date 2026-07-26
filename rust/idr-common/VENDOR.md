# Vendored: idr-common

This is a **verbatim copy** of the `idr-common` crate from Lee's `idr-main`
workspace (`crates/idr-common`) — the shared `idr_common::IdrEvent` types the
whole IDR platform serializes. It is vendored here so the wire-compatibility
proof (`../idr-common-parity`) and the serving bridge (`../idr-intelligence-rt`)
build against the real event schema **in CI**, not just on a machine that has
the idr-main checkout.

## Why only this crate

`idr-main` is a ~700MB+ mixed Rust/Python/TypeScript platform (BGP daemon, eBPF,
hardware sensors, sentinel correlator, a Next.js dashboard, and large data
blobs). This repo is a focused ML sidecar; it integrates with idr-main only
through the `IdrEvent` envelope, which lives entirely in `idr-common`. Vendoring
just this crate makes the integration real and testable without turning the ML
repo into a monorepo of the whole platform.

## Provenance & sync

- Source: `idr-main/crates/idr-common/src/{lib,events,alert,config,reputation}.rs`
- The workspace-inherited manifest fields (version `0.1.0`, edition 2024,
  license, and the serde/serde_json/chrono/uuid/thiserror deps) are pinned in
  `Cargo.toml` to idr-main's workspace values; its own `[workspace]` detaches it.
- The upstream is **not** the published `github.com/leelupton/idr` — that repo
  is behind this copy (it lacks `ExternalTriage`, `TriageClassification`, and
  the `BgpAnomaly` family). Re-copy from the local idr-main tree if idr-common
  changes there.

## License note (needs Lee's decision)

idr-common is **GPL-2.0-only**; the rest of this repo is **GPL-3.0-or-later**.
GPL-2.0-only is not compatible with GPLv3, so a downstream *combined
distribution* has a conflict to resolve. Both codebases are Lee's, so this is
his call — e.g. relicense idr-common to `GPL-2.0-or-later` (or GPL-3.0) upstream,
or keep the crates' licenses distinct and combine only at the wire (JSON) level,
which is how the bridge actually couples to it. The original license is
preserved here, unchanged, pending that decision.
