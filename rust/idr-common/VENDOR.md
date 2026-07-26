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

## License (resolved)

This vendored crate is **GPL-2.0-or-later**, relicensed from idr-main's
original **GPL-2.0-only** by the owner (Lee, who holds copyright on both
codebases). GPL-2.0-or-later is compatible with the repo's
**GPL-3.0-or-later**, so the combined-distribution conflict is resolved: a
downstream may take the whole work under GPLv3.

This is a deliberate divergence from upstream `idr-main/crates/idr-common`,
which remains GPL-2.0-only at last sync. When re-copying source from idr-main,
keep this crate's `license = "GPL-2.0-or-later"` in `Cargo.toml` (the `.rs`
files carry no license headers, so they stay byte-identical). Relicensing
idr-main upstream to match is optional and Lee's call.
