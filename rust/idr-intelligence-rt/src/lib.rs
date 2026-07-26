//! Streaming campaign scoring over an idr-intelligence ONNX export bundle.
//!
//! This crate is the W17 serving bridge: it consumes newline-delimited JSON in
//! the wire format serialized by `idr_common::IdrEvent`, carries per-entity S6
//! state through the exported `step.onnx` cell, and scores the entity graph
//! with the exported `head.onnx` relational head — reproducing the Python
//! `StreamingScorer` finding field-for-field. Every constant it needs (prior
//! tables, ATT&CK mapping, calibration, dimensions) is read from the bundle's
//! `manifest.json`; nothing is hardcoded twice.
//!
//! Deliberate scope, mirroring the Python streaming path: `entity_evidence`
//! is always empty (occlusion attribution needs history replay) and campaign
//! identity uses the first-event fallback id — the cross-window
//! `CampaignRegistry` stays on the Python side.
//!
//! Findings are advisory. The evidence boundary from `docs/ARCHITECTURE.md`
//! applies unchanged: output must be corroborated by the deterministic IDR
//! correlator and must never trigger an automated response on its own.

pub mod attack;
pub mod event;
pub mod features;
pub mod finding;
pub mod log;
pub mod manifest;
pub mod model;
pub mod scorer;

pub use event::RawEvent;
pub use finding::Finding;
pub use manifest::Manifest;
pub use scorer::{EvictionRecord, StreamingScorer};
