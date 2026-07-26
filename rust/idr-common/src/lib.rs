//! IDR Common - Shared types, event definitions, and cross-layer protocols
//!
//! This crate defines the canonical event types that flow through the
//! Triple-Check detection pipeline: Kernel → Network → Hardware.

pub mod events;
pub mod alert;
pub mod config;
pub mod reputation;
