//! Deterministic ATT&CK stage accumulation and next-stage prediction.
//!
//! A port of `attack.py` driven entirely by the manifest's tables, so the
//! auditable field idr-sentinel corroborates cannot drift from the Python
//! engine's mapping.

use serde::Serialize;

use crate::manifest::AttackSpec;

/// One observed attack stage, anchored to the first event that exhibited it.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Stage {
    pub tactic: String,
    pub technique: String,
    pub kind_type: String,
    pub first_event_id: String,
}

/// Record the stage for a kind the first time it is seen (dedup by kind).
pub fn observe(stages: &mut Vec<Stage>, spec: &AttackSpec, kind_type: &str, event_id: &str) {
    if stages.iter().any(|stage| stage.kind_type == kind_type) {
        return;
    }
    if let Some(mapping) = spec.kind_to_attack.get(kind_type) {
        stages.push(Stage {
            tactic: mapping.tactic.clone(),
            technique: mapping.technique.clone(),
            kind_type: kind_type.to_string(),
            first_event_id: event_id.to_string(),
        });
    }
}

/// Next unobserved kill-chain tactic after the furthest tactic in `stages`.
pub fn next_stage(stages: &[Stage], spec: &AttackSpec) -> String {
    let observed: Vec<usize> = stages
        .iter()
        .filter_map(|stage| {
            spec.tactic_order
                .iter()
                .position(|tactic| *tactic == stage.tactic)
        })
        .collect();
    let Some(&furthest) = observed.iter().max() else {
        return "unknown".to_string();
    };
    for index in furthest + 1..spec.tactic_order.len() {
        if !observed.contains(&index) {
            return spec.tactic_order[index].clone();
        }
    }
    "kill-chain-complete".to_string()
}
