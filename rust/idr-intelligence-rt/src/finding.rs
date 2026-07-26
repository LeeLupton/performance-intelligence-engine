//! The advisory finding, field-for-field compatible with Python's
//! `IntelligenceFinding.to_dict()` — including the W15 streaming exclusions
//! (`entity_evidence` empty) and the W16 defaults (`continues_campaign`,
//! `windows_observed`) for registry-less scoring.

use serde::Serialize;
use serde_json::{Map, Value};

use crate::attack::Stage;

#[derive(Debug, Clone, Serialize)]
pub struct Finding {
    pub campaign_id: String,
    pub escalation_probability: f64,
    pub raw_escalation_probability: f64,
    pub calibration: String,
    pub predicted_next_stage: String,
    pub observed_attack_stages: Vec<Stage>,
    pub related_entities: Vec<String>,
    pub entity_evidence: Vec<Value>,
    pub applied_suppressions: Vec<String>,
    pub evidence_event_ids: Vec<String>,
    pub model_version: String,
    pub graph_nodes: usize,
    pub graph_relations: Map<String, Value>,
    pub engine_version: String,
    pub feature_schema_hash: String,
    pub scored_at: String,
    pub feature_drift: Option<FeatureDrift>,
    pub continues_campaign: bool,
    pub windows_observed: u64,
}

/// Advisory PSI of scored event features against the training snapshot.
#[derive(Debug, Clone, Serialize)]
pub struct FeatureDrift {
    pub psi_max: f64,
    pub psi_mean: f64,
    pub flagged_features: Vec<String>,
}
