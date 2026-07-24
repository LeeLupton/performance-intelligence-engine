//! The export-bundle manifest: the complete cross-language serving contract.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

pub const EXPORT_FORMAT: &str = "idr-intelligence-onnx-v1";

#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub engine_version: String,
    pub model_version: String,
    pub feature_schema_hash: String,
    pub model: ModelSpec,
    pub calibration: Calibration,
    pub features: FeatureSpec,
    pub attack: AttackSpec,
    pub scoring: ScoringSpec,
    #[serde(default)]
    pub feature_stats: Option<FeatureStats>,
    pub graphs: Graphs,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelSpec {
    pub feature_dim: usize,
    pub hidden_dim: usize,
    pub state_dim: usize,
    pub time_mode: String,
    pub decay_half_life: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Calibration {
    pub temperature: f64,
    pub bias: f64,
    pub label: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FeatureSpec {
    pub names: Vec<String>,
    pub severity_weight: HashMap<String, f64>,
    pub severity_default: f64,
    pub kind_prior: HashMap<String, f64>,
    pub kind_prior_default: f64,
    pub delta_log_divisor: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AttackSpec {
    pub tactic_order: Vec<String>,
    pub kind_to_attack: HashMap<String, TacticTechnique>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TacticTechnique {
    pub tactic: String,
    pub technique: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ScoringSpec {
    pub top_k_default: usize,
    pub evidence_limit: usize,
}

/// Training-time feature histogram snapshot for PSI drift.
#[derive(Debug, Clone, Deserialize)]
pub struct FeatureStats {
    pub bin_edges: Vec<f64>,
    pub histograms: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Graphs {
    pub step: GraphSpec,
    pub head: GraphSpec,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GraphSpec {
    pub file: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

impl Manifest {
    pub fn load(bundle_dir: &Path) -> Result<Self, String> {
        let path = bundle_dir.join("manifest.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        let manifest: Manifest = serde_json::from_str(&text)
            .map_err(|error| format!("cannot parse {}: {error}", path.display()))?;
        if manifest.format != EXPORT_FORMAT {
            return Err(format!(
                "unrecognized export bundle format: {:?}",
                manifest.format
            ));
        }
        Ok(manifest)
    }
}
