//! The streaming scorer: a semantics-preserving port of `streaming.py` (and of
//! `export.OnnxStreamScorer`, its torch-free executable spec) over tract.

use std::collections::HashMap;
use std::path::Path;

use chrono::{DateTime, Utc};
use serde_json::{Map, Number, Value};

use crate::attack::{Stage, next_stage, observe};
use crate::event::RawEvent;
use crate::features::{DELTA_FEATURE_INDEX, Projection, normalize_delta, project_event};
use crate::finding::{FeatureDrift, Finding};
use crate::manifest::Manifest;
use crate::model::{HeadSession, StepSession};

/// One entity's carried scan state — everything streaming keeps per node.
struct EntityState {
    state: Vec<f32>,
    output: Vec<f32>,
    last_seen: DateTime<Utc>,
    evidence_ids: Vec<String>,
}

/// One entity dropped to stay within the node budget.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EvictionRecord {
    pub entity: String,
    pub last_seen: String,
    pub reason: String,
}

pub struct StreamingScorer {
    pub manifest: Manifest,
    step: StepSession,
    head: HeadSession,
    max_nodes: Option<usize>,
    order: Vec<String>,
    entities: HashMap<String, EntityState>,
    pub evictions: Vec<EvictionRecord>,
    relation_counts: Vec<(String, u64)>,
    edge_last_seen: HashMap<(String, String), DateTime<Utc>>,
    stages: Vec<Stage>,
    previous_time: Option<DateTime<Utc>>,
    first_event: Option<(DateTime<Utc>, String)>,
    drift_counts: Option<Vec<Vec<u64>>>,
    pub events_seen: u64,
}

impl StreamingScorer {
    /// Open an export bundle directory (step.onnx + head.onnx + manifest.json).
    pub fn open(bundle_dir: &Path, max_nodes: Option<usize>) -> Result<Self, String> {
        let manifest = Manifest::load(bundle_dir)?;
        if let Some(limit) = max_nodes
            && limit < 1
        {
            return Err("max_nodes must be at least 1".to_string());
        }
        let has_delta = manifest
            .graphs
            .step
            .inputs
            .iter()
            .any(|name| name == "delta_t");
        let step = StepSession::load(
            &bundle_dir.join(&manifest.graphs.step.file),
            manifest.model.feature_dim,
            manifest.model.hidden_dim,
            manifest.model.state_dim,
            has_delta,
        )
        .map_err(|error| format!("cannot load step graph: {error}"))?;
        let has_adjacency = manifest
            .graphs
            .head
            .inputs
            .iter()
            .any(|name| name == "adjacency");
        let head = HeadSession::load(
            &bundle_dir.join(&manifest.graphs.head.file),
            manifest.model.hidden_dim,
            has_adjacency,
        )
        .map_err(|error| format!("cannot load head graph: {error}"))?;
        let drift_counts = manifest.feature_stats.as_ref().map(|stats| {
            vec![vec![0u64; stats.bin_edges.len().saturating_sub(1)]; manifest.model.feature_dim]
        });
        Ok(Self {
            manifest,
            step,
            head,
            max_nodes,
            order: Vec::new(),
            entities: HashMap::new(),
            evictions: Vec::new(),
            relation_counts: Vec::new(),
            edge_last_seen: HashMap::new(),
            stages: Vec::new(),
            previous_time: None,
            first_event: None,
            drift_counts,
            events_seen: 0,
        })
    }

    /// Advance every entity the event mentions by one step; update the graph.
    pub fn ingest(&mut self, event: &RawEvent) -> Result<(), String> {
        if let Some(previous) = self.previous_time
            && event.timestamp < previous
        {
            return Err(format!(
                "out-of-order event {}: {} precedes the stream clock {}; sort or buffer upstream",
                event.id,
                event.timestamp.to_rfc3339(),
                previous.to_rfc3339()
            ));
        }
        let global_delta = self
            .previous_time
            .map(|previous| (event.timestamp - previous).as_seconds_f64())
            .unwrap_or(0.0);
        let projection: Projection = project_event(event, global_delta, &self.manifest.features);
        self.previous_time = Some(event.timestamp);
        let key = (event.timestamp, event.id.clone());
        if self.first_event.as_ref().is_none_or(|first| key < *first) {
            self.first_event = Some(key);
        }
        observe(
            &mut self.stages,
            &self.manifest.attack,
            &event.kind_type(),
            &event.id,
        );
        let hidden = self.manifest.model.hidden_dim;
        let state_dim = self.manifest.model.state_dim;
        let per_entity_delta = self.manifest.model.time_mode != "global";
        for entity in &projection.entities {
            let gap_seconds = self
                .entities
                .get(entity)
                .map(|existing| (event.timestamp - existing.last_seen).as_seconds_f64())
                .unwrap_or(0.0);
            let entity_delta =
                normalize_delta(gap_seconds, self.manifest.features.delta_log_divisor) as f32;
            let mut features = projection.features.clone();
            if per_entity_delta {
                features[DELTA_FEATURE_INDEX] = entity_delta;
            }
            if !self.entities.contains_key(entity) {
                self.order.push(entity.clone());
                self.entities.insert(
                    entity.clone(),
                    EntityState {
                        state: vec![0.0; hidden * state_dim],
                        output: vec![0.0; hidden],
                        last_seen: event.timestamp,
                        evidence_ids: Vec::new(),
                    },
                );
            }
            let existing = self.entities.get_mut(entity).expect("entity just ensured");
            let (state, output) = self
                .step
                .run(&features, &existing.state, &existing.output, entity_delta)
                .map_err(|error| format!("step inference failed for {entity}: {error}"))?;
            existing.state = state;
            existing.output = output;
            existing.last_seen = event.timestamp;
            if !existing.evidence_ids.contains(&event.id) {
                existing.evidence_ids.push(event.id.clone());
                let limit = self.manifest.scoring.evidence_limit;
                if existing.evidence_ids.len() > limit {
                    let excess = existing.evidence_ids.len() - limit;
                    existing.evidence_ids.drain(..excess);
                }
            }
            if let (Some(counts), Some(stats)) = (
                self.drift_counts.as_mut(),
                self.manifest.feature_stats.as_ref(),
            ) {
                for (index, value) in features.iter().enumerate() {
                    if let Some(bin) = histogram_bin(f64::from(*value), &stats.bin_edges) {
                        counts[index][bin] += 1;
                    }
                }
            }
        }
        for (left, right, relation) in &projection.edges {
            let pair = if left <= right {
                (left.clone(), right.clone())
            } else {
                (right.clone(), left.clone())
            };
            self.edge_last_seen.insert(pair, event.timestamp);
            match self
                .relation_counts
                .iter_mut()
                .find(|(name, _)| name == relation)
            {
                Some((_, count)) => *count += 1,
                None => self.relation_counts.push((relation.clone(), 1)),
            }
        }
        self.events_seen += 1;
        self.enforce_budget();
        Ok(())
    }

    /// Evict least-recently-seen entities past the budget, with an audit trail.
    fn enforce_budget(&mut self) {
        let Some(limit) = self.max_nodes else { return };
        if self.entities.len() <= limit {
            return;
        }
        // Rank by (last_seen, entity) descending — GraphBudget.apply's tiebreak.
        let mut ranked: Vec<&String> = self.order.iter().collect();
        ranked.sort_by(|a, b| {
            let left = (self.entities[*a].last_seen, *a);
            let right = (self.entities[*b].last_seen, *b);
            right.cmp(&left)
        });
        let evicted: Vec<String> = ranked[limit..]
            .iter()
            .map(|entity| (*entity).clone())
            .collect();
        for entity in &evicted {
            let state = self.entities.remove(entity).expect("evicted entity exists");
            self.evictions.push(EvictionRecord {
                entity: entity.clone(),
                last_seen: isoformat(state.last_seen),
                reason: "node_budget".to_string(),
            });
        }
        self.order
            .retain(|entity| self.entities.contains_key(entity));
        self.edge_last_seen.retain(|(left, right), _| {
            self.entities.contains_key(left) && self.entities.contains_key(right)
        });
    }

    /// Score the carried state now and return a finding.
    pub fn finding(
        &mut self,
        top_k: Option<usize>,
        suppressions: &[String],
    ) -> Result<Finding, String> {
        if self.order.is_empty() {
            return Err("no events ingested".to_string());
        }
        let top_k = top_k.unwrap_or(self.manifest.scoring.top_k_default);
        let nodes = self.order.len();
        let index_of: HashMap<&String, usize> = self
            .order
            .iter()
            .enumerate()
            .map(|(index, id)| (id, index))
            .collect();
        let has_adjacency = self
            .manifest
            .graphs
            .head
            .inputs
            .iter()
            .any(|name| name == "adjacency");
        let mut adjacency = Vec::new();
        if has_adjacency {
            adjacency = vec![0.0f32; nodes * nodes];
            for index in 0..nodes {
                adjacency[index * nodes + index] = 1.0;
            }
            let now = self
                .previous_time
                .expect("previous_time set after first ingest");
            for ((left, right), seen) in &self.edge_last_seen {
                let weight = match self.manifest.model.decay_half_life {
                    Some(half_life) => {
                        let age = (now - *seen).as_seconds_f64();
                        0.5f64.powf(age / half_life) as f32
                    }
                    None => 1.0,
                };
                let (i, j) = (index_of[left], index_of[right]);
                adjacency[i * nodes + j] = weight;
                adjacency[j * nodes + i] = weight;
            }
            degree_normalize(&mut adjacency, nodes);
        }
        let mut outputs = Vec::with_capacity(nodes * self.manifest.model.hidden_dim);
        for entity in &self.order {
            outputs.extend_from_slice(&self.entities[entity].output);
        }
        let (graph_logit, node_logits) = self
            .head
            .run(&outputs, &adjacency, nodes)
            .map_err(|error| format!("head inference failed: {error}"))?;
        let calibration = &self.manifest.calibration;
        let raw_probability = sigmoid(f64::from(graph_logit));
        let probability =
            sigmoid(f64::from(graph_logit) / calibration.temperature.max(1e-3) + calibration.bias);
        // Suppressions attenuate matching entities out of the ranking; the
        // finding itself and the campaign probability are untouched.
        let mut ranking: Vec<f64> = node_logits
            .iter()
            .map(|logit| sigmoid(f64::from(*logit)))
            .collect();
        let mut applied_suppressions = Vec::new();
        for (index, entity) in self.order.iter().enumerate() {
            let matched = suppressions.iter().any(|rule| {
                entity == rule || (rule.ends_with(':') && entity.starts_with(rule.as_str()))
            });
            if matched {
                ranking[index] = f64::NEG_INFINITY;
                applied_suppressions.push(entity.clone());
            }
        }
        let ranked = ranked_indices(&ranking, top_k);
        let related: Vec<String> = ranked
            .iter()
            .map(|index| self.order[*index].clone())
            .collect();
        let mut evidence = Vec::new();
        for index in &ranked {
            for event_id in &self.entities[&self.order[*index]].evidence_ids {
                if !evidence.contains(event_id) {
                    evidence.push(event_id.clone());
                }
            }
        }
        let first_event = self
            .first_event
            .as_ref()
            .expect("first_event set after first ingest");
        let campaign_prefix: String = first_event.1.chars().take(8).collect();
        let mut graph_relations = Map::new();
        for (relation, count) in &self.relation_counts {
            graph_relations.insert(relation.clone(), Value::Number(Number::from(*count)));
        }
        Ok(Finding {
            campaign_id: format!("idr-campaign-{campaign_prefix}"),
            escalation_probability: round6(probability),
            raw_escalation_probability: round6(raw_probability),
            calibration: calibration.label.clone(),
            predicted_next_stage: next_stage(&self.stages, &self.manifest.attack),
            observed_attack_stages: self.stages.clone(),
            related_entities: related,
            entity_evidence: Vec::new(),
            applied_suppressions,
            evidence_event_ids: evidence,
            model_version: self.manifest.model_version.clone(),
            graph_nodes: nodes,
            graph_relations,
            engine_version: self.manifest.engine_version.clone(),
            feature_schema_hash: self.manifest.feature_schema_hash.clone(),
            scored_at: isoformat(Utc::now()),
            feature_drift: self.feature_drift(),
            continues_campaign: false,
            windows_observed: 1,
        })
    }

    /// PSI of accumulated feature counts against the training snapshot.
    fn feature_drift(&self) -> Option<FeatureDrift> {
        let stats = self.manifest.feature_stats.as_ref()?;
        let counts = self.drift_counts.as_ref()?;
        let mut psi_values = Vec::with_capacity(stats.histograms.len());
        for (index, train_counts) in stats.histograms.iter().enumerate() {
            let train: Vec<f64> = train_counts.iter().map(|count| count + 1e-4).collect();
            let observed: Vec<f64> = counts[index]
                .iter()
                .map(|count| *count as f64 + 1e-4)
                .collect();
            let train_total: f64 = train.iter().sum();
            let observed_total: f64 = observed.iter().sum();
            let psi: f64 = train
                .iter()
                .zip(&observed)
                .map(|(t, o)| {
                    let (t, o) = (t / train_total, o / observed_total);
                    (o - t) * (o / t).ln()
                })
                .sum();
            psi_values.push(psi);
        }
        let psi_max = psi_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let psi_mean = psi_values.iter().sum::<f64>() / psi_values.len() as f64;
        let flagged = psi_values
            .iter()
            .enumerate()
            .filter(|(_, value)| **value >= 0.2)
            .map(|(index, _)| self.manifest.features.names[index].clone())
            .collect();
        Some(FeatureDrift {
            psi_max: round6(psi_max),
            psi_mean: round6(psi_mean),
            flagged_features: flagged,
        })
    }
}

/// Symmetric D^-1/2 A D^-1/2 normalization in f32, matching `graph.degree_normalize`.
fn degree_normalize(adjacency: &mut [f32], nodes: usize) {
    let mut inv_sqrt = vec![0.0f32; nodes];
    for (index, inv) in inv_sqrt.iter_mut().enumerate() {
        let degree: f32 = adjacency[index * nodes..(index + 1) * nodes].iter().sum();
        *inv = degree.max(1.0).powf(-0.5);
    }
    for i in 0..nodes {
        for j in 0..nodes {
            adjacency[i * nodes + j] *= inv_sqrt[i] * inv_sqrt[j];
        }
    }
}

/// np.histogram bin for one value: right-open bins, last bin closed.
fn histogram_bin(value: f64, edges: &[f64]) -> Option<usize> {
    let bins = edges.len().checked_sub(1)?;
    if bins == 0 || value < edges[0] || value > edges[bins] {
        return None;
    }
    if value == edges[bins] {
        return Some(bins - 1);
    }
    let mut bin = 0;
    while bin + 1 < bins && value >= edges[bin + 1] {
        bin += 1;
    }
    Some(bin)
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

/// Python round(x, 6): nearest multiple of 1e-6, ties to even.
fn round6(value: f64) -> f64 {
    (value * 1e6).round_ties_even() / 1e6
}

/// Python `datetime.isoformat()` shape: +00:00 offset, microseconds only when nonzero.
fn isoformat(timestamp: DateTime<Utc>) -> String {
    let precision = if timestamp.timestamp_subsec_micros() == 0 {
        chrono::SecondsFormat::Secs
    } else {
        chrono::SecondsFormat::Micros
    };
    timestamp.to_rfc3339_opts(precision, false)
}

/// Descending stable ranking over node probabilities: slice the top_k prefix,
/// then drop suppressed (-inf) entries — the exact order of operations
/// score_events/StreamingScorer use. Stability is the ADR-31 contract: exact
/// ties (structurally identical entities) rank in first-seen order.
pub(crate) fn ranked_indices(scores: &[f64], top_k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|a, b| {
        scores[*b]
            .partial_cmp(&scores[*a])
            .expect("ranking scores are never NaN")
    });
    order
        .into_iter()
        .take(top_k.min(scores.len()))
        .filter(|index| scores[*index].is_finite())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{histogram_bin, ranked_indices, round6};

    #[test]
    fn exact_ties_rank_in_first_seen_order() {
        // ADR-31: stable sort keeps equal scores in index (first-seen) order.
        assert_eq!(
            ranked_indices(&[0.5, 0.9, 0.5, 0.5, 0.7], 5),
            vec![1, 4, 0, 2, 3]
        );
    }

    #[test]
    fn suppressed_entries_are_sliced_then_filtered() {
        // -inf sorts last, enters the top_k slice only when finite entries run
        // out, and is filtered after slicing — matching np.argsort[:k] + isfinite.
        let scores = [f64::NEG_INFINITY, 0.4, 0.6];
        assert_eq!(ranked_indices(&scores, 3), vec![2, 1]);
        assert_eq!(ranked_indices(&scores, 1), vec![2]);
    }

    #[test]
    fn histogram_bins_match_numpy_edges() {
        let edges = [0.0, 0.5, 1.0];
        assert_eq!(histogram_bin(0.0, &edges), Some(0));
        assert_eq!(histogram_bin(0.5, &edges), Some(1)); // interior edge -> higher bin
        assert_eq!(histogram_bin(1.0, &edges), Some(1)); // last bin closed
        assert_eq!(histogram_bin(1.1, &edges), None);
        assert_eq!(histogram_bin(-0.1, &edges), None);
    }

    #[test]
    fn round6_is_ties_to_even() {
        assert_eq!(round6(0.1234565), 0.123456); // ties to even, like Python round()
        assert_eq!(round6(0.1234575), 0.123458);
        assert_eq!(round6(0.9999999), 1.0);
    }
}
