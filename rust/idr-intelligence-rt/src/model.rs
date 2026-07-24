//! tract-onnx sessions for the two exported graphs.
//!
//! The step cell has fixed shapes and is optimized once. The relational head
//! has a symbolic node axis; a runnable plan is concretized (and cached) per
//! observed node count — the graph is small, so re-optimizing is milliseconds.

use std::collections::HashMap;
use std::path::Path;

use std::sync::Arc;

use tract_onnx::prelude::*;

pub struct StepSession {
    plan: Arc<TypedRunnableModel>,
    pub has_delta: bool,
    hidden: usize,
    state_dim: usize,
    feature_dim: usize,
}

impl StepSession {
    pub fn load(
        path: &Path,
        feature_dim: usize,
        hidden: usize,
        state_dim: usize,
        has_delta: bool,
    ) -> TractResult<Self> {
        let mut model = tract_onnx::onnx()
            .model_for_path(path)?
            .with_input_fact(0, f32::fact([1, feature_dim]).into())?
            .with_input_fact(1, f32::fact([1, hidden, state_dim]).into())?
            .with_input_fact(2, f32::fact([1, hidden]).into())?;
        if has_delta {
            model = model.with_input_fact(3, f32::fact([1]).into())?;
        }
        Ok(Self {
            plan: model.into_optimized()?.into_runnable()?,
            has_delta,
            hidden,
            state_dim,
            feature_dim,
        })
    }

    /// Advance one entity by one event: (state, output) -> (state', output').
    pub fn run(
        &self,
        features: &[f32],
        state: &[f32],
        output: &[f32],
        delta: f32,
    ) -> TractResult<(Vec<f32>, Vec<f32>)> {
        let mut inputs: TVec<TValue> = tvec!(
            Tensor::from_shape(&[1, self.feature_dim], features)?.into(),
            Tensor::from_shape(&[1, self.hidden, self.state_dim], state)?.into(),
            Tensor::from_shape(&[1, self.hidden], output)?.into(),
        );
        if self.has_delta {
            inputs.push(Tensor::from_shape(&[1], &[delta])?.into());
        }
        let result = self.plan.run(inputs)?;
        Ok((
            result[0].view().as_slice::<f32>()?.to_vec(),
            result[1].view().as_slice::<f32>()?.to_vec(),
        ))
    }
}

pub struct HeadSession {
    proto: tract_onnx::prelude::InferenceModel,
    hidden: usize,
    cache: HashMap<usize, Arc<TypedRunnableModel>>,
}

impl HeadSession {
    pub fn load(path: &Path, hidden: usize) -> TractResult<Self> {
        Ok(Self {
            proto: tract_onnx::onnx().model_for_path(path)?,
            hidden,
            cache: HashMap::new(),
        })
    }

    /// Score N carried outputs against the normalized adjacency -> (graph logit, node logits).
    pub fn run(
        &mut self,
        outputs: &[f32],
        adjacency: &[f32],
        nodes: usize,
    ) -> TractResult<(f32, Vec<f32>)> {
        if !self.cache.contains_key(&nodes) {
            let plan = self
                .proto
                .clone()
                .with_input_fact(0, f32::fact([nodes, self.hidden]).into())?
                .with_input_fact(1, f32::fact([nodes, nodes]).into())?
                .into_optimized()?
                .into_runnable()?;
            self.cache.insert(nodes, plan);
        }
        let plan = &self.cache[&nodes];
        let result = plan.run(tvec!(
            Tensor::from_shape(&[nodes, self.hidden], outputs)?.into(),
            Tensor::from_shape(&[nodes, nodes], adjacency)?.into(),
        ))?;
        let graph_logit = result[0].view().as_slice::<f32>()?[0];
        Ok((graph_logit, result[1].view().as_slice::<f32>()?.to_vec()))
    }
}
