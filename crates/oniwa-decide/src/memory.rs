//! Fixed training-memory budget calculations for the System 1 model.

use crate::model::DecisionConfig;

pub const SYSTEM1_MEMORY_LIMIT_BYTES: usize = 100 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct FlatMemoryLayout {
    embedding: usize,
    parameters: usize,
    activations: usize,
}

impl FlatMemoryLayout {
    pub fn for_training(config: &DecisionConfig, batch_size: usize) -> Self {
        let c = config.dim;
        let ffn = config.ffn_dim;
        let layer_params = if config.quaternion_backbone {
            let cq = c / 4;
            let fq = ffn / 4;
            3 * cq * cq * 4 + cq * cq * 4 + 2 * cq * fq * 4 + cq * fq * 4 + 2 * c
        } else {
            c * 3 * c + c * c + c * 2 * ffn + ffn * c + 2 * c
        };
        let head_params = if config.use_quaternion_head {
            6 * (c / 4) * 4
        } else {
            c * config.num_choices + 2 * c
        };
        let embedding = config.vocab_size * c;
        let parameters = embedding + config.num_layers * layer_params + c + head_params;
        let n = batch_size * config.seq_len;
        // Mirrors the per-layer forward cache: normalized states, Q/K/V, attention, and SwiGLU buffers.
        let per_layer = n * (7 * c + 3 * ffn)
            + batch_size * config.num_heads * config.seq_len * config.seq_len
            + 2 * n;
        // MLM logits and their loss gradient coexist during the backward pass.
        let activations = n * c * 3 + config.num_layers * per_layer + 2 * n * config.vocab_size;
        Self {
            embedding,
            parameters,
            activations,
        }
    }

    pub fn embedding_elements(&self) -> usize {
        self.embedding
    }
    pub fn total_elements(&self) -> usize {
        self.parameters * 4 + self.activations
    }
    pub fn total_bytes(&self) -> usize {
        self.total_elements() * std::mem::size_of::<f32>()
    }
    pub fn fits_system1_budget(&self) -> bool {
        self.total_bytes() <= SYSTEM1_MEMORY_LIMIT_BYTES
    }
}
