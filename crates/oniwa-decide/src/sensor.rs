//! Semantic Sensor: Non-Autoregressive Environment State Sensor
//!
//! Exposes `oniwa-decide`'s bidirectional encoder output as a compact,
//! fixed-dimensional state embedding for downstream World Models,
//! JEPA architectures, or high-speed routing engines.

use crate::model::DecisionModel;
use oniwa_lm::tokenizer::CharTokenizer;
use serde::{Deserialize, Serialize};

/// Compact state embedding output by `SemanticSensor`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateEmbedding {
    /// Context vector output by the encoder mean-pooling layer [dim]
    pub vector: Vec<f32>,
    /// Dimensionality of the embedding vector
    pub dim: usize,
    /// Overall sensor confidence metric based on decision heads
    pub confidence: f32,
    /// L2 Euclidean norm of the vector
    pub norm: f32,
}

impl StateEmbedding {
    /// Compute cosine similarity between two state embeddings
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        if self.dim != other.dim || self.norm < 1e-12 || other.norm < 1e-12 {
            return 0.0;
        }

        let mut dot = 0.0f32;
        for i in 0..self.dim {
            dot += self.vector[i] * other.vector[i];
        }

        dot / (self.norm * other.norm)
    }

    /// Return an L2-normalized copy of the embedding vector
    pub fn normalized_vector(&self) -> Vec<f32> {
        if self.norm > 1e-12 {
            let inv_norm = 1.0 / self.norm;
            self.vector.iter().map(|&v| v * inv_norm).collect()
        } else {
            vec![0.0f32; self.dim]
        }
    }
}

/// Semantic Sensor: maps text/code observations to state representations
pub struct SemanticSensor<'a> {
    pub model: &'a DecisionModel,
    pub tokenizer: &'a CharTokenizer,
}

impl<'a> SemanticSensor<'a> {
    /// Create a new SemanticSensor referencing a model and tokenizer
    pub fn new(model: &'a DecisionModel, tokenizer: &'a CharTokenizer) -> Self {
        Self { model, tokenizer }
    }

    /// Sense input text and produce a `StateEmbedding` in a single forward pass
    pub fn sense(&self, text: &str) -> StateEmbedding {
        let tokens = self.tokenizer.encode(text);
        let b = 1;
        let t = tokens.len().min(self.model.config.seq_len);
        let mut padded = vec![0u16; self.model.config.seq_len];
        padded[..t].copy_from_slice(&tokens[..t]);

        let cache = self.model.forward(&padded, b, self.model.config.seq_len);

        let dim = self.model.config.dim;
        let mut vector = vec![0.0f32; dim];
        vector.copy_from_slice(&cache.pooled[..dim]);

        let mut norm_sq = 0.0f32;
        for &v in &vector {
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt();

        // Compute aggregate confidence from Choice and Noul
        let choice_probs = crate::loss::LossCalculator::softmax(
            &cache.choice_logits[..self.model.config.num_choices],
            self.model.config.temperature,
        );
        let max_choice_p = choice_probs.iter().cloned().fold(0.0f32, f32::max);

        let noul_p = crate::loss::LossCalculator::sigmoid(cache.noul_logits[0]);
        let noul_conf = (noul_p - 0.5).abs() * 2.0;

        let confidence = (max_choice_p + noul_conf) * 0.5;

        StateEmbedding {
            vector,
            dim,
            confidence,
            norm,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::DecisionConfig;
    use oniwa_lm::reproducibility::DeterministicRng;

    #[test]
    fn test_semantic_sensor_embedding_properties() {
        let config = DecisionConfig {
            vocab_size: 100,
            seq_len: 16,
            dim: 32,
            ..Default::default()
        };
        let mut rng = DeterministicRng::new(42);
        let model = DecisionModel::new(config, &mut rng);
        let tokenizer = CharTokenizer::build_from_text("abcdefghijklmnopqrstuvwxyz 0123456789");

        let sensor = SemanticSensor::new(&model, &tokenizer);

        let emb1 = sensor.sense("fn main() { println!(\"Hello\"); }");
        assert_eq!(emb1.dim, 32);
        assert_eq!(emb1.vector.len(), 32);
        assert!(emb1.norm > 0.0);
        assert!(emb1.confidence >= 0.0 && emb1.confidence <= 1.0);

        // Same input produces identical deterministic embedding
        let emb1_repeat = sensor.sense("fn main() { println!(\"Hello\"); }");
        let sim_self = emb1.cosine_similarity(&emb1_repeat);
        assert!(
            (sim_self - 1.0).abs() < 1e-5,
            "Self cosine similarity must be 1.0, got {}",
            sim_self
        );

        // Different input produces different embedding
        let emb2 = sensor.sense("春はあけぼの。やうやう白くなりゆく山ぎは");
        let sim_diff = emb1.cosine_similarity(&emb2);
        assert!(
            sim_diff < 0.999,
            "Different inputs should have distinguishable embeddings"
        );
    }
}
