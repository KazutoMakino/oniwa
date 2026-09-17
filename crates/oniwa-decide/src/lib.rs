//! # oniwa-decide
//!
//! 🌿 ONIWA: Organic Non-datacenter Intelligence Without Abuse
//! Pure-Rust TypeSafe decision engine inspired by the System One philosophy of TypeSafe AI.
//!
//! Ingests natural language or code in a single forward pass (sub-millisecond latency)
//! and outputs directly executable typed decisions and calibrated confidence values,
//! without requiring free-form autoregressive text generation.

pub mod dataset;
pub mod layers;
pub mod loss;
pub mod model;
pub mod quaternion;

pub use dataset::{DatasetGenerator, DocCategory};
pub use layers::quaternion_linear::QuaternionLinear;
pub use loss::{LossCalculator, LossConfig};
pub use model::{DecisionConfig, DecisionModel, RawDecision};
use oniwa_lm::tokenizer::CharTokenizer;
pub use quaternion::Quaternion;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Categorical choice decision primitive
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Choice<T> {
    pub value: T,
    pub probabilities: Vec<f32>,
    pub confidence: f32,
}

/// Binary truth probability decision primitive
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Noul {
    pub value: bool,
    pub probability: f32,
    pub confidence: f32,
}

/// Continuous score decision primitive
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Score {
    pub value: f32,
    pub confidence: f32,
}

/// Result of type-safe audit decision on code or documentation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditDecision {
    pub category: Choice<DocCategory>,
    pub syntax_anomaly: Noul,
    pub complexity: Score,
    pub inference_time_ms: u128,
}

/// High-level decision engine
pub struct DecisionEngine {
    pub model: DecisionModel,
    pub tokenizer: CharTokenizer,
}

impl DecisionEngine {
    pub fn new(model: DecisionModel, tokenizer: CharTokenizer) -> Self {
        Self { model, tokenizer }
    }

    /// Load model and configuration from a checkpoint directory
    pub fn load_from_dir<P: AsRef<Path>>(
        checkpoint_dir: P,
        tokenizer: CharTokenizer,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let meta_str = std::fs::read_to_string(checkpoint_dir.as_ref().join("meta.json"))?;
        let meta: serde_json::Value = serde_json::from_str(&meta_str)?;
        let config: DecisionConfig = serde_json::from_value(meta["config"].clone())?;

        let mut rng = oniwa_lm::reproducibility::DeterministicRng::new(0);
        let mut model = DecisionModel::new(config, &mut rng);
        model.load_checkpoint(checkpoint_dir.as_ref())?;

        Ok(Self { model, tokenizer })
    }

    /// Run type-safe audit inference on text in a single forward pass
    pub fn audit_text(&self, text: &str) -> AuditDecision {
        let start = std::time::Instant::now();
        let tokens = self.tokenizer.encode(text);
        let raw = self.model.decide(&tokens);
        let elapsed = start.elapsed().as_millis();

        let cat_val = match raw.choice_idx {
            0 => DocCategory::RustCode,
            1 => DocCategory::PythonCode,
            2 => DocCategory::LegalOrTechDoc,
            _ => DocCategory::Literature,
        };

        AuditDecision {
            category: Choice {
                value: cat_val,
                probabilities: raw.choice_probs,
                confidence: raw.choice_confidence,
            },
            syntax_anomaly: Noul {
                value: raw.noul_value,
                probability: raw.noul_prob,
                confidence: raw.noul_confidence,
            },
            complexity: Score {
                value: raw.score_value,
                confidence: raw.score_confidence,
            },
            inference_time_ms: elapsed,
        }
    }
}
