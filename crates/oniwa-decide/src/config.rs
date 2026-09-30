//! Configuration and Calibration Metadata for oniwa-decide
//!
//! Manages model hyper-parameters, post-hoc temperature scaling ($T$),
//! and checkpoint metadata loading/saving.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionConfig {
    pub vocab_size: usize,
    pub seq_len: usize,
    pub dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub num_choices: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_use_quaternion_head")]
    pub use_quaternion_head: bool,
    #[serde(default = "default_quaternion_backbone")]
    pub quaternion_backbone: bool,
    #[serde(default)]
    pub score_unit_interval: bool,
}

fn default_use_quaternion_head() -> bool {
    false
}

fn default_quaternion_backbone() -> bool {
    false
}

fn default_temperature() -> f32 {
    1.0
}

impl Default for DecisionConfig {
    fn default() -> Self {
        Self {
            vocab_size: 4721,
            seq_len: 128,
            dim: 128,
            num_layers: 4,
            num_heads: 4,
            head_dim: 32,
            ffn_dim: 256,
            num_choices: 4,
            temperature: 1.0,
            use_quaternion_head: false,
            quaternion_backbone: false,
            score_unit_interval: false,
        }
    }
}

impl DecisionConfig {
    /// System 1 target: V=4096, d_model=256, T=128, B=4 fits the 100 MiB budget.
    pub fn system1_mlm() -> Self {
        Self {
            vocab_size: 4_096,
            seq_len: 128,
            dim: 256,
            num_layers: 4,
            num_heads: 4,
            head_dim: 64,
            ffn_dim: 1_024,
            num_choices: 4,
            temperature: 1.0,
            use_quaternion_head: true,
            quaternion_backbone: true,
            score_unit_interval: true,
        }
    }

    /// Standard baseline configuration (~1.26M parameters)
    pub fn standard_baseline(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            use_quaternion_head: false,
            quaternion_backbone: false,
            score_unit_interval: false,
            ..Default::default()
        }
    }

    /// Quaternion head configuration (~1.26M parameters with 4x compressed head)
    pub fn quaternion_head(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            use_quaternion_head: true,
            quaternion_backbone: false,
            score_unit_interval: false,
            ..Default::default()
        }
    }

    /// Full Quaternion Transformer configuration (~315K parameters)
    /// Both backbone (Attention + SwiGLU) and decision head operate in quaternion space.
    pub fn full_quaternion_transformer(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            use_quaternion_head: true,
            quaternion_backbone: true,
            score_unit_interval: false,
            ..Default::default()
        }
    }

    /// Iso-parameter configuration (~315K parameters matching Full Q-Transformer scale)
    /// Shrinks hidden dimension from 128 to 64, head_dim from 32 to 16, and ffn_dim from 256 to 128
    pub fn iso_parameter(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            seq_len: 128,
            dim: 64,
            num_layers: 4,
            num_heads: 4,
            head_dim: 16,
            ffn_dim: 128,
            num_choices: 4,
            temperature: 1.0,
            use_quaternion_head: false,
            quaternion_backbone: false,
            score_unit_interval: false,
        }
    }

    /// Set post-hoc calibration temperature $T$. Clamped to minimum 1e-3 for stability.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp.max(1e-3);
        self
    }

    /// Load configuration from a `meta.json` file path or directory containing `meta.json`
    pub fn load_from_meta<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let p = path.as_ref();
        let target = if p.is_dir() {
            p.join("meta.json")
        } else {
            p.to_path_buf()
        };
        let content = fs::read_to_string(target)?;
        let val: serde_json::Value = serde_json::from_str(&content)?;
        let config: DecisionConfig = serde_json::from_value(val["config"].clone())?;
        Ok(config)
    }

    /// Update temperature in a `meta.json` file in-place
    pub fn update_meta_temperature<P: AsRef<Path>>(
        path: P,
        temperature: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let p = path.as_ref();
        let target = if p.is_dir() {
            p.join("meta.json")
        } else {
            p.to_path_buf()
        };
        let content = fs::read_to_string(&target)?;
        let mut val: serde_json::Value = serde_json::from_str(&content)?;
        if let Some(config_obj) = val.get_mut("config").and_then(|c| c.as_object_mut()) {
            config_obj.insert(
                "temperature".to_string(),
                serde_json::Value::from(temperature.max(1e-3)),
            );
        }
        let serialized = serde_json::to_string_pretty(&val)?;
        fs::write(&target, serialized)?;
        Ok(())
    }
}
