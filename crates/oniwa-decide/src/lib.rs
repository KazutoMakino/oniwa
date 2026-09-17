//! # oniwa-decide
//!
//! 🌿 ONIWA: Organic Non-datacenter Intelligence Without Abuse
//! TypeSafe AI「Jev」の System One 哲学に着想を得た、ピュアRust製 型安全意思決定エンジン。
//!
//! 自然言語やコードを単一フォワードパス（ミリ秒単位）で読み取り、
//! 自由な文章生成（自己回帰）を行わずに、プログラムが直接実行できる「型付けされた決定（Typed Decisions）」と
//! 較正された確信度（Calibrated Confidence）を出力します。

pub mod dataset;
pub mod layers;
pub mod loss;
pub mod model;

pub use dataset::{DatasetGenerator, DocCategory};
pub use loss::{LossCalculator, LossConfig};
pub use model::{DecisionConfig, DecisionModel, RawDecision};
use oniwa_lm::tokenizer::CharTokenizer;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// 選択決定プリミティブ (Jev 互換)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Choice<T> {
    pub value: T,
    pub probabilities: Vec<f32>,
    pub confidence: f32,
}

/// 真偽確率決定プリミティブ (Jev 互換)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Noul {
    pub value: bool,
    pub probability: f32,
    pub confidence: f32,
}

/// 連続数値評価プリミティブ (Jev 互換)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Score {
    pub value: f32,
    pub confidence: f32,
}

/// コード＆ドキュメント型安全監査の決定結果
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditDecision {
    pub category: Choice<DocCategory>,
    pub syntax_anomaly: Noul,
    pub complexity: Score,
    pub inference_time_ms: u128,
}

/// 高レベル決定エンジン
pub struct DecisionEngine {
    pub model: DecisionModel,
    pub tokenizer: CharTokenizer,
}

impl DecisionEngine {
    pub fn new(model: DecisionModel, tokenizer: CharTokenizer) -> Self {
        Self { model, tokenizer }
    }

    /// チェックポイントから読み込み
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

    /// テキストの型安全監査推論（単一フォワードパス）
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
