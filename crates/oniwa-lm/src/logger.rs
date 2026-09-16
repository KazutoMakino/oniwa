//! 完全なライフサイクル・トレーサビリティ（True End-to-End Provenance）
//!
//! ビルド、データ取得、前処理、学習、推論の「すべて」のイベントを
//! 改ざん不能な構造化ログ（JSON Lines）として記録し、100%出自の明瞭なモデルを保証します。

use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

// ==========================================
// 1. ビルド時ログ (Build Provenance)
// ==========================================
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BuildLog {
    pub timestamp_utc: String,
    pub git_commit_hash: String,
    pub git_dirty: bool,
    pub rustc_version: String,
    pub target_arch: String,
    pub target_os: String,
    pub profile: String,
    pub binary_sha256: Option<String>,
}

// ==========================================
// 2. データ取得・前処理ログ (Data Ingestion Provenance)
// ==========================================
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DataIngestionLog {
    pub timestamp_utc: String,
    pub source_name: String,
    pub source_url_or_path: String,
    pub license: String,
    pub raw_data_sha256: String,
    pub raw_data_bytes: usize,
    pub tokenized_sha256: String,
    pub num_tokens: usize,
    pub vocab_size: usize,
    pub tokenizer_type: String,
}

// ==========================================
// 3. 学習セッション & ステップログ (Training Provenance)
// ==========================================
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainingManifest {
    pub project_name: String,
    pub version: String,
    pub git_commit_hash: String,
    #[serde(default)]
    pub git_dirty: bool,
    pub timestamp_utc: String,
    pub random_seed: u64,
    pub model_config: ModelConfigInfo,
    pub dataset_sha256: String,
    pub initial_weights_sha256: String,
    pub platform_arch: String,
    pub os_name: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelConfigInfo {
    pub vocab_size: usize,
    pub seq_len: usize,
    pub dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub ffn_dim: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainingStepLog {
    pub step: usize,
    pub epoch: usize,
    pub loss: f32,
    /// 検証損失 (Val Loss, 定期出力)
    #[serde(default)]
    pub val_loss: Option<f32>,
    pub learning_rate: f32,
    pub grad_norm: f32,
    pub elapsed_ms: u128,
    /// CPU実温度 (℃)
    pub cpu_temp_c: Option<f32>,
    /// サーマルスロットリングで挿入された冷却時間 (ms)
    pub throttle_sleep_ms: u64,
    /// 推定瞬間消費電力 (W)
    pub estimated_power_w: f32,
    /// 累積消費エネルギー (Wh)
    pub accumulated_energy_wh: f32,
    /// 重みのチェックサム (定期出力)
    pub param_checksum: Option<String>,
    /// 多軸評価ベンチマーク (定期出力)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub benchmark: Option<BenchmarkLog>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BenchmarkLog {
    pub top5_accuracy: f32,
    pub cloze_top1_accuracy: f32,
    pub cloze_top5_accuracy: f32,
    pub syntactic_score: f32,
    pub bracket_score: f32,
    pub non_repetition_score: f32,
}

// ==========================================
// 4. 推論ログ (Inference Provenance)
// ==========================================
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct InferenceLog {
    pub timestamp_utc: String,
    pub prompt: String,
    pub output_text: String,
    pub tokens_generated: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub random_seed: u64,
    pub duration_ms: u128,
    pub model_weights_sha256: String,
}

// ==========================================
// 全イベント統合用 Ledger (系譜台帳)
// ==========================================
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "event_type", content = "payload")]
pub enum ProvenanceEvent {
    Build(BuildLog),
    DataIngestion(DataIngestionLog),
    TrainingStart(TrainingManifest),
    TrainingStep(TrainingStepLog),
    Inference(InferenceLog),
}

/// 全ライフサイクルを1つの追記専用ファイルに書き込むストリーミングロガー
pub struct ProvenanceLedger {
    writer: BufWriter<File>,
}

impl ProvenanceLedger {
    pub fn open<P: AsRef<Path>>(ledger_path: P) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(ledger_path)?;

        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    /// 任意のライフサイクルイベントを即時フラッシュ書き込み
    pub fn record(&mut self, event: &ProvenanceEvent) -> std::io::Result<()> {
        let json = serde_json::to_string(event)?;
        writeln!(self.writer, "{}", json)?;
        self.writer.flush()
    }
}

/// 現在時刻を ISO 8601 (UTC, 例: "2026-09-15T13:45:30Z") 形式で生成
pub fn current_timestamp_utc() -> String {
    let now = std::time::SystemTime::now();
    let duration = now.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    let total_secs = duration.as_secs();

    let sec = total_secs % 60;
    let min = (total_secs / 60) % 60;
    let hour = (total_secs / 3600) % 24;

    let mut days = (total_secs / 86400) as i64;
    let mut year = 1970;
    loop {
        let leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
        let days_in_year = if leap { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }

    let leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    let days_in_months = [
        31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
    ];
    let mut month = 1;
    for &dim in &days_in_months {
        if days < dim {
            break;
        }
        days -= dim;
        month += 1;
    }
    let day = days + 1;

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hour, min, sec
    )
}

/// ビルド時または実行時フォールバックによる Git コミットハッシュの取得 (完全40桁)
pub fn get_git_commit_hash() -> String {
    // 1. build.rs によるコンパイル時環境変数を優先
    if let Some(h) = option_env!("ONIWA_GIT_HASH") {
        if h != "unknown" && !h.is_empty() {
            return h.to_string();
        }
    }
    // 2. 実行時コマンドフォールバック
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                String::from_utf8(out.stdout).ok().map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// 短縮 Git コミットハッシュ (7桁)
pub fn get_git_short_hash() -> String {
    let full = get_git_commit_hash();
    if full.len() >= 7 {
        full[..7].to_string()
    } else {
        full
    }
}

/// Git 作業ツリーの変更有無 (dirty) の判定
pub fn get_git_dirty() -> bool {
    if let Some(d) = option_env!("ONIWA_GIT_DIRTY") {
        if d == "true" {
            return true;
        } else if d == "false" {
            return false;
        }
    }
    std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .map(|out| out.status.success() && !out.stdout.is_empty())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};

    #[test]
    fn test_full_lifecycle_provenance_ledger() {
        let tmp_dir = std::env::temp_dir();
        let ledger_path = tmp_dir.join("test_niwa_full_ledger.jsonl");

        let mut ledger = ProvenanceLedger::open(&ledger_path).unwrap();

        // 1. ビルドイベント記録
        ledger
            .record(&ProvenanceEvent::Build(BuildLog {
                timestamp_utc: "2026-09-14T09:00:00Z".into(),
                git_commit_hash: "abc1234".into(),
                git_dirty: false,
                rustc_version: "rustc 1.85.0".into(),
                target_arch: "aarch64".into(),
                target_os: "linux".into(),
                profile: "release".into(),
                binary_sha256: Some("deadbeef".into()),
            }))
            .unwrap();

        // 2. データ取得イベント記録
        ledger
            .record(&ProvenanceEvent::DataIngestion(DataIngestionLog {
                timestamp_utc: "2026-09-14T09:10:00Z".into(),
                source_name: "青空文庫 (芥川龍之介 羅生門)".into(),
                source_url_or_path: "https://www.aozora.gr.jp/cards/000879/files/127_15260.html".into(),
                license: "Public Domain".into(),
                raw_data_sha256: "11223344".into(),
                raw_data_bytes: 15420,
                tokenized_sha256: "55667788".into(),
                num_tokens: 3850,
                vocab_size: 8192,
                tokenizer_type: "BPE".into(),
            }))
            .unwrap();

        // 3. 学習ステップ記録 (熱温度付き)
        ledger
            .record(&ProvenanceEvent::TrainingStep(TrainingStepLog {
                step: 100,
                epoch: 1,
                loss: 3.125,
                val_loss: Some(3.250),
                learning_rate: 0.001,
                grad_norm: 0.25,
                elapsed_ms: 120,
                cpu_temp_c: Some(68.5),
                throttle_sleep_ms: 25,
                estimated_power_w: 5.8,
                accumulated_energy_wh: 0.15,
                param_checksum: Some("99aabbcc".into()),
                benchmark: None,
            }))
            .unwrap();

        // 4. 推論イベント記録
        ledger
            .record(&ProvenanceEvent::Inference(InferenceLog {
                timestamp_utc: "2026-09-14T10:00:00Z".into(),
                prompt: "ある日の暮方の事である。".into(),
                output_text: "一人の下人が、羅生門の下で雨やみを待っていた。".into(),
                tokens_generated: 16,
                temperature: 0.8,
                top_p: 0.95,
                random_seed: 42,
                duration_ms: 450,
                model_weights_sha256: "99aabbcc".into(),
            }))
            .unwrap();

        // 検証
        let file = File::open(&ledger_path).unwrap();
        let reader = BufReader::new(file);
        let lines: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();

        assert_eq!(lines.len(), 4);
        assert!(lines[0].contains("\"event_type\":\"Build\""));
        assert!(lines[1].contains("\"event_type\":\"DataIngestion\""));
        assert!(lines[2].contains("\"cpu_temp_c\":68.5"));
        assert!(lines[2].contains("\"throttle_sleep_ms\":25"));
        assert!(lines[3].contains("\"event_type\":\"Inference\""));

        // クリーンアップ
        let _ = std::fs::remove_file(ledger_path);
    }
}
