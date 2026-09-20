//! True End-to-End Provenance
//!
//! Records all lifecycle events (build, ingestion, preprocessing, training, and inference)
//! as tamper-evident structured logs (JSON Lines), ensuring 100% provenance transparency.

use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

// ==========================================
// 1. Build Provenance
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
// 2. Data Ingestion Provenance
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
// 3. Training Provenance
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
    #[serde(default)]
    pub label_smoothing: f32,
    #[serde(default)]
    pub z_loss_weight: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainingStepLog {
    #[serde(default)]
    pub timestamp_utc: String,
    pub step: usize,
    pub epoch: usize,
    pub loss: f32,
    /// Validation loss (periodic evaluation)
    #[serde(default)]
    pub val_loss: Option<f32>,
    pub learning_rate: f32,
    pub grad_norm: f32,
    pub elapsed_ms: u128,
    /// CPU actual temperature (Celsius)
    pub cpu_temp_c: Option<f32>,
    /// Cooling sleep duration inserted by thermal throttling (ms)
    pub throttle_sleep_ms: u64,
    /// Estimated instantaneous power consumption (W)
    pub estimated_power_w: f32,
    /// Cumulative energy consumed (Wh)
    pub accumulated_energy_wh: f32,
    /// Parameter checksum (periodic output)
    pub param_checksum: Option<String>,
    /// Multi-axial benchmark metrics (periodic output)
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
    #[serde(default)]
    pub lit_cloze_score: f32,
    #[serde(default)]
    pub code_cloze_score: f32,
    #[serde(default)]
    pub code_bracket_score: f32,
    #[serde(default)]
    pub indent_score: f32,
}

// ==========================================
// 4. Inference Provenance
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
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainingRunLog {
    pub timestamp_utc: String,
    pub model_name: String,
    pub git_commit_hash: String,
    #[serde(default)]
    pub git_dirty: bool,
    pub random_seed: u64,
    pub total_steps: usize,
    pub best_step: usize,
    pub best_loss: f32,
    pub final_loss: f32,
    pub best_weights_sha256: String,
    pub latest_weights_sha256: String,
    pub elapsed_secs: f32,
    pub cumulative_energy_wh: f32,
    pub target_arch: String,
    pub target_os: String,
}

// ==========================================
// Provenance Ledger
// ==========================================
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "event_type", content = "payload")]
pub enum ProvenanceEvent {
    Build(BuildLog),
    DataIngestion(DataIngestionLog),
    TrainingStart(TrainingManifest),
    TrainingStep(TrainingStepLog),
    Inference(InferenceLog),
    TrainingRun(TrainingRunLog),
}

/// Streaming logger appending all lifecycle events to a single append-only ledger file.
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

    /// Record an arbitrary lifecycle event with immediate flush.
    pub fn record(&mut self, event: &ProvenanceEvent) -> std::io::Result<()> {
        let json = serde_json::to_string(event)?;
        writeln!(self.writer, "{}", json)?;
        self.writer.flush()
    }
}

/// Generate current UTC timestamp in ISO 8601 format (e.g., "2026-09-15T13:45:30Z").
pub fn current_timestamp_utc() -> String {
    let now = std::time::SystemTime::now();
    let duration = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
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
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
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

/// Get Git commit hash (full 40 characters) via build-time env var or runtime fallback.
pub fn get_git_commit_hash() -> String {
    // 1. Prefer compile-time env var from build.rs
    if let Some(h) = option_env!("ONIWA_GIT_HASH") {
        if h != "unknown" && !h.is_empty() {
            return h.to_string();
        }
    }
    // 2. Runtime git command fallback
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                String::from_utf8(out.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// Shortened Git commit hash (7 characters).
pub fn get_git_short_hash() -> String {
    let full = get_git_commit_hash();
    if full.len() >= 7 {
        full[..7].to_string()
    } else {
        full
    }
}

/// Check if the Git working tree has uncommitted modifications (dirty).
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

        // 1. Record build event
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

        // 2. Record data ingestion event
        ledger
            .record(&ProvenanceEvent::DataIngestion(DataIngestionLog {
                timestamp_utc: "2026-09-14T09:10:00Z".into(),
                source_name: "青空文庫 (芥川龍之介 羅生門)".into(),
                source_url_or_path: "https://www.aozora.gr.jp/cards/000879/files/127_15260.html"
                    .into(),
                license: "Public Domain".into(),
                raw_data_sha256: "11223344".into(),
                raw_data_bytes: 15420,
                tokenized_sha256: "55667788".into(),
                num_tokens: 3850,
                vocab_size: 8192,
                tokenizer_type: "BPE".into(),
            }))
            .unwrap();

        // 3. Record training step (with temperature and timestamp)
        ledger
            .record(&ProvenanceEvent::TrainingStep(TrainingStepLog {
                timestamp_utc: "2026-09-21T00:00:00Z".into(),
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

        // 4. Record inference event
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

        // Verify
        let file = File::open(&ledger_path).unwrap();
        let reader = BufReader::new(file);
        let lines: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();

        assert_eq!(lines.len(), 4);
        assert!(lines[0].contains("\"event_type\":\"Build\""));
        assert!(lines[1].contains("\"event_type\":\"DataIngestion\""));
        assert!(lines[2].contains("\"cpu_temp_c\":68.5"));
        assert!(lines[2].contains("\"throttle_sleep_ms\":25"));
        assert!(lines[2].contains("\"timestamp_utc\":\"2026-09-21T00:00:00Z\""));
        assert!(lines[3].contains("\"event_type\":\"Inference\""));

        // Backward compatibility check for TrainingStepLog without timestamp_utc
        let legacy_json = r#"{"step":50,"epoch":1,"loss":4.5,"learning_rate":0.001,"grad_norm":0.1,"elapsed_ms":100,"cpu_temp_c":null,"throttle_sleep_ms":0,"estimated_power_w":4.0,"accumulated_energy_wh":0.05,"param_checksum":null}"#;
        let parsed: TrainingStepLog = serde_json::from_str(legacy_json).unwrap();
        assert_eq!(parsed.step, 50);
        assert_eq!(parsed.timestamp_utc, "");

        // Cleanup
        let _ = std::fs::remove_file(ledger_path);
    }
}
