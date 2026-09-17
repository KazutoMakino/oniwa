//! # oniwa-lm
//!
//! ONIWA: Organic Non-datacenter Intelligence Without Abuse
//! A minimal, pure-Rust deep learning training engine inspired by Andrej Karpathy's `llm.c`.
//! Designed with modern transformer primitives:
//! - RMSNorm (Root Mean Square Normalization)
//! - RoPE (Rotary Position Embedding)
//! - SwiGLU (Swish Gated Linear Unit)
//! - GQA (Grouped Query Attention)
//!
//! Features:
//! - Full Lifecycle Provenance & Audit Logging (`logger`)
//! - Deterministic Reproducibility with SHA-256 Checksums (`reproducibility`)
//! - Real-time Hardware Thermal Throttling (`thermal`)
//! - Zero-dependency Character Tokenizer (`tokenizer`)
//! - Minimal Transformer Model & Flat Buffer Layout (`model`)

pub mod benchmark;
pub mod layers;
pub mod logger;
pub mod model;
pub mod power;
pub mod reproducibility;
pub mod thermal;
pub mod tokenizer;

/// ワークスペースのルートディレクトリ（data/, logs/, Cargo.toml がある階層）を自動探索します。
pub fn find_workspace_root() -> std::path::PathBuf {
    // 1. カレントディレクトリまたはその親を走査
    if let Ok(mut dir) = std::env::current_dir() {
        loop {
            if dir.join("Cargo.toml").exists() && dir.join("data").is_dir() {
                return dir;
            }
            if !dir.pop() {
                break;
            }
        }
    }
    // 2. CARGO_MANIFEST_DIR の親を走査
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut dir = manifest_dir.to_path_buf();
    loop {
        if dir.join("Cargo.toml").exists() && dir.join("data").is_dir() {
            return dir;
        }
        if !dir.pop() {
            break;
        }
    }
    // 3. フォールバック
    manifest_dir.to_path_buf()
}
