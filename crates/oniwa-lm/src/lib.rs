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
pub mod quaternion;
pub mod reproducibility;
pub mod simd;
pub mod thermal;
pub mod tokenizer;

/// Automatically discovers the workspace root directory (containing data/, logs/, Cargo.toml).
pub fn find_workspace_root() -> std::path::PathBuf {
    // 1. Traverse current directory and its ancestors
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
    // 2. Traverse CARGO_MANIFEST_DIR and its ancestors
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
    // 3. Fallback
    manifest_dir.to_path_buf()
}
