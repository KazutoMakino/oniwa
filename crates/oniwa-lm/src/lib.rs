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
