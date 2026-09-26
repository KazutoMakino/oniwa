# Implementation Plan: refactor: Rust API Guidelines (RFC 430) Conformity

Harmonize acronym casing and public method signatures across `oniwa-decide` and `oniwa-lm` according to Rust API Guidelines (RFC 430) while preserving 100% backward compatibility for serialized checkpoints and configuration schemas.

## Proposed Changes

### `crates/oniwa-decide`
- `src/layers/rmsnorm.rs`:
  - Rename `RMSNorm` struct to `RmsNorm`.
  - Update method signatures: `c: usize` -> `dim: usize`.
- `src/layers/mlp.rs`:
  - Rename `SwiGLU` struct to `SwiGlu`.
  - Update method signatures: `c: usize` -> `dim: usize`, `ffn: usize` -> `ffn_dim: usize`.
- `src/layers/attention.rs`:
  - Rename `BidirectionalSelfAttention` struct to `Attention`.
  - Update method signatures: `c: usize` -> `dim: usize`, `nh: usize` -> `num_heads: usize`, `d_h: usize` -> `head_dim: usize`.
- `src/layers/quaternion_attention.rs`:
  - Rename `QuaternionSelfAttention` struct to `QuaternionAttention`.
  - Update references to `Attention::apply_rope`.
  - Update method signatures: `c: usize` -> `dim: usize`, `nh: usize` -> `num_heads: usize`.
- `src/layers/quaternion_mlp.rs`:
  - Rename `QuaternionSwiGLU` struct to `QuaternionSwiGlu`.
  - Update method signatures: `c: usize` -> `dim: usize`, `ffn: usize` -> `ffn_dim: usize`.
- `src/layers/quaternion_linear.rs`:
  - Review public argument names to ensure consistency.
- `src/layers/mod.rs`:
  - Re-export `Attention`, `SwiGlu`, `QuaternionAttention`, `QuaternionLinear`, `QuaternionSwiGlu`, `RmsNorm`.
- `src/model.rs`:
  - Update imports and layer invocations to match `RmsNorm`, `SwiGlu`, `Attention`, `QuaternionAttention`, `QuaternionSwiGlu`.
- `tests/decision_test.rs`:
  - Update test cases using old type names or signatures if any.

### `crates/oniwa-lm`
- `src/layers/rmsnorm.rs`:
  - Rename `RMSNorm` struct to `RmsNorm`.
  - Update signatures: `d: usize` -> `dim: usize`.
- `src/layers/mlp.rs`:
  - Rename `SwiGLU` struct to `SwiGlu`.
  - Update signatures: `c: usize` -> `dim: usize`, `ffn: usize` -> `ffn_dim: usize`.
- `src/layers/attention.rs`:
  - Keep `CausalSelfAttention` (standardizing argument names: `c: usize` -> `dim: usize`, `nh: usize` -> `num_heads: usize`, `d_h: usize` -> `head_dim: usize`).
- `src/layers/quaternion_head.rs`:
  - Update signatures: `n: usize` -> `num_tokens: usize`, `d: usize` -> `dim: usize`, `v: usize` -> `vocab_size: usize`.
- `src/model.rs`:
  - Update imports and layer calls for `RmsNorm`, `SwiGlu`.
- Binaries (`train.rs`, `chat.rs`, etc.):
  - Update references if applicable.

## Verification Plan

### Automated Tests
- Run `cargo test --workspace` to ensure all tests pass.
- Run `cargo fmt --all -- --check`.
- Run `cargo clippy --workspace --all-targets -- -D warnings`.
- Smoke test binary execution:
  - `cargo run --release -p oniwa-decide --bin gatekeeper -- --help`
  - `cargo run --release -p oniwa-lm --bin chat -- --help`
