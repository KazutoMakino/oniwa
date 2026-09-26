# Walkthrough: refactor: Rust API Guidelines (RFC 430) Conformity

We have refactored public types and signatures in `oniwa-decide` and `oniwa-lm` according to the Rust API Guidelines (RFC 430):
- Acronyms are treated as single words in CamelCase (`RmsNorm`, `SwiGlu`, `QuaternionAttention`, `QuaternionSwiGlu`).
- Redundant module scopes in type names were streamlined (`BidirectionalSelfAttention` -> `Attention`, `QuaternionSelfAttention` -> `QuaternionAttention`).
- Parameter names in public function signatures were expanded from cryptic abbreviations to standard self-documenting identifiers (`dim`, `num_heads`, `head_dim`, `ffn_dim`, `num_tokens`, `vocab_size`).
- Checkpoint JSON serialization backwards compatibility remains 100% intact.

## Changes by Component

### `crates/oniwa-decide`
- `src/layers/rmsnorm.rs`:
  - Renamed `RMSNorm` to `RmsNorm`.
  - Updated parameters `c: usize` -> `dim: usize`.
- `src/layers/mlp.rs`:
  - Renamed `SwiGLU` to `SwiGlu`.
  - Updated parameters `c: usize` -> `dim: usize`, `ffn: usize` -> `ffn_dim: usize`.
- `src/layers/attention.rs`:
  - Renamed `BidirectionalSelfAttention` to `Attention`.
  - Updated parameters `c: usize` -> `dim: usize`, `nh: usize` -> `num_heads: usize`, `d_h: usize` -> `head_dim: usize`.
- `src/layers/quaternion_attention.rs`:
  - Renamed `QuaternionSelfAttention` to `QuaternionAttention`.
  - Updated parameters `c: usize` -> `dim: usize`, `nh: usize` -> `num_heads: usize`.
  - Updated internal RoPE calls to `Attention::apply_rope`.
- `src/layers/quaternion_mlp.rs`:
  - Renamed `QuaternionSwiGLU` to `QuaternionSwiGlu`.
  - Updated parameters `c: usize` -> `dim: usize`, `ffn: usize` -> `ffn_dim: usize`.
- `src/layers/mod.rs`:
  - Updated re-exports to `Attention`, `SwiGlu`, `QuaternionAttention`, `QuaternionLinear`, `QuaternionSwiGlu`, `RmsNorm`.
- `src/model.rs`:
  - Updated layer calls in forward and backward passes to match `RmsNorm`, `SwiGlu`, `Attention`, `QuaternionAttention`, `QuaternionSwiGlu`.

### `crates/oniwa-lm`
- `src/layers/rmsnorm.rs`:
  - Renamed `RMSNorm` to `RmsNorm`.
  - Updated parameters `d: usize` -> `dim: usize`.
- `src/layers/mlp.rs`:
  - Renamed `SwiGLU` to `SwiGlu`.
  - Updated parameters `c: usize` -> `dim: usize`, `ffn: usize` -> `ffn_dim: usize`.
- `src/layers/attention.rs`:
  - Updated parameter names in `apply_rope`, `forward`, and `backward`: `c: usize` -> `dim: usize`, `nh: usize` -> `num_heads: usize`, `d_h: usize` -> `head_dim: usize`.
- `src/layers/quaternion_head.rs`:
  - Updated parameter names: `n: usize` -> `num_tokens: usize`, `d: usize` -> `dim: usize`, `v: usize` -> `vocab_size: usize`.
- `src/model.rs`:
  - Updated layer calls in forward and backward passes to use `RmsNorm` and `SwiGlu`.

## Verification Results

### Automated Tests & Lint
- `cargo test --workspace`: **All 82 tests passed** (100% green).
- `cargo fmt --all -- --check`: **Clean** (no diffs).
- `cargo clippy --workspace --all-targets -- -D warnings`: **Clean** (0 warnings).

### Smoke Tests
- `cargo run --release -p oniwa-decide --bin gatekeeper -- --help`: **Success**.
- `cargo run --release -p oniwa-lm --bin chat -- --help`: **Success** (loads checkpoint & runs interactive REPL successfully).
- `cargo run --release -p oniwa-lm --bin train -- --steps 1`: **Success** (checkpoint restore & 1-step training verified).

## AI Self-Check Checklist

- [x] **Specification Compliance**: `RmsNorm`, `SwiGlu`, `QuaternionAttention`, `QuaternionSwiGlu`, and clarified argument names (`dim`, `num_heads`, `head_dim`, `ffn_dim`, etc.) are fully implemented.
- [x] **Scope Discipline**: Checkpoint JSON formats and config schemas (`ModelConfig`, `DecisionConfig`) were untouched and maintain backward compatibility. Unnecessary file moving or inner computational kernel disruption was avoided.
- [x] **Affected Files Match**: Only the targeted layer and model files in `oniwa-decide` and `oniwa-lm` were modified.
- [x] **Formatting Compliance**: `cargo fmt --all -- --check` passes with zero diffs.
- [x] **Test Passing**: `cargo test --workspace` passes 100% (Green).
- [x] **Documentation Completeness**: Both English and Japanese `implement_plan` and `walkthrough` documents are created in `.agent/tasks/20260927-010300/`.
- [x] **Git Completeness**: Both code files and `.agent/tasks/20260927-010300/` directory files are staged and committed.
