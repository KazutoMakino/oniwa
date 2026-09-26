# Walkthrough: ARM NEON SIMD Direct Vectorization for oniwa-decide Layers

## Overview
Directly vectorized hot inference and forward passes in `oniwa-decide` layers (`rmsnorm.rs`, `mlp.rs`, `quaternion_mlp.rs`, `attention.rs`, `quaternion_attention.rs`) using new ARM NEON SIMD (`float32x4_t`) and robust scalar fallback primitives in `oniwa-lm::simd`.

## Changes Summary
- `crates/oniwa-lm/src/simd.rs`:
  - Added `dot_product_simd(a: &[f32], b: &[f32]) -> f32`
  - Added `sum_squares_simd(x: &[f32]) -> f32`
  - Added `mul_slices_simd(out: &mut [f32], a: &[f32], b: &[f32])`
  - Added `mul_slices_assign_simd(a: &mut [f32], b: &[f32])`
  - Added `scale_slice_simd(out: &mut [f32], a: &[f32], scalar: f32)`
  - Used `as_chunks::<4>()` and `as_chunks_mut::<4>()` ensuring safe and idiomatic slice chunking in Rust 1.98.
  - Added numerical equivalence unit tests (comparing SIMD against scalar implementation across varied slice lengths, verifying tolerance $< 10^{-5}$).
- `crates/oniwa-decide/src/simd/mod.rs`:
  - Verified re-exports of all primitives from `oniwa_lm::simd`.
  - Added equivalence unit tests within `oniwa-decide`.
- `crates/oniwa-decide/src/layers/rmsnorm.rs`:
  - Vectorized forward pass sum of squares calculation with `sum_squares_simd`.
  - Vectorized normalization scaling and weight multiplication with `scale_slice_simd` and `mul_slices_assign_simd`.
- `crates/oniwa-decide/src/layers/mlp.rs`:
  - Vectorized forward SwiGlu activation multiplication `act_h = silu(g) * u` with `mul_slices_assign_simd`.
- `crates/oniwa-decide/src/layers/quaternion_mlp.rs`:
  - Vectorized forward component-wise SwiGlu activation multiplication with `mul_slices_assign_simd`.
- `crates/oniwa-decide/src/layers/attention.rs`:
  - Vectorized bidirectional attention QK dot product scoring with `dot_product_simd`.
  - Optimized attention output accumulation across heads and tokens.
- `crates/oniwa-decide/src/layers/quaternion_attention.rs`:
  - Vectorized bidirectional attention quaternion inner product calculation ($\text{Re}(q \otimes k^*)$) using `dot_product_simd`.
  - Optimized attention output aggregation.

## Verification & Self-Check Log

### AI Self-Check Checklist
- [x] **Specification Compliance**: Primitives added to `oniwa-lm/src/simd.rs`, properly re-exported and integrated across RmsNorm, SwiGlu, and Attention layers in `oniwa-decide`.
- [x] **Strict Scope**: Zero breaking changes to public function signatures or buffer layouts; preserved standard mathematical implementations (`f32::exp`, `f32::sqrt`) without dangerous approximations.
- [x] **Impact Range Matching**: Only modified files defined in the task spec and associated docs.
- [x] **Formatting Adherence**: `cargo fmt --all -- --check` passes with zero diffs.
- [x] **Tests Passing**:
  - `cargo test --workspace` passes 100% green (all unit tests, equivalence tests, gradchecks, and integration tests passed).
  - `cargo clippy --workspace --all-targets -- -D warnings` passed with zero warnings.
- [x] **Documentation Coverage**: Both English and Japanese `implement_plan` and `walkthrough` documents are created in `.agent/tasks/20260927-024200/`.
- [x] **Git Completeness**: Source code files and `.agent/tasks/20260927-024200/` files are tracked and staged.

### Smoke Tests
- `cargo run --release -p oniwa-decide --bin gatekeeper -- --help`: Success
- `cargo run --release -p oniwa-decide --bin audit -- "pub fn test() {}"`: Success (Inference completed in 639ms with 92.5% confidence for RustCode category).
