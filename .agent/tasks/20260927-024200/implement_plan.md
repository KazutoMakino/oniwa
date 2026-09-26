# Implementation Plan: ARM NEON SIMD Direct Vectorization for oniwa-decide Layers

## Overview
Vectorize the hot forward/inference loops in `oniwa-decide` layers (RmsNorm, SwiGlu, Attention, Quaternion Attention) using ARM NEON SIMD (`float32x4_t`) with robust scalar fallbacks, and verify numerical equivalence and test passes.

## Scope & Changes
1. **SIMD Primitives Addition (`crates/oniwa-lm/src/simd.rs`)**:
   - `sum_squares_simd(x: &[f32]) -> f32`: Sum of squares using NEON FMA/accumulate and remainder fallback.
   - `mul_slices_simd(out: &mut [f32], a: &[f32], b: &[f32])`: Element-wise product `out[i] = a[i] * b[i]`.
   - `scale_slice_simd(out: &mut [f32], a: &[f32], scalar: f32)`: Vector scaling `out[i] = a[i] * scalar`.
   - Add equivalence tests with scalar reference implementation (tolerance $< 10^{-5}$).
2. **Re-export Confirmation (`crates/oniwa-decide/src/simd/mod.rs`)**:
   - Ensure new primitives are visible and accessible in `oniwa-decide`.
3. **Layer Integration (`crates/oniwa-decide/src/layers/`)**:
   - `rmsnorm.rs`: Vectorize forward pass sum of squares and scaling with weight.
   - `mlp.rs`: Vectorize forward SwiGlu activation multiplication `act_h = silu(g) * u`.
   - `quaternion_mlp.rs`: Vectorize forward component-wise SwiGlu multiplication `act_h = silu(g) * u`.
   - `attention.rs`: Vectorize inner dot product in attention score calculation and attention output accumulation.
   - `quaternion_attention.rs`: Check and optimize dot product and attention loops.
4. **Verification**:
   - `cargo test --workspace`
   - `cargo fmt --all -- --check`
   - `cargo clippy --workspace --all-targets -- -D warnings`
   - Smoke tests (`cargo run --release -p oniwa-decide --bin gatekeeper -- --help`, `audit`).
5. **Documentation & Process**:
   - Complete `implement_plan.ja.md`, `walkthrough.md`, `walkthrough.ja.md` in `.agent/tasks/20260927-024200/`.
