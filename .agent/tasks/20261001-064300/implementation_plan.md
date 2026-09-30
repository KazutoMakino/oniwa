# [Implementation Plan]: Parallel Hunk Scanning, NEON Vectorization for Attention Softmax, and Checkpoint/Training Evaluation

## 1. Objectives & Overview
1. **Parallel Diff Hunk Scanning**: In `gatekeeper.rs`, replace sequential hunk auditing with multi-threaded parallel execution via Rayon (`into_par_iter()`), processing all hunks concurrently.
2. **SIMD Vectorization for Attention Softmax**: Add `max_element_simd` and `sum_slice_simd` (with ARM NEON `vmaxvq_f32` and `vaddvq_f32` / `vaddq_f32`) in `oniwa-lm::simd`, re-export in `oniwa-decide::simd`, and vectorize Attention row max and sum-exp reductions.
3. **Training & Inference Evaluation**: Execute training steps to ensure the model trains stably with the new SIMD paths, evaluate accuracy and multi-hunk pre-commit latency, and present next actionable steps.

## 2. Proposed Changes
- `crates/oniwa-lm/src/simd.rs`:
  - Add `max_element_simd(x: &[f32]) -> f32`
  - Add `sum_slice_simd(x: &[f32]) -> f32`
  - Unit tests for numerical equivalence.
- `crates/oniwa-decide/src/simd/mod.rs`:
  - Re-export the new SIMD functions and add tests.
- `crates/oniwa-decide/src/layers/attention.rs` & `crates/oniwa-decide/src/layers/quaternion_attention.rs`:
  - Utilize `max_element_simd` and `sum_slice_simd` in the Attention Softmax rows.
- `crates/oniwa-decide/src/bin/gatekeeper.rs`:
  - Use `rayon::prelude::*` for concurrent hunk evaluation.
- Training / Benchmark run:
  - Run `train` steps and measure `bench` and `gatekeeper`.

## 3. Verification Plan
- `cargo test -p oniwa-lm simd::tests`
- `cargo test -p oniwa-decide`
- `cargo test --workspace`
- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo run --release -p oniwa-decide --bin bench -- --iters 5`
- `cargo run --release -p oniwa-decide --bin gatekeeper -- --stdin --bench`
