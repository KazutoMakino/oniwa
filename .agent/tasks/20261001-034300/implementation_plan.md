# [Implementation Plan]: Decouple Inference Path & Adaptive NEON SIMD Optimization for Edge Pre-commit Latency

## 1. Objectives & Overview
- Decouple the inference path in `oniwa-decide` (`DecisionModel::forward_inference`) to completely skip the tied MLM vocabulary projection ($V \times C = 4096 \times 256$ dot products) and eliminate `LayerCache` / `ForwardCache` allocations not needed during inference.
- Implement ARM NEON SIMD vectorization for Mean Pooling (`accumulate_slice_simd` in `oniwa-lm::simd` and re-exported in `oniwa-decide::simd`).
- Adapt sequence length in `DecisionEngine::audit_text` and `gatekeeper` to the valid token length $T_{\text{valid}}$, padded up to the nearest multiple of 4 ($4 \le T \le \text{seq\_len}$) to preserve Quaternion algebra and RoPE invariants while drastically reducing attention and MLP computational overhead on short snippets.
- Guarantee strict numerical equivalence ($< 10^{-5}$) between `forward` decision head logits and `forward_inference`.

## 2. Proposed Changes

### `crates/oniwa-lm/src/simd.rs`
- Add `accumulate_slice_simd(acc: &mut [f32], x: &[f32])`:
  - On `aarch64`: 4-wide NEON vector addition using `vld1q_f32`, `vaddq_f32`, and `vst1q_f32` with scalar loop for remainder.
  - On non-aarch64: scalar accumulation loop `acc[i] += x[i]`.
- Add unit equivalence test `test_accumulate_slice_simd_equivalence`.

### `crates/oniwa-decide/src/simd/mod.rs`
- Re-export `accumulate_slice_simd`.
- Add equivalence test in `oniwa-decide::simd::tests`.

### `crates/oniwa-decide/src/model.rs`
- Add `DecisionModel::forward_inference(&self, tokens: &[u16], b: usize, t: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)`:
  - Returns `(pooled, choice_logits, noul_logits, score_preds)`.
  - Reuses layer buffers without saving intermediate caches (`LayerCache`).
  - Skips MLM vocabulary projection entirely.
  - Uses `accumulate_slice_simd` and `scale_slice_simd` for Mean Pooling.
- Update `DecisionModel::decide(&self, tokens: &[u16]) -> RawDecision`:
  - Compute $T_{\text{valid}}$ rounded up to multiple of 4, minimum 4, maximum `self.config.seq_len`.
  - Prepare padded tokens of length $T_{\text{valid}}$.
  - Call `self.forward_inference(&padded, 1, t_valid)`.
- Keep `forward` and `forward_with_profile` intact for training and benchmark backward-compatibility.

### `crates/oniwa-decide/src/lib.rs`
- In `DecisionEngine::audit_text(&self, text: &str)`:
  - Tokenize text.
  - Pass tokens directly into `self.model.decide(&tokens)`, which dynamically adapts sequence length.

### `crates/oniwa-decide/src/bin/gatekeeper.rs`
- Update hunk scanning and benchmarking to leverage the faster adaptive inference.

### `crates/oniwa-decide/tests/decision_test.rs`
- Add unit test `test_forward_inference_equivalence`:
  - Verifies that `forward` and `forward_inference` produce identical Choice logits, Noul logits, and Score predictions within $< 10^{-5}$ tolerance across multiple seeds and configurations.

## 3. Verification Plan
- Phase 2: `cargo test -p oniwa-lm simd::tests`
- Phase 3: `cargo test -p oniwa-decide --test decision_test test_forward_inference_equivalence`
- Phase 4: `cargo test -p oniwa-decide --test regression_snippets`
- Phase 5:
  - `cargo fmt --all -- --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`
  - `cargo test --workspace`
- Phase 6:
  - Latency benchmark: `cargo run --release -p oniwa-decide --bin bench -- --iters 5 --profile-layers`
  - Smoke test: `cargo run --release -p oniwa-decide --bin gatekeeper -- --bench`
