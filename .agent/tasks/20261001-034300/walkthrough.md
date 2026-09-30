# [Walkthrough]: Decouple Inference Path & Adaptive NEON SIMD Optimization for Edge Pre-commit Latency

## 1. Overview of Changes
In this task, we addressed the edge pre-commit latency bottleneck of `oniwa-decide` (System 1) by:
1. **Decoupling the Dedicated Inference Path (`forward_inference`)**:
   - Skips tied MLM vocabulary projection ($V \times C = 4096 \times 256$ inner products) completely when computing decision heads (Choice, Noul, Score).
   - Reuses per-layer scratchpad vectors to avoid heap allocations and intermediate `LayerCache` activations.
2. **ARM NEON SIMD Accelerated Mean Pooling**:
   - Added `accumulate_slice_simd` in `oniwa-lm::simd` using `vld1q_f32`, `vaddq_f32`, and `vst1q_f32` (re-exported in `oniwa-decide::simd`).
   - Integrated `accumulate_slice_simd` and `scale_slice_simd` into Mean Pooling.
3. **Adaptive Sequence Length Slicing ($T_{\text{valid}}$)**:
   - In `DecisionModel::decide`, dynamic alignment rounds up $T_{\text{valid}}$ to the nearest multiple of 4 ($4 \le T \le \text{seq\_len}$), preserving Quaternion algebra and RoPE invariants while reducing quadratic attention and MLP operations for shorter snippets.
4. **Strict Numerical Equivalence Verification**:
   - Added `test_forward_inference_equivalence` in `tests/decision_test.rs` covering all combinations of Quaternion head and Quaternion backbone configurations, verifying that logits and predictions match within $< 10^{-5}$ tolerance.

---

## 2. Benchmark & Latency Results

### Standalone Pre-commit Diff Scan (`gatekeeper`)
- Tested on standard git diff snippet via `--stdin --bench`:
  - **Latency**: **145ms** (Achieved target goal of 100ms–150ms range).
  - **Verdict**: `PASS` (Noul=false, conf=59.2%, Score=0.47).

### Model Latency Benchmark (`bench --iters 5`)
- Full Quaternion (`770K` params):
  - **p50**: **42.17 ms**
  - **Mean**: **54.44 ms**
- Iso-Parameter Real (`467K` params):
  - **p50**: **63.04 ms**
  - **Mean**: **66.29 ms**
- Standard Baseline (`1.26M` params):
  - **p50**: **247.30 ms** (reduced from >540ms)

---

## 3. Self-Checklist Verification

- [x] **Specification Compliance**: `forward_inference` skips MLM projection, uses SIMD Mean Pooling, and leverages adaptive length slicing.
- [x] **Equivalence Verification**: `test_forward_inference_equivalence` passed with diff $< 10^{-5}$.
- [x] **Performance Goal**: Gatekeeper latency achieved 145ms (within 100ms–150ms target).
- [x] **Scope Preservation**: Training `forward`, `backward`, and `train.rs` remain completely untouched.
- [x] **Impacted Files**: Only designated files in `oniwa-lm`, `oniwa-decide`, and task documentation were modified.
- [x] **Formatting Cleanliness**: `cargo fmt --all -- --check` passed cleanly.
- [x] **Workspace Tests**: `cargo test --workspace` passed 100% green (all unit, integration, and doc tests).
- [x] **Branch Safety**: Working on dedicated branch `111/perf/decouple-inference-path-adaptive-neon` (not `main`).
- [x] **Document Completeness**: Both English and Japanese `implementation_plan` and `walkthrough` documents are created in `.agent/tasks/20261001-034300/`.
- [x] **Git Completeness**: Source code changes and task documentation staged together.
