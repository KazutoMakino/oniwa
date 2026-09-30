# ⏱️ Comprehensive Benchmark Report: Quaternion Decision Engine

<p align="left">
  <b>English</b> | <a href="quaternion_report.ja.md">日本語 (Japanese)</a>
</p>

> **Evaluation Environment**: Raspberry Pi 4 (Quad-Core ARM Cortex-A72 @ 1.5GHz, aarch64 Linux)  
> **Framework**: Pure Rust (`oniwa-decide`), zero external ML frameworks, zero Python/CUDA runtime.  
> **Statistical Significance**: 5 distinct random seeds ($seed \in \{42, 1042, 2042, 3042, 4042\}$), 50 total inferences per configuration.

---

## 1. Executive Summary

This report evaluates the hardware efficiency and architectural scalability of four configurations in the `oniwa-decide` System One decision engine:
1. **Standard Baseline**: Real-valued Transformer backbone (4 layers, dim 128) + Real-valued decision head.
2. **Quaternion Head**: Real-valued Transformer backbone (4 layers, dim 128) + QuaternionDecisionHead (4× head compression).
3. **Full Q-Transformer**: Complete quaternion backbone (Q-Attention + Q-SwiGLU) + QuaternionDecisionHead (~315K core params, 770K total with embeddings).
4. **Iso-Parameter Real**: Real-valued baseline matched to the parameter footprint of the Full Q-Transformer (~467K params).

---

## 2. Hardware Profiling & Latency Distribution

Measurements averaged across 5 random seeds (Mean ± Std Error):

| Configuration | Parameters | Latency p50 (ms) | Latency p95 (ms) | Latency p99 (ms) | Latency Mean (ms) | RSS (KB) | Energy (J/inf) | Net Compute Power |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Standard Baseline** | 1,261,568 | 255.62 | 318.32 | 363.44 | 267.65 ± 1.8 | 8,206 | 0.935 J | 3.49 W |
| **Quaternion Head** | 1,261,568 | 253.18 | 321.01 | 344.93 | 264.95 ± 1.7 | 8,292 | 0.926 J | 3.49 W |
| **Full Q-Transformer** | **770,048** | **40.27** | **42.48** | **45.42** | **40.70 ± 0.2** | **6,372** | **0.140 J** | **3.44 W** |
| **Iso-Parameter Real** | 466,944 | 54.66 | 66.38 | 76.32 | 56.79 ± 0.6 | 9,203 | 0.197 J | 3.47 W |

---

## 3. Key Findings

1. **6.58× Latency Acceleration (NEON SIMD Vectorization)**:
   - With ARM NEON Softmax/Hamilton SIMD and Adaptive Sequence optimizations, the **Full Q-Transformer** achieves an average inference latency of **40.70 ms** (p50: **40.27 ms**) compared to **267.65 ms** for the Standard Baseline, providing a **6.58× speedup**. It also significantly outperforms the iso-parameter real model (56.79 ms).
2. **85.0% Energy Reduction**:
   - Energy per inference drops from **0.935 Joules** down to **0.140 Joules**, drastically reducing the thermal and electrical load on low-power edge nodes.
3. **RSS Memory Footprint**:
   - Resident set size remains tightly bounded between **6.4 MB** and **9.2 MB**, allowing smooth operation even on 512MB RAM edge systems.
4. **System One vs System Two Speedup**:
   - Compared to 50-token autoregressive generation in `oniwa-lm` (23,296 ms), Full Q-Transformer decision inference executes in 77 ms — **303.5× faster** with **20.7× information entropy compression**.

---

## 4. Layer-by-Layer Microbenchmark Breakdown

Per-phase execution latency during forward pass (Raspberry Pi 4 empirical measurements):

| Phase | Standard Baseline | Quaternion Head | Full Q-Transformer | Iso-Parameter Real |
|:---|:---:|:---:|:---:|:---:|
| **Embedding** | 0.045 ms (0.02%) | 0.041 ms (0.02%) | 0.042 ms (0.10%) | 0.022 ms (0.04%) |
| **Layer 0 (Attn + MLP)** | 64.301 ms (22.37%) | 63.480 ms (25.34%) | **10.216 ms (24.97%)** | 14.013 ms (24.88%) |
| **Layer 1 (Attn + MLP)** | 68.708 ms (23.91%) | 62.022 ms (24.76%) | **10.084 ms (24.65%)** | 13.725 ms (24.36%) |
| **Layer 2 (Attn + MLP)** | 79.768 ms (27.76%) | 62.146 ms (24.81%) | **10.138 ms (24.78%)** | 14.619 ms (25.95%) |
| **Layer 3 (Attn + MLP)** | 74.519 ms (25.93%) | 62.745 ms (25.05%) | **10.388 ms (25.39%)** | 13.929 ms (24.73%) |
| **Mean Pooling & LN** | 0.028 ms (0.01%) | 0.031 ms (0.01%) | 0.026 ms (0.06%) | 0.016 ms (0.03%) |
| **Decision Heads** | 0.011 ms (0.00%) | 0.012 ms (0.00%) | 0.013 ms (0.03%) | 0.008 ms (0.01%) |
| **Total Forward Measured** | **331.03 ms** | **290.23 ms** | **80.05 ms (with cache)** | **75.80 ms** |

> **Takeaway**: Each Transformer layer in the Full Q-Transformer (Q-Attention + Q-SwiGLU) executes in **~10.2 ms**. Thanks to vectorized NEON SIMD Hamilton products, this achieves a **6.3× to 7.8× speedup per layer** compared to real-valued layers of matching hidden dimensionality (~64 to 79 ms).
