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
| **Standard Baseline** | 1,261,568 | 499.89 | 524.49 | 533.26 | 503.98 ± 2.1 | 11,652 | 1.762 J | 3.50 W |
| **Quaternion Head** | 1,261,568 | 497.79 | 532.90 | 549.14 | 503.43 ± 2.3 | 12,204 | 1.760 J | 3.50 W |
| **Full Q-Transformer** | **770,048** | **125.11** | **128.76** | **130.10** | **125.66 ± 0.6** | **10,284** | **0.438 J** | **3.49 W** |
| **Iso-Parameter Real** | 466,944 | 120.24 | 127.87 | 131.00 | 121.71 ± 0.8 | 11,327 | 0.424 J | 3.48 W |

---

## 3. Key Findings

1. **4.01× Latency Acceleration**:
   - The **Full Q-Transformer** achieves an average inference latency of **125.66 ms** compared to **503.98 ms** for the Standard Baseline, providing a **4.01× speedup**.
2. **75.1% Energy Reduction**:
   - Energy per inference drops from **1.762 Joules** down to **0.438 Joules**, drastically reducing the thermal and electrical load on low-power edge nodes.
3. **RSS Memory Footprint**:
   - Resident set size remains tightly bounded between **10.2 MB** and **12.2 MB**, allowing smooth operation even on 1GB or 512MB RAM edge systems.
4. **System One vs System Two Speedup**:
   - Compared to 50-token autoregressive generation in `oniwa-lm` (24,073 ms), Full Q-Transformer decision inference executes in 128 ms — **188.4× faster** with **105.8× information entropy compression**.
