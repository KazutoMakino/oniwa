# ⚖️ Lossy Compression Benchmark Report

<p align="left">
  <b>English</b> | <a href="lossy-compression-report.ja.md">日本語 (Japanese)</a>
</p>

> **Status**: Verified Empirical Data  
> **Hardware**: Raspberry Pi 4 (Quad-Core ARM Cortex-A72 @ 1.5GHz, Pure Rust, No GPU)  
> **Hypothesis**: [H3: Lossy Compression in System One Hypotheses](../design/system-one-hypotheses.md)

---

## 1. Executive Summary

This benchmark provides concrete empirical proof for Hypothesis H3: **"Decisions are irreversible lossy compression, hence ultra-fast."**

- **Information Compression**: System One decisions output an average of **5.0 bits**, compressing information by **22.1×** relative to a 50-token LLM output (**110.2 bits**).
- **Speed Advantage**: System One outputs decisions in **500 ms** on average, executing **47.4× faster** than token-by-token autoregression (**23744 ms**).

## 2. Experimental Results Table

| Domain | Sample | S2 Latency (LLM) | S1 Latency (Decide) | S2 Entropy $H(Y)$ | S1 Entropy $H(D)$ | Compression Ratio | Speedup Ratio |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Rust Code** | Fibonacci function | 24186 ms | 519 ms | 93.5 bits | 5.4 bits | **17.4×** | **46.6×** |
| **Python Code** | Binary search | 24056 ms | 496 ms | 58.0 bits | 4.6 bits | **12.5×** | **48.5×** |
| **Tech / Law Doc** | Cryptographic protocol | 24237 ms | 491 ms | 113.9 bits | 5.6 bits | **20.4×** | **49.4×** |
| **Literature** | Run, Melos! excerpt | 22497 ms | 496 ms | 175.2 bits | 4.3 bits | **40.3×** | **45.4×** |
| **Average / Overall** | - | **23744 ms** | **500 ms** | **110.2 bits** | **5.0 bits** | **22.1×** | **47.4×** |

## 3. Sample Details & Decisions

### Sample 1: Rust Code (Fibonacci function)

- **System One Decision**: `RustCode (P=59.4%), Anomaly=true, Score=3.29`
- **System Two Generation Excerpt**: `        >>> if is a condirection and a surrecursin`

### Sample 2: Python Code (Binary search)

- **System One Decision**: `PythonCode (P=88.7%), Anomaly=true, Score=2.99`
- **System Two Generation Excerpt**: `.          //    .. methowePurePosixPath` popowing`

### Sample 3: Tech / Law Doc (Cryptographic protocol)

- **System One Decision**: `LegalOrTechDoc (P=55.4%), Anomaly=true, Score=1.13`
- **System Two Generation Excerpt**: ` RIn with on the it rent an will be retriallec:檀忘 `

### Sample 4: Literature (Run, Melos! excerpt)

- **System One Decision**: `Literature (P=93.6%), Anomaly=false, Score=1.10`
- **System Two Generation Excerpt**: ` 　との武右衛門のは、私の一緒に、またベッテかに頬をかけて、動きに、中年の少年生の影を見たようにして`

---

*Reproduce this benchmark locally: `cargo run --release -p oniwa-decide --bin compress_eval`*
