# Phase 3: Development & Execution Handover Guide (16GB RAM)

<p align="left">
  <b>English</b> | <a href="04_development_handoff.ja.md">日本語 (Japanese)</a>
</p>



> [!NOTE]
> **Current Specification**: This document reflects the foundational/historical design. For current production specifications (Byte-level BPE, MLM multi-task learning, 100MB memory budget, and quaternion algebra integration), please refer to the latest primary specification: [10. ONIWA Production Roadmap Specification](10_system_integration_roadmap.md). Historical revisions are tracked via Git commit history and release tags.

This document provides a comprehensive guide for continuing implementation, testing, and training of `oniwa-lm` on another machine (e.g., 16GB RAM environment).

---

## 1. Prerequisites & Recommended Specs

* **Target Hardware**: PC / Laptop with 16GB RAM (Linux / macOS / WSL2)
* **Rust Toolchain**: 1.75 or later (`rustup update` recommended)
* **External Dependencies**: **Zero** (no Python, PyTorch, or CUDA drivers required; builds and runs entirely with `cargo`)

---

## 2. Quick Start Commands

After cloning or pulling the repository on the target machine:

```bash
# 1. Navigate to oniwa-lm crate
cd crates/oniwa-lm

# 2. Run all unit tests (gradient checks, thermal/power checks, reproducibility)
cargo test -- --nocapture

# 3. Quick training test (reset checkpoints, run 150 steps)
cargo run --release --bin train -- --reset

# 4. Interactive chat inference test
cargo run --release --bin chat
```

---

## 3. Recommended Hyperparameters for 16GB RAM

Configuration values designed for zero-swapping, comfortable in-memory execution:

```rust
pub struct Config {
    pub vocab_size: usize,   // e.g. 8,192
    pub seq_len: usize,      // 512 tokens
    pub dim: usize,          // 512 (hidden dimension)
    pub num_layers: usize,   // 8 layers
    pub num_heads: usize,    // 8 (Query heads)
    pub num_kv_heads: usize, // 2 (KV heads, GQA G=4)
    pub ffn_dim: usize,      // 1,365 (SwiGLU golden ratio: 512 * 8/3)
}
```

### Memory Footprint Breakdown (Total: ~910 MB)
* **Parameter weights**: ~140 MB (`f32`)
* **Gradient buffer**: ~140 MB (`f32`)
* **AdamW state buffer**: ~280 MB ($2 \times \text{Params}$ for $m$ and $v$)
* **Intermediate activations (B=4, T=512)**: ~350 MB
* **OS / Background Headroom**: **> 15 GB free** (safe to run concurrently with other desktop workloads)

---

## 4. Development Workflow & Implementation Tasks

When extending functionality within `src/layers/`:

### Step 1: Modern Primitives
1. **`src/layers/rope.rs` & `attention.rs`**:
   - Reference `02_modern_primitives.md` Section 2.
   - Forward rotation and backward reverse rotation (sign inversion).
2. **`src/layers/mlp.rs` (SwiGLU)**:
   - Reference `02_modern_primitives.md` Section 3.
   - Backward pass using $\text{Swish}'(u)$.
3. **`src/layers/rmsnorm.rs`**:
   - Reference `02_modern_primitives.md` Section 1.

### Step 2: Training Loop & Optimizer Integration
- Reference `01_llm_c_architecture.md` and `model.rs` to maintain manual backward traversal and flat-array AdamW optimization.

---

## 5. Document Cross-Reference Map

* **Architectural Decomposition**: [01_llm_c_architecture.md](01_llm_c_architecture.md)
* **Mathematical Formulations**: [02_modern_primitives.md](02_modern_primitives.md)
* **Memory & Lifecycle Design**: [03_rust_memory_model.md](03_rust_memory_model.md)
* **Layer Implementation Reference**: [`crates/oniwa-lm/src/layers/rmsnorm.rs`](../crates/oniwa-lm/src/layers/rmsnorm.rs)
