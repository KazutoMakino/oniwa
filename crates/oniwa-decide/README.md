# ONIWA Decide (TypeSafe System One Decision Engine)

<p align="left">
  <b>English</b> | <a href="README.ja.md">日本語 (Japanese)</a>
</p>

`oniwa-decide` is a pure-Rust, type-safe decision engine inspired by the System One philosophy of TypeSafe AI.

Foregoing autoregressive token generation, it ingests code or text in a single forward pass and outputs **type-safe decision primitives (Choice, Noul, Score)** with calibrated confidence values, designed for CPU-only, sub-100ms execution on edge hardware like Raspberry Pi 4.

---

## Architectural Highlights

1. **Non-Autoregressive Single Forward Pass**:
   - Captures global sequence context via a bidirectional Transformer encoder without generation loops.
   - Outputs `Choice<T>` (categorical classification with label smoothing), `Noul` (binary anomaly flag with Brier score regularization), and `Score` (continuous metric with Smooth L1 / Huber loss).
2. **Flat Memory 100MB Budget Enforcement**:
   - `FlatMemoryLayout` (`src/memory.rs`) ensures the cumulative footprint of embeddings, parameters, optimizer state, and backward activations strictly stays within **100MB** (~32–42MB for $V=4,096$).
   - Mathematically prevents OOM crashes in resource-constrained edge environments.
3. **MLM (Masked Language Modeling) & Killer Pattern Integration**:
   - Equipped with a BERT-style bidirectional MLM head to uncover semantic inconsistencies, doc comment mismatches, and non-commutative ordering bugs that bypass standard AST analyzers.
   - Balances token restoration loss ($\mathcal{L}_{\text{MLM}}$) and sequence decision loss in a unified multi-task training loop.
4. **Quaternion Algebra Backbone & Heads**:
   - `QuaternionLinear` layers leverage Hamilton products to compress weight parameters to roughly $1/4$ of real-valued equivalents while representing non-commutative sequence dynamics.
   - Accelerated via ARM NEON SIMD vector instructions.
5. **On-Device Autonomous Training & Thermal/Power Telemetry**:
   - Integrates hardware dynamic throttling (monitoring `/sys/class/thermal/`) and real-time energy tracking (Joules/step) shared with `oniwa-lm`.

---

## Decision Primitives

| Primitive | Type | Purpose | Loss Function & Calibration |
| :--- | :---: | :--- | :--- |
| **`Choice<T>`** | Enum / Categorical | Document category / routing decision (Rust, Python, Tech/Legal doc, Literature) | Cross-Entropy with Label Smoothing (0.05) |
| **`Noul`** | `bool` (Binary) | Syntactic/semantic anomaly or killer pattern flag | Binary Cross-Entropy with Brier Score Regularization |
| **`Score`** | `f32` (1.0–5.0) | Syntactic complexity and quality metric | Smooth L1 (Huber Loss, $\delta=0.5$) |

---

## CLI Usage

### 1. Type-Safe Audit / Inference (`audit`)
```bash
# Perform type-safe audit on code or prose
cargo run --release -p oniwa-decide --bin audit -- "pub fn fibonacci(n: u64) -> u64 { ... }"
```

### 2. Self-Supervised Multi-Task Training (`train`)
```bash
# Basic training run (300 steps)
cargo run --release -p oniwa-decide --bin train -- --steps 300 --seed 42

# Train with Phase 1 BPE vocabulary and Phase 2 killer pattern dataset
cargo run --release -p oniwa-decide --bin train -- --steps 500 --bpe data/bpe_vocab.json --data data/killer_patterns.jsonl

# Train with quaternion head configuration
cargo run --release -p oniwa-decide --bin train -- --steps 300 --config quaternion_head

# Incrementally continue training (+100 steps)
cargo run --release -p oniwa-decide --bin train -- --add-steps 100
```

### 3. Lossy Compression Evaluation Benchmark (`compress-eval`)
```bash
# Evaluate irreversible information compression vs autoregressive LLM
cargo run --release -p oniwa-decide --bin compress-eval
```

### 4. Git Gatekeeper CLI (`gatekeeper`)
```bash
# Scan git diff and block anomalous code
git diff --cached | cargo run --release -p oniwa-decide --bin gatekeeper -- --stdin
```

---

## Specifications & Roadmap References
- [10. ONIWA Production Roadmap Specification (System 1 × System 2 Integration)](../../docs/10_system_integration_roadmap.md)
- [System One Hypotheses Design Document](../../docs/design/system-one-hypotheses.md)
- [Academic Quaternion Research Roadmap](../../docs/09_roadmap.md)
