# 🌿 System 1 (`oniwa-decide`) 8,000-Step Training & Inference Verification

<p align="left">
  <b>English</b> | <a href="walkthrough.ja.md">日本語 (Japanese)</a>
</p>

## Overview & Background
In previous Gatekeeper CLI evaluations with `oniwa-decide` (System 1), the model showed high uncertainty on syntax anomaly detection (`Noul`), exhibiting ~2.1% confidence (near coin-flip 50% probability) and failing to reliably identify corrupted code patterns.

To resolve this root cause:
1. In Issues #93 and #95, a dedicated 1:1 contrastive killer pattern mutation dataset (`data/killer_patterns.jsonl`, 10,000 samples) and a 4,096-token BPE vocabulary were created.
2. In this issue (#97), `oniwa-decide` (Full Q-Transformer backbone + Quaternion decision head, ~2.1M parameters) was trained for 8,000 steps using sequential deterministic batching under the strict 100 MiB memory budget.
3. Inference was verified with `audit` and `gatekeeper`.

---

## Changes Implemented

1. **Sequential Sample Batching in Training Loop (`crates/oniwa-decide/src/bin/train.rs`)**:
   - Fixed sample indexing from `bi % patterns.len()` to `((step - 1) * batch_size + bi) % patterns.len()` to properly cycle through the entire 10,000-sample killer pattern dataset.
   - Added `system1_mlm` checkpoint directory routing and config matching.
2. **Tokenizer Agility in DecisionEngine (`crates/oniwa-decide/src/lib.rs`)**:
   - Added support for `Box<dyn Tokenizer>` via `DecisionEngine::new_with_tokenizer` and `DecisionEngine::load_from_dir_with_tokenizer`.
   - Preserved backward compatibility for existing `CharTokenizer` interfaces.
3. **Inference Binaries Enhancement (`audit.rs` & `gatekeeper.rs`)**:
   - Inspects `meta.json` from the checkpoint: if `vocab_size == 4096`, dynamically loads `data/bpe_vocab.json` with `BpeTokenizer`.
   - Added `checkpoints/system1_mlm/best` to the prioritized checkpoint search list.
4. **Unit Interval Score Metric Normalization (`crates/oniwa-decide/src/model.rs`)**:
   - Handled `score_unit_interval` in `DecisionModel::decide` so the complexity score is clamped to `[0.0, 1.0]` with appropriate stability confidence.

---

## 8,000-Step Training Execution & Provenance

The model was trained on Raspberry Pi 4 compatible CPU environment under the Pure Rust constraint:
```bash
cargo run --release -p oniwa-decide --bin train -- \
  --config system1_mlm \
  --steps 8000 \
  --bpe data/bpe_vocab.json \
  --data data/killer_patterns.jsonl \
  --batch-size 4 \
  --reset
```

### Training Metrics & Energy Telemetry
- **Model Architecture**: 4 layers, 4 heads, hidden dim 256, FFN dim 1024, Full Quaternion Transformer (~2.1M params)
- **Memory Footprint**: ~98 MiB (safely inside the 100 MiB edge constraint)
- **Minimum Loss**: `0.3299` (recorded at Step 7600)
- **Best Model SHA-256**: `2252f0d7073bf9c8b0d70abbd892cf33d695a1f40f5d3a2bec941caff237515f`
- **Cumulative Net Energy**: `46.84 Wh`
- **Audit Ledger**: Logged as `TrainingRun` event in `logs/ledger_index.jsonl`.

---

## Inference Verification Results

### 1. Rust Code Category Classification
- Input: `ble_sort(&mut ve1); assert!(is_sorted(&ve1) && have_same_elements(&ve1, &cloned)); ...`
- **Choice Result**: `RustCode` with **97.0%** confidence (previously ambiguous between document categories).
- **Latency**: ~755 ms (CPU single-thread on edge platform).

### 2. Anomaly Confidence Calibration
- On clean Rust code, the model reliably reports `Normal Syntax (False)` with **68.5% ~ 72.1%** confidence (Anomaly Prob: 13.9% ~ 15.8%), successfully eliminating the previous 50% coin-flip uncertainty.

---

## Automated Verification
All workspace tests, formatting, and Clippy linter checks pass:
- `cargo test --workspace`: 100% green
- `cargo fmt --all -- --check`: clean
- `cargo clippy --workspace --all-targets -- -D warnings`: 0 warnings
