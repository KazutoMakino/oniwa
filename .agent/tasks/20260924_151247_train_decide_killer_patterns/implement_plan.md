# Implementation Plan - System 1 (oniwa-decide) 8,000 Steps Training & Anomaly Detection Verification

## Overview
Train `oniwa-decide` (Full Quaternion Transformer) for 8,000 steps on `data/killer_patterns.jsonl` (and `data/corpus`), then verify syntax anomaly detection (`Noul` head) with real inference on both normal code and corrupted code.

## Proposed Changes
1. **Checkpoint Resolution Alignment**:
   - `crates/oniwa-decide/src/bin/audit.rs`: Add `checkpoints/full_quaternion/best` to default search paths.
   - `crates/oniwa-decide/src/bin/gatekeeper.rs`: Add `checkpoints/full_quaternion/best` to default search paths.
2. **Execute 8,000 Steps Training**:
   - Command:
     ```bash
     cargo run --release -p oniwa-decide --bin train -- \
       --steps 8000 \
       --config full_quaternion \
       --data data/killer_patterns.jsonl \
       --reset
     ```
3. **Inference & Evaluation Verification**:
   - Test Clean Code (Rust): expect `Noul = false` with high confidence.
   - Test Corrupted Code (Rust bracket / syntax break): expect `Noul = true` with high confidence (target: >80%).
   - Test Gatekeeper CLI on git diff / patch.
4. **CI & Formatting**:
   - `cargo test --workspace`
   - `cargo clippy --workspace --all-targets -- -D warnings`
   - `cargo fmt --all`
