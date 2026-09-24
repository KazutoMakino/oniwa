# Walkthrough - Step 1: Killer Pattern Contrastive Mutation Generator (`gen-killer-patterns`)

## Overview
Implemented the standalone deterministic killer pattern contrastive dataset generator (`pipelines/src/bin/gen-killer-patterns.rs`) for System 1 (`oniwa-decide`), enabling automatic synthesis of 1:1 clean vs. mutated code/doc slices to feed the anomaly detection head (`Noul`) and complexity score head.

## Changes Made
- **`pipelines/src/bin/gen-killer-patterns.rs`**:
  - Implemented CLI supporting `--corpus`, `--output`, `--vocab`, `--bpe`, `--pairs`, `--seed`.
  - Produces paired samples:
    - Clean (`Noul = 0.0`, `Score in [0.0, 1.0]`, `mask_indices = []`).
    - Mutated (`Noul = 1.0`, `Score in [0.70, 0.90]`, `mask_indices = [mut_pos]`).
  - Ensures strictly valid token boundaries and non-empty sequences matching `KillerPatternDataset` requirements.
- **`crates/oniwa-decide/tests/decision_test.rs`**:
  - Added `test_generated_killer_patterns_jsonl_roundtrip` to test loading the generated 10,000-record JSONL file and round-tripping batches through `KillerPatternDataset::write_batch`.

## Verification & Self-Check
- [x] **Specification Compliance**: Generated 5,000 contrastive pairs (10,000 records) into `data/killer_patterns.jsonl`.
- [x] **Formatting (`cargo fmt --all`)**: Checked and applied without remaining diffs.
- [x] **Clippy (`cargo clippy --workspace --all-targets -- -D warnings`)**: 0 warnings.
- [x] **Tests (`cargo test --workspace`)**: All workspace tests passed (24/24 in decision_test, 36/36 in oniwa_lm, 9/9 in oniwa_dataset, etc.).
- [x] **Git Completeness**: Source code and all task files in `.agent/tasks/20260924_125029_feat_gen_killer_patterns/` are tracked and ready.
