# Implementation Plan - Step 1: Killer Pattern Contrastive Mutation Generator (`gen-killer-patterns`)

## Overview
Implement deterministic killer pattern contrastive mutation dataset generator (`pipelines/src/bin/gen-killer-patterns.rs`) and verify with `KillerPatternDataset` in `crates/oniwa-decide`.

## User Review Required
No breaking architectural changes in this step. Dataset format conforms directly to `KillerPatternDataset::from_jsonl_reader`.

## Proposed Changes

### `pipelines/Cargo.toml` & `pipelines/src/bin/gen-killer-patterns.rs`
- Add binary target or rely on cargo default binary discovery `pipelines/src/bin/gen-killer-patterns.rs`.
- Command line arguments:
  - `--corpus <DIR>`: default `data/corpus`
  - `--output <FILE>`: default `data/killer_patterns.jsonl`
  - `--pairs <COUNT>`: default 5000 (total samples: 10,000)
  - `--seed <U64>`: default 42
  - `--bpe-vocab <FILE>`: optional BPE vocabulary path for token-level index checking
- Generation logic:
  - Read corpus files across 4 categories (Rust, Python, Tech/Legal, Literature).
  - Use `DeterministicRng` for reproducible selection.
  - Slice snippets of reasonable token/char length (~128-256 tokens).
  - For each pair:
    - Normal slice (`Noul = 0.0`, `Score = compute_complexity / 5.0` or standard score in `[0.0, 1.0]`, `mask_indices = []`). Note that `KillerPatternDataset` checks `raw.labels.noul` and `score` in `[0.0, 1.0]`. Score can be normalized to `[0.0, 1.0]` by dividing complexity score (1.0..5.0) or mapped directly. Specifically, in `KillerPatternDataset`, `raw.labels.score` must be in `0.0..=1.0`. So `(complexity - 1.0) / 4.0` or `score / 5.0` will be used (normalized to [0.0, 1.0]).
    - Mutated slice (`Noul = 1.0`, `Score = mutated_score / 5.0`, `mask_indices = [mutated_token_indices]`).
    - Mutate 1 token/syntax position (bracket deletion/substitution, indentation breakage, or syntax corruption), verify that `mask_indices` are strictly ascending and unique, and that all indices are `< written_tokens` if encoded with tokenizer.
- Output deterministic JSONL format.

### `crates/oniwa-decide/tests/decision_test.rs`
- Add unit test verifying that generated killer pattern JSONL can be parsed and passed to `KillerPatternDataset::write_batch` without any index out of bounds or parse errors.

## Verification Plan
1. `cargo check --workspace`
2. `cargo run --release -p oniwa-pipeline --bin gen-killer-patterns -- --pairs 5000 --output data/killer_patterns.jsonl`
3. `cargo test --workspace`
4. `cargo fmt --all -- --check`
5. `cargo clippy --workspace --all-targets -- -D warnings`
