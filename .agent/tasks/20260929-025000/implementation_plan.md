# Implementation Plan: oniwa-decide Layer-by-Layer Microbenchmark Profiling and Hotspot Identification

## Summary
To drive inference latency down towards the Phase 4 target of ~100ms without speculative optimization or YAGNI violations, we introduce layer-by-layer microbenchmark profiling for `oniwa-decide`. We add `DecisionModel::forward_with_profile`, telemetry structures in `profiler.rs`, and a `--profile-layers` option in `crates/oniwa-decide/src/bin/bench.rs` with ASCII table visualization and JSONL logging.

## Proposed Changes

### 1. `crates/oniwa-decide/src/model.rs`
- Define `ForwardBreakdown` struct:
  ```rust
  #[derive(Clone, Debug, Default, Serialize, Deserialize)]
  pub struct ForwardBreakdown {
      pub embedding_ms: f64,
      pub layer_ms: Vec<f64>,
      pub pooling_ms: f64,
      pub heads_ms: f64,
      pub total_ms: f64,
  }
  ```
- Implement `DecisionModel::forward_with_profile(&self, tokens: &[u16], b: usize, t: usize) -> (ForwardCache, ForwardBreakdown)`.
- Keep existing `forward` completely intact and unmodified to prevent any regression or overhead in training loops.

### 2. `crates/oniwa-decide/src/profiler.rs`
- Add `LayerBreakdownRecord` struct or integrate `ForwardBreakdown` into telemetry structures for JSONL serializability.
- Re-export `ForwardBreakdown` if appropriate.

### 3. `crates/oniwa-decide/src/bin/bench.rs`
- Add `--profile-layers` / `--breakdown` command line flags.
- When enabled, run inferences using `forward_with_profile` and compute average breakdown across runs.
- Format and display an ASCII breakdown table (Phase name, Duration in ms, Percentage %).
- When `--output` is specified, export the breakdown metrics to JSONL.

### 4. `crates/oniwa-decide/tests/decision_test.rs`
- Add unit test `test_forward_with_profile` to verify that `ForwardBreakdown` has non-negative timings, layer count matches model configuration, and total_ms is consistent with sum of components.

## Verification Plan
- `cargo check -p oniwa-decide`
- `cargo test -p oniwa-decide --test decision_test test_forward_with_profile`
- `cargo run --release -p oniwa-decide --bin bench -- --iters 1 --profile-layers`
- `cargo test --workspace`
- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
