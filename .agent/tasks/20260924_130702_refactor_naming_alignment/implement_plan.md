# Implementation Plan - Step 2: Rust Idiomatic Naming Alignment & Documentation Sync

## Overview
Align naming conventions and idioms across `oniwa` workspace with official Rust API Guidelines (RFC 430):
- Binary target name and source file renaming to `kebab-case` (`compress_eval.rs` -> `compress-eval.rs`).
- Getter conventions: Eliminate `get_` prefixes from simple field accessors (e.g. `get_extension` -> `extension`).
- Constructor/conversion traits: Keep idiomatic `new`, `as_`, `to_`, `into_`.
- Synchronize command usage and binary references in README and crate documentations.
- Preserve static benchmark data in `docs/benchmarks/*`.
- Verify full test suite and clippy zero-warning pass.

## Proposed Changes

### 1. Binary Naming & File Renaming
- `crates/oniwa-decide/src/bin/compress_eval.rs` -> `crates/oniwa-decide/src/bin/compress-eval.rs`
- Update `crates/oniwa-decide/Cargo.toml` `[[bin]]` section to `name = "compress-eval", path = "src/bin/compress-eval.rs"`.

### 2. Rust API Guidelines (RFC 430) Audit
- Scan for `fn get_` across `crates/` and `pipelines/`:
  - E.g., `get_extension(&str)` in `crates/oniwa-decide/src/bin/gatekeeper.rs` and `tests/decision_test.rs` -> `extension(&str)` or idiomatic method.
  - Review public APIs and methods for non-idiomatic `get_` getters.
- Audit struct fields / enums for snake_case / CamelCase conformity.

### 3. Documentation Sync
- Update `README.md`, `README.ja.md`:
  - Check any reference to `compress_eval` -> `compress-eval`.
  - Check `train_bpe` vs `train-bpe`.
- Update `crates/oniwa-decide/README.md`, `crates/oniwa-decide/README.ja.md`.
- Update `pipelines/README.md`, `pipelines/README.ja.md`.

## Verification Plan
1. `cargo test --workspace`
2. `cargo clippy --workspace --all-targets -- -D warnings`
3. `cargo fmt --all -- --check`
