# Implementation Plan: Compact UTC Timestamps & degC Temperature Unification

## Background & Objectives
During training runs and benchmarking across `oniwa-lm` and `oniwa-decide`, progress lines printed to stdout and plain text logs (`*train*.log`) lacked precise timestamps, complicating time-series analysis and background job monitoring. Furthermore, temperature notation varied between `C` and `℃`.

This plan implements:
1. Pure Rust UTC time formatter `current_utc_display()` in `crates/oniwa-lm/src/logger.rs` producing `YYYY-MM-DD HH:MM:SS UTC`.
2. Prefixing progress log lines with `[YYYY-MM-DD HH:MM:SS UTC]` in `crates/oniwa-lm/src/bin/train.rs` and `crates/oniwa-decide/src/bin/train.rs`.
3. Unifying temperature unit representations to `degC` across stdout and log text (e.g., `72.4 degC`), while strictly preserving JSON/JSONL key names (`cpu_temp_c`, `thermal_celsius`).
4. Unit tests covering `current_utc_display()` and epoch conversions.

---

## Scope & Target Files

1. `crates/oniwa-lm/src/logger.rs`:
   - Implement `format_utc_display(epoch_secs: u64) -> String` and `current_utc_display() -> String`.
   - Add unit tests for known epoch timestamps (including leap years and standard dates).
2. `crates/oniwa-lm/src/bin/train.rs`:
   - Prefix progress output (console and Growth Journal table) with `[<timestamp>]`.
   - Update temperature display formatting to `{:.1} degC`.
3. `crates/oniwa-decide/src/bin/train.rs`:
   - Prefix step progress lines with `[<timestamp>]`.
   - Update temperature display from `{:.1}C` to `{:.1} degC`.
4. `crates/oniwa-decide/src/bin/bench.rs`:
   - Ensure temperature output displays as `degC`.
5. `.agent/tasks/20260924-004500/`:
   - Documentation: `implement_plan.md`, `implement_plan.ja.md`, `walkthrough.md`, `walkthrough.ja.md`.

---

## Verification Plan
1. `cargo test --workspace`
2. `cargo fmt --all -- --check`
3. `cargo clippy --workspace --all-targets -- -D warnings`
4. Smoke run: `cargo run --release -p oniwa-lm --bin train -- --steps 2` and `cargo run --release -p oniwa-decide --bin train -- --steps 2`
5. AI Self-Check verification checklist in `walkthrough.md`.
