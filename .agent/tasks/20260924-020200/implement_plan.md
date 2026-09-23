# Implementation Plan: Gatekeeper CLI for oniwa-decide

## Summary

Implement a standalone **Gatekeeper CLI** binary (`gatekeeper.rs`) within the `oniwa-decide` crate that scans Git staging diffs at millisecond latency using the trained `DecisionEngine`. This serves as a paper-grade evaluation tool for the ONIWA project, demonstrating real-world applicability of the System 1 decision engine on code review tasks.

## Architecture

```
git diff --cached → Unified Diff Parser → Hunk Extractor → DecisionEngine → Verdict (PASS/WARNING/BLOCK)
                                                                            ↓ (--bench mode)
                                                              logs/benchmarks/gatekeeper_eval.jsonl
```

## Files Affected

### [NEW] `crates/oniwa-decide/src/bin/gatekeeper.rs`
- **CLI argument parsing**: `--bench`, `--checkpoint <path>`, `--stdin`, `--help`
- **Unified diff parser**: Extracts hunks from `git diff --cached` or stdin
- **Extension filter**: `.rs`, `.py` (code), `.md`, `.txt` (document), skip binary/lock files
- **Hunk extraction**: 128–256 token windows from added lines + context
- **DecisionEngine integration**: Loads from checkpoint, runs `audit_text()` per hunk
- **Verdict policy**:
  - Code (`.rs`, `.py`): Noul==true && confidence ≥ 80% → BLOCK (exit 1)
  - Document (`.md`, `.txt`): anomaly → WARNING (exit 0)
  - No anomaly → PASS (exit 0)
- **Benchmark mode** (`--bench`): Logs JSONL records with inference time, entropy, confidence

### [MODIFY] `crates/oniwa-decide/tests/decision_test.rs`
- Add 9 new tests for diff parsing and hunk extraction:
  - Empty diff handling
  - Rust diff extraction
  - Binary extension skip
  - Document classification
  - Deletion-only diff skip
  - Lock file skip
  - Multi-file diff extraction
  - Extension classification coverage
  - Python diff code classification

## Verification Plan

1. `cargo fmt --all -- --check` — No formatting issues
2. `cargo clippy --workspace --all-targets -- -D warnings` — No clippy warnings
3. `cargo test --workspace` — All tests pass (100% green)
4. `cargo run --release -p oniwa-decide --bin gatekeeper -- --help` — Help message displayed
5. `cargo run --release -p oniwa-decide --bin gatekeeper -- --bench` — Benchmark mode runs

## Issue-Driven Workflow

- **Issue**: #91
- **Branch**: `91/feat/gatekeeper-cli-standalone-engine`
- **Commit message**: `feat: implement Gatekeeper CLI for paper-grade standalone gate-keeping engine (#91)`
