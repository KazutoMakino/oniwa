# Implementation Plan: System 1 Calibration, Git Pre-commit Hook Integration & On-Demand System 2 Cascade

## 1. Goal and Background
The objective is to resolve overconfident misclassification in System 1 (`oniwa-decide`) on short code snippets, integrate a fast diff-chunk based pre-commit hook into Git workflow with actionable hard-block diagnostics and bypass guidance, and establish an on-demand, memory-safe System 2 (`oniwa-lm`) escalation bridge for Raspberry Pi 4 edge constraints.

## 2. User Review & Non-Goals
- **Non-Goals**:
  - No breaking modifications to Choice category definitions (`DocCategory`).
  - No background IPC daemon or persistent server process (keeping process lifecycle strictly on-demand to protect Raspberry Pi 4 memory).
  - No binary raw state-embedding direct transmission as a tight coupling interface; communication uses structured prompts (diff chunk + System 1 reason and scores) over standard CLI / stdio.
- **Backward Compatibility**:
  - `DecisionConfig` and `meta.json` deserialization must maintain `#[serde(default)]` and backwards compatibility with existing checkpoints.

## 3. Proposed Changes

### Phase B: Temperature Calibration & Dataset Enhancement
- **`crates/oniwa-decide/src/config.rs` & `src/model.rs`**:
  - Add `crates/oniwa-decide/src/config.rs` to modularize configuration loading, post-hoc temperature adjustments, and validation.
  - In `crates/oniwa-decide/src/model.rs`, integrate `config.rs` and ensure `temperature` $T$ scales logits properly in decision routines.
- **Short Snippet & Ambiguous Code Regression Tests**:
  - Create `crates/oniwa-decide/tests/regression_snippets.rs`.
  - Add test cases evaluating ambiguous short snippets, bracket defects, and calibration behavior.

### Phase C: Git Hook / Pre-commit Integration with Diff-Chunk Gatekeeper
- **`crates/oniwa-decide/src/bin/gatekeeper.rs`**:
  - Enhance `git diff --cached` extraction to handle chunk-based inspection and non-code/empty diff safe bypass (`exit 0`).
  - Format terminal output with clear rationale, confidence scores, and explicit instructions for emergency bypass (`git commit --no-verify` or `ONIWA_BYPASS=1`).
- **`.githooks/pre-commit`**:
  - Integrate `gatekeeper --router` execution before or alongside `cargo fmt` and `cargo clippy`.
  - Check `ONIWA_BYPASS=1` environment variable to allow emergency developer bypass.

### Phase A: On-Demand System 2 Cascade Integration
- **`crates/oniwa-decide/src/cascade.rs` (and export in `lib.rs`)**:
  - Implement on-demand escalation runner `System2CascadeRunner`:
    - Formats structured prompt: `[System 1 Verdict: <reason>, Entropy: <e>] Diff Chunk:\n<code>`.
    - Spawns `oniwa-lm` binary or runs inference on-demand, capturing output and releasing memory immediately upon completion.
    - Gracefully handles process/memory errors on low-spec edge hardware (Raspberry Pi 4 fallback).
  - Add unit tests verifying mock/on-demand process invocation and graceful error handling.

## 4. Verification Plan
### Automated Tests
1. `cargo test -p oniwa-decide --test regression_snippets`
2. `cargo test -p oniwa-decide`
3. `cargo test --workspace`
4. Formatting and Clippy checks:
   - `cargo fmt --all -- --check`
   - `cargo clippy --workspace --all-targets -- -D warnings`
5. Manual test of gatekeeper on staged diffs and bypass options.
