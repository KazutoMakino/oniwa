# Walkthrough: System 1 Calibration, Git Pre-commit Hook Integration & On-Demand System 2 Cascade

## Summary
Successfully implemented and verified the end-to-end Issue-driven pipeline covering:
1. **Phase B (Calibration & Robustness)**: Post-hoc temperature scaling ($T$) configurable via metadata/config and thoroughly tested with short snippet regression suites (`tests/regression_snippets.rs`).
2. **Phase C (Git Hook Integration)**: Optimized `git diff --cached` chunk parsing in `gatekeeper`, formatted informative blocking output with rationale, and provided clear bypass guidance via `--no-verify` or `ONIWA_BYPASS=1`.
3. **Phase A (On-Demand System 2 Cascade)**: Added `oniwa_decide::cascade` (`System2CascadeRunner` and `EscalationPrompt`) to handle on-demand execution and immediate memory release, safeguarding Raspberry Pi 4 edge resource limits.

## Changes Made
- [crates/oniwa-decide/src/config.rs](file:///home/multi/GitHub/oniwa/crates/oniwa-decide/src/config.rs): Added standalone configuration module handling hyper-parameters, metadata JSON loading, and dynamic temperature scaling updates.
- [crates/oniwa-decide/src/model.rs](file:///home/multi/GitHub/oniwa/crates/oniwa-decide/src/model.rs): Re-exported `DecisionConfig` from `crate::config`, maintaining full backward compatibility.
- [crates/oniwa-decide/src/cascade.rs](file:///home/multi/GitHub/oniwa/crates/oniwa-decide/src/cascade.rs): Implemented `EscalationPrompt`, `System2CascadeRunner`, and `System2Result` for on-demand System 2 delegation and edge fallback.
- [crates/oniwa-decide/src/lib.rs](file:///home/multi/GitHub/oniwa/crates/oniwa-decide/src/lib.rs): Exported `config` and `cascade` modules.
- [crates/oniwa-decide/src/bin/gatekeeper.rs](file:///home/multi/GitHub/oniwa/crates/oniwa-decide/src/bin/gatekeeper.rs): Added `ONIWA_BYPASS=1` emergency bypass check, `--cascade` flag support, and detailed recovery guidance in block messages.
- [.githooks/pre-commit](file:///home/multi/GitHub/oniwa/.githooks/pre-commit): Integrated `cargo run -p oniwa-decide --bin gatekeeper -- --router` and `ONIWA_BYPASS` bypass check.
- [crates/oniwa-decide/tests/regression_snippets.rs](file:///home/multi/GitHub/oniwa/crates/oniwa-decide/tests/regression_snippets.rs): Added unit tests for temperature scaling, metadata updates, and short snippet entropy escalation.

## Validation Results
- `cargo test -p oniwa-decide --test regression_snippets`: PASSED (4/4 tests passed)
- `cargo test -p oniwa-decide`: PASSED (26 unit + 9 gatekeeper + 25 decision + 4 regression tests passed)
- `cargo test --all`: PASSED (115 total tests passed across workspace)
- `cargo fmt --all -- --check`: PASSED
- `cargo clippy --workspace --all-targets -- -D warnings`: PASSED

## Self-Check Verification
- [x] **Specification Compliance**: Temperature parameter modularized, diff chunk gatekeeper implemented, on-demand cascade integration established.
- [x] **Scope Guardrails**: No category schema breaking changes, no persistent daemon, no binary state-embedding coupling.
- [x] **Impacted Files**: Only files specified in plan modified or created.
- [x] **Formatting & Linting**: Formatted cleanly and verified with zero Clippy warnings.
- [x] **Branch Safety**: Verified not on main/master (`109/feat/sys1-calibration-hook-cascade`).
- [x] **Documentation**: Japanese and English implementation plans and walkthroughs created under `.agent/tasks/20261001-023800/`.
- [x] **Git Completeness**: All task documents and source code included.
